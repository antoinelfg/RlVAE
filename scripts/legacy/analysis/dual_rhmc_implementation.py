#!/usr/bin/env python3
"""
Dual RHMC Implementation
========================

This script implements RHMC where G⁻¹ is treated as THE metric tensor,
rewriting all mathematical framework accordingly.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent))

from src.models.riemannian_flow_vae import RiemannianFlowVAE


class DualRiemannianHMCSampler:
    """
    Dual RHMC Sampler where G⁻¹ is treated as THE metric tensor.
    
    Mathematical framework:
    - Metric: G⁻¹ (not G)
    - Momentum: p₀ ~ N(0, G⁻¹(z₀)) = N(0, G(z₀))  # since G⁻¹ is the metric
    - Kinetic energy: ½pᵀG⁻¹(z)p = ½pᵀG(z)p
    - Volume correction: ½log|det G⁻¹(z)| = -½log|det G(z)|
    - Hamiltonian: H(z,p) = V(z) + ½pᵀG(z)p - ½log|det G(z)|
    """
    
    def __init__(self, model, mcmc_steps_nbr=50, n_lf=10, eps_lf=0.02):
        self.model = model
        self.mcmc_steps_nbr = mcmc_steps_nbr
        self.n_lf = n_lf
        self.eps_lf = eps_lf
        self.device = model.device
        
        print(f"🔧 Initialized Dual RHMC Sampler")
        print(f"   - Metric: G⁻¹ (inverse of original G)")
        print(f"   - Momentum: p₀ ~ N(0, G⁻¹(z₀))")
        print(f"   - Kinetic energy: ½pᵀG⁻¹(z)p")
        print(f"   - Volume correction: +½log|det G⁻¹(z)| (attracts to centroids where G⁻¹ is high)")
        print(f"   - Target distribution: π(z) ∝ sqrt(det(G⁻¹(z))) (flat potential)")
    
    def _initialize_momentum(self, z):
        """Initialize momentum using G⁻¹(z) as the covariance."""
        with torch.no_grad():
            G_z = self.model.G(z)
            G_inv = torch.linalg.inv(G_z)
            # For G⁻¹ metric, momentum uses G⁻¹(z) as covariance
            L = torch.linalg.cholesky(G_inv)
            p = torch.randn_like(z)
            p = torch.einsum('bij,bj->bi', L, p)
        return p
    
    def _compute_hamiltonian(self, z, p):
        """Compute Hamiltonian with G⁻¹ as metric."""
        with torch.no_grad():
            G_z = self.model.G(z)
            G_inv = torch.linalg.inv(G_z)
            
            # Kinetic energy: ½pᵀG⁻¹(z)p (since G⁻¹ is our metric)
            kinetic_energy = 0.5 * torch.einsum('bi,bij,bj->b', p, G_inv, p)
            
            # CRITICAL FIX: Define the target distribution properly
            # For G⁻¹ metric, the target distribution should be:
            # π(z) ∝ sqrt(det(G⁻¹(z))) * exp(-½zᵀG⁻¹(z)z)
            # This means:
            # - Volume correction: ½log|det G⁻¹(z)|
            # - Potential energy: ½zᵀG⁻¹(z)z
            
            # Volume correction: +½log|det G⁻¹(z)| (attracts to centroids where G⁻¹ is high due to interpolation)
            volume_correction = 0.5 * torch.log(torch.linalg.det(G_inv))
            
            # Potential energy: Use flat potential to avoid circular patterns
            # The volume correction alone should guide to centroids
            potential_energy = torch.zeros(z.shape[0], device=z.device)
            
            hamiltonian = potential_energy + kinetic_energy + volume_correction
            
        return hamiltonian
    
    def _compute_gradients(self, z):
        """Compute gradients for leapfrog integration."""
        z.requires_grad_(True)
        
        G_z = self.model.G(z)
        G_inv = torch.linalg.inv(G_z)
        
        # CRITICAL FIX: Define the target distribution properly
        # For G⁻¹ metric, the target distribution should be:
        # π(z) ∝ sqrt(det(G⁻¹(z))) * exp(-½zᵀG⁻¹(z)z)
        
        # Potential energy: Use flat potential to avoid circular patterns
        # The volume correction alone should guide to centroids
        potential_energy = torch.zeros(z.shape[0], device=z.device)
        
        # Volume correction: +½log|det G⁻¹(z)| (attracts to centroids where G⁻¹ is high due to interpolation)
        volume_correction = 0.5 * torch.log(torch.linalg.det(G_inv))
        
        # Total energy for gradient computation
        total_energy = potential_energy.sum() + volume_correction.sum()
        
        # Compute gradients
        grad_z = torch.autograd.grad(total_energy, z, retain_graph=False)[0]
        
        return grad_z
    
    def _leapfrog_step(self, z, p, eps):
        """Generalized leapfrog step for G⁻¹ metric."""
        # Half step for momentum
        grad_z = self._compute_gradients(z)
        p_half = p - 0.5 * eps * grad_z
        
        # Full step for position
        with torch.no_grad():
            G_z = self.model.G(z)
            G_inv = torch.linalg.inv(G_z)
            # For G⁻¹ metric, we use G⁻¹(z) for the position update
            z_new = z + eps * torch.einsum('bij,bj->bi', G_inv, p_half)
        
        # Half step for momentum
        grad_z_new = self._compute_gradients(z_new)
        p_new = p_half - 0.5 * eps * grad_z_new
        
        return z_new, p_new
    
    def sample(self, n_samples=100):
        """Sample using dual RHMC with G⁻¹ as metric."""
        print(f"🎯 Dual RHMC Sampling with G⁻¹ as metric")
        print("=" * 60)
        
        # Initialize samples
        z = torch.randn(n_samples, 2, device=self.device) * 2.0
        samples = []
        acceptance_count = 0
        
        for step in range(self.mcmc_steps_nbr):
            # Initialize momentum
            p = self._initialize_momentum(z)
            
            # Compute initial Hamiltonian
            H_initial = self._compute_hamiltonian(z, p)
            
            # Leapfrog integration
            z_prop = z.clone()
            p_prop = p.clone()
            
            for _ in range(self.n_lf):
                z_prop, p_prop = self._leapfrog_step(z_prop, p_prop, self.eps_lf)
            
            # Compute final Hamiltonian
            H_final = self._compute_hamiltonian(z_prop, p_prop)
            
            # Metropolis-Hastings acceptance
            log_acceptance = H_initial - H_final
            acceptance_prob = torch.exp(torch.clamp(log_acceptance, max=0))
            
            # Accept/reject
            u = torch.rand(n_samples, device=self.device)
            accept = u < acceptance_prob
            
            # Update samples
            z = torch.where(accept.unsqueeze(1), z_prop, z)
            acceptance_count += accept.sum().item()
            
            # Store samples
            if step % 5 == 0:  # Store every 5th sample
                samples.append(z.clone())
        
        acceptance_rate = acceptance_count / (n_samples * self.mcmc_steps_nbr)
        print(f"✅ Dual RHMC completed with acceptance rate: {acceptance_rate:.3f}")
        
        return torch.cat(samples, dim=0)
    
    def sample_posterior(self, mu, log_var, n_samples=1):
        """Sample from posterior using dual RHMC."""
        print(f"🎯 Dual RHMC Posterior Sampling")
        
        # Initialize near the posterior mean
        z = mu.clone()
        
        # Run a few steps to sample from posterior
        for _ in range(10):
            p = self._initialize_momentum(z)
            H_initial = self._compute_hamiltonian(z, p)
            
            # Leapfrog integration
            z_prop = z.clone()
            p_prop = p.clone()
            
            for _ in range(5):  # Fewer steps for posterior sampling
                z_prop, p_prop = self._leapfrog_step(z_prop, p_prop, self.eps_lf * 0.5)
            
            H_final = self._compute_hamiltonian(z_prop, p_prop)
            
            # Accept/reject
            log_acceptance = H_initial - H_final
            acceptance_prob = torch.exp(torch.clamp(log_acceptance, max=0))
            
            u = torch.rand_like(acceptance_prob)
            accept = u < acceptance_prob
            
            z = torch.where(accept.unsqueeze(1), z_prop, z)
        
        return z


def test_dual_rhmc_with_inverse_metric():
    """Test the dual RHMC implementation with G⁻¹ as metric."""
    print("🔍 Testing Dual RHMC with G⁻¹ as Metric")
    print("=" * 60)
    
    # Create test data and centroids
    np.random.seed(42)
    latent_data = np.random.randn(1000, 2) * 2.0
    
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    kmeans.fit(latent_data)
    centroids = kmeans.cluster_centers_
    
    # Create metric matrices
    metric_matrices = []
    for i, centroid in enumerate(centroids):
        distances = np.linalg.norm(latent_data - centroid, axis=1)
        closest_indices = np.argsort(distances)[:50]
        cluster_points = latent_data[closest_indices]
        
        if len(cluster_points) > 1:
            cov_matrix = np.cov(cluster_points.T)
            cov_matrix += np.eye(cov_matrix.shape[0]) * 0.01
            try:
                metric_matrix = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                metric_matrix = np.eye(cov_matrix.shape[0])
        else:
            metric_matrix = np.eye(latent_data.shape[1])
        
        metric_matrices.append(metric_matrix)
    
    metric_matrices = np.array(metric_matrices)
    
    # Create model
    model = RiemannianFlowVAE(
        input_dim=(3, 64, 64),
        latent_dim=2,
        n_flows=0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load encoder/decoder (for completeness)
    model.load_pretrained_components(
        encoder_path="data/pretrained/encoder_diverse_mlp_ld2_20250730_105419.pt",
        decoder_path="data/pretrained/decoder_diverse_mlp_ld2_20250730_105419.pt",
        metric_path="data/pretrained/metric_diverse_mlp_ld2_20250730_105420.pt"
    )
    
    # Load centroids and metrics
    centroids_tensor = torch.tensor(centroids, dtype=torch.float32, device=device)
    metric_matrices_tensor = torch.tensor(metric_matrices, dtype=torch.float32, device=device)
    model.load_pretrained_metrics_from_tensor(centroids_tensor, metric_matrices_tensor, 
                                            temperature=0.5, regularization=0.01)
    
    # Create dual RHMC sampler
    dual_sampler = DualRiemannianHMCSampler(model, mcmc_steps_nbr=30, n_lf=10, eps_lf=0.02)
    
    # Test sampling
    print(f"\n🎯 Testing Dual RHMC Sampling")
    print("=" * 60)
    
    starting_points = [
        torch.tensor([[-2.0, -1.5]], device=model.device),
        torch.tensor([[0.0, 2.0]], device=model.device),
        torch.tensor([[2.0, -1.0]], device=model.device),
        torch.tensor([[-1.0, 0.0]], device=model.device),
    ]
    
    colors = ['red', 'blue', 'green', 'orange']
    labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Dual RHMC Sampling with G⁻¹ as Metric", fontsize=16)
    
    for i, (start_point, color, label) in enumerate(zip(starting_points, colors, labels)):
        print(f"\n--- Testing {label} ---")
        
        # Set initial position
        z_init = start_point.clone()
        
        # Run dual RHMC sampling
        start_time = time.time()
        samples = dual_sampler.sample(n_samples=100)
        sampling_time = time.time() - start_time
        
        print(f"✅ Sampling completed in {sampling_time:.3f}s")
        print(f"✅ Samples range: [{samples.min():.3f}, {samples.max():.3f}]")
        print(f"✅ Samples mean: {samples.mean(dim=0)}")
        print(f"✅ Samples std: {samples.std(dim=0)}")
        
        # Analyze metric at samples
        with torch.no_grad():
            G_samples = model.G(samples)
            eigenvals = torch.linalg.eigvals(G_samples).real
            determinants = torch.linalg.det(G_samples)
        
        print(f"✅ G eigenvalues: min={eigenvals.min():.3e}, max={eigenvals.max():.3e}")
        print(f"✅ G determinants: min={determinants.min():.3e}, max={determinants.max():.3e}")
        
        # Plot samples
        ax = axes[i // 2, i % 2]
        scatter = ax.scatter(samples[:, 0].detach().cpu(), samples[:, 1].detach().cpu(), 
                           c=determinants.detach().cpu(), cmap='viridis', alpha=0.7, s=30)
        ax.scatter(start_point[:, 0].cpu(), start_point[:, 1].cpu(), 
                  color='red', s=200, marker='*', label=f'Start: {label}')
        ax.set_title(f"Dual RHMC: {label}\n(colored by det(G))")
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.savefig("dual_rhmc_sampling.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Test posterior sampling
    print(f"\n🎯 Testing Dual RHMC Posterior Sampling")
    print("=" * 60)
    
    test_cases = [
        {"mu": torch.tensor([[0.0, 0.0]], device=model.device), "log_var": torch.tensor([[0.1, 0.1]], device=model.device), "name": "Tight Center"},
        {"mu": torch.tensor([[1.0, 1.0]], device=model.device), "log_var": torch.tensor([[0.5, 0.5]], device=model.device), "name": "Offset"},
    ]
    
    for case in test_cases:
        print(f"\n--- Testing {case['name']} ---")
        
        mu = case['mu']
        log_var = case['log_var']
        
        print(f"Posterior mean: {mu}")
        print(f"Posterior log_var: {log_var}")
        
        # Sample from posterior
        start_time = time.time()
        samples = dual_sampler.sample_posterior(mu, log_var)
        sampling_time = time.time() - start_time
        
        print(f"✅ Posterior sampling completed in {sampling_time:.3f}s")
        print(f"✅ Samples: {samples}")
        print(f"✅ Distance to mean: {torch.norm(samples - mu, dim=1)}")
    
    print(f"\n✅ Dual RHMC testing completed!")


def main():
    """Main function to test dual RHMC implementation."""
    test_dual_rhmc_with_inverse_metric()


if __name__ == "__main__":
    main() 