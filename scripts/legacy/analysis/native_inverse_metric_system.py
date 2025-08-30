#!/usr/bin/env python3
"""
Native Inverse Metric System
=============================

Build everything from scratch with G⁻¹ as the fundamental metric.
No more G -> G⁻¹ conversions - G⁻¹ is the native metric.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Tuple, Optional

sys.path.append(str(Path(__file__).parent))

class NativeInverseMetricTensor(nn.Module):
    """
    Metric tensor that natively works with G⁻¹ (inverse metric).
    
    Mathematical Framework:
    - Native metric: G⁻¹(z) (positive definite matrix)
    - Direct interpolation in G⁻¹ space
    - No G computation - everything based on G⁻¹
    """
    
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.centroids = None
        self.inverse_metrics = None  # These ARE the fundamental metrics (G⁻¹)
        self.temperature = 1.0
        self.regularization = 1e-4
        
        print("🔧 Native Inverse Metric Tensor initialized")
        print("   - Fundamental metric: G⁻¹(z)")
        print("   - No G computation required")
    
    def load_inverse_metrics(self, centroids: torch.Tensor, inverse_metrics: torch.Tensor, 
                           temperature: float = 1.0, regularization: float = 1e-4):
        """
        Load centroids and their corresponding G⁻¹ matrices.
        
        Args:
            centroids: [n_centroids, latent_dim] centroid positions
            inverse_metrics: [n_centroids, latent_dim, latent_dim] G⁻¹ matrices
            temperature: smoothness parameter for interpolation
            regularization: regularization for numerical stability
        """
        self.centroids = centroids.clone()
        self.inverse_metrics = inverse_metrics.clone()
        self.temperature = temperature
        self.regularization = regularization
        
        # Ensure all G⁻¹ matrices are positive definite
        for i in range(len(self.inverse_metrics)):
            eigenvals, eigenvecs = torch.linalg.eigh(self.inverse_metrics[i])
            eigenvals = torch.clamp(eigenvals, min=1e-6)  # Ensure positive definiteness
            self.inverse_metrics[i] = eigenvecs @ torch.diag(eigenvals) @ eigenvecs.T
        
        print(f"✅ Loaded {len(centroids)} centroids with native G⁻¹ metrics")
        print(f"   Temperature: {temperature}, Regularization: {regularization}")
        
        # Log det(G⁻¹) range for verification
        det_G_inv = torch.linalg.det(self.inverse_metrics)
        print(f"   det(G⁻¹) range: [{det_G_inv.min().item():.2e}, {det_G_inv.max().item():.2e}]")
    
    def G_inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute G⁻¹(z) by interpolating between centroid G⁻¹ matrices.
        
        Args:
            z: [batch_size, latent_dim] latent points
            
        Returns:
            G_inv: [batch_size, latent_dim, latent_dim] inverse metric matrices
        """
        if self.centroids is None:
            # Default to identity if no metrics loaded
            batch_size = z.shape[0]
            return torch.eye(self.latent_dim, device=z.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        batch_size = z.shape[0]
        device = z.device
        
        # Compute distances to centroids
        # z: [batch_size, latent_dim], centroids: [n_centroids, latent_dim]
        distances = torch.cdist(z, self.centroids)  # [batch_size, n_centroids]
        
        # Compute weights using exponential kernel
        weights = torch.exp(-distances / self.temperature)  # [batch_size, n_centroids]
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)  # Normalize
        
        # Interpolate G⁻¹ matrices
        # weights: [batch_size, n_centroids], inverse_metrics: [n_centroids, latent_dim, latent_dim]
        G_inv = torch.einsum('bn,nij->bij', weights, self.inverse_metrics)
        
        # Add regularization for numerical stability
        reg_matrix = self.regularization * torch.eye(self.latent_dim, device=device)
        G_inv = G_inv + reg_matrix.unsqueeze(0)
        
        return G_inv
    
    def log_det_G_inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log|det(G⁻¹(z))|.
        
        Args:
            z: [batch_size, latent_dim] latent points
            
        Returns:
            log_det: [batch_size] log determinants
        """
        G_inv = self.G_inverse(z)
        
        # Use slogdet for numerical stability
        sign, log_det = torch.linalg.slogdet(G_inv)
        
        # Ensure positive determinants (sign should be 1)
        log_det = torch.where(sign > 0, log_det, torch.full_like(log_det, -20.0))
        
        return log_det
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both G⁻¹ and log|det(G⁻¹)|.
        
        Args:
            z: [batch_size, latent_dim] latent points
            
        Returns:
            G_inv: [batch_size, latent_dim, latent_dim] inverse metric
            log_det_G_inv: [batch_size] log determinants
        """
        G_inv = self.G_inverse(z)
        log_det_G_inv = self.log_det_G_inverse(z)
        
        return G_inv, log_det_G_inv


class NativeInverseRHMC:
    """
    Riemannian HMC sampler built natively for G⁻¹ metric.
    
    Mathematical Framework:
    - Kinetic energy: T(p, z) = ½ pᵀ G⁻¹(z) p
    - Potential energy: V(z) = 0 (flat potential)
    - Volume correction: -½ log|det(G⁻¹(z))| (negative to attract to high det regions)
    - Hamiltonian: H(z, p) = T(p, z) + V(z) - ½ log|det(G⁻¹(z))|
    """
    
    def __init__(self, metric_tensor: NativeInverseMetricTensor, 
                 step_size: float = 1e-5, n_leapfrog: int = 50, n_steps: int = 100):
        self.metric_tensor = metric_tensor
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.n_steps = n_steps
        
        print("🎯 Native Inverse RHMC Sampler initialized")
        print(f"   - Native metric: G⁻¹(z)")
        print(f"   - Kinetic energy: ½ pᵀ G⁻¹(z) p")
        print(f"   - Volume correction: -½ log|det(G⁻¹(z))| (attracts to high det)")
        print(f"   - Step size: {step_size}")
        print(f"   - Leapfrog steps: {n_leapfrog}")
    
    def _sample_momentum(self, z: torch.Tensor) -> torch.Tensor:
        """
        Sample momentum p ~ N(0, G⁻¹(z)).
        
        Since G⁻¹ is the metric, momentum covariance is G⁻¹.
        """
        batch_size = z.shape[0]
        G_inv = self.metric_tensor.G_inverse(z)
        
        # Cholesky decomposition of G⁻¹
        try:
            L = torch.linalg.cholesky(G_inv)
        except RuntimeError:
            # Fallback: add small regularization
            reg = 1e-6 * torch.eye(z.shape[1], device=z.device)
            G_inv_reg = G_inv + reg.unsqueeze(0)
            L = torch.linalg.cholesky(G_inv_reg)
        
        # Sample p ~ N(0, G⁻¹)
        xi = torch.randn(batch_size, z.shape[1], device=z.device)
        p = torch.einsum('bij,bj->bi', L, xi)
        
        return p
    
    def _kinetic_energy(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute kinetic energy T(p, z) = ½ pᵀ G⁻¹(z) p.
        """
        G_inv = self.metric_tensor.G_inverse(z)
        
        # T = ½ pᵀ G⁻¹ p
        kinetic = 0.5 * torch.einsum('bi,bij,bj->b', p, G_inv, p)
        
        return kinetic
    
    def _potential_energy(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute potential energy V(z) = 0 (flat potential).
        """
        return torch.zeros(z.shape[0], device=z.device)
    
    def _volume_correction(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute volume correction: -½ log|det(G⁻¹(z))|.
        
        Negative sign attracts to regions with high det(G⁻¹).
        """
        log_det_G_inv = self.metric_tensor.log_det_G_inverse(z)
        return -0.5 * log_det_G_inv
    
    def _hamiltonian(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Compute total Hamiltonian H(z, p) = T + V + volume_correction.
        """
        kinetic = self._kinetic_energy(p, z)
        potential = self._potential_energy(z)
        volume = self._volume_correction(z)
        
        return kinetic + potential + volume
    
    def _compute_gradients(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute ∇_z H for the leapfrog integrator.
        """
        z_clone = z.clone().requires_grad_(True)
        
        # Compute only the volume correction gradient (potential is zero)
        volume_correction = self._volume_correction(z_clone)
        total_energy = volume_correction.sum()
        
        grad_z = torch.autograd.grad(total_energy, z_clone, create_graph=False)[0]
        
        return grad_z
    
    def _leapfrog_step(self, z: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one leapfrog integration step.
        """
        z_new = z.clone()
        p_new = p.clone()
        
        for step in range(self.n_leapfrog):
            # Half momentum step
            grad_z = self._compute_gradients(z_new)
            p_new = p_new - 0.5 * self.step_size * grad_z
            
            # Full position step: z += ε G⁻¹(z) p
            G_inv = self.metric_tensor.G_inverse(z_new)
            z_new = z_new + self.step_size * torch.einsum('bij,bj->bi', G_inv, p_new)
            
            # Half momentum step
            grad_z = self._compute_gradients(z_new)
            p_new = p_new - 0.5 * self.step_size * grad_z
        
        return z_new, p_new
    
    def sample(self, n_samples: int = 1000, initial_z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Run native inverse RHMC sampling.
        """
        # Determine device from metric data
        if self.metric_tensor.centroids is not None:
            device = self.metric_tensor.centroids.device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if initial_z is not None:
            z = initial_z.clone()
        else:
            z = torch.randn(n_samples, self.metric_tensor.latent_dim, device=device)
        
        samples = []
        accepted = 0
        
        print(f"🎯 Native Inverse RHMC Sampling")
        print("=" * 50)
        
        for step in range(self.n_steps):
            # Sample momentum
            p = self._sample_momentum(z)
            
            # Store current state
            z_current = z.clone()
            p_current = p.clone()
            
            # Leapfrog integration
            z_proposed, p_proposed = self._leapfrog_step(z_current, p_current)
            
            # Metropolis acceptance
            H_current = self._hamiltonian(z_current, p_current)
            H_proposed = self._hamiltonian(z_proposed, p_proposed)
            
            # Accept/reject
            alpha = torch.exp(H_current - H_proposed)
            alpha = torch.clamp(alpha, 0, 1)
            
            accept_mask = torch.rand(n_samples, device=device) < alpha
            z = torch.where(accept_mask.unsqueeze(1), z_proposed, z)
            
            accepted += accept_mask.sum().item()
            
            # Store samples
            if step >= self.n_steps // 4:  # Skip burn-in
                samples.append(z.clone())
            
            # Progress
            if (step + 1) % (self.n_steps // 4) == 0:
                acc_rate = accepted / (n_samples * (step + 1))
                print(f"Step {step+1}/{self.n_steps}: acceptance_rate={acc_rate:.3f}")
        
        final_acceptance = accepted / (n_samples * self.n_steps)
        print(f"✅ Native RHMC completed: acceptance_rate={final_acceptance:.3f}")
        
        # Return concatenated samples
        all_samples = torch.cat(samples, dim=0)
        return all_samples


def create_native_inverse_metric_data():
    """Create native G⁻¹ metric data from centroids."""
    print("🔧 Creating Native Inverse Metric Data")
    print("=" * 50)
    
    # Define strategic centroids
    centroids = torch.tensor([
        [0.0, 0.0],     # Origin
        [2.0, 0.0],     # Right
        [0.0, 2.0],     # Top  
        [-2.0, 0.0],    # Left
        [0.0, -2.0],    # Bottom
        [1.5, 1.5],     # Top-right
        [-1.5, -1.5],   # Bottom-left
        [1.5, -1.5],    # Bottom-right
        [-1.5, 1.5],    # Top-left
    ], dtype=torch.float32)
    
    # Create diverse G⁻¹ matrices (these are the FUNDAMENTAL metrics)
    inverse_metrics = []
    
    for i, centroid in enumerate(centroids):
        # Create diverse eigenvalues for G⁻¹
        base_scale = 500.0 + i * 300.0
        
        # Vary eigenvalue ratios for diversity
        if i % 3 == 0:
            eigenvals = torch.tensor([base_scale, base_scale * 0.5])  # Anisotropic
        elif i % 3 == 1:
            eigenvals = torch.tensor([base_scale, base_scale * 0.8])  # Moderate anisotropy
        else:
            eigenvals = torch.tensor([base_scale, base_scale])        # Isotropic
        
        # Create rotation for variety
        angle = i * np.pi / 4  # Different orientations
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)
        
        # G⁻¹ = R D R^T where D is diagonal eigenvalues
        G_inv = rotation @ torch.diag(eigenvals) @ rotation.T
        
        inverse_metrics.append(G_inv)
        
        print(f"Centroid {i}: {centroid.tolist()}")
        print(f"  G⁻¹ eigenvalues: {eigenvals.tolist()}")
        print(f"  det(G⁻¹): {torch.det(G_inv):.2e}")
    
    inverse_metrics = torch.stack(inverse_metrics)
    
    print(f"✅ Created {len(centroids)} native G⁻¹ metrics")
    print(f"   det(G⁻¹) range: [{torch.det(inverse_metrics).min():.2e}, {torch.det(inverse_metrics).max():.2e}]")
    
    return centroids, inverse_metrics


def test_native_inverse_system():
    """Test the complete native inverse metric system."""
    print("🚀 TESTING NATIVE INVERSE METRIC SYSTEM")
    print("=" * 70)
    
    # Create native G⁻¹ data
    centroids, inverse_metrics = create_native_inverse_metric_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    centroids = centroids.to(device)
    inverse_metrics = inverse_metrics.to(device)
    
    # Create native metric tensor
    metric_tensor = NativeInverseMetricTensor(latent_dim=2)
    metric_tensor.load_inverse_metrics(
        centroids, inverse_metrics, 
        temperature=2.0, regularization=1e-4
    )
    
    # Test metric interpolation
    print(f"\n🔬 Testing Metric Interpolation")
    test_points = torch.tensor([
        [0.0, 0.0],   # At centroid
        [0.5, 0.5],   # Between centroids
        [3.0, 3.0],   # Far from centroids
    ], device=device)
    
    G_inv_test, log_det_test = metric_tensor(test_points)
    
    for i, point in enumerate(test_points):
        print(f"Point {point.tolist()}: det(G⁻¹) = {torch.exp(log_det_test[i]).item():.2e}")
    
    # Create native RHMC sampler
    sampler = NativeInverseRHMC(
        metric_tensor, 
        step_size=1e-4, 
        n_leapfrog=100, 
        n_steps=200
    )
    
    # Run sampling
    print(f"\n🎯 Running Native Inverse RHMC")
    samples = sampler.sample(n_samples=1000)
    
    print(f"Generated {len(samples)} samples")
    print(f"Sample range: z1=[{samples[:, 0].min():.3f}, {samples[:, 0].max():.3f}], z2=[{samples[:, 1].min():.3f}, {samples[:, 1].max():.3f}]")
    
    # Analyze results
    print(f"\n📊 Analyzing Results")
    
    # Compute centroid distances
    min_distances = []
    for sample in samples:
        distances = torch.norm(centroids - sample.unsqueeze(0), dim=1)
        min_dist = torch.min(distances).item()
        min_distances.append(min_dist)
    
    overall_min = min(min_distances)
    mean_min = np.mean(min_distances)
    
    # Count close samples
    very_close = sum(1 for d in min_distances if d < 0.1)
    close = sum(1 for d in min_distances if d < 0.2)
    
    print(f"Minimum distance to centroids: {overall_min:.6f}")
    print(f"Mean distance to centroids: {mean_min:.4f}")
    print(f"Very close samples (<0.1): {very_close}/{len(samples)} ({100*very_close/len(samples):.1f}%)")
    print(f"Close samples (<0.2): {close}/{len(samples)} ({100*close/len(samples):.1f}%)")
    
    # Compute det(G⁻¹) at samples
    with torch.no_grad():
        _, log_det_samples = metric_tensor(samples)
        det_G_inv_samples = torch.exp(log_det_samples)
        
        _, log_det_centroids = metric_tensor(centroids)
        det_G_inv_centroids = torch.exp(log_det_centroids)
    
    print(f"det(G⁻¹) at samples - Min: {det_G_inv_samples.min().item():.2e}, Max: {det_G_inv_samples.max().item():.2e}")
    print(f"det(G⁻¹) at centroids - Min: {det_G_inv_centroids.min().item():.2e}, Max: {det_G_inv_centroids.max().item():.2e}")
    
    return samples, centroids, inverse_metrics, metric_tensor


if __name__ == "__main__":
    # Run the complete native inverse system test
    samples, centroids, inverse_metrics, metric_tensor = test_native_inverse_system()
    
    print(f"\n🎉 NATIVE INVERSE METRIC SYSTEM TEST COMPLETE!")
    print(f"✅ Everything built from scratch with G⁻¹ as fundamental metric")
    print(f"✅ No G->G⁻¹ conversions - pure native implementation")
    print(f"✅ Ready for comprehensive analysis and visualization")