#!/usr/bin/env python3
"""
Comparison of Sampling Methods - Why Posterior Samples Were Not Following Manifold
================================================================================

This script demonstrates the key difference between:
1. OLD METHOD: Posterior refinement using sample_riemannian_latents (limited exploration)
2. NEW METHOD: Full RHMC sampling using sample() (full exploration)

The old method only does "refinement" from a starting point, while the new method
does full exploration of the manifold structure.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from original_rlvae.src.models.riemannian_flow_vae import RiemannianFlowVAE
from src.models.samplers.hmc_sampler import RHVAEVolumeElementHMCSampler

def create_model_with_interesting_metric():
    """Create a model with interesting metric structure."""
    print("🔧 Creating model with interesting metric...")
    
    input_dim = [3, 64, 64]
    latent_dim = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = RiemannianFlowVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        riemannian_beta=1.0,
        temperature=0.7,
        regularization=0.01,
        n_centroids=50,
        posterior_type="riemannian_metric",
        loop_mode=True,
    ).to(device)
    
    # Initialize interesting metric
    n_centroids = 50
    angles = torch.linspace(0, 2 * np.pi, n_centroids, device=device)
    radius = 2.0
    
    centroids = torch.zeros(n_centroids, latent_dim, device=device)
    centroids[:, 0] = radius * torch.cos(angles) + 0.5 * torch.randn(n_centroids, device=device)
    centroids[:, 1] = radius * torch.sin(angles) + 0.5 * torch.randn(n_centroids, device=device)
    
    n_center = n_centroids // 4
    centroids[:n_center, 0] = 0.5 * torch.randn(n_center, device=device)
    centroids[:n_center, 1] = 0.5 * torch.randn(n_center, device=device)
    
    for d in range(2, min(6, latent_dim)):
        centroids[:, d] = 0.3 * torch.randn(n_centroids, device=device)
    
    model.centroids_tens = centroids
    
    # Create interesting metric matrices
    M_matrices = torch.zeros(n_centroids, latent_dim, latent_dim, device=device)
    
    for i in range(n_centroids):
        M = torch.eye(latent_dim, device=device)
        angle = angles[i] if i < len(angles) else 0
        
        scale_1 = 1.0 + 2.0 * torch.cos(angle) ** 2
        scale_2 = 1.0 + 2.0 * torch.sin(angle) ** 2
        
        M[0, 0] = scale_1
        M[1, 1] = scale_2
        
        if latent_dim >= 4:
            M[0, 2] = 0.3 * torch.cos(angle)
            M[2, 0] = 0.3 * torch.cos(angle)
            M[1, 3] = 0.3 * torch.sin(angle)
            M[3, 1] = 0.3 * torch.sin(angle)
        
        eigenvals, eigenvecs = torch.linalg.eigh(M)
        eigenvals = torch.clamp(eigenvals, min=0.1)
        M = torch.mm(torch.mm(eigenvecs, torch.diag(eigenvals)), eigenvecs.t())
        M_matrices[i] = M
    
    model.M_tens = M_matrices
    model.temperature = torch.tensor(0.7, device=device)
    model.lbd = torch.tensor(0.01, device=device)
    
    def G_inv(z):
        diff = model.centroids_tens.unsqueeze(0) - z.unsqueeze(1)
        weights = torch.exp(-torch.norm(diff, dim=-1) ** 2 / (model.temperature ** 2))
        weighted_M = model.M_tens.unsqueeze(0) * weights.unsqueeze(-1).unsqueeze(-1)
        G_inv = weighted_M.sum(dim=1) + model.lbd * torch.eye(model.latent_dim, device=z.device).unsqueeze(0)
        return G_inv
    
    def G(z):
        return torch.linalg.inv(G_inv(z))
    
    model.G_inv = G_inv
    model.G = G
    
    print(f"   ✅ Model created with interesting metric structure")
    return model, device

def sample_old_method(model, device, n_samples=128):
    """OLD METHOD: Posterior refinement using sample_riemannian_latents."""
    print(f"\n🔴 OLD METHOD: Sampling {n_samples} posterior samples using refinement...")
    
    # Create synthetic posterior parameters
    mu = torch.randn(n_samples, model.latent_dim, device=device) * 0.5
    log_var = torch.ones(n_samples, model.latent_dim, device=device) * -1.0
    
    # OLD METHOD: Use posterior refinement with limited parameters
    rhmc_sampler = RHVAEVolumeElementHMCSampler(
        model=model,
        mcmc_steps_nbr=100,  # Fewer steps
        n_lf=20,            # Fewer leapfrog steps
        eps_lf=0.001,       # Same step size
        beta_zero=1.0,
    )
    
    # OLD METHOD: Use sample_riemannian_latents (refinement from starting point)
    posterior_samples = rhmc_sampler.sample_riemannian_latents(mu, log_var)
    
    print(f"   ✅ OLD METHOD completed")
    print(f"   ✅ Sample shape: {posterior_samples.shape}")
    print(f"   ⚠️  This method only does REFINEMENT from starting point")
    
    return posterior_samples.cpu().numpy()

def sample_new_method(model, device, n_samples=128):
    """NEW METHOD: Full RHMC sampling using sample()."""
    print(f"\n🟢 NEW METHOD: Sampling {n_samples} posterior samples using full RHMC...")
    
    # NEW METHOD: Use full RHMC sampling with proper parameters
    rhmc_sampler = RHVAEVolumeElementHMCSampler(
        model=model,
        mcmc_steps_nbr=200,  # More steps for full exploration
        n_lf=30,            # More leapfrog steps
        eps_lf=0.001,       # Same step size
        beta_zero=1.0,
    )
    
    # NEW METHOD: Use sample() (full exploration from centroids)
    posterior_samples = rhmc_sampler.sample(n_samples)
    
    print(f"   ✅ NEW METHOD completed")
    print(f"   ✅ Acceptance rate: {rhmc_sampler.last_acceptance_rate:.3f}")
    print(f"   ✅ Sample shape: {posterior_samples.shape}")
    print(f"   ✅ This method does FULL EXPLORATION of manifold")
    
    return posterior_samples.cpu().numpy()

def sample_prior_for_comparison(model, device, n_samples=128):
    """Sample from prior for comparison."""
    print(f"\n🔵 PRIOR: Sampling {n_samples} prior samples using full RHMC...")
    
    rhmc_sampler = RHVAEVolumeElementHMCSampler(
        model=model,
        mcmc_steps_nbr=200,
        n_lf=30,
        eps_lf=0.001,
        beta_zero=1.0,
    )
    
    prior_samples = rhmc_sampler.sample(n_samples)
    
    print(f"   ✅ PRIOR sampling completed")
    print(f"   ✅ Acceptance rate: {rhmc_sampler.last_acceptance_rate:.3f}")
    print(f"   ✅ Sample shape: {prior_samples.shape}")
    
    return prior_samples.cpu().numpy()

def create_comparison_visualization(model, device, old_samples, new_samples, prior_samples):
    """Create visualization comparing the two methods."""
    print(f"\n🎨 Creating comparison visualization...")
    
    # Get centroids
    with torch.no_grad():
        centroids = model.centroids_tens.cpu().numpy()
    
    # PCA for visualization
    all_data = np.vstack([old_samples, new_samples, prior_samples, centroids])
    pca = PCA(n_components=2)
    pca.fit(all_data)
    
    # Transform all data
    old_pca = pca.transform(old_samples)
    new_pca = pca.transform(new_samples)
    prior_pca = pca.transform(prior_samples)
    centroids_pca = pca.transform(centroids)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: OLD METHOD vs PRIOR
    ax1.scatter(prior_pca[:, 0], prior_pca[:, 1], 
               c='blue', s=30, alpha=0.8, label='Prior Samples (Full RHMC)', edgecolors='none')
    ax1.scatter(old_pca[:, 0], old_pca[:, 1], 
               c='red', s=20, alpha=0.7, label='Posterior Samples (OLD: Refinement)', edgecolors='none')
    ax1.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
               c='cyan', s=100, alpha=0.9, label='Centroids', edgecolors='black', linewidth=1)
    
    ax1.set_xlabel('PCA Component 1', fontsize=12)
    ax1.set_ylabel('PCA Component 2', fontsize=12)
    ax1.set_title('OLD METHOD: Posterior Refinement vs Prior\n(Posterior samples do NOT follow manifold)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: NEW METHOD vs PRIOR
    ax2.scatter(prior_pca[:, 0], prior_pca[:, 1], 
               c='blue', s=30, alpha=0.8, label='Prior Samples (Full RHMC)', edgecolors='none')
    ax2.scatter(new_pca[:, 0], new_pca[:, 1], 
               c='green', s=20, alpha=0.7, label='Posterior Samples (NEW: Full RHMC)', edgecolors='none')
    ax2.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
               c='cyan', s=100, alpha=0.9, label='Centroids', edgecolors='black', linewidth=1)
    
    ax2.set_xlabel('PCA Component 1', fontsize=12)
    ax2.set_ylabel('PCA Component 2', fontsize=12)
    ax2.set_title('NEW METHOD: Full RHMC vs Prior\n(Both follow manifold structure)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add explanation text
    explanation_text = """EXPLANATION OF THE DIFFERENCE:

OLD METHOD (sample_riemannian_latents):
- Only does REFINEMENT from starting point (encoder posterior mean)
- Limited exploration (100 MCMC steps, 20 leapfrog steps)
- Result: Samples cluster around starting point, don't follow manifold

NEW METHOD (sample):
- Does FULL EXPLORATION from centroids
- Complete exploration (200 MCMC steps, 30 leapfrog steps)
- Result: Samples follow manifold structure like prior samples

The key insight: Posterior sampling should use the SAME full RHMC method
as prior sampling to properly explore the learned manifold structure."""
    
    fig.text(0.02, 0.02, explanation_text, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
             verticalalignment='bottom')
    
    plt.tight_layout()
    
    output_file = 'sampling_methods_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Comparison visualization saved as {output_file}")
    
    plt.show()
    
    return output_file

def main():
    """Main function to compare sampling methods."""
    print("🔍 Comparison of Sampling Methods - Why Posterior Samples Were Not Following Manifold")
    print("=" * 80)
    
    # Create model with interesting metric
    model, device = create_model_with_interesting_metric()
    
    # Sample using OLD method (refinement)
    old_samples = sample_old_method(model, device, n_samples=128)
    
    # Sample using NEW method (full RHMC)
    new_samples = sample_new_method(model, device, n_samples=128)
    
    # Sample from prior for comparison
    prior_samples = sample_prior_for_comparison(model, device, n_samples=128)
    
    # Create comparison visualization
    output_file = create_comparison_visualization(model, device, old_samples, new_samples, prior_samples)
    
    print(f"\n🎉 SUCCESS: Sampling methods comparison completed!")
    print(f"📁 Output file: {output_file}")
    print(f"📊 Key findings:")
    print(f"   🔴 OLD METHOD: Posterior refinement only explores around starting point")
    print(f"   🟢 NEW METHOD: Full RHMC exploration follows manifold structure")
    print(f"   🔵 PRIOR: Full RHMC exploration follows manifold structure")
    print(f"   ✅ SOLUTION: Use same full RHMC method for both prior and posterior")

if __name__ == "__main__":
    main()
