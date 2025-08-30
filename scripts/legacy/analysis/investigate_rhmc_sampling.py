#!/usr/bin/env python3
"""
Investigate RHMC Sampling with Real Data
========================================

This script investigates the RHMC sampling behavior with real data,
creates visualizations of the manifolds, and tests with different metrics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
sys.path.append(str(Path(__file__).parent))

from src.models.riemannian_flow_vae import RiemannianFlowVAE
from src.models.samplers.hmc_sampler import RiemannianHMCSampler


def create_diverse_metric(latent_dim=2, n_centroids=10, device='cuda'):
    """Create a more diverse metric tensor for testing."""
    print("🔧 Creating diverse metric tensor...")
    
    # Create diverse centroids
    centroids = torch.randn(n_centroids, latent_dim, device=device) * 2.0
    
    # Create diverse metric matrices with different scales and orientations
    metric_matrices = []
    for i in range(n_centroids):
        # Create a random positive definite matrix
        A = torch.randn(latent_dim, latent_dim, device=device)
        # Make it symmetric positive definite
        M = A @ A.T + torch.eye(latent_dim, device=device) * 0.1
        # Add some diversity in scale
        scale = torch.rand(1, device=device) * 5.0 + 0.5
        M = M * scale
        metric_matrices.append(M)
    
    metric_matrices = torch.stack(metric_matrices)
    
    print(f"✅ Created diverse metric with {n_centroids} centroids")
    print(f"✅ Metric matrices range: [{metric_matrices.min():.3f}, {metric_matrices.max():.3f}]")
    
    return centroids, metric_matrices


def analyze_metric_properties(model, z_points, title="Metric Analysis"):
    """Analyze metric properties at given points."""
    print(f"\n🔍 {title}")
    
    with torch.no_grad():
        G_z = model.G(z_points)
        G_inv_z = model.G_inv(z_points)
        
        # Analyze eigenvalues
        eigenvals = torch.linalg.eigvals(G_z)
        eigenvals_real = eigenvals.real
        eigenvals_imag = eigenvals.imag
        
        print(f"✅ Metric shape: {G_z.shape}")
        print(f"✅ Eigenvalues real: min={eigenvals_real.min():.3e}, max={eigenvals_real.max():.3e}")
        print(f"✅ Eigenvalues imag: min={eigenvals_imag.min():.3e}, max={eigenvals_imag.max():.3e}")
        print(f"✅ Condition number: {(eigenvals_real.max() / (eigenvals_real.min() + 1e-8)):.2e}")
        
        # Check positive definiteness
        is_positive_definite = torch.all(eigenvals_real > 1e-6)
        print(f"✅ Positive definite: {is_positive_definite}")
        
        # Analyze determinant
        det_G = torch.linalg.det(G_z)
        print(f"✅ Determinant range: [{det_G.min():.3e}, {det_G.max():.3e}]")
        
        return {
            'G_z': G_z,
            'G_inv_z': G_inv_z,
            'eigenvals': eigenvals,
            'det_G': det_G,
            'is_positive_definite': is_positive_definite
        }


def visualize_metric_manifold(model, z_points, title="Metric Manifold"):
    """Visualize the metric manifold at given points."""
    print(f"\n🎨 Creating {title} visualization...")
    
    # Analyze metric properties
    metric_data = analyze_metric_properties(model, z_points, title)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot 1: Original points
    axes[0, 0].scatter(z_points[:, 0].cpu(), z_points[:, 1].cpu(), alpha=0.7)
    axes[0, 0].set_title("Original Points")
    axes[0, 0].set_xlabel("z₁")
    axes[0, 0].set_ylabel("z₂")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Metric eigenvalues
    eigenvals_real = metric_data['eigenvals'].real
    axes[0, 1].scatter(z_points[:, 0].cpu(), z_points[:, 1].cpu(), 
                       c=eigenvals_real[:, 0].cpu(), cmap='viridis', alpha=0.7)
    axes[0, 1].set_title("λ₁ (First Eigenvalue)")
    axes[0, 1].set_xlabel("z₁")
    axes[0, 1].set_ylabel("z₂")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Metric determinant
    axes[0, 2].scatter(z_points[:, 0].cpu(), z_points[:, 1].cpu(), 
                       c=metric_data['det_G'].cpu(), cmap='plasma', alpha=0.7)
    axes[0, 2].set_title("det(G)")
    axes[0, 2].set_xlabel("z₁")
    axes[0, 2].set_ylabel("z₂")
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Metric magnitude (trace)
    trace_G = torch.diagonal(metric_data['G_z'], dim1=-2, dim2=-1).sum(dim=-1)
    axes[1, 0].scatter(z_points[:, 0].cpu(), z_points[:, 1].cpu(), 
                       c=trace_G.cpu(), cmap='hot', alpha=0.7)
    axes[1, 0].set_title("tr(G)")
    axes[1, 0].set_xlabel("z₁")
    axes[1, 0].set_ylabel("z₂")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Condition number
    condition_number = eigenvals_real.max(dim=1)[0] / (eigenvals_real.min(dim=1)[0] + 1e-8)
    axes[1, 1].scatter(z_points[:, 0].cpu(), z_points[:, 1].cpu(), 
                       c=condition_number.cpu(), cmap='coolwarm', alpha=0.7)
    axes[1, 1].set_title("Condition Number")
    axes[1, 1].set_xlabel("z₁")
    axes[1, 1].set_ylabel("z₂")
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Metric anisotropy
    anisotropy = eigenvals_real.max(dim=1)[0] - eigenvals_real.min(dim=1)[0]
    axes[1, 2].scatter(z_points[:, 0].cpu(), z_points[:, 1].cpu(), 
                       c=anisotropy.cpu(), cmap='RdYlBu', alpha=0.7)
    axes[1, 2].set_title("Anisotropy (λ_max - λ_min)")
    axes[1, 2].set_xlabel("z₁")
    axes[1, 2].set_ylabel("z₂")
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"metric_manifold_{title.lower().replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return metric_data


def test_rhmc_sampling_with_real_data(model, n_samples=100, n_steps=50):
    """Test RHMC sampling with real data and visualize results."""
    print(f"\n🚀 Testing RHMC Sampling with Real Data")
    print("=" * 60)
    
    # Create RHMC sampler
    rhmc_sampler = RiemannianHMCSampler(model, mcmc_steps_nbr=n_steps, n_lf=10, eps_lf=0.02)
    
    # Test points in different regions
    test_regions = [
        torch.randn(n_samples, 2, device=model.device) * 0.5,  # Near origin
        torch.randn(n_samples, 2, device=model.device) * 2.0,  # Far from origin
        torch.randn(n_samples, 2, device=model.device) * 1.0 + torch.tensor([[1.0, 1.0]], device=model.device),  # Offset
    ]
    
    region_names = ["Near Origin", "Far from Origin", "Offset Region"]
    
    for i, (z_init, region_name) in enumerate(zip(test_regions, region_names)):
        print(f"\n--- Testing {region_name} ---")
        
        # Analyze initial metric
        print(f"Initial points range: [{z_init.min():.3f}, {z_init.max():.3f}]")
        initial_metric = analyze_metric_properties(model, z_init, f"Initial Metric - {region_name}")
        
        # Run RHMC sampling
        start_time = time.time()
        samples = rhmc_sampler.sample(n_samples)
        sampling_time = time.time() - start_time
        
        print(f"✅ RHMC sampling completed in {sampling_time:.3f}s")
        print(f"✅ Samples range: [{samples.min():.3f}, {samples.max():.3f}]")
        print(f"✅ Samples mean: {samples.mean(dim=0)}")
        print(f"✅ Samples std: {samples.std(dim=0)}")
        
        # Analyze final metric
        final_metric = analyze_metric_properties(model, samples, f"Final Metric - {region_name}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"RHMC Sampling - {region_name}", fontsize=16)
        
        # Plot 1: Initial vs Final points
        axes[0, 0].scatter(z_init[:, 0].cpu(), z_init[:, 1].cpu(), alpha=0.6, label='Initial', color='blue')
        axes[0, 0].scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), alpha=0.6, label='RHMC Samples', color='red')
        axes[0, 0].set_title("Initial vs RHMC Samples")
        axes[0, 0].set_xlabel("z₁")
        axes[0, 0].set_ylabel("z₂")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Metric eigenvalues at samples
        eigenvals_samples = torch.linalg.eigvals(final_metric['G_z']).real
        axes[0, 1].scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), 
                          c=eigenvals_samples[:, 0].cpu(), cmap='viridis', alpha=0.7)
        axes[0, 1].set_title("λ₁ at RHMC Samples")
        axes[0, 1].set_xlabel("z₁")
        axes[0, 1].set_ylabel("z₂")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Determinant at samples
        axes[1, 0].scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), 
                          c=final_metric['det_G'].cpu(), cmap='plasma', alpha=0.7)
        axes[1, 0].set_title("det(G) at RHMC Samples")
        axes[1, 0].set_xlabel("z₁")
        axes[1, 0].set_ylabel("z₂")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Sample distribution
        axes[1, 1].hist2d(samples[:, 0].cpu(), samples[:, 1].cpu(), bins=20, cmap='hot')
        axes[1, 1].set_title("RHMC Sample Distribution")
        axes[1, 1].set_xlabel("z₁")
        axes[1, 1].set_ylabel("z₂")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"rhmc_sampling_{region_name.lower().replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
        plt.show()


def test_posterior_sampling_with_real_data(model, n_samples=50):
    """Test RHMC posterior sampling with real data."""
    print(f"\n🎯 Testing RHMC Posterior Sampling with Real Data")
    print("=" * 60)
    
    # Create RHMC sampler
    rhmc_sampler = RiemannianHMCSampler(model, mcmc_steps_nbr=20, n_lf=5, eps_lf=0.01)
    
    # Test different posterior parameters
    test_cases = [
        {"mu": torch.tensor([[0.0, 0.0]], device=model.device), "log_var": torch.tensor([[0.1, 0.1]], device=model.device), "name": "Tight Posterior"},
        {"mu": torch.tensor([[1.0, 1.0]], device=model.device), "log_var": torch.tensor([[0.5, 0.5]], device=model.device), "name": "Offset Posterior"},
        {"mu": torch.tensor([[-1.0, 0.5]], device=model.device), "log_var": torch.tensor([[1.0, 0.2]], device=model.device), "name": "Asymmetric Posterior"},
    ]
    
    for case in test_cases:
        print(f"\n--- Testing {case['name']} ---")
        
        mu = case['mu']
        log_var = case['log_var']
        
        print(f"Posterior mean: {mu}")
        print(f"Posterior log_var: {log_var}")
        
        # Sample from posterior
        start_time = time.time()
        samples = rhmc_sampler.sample_posterior(mu, log_var)
        sampling_time = time.time() - start_time
        
        print(f"✅ Posterior sampling completed in {sampling_time:.3f}s")
        print(f"✅ Samples: {samples}")
        print(f"✅ Distance to mean: {torch.norm(samples - mu, dim=1)}")
        
        # Analyze metric at samples
        metric_data = analyze_metric_properties(model, samples, f"Posterior Metric - {case['name']}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"RHMC Posterior Sampling - {case['name']}", fontsize=16)
        
        # Plot 1: Samples vs mean
        axes[0].scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), alpha=0.7, label='RHMC Samples')
        axes[0].scatter(mu[:, 0].cpu(), mu[:, 1].cpu(), color='red', s=100, marker='*', label='Posterior Mean')
        axes[0].set_title("Samples vs Posterior Mean")
        axes[0].set_xlabel("z₁")
        axes[0].set_ylabel("z₂")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Metric eigenvalues
        eigenvals = torch.linalg.eigvals(metric_data['G_z']).real
        axes[1].scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), 
                       c=eigenvals[:, 0].cpu(), cmap='viridis', alpha=0.7)
        axes[1].set_title("λ₁ at Samples")
        axes[1].set_xlabel("z₁")
        axes[1].set_ylabel("z₂")
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Determinant
        axes[2].scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), 
                       c=metric_data['det_G'].cpu(), cmap='plasma', alpha=0.7)
        axes[2].set_title("det(G) at Samples")
        axes[2].set_xlabel("z₁")
        axes[2].set_ylabel("z₂")
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"rhmc_posterior_{case['name'].lower().replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
        plt.show()


def compare_metrics_and_sampling():
    """Compare different metrics and their sampling behavior."""
    print(f"\n🔬 Comparing Different Metrics and Sampling Behavior")
    print("=" * 60)
    
    # Create model
    model = RiemannianFlowVAE(
        input_dim=(3, 64, 64),
        latent_dim=2,
        n_flows=0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Test 1: Loaded metric
    print("\n--- Test 1: Loaded Metric ---")
    model.load_pretrained_components(
        encoder_path="data/pretrained/encoder_diverse_mlp_ld2_20250730_105419.pt",
        decoder_path="data/pretrained/decoder_diverse_mlp_ld2_20250730_105419.pt",
        metric_path="data/pretrained/metric_diverse_mlp_ld2_20250730_105420.pt"
    )
    
    # Test points
    z_test = torch.randn(100, 2, device=device) * 2.0
    visualize_metric_manifold(model, z_test, "Loaded Metric")
    test_rhmc_sampling_with_real_data(model, n_samples=50, n_steps=30)
    
    # Test 2: Diverse metric
    print("\n--- Test 2: Diverse Metric ---")
    centroids, metric_matrices = create_diverse_metric(latent_dim=2, n_centroids=20, device=device)
    model.load_pretrained_metrics_from_tensor(centroids, metric_matrices, temperature=0.5, regularization=0.01)
    
    z_test = torch.randn(100, 2, device=device) * 2.0
    visualize_metric_manifold(model, z_test, "Diverse Metric")
    test_rhmc_sampling_with_real_data(model, n_samples=50, n_steps=30)
    
    # Test 3: Very diverse metric
    print("\n--- Test 3: Very Diverse Metric ---")
    centroids = torch.randn(30, 2, device=device) * 3.0
    metric_matrices = []
    for i in range(30):
        # Create highly anisotropic matrices
        A = torch.randn(2, 2, device=device) * 2.0
        M = A @ A.T + torch.eye(2, device=device) * 0.1
        # Add extreme scales
        scale = torch.rand(1, device=device) * 10.0 + 0.1
        M = M * scale
        metric_matrices.append(M)
    
    metric_matrices = torch.stack(metric_matrices)
    model.load_pretrained_metrics_from_tensor(centroids, metric_matrices, temperature=0.3, regularization=0.001)
    
    z_test = torch.randn(100, 2, device=device) * 3.0
    visualize_metric_manifold(model, z_test, "Very Diverse Metric")
    test_rhmc_sampling_with_real_data(model, n_samples=50, n_steps=30)


def main():
    """Main investigation function."""
    print("🔍 RHMC Sampling Investigation with Real Data")
    print("=" * 60)
    
    # Compare different metrics and their sampling behavior
    compare_metrics_and_sampling()
    
    # Test posterior sampling
    print(f"\n🎯 Testing Posterior Sampling")
    print("=" * 60)
    
    model = RiemannianFlowVAE(
        input_dim=(3, 64, 64),
        latent_dim=2,
        n_flows=0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load diverse metric for posterior testing
    centroids, metric_matrices = create_diverse_metric(latent_dim=2, n_centroids=15, device=device)
    model.load_pretrained_metrics_from_tensor(centroids, metric_matrices, temperature=0.4, regularization=0.01)
    
    test_posterior_sampling_with_real_data(model, n_samples=30)
    
    print(f"\n✅ Investigation completed! Check the generated plots for analysis.")


if __name__ == "__main__":
    main() 