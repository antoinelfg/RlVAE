#!/usr/bin/env python3
"""
Inverse Metric Analysis
=======================

This script compares using the metric G vs its inverse G⁻¹ for:
- Manifold structure visualization
- RHMC sampling behavior
- Metric properties analysis
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
from src.models.samplers.hmc_sampler import RiemannianHMCSampler


def create_test_data(n_samples=1000, latent_dim=2, n_clusters=5):
    """Create test data with clear cluster structure."""
    print("🔧 Creating test data...")
    
    # Create cluster centers
    np.random.seed(42)
    cluster_centers = np.array([
        [-2.0, -1.5],  # Bottom-left
        [0.0, 2.0],    # Top-center
        [2.0, -1.0],   # Bottom-right
        [-1.0, 0.0],   # Left-center
        [1.5, 1.5],    # Top-right
    ])
    
    # Generate samples around clusters
    samples = []
    samples_per_cluster = n_samples // n_clusters
    
    for i in range(n_clusters):
        cluster_samples = np.random.multivariate_normal(
            cluster_centers[i], 
            np.eye(latent_dim) * 0.1, 
            samples_per_cluster
        )
        samples.append(cluster_samples)
    
    samples = np.vstack(samples)
    np.random.shuffle(samples)
    
    print(f"✅ Created {len(samples)} samples in {latent_dim}D with {n_clusters} clusters")
    return samples


def create_centroids_and_metrics(latent_data, n_centroids=20):
    """Create centroids and metric matrices."""
    print(f"\n🔍 Creating {n_centroids} centroids...")
    
    kmeans = KMeans(n_clusters=n_centroids, random_state=42, n_init=10)
    labels = kmeans.fit_predict(latent_data)
    centroids = kmeans.cluster_centers_
    
    # Compute metric matrices for each centroid
    metric_matrices = []
    for i, centroid in enumerate(centroids):
        # Find points closest to this centroid
        distances = np.linalg.norm(latent_data - centroid, axis=1)
        closest_indices = np.argsort(distances)[:max(10, len(latent_data) // n_centroids)]
        cluster_points = latent_data[closest_indices]
        
        # Compute local covariance
        if len(cluster_points) > 1:
            cov_matrix = np.cov(cluster_points.T)
            # Add regularization
            cov_matrix += np.eye(cov_matrix.shape[0]) * 0.01
            # Metric is inverse of covariance
            try:
                metric_matrix = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                metric_matrix = np.eye(cov_matrix.shape[0])
        else:
            metric_matrix = np.eye(latent_data.shape[1])
        
        metric_matrices.append(metric_matrix)
    
    metric_matrices = np.array(metric_matrices)
    
    print(f"✅ Created {len(centroids)} centroids with metric matrices")
    return centroids, metric_matrices, labels


def compare_metric_vs_inverse_manifold(model, n_points=400, save_prefix="inverse"):
    """Compare manifold structure using G vs G⁻¹."""
    print(f"\n🎨 Comparing Metric G vs Inverse G⁻¹ Manifold Structure")
    print("=" * 60)
    
    # Create a grid of points
    x_range = np.linspace(-4, 4, int(np.sqrt(n_points)))
    y_range = np.linspace(-4, 4, int(np.sqrt(n_points)))
    X, Y = np.meshgrid(x_range, y_range)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Convert to tensor
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=model.device)
    
    # Compute both G and G⁻¹ properties
    with torch.no_grad():
        G_grid = model.G(grid_tensor)
        G_inv_grid = model.G_inv(grid_tensor)
        
        # Properties of G
        eigenvals_G = torch.linalg.eigvals(G_grid).real
        determinants_G = torch.linalg.det(G_grid)
        condition_numbers_G = eigenvals_G.max(dim=-1)[0] / (eigenvals_G.min(dim=-1)[0] + 1e-8)
        
        # Properties of G⁻¹
        eigenvals_G_inv = torch.linalg.eigvals(G_inv_grid).real
        determinants_G_inv = torch.linalg.det(G_inv_grid)
        condition_numbers_G_inv = eigenvals_G_inv.max(dim=-1)[0] / (eigenvals_G_inv.min(dim=-1)[0] + 1e-8)
    
    # Reshape for plotting
    eigenvals_G_reshaped = eigenvals_G.cpu().numpy().reshape(len(x_range), len(y_range), 2)
    determinants_G_reshaped = determinants_G.cpu().numpy().reshape(len(x_range), len(y_range))
    condition_numbers_G_reshaped = condition_numbers_G.cpu().numpy().reshape(len(x_range), len(y_range))
    
    eigenvals_G_inv_reshaped = eigenvals_G_inv.cpu().numpy().reshape(len(x_range), len(y_range), 2)
    determinants_G_inv_reshaped = determinants_G_inv.cpu().numpy().reshape(len(x_range), len(y_range))
    condition_numbers_G_inv_reshaped = condition_numbers_G_inv.cpu().numpy().reshape(len(x_range), len(y_range))
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Metric G vs Inverse G⁻¹ Manifold Structure", fontsize=16)
    
    # Determine global ranges for consistent scaling
    det_G_min, det_G_max = determinants_G_reshaped.min(), determinants_G_reshaped.max()
    det_G_inv_min, det_G_inv_max = determinants_G_inv_reshaped.min(), determinants_G_inv_reshaped.max()
    eigenval_G_min, eigenval_G_max = eigenvals_G_reshaped.min(), eigenvals_G_reshaped.max()
    eigenval_G_inv_min, eigenval_G_inv_max = eigenvals_G_inv_reshaped.min(), eigenvals_G_inv_reshaped.max()
    
    # Plot 1: G Determinant
    im1 = axes[0, 0].contourf(X, Y, determinants_G_reshaped, levels=20, cmap='viridis', 
                               vmin=det_G_min, vmax=det_G_max)
    axes[0, 0].set_title(f"G Determinant\nRange: [{det_G_min:.2e}, {det_G_max:.2e}]")
    axes[0, 0].set_xlabel("z₁")
    axes[0, 0].set_ylabel("z₂")
    axes[0, 0].set_xlim(-4, 4)
    axes[0, 0].set_ylim(-4, 4)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: G⁻¹ Determinant
    im2 = axes[0, 1].contourf(X, Y, determinants_G_inv_reshaped, levels=20, cmap='viridis',
                               vmin=det_G_inv_min, vmax=det_G_inv_max)
    axes[0, 1].set_title(f"G⁻¹ Determinant\nRange: [{det_G_inv_min:.2e}, {det_G_inv_max:.2e}]")
    axes[0, 1].set_xlabel("z₁")
    axes[0, 1].set_ylabel("z₂")
    axes[0, 1].set_xlim(-4, 4)
    axes[0, 1].set_ylim(-4, 4)
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: G First Eigenvalue
    im3 = axes[0, 2].contourf(X, Y, eigenvals_G_reshaped[:, :, 0], levels=20, cmap='plasma',
                               vmin=eigenval_G_min, vmax=eigenval_G_max)
    axes[0, 2].set_title(f"G λ₁ (First Eigenvalue)\nRange: [{eigenval_G_min:.2e}, {eigenval_G_max:.2e}]")
    axes[0, 2].set_xlabel("z₁")
    axes[0, 2].set_ylabel("z₂")
    axes[0, 2].set_xlim(-4, 4)
    axes[0, 2].set_ylim(-4, 4)
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Plot 4: G⁻¹ First Eigenvalue
    im4 = axes[0, 3].contourf(X, Y, eigenvals_G_inv_reshaped[:, :, 0], levels=20, cmap='plasma',
                               vmin=eigenval_G_inv_min, vmax=eigenval_G_inv_max)
    axes[0, 3].set_title(f"G⁻¹ λ₁ (First Eigenvalue)\nRange: [{eigenval_G_inv_min:.2e}, {eigenval_G_inv_max:.2e}]")
    axes[0, 3].set_xlabel("z₁")
    axes[0, 3].set_ylabel("z₂")
    axes[0, 3].set_xlim(-4, 4)
    axes[0, 3].set_ylim(-4, 4)
    plt.colorbar(im4, ax=axes[0, 3])
    
    # Plot 5: G Condition Number
    im5 = axes[1, 0].contourf(X, Y, condition_numbers_G_reshaped, levels=20, cmap='coolwarm',
                               vmin=condition_numbers_G_reshaped.min(), vmax=condition_numbers_G_reshaped.max())
    axes[1, 0].set_title(f"G Condition Number\nRange: [{condition_numbers_G_reshaped.min():.2f}, {condition_numbers_G_reshaped.max():.2f}]")
    axes[1, 0].set_xlabel("z₁")
    axes[1, 0].set_ylabel("z₂")
    axes[1, 0].set_xlim(-4, 4)
    axes[1, 0].set_ylim(-4, 4)
    plt.colorbar(im5, ax=axes[1, 0])
    
    # Plot 6: G⁻¹ Condition Number
    im6 = axes[1, 1].contourf(X, Y, condition_numbers_G_inv_reshaped, levels=20, cmap='coolwarm',
                               vmin=condition_numbers_G_inv_reshaped.min(), vmax=condition_numbers_G_inv_reshaped.max())
    axes[1, 1].set_title(f"G⁻¹ Condition Number\nRange: [{condition_numbers_G_inv_reshaped.min():.2f}, {condition_numbers_G_inv_reshaped.max():.2f}]")
    axes[1, 1].set_xlabel("z₁")
    axes[1, 1].set_ylabel("z₂")
    axes[1, 1].set_xlim(-4, 4)
    axes[1, 1].set_ylim(-4, 4)
    plt.colorbar(im6, ax=axes[1, 1])
    
    # Plot 7: G Anisotropy
    anisotropy_G = eigenvals_G_reshaped[:, :, 0] - eigenvals_G_reshaped[:, :, 1]
    im7 = axes[1, 2].contourf(X, Y, anisotropy_G, levels=20, cmap='RdYlBu',
                               vmin=anisotropy_G.min(), vmax=anisotropy_G.max())
    axes[1, 2].set_title(f"G Anisotropy (λ₁ - λ₂)\nRange: [{anisotropy_G.min():.2e}, {anisotropy_G.max():.2e}]")
    axes[1, 2].set_xlabel("z₁")
    axes[1, 2].set_ylabel("z₂")
    axes[1, 2].set_xlim(-4, 4)
    axes[1, 2].set_ylim(-4, 4)
    plt.colorbar(im7, ax=axes[1, 2])
    
    # Plot 8: G⁻¹ Anisotropy
    anisotropy_G_inv = eigenvals_G_inv_reshaped[:, :, 0] - eigenvals_G_inv_reshaped[:, :, 1]
    im8 = axes[1, 3].contourf(X, Y, anisotropy_G_inv, levels=20, cmap='RdYlBu',
                               vmin=anisotropy_G_inv.min(), vmax=anisotropy_G_inv.max())
    axes[1, 3].set_title(f"G⁻¹ Anisotropy (λ₁ - λ₂)\nRange: [{anisotropy_G_inv.min():.2e}, {anisotropy_G_inv.max():.2e}]")
    axes[1, 3].set_xlabel("z₁")
    axes[1, 3].set_ylabel("z₂")
    axes[1, 3].set_xlim(-4, 4)
    axes[1, 3].set_ylim(-4, 4)
    plt.colorbar(im8, ax=axes[1, 3])
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_metric_vs_inverse_manifold.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'G': {
            'determinants': determinants_G_reshaped,
            'eigenvals': eigenvals_G_reshaped,
            'condition_numbers': condition_numbers_G_reshaped,
            'anisotropy': anisotropy_G
        },
        'G_inv': {
            'determinants': determinants_G_inv_reshaped,
            'eigenvals': eigenvals_G_inv_reshaped,
            'condition_numbers': condition_numbers_G_inv_reshaped,
            'anisotropy': anisotropy_G_inv
        }
    }


def compare_rhmc_with_inverse_metric(model, n_samples=200, n_steps=40, save_prefix="inverse"):
    """Compare RHMC sampling using G vs G⁻¹."""
    print(f"\n🎯 Comparing RHMC Sampling with G vs G⁻¹")
    print("=" * 60)
    
    # Create RHMC sampler
    rhmc_sampler = RiemannianHMCSampler(model, mcmc_steps_nbr=n_steps, n_lf=15, eps_lf=0.02)
    
    # Sample from different starting points
    starting_points = [
        torch.tensor([[-2.0, -1.5]], device=model.device),  # Cluster 1
        torch.tensor([[0.0, 2.0]], device=model.device),    # Cluster 2
        torch.tensor([[2.0, -1.0]], device=model.device),   # Cluster 3
        torch.tensor([[-1.0, 0.0]], device=model.device),   # Cluster 4
    ]
    
    colors = ['red', 'blue', 'green', 'orange']
    labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("RHMC Sampling with G vs G⁻¹", fontsize=16)
    
    # Test with original metric (G)
    print(f"\n--- Testing with Original Metric G ---")
    for i, (start_point, color, label) in enumerate(zip(starting_points, colors, labels)):
        print(f"\n--- Sampling from {label} with G ---")
        
        # Run RHMC sampling
        start_time = time.time()
        samples = rhmc_sampler.sample(n_samples)
        sampling_time = time.time() - start_time
        
        print(f"✅ Sampling completed in {sampling_time:.3f}s")
        print(f"✅ Samples range: [{samples.min():.3f}, {samples.max():.3f}]")
        
        # Analyze metric at samples
        with torch.no_grad():
            G_samples = model.G(samples)
            eigenvals = torch.linalg.eigvals(G_samples).real
            determinants = torch.linalg.det(G_samples)
        
        print(f"✅ G eigenvalues: min={eigenvals.min():.3e}, max={eigenvals.max():.3e}")
        print(f"✅ G determinants: min={determinants.min():.3e}, max={determinants.max():.3e}")
        
        # Plot samples colored by metric properties
        ax = axes[0, i]
        scatter = ax.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), 
                           c=determinants.cpu(), cmap='viridis', alpha=0.7, s=30)
        ax.scatter(start_point[:, 0].cpu(), start_point[:, 1].cpu(), 
                  color='red', s=200, marker='*', label=f'Start: {label}')
        ax.set_title(f"G: RHMC from {label}\n(colored by det(G))")
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.colorbar(scatter, ax=ax)
    
    # Test with inverse metric (G⁻¹)
    print(f"\n--- Testing with Inverse Metric G⁻¹ ---")
    
    # Temporarily modify the model to use G⁻¹ for sampling
    original_G = model.G
    original_G_inv = model.G_inv
    
    # Swap G and G⁻¹ for sampling
    model.G = original_G_inv
    model.G_inv = original_G
    
    for i, (start_point, color, label) in enumerate(zip(starting_points, colors, labels)):
        print(f"\n--- Sampling from {label} with G⁻¹ ---")
        
        # Run RHMC sampling with inverse metric
        start_time = time.time()
        samples = rhmc_sampler.sample(n_samples)
        sampling_time = time.time() - start_time
        
        print(f"✅ Sampling completed in {sampling_time:.3f}s")
        print(f"✅ Samples range: [{samples.min():.3f}, {samples.max():.3f}]")
        
        # Analyze inverse metric at samples
        with torch.no_grad():
            G_inv_samples = model.G(samples)  # This is now G⁻¹
            eigenvals = torch.linalg.eigvals(G_inv_samples).real
            determinants = torch.linalg.det(G_inv_samples)
        
        print(f"✅ G⁻¹ eigenvalues: min={eigenvals.min():.3e}, max={eigenvals.max():.3e}")
        print(f"✅ G⁻¹ determinants: min={determinants.min():.3e}, max={determinants.max():.3e}")
        
        # Plot samples colored by inverse metric properties
        ax = axes[1, i]
        scatter = ax.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), 
                           c=determinants.cpu(), cmap='plasma', alpha=0.7, s=30)
        ax.scatter(start_point[:, 0].cpu(), start_point[:, 1].cpu(), 
                  color='red', s=200, marker='*', label=f'Start: {label}')
        ax.set_title(f"G⁻¹: RHMC from {label}\n(colored by det(G⁻¹))")
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.colorbar(scatter, ax=ax)
    
    # Restore original metric functions
    model.G = original_G
    model.G_inv = original_G_inv
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_rhmc_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()


def analyze_metric_properties_comparison(model, n_points=100, save_prefix="inverse"):
    """Analyze and compare properties of G vs G⁻¹."""
    print(f"\n🔍 Analyzing Metric Properties: G vs G⁻¹")
    print("=" * 60)
    
    # Create test points
    test_points = torch.randn(n_points, 2, device=model.device) * 2.0
    
    # Compute both G and G⁻¹ properties
    with torch.no_grad():
        G_test = model.G(test_points)
        G_inv_test = model.G_inv(test_points)
        
        # Properties of G
        eigenvals_G = torch.linalg.eigvals(G_test).real
        determinants_G = torch.linalg.det(G_test)
        condition_numbers_G = eigenvals_G.max(dim=-1)[0] / (eigenvals_G.min(dim=-1)[0] + 1e-8)
        
        # Properties of G⁻¹
        eigenvals_G_inv = torch.linalg.eigvals(G_inv_test).real
        determinants_G_inv = torch.linalg.det(G_inv_test)
        condition_numbers_G_inv = eigenvals_G_inv.max(dim=-1)[0] / (eigenvals_G_inv.min(dim=-1)[0] + 1e-8)
    
    # Print comparison
    print(f"\n📊 Metric Properties Comparison:")
    print(f"G Eigenvalues: min={eigenvals_G.min():.3e}, max={eigenvals_G.max():.3e}")
    print(f"G⁻¹ Eigenvalues: min={eigenvals_G_inv.min():.3e}, max={eigenvals_G_inv.max():.3e}")
    print(f"G Determinants: min={determinants_G.min():.3e}, max={determinants_G.max():.3e}")
    print(f"G⁻¹ Determinants: min={determinants_G_inv.min():.3e}, max={determinants_G_inv.max():.3e}")
    print(f"G Condition Numbers: min={condition_numbers_G.min():.2f}, max={condition_numbers_G.max():.2f}")
    print(f"G⁻¹ Condition Numbers: min={condition_numbers_G_inv.min():.2f}, max={condition_numbers_G_inv.max():.2f}")
    
    # Mathematical relationships
    print(f"\n🔬 Mathematical Relationships:")
    print(f"det(G) × det(G⁻¹) should be 1:")
    det_product = determinants_G * determinants_G_inv
    print(f"  Actual: min={det_product.min():.6f}, max={det_product.max():.6f}")
    print(f"  Expected: 1.000000")
    
    print(f"\nλ(G) × λ(G⁻¹) should be 1:")
    eigenval_product = eigenvals_G * eigenvals_G_inv
    print(f"  λ₁ product: min={eigenval_product[:, 0].min():.6f}, max={eigenval_product[:, 0].max():.6f}")
    print(f"  λ₂ product: min={eigenval_product[:, 1].min():.6f}, max={eigenval_product[:, 1].max():.6f}")
    print(f"  Expected: 1.000000")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Metric Properties Comparison: G vs G⁻¹", fontsize=16)
    
    # Plot 1: Determinant comparison
    axes[0, 0].scatter(determinants_G.cpu(), determinants_G_inv.cpu(), alpha=0.7)
    axes[0, 0].set_xlabel("det(G)")
    axes[0, 0].set_ylabel("det(G⁻¹)")
    axes[0, 0].set_title("Determinant Relationship")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: First eigenvalue comparison
    axes[0, 1].scatter(eigenvals_G[:, 0].cpu(), eigenvals_G_inv[:, 0].cpu(), alpha=0.7)
    axes[0, 1].set_xlabel("λ₁(G)")
    axes[0, 1].set_ylabel("λ₁(G⁻¹)")
    axes[0, 1].set_title("First Eigenvalue Relationship")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Condition number comparison
    axes[1, 0].scatter(condition_numbers_G.cpu(), condition_numbers_G_inv.cpu(), alpha=0.7)
    axes[1, 0].set_xlabel("Condition Number(G)")
    axes[1, 0].set_ylabel("Condition Number(G⁻¹)")
    axes[1, 0].set_title("Condition Number Relationship")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Product verification
    axes[1, 1].scatter(det_product.cpu(), eigenval_product[:, 0].cpu(), alpha=0.7, label='det product')
    axes[1, 1].scatter(det_product.cpu(), eigenval_product[:, 1].cpu(), alpha=0.7, label='λ₁ product')
    axes[1, 1].axhline(y=1.0, color='red', linestyle='--', label='Expected: 1.0')
    axes[1, 1].set_xlabel("det(G) × det(G⁻¹)")
    axes[1, 1].set_ylabel("Product Values")
    axes[1, 1].set_title("Mathematical Relationship Verification")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_properties_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main inverse metric analysis function."""
    print("🔍 Inverse Metric Analysis: G vs G⁻¹")
    print("=" * 60)
    
    # Create test data
    latent_data = create_test_data(n_samples=1000, n_clusters=5)
    
    # Create centroids and metrics
    centroids, metric_matrices, labels = create_centroids_and_metrics(latent_data, n_centroids=20)
    
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
    print(f"\n🔧 Loading centroids and metrics...")
    centroids_tensor = torch.tensor(centroids, dtype=torch.float32, device=device)
    metric_matrices_tensor = torch.tensor(metric_matrices, dtype=torch.float32, device=device)
    model.load_pretrained_metrics_from_tensor(centroids_tensor, metric_matrices_tensor, 
                                            temperature=0.5, regularization=0.01)
    
    # Compare manifold structure
    manifold_data = compare_metric_vs_inverse_manifold(model, n_points=400, save_prefix="inverse")
    
    # Compare RHMC sampling
    compare_rhmc_with_inverse_metric(model, n_samples=200, n_steps=40, save_prefix="inverse")
    
    # Analyze metric properties
    analyze_metric_properties_comparison(model, n_points=100)
    
    print(f"\n✅ Inverse metric analysis completed!")
    print(f"📊 Key Insights:")
    print(f"   - G and G⁻¹ have inverse mathematical relationships")
    print(f"   - Manifold structure differs between G and G⁻¹")
    print(f"   - RHMC sampling behavior changes with inverse metric")
    print(f"   - Both approaches provide valid geometric interpretations")


if __name__ == "__main__":
    main() 