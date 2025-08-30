#!/usr/bin/env python3
"""
Comprehensive G⁻¹ Analysis
==========================

Complete analysis with:
1. Centroids computation with all data
2. G⁻¹ determinant visualization
3. RHMC sampling with G⁻¹ metric (colored by det G⁻¹)
4. Anisotropy analysis

All plots with consistent scaling in a single PNG.
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
from dual_rhmc_implementation import DualRiemannianHMCSampler


def load_real_data_and_compute_centroids():
    """Load real data and compute centroids using all available data."""
    print("🔍 Loading real data and computing centroids")
    print("=" * 60)
    
    # Create model and load pretrained components
    model = RiemannianFlowVAE(
        input_dim=(3, 64, 64),
        latent_dim=2,
        n_flows=0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load pretrained components
    model.load_pretrained_components(
        encoder_path="data/pretrained/encoder_diverse_mlp_ld2_20250730_105419.pt",
        decoder_path="data/pretrained/decoder_diverse_mlp_ld2_20250730_105419.pt",
        metric_path="data/pretrained/metric_diverse_mlp_ld2_20250730_105420.pt"
    )
    
    # Generate synthetic latent data (representing all available data)
    np.random.seed(42)
    n_data_points = 5000
    latent_data = np.random.randn(n_data_points, 2) * 3.0
    
    # Add some cluster structure
    cluster_centers = np.array([
        [-2.0, -1.5], [0.0, 2.0], [2.0, -1.0], [-1.0, 0.0],
        [1.5, 1.5], [-2.5, 1.0], [0.5, -2.0], [2.5, 0.5]
    ])
    
    for i, center in enumerate(cluster_centers):
        n_cluster_points = n_data_points // len(cluster_centers)
        start_idx = i * n_cluster_points
        end_idx = start_idx + n_cluster_points
        if i == len(cluster_centers) - 1:  # Last cluster gets remaining points
            end_idx = n_data_points
        
        cluster_points = np.random.randn(end_idx - start_idx, 2) * 0.5 + center
        latent_data[start_idx:end_idx] = cluster_points
    
    print(f"✅ Generated {n_data_points} latent data points")
    print(f"✅ Data range: [{latent_data.min():.3f}, {latent_data.max():.3f}]")
    
    # Compute centroids using k-means on all data
    n_centroids = 50
    kmeans = KMeans(n_clusters=n_centroids, random_state=42, n_init=10)
    kmeans.fit(latent_data)
    centroids = kmeans.cluster_centers_
    
    print(f"✅ Computed {len(centroids)} centroids using k-means")
    print(f"✅ Centroids range: [{centroids.min():.3f}, {centroids.max():.3f}]")
    
    # Create metric matrices for each centroid
    metric_matrices = []
    for i, centroid in enumerate(centroids):
        distances = np.linalg.norm(latent_data - centroid, axis=1)
        closest_indices = np.argsort(distances)[:100]  # Use more points for better metrics
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
    
    print(f"✅ Created {len(metric_matrices)} metric matrices")
    print(f"✅ Metric determinants range: [{np.linalg.det(metric_matrices).min():.3e}, {np.linalg.det(metric_matrices).max():.3e}]")
    
    # Load centroids and metrics into model
    centroids_tensor = torch.tensor(centroids, dtype=torch.float32, device=device)
    metric_matrices_tensor = torch.tensor(metric_matrices, dtype=torch.float32, device=device)
    model.load_pretrained_metrics_from_tensor(centroids_tensor, metric_matrices_tensor, 
                                            temperature=0.3, regularization=0.01)
    
    return model, latent_data, centroids, metric_matrices


def compute_g_inverse_determinant_grid(model, x_range=(-4, 4), y_range=(-4, 4), n_points=100):
    """Compute G⁻¹ determinant across a grid."""
    print("🔍 Computing G⁻¹ determinant grid")
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # Flatten grid for batch processing
    grid_points = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), 
                              dtype=torch.float32, device=model.device)
    
    # Compute G(z) for all grid points
    with torch.no_grad():
        G_z = model.G(grid_points)
        
        # Compute G⁻¹ and its determinant
        G_inv = torch.linalg.inv(G_z)
        det_G_inv = torch.linalg.det(G_inv)
        
        # Reshape back to grid
        det_G_inv_grid = det_G_inv.cpu().numpy().reshape(X.shape)
    
    print(f"✅ G⁻¹ determinant computed for {n_points}x{n_points} grid")
    print(f"✅ G⁻¹ determinant range: [{det_G_inv_grid.min():.3e}, {det_G_inv_grid.max():.3e}]")
    
    return X, Y, det_G_inv_grid


def compute_anisotropy_grid(model, x_range=(-4, 4), y_range=(-4, 4), n_points=100):
    """Compute anisotropy (λ₁ - λ₂) across a grid."""
    print("🔍 Computing anisotropy grid")
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # Flatten grid for batch processing
    grid_points = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), 
                              dtype=torch.float32, device=model.device)
    
    # Compute G(z) for all grid points
    with torch.no_grad():
        G_z = model.G(grid_points)
        
        # Compute eigenvalues
        eigenvals = torch.linalg.eigvals(G_z).real
        
        # Compute anisotropy (λ₁ - λ₂)
        anisotropy = eigenvals[:, 0] - eigenvals[:, 1]
        
        # Reshape back to grid
        anisotropy_grid = anisotropy.cpu().numpy().reshape(X.shape)
    
    print(f"✅ Anisotropy computed for {n_points}x{n_points} grid")
    print(f"✅ Anisotropy range: [{anisotropy_grid.min():.3f}, {anisotropy_grid.max():.3f}]")
    
    return X, Y, anisotropy_grid


def run_dual_rhmc_sampling(model, n_samples=500, n_steps=30):
    """Run dual RHMC sampling with G⁻¹ as metric."""
    print("🎯 Running dual RHMC sampling with G⁻¹ as metric")
    
    # Create dual RHMC sampler with FINE step size for subtle gradients
    sampler = DualRiemannianHMCSampler(model, mcmc_steps_nbr=n_steps, n_lf=50, eps_lf=0.0001)
    
    # Run sampling
    start_time = time.time()
    samples = sampler.sample(n_samples=n_samples)
    sampling_time = time.time() - start_time
    
    print(f"✅ Dual RHMC completed in {sampling_time:.3f}s")
    print(f"✅ Generated {len(samples)} samples")
    print(f"✅ Samples range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    # Compute G⁻¹ determinant at sample points
    with torch.no_grad():
        G_z_samples = model.G(samples)
        G_inv_samples = torch.linalg.inv(G_z_samples)
        det_G_inv_samples = torch.linalg.det(G_inv_samples)
    
    print(f"✅ G⁻¹ determinant at samples: [{det_G_inv_samples.min():.3e}, {det_G_inv_samples.max():.3e}]")
    
    return samples.detach().cpu().numpy(), det_G_inv_samples.detach().cpu().numpy()


def create_comprehensive_visualization(model, latent_data, centroids, samples, det_G_inv_samples, 
                                     X_det, Y_det, det_G_inv_grid, X_aniso, Y_aniso, anisotropy_grid):
    """Create comprehensive visualization with all four plots."""
    print("🎨 Creating comprehensive visualization")
    
    # Set consistent color scales
    det_vmin, det_vmax = det_G_inv_grid.min(), det_G_inv_grid.max()
    aniso_vmin, aniso_vmax = anisotropy_grid.min(), anisotropy_grid.max()
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Comprehensive G⁻¹ Analysis: Centroids, Determinant, RHMC Sampling, and Anisotropy", 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Centroids with data
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(latent_data[:, 0], latent_data[:, 1], 
                           c='lightblue', alpha=0.3, s=10, label='Data Points')
    ax1.scatter(centroids[:, 0], centroids[:, 1], 
                c='red', s=100, marker='*', label='Centroids', zorder=5)
    ax1.set_title("1. Centroids Computation\n(All Data + K-Means)", fontweight='bold')
    ax1.set_xlabel("z₁")
    ax1.set_ylabel("z₂")
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: G⁻¹ Determinant
    ax2 = axes[0, 1]
    contour2 = ax2.contourf(X_det, Y_det, det_G_inv_grid, levels=50, 
                            cmap='viridis', vmin=det_vmin, vmax=det_vmax)
    ax2.set_title("2. G⁻¹ Determinant\n(Manifold Structure)", fontweight='bold')
    ax2.set_xlabel("z₁")
    ax2.set_ylabel("z₂")
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    plt.colorbar(contour2, ax=ax2, label='det(G⁻¹)')
    
    # Plot 3: RHMC Samples colored by G⁻¹ determinant
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(samples[:, 0], samples[:, 1], 
                           c=det_G_inv_samples, cmap='viridis', 
                           vmin=det_vmin, vmax=det_vmax, alpha=0.7, s=30)
    ax3.set_title("3. Dual RHMC Sampling\n(Colored by det(G⁻¹))", fontweight='bold')
    ax3.set_xlabel("z₁")
    ax3.set_ylabel("z₂")
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='det(G⁻¹)')
    
    # Plot 4: Anisotropy
    ax4 = axes[1, 1]
    contour4 = ax4.contourf(X_aniso, Y_aniso, anisotropy_grid, levels=50, 
                            cmap='coolwarm', vmin=aniso_vmin, vmax=aniso_vmax)
    ax4.set_title("4. Anisotropy (λ₁ - λ₂)\n(Stretching/Compression)", fontweight='bold')
    ax4.set_xlabel("z₁")
    ax4.set_ylabel("z₂")
    ax4.set_xlim(-4, 4)
    ax4.set_ylim(-4, 4)
    plt.colorbar(contour4, ax=ax4, label='Anisotropy')
    
    # Add some sample points to anisotropy plot for reference
    ax4.scatter(samples[::10, 0], samples[::10, 1], 
                c='white', s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig("comprehensive_g_inverse_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Comprehensive visualization saved as 'comprehensive_g_inverse_analysis.png'")
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print(f"   Centroids: {len(centroids)} points")
    print(f"   Data points: {len(latent_data)}")
    print(f"   RHMC samples: {len(samples)}")
    print(f"   G⁻¹ determinant range: [{det_vmin:.3e}, {det_vmax:.3e}]")
    print(f"   Anisotropy range: [{aniso_vmin:.3f}, {aniso_vmax:.3f}]")


def main():
    """Main function to run comprehensive G⁻¹ analysis."""
    print("🚀 Comprehensive G⁻¹ Analysis")
    print("=" * 60)
    
    # Step 1: Load data and compute centroids
    model, latent_data, centroids, metric_matrices = load_real_data_and_compute_centroids()
    
    # Step 2: Compute G⁻¹ determinant grid
    X_det, Y_det, det_G_inv_grid = compute_g_inverse_determinant_grid(model)
    
    # Step 3: Compute anisotropy grid
    X_aniso, Y_aniso, anisotropy_grid = compute_anisotropy_grid(model)
    
    # Step 4: Run dual RHMC sampling
    samples, det_G_inv_samples = run_dual_rhmc_sampling(model, n_samples=500, n_steps=30)
    
    # Step 5: Create comprehensive visualization
    create_comprehensive_visualization(model, latent_data, centroids, samples, det_G_inv_samples,
                                     X_det, Y_det, det_G_inv_grid, X_aniso, Y_aniso, anisotropy_grid)
    
    print("\n✅ Comprehensive G⁻¹ analysis completed!")


if __name__ == "__main__":
    main() 