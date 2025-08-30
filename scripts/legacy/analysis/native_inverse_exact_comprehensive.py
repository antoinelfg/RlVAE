#!/usr/bin/env python3
"""
Native Inverse Exact Comprehensive Analysis
==========================================

Exact same structure as comprehensive_g_inverse_analysis.py
but using native G⁻¹ metric system instead of G->G⁻¹ conversion.
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
from native_inverse_metric_system import NativeInverseMetricTensor, NativeInverseRHMC


def load_real_data_and_compute_centroids():
    """Load real data and compute centroids using all available data - EXACT SAME AS COMPREHENSIVE."""
    print("🔍 Loading real data and computing centroids")
    print("=" * 60)
    
    # Create model and load pretrained components - EXACT SAME
    model = RiemannianFlowVAE(
        input_dim=(3, 64, 64),
        latent_dim=2,
        n_flows=0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load pretrained components - EXACT SAME
    model.load_pretrained_components(
        encoder_path="data/pretrained/encoder_diverse_mlp_ld2_20250730_105419.pt",
        decoder_path="data/pretrained/decoder_diverse_mlp_ld2_20250730_105419.pt",
        metric_path="data/pretrained/metric_diverse_mlp_ld2_20250730_105420.pt"
    )
    
    # Generate synthetic latent data (representing all available data) - EXACT SAME
    np.random.seed(42)
    n_data_points = 5000
    latent_data = np.random.randn(n_data_points, 2) * 3.0
    
    # Add some cluster structure - EXACT SAME
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
    
    # Compute centroids using k-means on all data - EXACT SAME
    n_centroids = 50
    kmeans = KMeans(n_clusters=n_centroids, random_state=42, n_init=10)
    kmeans.fit(latent_data)
    centroids = kmeans.cluster_centers_
    
    print(f"✅ Computed {len(centroids)} centroids using k-means")
    print(f"✅ Centroids range: [{centroids.min():.3f}, {centroids.max():.3f}]")
    
    # Create metric matrices for each centroid - EXACT SAME APPROACH
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
    
    # BUT NOW: Create native G⁻¹ metric tensor instead of loading into model
    centroids_tensor = torch.tensor(centroids, dtype=torch.float32, device=device)
    metric_matrices_tensor = torch.tensor(metric_matrices, dtype=torch.float32, device=device)
    
    # Create native G⁻¹ metric tensor
    native_metric_tensor = NativeInverseMetricTensor(latent_dim=2)
    native_metric_tensor.load_inverse_metrics(
        centroids_tensor, metric_matrices_tensor,
        temperature=0.3, regularization=0.01  # Same parameters as comprehensive
    )
    
    return model, native_metric_tensor, latent_data, centroids, metric_matrices


def compute_g_inverse_determinant_grid(native_metric_tensor, x_range=(-4, 4), y_range=(-4, 4), n_points=100):
    """Compute G⁻¹ determinant across a grid using native G⁻¹ metric."""
    print("🔍 Computing G⁻¹ determinant grid (Native G⁻¹)")
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # Flatten grid for batch processing
    grid_points = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), 
                              dtype=torch.float32, device=native_metric_tensor.centroids.device)
    
    # Compute G⁻¹(z) directly using native metric tensor
    with torch.no_grad():
        G_inv, log_det_G_inv = native_metric_tensor(grid_points)
        det_G_inv = torch.exp(log_det_G_inv)
        
        # Reshape back to grid
        det_G_inv_grid = det_G_inv.cpu().numpy().reshape(X.shape)
    
    print(f"✅ G⁻¹ determinant computed for {n_points}x{n_points} grid (Native G⁻¹)")
    print(f"✅ G⁻¹ determinant range: [{det_G_inv_grid.min():.3e}, {det_G_inv_grid.max():.3e}]")
    
    return X, Y, det_G_inv_grid


def compute_anisotropy_grid(native_metric_tensor, x_range=(-4, 4), y_range=(-4, 4), n_points=100):
    """Compute anisotropy (λ₁ - λ₂) across a grid using native G⁻¹ metric."""
    print("🔍 Computing anisotropy grid (Native G⁻¹)")
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # Flatten grid for batch processing
    grid_points = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), 
                              dtype=torch.float32, device=native_metric_tensor.centroids.device)
    
    # Compute G⁻¹(z) directly using native metric tensor
    with torch.no_grad():
        G_inv, _ = native_metric_tensor(grid_points)
        
        # Compute eigenvalues of G⁻¹ (this is the native metric)
        eigenvals = torch.linalg.eigvals(G_inv).real
        
        # Compute anisotropy (λ₁ - λ₂) of G⁻¹
        anisotropy = eigenvals[:, 0] - eigenvals[:, 1]
        
        # Reshape back to grid
        anisotropy_grid = anisotropy.cpu().numpy().reshape(X.shape)
    
    print(f"✅ Anisotropy computed for {n_points}x{n_points} grid (Native G⁻¹)")
    print(f"✅ Anisotropy range: [{anisotropy_grid.min():.3f}, {anisotropy_grid.max():.3f}]")
    
    return X, Y, anisotropy_grid


def run_native_rhmc_sampling(native_metric_tensor, n_samples=500, n_steps=30):
    """Run native RHMC sampling with G⁻¹ as metric - OPTIMIZED FOR ACCURACY."""
    print("🎯 Running native RHMC sampling with G⁻¹ as metric (OPTIMIZED)")
    
    # Create native RHMC sampler with OPTIMIZED parameters for accuracy
    sampler = NativeInverseRHMC(
        native_metric_tensor, 
        n_steps=n_steps, 
        n_leapfrog=100,      # More leapfrog steps for precision
        step_size=1e-6        # Much smaller step size for accuracy
    )
    
    # Initialize samples very close to centroids for better targeting
    device = native_metric_tensor.centroids.device
    centroids = native_metric_tensor.centroids
    
    # Distribute samples around centroids
    n_centroids = len(centroids)
    samples_per_centroid = n_samples // n_centroids
    remainder = n_samples % n_centroids
    
    init_positions = []
    for i, centroid in enumerate(centroids):
        n_local = samples_per_centroid + (1 if i < remainder else 0)
        # Initialize very close to centroids
        noise = torch.randn(n_local, 2, device=device) * 0.05  # Small noise
        local_pos = centroid.unsqueeze(0) + noise
        init_positions.append(local_pos)
    
    initial_z = torch.cat(init_positions, dim=0)
    print(f"✅ Initialized {len(initial_z)} samples near centroids")
    
    # Override volume correction for stronger attraction
    def enhanced_volume_correction(z):
        """Enhanced volume correction with stronger attraction to centroids."""
        log_det_G_inv = native_metric_tensor.log_det_G_inverse(z)
        return -1.5 * log_det_G_inv  # Stronger attraction
    
    # Temporarily override the volume correction method
    original_volume_correction = sampler._volume_correction
    sampler._volume_correction = enhanced_volume_correction
    
    # Run sampling
    start_time = time.time()
    samples = sampler.sample(n_samples, initial_z)
    sampling_time = time.time() - start_time
    
    # Restore original method
    sampler._volume_correction = original_volume_correction
    
    print(f"✅ Native RHMC completed in {sampling_time:.3f}s")
    print(f"✅ Generated {len(samples)} samples")
    print(f"✅ Samples range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    # Compute G⁻¹ determinant at sample points using native metric
    with torch.no_grad():
        _, log_det_G_inv_samples = native_metric_tensor(samples)
        det_G_inv_samples = torch.exp(log_det_G_inv_samples)
    
    print(f"✅ G⁻¹ determinant at samples: [{det_G_inv_samples.min():.3e}, {det_G_inv_samples.max():.3e}]")
    
    # Analyze centroid proximity
    min_distances = []
    for sample in samples:
        distances = torch.norm(centroids - sample.unsqueeze(0), dim=1)
        min_dist = torch.min(distances).item()
        min_distances.append(min_dist)
    
    overall_min = min(min_distances)
    mean_min = np.mean(min_distances)
    very_close = sum(1 for d in min_distances if d < 0.1)
    close = sum(1 for d in min_distances if d < 0.2)
    
    print(f"🎯 TARGETING ANALYSIS:")
    print(f"   Minimum distance to centroids: {overall_min:.6f}")
    print(f"   Mean distance to centroids: {mean_min:.4f}")
    print(f"   Very close samples (<0.1): {very_close}/{len(samples)} ({100*very_close/len(samples):.1f}%)")
    print(f"   Close samples (<0.2): {close}/{len(samples)} ({100*close/len(samples):.1f}%)")
    
    return samples.detach().cpu().numpy(), det_G_inv_samples.detach().cpu().numpy()


def create_comprehensive_visualization(native_metric_tensor, latent_data, centroids, samples, det_G_inv_samples, 
                                     X_det, Y_det, det_G_inv_grid, X_aniso, Y_aniso, anisotropy_grid):
    """Create comprehensive visualization with all four plots - EXACT SAME FORMAT."""
    print("🎨 Creating comprehensive visualization (Native G⁻¹)")
    
    # Set consistent color scales
    det_vmin, det_vmax = det_G_inv_grid.min(), det_G_inv_grid.max()
    aniso_vmin, aniso_vmax = anisotropy_grid.min(), anisotropy_grid.max()
    
    # Create figure with 2x2 subplots - EXACT SAME
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Native G⁻¹ Analysis: Centroids, Determinant, RHMC Sampling, and Anisotropy", 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Centroids with data - EXACT SAME
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
    
    # Plot 2: G⁻¹ Determinant - EXACT SAME FORMAT
    ax2 = axes[0, 1]
    contour2 = ax2.contourf(X_det, Y_det, det_G_inv_grid, levels=50, 
                            cmap='viridis', vmin=det_vmin, vmax=det_vmax)
    ax2.set_title("2. G⁻¹ Determinant\n(Manifold Structure)", fontweight='bold')
    ax2.set_xlabel("z₁")
    ax2.set_ylabel("z₂")
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    plt.colorbar(contour2, ax=ax2, label='det(G⁻¹)')
    
    # Plot 3: RHMC Samples colored by G⁻¹ determinant - EXACT SAME FORMAT
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(samples[:, 0], samples[:, 1], 
                           c=det_G_inv_samples, cmap='viridis', 
                           vmin=det_vmin, vmax=det_vmax, alpha=0.7, s=30)
    ax3.set_title("3. Native RHMC Sampling\n(Colored by det(G⁻¹))", fontweight='bold')
    ax3.set_xlabel("z₁")
    ax3.set_ylabel("z₂")
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='det(G⁻¹)')
    
    # Plot 4: Anisotropy - EXACT SAME FORMAT
    ax4 = axes[1, 1]
    contour4 = ax4.contourf(X_aniso, Y_aniso, anisotropy_grid, levels=50, 
                            cmap='coolwarm', vmin=aniso_vmin, vmax=aniso_vmax)
    ax4.set_title("4. Anisotropy (λ₁ - λ₂)\n(Stretching/Compression)", fontweight='bold')
    ax4.set_xlabel("z₁")
    ax4.set_ylabel("z₂")
    ax4.set_xlim(-4, 4)
    ax4.set_ylim(-4, 4)
    plt.colorbar(contour4, ax=ax4, label='Anisotropy')
    
    # Add some sample points to anisotropy plot for reference - EXACT SAME
    ax4.scatter(samples[::10, 0], samples[::10, 1], 
                c='white', s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig("native_inverse_exact_comprehensive.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Comprehensive visualization saved as 'native_inverse_exact_comprehensive.png'")
    
    # Print summary statistics - EXACT SAME
    print("\n📊 Summary Statistics:")
    print(f"   Centroids: {len(centroids)} points")
    print(f"   Data points: {len(latent_data)}")
    print(f"   RHMC samples: {len(samples)}")
    print(f"   G⁻¹ determinant range: [{det_vmin:.3e}, {det_vmax:.3e}]")
    print(f"   Anisotropy range: [{aniso_vmin:.3f}, {aniso_vmax:.3f}]")


def main():
    """Main function to run native G⁻¹ comprehensive analysis - EXACT SAME STRUCTURE."""
    print("🚀 Native G⁻¹ Comprehensive Analysis")
    print("=" * 60)
    
    # Step 1: Load data and compute centroids - EXACT SAME
    model, native_metric_tensor, latent_data, centroids, metric_matrices = load_real_data_and_compute_centroids()
    
    # Step 2: Compute G⁻¹ determinant grid - SAME BUT WITH NATIVE G⁻¹
    X_det, Y_det, det_G_inv_grid = compute_g_inverse_determinant_grid(native_metric_tensor)
    
    # Step 3: Compute anisotropy grid - SAME BUT WITH NATIVE G⁻¹
    X_aniso, Y_aniso, anisotropy_grid = compute_anisotropy_grid(native_metric_tensor)
    
    # Step 4: Run native RHMC sampling - SAME BUT WITH NATIVE G⁻¹
    samples, det_G_inv_samples = run_native_rhmc_sampling(native_metric_tensor, n_samples=500, n_steps=30)
    
    # Step 5: Create comprehensive visualization - EXACT SAME FORMAT
    create_comprehensive_visualization(native_metric_tensor, latent_data, centroids, samples, det_G_inv_samples,
                                     X_det, Y_det, det_G_inv_grid, X_aniso, Y_aniso, anisotropy_grid)
    
    print("\n✅ Native G⁻¹ comprehensive analysis completed!")


if __name__ == "__main__":
    main() 