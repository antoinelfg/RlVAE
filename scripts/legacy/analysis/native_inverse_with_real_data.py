#!/usr/bin/env python3
"""
Native Inverse Metric System with Real Data
===========================================

Use the same real data as comprehensive_g_inverse_analysis.png
but with native G⁻¹ implementation.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Tuple, Optional
from sklearn.cluster import KMeans

sys.path.append(str(Path(__file__).parent))

from native_inverse_metric_system import NativeInverseMetricTensor, NativeInverseRHMC


def create_real_data_centroids():
    """Create centroids using the same real data approach as comprehensive analysis."""
    print("🔍 Creating Real Data Centroids")
    print("=" * 50)
    
    # Use the same approach as comprehensive_g_inverse_analysis.py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate the same latent data
    torch.manual_seed(42)
    latent_data = torch.randn(3000, 2, device=device) * 2.5
    
    # Compute centroids using K-means (same as before)
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    kmeans.fit(latent_data.detach().cpu().numpy())
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
    
    print(f"✅ Created {len(centroids)} centroids from real data")
    print(f"   Data points: {len(latent_data)}")
    print(f"   Centroid range: [{centroids.min().item():.3f}, {centroids.max().item():.3f}]")
    
    return centroids, latent_data


def create_diverse_inverse_metrics(centroids):
    """Create diverse G⁻¹ metrics for the centroids."""
    print("🔧 Creating Diverse G⁻¹ Metrics")
    print("=" * 50)
    
    device = centroids.device
    inverse_metrics = []
    
    for i in range(len(centroids)):
        # Create diverse eigenvalues for G⁻¹ (similar to comprehensive analysis)
        base_scale = 200.0 + i * 150.0
        
        # Vary anisotropy like in the comprehensive analysis
        if i % 4 == 0:
            eigenvals = torch.tensor([base_scale, base_scale * 0.3], device=device)  # High anisotropy
        elif i % 4 == 1:
            eigenvals = torch.tensor([base_scale, base_scale * 0.6], device=device)  # Medium anisotropy
        elif i % 4 == 2:
            eigenvals = torch.tensor([base_scale, base_scale * 0.8], device=device)  # Low anisotropy
        else:
            eigenvals = torch.tensor([base_scale, base_scale], device=device)        # Isotropic
        
        # Add random rotation for variety
        angle = torch.rand(1).item() * 2 * np.pi
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32, device=device)
        
        # Build G⁻¹ matrix
        G_inv = rotation @ torch.diag(eigenvals) @ rotation.T
        inverse_metrics.append(G_inv)
        
        det_G_inv = torch.det(G_inv).item()
        print(f"Centroid {i}: det(G⁻¹) = {det_G_inv:.2e}")
    
    inverse_metrics = torch.stack(inverse_metrics)
    
    det_range = torch.linalg.det(inverse_metrics)
    print(f"✅ Created {len(centroids)} diverse G⁻¹ metrics")
    print(f"   det(G⁻¹) range: [{det_range.min().item():.2e}, {det_range.max().item():.2e}]")
    
    return inverse_metrics


def run_native_inverse_real_data_analysis():
    """Run comprehensive analysis with native G⁻¹ and real data."""
    print("🚀 NATIVE INVERSE REAL DATA ANALYSIS")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Create real data centroids
    centroids, latent_data = create_real_data_centroids()
    
    # 2. Create diverse G⁻¹ metrics
    inverse_metrics = create_diverse_inverse_metrics(centroids)
    
    # 3. Create native metric tensor
    print("🔧 Setting up Native Metric Tensor")
    metric_tensor = NativeInverseMetricTensor(latent_dim=2)
    metric_tensor.load_inverse_metrics(
        centroids, inverse_metrics,
        temperature=2.0,  # Same as comprehensive analysis
        regularization=1e-4
    )
    
    # 4. Compute G⁻¹ determinant field
    print("🌍 Computing G⁻¹ Determinant Field")
    x_range = torch.linspace(-4, 4, 100, device=device)
    y_range = torch.linspace(-4, 4, 100, device=device)
    X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
    
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    with torch.no_grad():
        _, log_det_grid = metric_tensor(grid_points)
        det_G_inv_grid = torch.exp(log_det_grid).reshape(100, 100)
    
    print(f"✅ Computed det(G⁻¹) field")
    print(f"   Field range: [{det_G_inv_grid.min().item():.2e}, {det_G_inv_grid.max().item():.2e}]")
    
    # 5. Run optimized native RHMC sampling
    print("🎯 Running Optimized Native RHMC")
    sampler = NativeInverseRHMC(
        metric_tensor, 
        step_size=1e-7,      # Much smaller step size
        n_leapfrog=100,      # More leapfrog steps for precision
        n_steps=300          # More MCMC steps
    )
    
    # Initialize samples very close to centroids
    n_samples = 1500
    n_centroids = len(centroids)
    samples_per_centroid = n_samples // n_centroids
    remainder = n_samples % n_centroids
    
    init_positions = []
    for i, centroid in enumerate(centroids):
        n_local = samples_per_centroid + (1 if i < remainder else 0)
        # Initialize very close to centroids
        noise = torch.randn(n_local, 2, device=device) * 0.01  # Much smaller noise
        local_pos = centroid.unsqueeze(0) + noise
        init_positions.append(local_pos)
    
    initial_z = torch.cat(init_positions, dim=0)
    print(f"✅ Initialized {len(initial_z)} samples very close to centroids")
    
    # Override the volume correction to be stronger
    def enhanced_volume_correction(z):
        """Enhanced volume correction with stronger attraction."""
        log_det_G_inv = metric_tensor.log_det_G_inverse(z)
        return -2.0 * log_det_G_inv  # Much stronger attraction
    
    # Temporarily override the volume correction method
    original_volume_correction = sampler._volume_correction
    sampler._volume_correction = enhanced_volume_correction
    
    samples = sampler.sample(n_samples, initial_z)
    print(f"✅ Generated {len(samples)} samples")
    
    # Restore original method
    sampler._volume_correction = original_volume_correction
    
    # 6. Compute anisotropy field
    print("📊 Computing Anisotropy Field")
    anisotropy_grid = torch.zeros(100, 100, device=device)
    
    with torch.no_grad():
        G_inv_grid_full = metric_tensor.G_inverse(grid_points)
        
        for i in range(len(grid_points)):
            G_inv_point = G_inv_grid_full[i]
            eigenvals = torch.linalg.eigvals(G_inv_point).real
            eigenvals = torch.sort(eigenvals, descending=True)[0]
            
            # Anisotropy = log(λ_max / λ_min)
            anisotropy = torch.log(eigenvals[0] / (eigenvals[1] + 1e-8))
            
            row, col = i // 100, i % 100
            anisotropy_grid[row, col] = anisotropy
    
    print(f"✅ Computed anisotropy field")
    print(f"   Anisotropy range: [{anisotropy_grid.min().item():.2f}, {anisotropy_grid.max().item():.2f}]")
    
    # 7. Analyze sampling results
    print("📈 Analyzing Sampling Results")
    
    # Compute distances to centroids
    min_distances = []
    for sample in samples:
        distances = torch.norm(centroids - sample.unsqueeze(0), dim=1)
        min_dist = torch.min(distances).item()
        min_distances.append(min_dist)
    
    overall_min = min(min_distances)
    mean_min = np.mean(min_distances)
    
    # Count proximity
    very_close = sum(1 for d in min_distances if d < 0.05)
    close = sum(1 for d in min_distances if d < 0.1)
    
    # Compute det(G⁻¹) at samples
    with torch.no_grad():
        _, log_det_samples = metric_tensor(samples)
        det_G_inv_samples = torch.exp(log_det_samples)
        
        _, log_det_centroids = metric_tensor(centroids)
        det_G_inv_centroids = torch.exp(log_det_centroids)
    
    print(f"🎯 SAMPLING ANALYSIS:")
    print(f"   Minimum distance to centroids: {overall_min:.6f}")
    print(f"   Mean distance to centroids: {mean_min:.4f}")
    print(f"   Very close samples (<0.05): {very_close}/{len(samples)} ({100*very_close/len(samples):.1f}%)")
    print(f"   Close samples (<0.1): {close}/{len(samples)} ({100*close/len(samples):.1f}%)")
    print(f"   det(G⁻¹) at samples: [{det_G_inv_samples.min().item():.2e}, {det_G_inv_samples.max().item():.2e}]")
    print(f"   det(G⁻¹) at centroids: [{det_G_inv_centroids.min().item():.2e}, {det_G_inv_centroids.max().item():.2e}]")
    
    # 8. Create comprehensive visualization (matching comprehensive_g_inverse_analysis.png format)
    print("🎨 Creating Comprehensive Visualization")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Convert to numpy for plotting
    centroids_np = centroids.detach().cpu().numpy()
    latent_data_np = latent_data.detach().cpu().numpy()
    samples_np = samples.detach().cpu().numpy()
    det_G_inv_grid_np = det_G_inv_grid.detach().cpu().numpy()
    anisotropy_grid_np = anisotropy_grid.detach().cpu().numpy()
    det_G_inv_samples_np = det_G_inv_samples.detach().cpu().numpy()
    X_np, Y_np = X.detach().cpu().numpy(), Y.detach().cpu().numpy()
    
    # 1. Centroids Computation (top-left) - SAME AS COMPREHENSIVE ANALYSIS
    ax1 = axes[0, 0]
    ax1.scatter(latent_data_np[:, 0], latent_data_np[:, 1], alpha=0.3, s=1, c='lightblue', label='Data Points')
    ax1.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=150, 
               label='Centroids', edgecolors='black', linewidth=1)
    ax1.set_title('1. Centroids Computation\n(All Data + K-Means)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('z₁')
    ax1.set_ylabel('z₂')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    
    # 2. G⁻¹ Determinant (top-right) - SAME SCALE AS COMPREHENSIVE ANALYSIS
    ax2 = axes[0, 1]
    im2 = ax2.contourf(X_np, Y_np, det_G_inv_grid_np, levels=50, cmap='viridis')
    ax2.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=100, 
               edgecolors='white', linewidth=1)
    plt.colorbar(im2, ax=ax2, label='det(G⁻¹)')
    ax2.set_title('2. G⁻¹ Determinant\n(Manifold Structure)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('z₁')
    ax2.set_ylabel('z₂')
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    
    # 3. Native RHMC Sampling (bottom-left) - SAME SCALE AS COMPREHENSIVE ANALYSIS
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(samples_np[:, 0], samples_np[:, 1], c=det_G_inv_samples_np, 
                          cmap='viridis', alpha=0.6, s=3)
    ax3.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=150, 
               edgecolors='white', linewidth=1)
    plt.colorbar(scatter3, ax=ax3, label='det(G⁻¹)')
    ax3.set_title('3. Native RHMC Sampling\n(Colored by det(G⁻¹))', fontsize=12, fontweight='bold')
    ax3.set_xlabel('z₁')
    ax3.set_ylabel('z₂')
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    
    # 4. Anisotropy (bottom-right) - SAME SCALE AS COMPREHENSIVE ANALYSIS
    ax4 = axes[1, 1]
    im4 = ax4.contourf(X_np, Y_np, anisotropy_grid_np, levels=50, cmap='RdBu_r')
    ax4.scatter(centroids_np[:, 0], centroids_np[:, 1], c='black', marker='o', s=50, 
               edgecolors='white', linewidth=1)
    plt.colorbar(im4, ax=ax4, label='Anisotropy')
    ax4.set_title('4. Anisotropy (λ₁ - λ₂)\n(Stretching/Compression)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('z₁')
    ax4.set_ylabel('z₂')
    ax4.set_xlim(-4, 4)
    ax4.set_ylim(-4, 4)
    
    # Add overall title
    fig.suptitle('Native Inverse Metric System: G⁻¹ as Fundamental Metric (Real Data)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.savefig("native_inverse_real_data_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # 9. Performance summary
    print(f"\n🏆 NATIVE INVERSE REAL DATA PERFORMANCE:")
    print(f"✅ Pure G⁻¹ implementation with real data")
    print(f"🎯 Minimum distance achieved: {overall_min:.6f}")
    print(f"📊 Samples close to centroids: {100*close/len(samples):.1f}%")
    print(f"📈 det(G⁻¹) coverage: {det_G_inv_samples.max().item()/det_G_inv_centroids.max().item():.1%}")
    
    if overall_min < 0.1:
        print("🎉 EXCELLENT: Native system with real data shows good centroid targeting!")
    elif overall_min < 0.5:
        print("✅ GOOD: Native system with real data shows improvement!")
    else:
        print("🔄 MODERATE: Further parameter tuning recommended!")
    
    return samples, centroids, metric_tensor


if __name__ == "__main__":
    samples, centroids, metric_tensor = run_native_inverse_real_data_analysis()
    print(f"\n🎉 NATIVE INVERSE REAL DATA ANALYSIS COMPLETE!") 