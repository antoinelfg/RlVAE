#!/usr/bin/env python3
"""
Investigate Visualization Computation
====================================

Deep dive into why the determinant visualization appears backwards.
We'll examine every step of the computation pipeline.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.components.native_inverse_metric import NativeInverseMetricTensor

def investigate_full_pipeline():
    """Investigate the complete visualization pipeline used in test_real_rhmc_manifold.py"""
    print("🔍 INVESTIGATING FULL VISUALIZATION PIPELINE")
    print("="*60)
    
    # Load the same data as the main script
    print("1. Loading real Sprites data (same as main script)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create realistic test data that mimics the real Sprites latent space
    # Based on the range we saw: [-2.045, 2.126]
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create clusters that mimic real data distribution
    n_clusters = 5
    points_per_cluster = 300
    latent_data_parts = []
    
    cluster_centers = [
        [-1.5, 0.5], [0.0, 1.0], [1.0, -0.5], [-0.5, -1.0], [1.5, 1.5]
    ]
    
    for center in cluster_centers:
        cluster = torch.randn(points_per_cluster, 2, device=device) * 0.3 + torch.tensor(center, device=device)
        latent_data_parts.append(cluster)
    
    latent_data = torch.cat(latent_data_parts, dim=0)
    
    # Add some noise points
    noise_points = torch.randn(500, 2, device=device) * 1.8
    latent_data = torch.cat([latent_data, noise_points], dim=0)
    
    print(f"   Simulated latent data shape: {latent_data.shape}")
    print(f"   Data range: [{latent_data.min():.3f}, {latent_data.max():.3f}]")
    
    # Create dummy model
    class DummyModel:
        pass
    model = DummyModel()
    
    # Step 2: Create metric EXACTLY like the main script
    print("\n2. Creating metric (same parameters as main script)...")
    native_metric = NativeInverseMetricTensor.from_model_data(
        model, latent_data, n_centroids=25, temperature=2.0, device=device
    )
    
    centroids = native_metric.centroids.cpu()
    print(f"   Created {len(centroids)} centroids")
    print(f"   Temperature: 2.0 (same as main script)")
    
    # Step 3: Create grid EXACTLY like the main script
    print("\n3. Creating grid (same as main script)...")
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    grid_points = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), 
                              dtype=torch.float32, device=native_metric.device)
    
    print(f"   Grid shape: {X.shape}")
    print(f"   Grid points: {grid_points.shape}")
    
    # Step 4: Compute determinants EXACTLY like the main script
    print("\n4. Computing determinants (same as main script)...")
    with torch.no_grad():
        G_inv, log_det_G_inv = native_metric(grid_points)
        det_G_inv_grid = torch.exp(log_det_G_inv).cpu().numpy().reshape(X.shape)
    
    vmin, vmax = det_G_inv_grid.min(), det_G_inv_grid.max()
    print(f"   det(G⁻¹) grid range: [{vmin:.3e}, {vmax:.3e}]")
    print(f"   log det(G⁻¹) range: [{log_det_G_inv.min().item():.3f}, {log_det_G_inv.max().item():.3f}]")
    
    # Step 5: Analyze specific regions
    print("\n5. Analyzing specific regions...")
    
    # Test points near centroids vs far from centroids
    test_points = []
    labels = []
    
    # Points near centroids
    for i in range(min(5, len(centroids))):
        near_point = centroids[i] + torch.randn(2) * 0.1
        test_points.append(near_point)
        labels.append(f"Near centroid {i}")
    
    # Points far from all centroids
    far_points = [
        torch.tensor([-3.5, -3.5]),
        torch.tensor([3.5, 3.5]),
        torch.tensor([-3.5, 3.5]),
        torch.tensor([3.5, -3.5])
    ]
    
    for i, point in enumerate(far_points):
        test_points.append(point)
        labels.append(f"Far point {i}")
    
    print("   Point analysis:")
    for point, label in zip(test_points, labels):
        with torch.no_grad():
            G_inv_test, log_det_test = native_metric(point.unsqueeze(0).to(device))
            det_test = torch.exp(log_det_test).item()
        
        # Distance to nearest centroid
        distances = torch.norm(point.unsqueeze(0) - centroids, dim=1)
        nearest_dist = distances.min().item()
        
        print(f"     {label}: det={det_test:.3e}, dist_to_centroid={nearest_dist:.3f}")
    
    # Step 6: Create detailed visualization analysis
    print("\n6. Creating detailed visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Raw data and centroids
    ax1 = axes[0, 0]
    latent_np = latent_data.cpu().numpy()
    ax1.scatter(latent_np[:, 0], latent_np[:, 1], alpha=0.3, s=5, c='lightblue', label='Latent Data')
    ax1.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='*', 
               edgecolors='black', linewidth=1, label='Centroids')
    ax1.set_title('1. Raw Data & Centroids', fontweight='bold')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: det(G⁻¹) with viridis (same as main script)
    ax2 = axes[0, 1]
    contour2 = ax2.contourf(X, Y, det_G_inv_grid, levels=50, cmap='viridis', alpha=0.8)
    ax2.contour(X, Y, det_G_inv_grid, levels=15, colors='white', alpha=0.4, linewidths=0.5)
    ax2.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='*', 
               edgecolors='white', linewidth=2, label='Centroids')
    plt.colorbar(contour2, ax=ax2, label='det(G⁻¹)')
    ax2.set_title('2. det(G⁻¹) - Viridis\n(Yellow=High, Purple=Low)', fontweight='bold')
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.legend()
    
    # Plot 3: det(G⁻¹) with INVERTED colormap
    ax3 = axes[0, 2]
    contour3 = ax3.contourf(X, Y, det_G_inv_grid, levels=50, cmap='viridis_r', alpha=0.8)
    ax3.contour(X, Y, det_G_inv_grid, levels=15, colors='white', alpha=0.4, linewidths=0.5)
    ax3.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='*', 
               edgecolors='white', linewidth=2, label='Centroids')
    plt.colorbar(contour3, ax=ax3, label='det(G⁻¹)')
    ax3.set_title('3. det(G⁻¹) - Viridis Reversed\n(Purple=High, Yellow=Low)', fontweight='bold')
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    ax3.legend()
    
    # Plot 4: log det(G⁻¹)
    ax4 = axes[1, 0]
    log_det_grid = log_det_G_inv.cpu().numpy().reshape(X.shape)
    contour4 = ax4.contourf(X, Y, log_det_grid, levels=50, cmap='viridis', alpha=0.8)
    ax4.contour(X, Y, log_det_grid, levels=15, colors='white', alpha=0.4, linewidths=0.5)
    ax4.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='*', 
               edgecolors='white', linewidth=2, label='Centroids')
    plt.colorbar(contour4, ax=ax4, label='log det(G⁻¹)')
    ax4.set_title('4. log det(G⁻¹)', fontweight='bold')
    ax4.set_xlim(-4, 4)
    ax4.set_ylim(-4, 4)
    ax4.legend()
    
    # Plot 5: Distance to nearest centroid
    ax5 = axes[1, 1]
    # Compute distance to nearest centroid for each grid point
    distances_grid = np.zeros_like(det_G_inv_grid)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = torch.tensor([X[i,j], Y[i,j]])
            dists = torch.norm(point.unsqueeze(0) - centroids, dim=1)
            distances_grid[i,j] = dists.min().item()
    
    contour5 = ax5.contourf(X, Y, distances_grid, levels=50, cmap='plasma', alpha=0.8)
    ax5.scatter(centroids[:, 0], centroids[:, 1], c='white', s=100, marker='*', 
               edgecolors='black', linewidth=2, label='Centroids')
    plt.colorbar(contour5, ax=ax5, label='Distance to Centroid')
    ax5.set_title('5. Distance to Nearest Centroid', fontweight='bold')
    ax5.set_xlim(-4, 4)
    ax5.set_ylim(-4, 4)
    ax5.legend()
    
    # Plot 6: Correlation analysis
    ax6 = axes[1, 2]
    # Flatten grids for correlation
    det_flat = det_G_inv_grid.flatten()
    dist_flat = distances_grid.flatten()
    
    ax6.scatter(dist_flat, det_flat, alpha=0.3, s=1)
    ax6.set_xlabel('Distance to Nearest Centroid')
    ax6.set_ylabel('det(G⁻¹)')
    ax6.set_title('6. Correlation:\nDistance vs det(G⁻¹)', fontweight='bold')
    
    # Compute correlation
    correlation = np.corrcoef(dist_flat, det_flat)[0, 1]
    ax6.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax6.transAxes, fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('investigate_visualization_computation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Step 7: Summary analysis
    print(f"\n7. SUMMARY ANALYSIS:")
    print(f"   - det(G⁻¹) range: [{vmin:.3e}, {vmax:.3e}]")
    print(f"   - Distance vs det(G⁻¹) correlation: {correlation:.3f}")
    
    if correlation > 0:
        print(f"   ⚠️  POSITIVE correlation: det(G⁻¹) INCREASES with distance from centroids")
        print(f"       This is COUNTERINTUITIVE - should be negative!")
    else:
        print(f"   ✅ NEGATIVE correlation: det(G⁻¹) DECREASES with distance from centroids")
        print(f"       This matches intuition!")
    
    # Step 8: Check if there's a temperature effect
    print(f"\n8. TEMPERATURE EFFECT ANALYSIS:")
    print(f"   Testing different temperatures...")
    
    temperatures = [0.5, 1.0, 2.0, 5.0]
    for temp in temperatures:
        temp_metric = NativeInverseMetricTensor.from_model_data(
            model, latent_data, n_centroids=25, temperature=temp, device=device
        )
        
        # Test a point near centroid vs far
        near_point = centroids[0].to(device) + torch.tensor([0.1, 0.1]).to(device)
        far_point = torch.tensor([3.0, 3.0]).to(device)
        
        with torch.no_grad():
            _, log_det_near = temp_metric(near_point.unsqueeze(0))
            _, log_det_far = temp_metric(far_point.unsqueeze(0))
            
            det_near = torch.exp(log_det_near).item()
            det_far = torch.exp(log_det_far).item()
        
        print(f"     T={temp}: det_near={det_near:.3e}, det_far={det_far:.3e}, ratio={det_near/det_far:.3f}")
    
    print(f"\n✅ Investigation complete! Check investigate_visualization_computation.png")
    return det_G_inv_grid, distances_grid, correlation

if __name__ == "__main__":
    det_grid, dist_grid, corr = investigate_full_pipeline()