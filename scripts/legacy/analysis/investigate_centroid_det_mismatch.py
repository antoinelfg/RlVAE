#!/usr/bin/env python3
"""
Investigate Centroid-det(G⁻¹) Mismatch
======================================

This script investigates why the highest det(G⁻¹) might not be near centroids.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.components.native_inverse_metric import NativeInverseMetricTensor

def investigate_centroid_det_mismatch():
    """Investigate why highest det(G⁻¹) is not near centroids."""
    print("🔍 INVESTIGATING CENTROID-det(G⁻¹) MISMATCH")
    print("="*55)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained model's latent data
    checkpoint_path = "outputs/checkpoints/epoch=31-val_loss=5.884.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract latent data (simplified version)
    latent_data = torch.randn(200, 16, device=device)  # Simulate latent data
    print(f"📊 Latent data shape: {latent_data.shape}")
    
    # Create metric
    class DummyModel:
        pass
    model = DummyModel()
    
    native_metric = NativeInverseMetricTensor.from_model_data(
        model, latent_data, 
        n_centroids=25,
        temperature=0.5,
        device=device
    )
    
    centroids = native_metric.centroids
    print(f"🎯 Created metric with {len(centroids)} centroids")
    
    # Create visualization grid
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    
    # Create 16D grid points
    grid_2d = np.column_stack([X.ravel(), Y.ravel()])
    latent_mean = latent_data.mean(dim=0).cpu().numpy()
    
    grid_points_16d = np.zeros((len(grid_2d), 16))
    grid_points_16d[:, :2] = grid_2d
    grid_points_16d[:, 2:] = latent_mean[2:]
    
    grid_points = torch.tensor(grid_points_16d, dtype=torch.float32, device=device)
    
    # Compute det(G⁻¹) for all grid points
    with torch.no_grad():
        G_inv, log_det_G_inv = native_metric(grid_points)
        det_grid = torch.exp(log_det_G_inv).cpu().numpy().reshape(X.shape)
    
    print(f"✅ det(G⁻¹) range: [{det_grid.min():.1f}, {det_grid.max():.1f}]")
    
    # Find the maximum det(G⁻¹) location
    max_det_idx = np.unravel_index(np.argmax(det_grid), det_grid.shape)
    max_det_pos = (X[max_det_idx], Y[max_det_idx])
    max_det_value = det_grid[max_det_idx]
    
    print(f"🎯 Maximum det(G⁻¹): {max_det_value:.1f} at position {max_det_pos}")
    
    # Find distances from max det position to all centroids
    max_det_point_2d = torch.tensor([max_det_pos[0], max_det_pos[1]], device=device)
    distances_to_centroids = torch.norm(max_det_point_2d.unsqueeze(0) - centroids[:, :2], dim=1)
    min_distance = distances_to_centroids.min().item()
    closest_centroid_idx = torch.argmin(distances_to_centroids).item()
    closest_centroid = centroids[closest_centroid_idx][:2].cpu().numpy()
    
    print(f"📍 Distance to closest centroid: {min_distance:.3f}")
    print(f"📍 Closest centroid position: {closest_centroid}")
    print(f"📍 Closest centroid index: {closest_centroid_idx}")
    
    # Check if this is indeed weird
    if min_distance > 1.0:
        print("⚠️  WARNING: Maximum det(G⁻¹) is far from centroids!")
        print("   This suggests a potential issue with the metric computation.")
    else:
        print("✅ Maximum det(G⁻¹) is reasonably close to centroids.")
    
    # Let's investigate the metric computation at the max det point
    print(f"\n🔧 INVESTIGATING METRIC AT MAX DET POINT:")
    print("-" * 45)
    
    # Create the 16D point at max det position
    max_det_point_16d = torch.zeros(16, device=device)
    max_det_point_16d[:2] = torch.tensor([max_det_pos[0], max_det_pos[1]], device=device)
    max_det_point_16d[2:] = latent_data.mean(dim=0)[2:]
    
    # Compute distances to all centroids
    distances = torch.norm(max_det_point_16d.unsqueeze(0) - centroids, dim=1)
    weights = torch.softmax(-distances / 0.5, dim=0)
    
    print(f"📊 Distances to centroids: [{distances.min():.3f}, {distances.max():.3f}]")
    print(f"📊 Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"📊 Sum of weights: {weights.sum():.3f}")
    
    # Show top 5 closest centroids
    sorted_indices = torch.argsort(distances)
    print(f"\n🎯 Top 5 closest centroids:")
    for i in range(5):
        idx = sorted_indices[i].item()
        dist = distances[idx].item()
        weight = weights[idx].item()
        print(f"   {i+1}. Centroid {idx}: distance={dist:.3f}, weight={weight:.3f}")
    
    # Let's also check what happens at the centroid positions themselves
    print(f"\n🔍 CHECKING DET(G⁻¹) AT CENTROID POSITIONS:")
    print("-" * 45)
    
    centroid_dets = []
    for i, centroid in enumerate(centroids):
        # Create 16D point at centroid position
        centroid_16d = centroid.clone()
        centroid_16d[2:] = latent_data.mean(dim=0)[2:]  # Use mean for other dims
        
        # Compute det(G⁻¹) at this centroid
        with torch.no_grad():
            G_inv_centroid, log_det_centroid = native_metric(centroid_16d.unsqueeze(0))
            det_centroid = torch.exp(log_det_centroid).item()
        
        centroid_dets.append(det_centroid)
        print(f"   Centroid {i}: det(G⁻¹) = {det_centroid:.1f}")
    
    max_centroid_det = max(centroid_dets)
    max_centroid_idx = centroid_dets.index(max_centroid_det)
    
    print(f"\n🎯 Maximum det(G⁻¹) at centroids: {max_centroid_det:.1f} (centroid {max_centroid_idx})")
    print(f"🎯 Maximum det(G⁻¹) overall: {max_det_value:.1f}")
    print(f"🎯 Ratio: {max_det_value / max_centroid_det:.2f}")
    
    if max_det_value > max_centroid_det * 1.1:
        print("⚠️  WARNING: Maximum det(G⁻¹) is significantly higher than at any centroid!")
        print("   This suggests the metric computation might have issues.")
    else:
        print("✅ Maximum det(G⁻¹) is reasonably close to centroid values.")
    
    # Create visualization to show the issue
    print(f"\n🎨 CREATING INVESTIGATION VISUALIZATION:")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: det(G⁻¹) with max point highlighted
    ax1 = axes[0, 0]
    contour1 = ax1.contourf(X, Y, det_grid, levels=30, cmap='viridis', alpha=0.8)
    ax1.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='red', s=100, marker='*',
               edgecolors='white', linewidth=2, label='Centroids')
    ax1.scatter(max_det_pos[0], max_det_pos[1], c='yellow', s=200, marker='X',
               edgecolors='black', linewidth=3, label='Max det(G⁻¹)', zorder=10)
    plt.colorbar(contour1, ax=ax1, label='det(G⁻¹)')
    ax1.set_title('1. det(G⁻¹) with Max Point\n(Yellow X = max det, Red * = centroids)', fontweight='bold')
    ax1.set_xlabel('z₁ (first dimension)')
    ax1.set_ylabel('z₂ (second dimension)')
    ax1.legend()
    
    # Plot 2: Distance to nearest centroid
    distance_grid = np.zeros_like(det_grid)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point_2d = torch.tensor([X[i,j], Y[i,j]], device=device)
            distances = torch.norm(point_2d.unsqueeze(0) - centroids[:, :2], dim=1)
            distance_grid[i,j] = distances.min().item()
    
    ax2 = axes[0, 1]
    contour2 = ax2.contourf(X, Y, distance_grid, levels=30, cmap='plasma_r', alpha=0.8)
    ax2.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='white', s=100, marker='*',
               edgecolors='black', linewidth=2)
    ax2.scatter(max_det_pos[0], max_det_pos[1], c='yellow', s=200, marker='X',
               edgecolors='black', linewidth=3, zorder=10)
    plt.colorbar(contour2, ax=ax2, label='Distance to Nearest Centroid')
    ax2.set_title('2. Distance to Nearest Centroid\n(Should be low where det is high)', fontweight='bold')
    ax2.set_xlabel('z₁ (first dimension)')
    ax2.set_ylabel('z₂ (second dimension)')
    
    # Plot 3: Correlation scatter plot
    ax3 = axes[1, 0]
    det_flat = det_grid.flatten()
    dist_flat = distance_grid.flatten()
    ax3.scatter(dist_flat, det_flat, alpha=0.3, s=1)
    ax3.scatter(min_distance, max_det_value, c='yellow', s=200, marker='X',
               edgecolors='black', linewidth=3, zorder=10, label='Max det point')
    ax3.set_xlabel('Distance to Nearest Centroid')
    ax3.set_ylabel('det(G⁻¹)')
    ax3.set_title('3. Distance vs det(G⁻¹)\n(Max point should be in bottom-right)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Centroid det values
    ax4 = axes[1, 1]
    centroid_positions = centroids.cpu().numpy()
    scatter = ax4.scatter(centroid_positions[:, 0], centroid_positions[:, 1], 
                         c=centroid_dets, s=150, cmap='viridis', marker='*',
                         edgecolors='white', linewidth=2)
    ax4.scatter(max_det_pos[0], max_det_pos[1], c='yellow', s=200, marker='X',
               edgecolors='black', linewidth=3, zorder=10, label='Max det point')
    plt.colorbar(scatter, ax=ax4, label='det(G⁻¹) at Centroids')
    ax4.set_title('4. det(G⁻¹) at Centroids\n(Should be highest at centroids)', fontweight='bold')
    ax4.set_xlabel('z₁ (first dimension)')
    ax4.set_ylabel('z₂ (second dimension)')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('investigate_centroid_det_mismatch.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Investigation visualization saved to: investigate_centroid_det_mismatch.png")
    
    # Summary
    print(f"\n📝 SUMMARY:")
    print("=" * 20)
    print(f"Maximum det(G⁻¹): {max_det_value:.1f} at {max_det_pos}")
    print(f"Distance to closest centroid: {min_distance:.3f}")
    print(f"Maximum det at centroids: {max_centroid_det:.1f}")
    print(f"Ratio: {max_det_value / max_centroid_det:.2f}")
    
    if min_distance > 1.0 or max_det_value > max_centroid_det * 1.1:
        print("\n⚠️  POTENTIAL ISSUES DETECTED:")
        print("1. Maximum det(G⁻¹) is far from centroids")
        print("2. Maximum det(G⁻¹) is higher than at any centroid")
        print("3. This suggests the metric computation might have problems")
        print("\n🔧 POSSIBLE CAUSES:")
        print("1. Temperature parameter too low (0.5 might be too aggressive)")
        print("2. Regularization parameter affecting the metric")
        print("3. Issues with the 16D to 2D projection")
        print("4. Problems with the centroid selection or metric matrices")
    else:
        print("\n✅ METRIC BEHAVIOR LOOKS REASONABLE")
        print("The maximum det(G⁻¹) is appropriately close to centroids")

if __name__ == "__main__":
    investigate_centroid_det_mismatch() 