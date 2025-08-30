#!/usr/bin/env python3
"""
Investigate RHMC Components
===========================

Find what's REALLY making RHMC follow the data so well.
It's probably not just det(G⁻¹) - let's find the real driver!
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.components.native_inverse_metric import NativeInverseMetricTensor

def analyze_rhmc_components():
    """Analyze all components that influence RHMC behavior."""
    print("🔍 INVESTIGATING WHAT MAKES RHMC FOLLOW DATA")
    print("="*60)
    
    # Load the same setup as the main script
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simulate the latent data (same as before)
    torch.manual_seed(42)
    latent_data = torch.randn(6400, 2, device=device) * 1.5
    latent_data = torch.clamp(latent_data, -2.1, 2.2)
    
    class DummyModel:
        pass
    model = DummyModel()
    
    # Create metric with the fixed temperature
    native_metric = NativeInverseMetricTensor.from_model_data(
        model, latent_data, 
        n_centroids=25,
        temperature=0.5,
        device=device
    )
    
    centroids = native_metric.centroids
    print(f"   Created metric with {len(centroids)} centroids")
    
    # Create grid for analysis
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    grid_points = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), 
                              dtype=torch.float32, device=device)
    
    print("\n1. Computing all RHMC components on grid...")
    
    # Component 1: det(G⁻¹) - what we've been looking at
    with torch.no_grad():
        G_inv, log_det_G_inv = native_metric(grid_points)
        det_G_inv_grid = torch.exp(log_det_G_inv).cpu().numpy().reshape(X.shape)
    
    print(f"   det(G⁻¹) range: [{det_G_inv_grid.min():.0f}, {det_G_inv_grid.max():.0f}]")
    
    # Component 2: Distance to nearest centroid (what RHMC potential uses!)
    print("\n2. Computing distance to nearest centroid...")
    distance_grid = np.zeros_like(det_G_inv_grid)
    nearest_centroid_grid = np.zeros_like(det_G_inv_grid, dtype=int)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = torch.tensor([X[i,j], Y[i,j]], device=device)
            distances = torch.norm(point.unsqueeze(0) - centroids, dim=1)
            distance_grid[i,j] = distances.min().item()
            nearest_centroid_grid[i,j] = distances.argmin().item()
    
    print(f"   Distance range: [{distance_grid.min():.3f}, {distance_grid.max():.3f}]")
    
    # Component 3: RHMC potential energy (the key!)
    print("\n3. Computing RHMC potential energy...")
    potential_grid = np.zeros_like(det_G_inv_grid)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = torch.tensor([X[i,j], Y[i,j]], device=device)
            
            # RHMC potential computation (from the RHMC code)
            distances = torch.norm(point.unsqueeze(0) - centroids, dim=1)
            closest_centroid_idx = torch.argmin(distances)
            closest_centroid = centroids[closest_centroid_idx]
            
            # This is the potential gradient that attracts to centroids!
            potential_grad = -2.0 * (point - closest_centroid)  # Same as RHMC
            potential_energy = 0.5 * torch.sum((point - closest_centroid) ** 2).item()
            
            potential_grid[i,j] = potential_energy
    
    print(f"   Potential energy range: [{potential_grid.min():.3f}, {potential_grid.max():.3f}]")
    
    # Component 4: Multi-centroid attraction (from improved RHMC)
    print("\n4. Computing multi-centroid attraction field...")
    attraction_grid = np.zeros_like(det_G_inv_grid)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = torch.tensor([X[i,j], Y[i,j]], device=device)
            
            # Multi-centroid gradient (from improved RHMC)
            distances = torch.norm(point.unsqueeze(0) - centroids, dim=1)
            k_nearest = 3
            _, nearest_indices = torch.topk(distances, k_nearest, largest=False)
            
            multi_centroid_force = 0.0
            for idx in nearest_indices:
                weight = 1.0 / (distances[idx] + 1e-6)
                force = weight * torch.norm(centroids[idx] - point).item()
                multi_centroid_force += force
            
            attraction_grid[i,j] = multi_centroid_force / k_nearest
    
    print(f"   Attraction field range: [{attraction_grid.min():.3f}, {attraction_grid.max():.3f}]")
    
    # Component 5: Actual data density (ground truth)
    print("\n5. Computing actual data density...")
    data_density_grid = np.zeros_like(det_G_inv_grid)
    
    latent_cpu = latent_data.cpu().numpy()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i,j], Y[i,j]])
            
            # Count data points within radius
            distances_to_data = np.linalg.norm(latent_cpu - point, axis=1)
            density = np.sum(distances_to_data < 0.3)  # Within radius 0.3
            data_density_grid[i,j] = density
    
    print(f"   Data density range: [{data_density_grid.min():.0f}, {data_density_grid.max():.0f}]")
    
    # Analyze correlations with data density
    print("\n6. Correlation analysis with actual data density:")
    flat_density = data_density_grid.flatten()
    
    correlations = {
        "det(G⁻¹)": np.corrcoef(flat_density, det_G_inv_grid.flatten())[0,1],
        "Distance to centroid": np.corrcoef(flat_density, -distance_grid.flatten())[0,1],  # Negative distance
        "Potential energy": np.corrcoef(flat_density, -potential_grid.flatten())[0,1],  # Negative potential
        "Attraction field": np.corrcoef(flat_density, attraction_grid.flatten())[0,1],
    }
    
    print("   Correlations with data density:")
    for name, corr in correlations.items():
        status = "🔥 STRONG" if abs(corr) > 0.7 else "✅ GOOD" if abs(corr) > 0.4 else "❌ WEAK"
        print(f"     {name}: {corr:.3f} {status}")
    
    # Find the best predictor
    best_component = max(correlations.items(), key=lambda x: abs(x[1]))
    print(f"\n   🎯 BEST DATA PREDICTOR: {best_component[0]} (corr: {best_component[1]:.3f})")
    
    # Create comprehensive visualization
    print("\n7. Creating comprehensive component visualization...")
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    
    # Plot 1: det(G⁻¹) (what we thought was important)
    ax1 = axes[0, 0]
    contour1 = ax1.contourf(X, Y, det_G_inv_grid, levels=30, cmap='viridis', alpha=0.8)
    ax1.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='red', s=100, marker='*',
               edgecolors='white', linewidth=2)
    ax1.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='white', s=1, alpha=0.3)
    plt.colorbar(contour1, ax=ax1, label='det(G⁻¹)')
    ax1.set_title(f'1. det(G⁻¹)\nCorr with data: {correlations["det(G⁻¹)"]:.3f}', fontweight='bold')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    
    # Plot 2: Distance to nearest centroid
    ax2 = axes[0, 1]
    contour2 = ax2.contourf(X, Y, distance_grid, levels=30, cmap='plasma_r', alpha=0.8)  # Reversed so low distance = high value
    ax2.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='white', s=100, marker='*',
               edgecolors='black', linewidth=2)
    ax2.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='white', s=1, alpha=0.3)
    plt.colorbar(contour2, ax=ax2, label='Distance to Centroid')
    ax2.set_title(f'2. Distance to Centroid\nCorr with data: {correlations["Distance to centroid"]:.3f}', fontweight='bold')
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    
    # Plot 3: RHMC potential energy (this might be the key!)
    ax3 = axes[1, 0]
    contour3 = ax3.contourf(X, Y, potential_grid, levels=30, cmap='hot_r', alpha=0.8)  # Reversed so low energy = high value
    ax3.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='cyan', s=100, marker='*',
               edgecolors='black', linewidth=2)
    ax3.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='white', s=1, alpha=0.3)
    plt.colorbar(contour3, ax=ax3, label='Potential Energy')
    ax3.set_title(f'3. RHMC Potential Energy\nCorr with data: {correlations["Potential energy"]:.3f}', fontweight='bold')
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    
    # Plot 4: Multi-centroid attraction
    ax4 = axes[1, 1]
    contour4 = ax4.contourf(X, Y, attraction_grid, levels=30, cmap='magma', alpha=0.8)
    ax4.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='yellow', s=100, marker='*',
               edgecolors='black', linewidth=2)
    ax4.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='white', s=1, alpha=0.3)
    plt.colorbar(contour4, ax=ax4, label='Attraction Field')
    ax4.set_title(f'4. Multi-Centroid Attraction\nCorr with data: {correlations["Attraction field"]:.3f}', fontweight='bold')
    ax4.set_xlim(-4, 4)
    ax4.set_ylim(-4, 4)
    
    # Plot 5: Actual data density (ground truth)
    ax5 = axes[2, 0]
    contour5 = ax5.contourf(X, Y, data_density_grid, levels=30, cmap='Blues', alpha=0.8)
    ax5.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='red', s=100, marker='*',
               edgecolors='white', linewidth=2)
    ax5.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='red', s=1, alpha=0.5)
    plt.colorbar(contour5, ax=ax5, label='Data Point Count')
    ax5.set_title('5. ACTUAL DATA DENSITY\n(Ground Truth)', fontweight='bold')
    ax5.set_xlim(-4, 4)
    ax5.set_ylim(-4, 4)
    
    # Plot 6: Best component overlay
    ax6 = axes[2, 1]
    
    # Get the best component data
    if "Potential energy" in best_component[0]:
        best_data = potential_grid
        cmap = 'hot_r'
        label = 'Potential Energy'
    elif "Distance" in best_component[0]:
        best_data = distance_grid
        cmap = 'plasma_r'
        label = 'Distance'
    elif "Attraction" in best_component[0]:
        best_data = attraction_grid
        cmap = 'magma'
        label = 'Attraction'
    else:
        best_data = det_G_inv_grid
        cmap = 'viridis'
        label = 'det(G⁻¹)'
    
    contour6 = ax6.contourf(X, Y, best_data, levels=30, cmap=cmap, alpha=0.6)
    
    # Overlay actual data points
    scatter_data = ax6.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='white', s=10, alpha=0.8,
                              edgecolors='black', linewidth=0.5, label='Real Data')
    ax6.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='red', s=150, marker='*',
               edgecolors='white', linewidth=2, label='Centroids')
    
    plt.colorbar(contour6, ax=ax6, label=label)
    ax6.set_title(f'6. BEST PREDICTOR: {best_component[0]}\nCorr: {best_component[1]:.3f}', fontweight='bold')
    ax6.set_xlim(-4, 4)
    ax6.set_ylim(-4, 4)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('investigate_rhmc_components.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Component analysis complete! Check investigate_rhmc_components.png")
    print(f"\n🎯 CONCLUSION:")
    print(f"   The component that REALLY makes RHMC follow data is:")
    print(f"   >>> {best_component[0]} (correlation: {best_component[1]:.3f}) <<<")
    
    return correlations, best_component

if __name__ == "__main__":
    correlations, best = analyze_rhmc_components()