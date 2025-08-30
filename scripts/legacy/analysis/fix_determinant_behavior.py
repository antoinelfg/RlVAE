#!/usr/bin/env python3
"""
Fix Determinant Behavior
========================

Solutions to fix the backwards det(G⁻¹) behavior.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.components.native_inverse_metric import NativeInverseMetricTensor

def create_fixed_metric(latent_data, device, method="lower_temp"):
    """Create a fixed metric that doesn't have backwards determinant behavior."""
    
    class DummyModel:
        pass
    model = DummyModel()
    
    if method == "lower_temp":
        print("🔧 FIX 1: Lower temperature (2.0 → 0.5)")
        native_metric = NativeInverseMetricTensor.from_model_data(
            model, latent_data, 
            n_centroids=25,
            temperature=0.5,  # Much lower temperature
            regularization=1e-4,
            device=device
        )
        
    elif method == "normalize_dets":
        print("🔧 FIX 2: Normalize determinants")
        # Create metric with normal temperature
        native_metric = NativeInverseMetricTensor.from_model_data(
            model, latent_data, 
            n_centroids=25,
            temperature=2.0,
            regularization=1e-4,
            device=device
        )
        
        # Normalize the determinants to reduce variation
        dets = torch.det(native_metric.inverse_metrics)
        mean_det = dets.mean()
        
        # Scale all matrices to have similar determinants
        for i in range(len(native_metric.inverse_metrics)):
            current_det = torch.det(native_metric.inverse_metrics[i])
            scale_factor = (mean_det / current_det).sqrt()
            native_metric.inverse_metrics[i] = native_metric.inverse_metrics[i] * scale_factor
        
        # Recompute log determinants
        native_metric.log_det_inverse_metrics = torch.log(torch.det(native_metric.inverse_metrics))
        
    elif method == "min_variance":
        print("🔧 FIX 3: Minimum variance regularization")
        native_metric = NativeInverseMetricTensor.from_model_data(
            model, latent_data, 
            n_centroids=25,
            temperature=1.0,  # Moderate temperature
            regularization=0.1,  # Higher regularization
            device=device
        )
    
    return native_metric

def test_all_fixes():
    """Test all the fixes and compare behaviors."""
    print("🧪 TESTING ALL DETERMINANT FIXES")
    print("="*50)
    
    # Load data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    latent_data = torch.randn(6400, 2, device=device) * 1.5
    latent_data = torch.clamp(latent_data, -2.1, 2.2)
    
    # We'll test the fixes directly
    
    # Test fixes
    fixes = ["lower_temp", "normalize_dets", "min_variance"]
    metrics = {}
    
    for fix in fixes:
        print(f"\n{len(metrics)+1}. Testing {fix}:")
        metrics[fix] = create_fixed_metric(latent_data, device, fix)
    
    # Create test grid
    x = np.linspace(-4, 4, 50)  # Smaller grid for speed
    y = np.linspace(-4, 4, 50)
    X, Y = np.meshgrid(x, y)
    grid_points = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), 
                              dtype=torch.float32, device=device)
    
    # Test each fix
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    test_points = [
        torch.tensor([0.0, 0.0]),     # Near data
        torch.tensor([-3.0, -3.0]),   # Empty corner
    ]
    
    for i, (name, metric) in enumerate(metrics.items()):
        print(f"\nTesting {name}:")
        
        # Compute grid
        with torch.no_grad():
            G_inv, log_det_G_inv = metric(grid_points)
            det_grid = torch.exp(log_det_G_inv).cpu().numpy().reshape(X.shape)
        
        vmin, vmax = det_grid.min(), det_grid.max()
        centroids = metric.centroids.cpu()
        
        # Test specific points
        for j, point in enumerate(test_points):
            with torch.no_grad():
                _, log_det_test = metric(point.unsqueeze(0).to(device))
                det_test = torch.exp(log_det_test).item()
            print(f"   Point {point.tolist()}: det = {det_test:.1f}")
        
        # Compute correlation
        distances_grid = np.zeros_like(det_grid)
        for ii in range(X.shape[0]):
            for jj in range(X.shape[1]):
                point = torch.tensor([X[ii,jj], Y[ii,jj]])
                dists = torch.norm(point.unsqueeze(0) - centroids, dim=1)
                distances_grid[ii,jj] = dists.min().item()
        
        correlation = np.corrcoef(distances_grid.flatten(), det_grid.flatten())[0, 1]
        print(f"   Distance vs det correlation: {correlation:.3f}")
        
        # Plot
        ax = axes[i]
        contour = ax.contourf(X, Y, det_grid, levels=30, cmap='viridis', alpha=0.8)
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='*',
                  edgecolors='white', linewidth=2)
        plt.colorbar(contour, ax=ax, label='det(G⁻¹)')
        
        status = "✅ FIXED" if correlation < -0.2 else "❌ STILL BROKEN"
        ax.set_title(f'{name}\nCorr: {correlation:.3f} {status}', fontweight='bold')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
    
    plt.tight_layout()
    plt.savefig('fix_determinant_behavior.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Fix testing complete! Check fix_determinant_behavior.png")

if __name__ == "__main__":
    test_all_fixes()