#!/usr/bin/env python3
"""
Investigate True RHMC Usage
===========================

Check if we're REALLY doing Riemannian HMC or just "fake RHMC".
Are we using the metric G⁻¹(z) properly in the dynamics?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.components.native_inverse_metric import NativeInverseMetricTensor

def analyze_true_rhmc_implementation():
    """Analyze if we're really using the Riemannian metric in RHMC dynamics."""
    print("🔍 INVESTIGATING TRUE RHMC IMPLEMENTATION")
    print("="*60)
    
    # Setup like the main script
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    latent_data = torch.randn(6400, 2, device=device) * 1.5
    latent_data = torch.clamp(latent_data, -2.1, 2.2)
    
    class DummyModel:
        pass
    model = DummyModel()
    
    native_metric = NativeInverseMetricTensor.from_model_data(
        model, latent_data, n_centroids=25, temperature=0.5, device=device
    )
    
    centroids = native_metric.centroids
    print(f"   Created metric with {len(centroids)} centroids")
    
    # Let's trace through the ACTUAL RHMC step from the code
    print("\n1. ANALYZING THE ACTUAL RHMC STEP IMPLEMENTATION...")
    
    def analyze_rhmc_step(z, momentum, native_metric, verbose=True):
        """Analyze what the RHMC step is actually doing."""
        if verbose:
            print(f"   Input z: [{z[0]:.3f}, {z[1]:.3f}]")
            print(f"   Input momentum: [{momentum[0]:.3f}, {momentum[1]:.3f}]")
        
        with torch.no_grad():
            # Step 1: Get metric at current position
            G_inv, log_det_G_inv = native_metric(z.unsqueeze(0))
            G_inv = G_inv.squeeze(0)
            det_G_inv = torch.exp(log_det_G_inv).squeeze(0)
            
            if verbose:
                print(f"   G⁻¹ matrix:\n     {G_inv}")
                print(f"   det(G⁻¹): {det_G_inv.item():.1f}")
            
            # Step 2: Compute potential energy gradient (THIS IS THE KEY!)
            distances = torch.norm(z.unsqueeze(0) - centroids, dim=1)
            closest_centroid_idx = torch.argmin(distances)
            closest_centroid = centroids[closest_centroid_idx]
            
            # This is just EUCLIDEAN gradient!
            potential_grad = -2.0 * (z - closest_centroid)
            
            if verbose:
                print(f"   Closest centroid: [{closest_centroid[0]:.3f}, {closest_centroid[1]:.3f}]")
                print(f"   Euclidean potential grad: [{potential_grad[0]:.3f}, {potential_grad[1]:.3f}]")
            
            # Step 3: Multi-centroid part (also Euclidean)
            k_nearest = 3
            _, nearest_indices = torch.topk(distances, k_nearest, largest=False)
            multi_centroid_grad = torch.zeros_like(z)
            for idx in nearest_indices:
                weight = 1.0 / (distances[idx] + 1e-6)
                multi_centroid_grad += weight * (centroids[idx] - z)
            multi_centroid_grad = 0.5 * multi_centroid_grad / k_nearest
            
            total_grad = potential_grad + multi_centroid_grad
            
            if verbose:
                print(f"   Total gradient (Euclidean): [{total_grad[0]:.3f}, {total_grad[1]:.3f}]")
            
            # Step 4: Adaptive step size
            base_eps = 0.015
            adaptive_eps = base_eps / (1.0 + 0.1 * det_G_inv)
            
            if verbose:
                print(f"   Adaptive step size: {adaptive_eps:.6f}")
            
            # Step 5: Update momentum - THIS IS WHERE RIEMANNIAN SHOULD MATTER!
            momentum_new = momentum - adaptive_eps * total_grad
            
            if verbose:
                print(f"   New momentum: [{momentum_new[0]:.3f}, {momentum_new[1]:.3f}]")
            
            # Step 6: Update position using metric - THIS IS THE RIEMANNIAN PART!
            z_new = z + adaptive_eps * torch.mv(G_inv, momentum_new)
            
            if verbose:
                print(f"   G⁻¹ @ momentum: [{torch.mv(G_inv, momentum_new)[0]:.3f}, {torch.mv(G_inv, momentum_new)[1]:.3f}]")
                print(f"   New z: [{z_new[0]:.3f}, {z_new[1]:.3f}]")
                print(f"   Step vector: [{(z_new - z)[0]:.3f}, {(z_new - z)[1]:.3f}]")
            
            return z_new, momentum_new, {
                'G_inv': G_inv,
                'det_G_inv': det_G_inv,
                'potential_grad': total_grad,
                'adaptive_eps': adaptive_eps,
                'momentum_new': momentum_new,
                'step_vector': z_new - z
            }
    
    # Test a specific point
    print("\n2. TESTING RHMC STEP AT SPECIFIC POINT...")
    test_z = torch.tensor([1.0, 0.5], device=device)
    test_momentum = torch.tensor([0.1, -0.2], device=device)
    
    z_new, momentum_new, info = analyze_rhmc_step(test_z, test_momentum, native_metric)
    
    # Now let's compare: what if we used IDENTITY metric vs actual metric?
    print("\n3. COMPARING WITH IDENTITY METRIC...")
    
    # Identity metric case
    G_inv_identity = torch.eye(2, device=device)
    z_new_identity = test_z + info['adaptive_eps'] * torch.mv(G_inv_identity, info['momentum_new'])
    
    print(f"   With G⁻¹ (actual):   [{z_new[0]:.3f}, {z_new[1]:.3f}]")
    print(f"   With I (identity):   [{z_new_identity[0]:.3f}, {z_new_identity[1]:.3f}]")
    print(f"   Difference:          [{(z_new - z_new_identity)[0]:.3f}, {(z_new - z_new_identity)[1]:.3f}]")
    
    metric_effect_magnitude = torch.norm(z_new - z_new_identity).item()
    print(f"   Metric effect magnitude: {metric_effect_magnitude:.6f}")
    
    # Let's test this across the space
    print("\n4. TESTING METRIC EFFECT ACROSS SPACE...")
    
    test_points = torch.tensor([
        [0.0, 0.0],     # Near center
        [2.0, 1.0],     # Near some centroid
        [-2.0, -1.0],   # Different region
        [3.0, 3.0],     # Far from centroids
    ], device=device)
    
    metric_effects = []
    
    for i, point in enumerate(test_points):
        momentum = torch.randn(2, device=device) * 0.1
        z_new, _, info = analyze_rhmc_step(point, momentum, native_metric, verbose=False)
        
        # Compare with identity
        z_new_identity = point + info['adaptive_eps'] * torch.mv(torch.eye(2, device=device), info['momentum_new'])
        effect = torch.norm(z_new - z_new_identity).item()
        metric_effects.append(effect)
        
        print(f"   Point {i} {point.tolist()}: metric effect = {effect:.6f}")
    
    avg_effect = np.mean(metric_effects)
    print(f"   Average metric effect: {avg_effect:.6f}")
    
    # Create visualization showing the difference
    print("\n5. VISUALIZING METRIC VS IDENTITY EFFECTS...")
    
    # Create grid
    x = np.linspace(-3, 3, 20)
    y = np.linspace(-3, 3, 20)
    X, Y = np.meshgrid(x, y)
    
    metric_effect_grid = np.zeros_like(X)
    direction_diff_grid = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = torch.tensor([X[i,j], Y[i,j]], device=device, dtype=torch.float32)
            momentum = torch.tensor([0.1, 0.1], device=device)  # Fixed momentum for comparison
            
            z_new, _, info = analyze_rhmc_step(point, momentum, native_metric, verbose=False)
            z_new_identity = point + info['adaptive_eps'] * torch.mv(torch.eye(2, device=device), info['momentum_new'])
            
            effect = torch.norm(z_new - z_new_identity).item()
            metric_effect_grid[i,j] = effect
            
            # Direction difference
            direction_metric = (z_new - point) / torch.norm(z_new - point + 1e-8)
            direction_identity = (z_new_identity - point) / torch.norm(z_new_identity - point + 1e-8)
            direction_diff = torch.norm(direction_metric - direction_identity).item()
            direction_diff_grid[i,j] = direction_diff
    
    # Plot the results
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Metric effect magnitude
    ax1 = axes[0, 0]
    contour1 = ax1.contourf(X, Y, metric_effect_grid, levels=20, cmap='plasma', alpha=0.8)
    ax1.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='white', s=100, marker='*',
               edgecolors='black', linewidth=2)
    plt.colorbar(contour1, ax=ax1, label='|step_metric - step_identity|')
    ax1.set_title('1. Metric Effect Magnitude\n(How much G⁻¹ changes the step)', fontweight='bold')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    
    # Plot 2: Direction difference
    ax2 = axes[0, 1]
    contour2 = ax2.contourf(X, Y, direction_diff_grid, levels=20, cmap='viridis', alpha=0.8)
    ax2.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='red', s=100, marker='*',
               edgecolors='white', linewidth=2)
    plt.colorbar(contour2, ax=ax2, label='Direction difference')
    ax2.set_title('2. Step Direction Difference\n(How much G⁻¹ changes direction)', fontweight='bold')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    
    # Plot 3: Show actual G⁻¹ determinant for comparison
    with torch.no_grad():
        grid_points = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), 
                                  dtype=torch.float32, device=device)
        G_inv, log_det_G_inv = native_metric(grid_points)
        det_grid = torch.exp(log_det_G_inv).cpu().numpy().reshape(X.shape)
    
    ax3 = axes[1, 0]
    contour3 = ax3.contourf(X, Y, det_grid, levels=20, cmap='hot', alpha=0.8)
    ax3.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='cyan', s=100, marker='*',
               edgecolors='black', linewidth=2)
    plt.colorbar(contour3, ax=ax3, label='det(G⁻¹)')
    ax3.set_title('3. Metric Determinant det(G⁻¹)\n(What we thought was important)', fontweight='bold')
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-3, 3)
    
    # Plot 4: Correlation analysis
    ax4 = axes[1, 1]
    
    # Flatten for correlation
    metric_flat = metric_effect_grid.flatten()
    det_flat = det_grid.flatten()
    direction_flat = direction_diff_grid.flatten()
    
    # Plot metric effect vs determinant
    ax4.scatter(det_flat, metric_flat, alpha=0.6, s=20, label='Effect vs det(G⁻¹)')
    ax4.set_xlabel('det(G⁻¹)')
    ax4.set_ylabel('Metric Effect Magnitude')
    
    corr_effect_det = np.corrcoef(det_flat, metric_flat)[0,1]
    ax4.set_title(f'4. Metric Effect vs det(G⁻¹)\nCorrelation: {corr_effect_det:.3f}', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('investigate_true_rhmc_usage.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Final analysis
    print(f"\n6. FINAL ANALYSIS:")
    print(f"   Average metric effect magnitude: {avg_effect:.6f}")
    print(f"   Max metric effect: {np.max(metric_effects):.6f}")
    print(f"   Correlation (metric effect vs det(G⁻¹)): {corr_effect_det:.3f}")
    
    if avg_effect < 1e-3:
        print(f"   🚨 WARNING: Metric effects are tiny! We might be doing 'fake RHMC'")
        print(f"       The metric G⁻¹ has minimal impact on the actual dynamics")
    elif avg_effect > 0.01:
        print(f"   ✅ GOOD: Metric has significant effect on dynamics")
        print(f"        We're doing real Riemannian HMC!")
    else:
        print(f"   ⚠️  MODERATE: Metric has some effect, but not dominant")
    
    print(f"\n🎯 CONCLUSION ABOUT G⁻¹ FORMULA:")
    print(f"   The formula G⁻¹(z) = Σ w_j(z) M_j + λI is being computed correctly.")
    print(f"   However, its EFFECT on RHMC dynamics might be small compared to")
    print(f"   the potential gradient that attracts to centroids.")
    
    print(f"\n✅ Investigation complete! Check investigate_true_rhmc_usage.png")
    
    return avg_effect, corr_effect_det

if __name__ == "__main__":
    avg_effect, correlation = analyze_true_rhmc_implementation()