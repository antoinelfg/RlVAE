#!/usr/bin/env python3
"""
Fix True RHMC Implementation
============================

Implement proper Riemannian HMC that actually uses the metric G⁻¹(z).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.components.native_inverse_metric import NativeInverseMetricTensor

def proper_rhmc_step(z, momentum, native_metric, eps=0.01):
    """
    Proper Riemannian HMC step using the metric correctly.
    
    In true RHMC:
    1. p' = p - eps * ∇_z U(z)  (momentum update)
    2. z' = z + eps * G⁻¹(z) @ p'  (position update using metric)
    
    Where ∇_z U(z) should include the volume correction term.
    """
    with torch.no_grad():
        # Get metric at current position
        G_inv, log_det_G_inv = native_metric(z.unsqueeze(0))
        G_inv = G_inv.squeeze(0)
        
        # Compute potential energy gradient (attraction to centroids)
        centroids = native_metric.centroids
        distances = torch.norm(z.unsqueeze(0) - centroids, dim=1)
        closest_centroid_idx = torch.argmin(distances)
        closest_centroid = centroids[closest_centroid_idx]
        
        # Main potential: quadratic wells around centroids
        potential_grad = 2.0 * (z - closest_centroid)  # Note: positive for repulsion from minimum
        
        # Add multi-centroid attraction for smoother manifold
        k_nearest = 3
        _, nearest_indices = torch.topk(distances, k_nearest, largest=False)
        multi_centroid_grad = torch.zeros_like(z)
        for idx in nearest_indices:
            weight = 1.0 / (distances[idx] + 1e-6)
            multi_centroid_grad += weight * (z - centroids[idx])  # Repulsion from minimum
        multi_centroid_grad = 0.1 * multi_centroid_grad / k_nearest
        
        # Volume correction term: ∇_z log det(G⁻¹(z))
        # This is the proper Riemannian correction!
        eps_small = 1e-4
        volume_grad = torch.zeros_like(z)
        for i in range(len(z)):
            z_plus = z.clone()
            z_minus = z.clone()
            z_plus[i] += eps_small
            z_minus[i] -= eps_small
            
            _, log_det_plus = native_metric(z_plus.unsqueeze(0))
            _, log_det_minus = native_metric(z_minus.unsqueeze(0))
            
            volume_grad[i] = (log_det_plus - log_det_minus) / (2 * eps_small)
        
        # Total gradient (proper Riemannian!)
        total_grad = potential_grad + multi_centroid_grad + 0.5 * volume_grad
        
        # Update momentum (standard)
        momentum_new = momentum - eps * total_grad
        
        # Update position using metric (the Riemannian part!)
        # z' = z + eps * G⁻¹ @ p
        z_new = z + eps * torch.mv(G_inv, momentum_new)
        
        return z_new, momentum_new

def compare_rhmc_implementations():
    """Compare the old 'fake' RHMC with proper Riemannian HMC."""
    print("🔧 COMPARING RHMC IMPLEMENTATIONS")
    print("="*50)
    
    # Setup
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
    
    print(f"   Created metric with {len(native_metric.centroids)} centroids")
    
    # Test both implementations on the same starting point
    print("\n1. TESTING SINGLE STEP COMPARISON...")
    
    start_z = torch.tensor([1.0, 0.5], device=device)
    start_momentum = torch.tensor([0.1, -0.2], device=device)
    
    print(f"   Starting point: [{start_z[0]:.3f}, {start_z[1]:.3f}]")
    print(f"   Starting momentum: [{start_momentum[0]:.3f}, {start_momentum[1]:.3f}]")
    
    # Old implementation (from the original code)
    def old_rhmc_step(z, momentum, eps=0.015):
        with torch.no_grad():
            G_inv, log_det_G_inv = native_metric(z.unsqueeze(0))
            G_inv = G_inv.squeeze(0)
            det_G_inv = torch.exp(log_det_G_inv).squeeze(0)
            
            distances = torch.norm(z.unsqueeze(0) - native_metric.centroids, dim=1)
            closest_centroid_idx = torch.argmin(distances)
            closest_centroid = native_metric.centroids[closest_centroid_idx]
            
            potential_grad = -2.0 * (z - closest_centroid)
            
            k_nearest = 3
            _, nearest_indices = torch.topk(distances, k_nearest, largest=False)
            multi_centroid_grad = torch.zeros_like(z)
            for idx in nearest_indices:
                weight = 1.0 / (distances[idx] + 1e-6)
                multi_centroid_grad += weight * (native_metric.centroids[idx] - z)
            multi_centroid_grad = 0.5 * multi_centroid_grad / k_nearest
            
            total_grad = potential_grad + multi_centroid_grad
            
            # The problematic adaptive step size!
            adaptive_eps = eps / (1.0 + 0.1 * det_G_inv)
            
            momentum_new = momentum - adaptive_eps * total_grad
            z_new = z + adaptive_eps * torch.mv(G_inv, momentum_new)
            
            return z_new, momentum_new
    
    # Compare results
    z_old, momentum_old = old_rhmc_step(start_z.clone(), start_momentum.clone())
    z_new, momentum_new = proper_rhmc_step(start_z.clone(), start_momentum.clone(), native_metric, eps=0.01)
    
    print(f"\n   OLD RHMC result:")
    print(f"     New z: [{z_old[0]:.6f}, {z_old[1]:.6f}]")
    print(f"     Step size: [{(z_old - start_z)[0]:.6f}, {(z_old - start_z)[1]:.6f}]")
    print(f"     Step magnitude: {torch.norm(z_old - start_z):.6f}")
    
    print(f"\n   PROPER RHMC result:")
    print(f"     New z: [{z_new[0]:.6f}, {z_new[1]:.6f}]")
    print(f"     Step size: [{(z_new - start_z)[0]:.6f}, {(z_new - start_z)[1]:.6f}]")
    print(f"     Step magnitude: {torch.norm(z_new - start_z):.6f}")
    
    step_ratio = torch.norm(z_new - start_z) / torch.norm(z_old - start_z)
    print(f"\n   Step magnitude ratio (proper/old): {step_ratio:.3f}")
    
    # Test trajectories
    print("\n2. COMPARING FULL TRAJECTORIES...")
    
    def run_trajectory(z_start, momentum_start, rhmc_func, n_steps=50, label=""):
        trajectory = [z_start.clone().cpu()]
        z, momentum = z_start.clone(), momentum_start.clone()
        
        for step in range(n_steps):
            if rhmc_func == old_rhmc_step:
                z, momentum = rhmc_func(z, momentum)
            else:
                z, momentum = rhmc_func(z, momentum, native_metric)
            z = torch.clamp(z, -4.0, 4.0)  # Same bounds as original
            trajectory.append(z.clone().cpu())
        
        trajectory = torch.stack(trajectory)
        print(f"   {label} trajectory:")
        print(f"     Start: [{trajectory[0, 0]:.3f}, {trajectory[0, 1]:.3f}]")
        print(f"     End: [{trajectory[-1, 0]:.3f}, {trajectory[-1, 1]:.3f}]")
        print(f"     Total distance: {torch.norm(trajectory[-1] - trajectory[0]):.3f}")
        
        return trajectory
    
    start_points = [
        torch.tensor([0.0, 0.0], device=device),
        torch.tensor([2.0, 1.0], device=device),
        torch.tensor([-1.0, -1.0], device=device),
    ]
    
    trajectories_old = []
    trajectories_new = []
    
    for i, start_z in enumerate(start_points):
        momentum = torch.randn(2, device=device) * 0.1
        
        print(f"\n   Trajectory {i+1} from [{start_z[0]:.1f}, {start_z[1]:.1f}]:")
        
        traj_old = run_trajectory(start_z, momentum.clone(), old_rhmc_step, label="OLD")
        traj_new = run_trajectory(start_z, momentum.clone(), proper_rhmc_step, label="PROPER")
        
        trajectories_old.append(traj_old)
        trajectories_new.append(traj_new)
    
    # Create visualization
    print("\n3. CREATING VISUALIZATION...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    centroids_cpu = native_metric.centroids.cpu()
    latent_cpu = latent_data.cpu()
    
    # Plot 1: Old RHMC trajectories
    ax1 = axes[0, 0]
    ax1.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='lightblue', s=1, alpha=0.3, label='Data')
    ax1.scatter(centroids_cpu[:, 0], centroids_cpu[:, 1], c='red', s=100, marker='*',
               edgecolors='black', linewidth=1, label='Centroids')
    
    colors = ['orange', 'green', 'purple']
    for i, traj in enumerate(trajectories_old):
        ax1.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2, alpha=0.8,
                label=f'Trajectory {i+1}')
        ax1.scatter(traj[0, 0], traj[0, 1], color=colors[i], s=100, marker='o', 
                   edgecolors='black', zorder=10)
    
    ax1.set_title('1. OLD RHMC (Tiny Steps)', fontweight='bold')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: New RHMC trajectories
    ax2 = axes[0, 1]
    ax2.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='lightblue', s=1, alpha=0.3, label='Data')
    ax2.scatter(centroids_cpu[:, 0], centroids_cpu[:, 1], c='red', s=100, marker='*',
               edgecolors='black', linewidth=1, label='Centroids')
    
    for i, traj in enumerate(trajectories_new):
        ax2.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2, alpha=0.8,
                label=f'Trajectory {i+1}')
        ax2.scatter(traj[0, 0], traj[0, 1], color=colors[i], s=100, marker='o', 
                   edgecolors='black', zorder=10)
    
    ax2.set_title('2. PROPER RHMC (Real Steps)', fontweight='bold')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Step size comparison
    ax3 = axes[1, 0]
    
    # Compute step sizes across space
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, y)
    
    step_sizes_old = np.zeros_like(X)
    step_sizes_new = np.zeros_like(X)
    
    test_momentum = torch.tensor([0.1, 0.1], device=device)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = torch.tensor([X[i,j], Y[i,j]], device=device, dtype=torch.float32)
            
            z_old, _ = old_rhmc_step(point, test_momentum.clone())
            z_new, _ = proper_rhmc_step(point, test_momentum.clone(), native_metric)
            
            step_sizes_old[i,j] = torch.norm(z_old - point).item()
            step_sizes_new[i,j] = torch.norm(z_new - point).item()
    
    contour3 = ax3.contourf(X, Y, step_sizes_old, levels=20, cmap='plasma', alpha=0.8)
    ax3.scatter(centroids_cpu[:, 0], centroids_cpu[:, 1], c='white', s=100, marker='*',
               edgecolors='black', linewidth=2)
    plt.colorbar(contour3, ax=ax3, label='Step Size')
    ax3.set_title('3. OLD RHMC Step Sizes\n(Tiny due to det adaptation)', fontweight='bold')
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-3, 3)
    
    # Plot 4: New step sizes
    ax4 = axes[1, 1]
    contour4 = ax4.contourf(X, Y, step_sizes_new, levels=20, cmap='plasma', alpha=0.8)
    ax4.scatter(centroids_cpu[:, 0], centroids_cpu[:, 1], c='white', s=100, marker='*',
               edgecolors='black', linewidth=2)
    plt.colorbar(contour4, ax=ax4, label='Step Size')
    ax4.set_title('4. PROPER RHMC Step Sizes\n(Reasonable and metric-aware)', fontweight='bold')
    ax4.set_xlim(-3, 3)
    ax4.set_ylim(-3, 3)
    
    plt.tight_layout()
    plt.savefig('fix_true_rhmc.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Comparison complete! Check fix_true_rhmc.png")
    print(f"\n🎯 SUMMARY:")
    print(f"   OLD RHMC: Tiny steps due to det(G⁻¹) adaptation")
    print(f"   PROPER RHMC: Uses metric correctly with volume correction")
    print(f"   Step ratio: {step_ratio:.1f}x larger steps with proper implementation")
    
    return trajectories_old, trajectories_new

if __name__ == "__main__":
    old_trajs, new_trajs = compare_rhmc_implementations()