#!/usr/bin/env python3
"""
Investigate Metric Confusion
===========================

Investigate the exact confusion between G and G⁻¹ to understand where RHMC is actually going.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.models.riemannian_flow_vae import RiemannianFlowVAE
from dual_rhmc_implementation import DualRiemannianHMCSampler


def investigate_metric_behavior():
    """Investigate what's actually happening with the metrics."""
    print("🔍 Investigating Metric Behavior")
    print("=" * 60)
    
    # Create model
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
    
    # Create test data and centroids
    np.random.seed(42)
    latent_data = np.random.randn(1000, 2) * 2.0
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    kmeans.fit(latent_data)
    centroids = kmeans.cluster_centers_
    
    # Create metric matrices
    metric_matrices = []
    for i, centroid in enumerate(centroids):
        distances = np.linalg.norm(latent_data - centroid, axis=1)
        closest_indices = np.argsort(distances)[:50]
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
    
    # Load centroids and metrics into model
    centroids_tensor = torch.tensor(centroids, dtype=torch.float32, device=device)
    metric_matrices_tensor = torch.tensor(metric_matrices, dtype=torch.float32, device=device)
    model.load_pretrained_metrics_from_tensor(centroids_tensor, metric_matrices_tensor, 
                                            temperature=0.3, regularization=0.01)
    
    # Test specific points
    test_points = torch.tensor([
        [0.0, 0.0],      # Center
        [1.0, 1.0],      # Near a centroid
        [-1.0, -1.0],    # Near a centroid
        [3.0, 0.0],      # Border region
    ], dtype=torch.float32, device=device)
    
    print("\n📊 Analyzing Metric Behavior at Test Points:")
    print("=" * 60)
    
    for i, point in enumerate(test_points):
        print(f"\n--- Point {i+1}: {point.cpu().numpy()} ---")
        
        # Compute G and G⁻¹
        with torch.no_grad():
            G_z = model.G(point.unsqueeze(0))
            G_inv = torch.linalg.inv(G_z)
            det_G = torch.linalg.det(G_z)
            det_G_inv = torch.linalg.det(G_inv)
        
        print(f"det(G): {det_G.item():.3e}")
        print(f"det(G⁻¹): {det_G_inv.item():.3e}")
        print(f"det(G) * det(G⁻¹): {det_G.item() * det_G_inv.item():.6f}")
        
        # Find nearest centroid
        distances_to_centroids = torch.norm(point.unsqueeze(0) - centroids_tensor, dim=1)
        nearest_centroid_idx = torch.argmin(distances_to_centroids)
        nearest_centroid = centroids_tensor[nearest_centroid_idx]
        min_distance = distances_to_centroids[nearest_centroid_idx]
        
        print(f"Nearest centroid: {nearest_centroid.cpu().numpy()}")
        print(f"Distance to centroid: {min_distance.item():.3f}")
        
        # Check if this is a high or low det(G⁻¹) region
        if det_G_inv.item() > 1e-2:
            print("📍 HIGH det(G⁻¹) region (near centroid)")
        else:
            print("📍 LOW det(G⁻¹) region (away from centroid)")
    
    return model


def test_rhmc_exploration():
    """Test where RHMC actually explores."""
    print("\n🎯 Testing RHMC Exploration")
    print("=" * 60)
    
    model = investigate_metric_behavior()
    
    # Create RHMC sampler
    sampler = DualRiemannianHMCSampler(model, mcmc_steps_nbr=20, n_lf=5, eps_lf=0.01)
    
    # Sample from different starting points
    starting_points = [
        torch.tensor([[0.0, 0.0]], device=model.device),      # Center (low det(G⁻¹))
        torch.tensor([[1.0, 1.0]], device=model.device),      # Near centroid (high det(G⁻¹))
        torch.tensor([[-1.0, -1.0]], device=model.device),    # Near centroid (high det(G⁻¹))
        torch.tensor([[3.0, 0.0]], device=model.device),      # Border (low det(G⁻¹))
    ]
    
    for i, start_point in enumerate(starting_points):
        print(f"\n--- Starting from point {i+1}: {start_point.cpu().numpy().flatten()} ---")
        
        # Check initial det(G⁻¹)
        with torch.no_grad():
            G_start = model.G(start_point)
            G_inv_start = torch.linalg.inv(G_start)
            det_G_inv_start = torch.linalg.det(G_inv_start)
        
        print(f"Initial det(G⁻¹): {det_G_inv_start.item():.3e}")
        
        # Run a few RHMC steps manually
        z = start_point.clone()
        for step in range(5):
            # Initialize momentum
            p = sampler._initialize_momentum(z)
            
            # Compute Hamiltonian
            H_initial = sampler._compute_hamiltonian(z, p)
            
            # One leapfrog step
            z_new, p_new = sampler._leapfrog_step(z, p, sampler.eps_lf)
            
            # Check new det(G⁻¹)
            with torch.no_grad():
                G_new = model.G(z_new)
                G_inv_new = torch.linalg.inv(G_new)
                det_G_inv_new = torch.linalg.det(G_inv_new)
            
            print(f"Step {step+1}: z={z_new.detach().cpu().numpy().flatten()}, det(G⁻¹)={det_G_inv_new.item():.3e}")
            
            # Update position
            z = z_new
        
        # Final position analysis
        print(f"Final position: {z.detach().cpu().numpy().flatten()}")
        print(f"Final det(G⁻¹): {det_G_inv_new.item():.3e}")
        
        if det_G_inv_new.item() > det_G_inv_start.item():
            print("✅ RHMC moved toward HIGHER det(G⁻¹) (CORRECT for G⁻¹ metric)")
        else:
            print("❌ RHMC moved toward LOWER det(G⁻¹) (WRONG for G⁻¹ metric)")


def visualize_gradient_direction():
    """Visualize which direction the gradients are pointing."""
    print("\n🧭 Visualizing Gradient Direction")
    print("=" * 60)
    
    model = investigate_metric_behavior()
    
    # Create a grid of points
    x = np.linspace(-3, 3, 10)
    y = np.linspace(-3, 3, 10)
    X, Y = np.meshgrid(x, y)
    
    # Test points
    test_points = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), 
                              dtype=torch.float32, device=model.device)
    
    gradients = []
    det_G_inv_values = []
    
    print("Computing gradients at grid points...")
    
    for i in range(len(test_points)):
        point = test_points[i].clone()
        point.requires_grad_(True)
        
        G_z = model.G(point.unsqueeze(0))
        G_inv = torch.linalg.inv(G_z)
        
        # Total energy (what RHMC is trying to minimize)
        potential_energy = 0.5 * torch.einsum('bi,bij,bj->b', point.unsqueeze(0), G_inv, point.unsqueeze(0))
        volume_correction = 0.5 * torch.log(torch.linalg.det(G_inv))
        total_energy = potential_energy + volume_correction
        
        # Compute gradient
        grad = torch.autograd.grad(total_energy, point)[0]
        gradients.append(grad.cpu().numpy())
        
        # Store det(G⁻¹)
        with torch.no_grad():
            det_G_inv = torch.linalg.det(G_inv)
            det_G_inv_values.append(det_G_inv.item())
    
    gradients = np.array(gradients)
    det_G_inv_values = np.array(det_G_inv_values)
    
    # Reshape for plotting
    grad_x = gradients[:, 0].reshape(X.shape)
    grad_y = gradients[:, 1].reshape(Y.shape)
    det_G_inv_grid = det_G_inv_values.reshape(X.shape)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: det(G⁻¹) with gradient arrows
    ax1 = axes[0]
    contour = ax1.contourf(X, Y, det_G_inv_grid, levels=20, cmap='viridis')
    ax1.quiver(X, Y, grad_x, grad_y, alpha=0.7, scale=50, color='red')
    ax1.set_title("det(G⁻¹) with Energy Gradients\n(Red arrows show where RHMC moves)")
    ax1.set_xlabel("z₁")
    ax1.set_ylabel("z₂")
    plt.colorbar(contour, ax=ax1, label='det(G⁻¹)')
    
    # Plot 2: Gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    ax2 = axes[1]
    contour2 = ax2.contourf(X, Y, grad_magnitude, levels=20, cmap='plasma')
    ax2.set_title("Gradient Magnitude")
    ax2.set_xlabel("z₁")
    ax2.set_ylabel("z₂")
    plt.colorbar(contour2, ax=ax2, label='|∇Energy|')
    
    plt.tight_layout()
    plt.savefig("gradient_direction_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ det(G⁻¹) range: [{det_G_inv_values.min():.3e}, {det_G_inv_values.max():.3e}]")
    print(f"✅ Gradient magnitude range: [{grad_magnitude.min():.3e}, {grad_magnitude.max():.3e}]")
    
    # Check if gradients point toward high det(G⁻¹) regions
    high_det_mask = det_G_inv_values > np.percentile(det_G_inv_values, 75)
    low_det_mask = det_G_inv_values < np.percentile(det_G_inv_values, 25)
    
    print(f"✅ High det(G⁻¹) points: {np.sum(high_det_mask)}")
    print(f"✅ Low det(G⁻¹) points: {np.sum(low_det_mask)}")


if __name__ == "__main__":
    investigate_metric_behavior()
    test_rhmc_exploration()
    visualize_gradient_direction()