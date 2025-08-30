#!/usr/bin/env python3
"""
Show Metric Construction Process
===============================

This script shows exactly how the covariance matrices (Mⱼ) are used to construct the metric G⁻¹.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.components.native_inverse_metric import NativeInverseMetricTensor

def show_metric_construction():
    """Show how the metric G⁻¹ is constructed from covariance matrices."""
    print("🔧 SHOWING METRIC CONSTRUCTION PROCESS")
    print("="*50)
    
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
    inverse_metrics = native_metric.inverse_metrics  # These are the Mⱼ matrices
    print(f"🎯 Created metric with {len(centroids)} centroids")
    print(f"📊 Inverse metrics shape: {inverse_metrics.shape}")
    
    # Choose a test point
    test_point_2d = torch.tensor([0.0, 0.0], device=device)
    test_point_16d = torch.zeros(16, device=device)
    test_point_16d[:2] = test_point_2d
    test_point_16d[2:] = latent_data.mean(dim=0)[2:]
    
    print(f"\n🎯 ANALYZING METRIC AT TEST POINT: {test_point_2d.cpu().numpy()}")
    print("-" * 50)
    
    # Step 1: Compute distances to all centroids
    distances = torch.norm(test_point_16d.unsqueeze(0) - centroids, dim=1)
    print(f"📏 Distances to centroids: [{distances.min():.3f}, {distances.max():.3f}]")
    
    # Step 2: Compute weights using softmax
    weights = torch.softmax(-distances / 0.5, dim=0)  # temperature = 0.5
    print(f"⚖️  Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"⚖️  Sum of weights: {weights.sum():.3f}")
    
    # Show top 5 weights
    sorted_indices = torch.argsort(weights, descending=True)
    print(f"\n🎯 Top 5 contributing centroids:")
    for i in range(5):
        idx = sorted_indices[i].item()
        dist = distances[idx].item()
        weight = weights[idx].item()
        print(f"   {i+1}. Centroid {idx}: distance={dist:.3f}, weight={weight:.3f}")
    
    # Step 3: Show the metric construction formula
    print(f"\n🔧 METRIC CONSTRUCTION FORMULA:")
    print("-" * 35)
    print("G⁻¹(z) = Σ wⱼ(z) Mⱼ + λI")
    print("where:")
    print("  - wⱼ(z) = softmax(-||z - cⱼ|| / temperature)")
    print("  - Mⱼ are the inverse metric matrices at centroids")
    print("  - λI is regularization (λ = 0.0001)")
    
    # Step 4: Compute the actual metric at this point
    with torch.no_grad():
        G_inv_test, log_det_test = native_metric(test_point_16d.unsqueeze(0))
        G_inv_test = G_inv_test[0]  # Remove batch dimension
        det_test = torch.exp(log_det_test).item()
    
    print(f"\n✅ COMPUTED METRIC:")
    print(f"   det(G⁻¹(z)) = {det_test:.1f}")
    print(f"   G⁻¹(z) shape: {G_inv_test.shape}")
    
    # Step 5: Manually reconstruct the metric to verify
    print(f"\n🔍 MANUAL RECONSTRUCTION:")
    print("-" * 30)
    
    # Initialize with regularization
    lambda_reg = 0.0001
    G_inv_manual = torch.eye(16, device=device) * lambda_reg
    
    # Add weighted contributions from each centroid
    for i in range(len(centroids)):
        weight = weights[i].item()
        M = inverse_metrics[i]
        contribution = weight * M
        G_inv_manual += contribution
        
        if i < 5:  # Show first 5 contributions
            det_contribution = torch.det(contribution).item()
            print(f"   Centroid {i}: weight={weight:.3f}, det(contribution)={det_contribution:.1f}")
    
    det_manual = torch.det(G_inv_manual).item()
    print(f"   Manual det(G⁻¹(z)) = {det_manual:.1f}")
    print(f"   Difference: {abs(det_test - det_manual):.3f}")
    
    # Step 6: Create visualization
    print(f"\n🎨 CREATING CONSTRUCTION VISUALIZATION:")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Centroid positions with weights
    ax1 = axes[0, 0]
    centroid_positions = centroids.cpu().numpy()
    scatter = ax1.scatter(centroid_positions[:, 0], centroid_positions[:, 1], 
                         c=weights.cpu().numpy(), s=150, cmap='viridis', marker='*',
                         edgecolors='white', linewidth=2)
    ax1.scatter(test_point_2d[0].item(), test_point_2d[1].item(), c='red', s=200, marker='X',
               edgecolors='black', linewidth=3, label='Test Point', zorder=10)
    plt.colorbar(scatter, ax=ax1, label='Weight wⱼ(z)')
    ax1.set_title('1. Centroid Weights at Test Point\n(Red X = test point)', fontweight='bold')
    ax1.set_xlabel('z₁ (first dimension)')
    ax1.set_ylabel('z₂ (second dimension)')
    ax1.legend()
    
    # Plot 2: Weight distribution
    ax2 = axes[0, 1]
    centroid_indices = np.arange(len(centroids))
    ax2.bar(centroid_indices, weights.cpu().numpy(), alpha=0.7)
    ax2.set_xlabel('Centroid Index')
    ax2.set_ylabel('Weight wⱼ(z)')
    ax2.set_title('2. Weight Distribution\nsoftmax(-||z - cⱼ|| / T)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distance vs weight
    ax3 = axes[0, 2]
    ax3.scatter(distances.cpu().numpy(), weights.cpu().numpy(), alpha=0.7, s=50)
    ax3.set_xlabel('Distance to Centroid')
    ax3.set_ylabel('Weight')
    ax3.set_title('3. Distance vs Weight\n(Should be decreasing)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Show the final G⁻¹ matrix
    ax4 = axes[1, 0]
    im = ax4.imshow(G_inv_test.cpu().numpy(), cmap='viridis', alpha=0.8)
    ax4.set_title(f'4. Final G⁻¹(z) Matrix\ndet = {det_test:.1f}', fontweight='bold')
    plt.colorbar(im, ax=ax4)
    
    # Plot 5: Show a few individual Mⱼ matrices
    ax5 = axes[1, 1]
    # Show the matrix with highest weight
    max_weight_idx = torch.argmax(weights).item()
    M_max_weight = inverse_metrics[max_weight_idx].cpu().numpy()
    im = ax5.imshow(M_max_weight, cmap='viridis', alpha=0.8)
    ax5.set_title(f'5. Highest Weight Matrix M_{max_weight_idx}\nweight = {weights[max_weight_idx]:.3f}', fontweight='bold')
    plt.colorbar(im, ax=ax5)
    
    # Plot 6: Construction breakdown
    ax6 = axes[1, 2]
    # Show the contribution breakdown
    top_contributions = []
    top_labels = []
    for i in range(5):
        idx = sorted_indices[i].item()
        weight = weights[idx].item()
        det_M = torch.det(inverse_metrics[idx]).item()
        contribution = weight * det_M
        top_contributions.append(contribution)
        top_labels.append(f'C{idx}')
    
    ax6.bar(top_labels, top_contributions, alpha=0.7, color='orange')
    ax6.set_ylabel('Weight × det(Mⱼ)')
    ax6.set_title('6. Top 5 Contributions\nto det(G⁻¹(z))', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('show_metric_construction.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Construction visualization saved to: show_metric_construction.png")
    
    # Summary
    print(f"\n📝 SUMMARY:")
    print("=" * 20)
    print(f"Test point: {test_point_2d.cpu().numpy()}")
    print(f"Number of centroids: {len(centroids)}")
    print(f"Temperature: 0.5")
    print(f"Regularization: 0.0001")
    print(f"Final det(G⁻¹(z)): {det_test:.1f}")
    print(f"Top contributor: Centroid {sorted_indices[0].item()} (weight={weights[sorted_indices[0]]:.3f})")
    
    print(f"\n💡 KEY INSIGHTS:")
    print("=" * 20)
    print("1. Each centroid has its own 16×16 metric matrix Mⱼ")
    print("2. Weights are computed using softmax over distances")
    print("3. The final metric is a weighted combination: G⁻¹(z) = Σ wⱼ(z) Mⱼ + λI")
    print("4. Temperature controls how 'sharp' the weighting is")
    print("5. Regularization ensures the metric is well-conditioned")

if __name__ == "__main__":
    show_metric_construction() 