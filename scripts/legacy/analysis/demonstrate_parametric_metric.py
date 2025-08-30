#!/usr/bin/env python3
"""
Demonstrate Parametric Metric
============================

This script demonstrates the new parametric metric that uses neural networks
to parametrize the lower triangular matrices Lψᵢ.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.components.parametric_inverse_metric import ParametricInverseMetricTensor
from src.models.components.native_inverse_metric import NativeInverseMetricTensor
import torch.nn as nn

def demonstrate_parametric_metric():
    """Demonstrate the parametric metric and compare with fixed metric."""
    print("🎯 DEMONSTRATING PARAMETRIC METRIC")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 16
    n_centroids = 10
    batch_size = 100
    
    # Create test data
    latent_data = torch.randn(500, latent_dim, device=device)
    print(f"📊 Test data shape: {latent_data.shape}")
    
    # Create parametric metric
    print(f"\n🔧 CREATING PARAMETRIC METRIC:")
    print("-" * 30)
    parametric_metric = ParametricInverseMetricTensor(
        latent_dim=latent_dim,
        n_centroids=n_centroids,
        temperature=1.0,
        regularization=1e-4,
        hidden_dim=64,
        num_layers=2,
        device=device
    )
    
    # Initialize parametric metric with better initialization
    with torch.no_grad():
        # Initialize centroids with data
        indices = torch.randperm(len(latent_data))[:n_centroids]
        parametric_metric.centroids.data = latent_data[indices].clone()
        
        # Initialize L matrices with identity + small noise
        for i, net in enumerate(parametric_metric.l_triangular_nets):
            for module in net.modules():
                if isinstance(module, torch.nn.Linear):
                    if module.out_features == parametric_metric.latent_dim * (parametric_metric.latent_dim + 1) // 2:
                        # Output layer - initialize to produce identity-like matrices
                        nn.init.normal_(module.weight, mean=0.0, std=0.1)
                        nn.init.zeros_(module.bias)
                    else:
                        # Hidden layers
                        nn.init.xavier_uniform_(module.weight)
                        nn.init.zeros_(module.bias)
    
    # Create fixed metric for comparison
    print(f"\n🔧 CREATING FIXED METRIC:")
    print("-" * 25)
    class DummyModel:
        pass
    model = DummyModel()
    
    fixed_metric = NativeInverseMetricTensor.from_model_data(
        model, latent_data, 
        n_centroids=n_centroids,
        temperature=1.0,
        device=device
    )
    
    # Test both metrics
    print(f"\n🧪 TESTING BOTH METRICS:")
    print("-" * 25)
    
    # Create test points
    z = torch.randn(batch_size, latent_dim, device=device)
    
    # Test parametric metric
    with torch.no_grad():
        G_inv_param, log_det_param = parametric_metric(z)
        det_param = torch.exp(log_det_param)
        
        # Debug: Check L matrices
        L_matrices = parametric_metric.get_lower_triangular_matrices()
        print(f"   Debug - L matrices shape: {L_matrices.shape}")
        print(f"   Debug - L det range: [{torch.det(L_matrices).min():.1f}, {torch.det(L_matrices).max():.1f}]")
        print(f"   Debug - L L^T det range: [{torch.det(torch.matmul(L_matrices, L_matrices.transpose(-2, -1))).min():.1f}, {torch.det(torch.matmul(L_matrices, L_matrices.transpose(-2, -1))).max():.1f}]")
        
        # Debug: Check weights
        weights = parametric_metric.compute_weights(z)
        print(f"   Debug - Weights range: [{weights.min():.3f}, {weights.max():.3f}]")
        print(f"   Debug - Weights sum: {weights.sum(dim=1).mean():.3f}")
    
    # Test fixed metric
    with torch.no_grad():
        G_inv_fixed, log_det_fixed = fixed_metric(z)
        det_fixed = torch.exp(log_det_fixed)
    
    print(f"✅ Parametric metric:")
    print(f"   det(G⁻¹) range: [{det_param.min():.1f}, {det_param.max():.1f}]")
    print(f"   Mean det: {det_param.mean():.1f}")
    print(f"   Std det: {det_param.std():.1f}")
    
    print(f"✅ Fixed metric:")
    print(f"   det(G⁻¹) range: [{det_fixed.min():.1f}, {det_fixed.max():.1f}]")
    print(f"   Mean det: {det_fixed.mean():.1f}")
    print(f"   Std det: {det_fixed.std():.1f}")
    
    # Get metric info for parametric
    param_info = parametric_metric.get_metric_info()
    
    # Get metric info for fixed (create manually)
    fixed_info = {
        'centroids': fixed_metric.centroids.data.clone(),
        'L_LT_matrices': fixed_metric.inverse_metrics.data.clone(),
        'dets_L_LT': torch.det(fixed_metric.inverse_metrics).data.clone(),
        'temperature': fixed_metric.temperature,
        'regularization': fixed_metric.regularization
    }
    
    print(f"\n📊 METRIC COMPARISON:")
    print("-" * 25)
    print(f"Parametric L matrices det range: [{param_info['dets_L'].min():.1f}, {param_info['dets_L'].max():.1f}]")
    print(f"Fixed M matrices det range: [{fixed_info['dets_L_LT'].min():.1f}, {fixed_info['dets_L_LT'].max():.1f}]")
    
    # Create comprehensive visualization
    print(f"\n🎨 CREATING COMPARISON VISUALIZATION:")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Parametric L matrices
    ax1 = axes[0, 0]
    L_matrices = param_info['L_matrices']
    dets_L = param_info['dets_L']
    
    # Show a few L matrices
    for i in range(min(3, len(L_matrices))):
        L = L_matrices[i].cpu().numpy()
        im = ax1.imshow(L, cmap='viridis', alpha=0.8)
        ax1.set_title(f'1. Parametric L Matrix {i}\ndet = {dets_L[i]:.1f}', fontweight='bold')
        plt.colorbar(im, ax=ax1)
        break  # Show only first one
    
    # Plot 2: Fixed M matrices
    ax2 = axes[0, 1]
    M_matrices = fixed_info['L_LT_matrices']  # These are the M matrices
    dets_M = fixed_info['dets_L_LT']
    
    # Show a few M matrices
    for i in range(min(3, len(M_matrices))):
        M = M_matrices[i].cpu().numpy()
        im = ax2.imshow(M, cmap='viridis', alpha=0.8)
        ax2.set_title(f'2. Fixed M Matrix {i}\ndet = {dets_M[i]:.1f}', fontweight='bold')
        plt.colorbar(im, ax=ax2)
        break  # Show only first one
    
    # Plot 3: Determinant comparison
    ax3 = axes[0, 2]
    ax3.hist(dets_L.cpu().numpy(), bins=15, alpha=0.7, label='Parametric L', color='blue')
    ax3.hist(dets_M.cpu().numpy(), bins=15, alpha=0.7, label='Fixed M', color='red')
    ax3.set_xlabel('Determinant')
    ax3.set_ylabel('Frequency')
    ax3.set_title('3. Determinant Distribution\nL vs M matrices', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Centroid positions comparison
    ax4 = axes[1, 0]
    param_centroids = param_info['centroids'].cpu().numpy()
    fixed_centroids = fixed_info['centroids'].cpu().numpy()
    
    ax4.scatter(param_centroids[:, 0], param_centroids[:, 1], c='blue', s=100, marker='o', 
               label='Parametric', alpha=0.7)
    ax4.scatter(fixed_centroids[:, 0], fixed_centroids[:, 1], c='red', s=100, marker='s', 
               label='Fixed', alpha=0.7)
    ax4.set_xlabel('z₁ (first dimension)')
    ax4.set_ylabel('z₂ (second dimension)')
    ax4.set_title('4. Centroid Positions\nParametric vs Fixed', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Final metric comparison
    ax5 = axes[1, 1]
    ax5.hist(det_param.cpu().numpy(), bins=20, alpha=0.7, label='Parametric', color='blue')
    ax5.hist(det_fixed.cpu().numpy(), bins=20, alpha=0.7, label='Fixed', color='red')
    ax5.set_xlabel('det(G⁻¹(z))')
    ax5.set_ylabel('Frequency')
    ax5.set_title('5. Final Metric Comparison\ndet(G⁻¹(z)) distribution', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Weight comparison
    ax6 = axes[1, 2]
    param_weights = parametric_metric.compute_weights(z)
    fixed_weights = fixed_metric.compute_weights(z)
    
    ax6.hist(param_weights.detach().cpu().numpy().flatten(), bins=20, alpha=0.7, label='Parametric', color='blue')
    ax6.hist(fixed_weights.detach().cpu().numpy().flatten(), bins=20, alpha=0.7, label='Fixed', color='red')
    ax6.set_xlabel('Weight w(z)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('6. Weight Distribution\nw(z) comparison', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parametric_vs_fixed_metric.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Comparison visualization saved to: parametric_vs_fixed_metric.png")
    
    # Show the key differences
    print(f"\n🔍 KEY DIFFERENCES:")
    print("=" * 25)
    print("1. PARAMETRIC METRIC:")
    print("   - Uses neural networks to generate L matrices")
    print("   - L matrices are learned and can adapt")
    print("   - Formula: G⁻¹(z) = Σᵢ Lψᵢ Lψᵢᵀ exp(-||z - cᵢ||² / T²) + λI")
    print("   - More flexible and trainable")
    
    print("\n2. FIXED METRIC:")
    print("   - Uses pre-computed covariance matrices")
    print("   - M matrices are fixed based on data")
    print("   - Formula: G⁻¹(z) = Σᵢ wᵢ(z) Mᵢ + λI")
    print("   - Simpler but less flexible")
    
    print(f"\n💡 ADVANTAGES OF PARAMETRIC METRIC:")
    print("=" * 35)
    print("✅ Can be trained end-to-end with the VAE")
    print("✅ More flexible and adaptive")
    print("✅ Can learn complex geometric structures")
    print("✅ Better integration with deep learning pipelines")
    print("✅ Can be regularized and optimized")
    
    print(f"\n⚠️  CONSIDERATIONS:")
    print("=" * 20)
    print("⚠️  More parameters to train")
    print("⚠️  Requires careful initialization")
    print("⚠️  May need more training data")
    print("⚠️  Computational overhead")

def train_parametric_metric_example():
    """Show how to train the parametric metric."""
    print(f"\n🎓 TRAINING EXAMPLE:")
    print("=" * 25)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 16
    n_centroids = 5
    
    # Create metric
    metric = ParametricInverseMetricTensor(
        latent_dim=latent_dim,
        n_centroids=n_centroids,
        temperature=1.0,
        regularization=1e-4,
        device=device
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(metric.parameters(), lr=1e-3)
    
    # Training loop example
    print("Training loop structure:")
    print("for epoch in range(n_epochs):")
    print("    for batch in dataloader:")
    print("        z = encoder(batch)")
    print("        G_inv, log_det = metric(z)")
    print("        loss = compute_metric_loss(G_inv, log_det)")
    print("        loss.backward()")
    print("        optimizer.step()")
    
    print(f"\n✅ The parametric metric is ready for training!")

if __name__ == "__main__":
    demonstrate_parametric_metric()
    train_parametric_metric_example() 