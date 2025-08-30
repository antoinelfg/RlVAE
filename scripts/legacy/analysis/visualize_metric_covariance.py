#!/usr/bin/env python3
"""
Visualize Metric Covariance Matrices
===================================

This script visualizes the covariance matrices (Mⱼ) that are used to construct the metric G⁻¹.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.components.native_inverse_metric import NativeInverseMetricTensor

def visualize_metric_covariance():
    """Visualize the covariance matrices used in the metric construction."""
    print("🔍 VISUALIZING METRIC COVARIANCE MATRICES")
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
    
    # Analyze the inverse metric matrices
    print(f"\n🔧 ANALYZING INVERSE METRIC MATRICES:")
    print("-" * 40)
    
    dets = []
    traces = []
    eigenvalues = []
    
    for i in range(len(inverse_metrics)):
        M = inverse_metrics[i].cpu().numpy()
        det_M = np.linalg.det(M)
        trace_M = np.trace(M)
        eigenvals = np.linalg.eigvals(M)
        
        dets.append(det_M)
        traces.append(trace_M)
        eigenvalues.append(eigenvals)
        
        print(f"   Centroid {i}: det(M) = {det_M:.1f}, trace(M) = {trace_M:.1f}")
    
    print(f"\n📊 STATISTICS:")
    print(f"   det(M) range: [{min(dets):.1f}, {max(dets):.1f}]")
    print(f"   trace(M) range: [{min(traces):.1f}, {max(traces):.1f}]")
    
    # Create comprehensive visualization
    print(f"\n🎨 CREATING COVARIANCE VISUALIZATION:")
    print("-" * 40)
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    # Plot 1: Overview of all inverse metric matrices
    ax1 = axes[0]
    # Show a few representative matrices
    for i in range(min(4, len(inverse_metrics))):
        M = inverse_metrics[i].cpu().numpy()
        im = ax1.imshow(M, cmap='viridis', alpha=0.8)
        ax1.set_title(f'1. Inverse Metric Matrix M_{i}\ndet = {dets[i]:.1f}', fontweight='bold')
        plt.colorbar(im, ax=ax1)
        break  # Show only first one for overview
    
    # Plot 2: Determinant distribution
    ax2 = axes[1]
    ax2.hist(dets, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax2.set_xlabel('det(Mⱼ)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('2. Distribution of det(Mⱼ)\nAcross all centroids', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Trace distribution
    ax3 = axes[2]
    ax3.hist(traces, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax3.set_xlabel('trace(Mⱼ)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('3. Distribution of trace(Mⱼ)\nAcross all centroids', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Eigenvalue analysis
    ax4 = axes[3]
    all_eigenvals = np.concatenate(eigenvalues)
    ax4.hist(all_eigenvals.real, bins=30, alpha=0.7, color='red', edgecolor='black')
    ax4.set_xlabel('Eigenvalues of Mⱼ')
    ax4.set_ylabel('Frequency')
    ax4.set_title('4. Eigenvalue Distribution\nAll Mⱼ matrices', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Centroid positions with det(M) coloring
    ax5 = axes[4]
    centroid_positions = centroids.cpu().numpy()
    scatter = ax5.scatter(centroid_positions[:, 0], centroid_positions[:, 1], 
                         c=dets, s=150, cmap='viridis', marker='*',
                         edgecolors='white', linewidth=2)
    plt.colorbar(scatter, ax=ax5, label='det(Mⱼ)')
    ax5.set_title('5. Centroid Positions\nColored by det(Mⱼ)', fontweight='bold')
    ax5.set_xlabel('z₁ (first dimension)')
    ax5.set_ylabel('z₂ (second dimension)')
    
    # Plot 6: Centroid positions with trace(M) coloring
    ax6 = axes[5]
    scatter = ax6.scatter(centroid_positions[:, 0], centroid_positions[:, 1], 
                         c=traces, s=150, cmap='plasma', marker='*',
                         edgecolors='white', linewidth=2)
    plt.colorbar(scatter, ax=ax6, label='trace(Mⱼ)')
    ax6.set_title('6. Centroid Positions\nColored by trace(Mⱼ)', fontweight='bold')
    ax6.set_xlabel('z₁ (first dimension)')
    ax6.set_ylabel('z₂ (second dimension)')
    
    # Plot 7: Correlation between det and trace
    ax7 = axes[6]
    ax7.scatter(traces, dets, alpha=0.7, s=50)
    ax7.set_xlabel('trace(Mⱼ)')
    ax7.set_ylabel('det(Mⱼ)')
    ax7.set_title('7. det(Mⱼ) vs trace(Mⱼ)\nCorrelation Analysis', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Show a few individual matrices
    ax8 = axes[7]
    # Show the matrix with highest determinant
    max_det_idx = np.argmax(dets)
    M_max = inverse_metrics[max_det_idx].cpu().numpy()
    im = ax8.imshow(M_max, cmap='viridis', alpha=0.8)
    ax8.set_title(f'8. Highest det(Mⱼ) Matrix\nCentroid {max_det_idx}, det = {dets[max_det_idx]:.1f}', fontweight='bold')
    plt.colorbar(im, ax=ax8)
    
    # Plot 9: Show the matrix with lowest determinant
    ax9 = axes[8]
    min_det_idx = np.argmin(dets)
    M_min = inverse_metrics[min_det_idx].cpu().numpy()
    im = ax9.imshow(M_min, cmap='viridis', alpha=0.8)
    ax9.set_title(f'9. Lowest det(Mⱼ) Matrix\nCentroid {min_det_idx}, det = {dets[min_det_idx]:.1f}', fontweight='bold')
    plt.colorbar(im, ax=ax9)
    
    # Plot 10: Eigenvalue spectrum for a few matrices
    ax10 = axes[9]
    for i in range(min(5, len(eigenvalues))):
        eigenvals = eigenvalues[i]
        ax10.scatter(range(len(eigenvals)), eigenvals.real, alpha=0.7, s=30, label=f'Centroid {i}')
    ax10.set_xlabel('Eigenvalue Index')
    ax10.set_ylabel('Eigenvalue Value')
    ax10.set_title('10. Eigenvalue Spectra\nFirst 5 centroids', fontweight='bold')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # Plot 11: Condition number analysis
    ax11 = axes[10]
    condition_numbers = [max(eigenvals.real) / min(eigenvals.real) for eigenvals in eigenvalues]
    ax11.hist(condition_numbers, bins=15, alpha=0.7, color='orange', edgecolor='black')
    ax11.set_xlabel('Condition Number')
    ax11.set_ylabel('Frequency')
    ax11.set_title('11. Condition Number Distribution\nλ_max/λ_min', fontweight='bold')
    ax11.grid(True, alpha=0.3)
    
    # Plot 12: Summary statistics
    ax12 = axes[11]
    stats = {
        'Mean det(Mⱼ)': np.mean(dets),
        'Std det(Mⱼ)': np.std(dets),
        'Mean trace(Mⱼ)': np.mean(traces),
        'Std trace(Mⱼ)': np.std(traces),
        'Mean condition': np.mean(condition_numbers),
        'Max condition': np.max(condition_numbers)
    }
    
    y_pos = np.arange(len(stats))
    ax12.barh(y_pos, [1]*len(stats), color='lightblue', alpha=0.7)
    ax12.set_yticks(y_pos)
    ax12.set_yticklabels(list(stats.keys()))
    ax12.set_xlim(0, 1.2)
    ax12.set_title('12. Summary Statistics', fontweight='bold')
    
    # Add value labels
    for i, (key, value) in enumerate(stats.items()):
        ax12.text(1.05, i, f'{value:.2f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualize_metric_covariance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Covariance visualization saved to: visualize_metric_covariance.png")
    
    # Additional analysis
    print(f"\n📊 DETAILED ANALYSIS:")
    print("=" * 25)
    print(f"Number of centroids: {len(centroids)}")
    print(f"Matrix dimension: {inverse_metrics.shape[1]}x{inverse_metrics.shape[2]}")
    print(f"det(Mⱼ) statistics:")
    print(f"   Mean: {np.mean(dets):.2f}")
    print(f"   Std: {np.std(dets):.2f}")
    print(f"   Min: {np.min(dets):.2f}")
    print(f"   Max: {np.max(dets):.2f}")
    print(f"trace(Mⱼ) statistics:")
    print(f"   Mean: {np.mean(traces):.2f}")
    print(f"   Std: {np.std(traces):.2f}")
    print(f"   Min: {np.min(traces):.2f}")
    print(f"   Max: {np.max(traces):.2f}")
    print(f"Condition number statistics:")
    print(f"   Mean: {np.mean(condition_numbers):.2f}")
    print(f"   Max: {np.max(condition_numbers):.2f}")
    
    # Check for potential issues
    print(f"\n🔍 POTENTIAL ISSUES:")
    print("=" * 25)
    
    if np.min(dets) < 0:
        print("⚠️  WARNING: Some matrices have negative determinants!")
    else:
        print("✅ All matrices have positive determinants")
    
    if np.max(condition_numbers) > 1000:
        print("⚠️  WARNING: Some matrices are very ill-conditioned!")
    else:
        print("✅ All matrices are reasonably well-conditioned")
    
    if np.std(dets) / np.mean(dets) > 0.5:
        print("⚠️  WARNING: High variance in determinants across centroids!")
    else:
        print("✅ Determinants are reasonably consistent across centroids")
    
    print(f"\n💡 INSIGHTS:")
    print("=" * 15)
    print("1. The Mⱼ matrices are the 'building blocks' of your metric")
    print("2. Each centroid has its own local metric structure")
    print("3. The final G⁻¹(z) is a weighted combination of these matrices")
    print("4. High condition numbers suggest anisotropic local geometry")
    print("5. Consistent determinants suggest stable metric construction")

if __name__ == "__main__":
    visualize_metric_covariance() 