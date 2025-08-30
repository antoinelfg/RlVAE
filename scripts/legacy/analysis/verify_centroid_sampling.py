#!/usr/bin/env python3
"""
Verify Centroid Sampling
=========================

Directly verify if RHMC samples are reaching centroids.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.models.riemannian_flow_vae import RiemannianFlowVAE
from dual_rhmc_implementation import DualRiemannianHMCSampler


def test_centroid_sampling():
    """Test if RHMC samples are actually reaching centroids."""
    print("🎯 Verifying Centroid Sampling")
    print("=" * 60)
    
    # Setup model
    model = RiemannianFlowVAE(input_dim=(3, 64, 64), latent_dim=2, n_flows=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load components
    model.load_pretrained_components(
        encoder_path="data/pretrained/encoder_diverse_mlp_ld2_20250730_105419.pt",
        decoder_path="data/pretrained/decoder_diverse_mlp_ld2_20250730_105419.pt",
        metric_path="data/pretrained/metric_diverse_mlp_ld2_20250730_105420.pt"
    )
    
    # Generate test centroids
    np.random.seed(42)
    torch.manual_seed(42)
    latent_data = torch.randn(1000, 2, device=device) * 3.0
    
    # Compute centroids using k-means
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    kmeans.fit(latent_data.detach().cpu().numpy())
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
    
    # Create diverse metric matrices
    metric_matrices = []
    for i in range(len(centroids)):
        # Create diverse eigenvalues
        eigenvals = torch.tensor([100.0 + i*50, 50.0 + i*25], device=device)
        metric_matrix = torch.diag(eigenvals)
        metric_matrices.append(metric_matrix)
    
    metric_matrices = torch.stack(metric_matrices)
    
    # Load metrics into model with HIGHER temperature for smoother transitions
    model.load_pretrained_metrics_from_tensor(centroids, metric_matrices, 
                                            temperature=2.0, regularization=0.001)
    
    print(f"✅ Loaded {len(centroids)} centroids")
    print(f"✅ Centroid positions:")
    for i, centroid in enumerate(centroids):
        print(f"   {i}: [{centroid[0].item():.3f}, {centroid[1].item():.3f}]")
    
    # Run RHMC sampling with smaller step size and more leapfrog steps
    sampler = DualRiemannianHMCSampler(model, mcmc_steps_nbr=200, n_lf=50, eps_lf=0.001)
    samples = sampler.sample(n_samples=1000)
    
    print(f"\n📊 Sample Analysis")
    print(f"✅ Generated {len(samples)} samples")
    print(f"✅ Sample range: [{samples.min().item():.3f}, {samples.max().item():.3f}]")
    
    # Compute distances to all centroids
    min_distances = []
    centroid_hits = []
    
    for threshold in [0.01, 0.05, 0.1, 0.2, 0.5]:
        hits = 0
        for centroid in centroids:
            distances = torch.norm(samples - centroid.unsqueeze(0), dim=1)
            min_dist = torch.min(distances).item()
            min_distances.append(min_dist)
            
            near_samples = torch.sum(distances < threshold).item()
            hits += near_samples
        
        print(f"Samples within {threshold:.2f} of any centroid: {hits}/{len(samples)} ({100*hits/len(samples):.1f}%)")
    
    overall_min_distance = min(min_distances)
    print(f"\n🎯 Overall minimum distance to any centroid: {overall_min_distance:.6f}")
    
    # Test det(G⁻¹) values at samples
    with torch.no_grad():
        G_z = model.G(samples)
        G_inv = torch.linalg.inv(G_z)
        det_G_inv = torch.linalg.det(G_inv)
        
        # Test det(G⁻¹) at centroids
        G_centroids = model.G(centroids)
        G_inv_centroids = torch.linalg.inv(G_centroids)
        det_G_inv_centroids = torch.linalg.det(G_inv_centroids)
    
    print(f"\n📈 det(G⁻¹) Analysis")
    print(f"det(G⁻¹) at samples - Min: {det_G_inv.min().item():.3e}, Max: {det_G_inv.max().item():.3e}")
    print(f"det(G⁻¹) at centroids - Min: {det_G_inv_centroids.min().item():.3e}, Max: {det_G_inv_centroids.max().item():.3e}")
    
    # Count samples in high det(G⁻¹) regions
    high_det_threshold = det_G_inv_centroids.min().item() * 0.5  # 50% of minimum centroid det
    high_det_samples = torch.sum(det_G_inv > high_det_threshold).item()
    
    print(f"Samples in high det(G⁻¹) regions (>{high_det_threshold:.2e}): {high_det_samples}/{len(samples)} ({100*high_det_samples/len(samples):.1f}%)")
    
    # Create detailed visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Centroids and samples
    ax1 = axes[0, 0]
    samples_np = samples.detach().cpu().numpy()
    centroids_np = centroids.detach().cpu().numpy()
    
    ax1.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.3, s=1, c='blue', label='RHMC Samples')
    ax1.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=200, label='Centroids', edgecolors='black')
    
    # Draw circles around centroids to show proximity
    for centroid in centroids_np:
        circle = plt.Circle((centroid[0], centroid[1]), 0.1, fill=False, color='red', linestyle='--', alpha=0.5)
        ax1.add_patch(circle)
    
    ax1.set_title('RHMC Samples vs Centroids')
    ax1.set_xlabel('z₁')
    ax1.set_ylabel('z₂')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Samples colored by det(G⁻¹)
    ax2 = axes[0, 1]
    det_G_inv_np = det_G_inv.detach().cpu().numpy()
    scatter = ax2.scatter(samples_np[:, 0], samples_np[:, 1], c=det_G_inv_np, cmap='viridis', alpha=0.6, s=1)
    ax2.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=200, edgecolors='black')
    plt.colorbar(scatter, ax=ax2, label='det(G⁻¹)')
    ax2.set_title('Samples Colored by det(G⁻¹)')
    ax2.set_xlabel('z₁')
    ax2.set_ylabel('z₂')
    
    # 3. Distance histogram
    ax3 = axes[1, 0]
    all_distances = []
    for centroid in centroids:
        distances = torch.norm(samples - centroid.unsqueeze(0), dim=1)
        min_dist_per_sample = torch.min(distances)
        all_distances.extend(distances.detach().cpu().numpy())
    
    # Find minimum distance to any centroid for each sample
    min_distances_per_sample = []
    for sample in samples:
        distances_to_all_centroids = torch.norm(centroids - sample.unsqueeze(0), dim=1)
        min_dist = torch.min(distances_to_all_centroids).item()
        min_distances_per_sample.append(min_dist)
    
    ax3.hist(min_distances_per_sample, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(0.1, color='red', linestyle='--', label='0.1 threshold')
    ax3.axvline(0.05, color='orange', linestyle='--', label='0.05 threshold')
    ax3.set_title('Distribution of Min Distances to Centroids')
    ax3.set_xlabel('Min Distance to Any Centroid')
    ax3.set_ylabel('Number of Samples')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. det(G⁻¹) histogram
    ax4 = axes[1, 1]
    ax4.hist(det_G_inv_np, bins=50, alpha=0.7, color='green', edgecolor='black', label='Sample det(G⁻¹)')
    
    # Add vertical lines for centroid det(G⁻¹) values
    det_G_inv_centroids_np = det_G_inv_centroids.detach().cpu().numpy()
    for i, det_val in enumerate(det_G_inv_centroids_np):
        ax4.axvline(det_val, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    ax4.set_title('Distribution of det(G⁻¹) Values')
    ax4.set_xlabel('det(G⁻¹)')
    ax4.set_ylabel('Number of Samples')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig("centroid_sampling_verification.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Final assessment
    print(f"\n🏆 ASSESSMENT:")
    very_close_count = sum(1 for d in min_distances_per_sample if d < 0.05)
    close_count = sum(1 for d in min_distances_per_sample if d < 0.1)
    
    print(f"Samples very close to centroids (<0.05): {very_close_count}/{len(samples)} ({100*very_close_count/len(samples):.1f}%)")
    print(f"Samples close to centroids (<0.1): {close_count}/{len(samples)} ({100*close_count/len(samples):.1f}%)")
    
    if very_close_count > len(samples) * 0.05:  # > 5% very close
        print("✅ SUCCESS: RHMC is successfully sampling near centroids!")
    elif close_count > len(samples) * 0.1:  # > 10% close
        print("🟡 PARTIAL SUCCESS: RHMC is sampling reasonably close to centroids")
    else:
        print("❌ ISSUE: RHMC is not effectively reaching centroids")
    
    return min_distances_per_sample, det_G_inv_np


if __name__ == "__main__":
    test_centroid_sampling()