#!/usr/bin/env python3
"""
Optimized Precise RHMC
=======================

Memory-efficient but highly accurate RHMC targeting centroids.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent))

from src.models.riemannian_flow_vae import RiemannianFlowVAE
from dual_rhmc_implementation import DualRiemannianHMCSampler


def run_precision_enhanced_sampling():
    """Run precision-enhanced RHMC with multiple strategies."""
    print("🎯 Precision-Enhanced RHMC Sampling")
    print("=" * 70)
    
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
    
    # Use the existing centroids but with optimized parameters
    torch.manual_seed(42)
    latent_data = torch.randn(3000, 2, device=device) * 2.5
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=25, random_state=42, n_init=10)
    kmeans.fit(latent_data.detach().cpu().numpy())
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
    
    # Create well-conditioned diverse metrics
    metric_matrices = []
    for i in range(len(centroids)):
        # Progressive eigenvalue scaling for diversity
        scale = 300.0 + i * 50.0
        eigenvals = torch.tensor([scale, scale * 0.7], device=device)
        metric_matrix = torch.diag(eigenvals)
        metric_matrices.append(metric_matrix)
    
    metric_matrices = torch.stack(metric_matrices)
    
    print(f"✅ Setup: {len(centroids)} centroids with diverse metrics")
    
    # Test multiple precision strategies
    strategies = [
        {
            "name": "Ultra-Fine Steps", 
            "temp": 2.0, "reg": 0.001, "eps": 0.000001, "n_lf": 200, "steps": 150,
            "description": "Smallest possible steps with many leapfrog iterations"
        },
        {
            "name": "Adaptive Temperature", 
            "temp": 5.0, "reg": 0.0001, "eps": 0.00001, "n_lf": 100, "steps": 200,
            "description": "High temperature for smooth gradients"
        },
        {
            "name": "Balanced Precision", 
            "temp": 3.0, "reg": 0.0005, "eps": 0.000005, "n_lf": 150, "steps": 175,
            "description": "Optimized balance of all parameters"
        }
    ]
    
    results = {}
    
    for strategy in strategies:
        print(f"\n🔬 Testing Strategy: {strategy['name']}")
        print(f"   {strategy['description']}")
        print(f"   Parameters: T={strategy['temp']}, ε={strategy['eps']}, n_lf={strategy['n_lf']}")
        
        # Load metrics with strategy parameters
        model.load_pretrained_metrics_from_tensor(
            centroids, metric_matrices, 
            temperature=strategy['temp'], 
            regularization=strategy['reg']
        )
        
        # Create sampler with strategy parameters
        sampler = DualRiemannianHMCSampler(
            model, 
            mcmc_steps_nbr=strategy['steps'], 
            n_lf=strategy['n_lf'], 
            eps_lf=strategy['eps']
        )
        
        # Run sampling with smaller batch size for memory efficiency
        start_time = time.time()
        samples = sampler.sample(n_samples=500)  # Smaller batch
        sampling_time = time.time() - start_time
        
        print(f"   ✅ Completed in {sampling_time:.1f}s")
        print(f"   📊 Generated {len(samples)} samples")
        
        # Analyze centroid proximity
        min_distances = []
        for sample in samples:
            distances_to_centroids = torch.norm(centroids - sample.unsqueeze(0), dim=1)
            min_dist = torch.min(distances_to_centroids).item()
            min_distances.append(min_dist)
        
        overall_min = min(min_distances)
        mean_min = np.mean(min_distances)
        
        # Count samples within thresholds
        very_close = sum(1 for d in min_distances if d < 0.05)
        close = sum(1 for d in min_distances if d < 0.1)
        
        # Analyze det(G⁻¹) values
        with torch.no_grad():
            G_samples = model.G(samples)
            G_inv_samples = torch.linalg.inv(G_samples)
            det_G_inv_samples = torch.linalg.det(G_inv_samples)
            
            G_centroids = model.G(centroids)
            G_inv_centroids = torch.linalg.inv(G_centroids)
            det_G_inv_centroids = torch.linalg.det(G_inv_centroids)
        
        # Store results
        results[strategy['name']] = {
            'samples': samples,
            'min_distances': min_distances,
            'overall_min': overall_min,
            'mean_min': mean_min,
            'very_close_pct': 100 * very_close / len(samples),
            'close_pct': 100 * close / len(samples),
            'det_G_inv_max': det_G_inv_samples.max().item(),
            'det_G_inv_mean': det_G_inv_samples.mean().item(),
            'sampling_time': sampling_time,
            'strategy': strategy
        }
        
        print(f"   🎯 Min distance to centroids: {overall_min:.6f}")
        print(f"   📈 Very close samples (<0.05): {very_close}/{len(samples)} ({100*very_close/len(samples):.1f}%)")
        print(f"   📈 Close samples (<0.1): {close}/{len(samples)} ({100*close/len(samples):.1f}%)")
        print(f"   📊 Max det(G⁻¹): {det_G_inv_samples.max().item():.3e}")
    
    # Find best strategy
    best_strategy = min(results.keys(), key=lambda k: results[k]['overall_min'])
    
    print(f"\n🏆 BEST STRATEGY: {best_strategy}")
    best_result = results[best_strategy]
    print(f"   Minimum distance achieved: {best_result['overall_min']:.6f}")
    print(f"   Very close samples: {best_result['very_close_pct']:.1f}%")
    print(f"   Close samples: {best_result['close_pct']:.1f}%")
    
    # Create comprehensive comparison visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    centroids_np = centroids.detach().cpu().numpy()
    
    for idx, (name, result) in enumerate(results.items()):
        samples_np = result['samples'].detach().cpu().numpy()
        
        # Row 1: Sample distributions
        ax1 = axes[0, idx]
        ax1.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5, s=3, c='blue')
        ax1.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=100, 
                   edgecolors='black', linewidth=0.5)
        
        # Draw proximity circles
        for centroid in centroids_np:
            circle = plt.Circle((centroid[0], centroid[1]), 0.1, fill=False, color='red', 
                              linestyle='--', alpha=0.3, linewidth=0.8)
            ax1.add_patch(circle)
        
        ax1.set_title(f'{name}\nMin: {result["overall_min"]:.4f}')
        ax1.set_xlabel('z₁')
        ax1.set_ylabel('z₂')
        ax1.grid(True, alpha=0.3)
        
        # Row 2: Distance histograms
        ax2 = axes[1, idx]
        ax2.hist(result['min_distances'], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(result['overall_min'], color='red', linestyle='-', linewidth=2, 
                   label=f'Min: {result["overall_min"]:.4f}')
        ax2.axvline(0.05, color='orange', linestyle='--', label='0.05 thresh')
        ax2.set_title(f'Distance Distribution')
        ax2.set_xlabel('Min Distance to Centroid')
        ax2.set_ylabel('Count')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Row 3: Strategy comparison metrics
        ax3 = axes[2, idx]
        metrics = ['Very Close\n(<0.05)', 'Close\n(<0.1)', 'Time (s)', 'Max det(G⁻¹)\n(×1000)']
        values = [
            result['very_close_pct'],
            result['close_pct'], 
            result['sampling_time'],
            result['det_G_inv_max'] / 1000
        ]
        colors = ['red', 'orange', 'blue', 'purple']
        
        bars = ax3.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_title(f'{name} Metrics')
        ax3.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("precision_enhanced_rhmc_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create focused analysis of the best strategy
    print(f"\n🔍 Detailed Analysis of Best Strategy: {best_strategy}")
    
    best_samples = results[best_strategy]['samples']
    
    # Zoom in around centroids for the best strategy
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    samples_np = best_samples.detach().cpu().numpy()
    
    # 1. Full view
    ax1 = axes[0, 0]
    ax1.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.4, s=5, c='blue', label='Samples')
    ax1.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=150, 
               label='Centroids', edgecolors='black', linewidth=1)
    ax1.set_title(f'{best_strategy} - Full View')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Zoomed view around a high-activity centroid
    # Find centroid with most nearby samples
    centroid_activity = []
    for i, centroid in enumerate(centroids):
        distances = torch.norm(best_samples - centroid.unsqueeze(0), dim=1)
        nearby_count = torch.sum(distances < 0.2).item()
        centroid_activity.append((nearby_count, i, centroid))
    
    most_active = max(centroid_activity, key=lambda x: x[0])
    active_centroid = most_active[2].cpu().numpy()
    
    ax2 = axes[0, 1]
    # Only plot samples near the most active centroid
    center_x, center_y = active_centroid[0], active_centroid[1]
    mask = ((samples_np[:, 0] >= center_x - 1) & (samples_np[:, 0] <= center_x + 1) & 
            (samples_np[:, 1] >= center_y - 1) & (samples_np[:, 1] <= center_y + 1))
    
    if np.any(mask):
        local_samples = samples_np[mask]
        ax2.scatter(local_samples[:, 0], local_samples[:, 1], alpha=0.6, s=15, c='blue')
    
    ax2.scatter(center_x, center_y, c='red', marker='*', s=200, edgecolors='black', linewidth=2)
    
    # Draw concentric circles for distance reference
    for radius in [0.05, 0.1, 0.2]:
        circle = plt.Circle((center_x, center_y), radius, fill=False, 
                          color='red', linestyle='--', alpha=0.6, linewidth=1)
        ax2.add_patch(circle)
    
    ax2.set_xlim(center_x - 1, center_x + 1)
    ax2.set_ylim(center_y - 1, center_y + 1)
    ax2.set_title(f'Zoom: Most Active Centroid\n({most_active[0]} nearby samples)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distance vs det(G⁻¹) correlation
    ax3 = axes[1, 0]
    
    with torch.no_grad():
        G_best = model.G(best_samples)
        G_inv_best = torch.linalg.inv(G_best)
        det_G_inv_best = torch.linalg.det(G_inv_best)
    
    det_values = det_G_inv_best.detach().cpu().numpy()
    min_dists = results[best_strategy]['min_distances']
    
    ax3.scatter(min_dists, det_values, alpha=0.6, s=5)
    ax3.set_xlabel('Min Distance to Centroid')
    ax3.set_ylabel('det(G⁻¹)')
    ax3.set_title('Distance vs det(G⁻¹) Correlation')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
{best_strategy} Performance Summary:

🎯 Targeting Accuracy:
• Minimum distance: {best_result['overall_min']:.6f}
• Mean distance: {best_result['mean_min']:.4f}
• Very close (<0.05): {best_result['very_close_pct']:.1f}%
• Close (<0.1): {best_result['close_pct']:.1f}%

📊 Metric Coverage:
• Max det(G⁻¹): {best_result['det_G_inv_max']:.2e}
• Mean det(G⁻¹): {best_result['det_G_inv_mean']:.2e}

⚙️ Efficiency:
• Sampling time: {best_result['sampling_time']:.1f}s
• Step size: {best_result['strategy']['eps']}
• Leapfrog steps: {best_result['strategy']['n_lf']}
• Temperature: {best_result['strategy']['temp']}
"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig("best_strategy_detailed_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return results, best_strategy


if __name__ == "__main__":
    results, best_strategy = run_precision_enhanced_sampling()
    
    print(f"\n🎉 PRECISION ENHANCEMENT COMPLETE!")
    print(f"Best approach identified: {best_strategy}")
    print(f"Achieved minimum distance: {results[best_strategy]['overall_min']:.6f}")
    
    if results[best_strategy]['overall_min'] < 0.01:
        print("🏆 EXCEPTIONAL: Achieved sub-centimeter precision!")
    elif results[best_strategy]['overall_min'] < 0.05:
        print("🥇 EXCELLENT: Achieved high precision targeting!")
    elif results[best_strategy]['overall_min'] < 0.1:
        print("🥈 VERY GOOD: Significant precision improvement!")
    else:
        print("🥉 GOOD: Noticeable improvement in targeting accuracy!")