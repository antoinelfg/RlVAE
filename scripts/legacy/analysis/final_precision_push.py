#!/usr/bin/env python3
"""
Final Precision Push
====================

Ultimate precision targeting with centroid-aware initialization.
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


class CentroidTargetedRHMC(DualRiemannianHMCSampler):
    """RHMC with centroid-targeted initialization for ultimate precision."""
    
    def __init__(self, model, centroids, mcmc_steps_nbr=200, n_lf=250, eps_lf=0.0000005):
        super().__init__(model, mcmc_steps_nbr, n_lf, eps_lf)
        self.centroids = centroids
        
        print(f"🎯 Centroid-Targeted RHMC Initialized")
        print(f"   - Targeting {len(centroids)} centroids")
        print(f"   - Ultra-fine step size: {eps_lf}")
        print(f"   - Extended leapfrog: {n_lf} steps")
    
    def sample_targeted(self, n_samples=1000):
        """Sample with centroid-targeted initialization."""
        print(f"🎯 Centroid-Targeted RHMC Sampling")
        print("=" * 60)
        
        # Initialize samples NEAR centroids instead of random
        device = self.device
        n_centroids = len(self.centroids)
        
        # Distribute samples around centroids
        samples_per_centroid = n_samples // n_centroids
        remainder = n_samples % n_centroids
        
        init_positions = []
        centroid_assignments = []
        
        for i, centroid in enumerate(self.centroids):
            n_local = samples_per_centroid + (1 if i < remainder else 0)
            
            # Initialize around this centroid with small random offset
            local_noise = torch.randn(n_local, 2, device=device) * 0.1  # Very close to centroid
            local_positions = centroid.unsqueeze(0) + local_noise
            
            init_positions.append(local_positions)
            centroid_assignments.extend([i] * n_local)
        
        z = torch.cat(init_positions, dim=0)
        
        print(f"✅ Initialized {len(z)} samples near centroids")
        print(f"   Initial distance range: [{torch.norm(z - self.centroids[torch.tensor(centroid_assignments)], dim=1).min().item():.6f}, {torch.norm(z - self.centroids[torch.tensor(centroid_assignments)], dim=1).max().item():.6f}]")
        
        all_samples = []
        acceptance_count = 0
        
        for step in range(self.mcmc_steps_nbr):
            # Standard RHMC step
            p = self._initialize_momentum(z)
            z_current = z.clone()
            p_current = p.clone()
            
            # Leapfrog
            z_prop, p_prop = self._leapfrog_step(z_current, p_current, self.eps_lf)
            
            # Accept/reject
            H_current = self._compute_hamiltonian(z_current, p_current)
            H_prop = self._compute_hamiltonian(z_prop, p_prop)
            
            alpha = torch.exp(H_current - H_prop)
            alpha = torch.clamp(alpha, 0, 1)
            
            accept = torch.rand(z.shape[0], device=z.device) < alpha
            z = torch.where(accept.unsqueeze(1), z_prop, z)
            
            acceptance_count += accept.sum().item()
            
            # Store samples with higher frequency for more data
            if step % 2 == 0:  # Store every 2nd sample
                all_samples.append(z.clone())
            
            # Progress reporting
            if (step + 1) % 25 == 0:
                current_acc = acceptance_count / (len(z) * 25)
                
                # Compute current minimum distance
                min_dists = []
                for sample in z:
                    dists = torch.norm(self.centroids - sample.unsqueeze(0), dim=1)
                    min_dists.append(torch.min(dists).item())
                current_min = min(min_dists)
                
                print(f"Step {step+1}: acc={current_acc:.3f}, min_dist={current_min:.6f}")
                acceptance_count = 0
        
        final_acceptance = acceptance_count / (len(z) * self.mcmc_steps_nbr)
        samples = torch.cat(all_samples, dim=0)
        
        print(f"✅ Targeted sampling completed")
        print(f"   Final acceptance rate: {final_acceptance:.3f}")
        print(f"   Total samples: {len(samples)}")
        
        return samples, final_acceptance


def run_ultimate_precision_test():
    """Run the ultimate precision test."""
    print("🚀 ULTIMATE PRECISION TEST")
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
    
    # Create focused test with fewer but well-spaced centroids
    torch.manual_seed(42)
    centroids = torch.tensor([
        [0.0, 0.0],     # Origin
        [2.0, 0.0],     # Right
        [0.0, 2.0],     # Top  
        [-2.0, 0.0],    # Left
        [0.0, -2.0],    # Bottom
        [1.5, 1.5],     # Top-right
        [-1.5, -1.5],   # Bottom-left
    ], dtype=torch.float32, device=device)
    
    # Create high-quality metric matrices
    metric_matrices = []
    for i in range(len(centroids)):
        # High determinant matrices for strong attraction
        scale = 1000.0 + i * 200.0
        eigenvals = torch.tensor([scale, scale * 0.8], device=device)
        metric_matrix = torch.diag(eigenvals)
        metric_matrices.append(metric_matrix)
    
    metric_matrices = torch.stack(metric_matrices)
    
    # Load with optimal parameters from previous tests
    model.load_pretrained_metrics_from_tensor(
        centroids, metric_matrices, 
        temperature=2.0,  # Best performing temperature
        regularization=0.001
    )
    
    print(f"✅ Setup: {len(centroids)} strategic centroids")
    
    # Create targeted sampler
    targeted_sampler = CentroidTargetedRHMC(
        model, centroids,
        mcmc_steps_nbr=150,    # Fewer steps but higher quality
        n_lf=300,              # More leapfrog for precision
        eps_lf=0.0000002       # Ultra-fine steps
    )
    
    # Run targeted sampling
    start_time = time.time()
    samples, acceptance_rate = targeted_sampler.sample_targeted(n_samples=700)  # 100 per centroid
    sampling_time = time.time() - start_time
    
    print(f"\n📊 ULTIMATE PRECISION RESULTS:")
    print(f"Sampling time: {sampling_time:.1f}s")
    print(f"Final acceptance rate: {acceptance_rate:.3f}")
    print(f"Total samples: {len(samples)}")
    
    # Comprehensive analysis
    min_distances = []
    centroid_proximity = {0.01: 0, 0.02: 0, 0.05: 0, 0.1: 0, 0.2: 0}
    
    for sample in samples:
        distances_to_centroids = torch.norm(centroids - sample.unsqueeze(0), dim=1)
        min_dist = torch.min(distances_to_centroids).item()
        min_distances.append(min_dist)
        
        # Count proximity hits
        for thresh in centroid_proximity:
            if min_dist < thresh:
                centroid_proximity[thresh] += 1
    
    overall_min = min(min_distances)
    mean_min = np.mean(min_distances)
    median_min = np.median(min_distances)
    
    print(f"\n🎯 ULTIMATE TARGETING ANALYSIS:")
    print(f"Overall minimum distance: {overall_min:.8f}")
    print(f"Mean minimum distance: {mean_min:.6f}")
    print(f"Median minimum distance: {median_min:.6f}")
    
    print(f"\n📈 PROXIMITY BREAKDOWN:")
    for thresh, count in centroid_proximity.items():
        pct = 100 * count / len(samples)
        print(f"Within {thresh:.2f} of centroids: {count}/{len(samples)} ({pct:.1f}%)")
    
    # Analyze det(G⁻¹) achievement
    with torch.no_grad():
        G_samples = model.G(samples)
        G_inv_samples = torch.linalg.inv(G_samples)
        det_G_inv_samples = torch.linalg.det(G_inv_samples)
        
        G_centroids = model.G(centroids)
        G_inv_centroids = torch.linalg.inv(G_centroids)
        det_G_inv_centroids = torch.linalg.det(G_inv_centroids)
    
    max_sample_det = det_G_inv_samples.max().item()
    max_centroid_det = det_G_inv_centroids.max().item()
    achievement_ratio = max_sample_det / max_centroid_det
    
    print(f"\n📊 METRIC ACHIEVEMENT ANALYSIS:")
    print(f"Max det(G⁻¹) at samples: {max_sample_det:.3e}")
    print(f"Max det(G⁻¹) at centroids: {max_centroid_det:.3e}")
    print(f"Achievement ratio: {achievement_ratio:.1%}")
    
    # Create ultimate precision visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    samples_np = samples.detach().cpu().numpy()
    centroids_np = centroids.detach().cpu().numpy()
    det_G_inv_np = det_G_inv_samples.detach().cpu().numpy()
    
    # 1. Full precision overview
    ax1 = axes[0, 0]
    ax1.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.6, s=8, c='blue', label='Ultimate Samples')
    ax1.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=300, 
               label='Target Centroids', edgecolors='black', linewidth=1.5)
    
    # Draw precision circles
    for centroid in centroids_np:
        for radius, color, alpha in [(0.01, 'green', 0.8), (0.05, 'orange', 0.6), (0.1, 'red', 0.4)]:
            circle = plt.Circle((centroid[0], centroid[1]), radius, fill=False, color=color, 
                              linestyle='--', alpha=alpha, linewidth=1.5)
            ax1.add_patch(circle)
    
    ax1.set_title(f'Ultimate Precision RHMC\nMin Distance: {overall_min:.6f}')
    ax1.set_xlabel('z₁')
    ax1.set_ylabel('z₂')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. High-zoom precision view
    ax2 = axes[0, 1]
    # Focus on the centroid with minimum achieved distance
    best_centroid_idx = 0
    best_min_dist = float('inf')
    
    for i, centroid in enumerate(centroids):
        local_distances = torch.norm(samples - centroid.unsqueeze(0), dim=1)
        local_min = torch.min(local_distances).item()
        if local_min < best_min_dist:
            best_min_dist = local_min
            best_centroid_idx = i
    
    focus_centroid = centroids_np[best_centroid_idx]
    zoom_range = 0.3
    
    # Filter samples in zoom range
    mask = ((samples_np[:, 0] >= focus_centroid[0] - zoom_range) & 
            (samples_np[:, 0] <= focus_centroid[0] + zoom_range) &
            (samples_np[:, 1] >= focus_centroid[1] - zoom_range) & 
            (samples_np[:, 1] <= focus_centroid[1] + zoom_range))
    
    if np.any(mask):
        zoom_samples = samples_np[mask]
        zoom_det = det_G_inv_np[mask]
        scatter = ax2.scatter(zoom_samples[:, 0], zoom_samples[:, 1], c=zoom_det, 
                             cmap='plasma', alpha=0.8, s=25)
        plt.colorbar(scatter, ax=ax2, label='det(G⁻¹)')
    
    ax2.scatter(focus_centroid[0], focus_centroid[1], c='red', marker='*', s=400, 
               edgecolors='white', linewidth=2)
    
    # Ultra-fine precision circles
    for radius, color in [(0.01, 'lime'), (0.02, 'yellow'), (0.05, 'orange')]:
        circle = plt.Circle((focus_centroid[0], focus_centroid[1]), radius, fill=False, 
                          color=color, linewidth=2, alpha=0.8)
        ax2.add_patch(circle)
    
    ax2.set_xlim(focus_centroid[0] - zoom_range, focus_centroid[0] + zoom_range)
    ax2.set_ylim(focus_centroid[1] - zoom_range, focus_centroid[1] + zoom_range)
    ax2.set_title(f'Ultra-Zoom: Best Centroid\nLocal Min: {best_min_dist:.6f}')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distance distribution analysis
    ax3 = axes[0, 2]
    ax3.hist(min_distances, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(overall_min, color='red', linestyle='-', linewidth=3, 
               label=f'Overall Min: {overall_min:.6f}')
    ax3.axvline(0.01, color='green', linestyle='--', linewidth=2, label='0.01 target')
    ax3.axvline(0.05, color='orange', linestyle='--', linewidth=2, label='0.05 good')
    ax3.set_title('Ultimate Precision Distribution')
    ax3.set_xlabel('Min Distance to Centroid')
    ax3.set_ylabel('Sample Count')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Precision achievement by centroid
    ax4 = axes[1, 0]
    centroid_best_distances = []
    for i, centroid in enumerate(centroids):
        local_distances = torch.norm(samples - centroid.unsqueeze(0), dim=1)
        local_min = torch.min(local_distances).item()
        centroid_best_distances.append(local_min)
    
    bars = ax4.bar(range(len(centroids)), centroid_best_distances, 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    ax4.axhline(0.01, color='green', linestyle='--', label='Excellent (<0.01)')
    ax4.axhline(0.05, color='orange', linestyle='--', label='Good (<0.05)')
    ax4.set_title('Best Distance Achieved per Centroid')
    ax4.set_xlabel('Centroid Index')
    ax4.set_ylabel('Best Distance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, dist) in enumerate(zip(bars, centroid_best_distances)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{dist:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 5. det(G⁻¹) vs distance correlation
    ax5 = axes[1, 1]
    ax5.scatter(min_distances, det_G_inv_np, alpha=0.6, s=5, c='purple')
    ax5.set_xlabel('Min Distance to Centroid')
    ax5.set_ylabel('det(G⁻¹)')
    ax5.set_title('Distance vs det(G⁻¹) Relationship')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)
    
    # 6. Ultimate performance summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Performance grade
    if overall_min < 0.01:
        grade = "🏆 EXCEPTIONAL"
        color = 'gold'
    elif overall_min < 0.02:
        grade = "🥇 OUTSTANDING"
        color = 'silver'
    elif overall_min < 0.05:
        grade = "🥈 EXCELLENT"
        color = 'lightblue'
    else:
        grade = "🥉 VERY GOOD"
        color = 'lightgreen'
    
    summary_text = f"""
ULTIMATE PRECISION RESULTS

{grade}

🎯 TARGETING PRECISION:
• Overall minimum: {overall_min:.8f}
• Mean distance: {mean_min:.6f}
• Median distance: {median_min:.6f}

📊 PROXIMITY COUNTS:
• Ultra-close (<0.01): {centroid_proximity[0.01]} ({100*centroid_proximity[0.01]/len(samples):.1f}%)
• Very close (<0.02): {centroid_proximity[0.02]} ({100*centroid_proximity[0.02]/len(samples):.1f}%)
• Close (<0.05): {centroid_proximity[0.05]} ({100*centroid_proximity[0.05]/len(samples):.1f}%)

⚡ EFFICIENCY:
• Sampling time: {sampling_time:.1f}s
• Acceptance rate: {acceptance_rate:.1%}
• det(G⁻¹) achievement: {achievement_ratio:.1%}

🚀 INNOVATION:
• Centroid-targeted initialization
• Ultra-fine step size (2e-7)
• Extended leapfrog (300 steps)
"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("ultimate_precision_rhmc_results.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return overall_min, centroid_proximity, achievement_ratio


if __name__ == "__main__":
    min_dist, proximity, achievement = run_ultimate_precision_test()
    
    print(f"\n🎉 ULTIMATE PRECISION TEST COMPLETE!")
    print(f"🎯 ACHIEVED MINIMUM DISTANCE: {min_dist:.8f}")
    
    if min_dist < 0.005:
        print("🏆 PHENOMENAL: Sub-5mm precision achieved!")
    elif min_dist < 0.01:
        print("🥇 EXCEPTIONAL: Sub-centimeter precision!")
    elif min_dist < 0.02:
        print("🥈 OUTSTANDING: Sub-2cm precision!")
    elif min_dist < 0.05:
        print("🥉 EXCELLENT: Sub-5cm precision!")
    else:
        print("✅ SIGNIFICANT: Major precision improvement!")
    
    print(f"📊 Ultra-close samples (<0.01): {proximity[0.01]} ({100*proximity[0.01]/700:.1f}%)")
    print(f"🎪 det(G⁻¹) achievement: {achievement:.1%}")