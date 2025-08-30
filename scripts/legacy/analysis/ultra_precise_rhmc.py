#!/usr/bin/env python3
"""
Ultra-Precise RHMC Sampling
============================

Push RHMC sampling to maximum accuracy for centroid targeting.
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


class UltraPreciseRHMC(DualRiemannianHMCSampler):
    """Ultra-precise RHMC with adaptive step sizing and enhanced targeting."""
    
    def __init__(self, model, mcmc_steps_nbr=300, n_lf=100, eps_lf=0.00001, 
                 adaptive_eps=True, target_acceptance=0.7):
        super().__init__(model, mcmc_steps_nbr, n_lf, eps_lf)
        self.adaptive_eps = adaptive_eps
        self.target_acceptance = target_acceptance
        self.eps_history = []
        self.acceptance_history = []
        
        print(f"🎯 Ultra-Precise RHMC Initialized")
        print(f"   - Adaptive step size: {adaptive_eps}")
        print(f"   - Target acceptance rate: {target_acceptance}")
        print(f"   - Initial step size: {eps_lf}")
        print(f"   - Leapfrog steps: {n_lf}")
        print(f"   - MCMC steps: {mcmc_steps_nbr}")
    
    def _adaptive_step_size(self, acceptance_rate, current_eps):
        """Adapt step size based on acceptance rate."""
        if not self.adaptive_eps:
            return current_eps
        
        # Target acceptance rate is 0.7 for optimal efficiency
        if acceptance_rate > self.target_acceptance + 0.1:
            # Too high acceptance - increase step size
            new_eps = current_eps * 1.1
        elif acceptance_rate < self.target_acceptance - 0.1:
            # Too low acceptance - decrease step size
            new_eps = current_eps * 0.9
        else:
            # Acceptance rate is good
            new_eps = current_eps
        
        # Clamp step size to reasonable bounds
        new_eps = torch.clamp(torch.tensor(new_eps), 0.000001, 0.01).item()
        
        return new_eps
    
    def _enhanced_momentum_initialization(self, z):
        """Enhanced momentum initialization with better conditioning."""
        with torch.no_grad():
            G_z = self.model.G(z)
            G_inv = torch.linalg.inv(G_z)
            
            # Use eigendecomposition for more stable sampling
            eigenvals, eigenvecs = torch.linalg.eigh(G_inv)
            eigenvals = torch.clamp(eigenvals, min=1e-8)  # Ensure positive definiteness
            
            # Create well-conditioned covariance
            sqrt_eigenvals = torch.sqrt(eigenvals)
            sqrt_cov = eigenvecs @ torch.diag_embed(sqrt_eigenvals) @ eigenvecs.transpose(-2, -1)
            
            # Sample momentum
            p = torch.einsum('bij,bj->bi', sqrt_cov, torch.randn_like(z))
            
        return p
    
    def _multi_scale_leapfrog(self, z, p, eps):
        """Multi-scale leapfrog with finer integration near centroids."""
        batch_size = z.shape[0]
        
        # Check if we're near centroids (high det(G⁻¹))
        with torch.no_grad():
            G_z = self.model.G(z)
            G_inv = torch.linalg.inv(G_z)
            det_G_inv = torch.linalg.det(G_inv)
            
            # Adaptive step size based on local metric properties
            # Use smaller steps in high-curvature regions (near centroids)
            local_eps = eps / (1.0 + 0.1 * torch.log(det_G_inv + 1e-6))
            local_eps = torch.clamp(local_eps, eps * 0.1, eps * 2.0)
        
        # Leapfrog with adaptive local step size
        z_new = z.clone()
        p_new = p.clone()
        
        for step in range(self.n_lf):
            # Half momentum step
            grad_z = self._compute_gradients(z_new)
            p_new = p_new - 0.5 * local_eps.unsqueeze(1) * grad_z
            
            # Full position step
            G_z_new = self.model.G(z_new)
            G_inv_new = torch.linalg.inv(G_z_new)
            z_new = z_new + local_eps.unsqueeze(1) * torch.einsum('bij,bj->bi', G_inv_new, p_new)
            
            # Update local step size for next iteration
            with torch.no_grad():
                G_z_updated = self.model.G(z_new)
                G_inv_updated = torch.linalg.inv(G_z_updated)
                det_G_inv_updated = torch.linalg.det(G_inv_updated)
                local_eps = eps / (1.0 + 0.1 * torch.log(det_G_inv_updated + 1e-6))
                local_eps = torch.clamp(local_eps, eps * 0.1, eps * 2.0)
            
            # Half momentum step
            grad_z_new = self._compute_gradients(z_new)
            p_new = p_new - 0.5 * local_eps.unsqueeze(1) * grad_z_new
        
        return z_new, p_new
    
    def sample_ultra_precise(self, n_samples=1000, warmup_steps=100):
        """Ultra-precise sampling with warmup and adaptation."""
        print(f"🎯 Ultra-Precise RHMC Sampling")
        print("=" * 60)
        
        # Initialize samples
        z = torch.randn(n_samples, 2, device=self.device) * 1.0  # Start closer to origin
        
        all_samples = []
        acceptance_count = 0
        current_eps = self.eps_lf
        
        total_steps = warmup_steps + self.mcmc_steps_nbr
        
        for step in range(total_steps):
            # Enhanced momentum initialization
            p = self._enhanced_momentum_initialization(z)
            
            z_current = z.clone()
            p_current = p.clone()
            
            # Multi-scale leapfrog
            z_prop, p_prop = self._multi_scale_leapfrog(z_current, p_current, current_eps)
            
            # Hamiltonian evaluation
            H_current = self._compute_hamiltonian(z_current, p_current)
            H_prop = self._compute_hamiltonian(z_prop, p_prop)
            
            # Metropolis accept/reject
            alpha = torch.exp(H_current - H_prop)
            alpha = torch.clamp(alpha, 0, 1)  # Ensure valid probabilities
            
            accept = torch.rand(z.shape[0], device=z.device) < alpha
            z = torch.where(accept.unsqueeze(1), z_prop, z)
            
            acceptance_count += accept.sum().item()
            
            # Adaptive step size (only during warmup)
            if step < warmup_steps and self.adaptive_eps and (step + 1) % 20 == 0:
                recent_acceptance = acceptance_count / (n_samples * 20)
                current_eps = self._adaptive_step_size(recent_acceptance, current_eps)
                
                self.eps_history.append(current_eps)
                self.acceptance_history.append(recent_acceptance)
                
                print(f"Step {step+1}: acc_rate={recent_acceptance:.3f}, eps={current_eps:.6f}")
                acceptance_count = 0  # Reset for next window
            
            # Store samples (after warmup)
            if step >= warmup_steps and step % 3 == 0:  # Store every 3rd sample for decorrelation
                all_samples.append(z.clone())
            
            # Progress reporting
            if (step + 1) % 50 == 0:
                current_acceptance = acceptance_count / (n_samples * min(50, step + 1))
                print(f"Step {step+1}/{total_steps}: current_acc={current_acceptance:.3f}")
        
        final_acceptance = acceptance_count / (n_samples * self.mcmc_steps_nbr)
        print(f"✅ Ultra-precise RHMC completed")
        print(f"   Final acceptance rate: {final_acceptance:.3f}")
        print(f"   Final step size: {current_eps:.6f}")
        
        samples = torch.cat(all_samples, dim=0)
        return samples, final_acceptance, current_eps


def test_ultra_precise_sampling():
    """Test ultra-precise RHMC sampling."""
    print("🚀 Testing Ultra-Precise RHMC Sampling")
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
    
    # Generate test data and centroids
    torch.manual_seed(42)
    latent_data = torch.randn(2000, 2, device=device) * 2.5
    
    # Compute centroids
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    kmeans.fit(latent_data.detach().cpu().numpy())
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
    
    # Create diverse metric matrices
    metric_matrices = []
    for i in range(len(centroids)):
        # Create diverse but well-conditioned metrics
        eigenvals = torch.tensor([200.0 + i*100, 100.0 + i*50], device=device)
        metric_matrix = torch.diag(eigenvals)
        metric_matrices.append(metric_matrix)
    
    metric_matrices = torch.stack(metric_matrices)
    
    # Load metrics with optimal temperature for smooth gradients
    model.load_pretrained_metrics_from_tensor(centroids, metric_matrices, 
                                            temperature=1.5, regularization=0.001)
    
    print(f"✅ Setup complete: {len(centroids)} centroids")
    
    # Create ultra-precise sampler
    ultra_sampler = UltraPreciseRHMC(
        model, 
        mcmc_steps_nbr=200,
        n_lf=150,
        eps_lf=0.000005,  # Very fine step size
        adaptive_eps=True,
        target_acceptance=0.65
    )
    
    # Run ultra-precise sampling
    start_time = time.time()
    samples, final_acceptance, final_eps = ultra_sampler.sample_ultra_precise(
        n_samples=2000, 
        warmup_steps=50
    )
    sampling_time = time.time() - start_time
    
    print(f"\n📊 Ultra-Precise Sampling Results:")
    print(f"Sampling time: {sampling_time:.1f}s")
    print(f"Total samples: {len(samples)}")
    print(f"Sample range: [{samples.min().item():.3f}, {samples.max().item():.3f}]")
    
    # Analyze centroid proximity
    print(f"\n🎯 Centroid Proximity Analysis:")
    
    min_distances = []
    centroid_hits = {thresh: 0 for thresh in [0.01, 0.05, 0.1, 0.2]}
    
    for sample in samples:
        distances_to_centroids = torch.norm(centroids - sample.unsqueeze(0), dim=1)
        min_dist = torch.min(distances_to_centroids).item()
        min_distances.append(min_dist)
        
        # Count hits at different thresholds
        for thresh in centroid_hits:
            if min_dist < thresh:
                centroid_hits[thresh] += 1
    
    overall_min = min(min_distances)
    mean_min = np.mean(min_distances)
    
    print(f"Overall minimum distance to centroids: {overall_min:.6f}")
    print(f"Mean minimum distance to centroids: {mean_min:.6f}")
    
    for thresh, hits in centroid_hits.items():
        percentage = 100 * hits / len(samples)
        print(f"Samples within {thresh:.2f} of centroids: {hits}/{len(samples)} ({percentage:.1f}%)")
    
    # Analyze det(G⁻¹) values
    with torch.no_grad():
        G_samples = model.G(samples)
        G_inv_samples = torch.linalg.inv(G_samples)
        det_G_inv_samples = torch.linalg.det(G_inv_samples)
        
        G_centroids = model.G(centroids)
        G_inv_centroids = torch.linalg.inv(G_centroids)
        det_G_inv_centroids = torch.linalg.det(G_inv_centroids)
    
    print(f"\n📈 det(G⁻¹) Analysis:")
    print(f"det(G⁻¹) at samples - Min: {det_G_inv_samples.min().item():.3e}, Max: {det_G_inv_samples.max().item():.3e}")
    print(f"det(G⁻¹) at centroids - Min: {det_G_inv_centroids.min().item():.3e}, Max: {det_G_inv_centroids.max().item():.3e}")
    
    # Count samples in high det(G⁻¹) regions
    high_det_threshold = det_G_inv_centroids.min().item() * 0.1  # 10% of minimum centroid det
    high_det_count = torch.sum(det_G_inv_samples > high_det_threshold).item()
    
    print(f"Samples in high det(G⁻¹) regions (>{high_det_threshold:.2e}): {high_det_count}/{len(samples)} ({100*high_det_count/len(samples):.1f}%)")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Centroids and samples overview
    ax1 = axes[0, 0]
    samples_np = samples.detach().cpu().numpy()
    centroids_np = centroids.detach().cpu().numpy()
    
    ax1.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.4, s=2, c='blue', label='Ultra-Precise Samples')
    ax1.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=200, 
               label='Centroids', edgecolors='black', linewidth=1)
    
    # Draw proximity circles
    for centroid in centroids_np:
        circle = plt.Circle((centroid[0], centroid[1]), 0.1, fill=False, color='red', 
                          linestyle='--', alpha=0.5, linewidth=1)
        ax1.add_patch(circle)
    
    ax1.set_title('Ultra-Precise RHMC: Samples vs Centroids')
    ax1.set_xlabel('z₁')
    ax1.set_ylabel('z₂')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Samples colored by det(G⁻¹)
    ax2 = axes[0, 1]
    det_G_inv_np = det_G_inv_samples.detach().cpu().numpy()
    scatter = ax2.scatter(samples_np[:, 0], samples_np[:, 1], c=det_G_inv_np, 
                         cmap='viridis', alpha=0.6, s=3)
    ax2.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=200, 
               edgecolors='black', linewidth=1)
    plt.colorbar(scatter, ax=ax2, label='det(G⁻¹)')
    ax2.set_title('Samples Colored by det(G⁻¹)')
    ax2.set_xlabel('z₁')
    ax2.set_ylabel('z₂')
    
    # 3. Distance histogram
    ax3 = axes[0, 2]
    ax3.hist(min_distances, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(0.1, color='red', linestyle='--', label='0.1 threshold')
    ax3.axvline(0.05, color='orange', linestyle='--', label='0.05 threshold')
    ax3.axvline(overall_min, color='purple', linestyle='-', linewidth=2, label=f'Min: {overall_min:.4f}')
    ax3.set_title('Distribution of Min Distances to Centroids')
    ax3.set_xlabel('Min Distance to Any Centroid')
    ax3.set_ylabel('Number of Samples')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. det(G⁻¹) histogram
    ax4 = axes[1, 0]
    ax4.hist(det_G_inv_np, bins=50, alpha=0.7, color='purple', edgecolor='black', label='Sample det(G⁻¹)')
    
    # Add vertical lines for centroid det(G⁻¹) values
    det_G_inv_centroids_np = det_G_inv_centroids.detach().cpu().numpy()
    for i, det_val in enumerate(det_G_inv_centroids_np[:5]):  # Show first 5 centroids
        ax4.axvline(det_val, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    ax4.set_title('Distribution of det(G⁻¹) Values')
    ax4.set_xlabel('det(G⁻¹)')
    ax4.set_ylabel('Number of Samples')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # 5. Adaptive step size evolution
    ax5 = axes[1, 1]
    if ultra_sampler.eps_history:
        ax5.plot(ultra_sampler.eps_history, 'b-', label='Step Size')
        ax5.set_title('Adaptive Step Size Evolution')
        ax5.set_xlabel('Adaptation Step')
        ax5.set_ylabel('Step Size (ε)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No Adaptation\n(Fixed Step Size)', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=14)
        ax5.set_title('Step Size Strategy')
    
    # 6. Acceptance rate evolution
    ax6 = axes[1, 2]
    if ultra_sampler.acceptance_history:
        ax6.plot(ultra_sampler.acceptance_history, 'g-', label='Acceptance Rate')
        ax6.axhline(ultra_sampler.target_acceptance, color='red', linestyle='--', 
                   label=f'Target: {ultra_sampler.target_acceptance}')
        ax6.set_title('Acceptance Rate Evolution')
        ax6.set_xlabel('Adaptation Step')
        ax6.set_ylabel('Acceptance Rate')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 1)
    else:
        ax6.text(0.5, 0.5, f'Final Acceptance:\n{final_acceptance:.3f}', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=14)
        ax6.set_title('Final Acceptance Rate')
    
    plt.tight_layout()
    plt.savefig("ultra_precise_rhmc_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Performance assessment
    print(f"\n🏆 ULTRA-PRECISE PERFORMANCE ASSESSMENT:")
    
    very_close_pct = 100 * centroid_hits[0.05] / len(samples)
    close_pct = 100 * centroid_hits[0.1] / len(samples)
    high_det_pct = 100 * high_det_count / len(samples)
    
    print(f"Samples very close to centroids (<0.05): {very_close_pct:.1f}%")
    print(f"Samples close to centroids (<0.1): {close_pct:.1f}%")
    print(f"Samples in high det(G⁻¹) regions: {high_det_pct:.1f}%")
    print(f"Overall minimum distance: {overall_min:.6f}")
    print(f"Final acceptance rate: {final_acceptance:.3f}")
    
    if very_close_pct > 5.0:
        print("🎉 EXCELLENT: Ultra-precise sampling is highly effective!")
    elif close_pct > 10.0:
        print("✅ GOOD: Ultra-precise sampling shows significant improvement")
    elif high_det_pct > 20.0:
        print("🟡 MODERATE: Some improvement in reaching high-metric regions")
    else:
        print("🔄 NEEDS REFINEMENT: Consider further parameter tuning")
    
    return samples, centroids, overall_min, final_acceptance


if __name__ == "__main__":
    test_ultra_precise_sampling()