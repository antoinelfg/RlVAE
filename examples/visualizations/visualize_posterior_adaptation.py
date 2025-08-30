#!/usr/bin/env python3
"""
Visualization script to observe posterior adaptation to the metric.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

# Add original_rlvae to path
sys.path.insert(0, 'original_rlvae')

def create_posterior_adaptation_visualization():
    """Create visualization showing posterior adaptation to metric."""
    print("🎨 Creating Posterior Adaptation Visualization")
    print("=" * 60)
    
    try:
        from src.models.riemannian_flow_vae import RiemannianFlowVAE
        
        # Create model with enhanced KL mechanism
        model = RiemannianFlowVAE(
            input_dim=[64, 64, 3],
            latent_dim=16,
            adaptive_kl_enabled=True,
            adaptive_kl_ramp_up_steps=5,
            adaptive_kl_alignment_weight=0.1,
            update_metric_during_training=True,
            metric_update_frequency=3
        )
        
        print("✅ Model created with enhanced KL mechanism")
        
        # Create figure for visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Posterior Adaptation to Metric During Training', fontsize=16)
        
        # Storage for tracking metrics
        beta_values = []
        kl_losses = []
        alignment_penalties = []
        adaptation_steps = []
        
        # Simulate training with metric updates
        batch_size = 32
        latent_dim = 16
        
        print("🔄 Simulating training with metric updates...")
        
        for step in range(15):
            # Generate dummy data
            mu = torch.randn(batch_size, latent_dim)
            log_var = torch.randn(batch_size, latent_dim)
            z_sample = torch.randn(batch_size, latent_dim)
            
            # Create metric matrices (simulating metric evolution)
            base_metric = torch.eye(latent_dim)
            # Add some evolution to the metric
            evolution_factor = step / 15.0
            evolved_metric = base_metric + evolution_factor * torch.randn(latent_dim, latent_dim) * 0.1
            evolved_metric = evolved_metric @ evolved_metric.T  # Make positive definite
            
            G_z = evolved_metric.unsqueeze(0).repeat(batch_size, 1, 1)
            
            # Mock metric function
            def mock_G(z):
                return evolved_metric.unsqueeze(0).repeat(z.shape[0], 1, 1)
            
            model.G = mock_G
            
            # Compute KL loss
            kl_loss = model.compute_riemannian_kl_loss(mu, log_var, z_sample)
            
            # Compute alignment penalty
            alignment_penalty = model._compute_metric_alignment_penalty(mu, log_var, G_z)
            
            # Store metrics
            beta_values.append(model.riemannian_beta)
            kl_losses.append(kl_loss.item())
            alignment_penalties.append(alignment_penalty.item())
            adaptation_steps.append(step)
            
            # Trigger metric update every few steps
            if step % 3 == 0 and step > 0:
                print(f"   Step {step}: Triggering metric update...")
                model._update_metric_during_training(mu, torch.randn(batch_size, 3, 64, 64))
                print(f"      Beta: {model.riemannian_beta:.4f}, KL: {kl_loss.item():.4f}, Alignment: {alignment_penalty.item():.4f}")
            
            # Update plots
            if step % 3 == 0:
                # Clear previous plots
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.clear()
                
                # Plot 1: Beta evolution
                ax1.plot(adaptation_steps, beta_values, 'b-', linewidth=2, marker='o')
                ax1.set_title('Adaptive Beta Evolution')
                ax1.set_xlabel('Training Step')
                ax1.set_ylabel('Riemannian Beta')
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: KL loss evolution
                ax2.plot(adaptation_steps, kl_losses, 'r-', linewidth=2, marker='s')
                ax2.set_title('KL Loss Evolution')
                ax2.set_xlabel('Training Step')
                ax2.set_ylabel('KL Loss')
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Alignment penalty evolution
                ax3.plot(adaptation_steps, alignment_penalties, 'g-', linewidth=2, marker='^')
                ax3.set_title('Metric Alignment Penalty')
                ax3.set_xlabel('Training Step')
                ax3.set_ylabel('Alignment Penalty')
                ax3.grid(True, alpha=0.3)
                
                # Plot 4: Posterior vs Metric visualization
                # Create 2D projection of posterior samples and metric structure
                if step > 0:
                    # Sample from posterior
                    posterior_samples = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
                    
                    # Project to 2D using PCA-like approach
                    samples_2d = posterior_samples[:, :2].detach().numpy()
                    
                    # Create metric visualization
                    x = np.linspace(-3, 3, 50)
                    y = np.linspace(-3, 3, 50)
                    X, Y = np.meshgrid(x, y)
                    
                    # Compute metric determinant at each point
                    Z = np.zeros_like(X)
                    for i in range(X.shape[0]):
                        for j in range(X.shape[1]):
                            point = torch.tensor([[X[i,j], Y[i,j]] + [0]*(latent_dim-2)], dtype=torch.float32)
                            metric_at_point = mock_G(point)[0, :2, :2]  # 2x2 submatrix
                            Z[i,j] = torch.det(metric_at_point).item()
                    
                    # Plot posterior samples and metric structure
                    contour = ax4.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
                    ax4.scatter(samples_2d[:, 0], samples_2d[:, 1], c='red', s=30, alpha=0.8, label='Posterior Samples')
                    ax4.set_title(f'Posterior vs Metric (Step {step})')
                    ax4.set_xlabel('Latent Dim 1')
                    ax4.set_ylabel('Latent Dim 2')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.pause(0.5)
        
        print("\n✅ Visualization completed!")
        print("\n📊 **Key Observations:**")
        print("1. Beta ramping: Starts low, increases with metric updates")
        print("2. KL loss evolution: Should show adaptation to metric")
        print("3. Alignment penalty: Measures posterior-metric compatibility")
        print("4. Posterior vs Metric: Visual alignment over time")
        
        # Save final visualization
        plt.savefig('posterior_adaptation_visualization.png', dpi=300, bbox_inches='tight')
        print("💾 Visualization saved as 'posterior_adaptation_visualization.png'")
        
        # Create summary plot
        fig_summary, ((ax1_sum, ax2_sum), (ax3_sum, ax4_sum)) = plt.subplots(2, 2, figsize=(15, 12))
        fig_summary.suptitle('Posterior Adaptation Summary', fontsize=16)
        
        # Summary plots
        ax1_sum.plot(adaptation_steps, beta_values, 'b-', linewidth=2, marker='o')
        ax1_sum.set_title('Adaptive Beta Evolution')
        ax1_sum.set_xlabel('Training Step')
        ax1_sum.set_ylabel('Riemannian Beta')
        ax1_sum.grid(True, alpha=0.3)
        
        ax2_sum.plot(adaptation_steps, kl_losses, 'r-', linewidth=2, marker='s')
        ax2_sum.set_title('KL Loss Evolution')
        ax2_sum.set_xlabel('Training Step')
        ax2_sum.set_ylabel('KL Loss')
        ax2_sum.grid(True, alpha=0.3)
        
        ax3_sum.plot(adaptation_steps, alignment_penalties, 'g-', linewidth=2, marker='^')
        ax3_sum.set_title('Metric Alignment Penalty')
        ax3_sum.set_xlabel('Training Step')
        ax3_sum.set_ylabel('Alignment Penalty')
        ax3_sum.grid(True, alpha=0.3)
        
        # Combined metric
        normalized_kl = np.array(kl_losses) / max(kl_losses) if max(kl_losses) > 0 else np.array(kl_losses)
        normalized_alignment = np.array(alignment_penalties) / max(alignment_penalties) if max(alignment_penalties) > 0 else np.array(alignment_penalties)
        combined_metric = normalized_kl + normalized_alignment
        
        ax4_sum.plot(adaptation_steps, combined_metric, 'purple', linewidth=2, marker='d')
        ax4_sum.set_title('Combined Adaptation Metric')
        ax4_sum.set_xlabel('Training Step')
        ax4_sum.set_ylabel('Combined Score')
        ax4_sum.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('posterior_adaptation_summary.png', dpi=300, bbox_inches='tight')
        print("💾 Summary saved as 'posterior_adaptation_summary.png'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_posterior_adaptation_visualization()
    sys.exit(0 if success else 1)
