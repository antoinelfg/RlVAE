#!/usr/bin/env python3
"""
Advanced visualization showing posterior distribution adapting to metric structure.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns

# Add original_rlvae to path
sys.path.insert(0, 'original_rlvae')

def create_advanced_posterior_visualization():
    """Create advanced visualization of posterior adaptation."""
    print("🎨 Creating Advanced Posterior Adaptation Visualization")
    print("=" * 65)
    
    try:
        from src.models.riemannian_flow_vae import RiemannianFlowVAE
        
        # Create model
        model = RiemannianFlowVAE(
            input_dim=[64, 64, 3],
            latent_dim=16,
            adaptive_kl_enabled=True,
            adaptive_kl_ramp_up_steps=5,
            adaptive_kl_alignment_weight=0.1
        )
        
        print("✅ Model created")
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Advanced Posterior Adaptation to Metric Structure', fontsize=16)
        
        # Storage
        all_metrics = []
        all_posteriors = []
        all_betas = []
        
        print("🔄 Simulating posterior adaptation...")
        
        for step in range(10):
            # Create evolving metric
            base_metric = torch.eye(16)
            evolution = step / 10.0
            
            # Create anisotropic metric (different scales in different directions)
            anisotropic_factor = 1.0 + evolution * 2.0
            metric = base_metric.clone()
            metric[0, 0] = anisotropic_factor  # Stretch in first dimension
            metric[1, 1] = 1.0 / anisotropic_factor  # Compress in second dimension
            
            # Add some rotation
            angle = evolution * np.pi / 4
            rotation = torch.tensor([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            
            # Apply rotation to first 2x2 block
            metric[:2, :2] = rotation @ metric[:2, :2] @ rotation.T
            
            # Ensure positive definiteness
            metric = metric @ metric.T
            
            # Create posterior parameters that adapt to metric
            batch_size = 50
            
            # Initial posterior (not adapted)
            if step == 0:
                mu = torch.randn(batch_size, 16) * 2.0
                log_var = torch.ones(batch_size, 16) * 0.5
            else:
                # Adapt posterior to metric structure
                # Sample from metric-aware distribution
                metric_sqrt = torch.linalg.cholesky(metric)
                mu = torch.randn(batch_size, 16) @ metric_sqrt.T
                
                # Adapt variance to metric eigenvalues
                eigenvals, _ = torch.linalg.eigh(metric)
                log_var = torch.log(eigenvals.unsqueeze(0).repeat(batch_size, 1))
            
            # Mock metric function
            def mock_G(z):
                return metric.unsqueeze(0).repeat(z.shape[0], 1, 1)
            
            model.G = mock_G
            
            # Compute metrics
            z_sample = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
            kl_loss = model.compute_riemannian_kl_loss(mu, log_var, z_sample)
            alignment_penalty = model._compute_metric_alignment_penalty(mu, log_var, metric.unsqueeze(0).repeat(batch_size, 1, 1))
            
            # Store data
            all_metrics.append(metric.detach().numpy())
            all_posteriors.append({
                'mu': mu.detach().numpy(),
                'log_var': log_var.detach().numpy(),
                'kl_loss': kl_loss.item(),
                'alignment': alignment_penalty.item()
            })
            all_betas.append(model.riemannian_beta)
            
            # Update plots every few steps
            if step % 2 == 0:
                print(f"   Step {step}: KL={kl_loss.item():.4f}, Alignment={alignment_penalty.item():.4f}")
                
                # Clear all axes
                for ax in axes.flat:
                    ax.clear()
                
                # Plot 1: Metric structure (2D slice)
                metric_2d = metric[:2, :2].detach().numpy()
                eigenvals_2d, eigenvecs_2d = np.linalg.eigh(metric_2d)
                
                # Create confidence ellipse
                theta = np.linspace(0, 2*np.pi, 100)
                ellipse_points = np.array([np.cos(theta), np.sin(theta)])
                scaled_points = eigenvecs_2d @ (ellipse_points * np.sqrt(eigenvals_2d))
                
                axes[0, 0].plot(scaled_points[0], scaled_points[1], 'b-', linewidth=2, label='Metric Structure')
                axes[0, 0].scatter([0], [0], c='red', s=100, marker='*', label='Origin')
                axes[0, 0].set_title(f'Metric Structure (Step {step})')
                axes[0, 0].set_xlabel('Latent Dim 1')
                axes[0, 0].set_ylabel('Latent Dim 2')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].set_aspect('equal')
                
                # Plot 2: Posterior samples vs metric
                posterior_samples = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
                samples_2d = posterior_samples[:, :2].detach().numpy()
                
                # Create metric contour
                x = np.linspace(-4, 4, 50)
                y = np.linspace(-4, 4, 50)
                X, Y = np.meshgrid(x, y)
                Z = np.zeros_like(X)
                
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        point = torch.tensor([[X[i,j], Y[i,j]] + [0]*14], dtype=torch.float32)
                        metric_at_point = mock_G(point)[0, :2, :2]
                        Z[i,j] = torch.det(metric_at_point).item()
                
                contour = axes[0, 1].contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
                axes[0, 1].scatter(samples_2d[:, 0], samples_2d[:, 1], c='red', s=30, alpha=0.8, label='Posterior Samples')
                axes[0, 1].set_title(f'Posterior vs Metric (Step {step})')
                axes[0, 1].set_xlabel('Latent Dim 1')
                axes[0, 1].set_ylabel('Latent Dim 2')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # Plot 3: KL loss evolution
                kl_values = [p['kl_loss'] for p in all_posteriors]
                steps = list(range(len(kl_values)))
                axes[0, 2].plot(steps, kl_values, 'r-', linewidth=2, marker='o')
                axes[0, 2].set_title('KL Loss Evolution')
                axes[0, 2].set_xlabel('Step')
                axes[0, 2].set_ylabel('KL Loss')
                axes[0, 2].grid(True, alpha=0.3)
                
                # Plot 4: Alignment penalty evolution
                alignment_values = [p['alignment'] for p in all_posteriors]
                axes[1, 0].plot(steps, alignment_values, 'g-', linewidth=2, marker='s')
                axes[1, 0].set_title('Alignment Penalty Evolution')
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('Alignment Penalty')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Plot 5: Beta evolution
                axes[1, 1].plot(steps, all_betas, 'b-', linewidth=2, marker='^')
                axes[1, 1].set_title('Adaptive Beta Evolution')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Riemannian Beta')
                axes[1, 1].grid(True, alpha=0.3)
                
                # Plot 6: Combined adaptation score
                normalized_kl = np.array(kl_values) / max(kl_values) if max(kl_values) > 0 else np.array(kl_values)
                normalized_alignment = np.array(alignment_values) / max(alignment_values) if max(alignment_values) > 0 else np.array(alignment_values)
                combined_score = normalized_kl + normalized_alignment
                
                axes[1, 2].plot(steps, combined_score, 'purple', linewidth=2, marker='d')
                axes[1, 2].set_title('Combined Adaptation Score')
                axes[1, 2].set_xlabel('Step')
                axes[1, 2].set_ylabel('Combined Score')
                axes[1, 2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.pause(1.0)
        
        print("\n✅ Advanced visualization completed!")
        
        # Save final visualization
        plt.savefig('advanced_posterior_adaptation.png', dpi=300, bbox_inches='tight')
        print("💾 Advanced visualization saved as 'advanced_posterior_adaptation.png'")
        
        # Create animation frames
        print("🎬 Creating animation frames...")
        for step in range(10):
            fig_anim, ax_anim = plt.subplots(1, 1, figsize=(10, 8))
            
            # Get data for this step
            metric = torch.tensor(all_metrics[step])
            posterior = all_posteriors[step]
            
            # Create metric visualization
            metric_2d = metric[:2, :2].numpy()
            eigenvals_2d, eigenvecs_2d = np.linalg.eigh(metric_2d)
            
            # Create confidence ellipse
            theta = np.linspace(0, 2*np.pi, 100)
            ellipse_points = np.array([np.cos(theta), np.sin(theta)])
            scaled_points = eigenvecs_2d @ (ellipse_points * np.sqrt(eigenvals_2d))
            
            # Plot metric structure
            ax_anim.plot(scaled_points[0], scaled_points[1], 'b-', linewidth=3, label='Metric Structure')
            
            # Plot posterior samples
            mu = torch.tensor(posterior['mu'])
            log_var = torch.tensor(posterior['log_var'])
            posterior_samples = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
            samples_2d = posterior_samples[:, :2].numpy()
            
            ax_anim.scatter(samples_2d[:, 0], samples_2d[:, 1], c='red', s=50, alpha=0.7, label='Posterior Samples')
            ax_anim.scatter([0], [0], c='green', s=200, marker='*', label='Origin')
            
            ax_anim.set_title(f'Posterior Adaptation to Metric (Step {step})\nKL: {posterior["kl_loss"]:.3f}, Alignment: {posterior["alignment"]:.3f}')
            ax_anim.set_xlabel('Latent Dim 1')
            ax_anim.set_ylabel('Latent Dim 2')
            ax_anim.legend()
            ax_anim.grid(True, alpha=0.3)
            ax_anim.set_aspect('equal')
            ax_anim.set_xlim(-4, 4)
            ax_anim.set_ylim(-4, 4)
            
            plt.tight_layout()
            plt.savefig(f'animation_frame_{step:02d}.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print("💾 Animation frames saved as 'animation_frame_XX.png'")
        
        print("\n📊 **Visualization Summary:**")
        print("1. Metric Structure: Shows how the Riemannian metric evolves")
        print("2. Posterior Samples: Shows how posterior adapts to metric")
        print("3. KL Loss: Measures divergence between posterior and prior")
        print("4. Alignment Penalty: Measures posterior-metric compatibility")
        print("5. Adaptive Beta: Shows how KL weight adapts over time")
        print("6. Combined Score: Overall adaptation quality")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in advanced visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_advanced_posterior_visualization()
    sys.exit(0 if success else 1)
