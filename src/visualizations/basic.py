"""
Basic Visualizations Module
===========================

Essential visualizations for RiemannianFlowVAE training:
- Cyclicity analysis
- Sequence trajectories 
- Reconstruction quality analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from .base import BaseVisualization


class BasicVisualizations(BaseVisualization):
    """Basic visualization suite for essential analysis."""
    
    def create_cyclicity_analysis(self, x_sample: torch.Tensor, epoch: int):
        """Analyze how well the model preserves/learns cyclicity."""
        print(f"🔄 Creating cyclicity analysis for epoch {epoch}")
        
        self.model.eval()
        with torch.no_grad():
            result = self.model_forward(x_sample)
            
            # Handle both ModelOutput and dict outputs
            if hasattr(result, 'recon_x'):
                recon_x = result.recon_x  # [batch_size, n_obs, 3, 64, 64]
                z_seq = result.z         # [batch_size, n_obs, latent_dim]
            else:
                # Handle dictionary output from modular model
                recon_x = result['reconstruction']  # [batch_size, n_obs, 3, 64, 64]
                z_seq = result['latent_samples']   # [batch_size, n_obs, latent_dim]
            
            batch_size, n_obs = x_sample.shape[:2]
            
            # Analyze original cyclicity
            orig_first_last_mse = []
            for i in range(batch_size):
                orig_mse = torch.mean((x_sample[i, 0] - x_sample[i, -1]) ** 2).item()
                orig_first_last_mse.append(orig_mse)
            
            # Analyze reconstruction cyclicity
            recon_first_last_mse = []
            for i in range(batch_size):
                recon_mse = torch.mean((recon_x[i, 0] - recon_x[i, -1]) ** 2).item()
                recon_first_last_mse.append(recon_mse)
            
            # Analyze latent cyclicity
            latent_first_last_mse = []
            latent_norms = []
            for i in range(batch_size):
                latent_mse = torch.mean((z_seq[i, 0] - z_seq[i, -1]) ** 2).item()
                latent_first_last_mse.append(latent_mse)
                latent_norms.append(torch.norm(z_seq[i], dim=-1).cpu().numpy())
            
            # Create visualization - SMALLER SIZE
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # SMALLER
            fig.suptitle(f'Cyclicity Analysis - Epoch {epoch}', fontsize=14)
            
            # Plot 1: Original vs Reconstructed MSE
            axes[0, 0].scatter(orig_first_last_mse, recon_first_last_mse, alpha=0.6)
            axes[0, 0].plot([0, max(orig_first_last_mse)], [0, max(orig_first_last_mse)], 'r--', alpha=0.5)
            axes[0, 0].set_xlabel('Original First-Last MSE')
            axes[0, 0].set_ylabel('Reconstructed First-Last MSE')
            axes[0, 0].set_title('Original vs Reconstructed Cyclicity')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Latent space cyclicity
            axes[0, 1].hist(latent_first_last_mse, bins=20, alpha=0.7)
            axes[0, 1].set_xlabel('Latent First-Last MSE')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title(f'Latent Cyclicity (mean: {np.mean(latent_first_last_mse):.2e})')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Example sequence - original
            seq_idx = 0
            for t in range(min(4, n_obs)):
                if t < 2:
                    row, col = 0, 2
                    axes[row, col].clear()
                    if t == 0:
                        axes[row, col].imshow(x_sample[seq_idx, 0].permute(1, 2, 0).cpu().numpy())
                        axes[row, col].set_title(f'Original: First (t=0)')
                    else:
                        axes[row, col].imshow(x_sample[seq_idx, -1].permute(1, 2, 0).cpu().numpy())
                        axes[row, col].set_title(f'Original: Last (t={n_obs-1})')
                    axes[row, col].axis('off')
            
            # Plot 4: Example sequence - reconstructed
            for t in range(min(2, n_obs)):
                row, col = 1, 0 + t
                if t == 0:
                    axes[row, col].imshow(recon_x[seq_idx, 0].permute(1, 2, 0).cpu().numpy())
                    axes[row, col].set_title(f'Recon: First (t=0)')
                else:
                    axes[row, col].imshow(recon_x[seq_idx, -1].permute(1, 2, 0).cpu().numpy())
                    axes[row, col].set_title(f'Recon: Last (t={n_obs-1})')
                axes[row, col].axis('off')
            
            # Plot 5: Latent trajectory
            axes[1, 2].clear()
            # Plot latent trajectory for first sequence
            z_traj = z_seq[seq_idx].cpu().numpy()  # [n_obs, latent_dim]
            
            # Use PCA for visualization
            if z_traj.shape[1] > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                z_pca = pca.fit_transform(z_traj)
                axes[1, 2].plot(z_pca[:, 0], z_pca[:, 1], 'o-', alpha=0.8, linewidth=2, markersize=8)
                axes[1, 2].scatter(z_pca[0, 0], z_pca[0, 1], color='green', s=100, marker='s', label='Start', zorder=5)
                axes[1, 2].scatter(z_pca[-1, 0], z_pca[-1, 1], color='red', s=100, marker='*', label='End', zorder=5)
                axes[1, 2].set_title(f'Latent Trajectory (PCA)\nFirst-Last Distance: {np.linalg.norm(z_pca[0] - z_pca[-1]):.3f}')
                axes[1, 2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                axes[1, 2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            else:
                axes[1, 2].plot(z_traj[:, 0], z_traj[:, 1], 'o-', alpha=0.8, linewidth=2, markersize=8)
                axes[1, 2].scatter(z_traj[0, 0], z_traj[0, 1], color='green', s=100, marker='s', label='Start')
                axes[1, 2].scatter(z_traj[-1, 0], z_traj[-1, 1], color='red', s=100, marker='*', label='End')
                axes[1, 2].set_title('Latent Trajectory')
                axes[1, 2].set_xlabel('Latent Dim 0')
                axes[1, 2].set_ylabel('Latent Dim 1')
                
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save and log
            filename = f'cyclicity_analysis_epoch_{epoch}.png'
            saved_file = self._safe_save_plt_figure(filename, dpi=300, bbox_inches='tight')
            
            # Log to WandB
            if self.should_log_to_wandb() and saved_file:
                wandb.log({
                    "basic/cyclicity_analysis": wandb.Image(saved_file, caption=f"Epoch {epoch}"),
                    "metrics/cyclicity_mse_mean": np.mean(orig_first_last_mse),
                    "metrics/latent_cyclicity_mean": np.mean(latent_first_last_mse),
                    "metrics/reconstruction_cyclicity_mean": np.mean(recon_first_last_mse),
                })
            
            plt.close()
        
        self.model.train()
    
    def create_sequence_trajectories(self, x_sample: torch.Tensor, epoch: int):
        """Create comprehensive visualization of sequence trajectories in latent space."""
        print(f"🧠 Creating sequence trajectory visualization for epoch {epoch}")
        
        self.model.eval()
        with torch.no_grad():
            result = self.model_forward(x_sample)
            
            # Handle both ModelOutput and dict outputs
            if hasattr(result, 'z'):
                z_seq = result.z  # [batch_size, n_obs, latent_dim]
            else:
                # Handle dictionary output from modular model
                z_seq = result['latent_samples']   # [batch_size, n_obs, latent_dim]
            
            batch_size, n_obs, latent_dim = z_seq.shape
            max_viz = self._get_viz_count()
            num_viz = min(max_viz, batch_size)
            
            # Create visualization - SMALLER SIZE
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # SMALLER
            fig.suptitle(f'Latent Sequence Trajectories - Epoch {epoch}', fontsize=14)
            
            # Prepare data for PCA
            z_flat = z_seq.reshape(-1, latent_dim).cpu().numpy()
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            z_pca = pca.fit_transform(z_flat)
            z_pca_seq = z_pca.reshape(batch_size, n_obs, 3)
            
            # Plot 1: 2D Trajectory Overview (PC1 vs PC2)
            ax = axes[0, 0]
            colors = plt.get_cmap("tab10")(np.linspace(0, 1, num_viz))
            for i in range(num_viz):
                traj = z_pca_seq[i, :, :2]
                ax.plot(traj[:, 0], traj[:, 1], 'o-', color=colors[i], alpha=0.7, label=f'Seq {i}', linewidth=2)
                ax.scatter(traj[0, 0], traj[0, 1], color=colors[i], s=100, marker='s', alpha=0.9, edgecolor='black')  # Start
                ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[i], s=100, marker='*', alpha=0.9, edgecolor='black')  # End
            
            ax.set_title(f'2D Trajectories (PC1 vs PC2)\nVariance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Temporal Evolution of PC1
            ax = axes[0, 1]
            for i in range(num_viz):
                ax.plot(range(n_obs), z_pca_seq[i, :, 0], 'o-', color=colors[i], alpha=0.7, label=f'Seq {i}')
            ax.set_title('PC1 Evolution Over Time')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('PC1 Value')
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Temporal Evolution of PC2
            ax = axes[0, 2]
            for i in range(num_viz):
                ax.plot(range(n_obs), z_pca_seq[i, :, 1], 'o-', color=colors[i], alpha=0.7, label=f'Seq {i}')
            ax.set_title('PC2 Evolution Over Time')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('PC2 Value')
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Trajectory Lengths
            ax = axes[1, 0]
            traj_lengths = []
            for i in range(batch_size):
                # Calculate total path length in PCA space
                diffs = np.diff(z_pca_seq[i], axis=0)
                lengths = np.linalg.norm(diffs, axis=1)
                total_length = np.sum(lengths)
                traj_lengths.append(total_length)
            
            ax.hist(traj_lengths, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title(f'Trajectory Lengths Distribution\nMean: {np.mean(traj_lengths):.3f}±{np.std(traj_lengths):.3f}')
            ax.set_xlabel('Total Path Length')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
            
            # Plot 5: Start vs End Point Analysis
            ax = axes[1, 1]
            start_points = z_pca_seq[:, 0, :2]  # [batch_size, 2]
            end_points = z_pca_seq[:, -1, :2]   # [batch_size, 2]
            distances = np.linalg.norm(end_points - start_points, axis=1)
            
            ax.scatter(start_points[:, 0], start_points[:, 1], alpha=0.6, s=50, label='Start Points', marker='s')
            ax.scatter(end_points[:, 0], end_points[:, 1], alpha=0.6, s=50, label='End Points', marker='*')
            
            # Draw lines connecting start to end
            for i in range(min(num_viz, batch_size)):
                ax.plot([start_points[i, 0], end_points[i, 0]], 
                       [start_points[i, 1], end_points[i, 1]], 
                       'k--', alpha=0.3, linewidth=1)
            
            ax.set_title(f'Start vs End Points\nMean Distance: {np.mean(distances):.3f}±{np.std(distances):.3f}')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 6: Distance Statistics
            ax = axes[1, 2]
            ax.hist(distances, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title('Start-End Distance Distribution')
            ax.set_xlabel('Start-End Distance')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save and log
            filename = f'sequence_trajectories_epoch_{epoch}.png'
            saved_file = self._safe_save_plt_figure(filename, dpi=300, bbox_inches='tight')
            
            # Log to WandB
            if self.should_log_to_wandb() and saved_file:
                wandb.log({
                    "basic/sequence_trajectories": wandb.Image(saved_file, caption=f"Epoch {epoch}"),
                })
            
            plt.close()
        
        self.model.train()
    
    def create_reconstruction_analysis(self, x_sample: torch.Tensor, epoch: int):
        """Create comprehensive reconstruction visualization."""
        print(f"🎬 Creating comprehensive reconstruction visualization for epoch {epoch}")
        
        self.model.eval()
        with torch.no_grad():
            result = self.model_forward(x_sample)
            
            # Handle both ModelOutput and dict outputs
            if hasattr(result, 'recon_x'):
                recon_x = result.recon_x  # [batch_size, n_obs, 3, 64, 64]
            else:
                # Handle dictionary output from modular model
                recon_x = result['reconstruction']  # [batch_size, n_obs, 3, 64, 64]
            
            batch_size, n_obs, channels, height, width = recon_x.shape
            max_viz = self._get_viz_count()
            
            # Create visualization - SMALLER SIZE
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # SMALLER
            fig.suptitle(f'Reconstruction Analysis - Epoch {epoch}', fontsize=14)
            
            # Plot original vs reconstructed for first sequence
            for i in range(min(3, n_obs)):
                # Original
                axes[0, i].imshow(x_sample[0, i].permute(1, 2, 0).cpu().numpy())
                axes[0, i].set_title(f'Original t={i}')
                axes[0, i].axis('off')
                
                # Reconstructed
                axes[1, i].imshow(recon_x[0, i].permute(1, 2, 0).cpu().numpy())
                axes[1, i].set_title(f'Reconstructed t={i}')
                axes[1, i].axis('off')
            
            # Save and log
            filename = f'reconstruction_analysis_epoch_{epoch}.png'
            saved_file = self._safe_save_plt_figure(filename, dpi=300, bbox_inches='tight')
            
            # Log to WandB
            if self.should_log_to_wandb() and saved_file:
                wandb.log({
                    "basic/reconstruction_analysis": wandb.Image(saved_file, caption=f"Epoch {epoch}"),
                })
        
        self.model.train() 