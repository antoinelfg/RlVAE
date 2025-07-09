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
        """Enhanced cyclicity analysis with velocity and energy metrics."""

        
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
            
            # Enhanced cyclicity metrics
            orig_first_last_mse = []
            recon_first_last_mse = []
            latent_first_last_mse = []
            latent_norms = []
            
            # New metrics: velocity and energy analysis
            latent_velocities = []
            latent_accelerations = []
            temporal_energies = []
            trajectory_curvatures = []
            
            for i in range(batch_size):
                # Original cyclicity metrics
                orig_mse = torch.mean((x_sample[i, 0] - x_sample[i, -1]) ** 2).item()
                orig_first_last_mse.append(orig_mse)
                
                recon_mse = torch.mean((recon_x[i, 0] - recon_x[i, -1]) ** 2).item()
                recon_first_last_mse.append(recon_mse)
                
                latent_mse = torch.mean((z_seq[i, 0] - z_seq[i, -1]) ** 2).item()
                latent_first_last_mse.append(latent_mse)
                latent_norms.append(torch.norm(z_seq[i], dim=-1).cpu().numpy())
                
                # Enhanced dynamics analysis
                z_traj = z_seq[i].cpu().numpy()  # [n_obs, latent_dim]
                
                # Compute velocities (finite differences)
                if n_obs > 1:
                    velocities = np.diff(z_traj, axis=0)  # [n_obs-1, latent_dim]
                    velocity_magnitudes = np.linalg.norm(velocities, axis=1)
                    latent_velocities.append(velocity_magnitudes)
                    
                    # Compute accelerations (second differences)
                    if n_obs > 2:
                        accelerations = np.diff(velocities, axis=0)  # [n_obs-2, latent_dim]
                        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
                        latent_accelerations.append(acceleration_magnitudes)
                    
                    # Compute temporal energy (kinetic energy analog)
                    temporal_energy = 0.5 * np.sum(velocity_magnitudes ** 2)
                    temporal_energies.append(temporal_energy)
                    
                    # Compute trajectory curvature using consecutive velocity vectors
                    curvatures = []
                    for j in range(len(velocities) - 1):
                        v1, v2 = velocities[j], velocities[j + 1]
                        if np.linalg.norm(v1) > 1e-8 and np.linalg.norm(v2) > 1e-8:
                            # Curvature as change in direction
                            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            cos_angle = np.clip(cos_angle, -1, 1)
                            angle_change = np.arccos(cos_angle)
                            curvatures.append(angle_change)
                    
                    if curvatures:
                        trajectory_curvatures.append(np.mean(curvatures))
                    else:
                        trajectory_curvatures.append(0.0)
            
            # Create enhanced visualization
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))  # Expanded to 3x4 grid
            fig.suptitle(f'Enhanced Cyclicity & Dynamics Analysis - Epoch {epoch}', fontsize=16)
            
            # Plot 1: Original vs Reconstructed MSE
            axes[0, 0].scatter(orig_first_last_mse, recon_first_last_mse, alpha=0.6, s=50)
            axes[0, 0].plot([0, max(orig_first_last_mse)], [0, max(orig_first_last_mse)], 'r--', alpha=0.5, label='Perfect reconstruction')
            
            # Add correlation coefficient
            corr_coef = np.corrcoef(orig_first_last_mse, recon_first_last_mse)[0, 1]
            axes[0, 0].text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=axes[0, 0].transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            axes[0, 0].set_xlabel('Original First-Last MSE')
            axes[0, 0].set_ylabel('Reconstructed First-Last MSE')
            axes[0, 0].set_title('Original vs Reconstructed Cyclicity')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Enhanced latent space cyclicity distribution
            axes[0, 1].hist(latent_first_last_mse, bins=20, alpha=0.7, density=True, label='Empirical')
            axes[0, 1].axvline(np.mean(latent_first_last_mse), color='red', linestyle='--', label=f'Mean: {np.mean(latent_first_last_mse):.2e}')
            axes[0, 1].axvline(np.median(latent_first_last_mse), color='orange', linestyle='--', label=f'Median: {np.median(latent_first_last_mse):.2e}')
            axes[0, 1].set_xlabel('Latent First-Last MSE')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].set_title('Latent Cyclicity Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Temporal energy analysis
            axes[0, 2].hist(temporal_energies, bins=15, alpha=0.7, color='green', density=True)
            axes[0, 2].axvline(np.mean(temporal_energies), color='red', linestyle='--', label=f'Mean: {np.mean(temporal_energies):.2e}')
            axes[0, 2].set_xlabel('Temporal Energy')
            axes[0, 2].set_ylabel('Density')
            axes[0, 2].set_title('Trajectory Energy Distribution')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # Plot 4: Trajectory curvature analysis
            axes[0, 3].hist(trajectory_curvatures, bins=15, alpha=0.7, color='purple', density=True)
            axes[0, 3].axvline(np.mean(trajectory_curvatures), color='red', linestyle='--', label=f'Mean: {np.mean(trajectory_curvatures):.3f}')
            axes[0, 3].set_xlabel('Mean Trajectory Curvature')
            axes[0, 3].set_ylabel('Density')
            axes[0, 3].set_title('Trajectory Curvature Distribution')
            axes[0, 3].legend()
            axes[0, 3].grid(True, alpha=0.3)
            
            # Row 2: Velocity and acceleration analysis
            # Plot 5: Velocity magnitude evolution
            axes[1, 0].clear()
            if latent_velocities:
                for i, vel_seq in enumerate(latent_velocities[:min(8, len(latent_velocities))]):
                    axes[1, 0].plot(vel_seq, alpha=0.7, linewidth=1.5, label=f'Seq {i}' if len(latent_velocities) <= 8 else None)
                axes[1, 0].set_xlabel('Time Step')
                axes[1, 0].set_ylabel('Velocity Magnitude')
                axes[1, 0].set_title('Latent Velocity Evolution')
                if len(latent_velocities) <= 8:
                    axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 6: Acceleration analysis
            axes[1, 1].clear()
            if latent_accelerations:
                all_accelerations = np.concatenate(latent_accelerations)
                axes[1, 1].hist(all_accelerations, bins=20, alpha=0.7, color='orange', density=True)
                axes[1, 1].axvline(np.mean(all_accelerations), color='red', linestyle='--', 
                                  label=f'Mean: {np.mean(all_accelerations):.2e}')
                axes[1, 1].set_xlabel('Acceleration Magnitude')
                axes[1, 1].set_ylabel('Density')
                axes[1, 1].set_title('Latent Acceleration Distribution')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            # Plot 7: Energy vs cyclicity correlation
            axes[1, 2].scatter(temporal_energies, latent_first_last_mse, alpha=0.6, s=50, color='green')
            if len(temporal_energies) > 1:
                energy_cyclicity_corr = np.corrcoef(temporal_energies, latent_first_last_mse)[0, 1]
                axes[1, 2].text(0.05, 0.95, f'r = {energy_cyclicity_corr:.3f}', 
                               transform=axes[1, 2].transAxes, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            axes[1, 2].set_xlabel('Temporal Energy')
            axes[1, 2].set_ylabel('Cyclicity Error')
            axes[1, 2].set_title('Energy vs Cyclicity Correlation')
            axes[1, 2].grid(True, alpha=0.3)
            
            # Plot 8: Velocity distribution across all sequences
            axes[1, 3].clear()
            if latent_velocities:
                all_velocities = np.concatenate(latent_velocities)
                axes[1, 3].hist(all_velocities, bins=25, alpha=0.7, color='blue', density=True)
                axes[1, 3].axvline(np.mean(all_velocities), color='red', linestyle='--', 
                                  label=f'Mean: {np.mean(all_velocities):.2e}')
                axes[1, 3].axvline(np.percentile(all_velocities, 95), color='orange', linestyle=':', 
                                  label=f'95th %: {np.percentile(all_velocities, 95):.2e}')
                axes[1, 3].set_xlabel('Velocity Magnitude')
                axes[1, 3].set_ylabel('Density')
                axes[1, 3].set_title('Overall Velocity Distribution')
                axes[1, 3].legend()
                axes[1, 3].grid(True, alpha=0.3)
            
            # Row 3: Example sequences and enhanced trajectory analysis
            seq_idx = 0
            
            # Plot 9: Original sequence comparison
            axes[2, 0].clear()
            axes[2, 0].imshow(x_sample[seq_idx, 0].permute(1, 2, 0).cpu().numpy())
            axes[2, 0].set_title(f'Original: First (t=0)')
            axes[2, 0].axis('off')
            
            # Plot 10: Original sequence comparison
            axes[2, 1].clear()
            axes[2, 1].imshow(x_sample[seq_idx, -1].permute(1, 2, 0).cpu().numpy())
            axes[2, 1].set_title(f'Original: Last (t={n_obs-1})')
            axes[2, 1].axis('off')
            
            # Plot 11: Enhanced latent trajectory with velocity vectors
            axes[2, 2].clear()
            z_traj = z_seq[seq_idx].cpu().numpy()  # [n_obs, latent_dim]
            
            # Use PCA for visualization with enhanced features
            if z_traj.shape[1] > 2:
                from sklearn.decomposition import PCA
                
                # Check for NaN values
                if np.any(np.isnan(z_traj)):
                    print(f"⚠️ NaN values detected in latent trajectory! Replacing with zeros.")
                    z_traj_clean = np.nan_to_num(z_traj, nan=0.0)
                else:
                    z_traj_clean = z_traj
                
                pca = PCA(n_components=2)
                z_pca = pca.fit_transform(z_traj_clean)
                
                # Plot trajectory
                axes[2, 2].plot(z_pca[:, 0], z_pca[:, 1], 'o-', alpha=0.8, linewidth=2, markersize=6)
                axes[2, 2].scatter(z_pca[0, 0], z_pca[0, 1], color='green', s=120, marker='s', 
                                  label='Start', zorder=5, edgecolor='black')
                axes[2, 2].scatter(z_pca[-1, 0], z_pca[-1, 1], color='red', s=120, marker='*', 
                                  label='End', zorder=5, edgecolor='black')
                
                # Add velocity vectors
                if len(z_pca) > 1:
                    velocities_pca = np.diff(z_pca, axis=0)
                    for i in range(0, len(velocities_pca), max(1, len(velocities_pca)//5)):  # Show every few vectors
                        start = z_pca[i]
                        vel = velocities_pca[i] * 0.5  # Scale for visibility
                        axes[2, 2].arrow(start[0], start[1], vel[0], vel[1], 
                                        head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.6)
                
                first_last_dist = np.linalg.norm(z_pca[0] - z_pca[-1])
                path_length = np.sum(np.linalg.norm(np.diff(z_pca, axis=0), axis=1))
                
                axes[2, 2].set_title(f'Enhanced Latent Trajectory (PCA)\nDist: {first_last_dist:.3f}, Path: {path_length:.3f}')
                axes[2, 2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                axes[2, 2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            else:
                axes[2, 2].plot(z_traj[:, 0], z_traj[:, 1], 'o-', alpha=0.8, linewidth=2, markersize=6)
                axes[2, 2].scatter(z_traj[0, 0], z_traj[0, 1], color='green', s=120, marker='s', label='Start')
                axes[2, 2].scatter(z_traj[-1, 0], z_traj[-1, 1], color='red', s=120, marker='*', label='End')
                axes[2, 2].set_title('Enhanced Latent Trajectory')
                axes[2, 2].set_xlabel('Latent Dim 0')
                axes[2, 2].set_ylabel('Latent Dim 1')
                
            axes[2, 2].legend()
            axes[2, 2].grid(True, alpha=0.3)
            
            # Plot 12: Statistical summary
            axes[2, 3].clear()
            axes[2, 3].axis('off')
            
            # Create comprehensive statistics text
            stats_text = f"""Enhanced Cyclicity Statistics:
            
Original MSE: {np.mean(orig_first_last_mse):.2e} ± {np.std(orig_first_last_mse):.2e}
Recon MSE: {np.mean(recon_first_last_mse):.2e} ± {np.std(recon_first_last_mse):.2e}
Latent MSE: {np.mean(latent_first_last_mse):.2e} ± {np.std(latent_first_last_mse):.2e}

Dynamics Metrics:
Avg Velocity: {np.mean([np.mean(v) for v in latent_velocities]):.2e}
Avg Energy: {np.mean(temporal_energies):.2e}
Avg Curvature: {np.mean(trajectory_curvatures):.3f}

Correlations:
Orig-Recon: {corr_coef:.3f}
Energy-Cyclicity: {energy_cyclicity_corr if len(temporal_energies) > 1 else 'N/A'}"""
            
            axes[2, 3].text(0.05, 0.95, stats_text, transform=axes[2, 3].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            axes[2, 3].set_title('Statistical Summary')
            
            plt.tight_layout()
            
            # Save and log with enhanced metrics
            filename = f'enhanced_cyclicity_analysis_epoch_{epoch}.png'
            saved_file = self._safe_save_plt_figure(filename, dpi=200, bbox_inches='tight')
            
            # Log to WandB with comprehensive metrics
            if self.should_log_to_wandb() and saved_file:
                wandb.log({
                    "basic/enhanced_cyclicity_analysis": wandb.Image(saved_file, caption=f"Enhanced Cyclicity Analysis - Epoch {epoch}"),
                    "metrics/cyclicity_mse_mean": np.mean(orig_first_last_mse),
                    "metrics/latent_cyclicity_mean": np.mean(latent_first_last_mse),
                    "metrics/reconstruction_cyclicity_mean": np.mean(recon_first_last_mse),
                    "metrics/temporal_energy_mean": np.mean(temporal_energies),
                    "metrics/trajectory_curvature_mean": np.mean(trajectory_curvatures),
                    "metrics/velocity_mean": np.mean([np.mean(v) for v in latent_velocities]) if latent_velocities else 0.0,
                    "metrics/orig_recon_correlation": corr_coef,
                })
            
            plt.close()
        
        self.model.train()
    
    def create_sequence_trajectories(self, x_sample: torch.Tensor, epoch: int):
        """Create comprehensive visualization of sequence trajectories in latent space.
        Now supports configurable number of sequences, optional clustering, and cluster-based coloring.
        Config options (from self.config or defaults):
            - sequence_viz_count: int or 'all' (default: 8)
            - cluster_sequences: bool (default: False)
            - n_clusters: int (default: 5)
            - cluster_coloring: bool (default: True)
            - show_cluster_centroids: bool (default: False)
        """
        import warnings
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        print(f"\U0001F9E0 Creating sequence trajectory visualization for epoch {epoch}")

        # --- Configurable parameters ---
        sequence_viz_count = getattr(self.config, 'sequence_viz_count', None)
        cluster_sequences = getattr(self.config, 'cluster_sequences', False)
        n_clusters = getattr(self.config, 'n_clusters', 5)
        cluster_coloring = getattr(self.config, 'cluster_coloring', True)
        show_cluster_centroids = getattr(self.config, 'show_cluster_centroids', False)

        self.model.eval()
        with torch.no_grad():
            result = self.model_forward(x_sample)
            # Handle both ModelOutput and dict outputs
            if hasattr(result, 'z'):
                z_seq = result.z  # [batch_size, n_obs, latent_dim]
            else:
                z_seq = result['latent_samples']   # [batch_size, n_obs, latent_dim]

            batch_size, n_obs, latent_dim = z_seq.shape
            # --- Determine number of sequences to plot ---
            if sequence_viz_count is None or sequence_viz_count == 'all':
                num_viz = batch_size
            else:
                num_viz = min(int(sequence_viz_count), batch_size)
    
            if num_viz > 128:
                warnings.warn(f"Plotting {num_viz} sequences may be slow or cluttered! Consider reducing sequence_viz_count.")
            # --- Prepare data for PCA ---
            z_flat = z_seq.reshape(-1, latent_dim).cpu().numpy()
            if np.any(np.isnan(z_flat)):
                print(f"⚠️ NaN values detected in latent representations! Shape: {z_flat.shape}")
                z_flat_clean = np.nan_to_num(z_flat, nan=0.0)
            else:
                z_flat_clean = z_flat
            pca = PCA(n_components=3)
            z_pca = pca.fit_transform(z_flat_clean)
            z_pca_seq = z_pca.reshape(batch_size, n_obs, 3)
            # --- Clustering (optional) ---
            if cluster_sequences and num_viz > 1:
                # Use mean of each sequence in PCA space for clustering
                seq_features = z_pca_seq[:num_viz, :, :2].mean(axis=1)
                kmeans = KMeans(n_clusters=min(n_clusters, num_viz), n_init=10, random_state=0)
                cluster_labels = kmeans.fit_predict(seq_features)
                n_clusters_actual = len(np.unique(cluster_labels))
                # Color map for clusters
                color_map = cm.get_cmap('tab10', n_clusters_actual)
                colors = [color_map(label) for label in cluster_labels]
            else:
                cluster_labels = np.zeros(num_viz, dtype=int)
                colors = cm.get_cmap('tab10')(np.linspace(0, 1, num_viz))
            # --- Plotting ---
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle(f'Latent Sequence Trajectories - Epoch {epoch}', fontsize=14)
            alpha = 0.7 if num_viz <= 32 else 0.4 if num_viz <= 64 else 0.2
            marker_size = 6 if num_viz <= 32 else 4 if num_viz <= 64 else 2
            # Plot 1: 2D PCA Trajectories (PC1 vs PC2)
            ax = axes[0, 0]
            for i in range(num_viz):
                traj = z_pca_seq[i, :, :2]
                color = colors[i] if cluster_coloring else cm.get_cmap('tab10')(i % 10)
                ax.plot(traj[:, 0], traj[:, 1], 'o-', color=color, alpha=alpha, linewidth=1, markersize=marker_size, label=f'Seq {i}' if num_viz <= 16 else None)
                ax.scatter(traj[0, 0], traj[0, 1], color=color, s=40, marker='s', alpha=0.9, edgecolor='black')  # Start
                ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=40, marker='*', alpha=0.9, edgecolor='black')  # End
            if cluster_sequences and cluster_coloring:
                handles = [plt.Line2D([0], [0], color=color_map(i), lw=2, label=f'Cluster {i}') for i in range(n_clusters_actual)]
                ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            elif num_viz <= 16:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.set_title(f'2D Trajectories (PC1 vs PC2)')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.grid(True, alpha=0.3)
            # Optionally plot cluster centroids
            if cluster_sequences and show_cluster_centroids:
                for i in range(n_clusters_actual):
                    centroid = kmeans.cluster_centers_[i]
                    ax.scatter(centroid[0], centroid[1], color=color_map(i), s=120, marker='X', edgecolor='black', label=f'Centroid {i}')
            # Plot 2: Temporal Evolution of PC1
            ax = axes[0, 1]
            for i in range(num_viz):
                color = colors[i] if cluster_coloring else cm.get_cmap('tab10')(i % 10)
                ax.plot(range(n_obs), z_pca_seq[i, :, 0], 'o-', color=color, alpha=alpha, linewidth=1, markersize=marker_size)
            ax.set_title('PC1 Evolution Over Time')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('PC1 Value')
            ax.grid(True, alpha=0.3)
            # Plot 3: Temporal Evolution of PC2
            ax = axes[0, 2]
            for i in range(num_viz):
                color = colors[i] if cluster_coloring else cm.get_cmap('tab10')(i % 10)
                ax.plot(range(n_obs), z_pca_seq[i, :, 1], 'o-', color=color, alpha=alpha, linewidth=1, markersize=marker_size)
            ax.set_title('PC2 Evolution Over Time')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('PC2 Value')
            ax.grid(True, alpha=0.3)
            # Plot 4: Trajectory Lengths
            ax = axes[1, 0]
            traj_lengths = []
            for i in range(num_viz):
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
            start_points = z_pca_seq[:num_viz, 0, :2]
            end_points = z_pca_seq[:num_viz, -1, :2]
            distances = np.linalg.norm(end_points - start_points, axis=1)
            ax.scatter(start_points[:, 0], start_points[:, 1], alpha=0.6, s=30, label='Start Points', marker='s')
            ax.scatter(end_points[:, 0], end_points[:, 1], alpha=0.6, s=30, label='End Points', marker='*')
            for i in range(min(num_viz, 64)):
                ax.plot([start_points[i, 0], end_points[i, 0]], [start_points[i, 1], end_points[i, 1]], 'k--', alpha=0.2, linewidth=0.7)
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
        """Create enhanced comprehensive reconstruction visualization with perceptual metrics."""
        print(f"🎬 Creating enhanced comprehensive reconstruction visualization for epoch {epoch}")
        
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
            max_viz = min(self._get_viz_count(), batch_size)
            
            # Enhanced metrics computation
            print("🔬 Computing enhanced reconstruction metrics...")
            
            # Pixel-level metrics
            mse_per_sample = []
            mae_per_sample = []
            
            # Perceptual metrics (simplified)
            ssim_scores = []
            psnr_scores = []
            
            # Temporal consistency metrics
            temporal_consistency_orig = []
            temporal_consistency_recon = []
            
            # Feature-level analysis (using simple gradients as features)
            feature_similarity_scores = []
            
            for i in range(max_viz):
                # Per-sample MSE and MAE
                sample_mse = torch.mean((x_sample[i] - recon_x[i]) ** 2, dim=(1, 2, 3)).cpu().numpy()
                sample_mae = torch.mean(torch.abs(x_sample[i] - recon_x[i]), dim=(1, 2, 3)).cpu().numpy()
                mse_per_sample.append(sample_mse)
                mae_per_sample.append(sample_mae)
                
                # SSIM and PSNR computation (simplified version)
                sample_ssim = []
                sample_psnr = []
                
                for t in range(n_obs):
                    orig_img = x_sample[i, t].cpu().numpy().transpose(1, 2, 0)
                    recon_img = recon_x[i, t].cpu().numpy().transpose(1, 2, 0)
                    
                    # Simplified SSIM calculation
                    orig_mean = np.mean(orig_img)
                    recon_mean = np.mean(recon_img)
                    orig_var = np.var(orig_img)
                    recon_var = np.var(recon_img)
                    covar = np.mean((orig_img - orig_mean) * (recon_img - recon_mean))
                    
                    c1, c2 = 0.01, 0.03
                    ssim = ((2 * orig_mean * recon_mean + c1) * (2 * covar + c2)) / \
                           ((orig_mean**2 + recon_mean**2 + c1) * (orig_var + recon_var + c2))
                    sample_ssim.append(max(0, min(1, ssim)))
                    
                    # PSNR calculation
                    mse_val = np.mean((orig_img - recon_img) ** 2)
                    if mse_val > 0:
                        psnr = 20 * np.log10(1.0 / np.sqrt(mse_val))
                        sample_psnr.append(psnr)
                    else:
                        sample_psnr.append(100.0)  # Perfect reconstruction
                
                ssim_scores.append(sample_ssim)
                psnr_scores.append(sample_psnr)
                
                # Temporal consistency: variance across time
                if n_obs > 1:
                    orig_temporal_var = np.var(x_sample[i].cpu().numpy(), axis=0)
                    recon_temporal_var = np.var(recon_x[i].cpu().numpy(), axis=0)
                    
                    temporal_consistency_orig.append(np.mean(orig_temporal_var))
                    temporal_consistency_recon.append(np.mean(recon_temporal_var))
                
                # Feature similarity using gradients
                orig_gradients = []
                recon_gradients = []
                
                for t in range(n_obs):
                    # Compute image gradients as simple features
                    orig_gray = np.mean(x_sample[i, t].cpu().numpy(), axis=0)
                    recon_gray = np.mean(recon_x[i, t].cpu().numpy(), axis=0)
                    
                    orig_grad_x = np.abs(np.diff(orig_gray, axis=1))
                    orig_grad_y = np.abs(np.diff(orig_gray, axis=0))
                    recon_grad_x = np.abs(np.diff(recon_gray, axis=1))
                    recon_grad_y = np.abs(np.diff(recon_gray, axis=0))
                    
                    # Pad to make same size
                    min_x_size = min(orig_grad_x.shape[1], recon_grad_x.shape[1])
                    min_y_size = min(orig_grad_y.shape[0], recon_grad_y.shape[0])
                    
                    if min_x_size > 0 and min_y_size > 0:
                        orig_grad_x = orig_grad_x[:, :min_x_size]
                        recon_grad_x = recon_grad_x[:, :min_x_size]
                        orig_grad_y = orig_grad_y[:min_y_size, :]
                        recon_grad_y = recon_grad_y[:min_y_size, :]
                        
                        orig_gradients.append(np.mean(orig_grad_x) + np.mean(orig_grad_y))
                        recon_gradients.append(np.mean(recon_grad_x) + np.mean(recon_grad_y))
                
                if orig_gradients and recon_gradients:
                    feature_similarity = np.corrcoef(orig_gradients, recon_gradients)[0, 1]
                    if not np.isnan(feature_similarity):
                        feature_similarity_scores.append(feature_similarity)
                    else:
                        feature_similarity_scores.append(0.0)
                else:
                    feature_similarity_scores.append(0.0)
            
            # Create enhanced comprehensive visualization
            fig, axes = plt.subplots(4, 4, figsize=(20, 20))
            fig.suptitle(f'Enhanced Reconstruction Analysis - Epoch {epoch}', fontsize=16)
            
            # Row 1: Example comparisons
            seq_idx = 0
            timesteps_to_show = [0, n_obs//3, 2*n_obs//3, n_obs-1] if n_obs >= 4 else list(range(n_obs))
            
            for i, t in enumerate(timesteps_to_show[:4]):
                # Original images
                if i < 4:
                    axes[0, i].imshow(x_sample[seq_idx, t].permute(1, 2, 0).cpu().numpy())
                    axes[0, i].set_title(f'Original t={t}')
                    axes[0, i].axis('off')
            
            # Row 2: Reconstructed images
            for i, t in enumerate(timesteps_to_show[:4]):
                if i < 4:
                    axes[1, i].imshow(recon_x[seq_idx, t].permute(1, 2, 0).cpu().numpy())
                    
                    # Add error as overlay
                    error = torch.mean((x_sample[seq_idx, t] - recon_x[seq_idx, t]) ** 2, dim=0).cpu().numpy()
                    error_normalized = (error - error.min()) / (error.max() - error.min() + 1e-8)
                    
                    axes[1, i].set_title(f'Reconstructed t={t}\nMSE: {np.mean(error):.2e}')
                    axes[1, i].axis('off')
            
            # Row 3: Quantitative analysis
            # Plot 1: MSE evolution across time
            axes[2, 0].clear()
            for i in range(min(max_viz, 6)):  # Show first 6 sequences
                axes[2, 0].plot(mse_per_sample[i], alpha=0.7, linewidth=2, label=f'Seq {i}' if max_viz <= 6 else None)
            axes[2, 0].set_xlabel('Timestep')
            axes[2, 0].set_ylabel('MSE')
            axes[2, 0].set_title('MSE Evolution Across Time')
            axes[2, 0].set_yscale('log')
            if max_viz <= 6:
                axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            
            # Plot 2: SSIM scores
            axes[2, 1].clear()
            if ssim_scores:
                all_ssim = [score for seq_ssim in ssim_scores for score in seq_ssim]
                axes[2, 1].hist(all_ssim, bins=20, alpha=0.7, color='green', density=True)
                axes[2, 1].axvline(np.mean(all_ssim), color='red', linestyle='--', 
                                  label=f'Mean: {np.mean(all_ssim):.3f}')
                axes[2, 1].set_xlabel('SSIM Score')
                axes[2, 1].set_ylabel('Density')
                axes[2, 1].set_title('SSIM Distribution')
                axes[2, 1].legend()
                axes[2, 1].grid(True, alpha=0.3)
            
            # Plot 3: PSNR scores
            axes[2, 2].clear()
            if psnr_scores:
                all_psnr = [score for seq_psnr in psnr_scores for score in seq_psnr]
                axes[2, 2].hist(all_psnr, bins=20, alpha=0.7, color='orange', density=True)
                axes[2, 2].axvline(np.mean(all_psnr), color='red', linestyle='--', 
                                  label=f'Mean: {np.mean(all_psnr):.1f} dB')
                axes[2, 2].set_xlabel('PSNR (dB)')
                axes[2, 2].set_ylabel('Density')
                axes[2, 2].set_title('PSNR Distribution')
                axes[2, 2].legend()
                axes[2, 2].grid(True, alpha=0.3)
            
            # Plot 4: Feature similarity
            axes[2, 3].clear()
            if feature_similarity_scores:
                axes[2, 3].hist(feature_similarity_scores, bins=15, alpha=0.7, color='purple', density=True)
                axes[2, 3].axvline(np.mean(feature_similarity_scores), color='red', linestyle='--', 
                                  label=f'Mean: {np.mean(feature_similarity_scores):.3f}')
                axes[2, 3].set_xlabel('Feature Similarity (Gradient Correlation)')
                axes[2, 3].set_ylabel('Density')
                axes[2, 3].set_title('Feature-Level Similarity')
                axes[2, 3].legend()
                axes[2, 3].grid(True, alpha=0.3)
            
            # Row 4: Advanced analysis
            # Plot 1: Temporal consistency comparison
            axes[3, 0].clear()
            if temporal_consistency_orig and temporal_consistency_recon:
                axes[3, 0].scatter(temporal_consistency_orig, temporal_consistency_recon, alpha=0.6, s=50)
                max_val = max(max(temporal_consistency_orig), max(temporal_consistency_recon))
                axes[3, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect consistency')
                
                consistency_corr = np.corrcoef(temporal_consistency_orig, temporal_consistency_recon)[0, 1]
                axes[3, 0].text(0.05, 0.95, f'r = {consistency_corr:.3f}', 
                               transform=axes[3, 0].transAxes,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                
                axes[3, 0].set_xlabel('Original Temporal Variance')
                axes[3, 0].set_ylabel('Reconstructed Temporal Variance')
                axes[3, 0].set_title('Temporal Consistency Analysis')
                axes[3, 0].legend()
                axes[3, 0].grid(True, alpha=0.3)
            
            # Plot 2: Error distribution by timestep
            axes[3, 1].clear()
            if mse_per_sample:
                timestep_errors = []
                for t in range(n_obs):
                    errors_at_t = [seq_mse[t] for seq_mse in mse_per_sample if t < len(seq_mse)]
                    timestep_errors.append(errors_at_t)
                
                if timestep_errors:
                    axes[3, 1].boxplot(timestep_errors, positions=range(n_obs), widths=0.6, patch_artist=True)
                    axes[3, 1].set_xlabel('Timestep')
                    axes[3, 1].set_ylabel('MSE')
                    axes[3, 1].set_title('Error Distribution by Timestep')
                    axes[3, 1].set_yscale('log')
                    axes[3, 1].grid(True, alpha=0.3)
            
            # Plot 3: Quality metrics correlation
            axes[3, 2].clear()
            if ssim_scores and psnr_scores:
                ssim_flat = [score for seq_ssim in ssim_scores for score in seq_ssim]
                psnr_flat = [score for seq_psnr in psnr_scores for score in seq_psnr]
                
                if len(ssim_flat) == len(psnr_flat):
                    axes[3, 2].scatter(ssim_flat, psnr_flat, alpha=0.6, s=30)
                    quality_corr = np.corrcoef(ssim_flat, psnr_flat)[0, 1]
                    axes[3, 2].text(0.05, 0.95, f'r = {quality_corr:.3f}', 
                                   transform=axes[3, 2].transAxes,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                    axes[3, 2].set_xlabel('SSIM Score')
                    axes[3, 2].set_ylabel('PSNR (dB)')
                    axes[3, 2].set_title('Quality Metrics Correlation')
                    axes[3, 2].grid(True, alpha=0.3)
            
            # Plot 4: Comprehensive statistics summary
            axes[3, 3].clear()
            axes[3, 3].axis('off')
            
            # Create comprehensive statistics text
            stats_text = f"""Enhanced Reconstruction Statistics:
            
Pixel-Level Metrics:
MSE Mean: {np.mean([np.mean(seq_mse) for seq_mse in mse_per_sample]):.2e}
MAE Mean: {np.mean([np.mean(seq_mae) for seq_mae in mae_per_sample]):.2e}

Perceptual Metrics:
SSIM Mean: {np.mean([score for seq_ssim in ssim_scores for score in seq_ssim]):.3f}
PSNR Mean: {np.mean([score for seq_psnr in psnr_scores for score in seq_psnr]):.1f} dB

Feature Analysis:
Gradient Correlation: {np.mean(feature_similarity_scores):.3f}

Temporal Metrics:
Consistency Corr: {consistency_corr if temporal_consistency_orig else 'N/A'}
Quality Correlation: {quality_corr if 'quality_corr' in locals() else 'N/A'}

Sequences Analyzed: {max_viz}
Timesteps: {n_obs}"""
            
            axes[3, 3].text(0.05, 0.95, stats_text, transform=axes[3, 3].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
            axes[3, 3].set_title('Statistical Summary')
            
            plt.tight_layout()
            
            # Save and log with enhanced metrics
            filename = f'enhanced_reconstruction_analysis_epoch_{epoch}.png'
            saved_file = self._safe_save_plt_figure(filename, dpi=200, bbox_inches='tight')
            
            # Log to WandB with comprehensive metrics
            if self.should_log_to_wandb() and saved_file:
                wandb.log({
                    "basic/enhanced_reconstruction_analysis": wandb.Image(saved_file, caption=f"Enhanced Reconstruction Analysis - Epoch {epoch}"),
                    "metrics/reconstruction_mse_mean": np.mean([np.mean(seq_mse) for seq_mse in mse_per_sample]),
                    "metrics/reconstruction_mae_mean": np.mean([np.mean(seq_mae) for seq_mae in mae_per_sample]),
                    "metrics/reconstruction_ssim_mean": np.mean([score for seq_ssim in ssim_scores for score in seq_ssim]),
                    "metrics/reconstruction_psnr_mean": np.mean([score for seq_psnr in psnr_scores for score in seq_psnr]),
                    "metrics/reconstruction_feature_similarity": np.mean(feature_similarity_scores),
                    "metrics/reconstruction_temporal_consistency": consistency_corr if temporal_consistency_orig else 0.0,
                })
            
            plt.close()
        
        self.model.train() 