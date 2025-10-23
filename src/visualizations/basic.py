"""
Basic Visualizations Module
===========================

Essential visualizations for RiemannianFlowVAE training:
- Cyclicity analysis
- Sequence trajectories 
- Reconstruction quality analysis
- Enhanced KL visualization with unified RHMC sampling
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
from typing import Optional, Tuple
from .base import BaseVisualization


class BasicVisualizations(BaseVisualization):
    """Basic visualization suite for essential analysis."""

    def _resolve_metric_path(self) -> Optional[Path]:
        """Resolve the absolute path to the metric checkpoint used in Stage C."""
        candidates = []
        try:
            metric_cfg = getattr(self.config.settings.model, 'metric', None)
            if metric_cfg is not None and hasattr(metric_cfg, 'path'):
                candidates.append(metric_cfg.path)
        except AttributeError:
            pass

        try:
            metric_cfg = getattr(getattr(self.config, 'model', None), 'metric', None)
            if metric_cfg is not None and hasattr(metric_cfg, 'path'):
                candidates.append(metric_cfg.path)
        except AttributeError:
            pass

        if hasattr(self.model, 'metric_path'):
            candidates.append(getattr(self.model, 'metric_path'))

        repo_root = Path(__file__).resolve().parents[2]

        for cand in candidates:
            if not cand:
                continue
            cand_path = Path(str(cand))
            # Direct absolute path
            if cand_path.is_absolute() and cand_path.exists():
                return cand_path
            # Relative to current working directory (Hydra run dir)
            run_path = (Path.cwd() / cand_path).resolve()
            if run_path.exists():
                return run_path
            # Relative to repository root
            repo_path = (repo_root / cand_path).resolve()
            if repo_path.exists():
                return repo_path
            # Fallback to pretrained directory with filename only
            if cand_path.parent == Path('.'):
                pretrained_path = (repo_root / 'data' / 'pretrained' / cand_path.name).resolve()
                if pretrained_path.exists():
                    return pretrained_path
        return None

    def _load_stage_b_pca_basis(self, latent_dim: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load (or lazily compute) the PCA basis captured during Stage B.

        Returns:
            Tuple of (components, mean) if available, else None.
        """
        metric_path = self._resolve_metric_path()
        if metric_path is None:
            return None

        cache_name = f"{metric_path.stem}_pca_ld{latent_dim}.npz"
        cache_path = metric_path.parent / cache_name

        if cache_path.exists():
            try:
                data = np.load(cache_path)
                components = data['components']
                mean = data['mean']
                print(f"✅ Loaded Stage-B PCA basis from {cache_path}")
                return components, mean
            except Exception as exc:
                print(f"⚠️ Failed to load cached PCA basis ({cache_path}): {exc}")

        try:
            metric_data = torch.load(metric_path, map_location='cpu')
        except Exception as exc:
            print(f"⚠️ Unable to load metric checkpoint at {metric_path}: {exc}")
            return None

        centroids = metric_data.get('centroids')
        if centroids is None:
            return None

        if isinstance(centroids, torch.Tensor):
            centroids_np = centroids.detach().cpu().numpy()
        else:
            centroids_np = np.asarray(centroids)

        if centroids_np.ndim != 2 or centroids_np.shape[0] < 2:
            return None

        from sklearn.decomposition import PCA

        n_components = min(2, centroids_np.shape[1])
        if n_components < 1:
            return None

        pca = PCA(n_components=n_components)
        pca.fit(centroids_np)

        components = pca.components_.astype(np.float32)
        mean = pca.mean_.astype(np.float32)

        try:
            np.savez(cache_path, components=components, mean=mean)
            print(f"💾 Cached Stage-B PCA basis at {cache_path}")
        except Exception as exc:
            print(f"⚠️ Could not cache PCA basis ({cache_path}): {exc}")

        return components, mean

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
                # Handle both dict format and ModelOutput format
                if isinstance(result, dict):
                    z_seq = result.get('latent_samples', result.get('z', None))
                else:
                    # ModelOutput object
                    z_seq = result.z if hasattr(result, 'z') else None
                
                if z_seq is None:
                    print("⚠️ Could not extract latent samples from model output")
                    return
                # Ensure z_seq is a tensor with shape [B, T, D]
                try:
                    if isinstance(z_seq, list):
                        z_seq = torch.stack([torch.as_tensor(z) for z in z_seq], dim=0)
                    elif isinstance(z_seq, tuple):
                        z_seq = torch.stack([torch.as_tensor(z) for z in list(z_seq)], dim=0)
                    else:
                        z_seq = torch.as_tensor(z_seq)
                    if z_seq.dim() == 2:
                        # [T, D] -> add batch dim assuming batch_size=1
                        z_seq = z_seq.unsqueeze(0)
                    elif z_seq.dim() > 3:
                        # In case extra dims exist, squeeze redundant dims
                        z_seq = z_seq.squeeze()
                except Exception as e:
                    print(f"⚠️ Failed to coerce latent_samples to tensor: {e}")
                    return
            
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
                
                try:
                    latent_mse = torch.mean((z_seq[i, 0] - z_seq[i, -1]) ** 2).item()
                except Exception as e:
                    print(f"⚠️ Latent cyclicity computation failed for seq {i}: {e}")
                    latent_mse = float('nan')
                latent_first_last_mse.append(latent_mse)
                try:
                    latent_norms.append(torch.norm(z_seq[i], dim=-1).cpu().numpy())
                except Exception:
                    pass
                
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
            try:
                z_traj = z_seq[seq_idx].cpu().numpy()  # [n_obs, latent_dim]
            except Exception as e:
                print(f"⚠️ Could not build latent trajectory for seq {seq_idx}: {e}")
                return
            
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
            self.last_cyclicity_path = saved_file

            # Log latent histogram + stats prior to the image so we can track drift in W&B dashboards
            if self.should_log_to_wandb():
                try:
                    z_tensor = torch.as_tensor(z_seq)
                    if z_tensor.dim() == 3:
                        # Histogram of timestep-0 latents (fallback to all if seq length < 1)
                        z_for_hist = z_tensor[:, 0] if z_tensor.shape[1] > 0 else z_tensor.reshape(-1, z_tensor.shape[-1])
                    else:
                        z_for_hist = z_tensor
                    z_flat = z_for_hist.reshape(-1).cpu().numpy()
                    wandb.log({
                        "basic/latent_hist_t0": wandb.Histogram(z_flat, num_bins=40),
                        "basic/latent_stats_t0/mean": float(np.mean(z_flat)) if z_flat.size else 0.0,
                        "basic/latent_stats_t0/std": float(np.std(z_flat)) if z_flat.size else 0.0,
                        "basic/latent_stats_t0/min": float(np.min(z_flat)) if z_flat.size else 0.0,
                        "basic/latent_stats_t0/max": float(np.max(z_flat)) if z_flat.size else 0.0,
                    })
                except Exception as exc:
                    print(f"⚠️ Failed to log latent histogram: {exc}")
            
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
                # Handle both dict format and ModelOutput format
                if isinstance(result, dict):
                    z_seq = result.get('latent_samples', result.get('z', None))
                else:
                    # ModelOutput object
                    z_seq = result.z if hasattr(result, 'z') else None
                
                if z_seq is None:
                    print("⚠️ Could not extract latent samples from model output")
                    return

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
            # Dynamically set n_components for PCA
            latent_dim_pca = z_flat.shape[1]
            n_components = min(3, latent_dim_pca)
            pca = PCA(n_components=n_components)
            z_pca = pca.fit_transform(z_flat_clean)
            z_pca_seq = z_pca.reshape(batch_size, n_obs, n_components)
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
            self.last_sequence_trajectories_path = saved_file
            # Log to WandB
            if self.should_log_to_wandb() and saved_file:
                wandb.log({
                    "basic/sequence_trajectories": wandb.Image(saved_file, caption=f"Epoch {epoch}"),
                })
            plt.close()
        self.model.train()
    
    def create_reconstruction_analysis(self, x_sample: torch.Tensor, epoch: int):
        """Create enhanced comprehensive reconstruction visualization with perceptual metrics."""
        print(f"[DEBUG] create_reconstruction_analysis called for epoch {epoch}")
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
                # Ensure both tensors are on the same device
                x_sample_i = x_sample[i].to(recon_x[i].device)
                # Per-sample MSE and MAE
                sample_mse = torch.mean((x_sample_i - recon_x[i]) ** 2, dim=(1, 2, 3)).cpu().numpy()
                sample_mae = torch.mean(torch.abs(x_sample_i - recon_x[i]), dim=(1, 2, 3)).cpu().numpy()
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
            self.last_reconstruction_path = saved_file
            
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

    def create_generation_grid(self, num_samples: int, epoch: int):
        """
        Generate a fancy grid of images sampled from the prior and log to wandb.
        Args:
            num_samples: Number of images to generate (should be a square number, e.g., 16 or 25)
            epoch: Current epoch (for labeling)
        """
        self.model.eval()
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, self.model.latent_dim, device=self.device)
            device = next(self.model.decoder.parameters()).device
            z = z.to(device)
            generated = self.model.decode(z)
            print(f"[DEBUG] Decoded output type: {type(generated)}; dir: {dir(generated)}")
            if hasattr(generated, 'recon_x'):
                print("[DEBUG] Using .recon_x from ModelOutput")
                generated = generated.recon_x
            elif hasattr(generated, 'output'):
                print("[DEBUG] Using .output from ModelOutput")
                generated = generated.output
            elif hasattr(generated, 'reconstruction'):
                print("[DEBUG] Using .reconstruction from ModelOutput")
                generated = generated.reconstruction
            else:
                print("[DEBUG] Decoded output is not a ModelOutput or has no recon_x/output/reconstruction.")
            generated = generated.cpu()
            # Clamp to [0, 1]
            generated = torch.clamp(generated, 0, 1)
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
        fig.suptitle(f"Fancy Generation Grid (Epoch {epoch})", fontsize=18, fontweight='bold', color='#4ECDC4')
        for i in range(grid_size * grid_size):
            ax = axes[i // grid_size, i % grid_size]
            if i < num_samples:
                img = generated[i].permute(1, 2, 0).numpy()
                ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        filename = f'generation_grid_epoch_{epoch}.png'
        saved_file = self._safe_save_plt_figure(filename, dpi=200, bbox_inches='tight')
        self.last_generation_path = saved_file
        if self.should_log_to_wandb() and saved_file:
            import wandb
            wandb.log({
                "final/generation_grid": wandb.Image(saved_file, caption=f"Fancy Generation Grid (Epoch {epoch})")
            })
        plt.close()
        self.model.train()

    def create_interpolation_grid(self, num_interpolations: int, steps: int, epoch: int):
        """
        Create a fancy grid of interpolated images between random latent pairs and log to wandb.
        Args:
            num_interpolations: Number of interpolation pairs (rows)
            steps: Number of interpolation steps (columns)
            epoch: Current epoch (for labeling)
        """
        self.model.eval()
        with torch.no_grad():
            device = next(self.model.decoder.parameters()).device
            z_start = torch.randn(num_interpolations, self.model.latent_dim, device=self.device).to(device)
            z_end = torch.randn(num_interpolations, self.model.latent_dim, device=self.device).to(device)
            all_imgs = []
            for i in range(num_interpolations):
                row_imgs = []
                for alpha in np.linspace(0, 1, steps):
                    z = (1 - alpha) * z_start[i] + alpha * z_end[i]
                    z = z.to(device)
                    img = self.model.decode(z.unsqueeze(0))
                    print(f"[DEBUG] Interp decode type: {type(img)}; dir: {dir(img)}")
                    if hasattr(img, 'recon_x'):
                        print("[DEBUG] Using .recon_x from ModelOutput (interp)")
                        img = img.recon_x
                    elif hasattr(img, 'output'):
                        print("[DEBUG] Using .output from ModelOutput (interp)")
                        img = img.output
                    elif hasattr(img, 'reconstruction'):
                        print("[DEBUG] Using .reconstruction from ModelOutput (interp)")
                        img = img.reconstruction
                    else:
                        print("[DEBUG] Interp decode is not a ModelOutput or has no recon_x/output/reconstruction.")
                    img = img.cpu()[0]
                    img = torch.clamp(img, 0, 1)
                    row_imgs.append(img)
                all_imgs.append(row_imgs)
        fig, axes = plt.subplots(num_interpolations, steps, figsize=(steps * 2, num_interpolations * 2))
        fig.suptitle(f"Fancy Interpolation Grid (Epoch {epoch})", fontsize=18, fontweight='bold', color='#FF6B6B')
        for i in range(num_interpolations):
            for j in range(steps):
                ax = axes[i, j]
                img = all_imgs[i][j].permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.axis('off')
                if j == 0:
                    ax.set_ylabel(f"Pair {i+1}", fontsize=12, color='#4ECDC4')
                if i == 0:
                    ax.set_title(f"α={j/(steps-1):.2f}", fontsize=12, color='#FF6B6B')
        plt.tight_layout()
        filename = f'interpolation_grid_epoch_{epoch}.png'
        saved_file = self._safe_save_plt_figure(filename, dpi=200, bbox_inches='tight')
        self.last_interpolation_path = saved_file
        if self.should_log_to_wandb() and saved_file:
            import wandb
            wandb.log({
                "final/interpolation_grid": wandb.Image(saved_file, caption=f"Fancy Interpolation Grid (Epoch {epoch})")
            })
        plt.close()
        self.model.train() 

    def create_enhanced_kl_visualization(self, x_sample: torch.Tensor, epoch: int):
        """
        Create enhanced KL visualization with unified RHMC sampling.
        Shows both prior and posterior samples following the manifold structure.
        Uses evolving filename pattern for consistent tracking.
        """
        print(f"🧠 Creating Enhanced KL Visualization (Epoch {epoch})")
        
        try:
            from src.models.samplers.hmc_sampler import RHVAEVolumeElementHMCSampler
            
            self.model.eval()
            # Encode without gradients, but perform RHMC sampling with gradients enabled
            with torch.no_grad():
                # Use RHMC for BOTH prior AND posterior sampling (UNIFIED approach)
                # Get encoder output for posterior initialization
                result = self.model_forward(x_sample)
                
                # Extract encoder means aligned exactly as used in training
                if isinstance(result, dict):
                    z_posterior = result.get('latent_samples', result.get('z', None))
                    mu_aligned = result.get('mu_aligned', result.get('mu', None))
                else:
                    z_posterior = getattr(result, 'latent_samples', getattr(result, 'z', None))
                    mu_aligned = getattr(result, 'mu_aligned', getattr(result, 'mu', None))

                if mu_aligned is None:
                    # Fallback: use latent samples from forward output
                    if z_posterior is not None:
                        mu_aligned = z_posterior
                        print("⚠️ Using forward latent_samples as μ for visualization (mu_aligned missing)")
                    else:
                        try:
                            enc_fallback = self.model.encoder(x_sample[:, 0])
                            if hasattr(enc_fallback, 'embedding'):
                                mu_aligned = enc_fallback.embedding.unsqueeze(1).expand(-1, x_sample.shape[1], -1)
                                print("⚠️ Using encoder fallback μ for visualization")
                            elif hasattr(enc_fallback, 'mu'):
                                mu_aligned = enc_fallback.mu.unsqueeze(1).expand(-1, x_sample.shape[1], -1)
                                print("⚠️ Using encoder fallback μ for visualization")
                            else:
                                raise AttributeError
                        except Exception:
                            print("⚠️ No aligned encoder means (mu_aligned) available for visualization")
                            return
                if z_posterior is None:
                    z_posterior = mu_aligned  # fallback

                # Ensure proper tensor format
                if isinstance(mu_aligned, list):
                    mu_aligned = torch.stack([torch.as_tensor(z) for z in mu_aligned], dim=0)
                elif isinstance(mu_aligned, tuple):
                    mu_aligned = torch.stack([torch.as_tensor(z) for z in list(mu_aligned)], dim=0)
                if isinstance(z_posterior, list):
                    z_posterior = torch.stack([torch.as_tensor(z) for z in z_posterior], dim=0)
                elif isinstance(z_posterior, tuple):
                    z_posterior = torch.stack([torch.as_tensor(z) for z in list(z_posterior)], dim=0)

                # Flatten tensors
                mu_aligned_flat = mu_aligned.reshape(-1, mu_aligned.shape[-1])  # [N, latent_dim]
                z_posterior_flat = z_posterior.reshape(-1, z_posterior.shape[-1])
                z_encoder_flat = mu_aligned_flat.clone()
                
                # Create RHMC sampler for both prior and posterior samples
                rhmc_sampler = RHVAEVolumeElementHMCSampler(
                    model=self.model,
                    mcmc_steps_nbr=200,
                    n_lf=30,
                    eps_lf=0.001,
                    beta_zero=1.0,
                )
                
                # Get actual posterior samples using the same method as training
                # This shows what the model actually uses during training
                try:
                    # Use the same RHMC posterior sampling as training – requires autograd
                    if hasattr(self.model, 'posterior_sampler_rhmc'):
                        print("✅ Using RHMC posterior sampling for visualization")
                        # Get proper log_var from encoder
                        enc_out_vis = self.model.encoder(x_sample[:, 0])  # Use first timestep
                        if hasattr(enc_out_vis, 'log_covariance'):
                            log_var = enc_out_vis.log_covariance
                        elif hasattr(enc_out_vis, 'log_var'):
                            log_var = enc_out_vis.log_var
                        else:
                            # Fallback to reasonable variance
                            log_var = torch.full_like(mu_aligned_flat, -1.0)  # std ≈ 0.6
                        # Perform RHMC sampling with gradients enabled
                        with torch.enable_grad():
                            rhmc_result = self.model.posterior_sampler_rhmc.sample_riemannian_rhmc_posterior(
                                mu=mu_aligned_flat,
                                log_var=log_var,
                                return_log_prob=False,
                                return_traj=False,
                                return_initial=False
                            )
                        z_posterior_flat = rhmc_result
                        print(f"✅ RHMC posterior samples shape: {z_posterior_flat.shape}")
                    elif hasattr(self.model, 'sample_metric_aware_posterior'):
                        print("✅ Using metric-aware sampling for visualization")
                        # Use metric-aware sampling with proper variance
                        enc_out_vis = self.model.encoder(x_sample[:, 0])
                        if hasattr(enc_out_vis, 'log_covariance'):
                            log_var = enc_out_vis.log_covariance
                        elif hasattr(enc_out_vis, 'log_var'):
                            log_var = enc_out_vis.log_var
                        else:
                            log_var = torch.full_like(mu_aligned_flat, -1.0)
                        z_posterior_flat = self.model.sample_metric_aware_posterior(
                            mu=mu_aligned_flat,
                            log_var=log_var
                        )
                        print(f"✅ Metric-aware posterior samples shape: {z_posterior_flat.shape}")
                    else:
                        # Fallback to encoder output
                        z_posterior_flat = mu_aligned_flat
                        print("⚠️ Using aligned encoder means as fallback")
                except Exception as e:
                    print(f"⚠️ Posterior sampling failed: {e}, using encoder means")
                    z_posterior_flat = mu_aligned_flat
                
                # ADD MORE ENCODER MEANS FOR BETTER COVERAGE
                # Sample additional encoder means from the dataset for better visualization
                try:
                    print("📊 Adding more encoder means for better coverage...")
                    # Get a larger sample from the dataset
                    n_additional_samples = min(200, len(x_sample) * 4)  # More samples
                    additional_x = x_sample[:n_additional_samples] if len(x_sample) >= n_additional_samples else x_sample
                    
                    # Get encoder means for additional samples
                    additional_result = self.model_forward(additional_x)
                    if isinstance(additional_result, dict):
                        additional_mu = additional_result.get('mu_aligned', additional_result.get('mu', None))
                    else:
                        additional_mu = getattr(additional_result, 'mu_aligned', getattr(additional_result, 'mu', None))
                    if additional_mu is None:
                        additional_mu = z_encoder_flat[:n_additional_samples]
                    else:
                        if isinstance(additional_mu, list):
                            additional_mu = torch.stack([torch.as_tensor(z) for z in additional_mu], dim=0)
                        elif isinstance(additional_mu, tuple):
                            additional_mu = torch.stack([torch.as_tensor(z) for z in list(additional_mu)], dim=0)
                        additional_mu = additional_mu.reshape(-1, additional_mu.shape[-1])
                    
                    # Combine with original encoder means
                    z_encoder_flat = torch.cat([z_encoder_flat, additional_mu], dim=0)
                    print(f"✅ Added {additional_mu.shape[0]} more encoder means (total: {z_encoder_flat.shape[0]})")
                except Exception as e:
                    print(f"⚠️ Could not add more encoder means: {e}")
                
                # Sample prior points using RHMC (volume‑element sampler uses analytic gradients)
                n_prior_samples = 200
                z_prior = rhmc_sampler.sample(n_samples=n_prior_samples)  # [n_prior_samples, latent_dim]
                
                # Get centroids for visualization
                if hasattr(self.model, 'centroids_tens'):
                    centroids = self.model.centroids_tens.detach().cpu().numpy()
                else:
                    print("⚠️ No centroids available for enhanced KL visualization")
                    centroids = np.array([])
                
                # Compute manifold structure (G^-1 determinant)
                z_grid = torch.linspace(-5, 5, 100, device=self.device)
                X, Y = torch.meshgrid(z_grid, z_grid, indexing='ij')
                Z_grid = torch.stack([X.flatten(), Y.flatten()], dim=1)  # [10000, 2]
                
                # Prepare 2D projection using Stage-B PCA basis when available
                posterior_np = z_posterior_flat.cpu().numpy()
                prior_np = z_prior.cpu().numpy()
                if isinstance(centroids, np.ndarray) and centroids.ndim == 2:
                    centroids_np = centroids
                else:
                    centroids_np = np.empty((0, posterior_np.shape[1]))

                stage_b_pca = self._load_stage_b_pca_basis(latent_dim=posterior_np.shape[1])
                if stage_b_pca is not None:
                    components, mean = stage_b_pca
                    print("✅ Using Stage-B PCA basis for enhanced KL visualization")

                    def project(arr: np.ndarray) -> np.ndarray:
                        return (arr - mean) @ components.T

                    def inverse(arr: np.ndarray) -> np.ndarray:
                        return arr @ components + mean

                    z_posterior_2d = project(posterior_np)
                    z_prior_2d = project(prior_np)
                    centroids_2d = project(centroids_np) if centroids_np.size else np.empty((0, components.shape[0]))
                else:
                    from sklearn.decomposition import PCA
                    all_samples = np.vstack([posterior_np, prior_np])
                    if centroids_np.size:
                        all_samples = np.vstack([all_samples, centroids_np])
                    pca = PCA(n_components=min(2, posterior_np.shape[1]))
                    pca.fit(all_samples)

                    def project(arr: np.ndarray, pca_obj=pca) -> np.ndarray:
                        return pca_obj.transform(arr)

                    def inverse(arr: np.ndarray, pca_obj=pca) -> np.ndarray:
                        return pca_obj.inverse_transform(arr)

                    z_posterior_2d = project(posterior_np)
                    z_prior_2d = project(prior_np)
                    centroids_2d = project(centroids_np) if centroids_np.size else np.empty((0, pca.n_components_))
                
                # FIX: Create grid for manifold structure visualization with DYNAMIC RANGE
                # Compute the actual range of all data in PCA space
                all_data_2d = np.vstack([z_posterior_2d, z_prior_2d])
                if len(centroids_2d) > 0:
                    all_data_2d = np.vstack([all_data_2d, centroids_2d])
                
                data_min = all_data_2d.min(axis=0)
                data_max = all_data_2d.max(axis=0)
                
                # Add padding to ensure full coverage
                padding = 1.0
                x_min, x_max = data_min[0] - padding, data_max[0] + padding
                y_min, y_max = data_min[1] - padding, data_max[1] + padding
                
                # Create grid with higher resolution covering the full data range
                x_grid = np.linspace(x_min, x_max, 150)  # Dynamic range + higher resolution
                y_grid = np.linspace(y_min, y_max, 150)  # Dynamic range + higher resolution
                X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
                Z_grid_2d = np.column_stack([X_grid.flatten(), Y_grid.flatten()])
                
                # Compute log det(G^{-1}) for each grid point (consistent with Stage B plots)
                det_G_grid = np.zeros(len(Z_grid_2d))
                for i, z_2d in enumerate(Z_grid_2d):
                    try:
                        # Project back to full latent space (approximate)
                        z_full = inverse(z_2d.reshape(1, -1))
                        z_tensor = torch.tensor(z_full, dtype=torch.float32, device=self.device)
                        
                        # Prefer orientation in precision space: log det(G^{-1})
                        logdetGinv = None
                        try:
                            if hasattr(self.model, 'modular_metric') and hasattr(self.model.modular_metric, 'compute_log_det_inverse_metric'):
                                logdet_tensor = self.model.modular_metric.compute_log_det_inverse_metric(z_tensor)
                                logdetGinv = logdet_tensor.reshape(-1)[0].item()
                            elif hasattr(self.model, 'metric') and hasattr(self.model.metric, 'compute_log_det_inverse_metric'):
                                logdet_tensor = self.model.metric.compute_log_det_inverse_metric(z_tensor)
                                logdetGinv = logdet_tensor.reshape(-1)[0].item()
                            elif hasattr(self.model, 'G_inv'):
                                G_inv = self.model.G_inv(z_tensor)
                                _, logdet_inv = torch.slogdet(G_inv.float())
                                logdetGinv = logdet_inv.item()
                            elif hasattr(self.model, 'modular_metric'):
                                G_inv = self.model.modular_metric.compute_inverse_metric(z_tensor)
                                _, logdet_inv = torch.slogdet(G_inv.float())
                                logdetGinv = logdet_inv.item()
                            elif hasattr(self.model, 'metric') and hasattr(self.model.metric, 'compute_inverse_metric'):
                                G_inv = self.model.metric.compute_inverse_metric(z_tensor)
                                _, logdet_inv = torch.slogdet(G_inv.float())
                                logdetGinv = logdet_inv.item()
                        except Exception:
                            logdetGinv = None

                        if logdetGinv is None:
                            # Fallback via G if needed: log det(G^{-1}) = - log det(G)
                            try:
                                if hasattr(self.model, 'G'):
                                    G = self.model.G(z_tensor)
                                elif hasattr(self.model, 'modular_metric'):
                                    G = self.model.modular_metric.compute_metric(z_tensor)
                                elif hasattr(self.model, 'metric') and hasattr(self.model.metric, 'compute_metric'):
                                    G = self.model.metric.compute_metric(z_tensor)
                                else:
                                    G = None
                                if G is not None:
                                    _, logdet = torch.slogdet(G.float())
                                    logdetGinv = (-logdet).item()
                            except Exception:
                                logdetGinv = None

                        det_G_grid[i] = np.exp(logdetGinv) if logdetGinv is not None else 1.0
                    except:
                        det_G_grid[i] = 1.0

                # Ensure we have a proper range for gradient visualization
                det_G_grid = det_G_grid.reshape(X_grid.shape)
                
                # Normalize to create a proper gradient
                det_min = np.min(det_G_grid)
                det_max = np.max(det_G_grid)
                
                # preserve raw determinant values even if the range is small to avoid artificial gradients
                det_G_grid = det_G_grid.reshape(X_grid.shape)
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Plot manifold structure as contour (log det G^{-1}) to match Stage B intuition
                det_log = np.log10(det_G_grid + 1e-16)
                
                # Create more levels for smoother gradient
                levels = np.linspace(det_log.min(), det_log.max(), 100)
                contour = ax.contourf(X_grid, Y_grid, det_log, levels=levels, cmap='viridis', alpha=0.7)
                cbar = plt.colorbar(contour, ax=ax, label='log₁₀(det(G⁻¹))')
                
                # Add contour lines for better definition
                contour_lines = ax.contour(X_grid, Y_grid, det_log, levels=levels[::10], colors='white', alpha=0.3, linewidths=0.5)

                # One-time metric sanity check
                try:
                    with torch.no_grad():
                        z_test = z_encoder_flat[: min(32, z_encoder_flat.shape[0])]
                        if hasattr(self.model, 'G'):
                            Gt = self.model.G(z_test).float()
                        elif hasattr(self.model, 'modular_metric'):
                            Gt = self.model.modular_metric.compute_metric(z_test).float()
                        else:
                            Gt = None
                        if hasattr(self.model, 'G_inv'):
                            Ginv_t = self.model.G_inv(z_test).float()
                        elif hasattr(self.model, 'modular_metric'):
                            Ginv_t = self.model.modular_metric.compute_inverse_metric(z_test).float()
                        else:
                            Ginv_t = None
                        if Gt is not None and Ginv_t is not None:
                            I = torch.eye(Gt.shape[-1], device=Gt.device).unsqueeze(0)
                            err = (torch.matmul(Gt, Ginv_t) - I).norm(dim=(-2, -1)).mean().item()
                            print(f"[METRIC CHECK] ||G G^-1 - I|| ≈ {err:.2e}")
                except Exception as _e:
                    pass
                
                # Plot posterior samples
                ax.scatter(z_posterior_2d[:, 0], z_posterior_2d[:, 1],
                          c='blue', s=20, alpha=0.6, label='Posterior Samples (Metric-Aware)')

                # Plot encoder means μ (green crosses) - these are the centers
                z_encoder_2d = project(z_encoder_flat.cpu().numpy())
                ax.scatter(z_encoder_2d[:, 0], z_encoder_2d[:, 1],
                          c='green', s=50, marker='x', alpha=0.8, linewidth=2,
                          label='Encoder Means μ (Centers)', zorder=4)

                # Plot prior samples
                ax.scatter(z_prior_2d[:, 0], z_prior_2d[:, 1],
                          c='red', s=20, alpha=0.6, label='Prior Samples (UNIFIED RHMC)')

                # Plot centroids
                if len(centroids_2d) > 0:
                    ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                              c='cyan', s=100, edgecolors='black', linewidth=2,
                              label='Centroids (Final)', zorder=5)

                ax.set_xlabel('Stage-B PCA Component 1')
                ax.set_ylabel('Stage-B PCA Component 2')
                ax.set_title(f'Enhanced KL Visualization: UNIFIED RHMC Sampling (Epoch {epoch})')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
    
                
                plt.tight_layout()
                
                # Use evolving filename pattern for consistent tracking
                filename = f'enhanced_kl_visualization.png'  # Fixed name without epoch
                saved_file = self._safe_save_plt_figure(filename, dpi=200, bbox_inches='tight')
                
                if self.should_log_to_wandb() and saved_file:
                    wandb.log({
                        f"enhanced_kl/visualization": wandb.Image(
                            saved_file, 
                            caption=f"Enhanced KL Visualization: UNIFIED RHMC Sampling (Epoch {epoch})"
                        )
                    })
                
                plt.close()
                print(f"✅ Enhanced KL visualization saved: {filename}")
                
        except Exception as e:
            print(f"⚠️ Enhanced KL visualization failed: {e}")
            import traceback
            traceback.print_exc()

    def create_rhmc_trajectory_overlay(
        self,
        x_sample: torch.Tensor,
        epoch: int,
        n_paths: int = 15
    ) -> None:
        """
        Create a dedicated visualization that overlays RHMC trajectories as arrows on top of
        the latent manifold. This complements the enhanced KL plot without modifying it.
        """
        print(f"🧭 Creating RHMC trajectory overlay (Epoch {epoch})")

        if not hasattr(self.model, 'posterior_sampler_rhmc'):
            print("⚠️ RHMC posterior sampler not available; skipping trajectory overlay.")
            return

        try:
            from matplotlib.patches import FancyArrowPatch
        except Exception:
            print("⚠️ Matplotlib arrow patches unavailable; skipping trajectory overlay.")
            return

        try:
            self.model.eval()
            device = self.device

            # Forward pass (no grad) to obtain encoder outputs
            with torch.no_grad():
                result = self.model_forward(x_sample)

            # Extract encoder latents
            if isinstance(result, dict):
                z_encoder = result.get('latent_samples', None)
            else:
                z_encoder = getattr(result, 'z', None)

            if z_encoder is None:
                print("⚠️ Could not extract encoder latents for trajectory overlay")
                return

            batch_size, n_obs, latent_dim = z_encoder.shape
            z_encoder_flat = z_encoder[:, 0].detach()  # Use first observation

            # Obtain log variance for RHMC sampler
            encoder_out = self.model.encoder(x_sample[:, 0])
            if hasattr(encoder_out, 'log_covariance'):
                log_var = encoder_out.log_covariance.detach()
            elif hasattr(encoder_out, 'log_var'):
                log_var = encoder_out.log_var.detach()
            elif isinstance(encoder_out, dict) and 'log_var' in encoder_out:
                log_var = encoder_out['log_var'].detach()
            else:
                log_var = torch.full_like(z_encoder_flat, -1.0)  # fallback variance

            # Posterior samples (context)
            try:
                with torch.enable_grad():
                    mu_for_samples = z_encoder_flat.clone().detach().requires_grad_(True)
                    log_var_samples = log_var.clone().detach()
                    posterior_samples = self.model.posterior_sampler_rhmc.sample_riemannian_rhmc_posterior(
                        mu=mu_for_samples,
                        log_var=log_var_samples,
                        return_log_prob=False,
                        return_traj=False,
                        return_initial=False
                    )
                z_posterior_flat = posterior_samples.detach()
            except Exception as e:
                print(f"⚠️ Posterior sampling failed ({e}); using encoder means as fallback")
                z_posterior_flat = z_encoder_flat

            # Prior samples (same volume-element sampler as enhanced KL)
            try:
                from src.models.samplers.hmc_sampler import RHVAEVolumeElementHMCSampler
                if not hasattr(self, '_rhmc_prior_sampler'):
                    self._rhmc_prior_sampler = RHVAEVolumeElementHMCSampler(self.model)
                rhmc_sampler = self._rhmc_prior_sampler
                z_prior = rhmc_sampler.sample(n_samples=200)
            except Exception:
                z_prior = torch.randn_like(z_posterior_flat)

            num_paths = min(n_paths, z_encoder_flat.shape[0])
            if num_paths == 0:
                print("⚠️ No encoder samples available for trajectories.")
                return
            idx_paths = torch.randperm(z_encoder_flat.shape[0], device=z_encoder_flat.device)[:num_paths]
            mu_selected = z_encoder_flat[idx_paths]
            log_var_selected = log_var[idx_paths]

            sampler = self.model.posterior_sampler_rhmc
            original_steps = int(getattr(sampler, 'rhmc_steps', 0))
            original_step_size = float(getattr(sampler, 'rhmc_step_size', 0.02))
            viz_steps = max(1, original_steps)
            viz_step_size = original_step_size if original_step_size > 0 else 0.02

            with torch.enable_grad():
                mu_req = mu_selected.detach().clone().requires_grad_(True)
                log_var_req = log_var_selected.detach().clone()
                try:
                    sampler.rhmc_steps = viz_steps
                    sampler.rhmc_step_size = viz_step_size
                    rhmc_output = sampler.sample_riemannian_rhmc_posterior(
                        mu=mu_req,
                        log_var=log_var_req,
                        return_log_prob=False,
                        return_traj=True,
                        return_initial=True
                    )
                finally:
                    sampler.rhmc_steps = original_steps
                    sampler.rhmc_step_size = original_step_size

            if isinstance(rhmc_output, tuple):
                z_final_sel = rhmc_output[0]
                z0_sel = None
                traj_info = None
                for item in rhmc_output[1:]:
                    if isinstance(item, torch.Tensor) and item.shape == z_final_sel.shape and z0_sel is None:
                        z0_sel = item
                    elif isinstance(item, dict):
                        traj_info = item
                if z0_sel is None:
                    z0_sel = mu_selected
            else:
                z_final_sel = rhmc_output
                z0_sel = mu_selected
                traj_info = None

            trajectory_points = []
            if isinstance(traj_info, dict) and 'trajectory' in traj_info:
                traj_states = traj_info['trajectory']
                for path_id in range(num_paths):
                    states = []
                    for entry in traj_states:
                        z_entry = entry.get('z', None)
                        if z_entry is None:
                            continue
                        states.append(z_entry[path_id].detach().cpu())
                    if states:
                        trajectory_points.append(torch.stack(states))
                    else:
                        trajectory_points.append(torch.stack([z0_sel[path_id].detach().cpu(),
                                                              z_final_sel[path_id].detach().cpu()]))
            else:
                for path_id in range(num_paths):
                    start = z0_sel[path_id].detach().cpu()
                    end = z_final_sel[path_id].detach().cpu()
                    trajectory_points.append(torch.stack([start, end]))

            samples_for_pca = [
                z_posterior_flat.detach().cpu().numpy(),
                z_prior.detach().cpu().numpy(),
                z_encoder_flat.detach().cpu().numpy(),
            ]

            if hasattr(self.model, 'centroids_tens') and self.model.centroids_tens is not None:
                centroids_np = self.model.centroids_tens.detach().cpu().numpy()
            else:
                centroids_np = np.empty((0, latent_dim))

            if centroids_np.size:
                samples_for_pca.append(centroids_np)

            traj_concat = []
            for pts in trajectory_points:
                traj_concat.append(pts.numpy())
            if traj_concat:
                samples_for_pca.append(np.concatenate(traj_concat, axis=0))

            all_for_pca = np.concatenate(samples_for_pca, axis=0)
            from sklearn.decomposition import PCA
            pca_components = min(2, latent_dim)
            pca = PCA(n_components=pca_components)
            pca.fit(all_for_pca)

            z_posterior_2d = pca.transform(z_posterior_flat.detach().cpu().numpy())
            z_prior_2d = pca.transform(z_prior.detach().cpu().numpy())
            z_encoder_2d = pca.transform(z_encoder_flat.detach().cpu().numpy())
            centroids_2d = pca.transform(centroids_np) if centroids_np.size else np.empty((0, 2))

            traj_proj = [pca.transform(pts.numpy()) for pts in trajectory_points]

            all_2d_points = [z_posterior_2d, z_prior_2d, z_encoder_2d]
            if centroids_2d.size:
                all_2d_points.append(centroids_2d)
            if traj_proj:
                all_2d_points.extend(traj_proj)

            all_concat = np.concatenate(all_2d_points, axis=0)
            data_min = np.percentile(all_concat, 1, axis=0)
            data_max = np.percentile(all_concat, 99, axis=0)
            padding = 0.15 * (data_max - data_min + 1e-6)
            x_min, x_max = data_min[0] - padding[0], data_max[0] + padding[0]
            y_min, y_max = data_min[1] - padding[1], data_max[1] + padding[1]

            grid_res = 160
            x_grid = np.linspace(x_min, x_max, grid_res)
            y_grid = np.linspace(y_min, y_max, grid_res)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            Z_grid_2d = np.column_stack([X_grid.flatten(), Y_grid.flatten()])

            latent_grid = pca.inverse_transform(Z_grid_2d)

            latent_grid_tensor = torch.tensor(latent_grid, dtype=torch.float32, device=device)
            with torch.no_grad():
                if hasattr(self.model, 'G_inv'):
                    G_eval = self.model.G_inv(latent_grid_tensor)
                    sign, logdet = torch.linalg.slogdet(G_eval.float())
                    det_log = logdet.cpu().numpy()
                elif hasattr(self.model, 'G'):
                    G_eval = self.model.G(latent_grid_tensor)
                    sign, logdet = torch.linalg.slogdet(G_eval.float())
                    det_log = (-logdet).cpu().numpy()
                elif hasattr(self.model, 'modular_metric'):
                    G_eval = self.model.modular_metric.compute_inverse_metric(latent_grid_tensor)
                    sign, logdet = torch.linalg.slogdet(G_eval.float())
                    det_log = logdet.cpu().numpy()
                else:
                    det_log = np.zeros(len(latent_grid_tensor))

            det_log = (det_log.reshape(X_grid.shape)) / np.log(10.0)

            fig, ax = plt.subplots(figsize=(10, 9))

            levels = np.linspace(det_log.min(), det_log.max(), 120)
            contour = ax.contourf(
                X_grid, Y_grid, det_log,
                levels=levels, cmap='viridis', alpha=0.92
            )
            plt.colorbar(contour, ax=ax, label='log₁₀(det(G⁻¹(z)))')

            ax.scatter(z_posterior_2d[:, 0], z_posterior_2d[:, 1],
                       c='royalblue', s=16, alpha=0.6, label='Posterior Samples')
            ax.scatter(z_prior_2d[:, 0], z_prior_2d[:, 1],
                       c='tomato', s=18, alpha=0.42, label='Prior Samples')
            ax.scatter(z_encoder_2d[:, 0], z_encoder_2d[:, 1],
                       c='lime', marker='x', s=50, linewidths=1.7, label='Encoder Means')
            if centroids_2d.size:
                ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                           facecolors='cyan', edgecolors='black', s=100, linewidths=1.2,
                           label='Centroids', zorder=5)

            cmap = plt.get_cmap('tab20')
            colors = [cmap(i % cmap.N) for i in range(len(traj_proj))]
            for idx_path, points_2d in enumerate(traj_proj):
                color = colors[idx_path]
                ax.plot(points_2d[:, 0], points_2d[:, 1],
                        color=color, linewidth=1.5, alpha=0.92, label=None)
                for step_idx in range(len(points_2d) - 1):
                    start = points_2d[step_idx]
                    end = points_2d[step_idx + 1]
                    if np.allclose(start, end):
                        continue
                    arrow = FancyArrowPatch(
                        posA=start, posB=end,
                        arrowstyle='->', mutation_scale=12,
                        color=color, linewidth=1.6, alpha=0.95
                    )
                    ax.add_patch(arrow)
                ax.scatter(points_2d[0, 0], points_2d[0, 1],
                           facecolors='none', edgecolors=color, s=75, linewidths=1.8)
                ax.scatter(points_2d[-1, 0], points_2d[-1, 1],
                           facecolors=color, edgecolors='white', s=55, linewidths=1.0)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('PCA Component 1')
            ax.set_ylabel('PCA Component 2')
            ax.set_title(f'RHMC Trajectory Overlay (Epoch {epoch})')
            ax.grid(True, alpha=0.25)
            ax.legend(loc='upper right')

            summary_text = (
                f"Trajectory Paths: {len(traj_proj)}\n"
                f"RHMC steps per path (viz): {viz_steps}\n"
                f"Posterior alpha: {getattr(sampler, 'rhmc_alpha', 'n/a')}\n"
                f"Latent dim: {latent_dim}, PCA dim: {pca_components}"
            )
            ax.text(
                0.02, 0.02, summary_text,
                transform=ax.transAxes,
                fontsize=8, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.72)
            )

            plt.tight_layout()
            filename = f'rhmc_trajectory_overlay.png'
            saved_file = self._safe_save_plt_figure(filename, dpi=200, bbox_inches='tight')
            if self.should_log_to_wandb() and saved_file:
                wandb.log({
                    f"enhanced_kl/rhmc_trajectory_overlay": wandb.Image(
                        saved_file,
                        caption=f"RHMC Trajectory Overlay (Epoch {epoch})"
                    )
                })
            plt.close(fig)
            print(f"✅ RHMC trajectory overlay saved: {filename}")

        except Exception as e:
            print(f"⚠️ RHMC trajectory overlay failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.model.train()

    def create_comprehensive_generation_visualization(self, generation_results: dict, fid_scores: dict = None, num_samples_per_method: int = 4, epoch: int = 0):
        """
        Create a comprehensive grid comparing generation methods, inspired by comprehensive_rlvae_analysis.py.
        Args:
            generation_results: dict mapping method name to images tensor (N, C, H, W) or (N, T, C, H, W)
            fid_scores: optional dict mapping method name to FID score
            num_samples_per_method: number of samples to show per method
            epoch: current epoch (for labeling)
        """
        import seaborn as sns
        methods = [m for m in generation_results.keys() if generation_results[m] is not None]
        n_methods = len(methods)
        if n_methods == 0:
            print("[BasicVisualizations] No generation results to visualize.")
            return
        grid_image = torch.zeros(3, n_methods * 64, num_samples_per_method * 64)
        for i, method in enumerate(methods):
            images = generation_results[method]
            if images.dim() == 5:
                images = images[:, 0]  # Take first frame if sequence
            for j in range(min(num_samples_per_method, len(images))):
                img = images[j]
                grid_image[:, i*64:(i+1)*64, j*64:(j+1)*64] = img
        grid_np = grid_image.permute(1, 2, 0).numpy()
        grid_np = np.clip(grid_np, 0, 1)
        fig, ax = plt.subplots(figsize=(num_samples_per_method * 2.5, n_methods * 2.5))
        ax.imshow(grid_np)
        ax.set_title(f"Comprehensive Generation Comparison (Epoch {epoch})", fontsize=18, fontweight='bold', color='#4ECDC4')
        ax.axis('off')
        palette = sns.color_palette("husl", n_methods)
        for i, method in enumerate(methods):
            label = method.capitalize()
            if fid_scores and method in fid_scores:
                label += f"\nFID: {fid_scores[method]:.1f}"
            ax.text(-10, i*64 + 32, label, rotation=0, ha='right', va='center',
                    fontweight='bold', color=palette[i], fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor=palette[i]))
        plt.tight_layout()
        filename = f'comprehensive_generation_grid_epoch_{epoch}.png'
        saved_file = self._safe_save_plt_figure(filename, dpi=200, bbox_inches='tight')
        self.last_comprehensive_generation_path = saved_file
        if self.should_log_to_wandb() and saved_file:
            import wandb
            wandb.log({
                "final/comprehensive_generation_grid": wandb.Image(saved_file, caption=f"Comprehensive Generation Comparison (Epoch {epoch})")
            })
        plt.close() 
