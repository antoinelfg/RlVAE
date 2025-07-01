"""
Manifold Visualizations Module
=============================

Metric tensor and manifold geometry visualizations:
- Metric heatmaps
- PCA analysis 
- Temporal metric evolution
- Curvature analysis (when enabled)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from .base import BaseVisualization


class ManifoldVisualizations(BaseVisualization):
    """Manifold and metric tensor visualization suite."""
    
    def create_metric_heatmaps(self, x_sample: torch.Tensor, epoch: int):
        """
        Create comprehensive manifold visualizations with proper metric computation across all timesteps.
        FIXED: Now computes metrics for flow-evolved latent coordinates at all timesteps.
        """
        print(f"🔬 Creating manifold visualizations for epoch {epoch}")
        
        if not hasattr(self.model, 'G'):
            print("⚠️ No metric tensor available for manifold visualization")
            return
            
        try:
            self.model.eval()
            with torch.no_grad():
                # Get the full model forward pass first to get properly flow-evolved latents
                result = self.model_forward(x_sample)
                z_seq = result['latent_samples'] if isinstance(result, dict) else result.z
            
                batch_size, n_obs, latent_dim = z_seq.shape
                print(f"📊 Analyzing {batch_size} sequences with {n_obs} timesteps")
                
                # ===== IMPROVED: Use flow-evolved coordinates for metric computation =====
                timestep_data = {}
                
                for t in range(n_obs):
                    print(f"📍 Computing metrics for timestep {t+1}/{n_obs} using flow-evolved coordinates")
                    
                    # CRITICAL FIX: Use the flow-evolved latent coordinates z_t
                    z_t = z_seq[:, t, :]  # [batch_size, latent_dim] - flow-evolved coordinates
                    
                    # Also get the original encoded representation for comparison
                    x_t = x_sample[:, t]
                    encoder_out = self.model.encoder(x_t)
                    mu_t = encoder_out.embedding
                    log_var_t = encoder_out.log_covariance
                    
                    # Generate analysis samples around the FLOW-EVOLVED coordinates
                    n_analysis_samples = min(500, batch_size * 50)  # Reasonable number
                    
                    # Method 1: Sample around flow-evolved coordinates
                    z_t_expanded = z_t.unsqueeze(1).expand(-1, n_analysis_samples // batch_size, -1)
                    z_t_expanded = z_t_expanded.reshape(-1, latent_dim)
                    
                    # Add some noise for variety
                    noise_scale = 0.1  # Small noise to explore neighborhood
                    noise = torch.randn_like(z_t_expanded) * noise_scale
                    z_samples_around_flow = z_t_expanded + noise
                    
                    # Method 2: Sample around encoder mean (for comparison)
                    mu_expanded = mu_t.unsqueeze(1).expand(-1, n_analysis_samples // batch_size, -1)
                    mu_expanded = mu_expanded.reshape(-1, latent_dim)
                    std_expanded = torch.exp(0.5 * log_var_t).unsqueeze(1).expand(-1, n_analysis_samples // batch_size, -1)
                    std_expanded = std_expanded.reshape(-1, latent_dim)
                    
                    eps = torch.randn_like(mu_expanded)
                    z_samples_reparam = mu_expanded + eps * std_expanded
                    
                    # ===== COMPUTE METRICS AT FLOW-EVOLVED COORDINATES =====
                    try:
                        print(f"   Computing G(z) at {len(z_samples_around_flow)} flow-evolved points...")
                        
                        # Compute metric tensor at flow-evolved coordinates
                        G_flow = self.model.G(z_samples_around_flow)  # [n_samples, latent_dim, latent_dim]
                        G_inv_flow = self.model.G_inv(z_samples_around_flow)
                        
                        # Extract metric properties
                        eigenvals_flow = torch.linalg.eigvals(G_inv_flow).real
                        det_G_inv_flow = torch.linalg.det(G_inv_flow)
                        
                        metric_properties = {
                            'eigenvals_mean': eigenvals_flow.mean(dim=0).cpu().numpy(),
                            'eigenvals_std': eigenvals_flow.std(dim=0).cpu().numpy(),
                            'condition_number': (eigenvals_flow.max(dim=1)[0] / (eigenvals_flow.min(dim=1)[0] + 1e-10)).mean().item(),
                            'det_G_inv': det_G_inv_flow.cpu().numpy(),
                            'log_det_G_inv': torch.log(torch.clamp(det_G_inv_flow, min=1e-12)).cpu().numpy(),
                            'eigenvals_all': eigenvals_flow.cpu().numpy(),
                            'samples_used': 'flow_evolved'  # Tag to indicate which coordinates were used
                        }
                        
                        print(f"   ✅ Computed metrics: det(G⁻¹) range [{det_G_inv_flow.min():.2e}, {det_G_inv_flow.max():.2e}]")
                        
                    except Exception as metric_error:
                        print(f"   ⚠️ Failed to compute metrics at flow coordinates: {metric_error}")
                        # Fallback to empty metrics
                        metric_properties = {'samples_used': 'failed'}
                    
                    # Apply flows for trajectory tracking (if not first timestep)
                    flow_intermediate = []
                    z_flow_traj = z_samples_reparam.clone()
                    
                    if t > 0:
                        # Apply flow sequence to see transformation
                        for flow_idx in range(t):
                            # Handle both legacy and modular model structures
                            flows = getattr(self.model, 'flows', None)
                            if flows is None and hasattr(self.model, 'flow_manager'):
                                flows = self.model.flow_manager.flows
                            
                            if flows is not None and flow_idx < len(flows):
                                try:
                                    z_prev = z_flow_traj.clone()
                                    flow_result = flows[flow_idx](z_flow_traj)
                                    z_flow_traj = flow_result.out
                                    
                                    flow_intermediate.append({
                                        'layer': flow_idx,
                                        'z_before': z_prev.cpu().numpy(),
                                        'z_after': z_flow_traj.cpu().numpy()
                                    })
                                except Exception as flow_error:
                                    print(f"   ⚠️ Flow {flow_idx} failed: {flow_error}")
                                    break
                    
                    # Store comprehensive timestep data
                    timestep_data[t] = {
                        'mu': mu_t.cpu().numpy(),
                        'log_var': log_var_t.cpu().numpy(),
                        'z_flow_evolved': z_t.cpu().numpy(),  # The actual flow-evolved coordinates
                        'z_reparam_samples': z_samples_reparam.cpu().numpy(),
                        'z_flow_neighborhood': z_samples_around_flow.cpu().numpy(),  # Samples around flow coordinates
                        'flow_intermediate': flow_intermediate,
                        'metric_properties': metric_properties,
                        'original_images': x_t.cpu().numpy()
                    }
                    
                    print(f"   ✅ Timestep {t} data collected (metrics: {len(metric_properties)} properties)")
                
                # Create the visualizations using properly computed metrics
                print(f"🎨 Creating visualizations with metrics for {len(timestep_data)} timesteps...")
                
                # Enhanced PCA analysis
                self._create_enhanced_pca_analysis(timestep_data, epoch)
                
                # Enhanced manifold heatmaps with flow-evolved metrics
                self._create_enhanced_manifold_heatmaps(timestep_data, epoch)
                
                # Temporal metric evolution analysis
                self._create_temporal_metric_analysis(timestep_data, epoch)
                
                print(f"✨ Manifold visualizations complete for epoch {epoch}")
            
        except Exception as e:
            print(f"⚠️ Manifold visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.model.train()
    
    def create_pca_analysis(self, x_sample: torch.Tensor, epoch: int):
        """Create PCA-based manifold analysis."""
        print(f"📈 Creating PCA analysis for epoch {epoch}")
        
        # This is handled as part of create_metric_heatmaps
        # to avoid code duplication and ensure consistency
        pass
        
    def create_temporal_analysis(self, x_sample: torch.Tensor, epoch: int):
        """Create temporal evolution analysis of the manifold."""
        print(f"⏱️ Creating temporal manifold analysis for epoch {epoch}")
        
        # This is handled as part of create_metric_heatmaps
        # to avoid code duplication and ensure consistency
        pass
    
    def _create_enhanced_pca_analysis(self, timestep_data, epoch):
        """Enhanced PCA analysis using flow-evolved coordinates."""
        print("📈 Creating enhanced PCA analysis with flow-evolved coordinates...")
        
        n_timesteps = len(timestep_data)
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))  # SMALLER
        fig.suptitle(f'Enhanced PCA Analysis (Flow-Evolved Coords) - Epoch {epoch}', fontsize=14)
        
        # Collect all flow-evolved coordinates for global PCA
        all_z_flow = np.concatenate([data['z_flow_evolved'] for data in timestep_data.values()], axis=0)
        all_z_reparam = np.concatenate([data['z_reparam_samples'][:100] for data in timestep_data.values()], axis=0)
        
        # Fit PCA on flow-evolved coordinates
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(all_z_flow)
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_timesteps))
        
        # Plot 1: Flow-evolved coordinates in PCA space
        ax = axes[0, 0]
        for t, data in timestep_data.items():
            z_pca = pca.transform(data['z_flow_evolved'])
            ax.scatter(z_pca[:, 0], z_pca[:, 1], c=[colors[t]], alpha=0.7, s=30, label=f't={t}')
        ax.set_title('Flow-Evolved Coordinates')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Metric determinant evolution
        ax = axes[0, 1]
        timesteps_with_metrics = []
        det_means = []
        det_stds = []
        
        for t, data in timestep_data.items():
            if 'det_G_inv' in data['metric_properties']:
                det_values = data['metric_properties']['det_G_inv']
                timesteps_with_metrics.append(t)
                det_means.append(np.mean(det_values))
                det_stds.append(np.std(det_values))
        
        if timesteps_with_metrics:
            ax.errorbar(timesteps_with_metrics, det_means, yerr=det_stds, 
                       marker='o', capsize=5, capthick=2, linewidth=2)
            ax.set_title('Metric Determinant Evolution')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Mean det(G⁻¹)')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No metric data\navailable', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Metric Determinant Evolution')
        
        # Save the rest as simplified versions
        for i in range(1, 9):
            row, col = divmod(i, 3)
            if row < 3 and col < 3 and (row != 0 or col < 2):
                ax = axes[row, col]
                ax.text(0.5, 0.5, f'Analysis {i+1}\n(Simplified)', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'Analysis Panel {i+1}')
        
        plt.tight_layout()
        filename = f'enhanced_pca_analysis_epoch_{epoch}.png'
        saved_file = self._safe_save_plt_figure(filename, dpi=150, bbox_inches='tight')
        
        # Log to WandB
        if self.should_log_to_wandb() and saved_file:
            wandb.log({"manifold/enhanced_pca_analysis": wandb.Image(saved_file, caption=f"Epoch {epoch} - Enhanced PCA")})
        
        plt.close()
        print(f"✅ Enhanced PCA analysis created")
    
    def _create_enhanced_manifold_heatmaps(self, timestep_data, epoch):
        """Create enhanced manifold heatmaps with flow-evolved metrics."""
        print("🌍 Creating enhanced manifold heatmaps with flow-evolved metrics...")
        
        n_timesteps = len(timestep_data)
        
        # Show first few timesteps
        timesteps_to_show = list(range(min(4, n_timesteps)))
        fig, axes = plt.subplots(3, len(timesteps_to_show), figsize=(4*len(timesteps_to_show), 12))  # SMALLER
        if len(timesteps_to_show) == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'Enhanced Manifold Heatmaps (Flow-Evolved Metrics) - Epoch {epoch}', fontsize=14)
        
        # Prepare global PCA
        all_z_flow = np.concatenate([timestep_data[t]['z_flow_evolved'] for t in timesteps_to_show], axis=0)
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(all_z_flow)
        
        for idx, t in enumerate(timesteps_to_show):
            data = timestep_data[t]
            
            # Row 1: Flow-evolved coordinates
            ax = axes[0, idx]
            z_pca = pca.transform(data['z_flow_evolved'])
            scatter = ax.scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.7, s=20)
            ax.set_title(f'Flow-Evolved Coords t={t}')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.grid(True, alpha=0.3)
            
            # Row 2: Metric determinant field
            ax = axes[1, idx]
            if 'det_G_inv' in data['metric_properties'] and 'z_flow_neighborhood' in data:
                z_neighborhood = data['z_flow_neighborhood']
                det_values = data['metric_properties']['det_G_inv']
                
                # Limit points for performance
                max_points = 200
                if len(z_neighborhood) > max_points:
                    indices = np.random.choice(len(z_neighborhood), max_points, replace=False)
                    z_neighborhood = z_neighborhood[indices]
                    det_values = det_values[indices]
                
                z_neigh_pca = pca.transform(z_neighborhood)
                
                # Create scatter plot with det(G^-1) values
                scatter = ax.scatter(z_neigh_pca[:, 0], z_neigh_pca[:, 1], 
                                   c=det_values, cmap='plasma', s=15, alpha=0.8,
                                   vmin=np.percentile(det_values, 5),
                                   vmax=np.percentile(det_values, 95))
                ax.set_title(f'Metric det(G⁻¹) t={t}')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                try:
                    plt.colorbar(scatter, ax=ax, shrink=0.6)
                except:
                    pass
            else:
                ax.text(0.5, 0.5, 'No metric data\navailable', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'Metric det(G⁻¹) t={t}')
                
            # Row 3: Log metric field for better visualization
            ax = axes[2, idx]
            if 'det_G_inv' in data['metric_properties'] and 'z_flow_neighborhood' in data:
                z_neighborhood = data['z_flow_neighborhood']
                det_values = data['metric_properties']['det_G_inv']
                
                # Limit points for performance
                max_points = 200
                if len(z_neighborhood) > max_points:
                    indices = np.random.choice(len(z_neighborhood), max_points, replace=False)
                    z_neighborhood = z_neighborhood[indices]
                    det_values = det_values[indices]
                
                z_neigh_pca = pca.transform(z_neighborhood)
                log_det_values = np.log10(np.clip(det_values, 1e-12, None))
                
                scatter = ax.scatter(z_neigh_pca[:, 0], z_neigh_pca[:, 1], 
                                   c=log_det_values, cmap='viridis', s=15, alpha=0.8)
                ax.set_title(f'Log₁₀ det(G⁻¹) t={t}')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                try:
                    plt.colorbar(scatter, ax=ax, shrink=0.6)
                except:
                    pass
            else:
                ax.text(0.5, 0.5, 'No metric data\navailable', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'Log₁₀ det(G⁻¹) t={t}')
        
        plt.tight_layout()
        filename = f'enhanced_manifold_heatmaps_epoch_{epoch}.png'
        saved_file = self._safe_save_plt_figure(filename, dpi=150, bbox_inches='tight')
        
        if self.should_log_to_wandb() and saved_file:
            wandb.log({"manifold/enhanced_heatmaps": wandb.Image(saved_file, caption=f"Epoch {epoch} - Manifold heatmaps")})
        
        plt.close()
        print(f"✅ Enhanced manifold heatmaps created")
    
    def _create_temporal_metric_analysis(self, timestep_data, epoch):
        """Create temporal evolution analysis of metrics."""
        print("⏱️ Creating temporal metric evolution analysis...")
        
        n_timesteps = len(timestep_data)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # SMALLER
        fig.suptitle(f'Temporal Metric Evolution Analysis - Epoch {epoch}', fontsize=14)
        
        # Extract timesteps with valid metric data
        timesteps_with_metrics = []
        det_means = []
        det_stds = []
        condition_numbers = []
        
        for t, data in timestep_data.items():
            if 'det_G_inv' in data['metric_properties']:
                det_values = data['metric_properties']['det_G_inv']
                timesteps_with_metrics.append(t)
                det_means.append(np.mean(det_values))
                det_stds.append(np.std(det_values))
                condition_numbers.append(data['metric_properties'].get('condition_number', np.nan))
        
        # Plot 1: Mean det(G^-1) evolution
        ax = axes[0, 0]
        if timesteps_with_metrics:
            ax.errorbar(timesteps_with_metrics, det_means, yerr=det_stds, 
                       marker='o', capsize=5, capthick=2, linewidth=2, color='blue')
            ax.set_title('Mean det(G⁻¹) Evolution')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Mean det(G⁻¹)')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No metric data\navailable', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Mean det(G⁻¹) Evolution')
        
        # Plot 2: Condition number evolution
        ax = axes[0, 1]
        if timesteps_with_metrics and not all(np.isnan(condition_numbers)):
            ax.plot(timesteps_with_metrics, condition_numbers, 'o-', linewidth=2, markersize=8, color='red')
            ax.set_title('Condition Number Evolution')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Condition Number')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No condition\nnumber data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Condition Number Evolution')
        
        # Plot 3: Metric variance evolution
        ax = axes[1, 0]
        if timesteps_with_metrics:
            variances = [std**2 for std in det_stds]
            ax.plot(timesteps_with_metrics, variances, 'o-', linewidth=2, markersize=8, color='green')
            ax.set_title('Metric Variance Evolution')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Var(det(G⁻¹))')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No variance\ndata available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Metric Variance Evolution')
        
        # Plot 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create summary text
        summary_text = f"TEMPORAL METRIC ANALYSIS\n"
        summary_text += "=" * 25 + "\n\n"
        summary_text += f"Total timesteps: {n_timesteps}\n"
        summary_text += f"Timesteps with metrics: {len(timesteps_with_metrics)}\n\n"
        
        if timesteps_with_metrics:
            summary_text += f"Metric Statistics:\n"
            summary_text += f"  Mean det(G⁻¹): {np.mean(det_means):.2e}\n"
            summary_text += f"  Range: [{np.min(det_means):.2e}, {np.max(det_means):.2e}]\n"
            if not all(np.isnan(condition_numbers)):
                summary_text += f"  Mean condition #: {np.nanmean(condition_numbers):.2f}\n"
            summary_text += f"  Mean variance: {np.mean([std**2 for std in det_stds]):.2e}\n"
        else:
            summary_text += "No metric data available\n"
        
        summary_text += f"\nCoordinates: Flow-evolved\n"
        summary_text += f"Epoch: {epoch}\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", alpha=0.8))
        
        plt.tight_layout()
        filename = f'temporal_metric_analysis_epoch_{epoch}.png'
        saved_file = self._safe_save_plt_figure(filename, dpi=150, bbox_inches='tight')
        
        if self.should_log_to_wandb() and saved_file:
            wandb.log({
                "manifold/temporal_analysis": wandb.Image(saved_file, caption=f"Epoch {epoch} - Temporal analysis"),
                "metrics/mean_det_G_inv": np.mean(det_means) if det_means else 0,
                "metrics/mean_condition_number": np.nanmean(condition_numbers) if condition_numbers else 0,
            })
        
        plt.close()
        print(f"✅ Temporal metric analysis created") 