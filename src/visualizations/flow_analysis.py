"""
Flow Analysis Visualizations Module
==================================

Flow Jacobian analysis and temporal evolution:
- Flow-based temporal plots
- Jacobian determinant evolution using built-in flow methods
- Flow stability and volume preservation analysis
- Interactive temporal animations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from .base import BaseVisualization

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class FlowAnalysisVisualizations(BaseVisualization):
    """Flow Jacobian and temporal evolution visualization suite with enhanced metrics."""
    
    def create_temporal_evolution(self, x_sample: torch.Tensor, epoch: int):
        """Create flow-based temporal metric evolution visualizations."""
        print(f"🌊 Creating enhanced temporal evolution analysis for epoch {epoch}")
        
        # Check for metric tensor and flows in both legacy and modular structures
        has_metric = (hasattr(self.model, 'G') and self.model.G is not None) or \
                     (hasattr(self.model, 'metric_tensor') and self.model.metric_tensor is not None)
        has_flows = self._get_flows() is not None
        
        if not has_metric or not has_flows:
            print("⚠️ No metric tensor or flows available for temporal evolution")
            return
            
        try:
            self.model.eval()
            with torch.no_grad():
                result = self.model_forward(x_sample)
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
                
                # Create PCA projection
                z_pca_seq, pca = self._prepare_pca_data(z_seq, n_components=2)
                
                # Compute enhanced flow metrics
                print("🔬 Computing enhanced flow metrics...")
                det_G_seq = self._compute_flow_evolved_det_G(z_seq)
                flow_jacobians = self._compute_flow_jacobians_enhanced(z_seq)
                flow_stability = self._compute_flow_stability_metrics(z_seq)
                volume_preservation = self._compute_volume_preservation_metrics(flow_jacobians)
                
                # Create comprehensive flow-based visualization
                self._create_enhanced_flow_temporal_plots(z_pca_seq, det_G_seq, flow_jacobians, 
                                                        flow_stability, volume_preservation, epoch, pca)
                
        except Exception as e:
            print(f"⚠️ Temporal evolution visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.model.train()
        
    def create_jacobian_analysis(self, x_sample: torch.Tensor, epoch: int):
        """Create enhanced flow Jacobian analysis visualizations."""
        print(f"📊 Creating enhanced flow Jacobian analysis for epoch {epoch}")
        
        flows = self._get_flows()
        if flows is None:
            print("⚠️ No flows available for Jacobian analysis")
            return
            
        try:
            self.model.eval()
            with torch.no_grad():
                result = self.model_forward(x_sample)
                # Handle both dict format and ModelOutput format
                if isinstance(result, dict):
                    z_seq = result.get('latent_samples', result.get('z', None))
                else:
                    # ModelOutput object
                    z_seq = result.z if hasattr(result, 'z') else None
                
                if z_seq is None:
                    print("⚠️ Could not extract latent samples from model output")
                    return
                
                # Compute enhanced flow metrics
                print("🔬 Computing enhanced Jacobian metrics...")
                det_G_seq = self._compute_flow_evolved_det_G(z_seq)
                flow_jacobians = self._compute_flow_jacobians_enhanced(z_seq)
                flow_stability = self._compute_flow_stability_metrics(z_seq)
                
                # Create detailed enhanced Jacobian analysis
                self._create_enhanced_jacobian_analysis(det_G_seq, flow_jacobians, flow_stability, epoch)
                
        except Exception as e:
            print(f"⚠️ Enhanced Jacobian analysis visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.model.train()
    
    def _compute_flow_evolved_det_G(self, z_seq):
        """Compute det(G) at flow-evolved coordinates for each timestep."""
        batch_size, n_obs, latent_dim = z_seq.shape
        det_G_seq = torch.zeros(batch_size, n_obs)
        
        # Get metric tensor from either legacy or modular structure
        metric_tensor = None
        if hasattr(self.model, 'G') and self.model.G is not None:
            metric_tensor = self.model.G
        elif hasattr(self.model, 'metric_tensor') and self.model.metric_tensor is not None:
            metric_tensor = self.model.metric_tensor
        
        if metric_tensor is None:
            print("⚠️ No metric tensor available for det(G) computation")
            return det_G_seq.cpu().numpy()
        
        for t in range(n_obs):
            z_t = z_seq[:, t, :]  # Flow-evolved coordinates at timestep t
            G_t = metric_tensor(z_t)
            det_G_t = torch.linalg.det(G_t)
            det_G_seq[:, t] = det_G_t
        
        return det_G_seq.cpu().numpy()
    
    def _compute_flow_jacobians_enhanced(self, z_seq):
        """Enhanced Jacobian computation using built-in flow methods when available."""
        batch_size, n_obs, latent_dim = z_seq.shape
        flow_jacobians = []
        flows = self._get_flows()
        
        if flows is None:
            print("⚠️ No flows available for Jacobian computation")
            return []
        

        
        for flow_idx, flow in enumerate(flows):
            if flow_idx >= n_obs - 1:
                break
                
            try:
                z_input = z_seq[:, flow_idx, :].clone().detach().requires_grad_(True)
                
                # Try to use built-in log_abs_det_jac method first (IAF flows have this)
                if hasattr(flow, 'log_abs_det_jac'):
                    print(f"✅ Using built-in log_abs_det_jac for flow {flow_idx}")
                    # Apply the flow
                    flow_result = flow(z_input)
                    if hasattr(flow_result, 'out'):
                        z_output = flow_result.out
                    else:
                        z_output = flow_result
                    
                    # Compute log absolute determinant of Jacobian
                    log_abs_det_jac = flow.log_abs_det_jac(z_input, z_output)
                    jac_det = torch.exp(log_abs_det_jac).detach()
                
                else:
                    # Fallback to manual Jacobian computation

                    jac_det = torch.zeros(batch_size, device=z_input.device)
                    
                    for i in range(batch_size):
                        try:
                            # Use functional Jacobian computation
                            jac_matrix = torch.autograd.functional.jacobian(
                                lambda x: flow(x.unsqueeze(0)).out.squeeze(0) if hasattr(flow(x.unsqueeze(0)), 'out') else flow(x.unsqueeze(0)).squeeze(0), 
                                z_input[i]
                            )
                            jac_det[i] = torch.abs(torch.linalg.det(jac_matrix))
                        except Exception as e:
                            print(f"⚠️ Jacobian computation failed for sample {i}, flow {flow_idx}: {e}")
                            jac_det[i] = 1.0  # Identity fallback
                
                flow_jacobians.append(jac_det.cpu().numpy())
                print(f"✅ Flow {flow_idx}: Jacobian det range [{jac_det.min():.3e}, {jac_det.max():.3e}]")
                
            except Exception as e:
                print(f"⚠️ Enhanced Jacobian computation failed for flow {flow_idx}: {e}")
                flow_jacobians.append(np.ones(batch_size))
        
        return flow_jacobians
    
    def _compute_flow_stability_metrics(self, z_seq):
        """Compute flow stability metrics including eigenvalue analysis."""
        flows = self._get_flows()
        if flows is None:
            return {}
        
        batch_size, n_obs, latent_dim = z_seq.shape
        stability_metrics = {}
        

        
        for flow_idx, flow in enumerate(flows):
            if flow_idx >= n_obs - 1:
                break
                
            try:
                z_input = z_seq[:, flow_idx, :].clone().detach().requires_grad_(True)
                
                # Compute Jacobian eigenvalues for stability analysis
                eigenvalues = []
                condition_numbers = []
                
                for i in range(min(batch_size, 8)):  # Limit for performance
                    try:
                        # Compute full Jacobian matrix
                        def flow_func(x):
                            result = flow(x.unsqueeze(0))
                            return result.out.squeeze(0) if hasattr(result, 'out') else result.squeeze(0)
                        
                        jac_matrix = torch.autograd.functional.jacobian(flow_func, z_input[i])
                        
                        # Compute eigenvalues
                        eigvals = torch.linalg.eigvals(jac_matrix)
                        eigenvalues.append(eigvals.cpu().numpy())
                        
                        # Compute condition number
                        cond_num = torch.linalg.cond(jac_matrix).item()
                        condition_numbers.append(cond_num)
                        
                    except Exception as e:
                        print(f"⚠️ Stability analysis failed for sample {i}, flow {flow_idx}: {e}")
                        eigenvalues.append(np.ones(latent_dim, dtype=complex))
                        condition_numbers.append(1.0)
                
                stability_metrics[flow_idx] = {
                    'eigenvalues': eigenvalues,
                    'condition_numbers': condition_numbers,
                    'mean_condition_number': np.mean(condition_numbers),
                    'max_eigenvalue_magnitude': np.max([np.abs(ev).max() for ev in eigenvalues])
                }
                
                print(f"✅ Flow {flow_idx}: Mean condition number: {np.mean(condition_numbers):.2f}")
                
            except Exception as e:
                print(f"⚠️ Flow stability computation failed for flow {flow_idx}: {e}")
                stability_metrics[flow_idx] = {
                    'eigenvalues': [],
                    'condition_numbers': [1.0],
                    'mean_condition_number': 1.0,
                    'max_eigenvalue_magnitude': 1.0
                }
        
        return stability_metrics
    
    def _compute_volume_preservation_metrics(self, flow_jacobians):
        """Compute volume preservation metrics from Jacobian determinants."""
        if not flow_jacobians:
            return {}
        
        print("📏 Computing volume preservation metrics...")
        
        volume_metrics = {}
        
        for flow_idx, jac_dets in enumerate(flow_jacobians):
            # Volume expansion/contraction analysis
            log_jac_dets = np.log(np.clip(jac_dets, 1e-12, None))
            
            volume_metrics[flow_idx] = {
                'mean_log_det': np.mean(log_jac_dets),
                'std_log_det': np.std(log_jac_dets),
                'volume_expansion_ratio': np.mean(jac_dets),
                'volume_preservation_score': 1.0 - np.abs(np.mean(log_jac_dets)),  # Closer to 1 is better
                'max_volume_change': np.max(np.abs(log_jac_dets))
            }
            
            print(f"✅ Flow {flow_idx}: Volume expansion ratio: {np.mean(jac_dets):.3f}, Preservation score: {volume_metrics[flow_idx]['volume_preservation_score']:.3f}")
        
        return volume_metrics
    
    def _create_enhanced_flow_temporal_plots(self, z_pca_seq, det_G_seq, flow_jacobians, 
                                           flow_stability, volume_preservation, epoch, pca):
        """Create enhanced comprehensive flow-based temporal visualization."""
        try:
            batch_size, n_obs, _ = z_pca_seq.shape
            
            # Create enhanced visualization with more panels
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            fig.suptitle(f'Enhanced Flow Analysis - Epoch {epoch}', fontsize=16)
            
            colors = plt.get_cmap("tab10")(np.linspace(0, 1, min(batch_size, 8)))
            timesteps = np.linspace(0, n_obs-1, min(4, n_obs), dtype=int)
            
            # Row 1: Flow-evolved det(G) spatial distribution
            for idx, t in enumerate(timesteps[:3]):
                ax = axes[0, idx]
                
                x_coords = z_pca_seq[:, t, 0]
                y_coords = z_pca_seq[:, t, 1]
                det_values = det_G_seq[:, t]
                
                scatter = ax.scatter(x_coords, y_coords, c=det_values, s=40,
                                   cmap='viridis', alpha=0.8, edgecolors='white',
                                   vmin=det_G_seq.min(), vmax=det_G_seq.max())
                
                # Draw trajectories up to this timestep
                for seq_idx in range(min(batch_size, 6)):
                    traj = z_pca_seq[seq_idx, :t+1, :]
                    ax.plot(traj[:, 0], traj[:, 1], color=colors[seq_idx], 
                           linewidth=1.2, alpha=0.6)
                
                ax.set_title(f'Flow-evolved det(G) at t={t}', fontsize=11)
                ax.set_xlabel('PC1', fontsize=10)
                ax.set_ylabel('PC2', fontsize=10)
                ax.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax, shrink=0.7)
            
            # Row 2: Volume preservation and Jacobian analysis
            if flow_jacobians:
                # Plot 1: Jacobian determinants over flows
                ax = axes[1, 0]
                flow_indices = list(range(len(flow_jacobians)))
                for i, jac_dets in enumerate(flow_jacobians):
                    ax.boxplot(jac_dets, positions=[i], widths=0.6, patch_artist=True,
                              boxprops=dict(facecolor=colors[i % len(colors)], alpha=0.7))
                ax.set_xlabel('Flow Index')
                ax.set_ylabel('|det(J)|')
                ax.set_title('Jacobian Determinants')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
                
                # Plot 2: Volume preservation metrics
                ax = axes[1, 1]
                if volume_preservation:
                    flows_vp = list(volume_preservation.keys())
                    expansion_ratios = [volume_preservation[f]['volume_expansion_ratio'] for f in flows_vp]
                    preservation_scores = [volume_preservation[f]['volume_preservation_score'] for f in flows_vp]
                    
                    ax.scatter(flows_vp, expansion_ratios, label='Expansion Ratio', s=60, alpha=0.8)
                    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Volume Preserving')
                    ax.set_xlabel('Flow Index')
                    ax.set_ylabel('Volume Expansion Ratio')
                    ax.set_title('Volume Preservation Analysis')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                # Plot 3: Flow stability analysis
                ax = axes[1, 2]
                if flow_stability:
                    flows_stab = list(flow_stability.keys())
                    condition_numbers = [flow_stability[f]['mean_condition_number'] for f in flows_stab]
                    max_eigenvals = [flow_stability[f]['max_eigenvalue_magnitude'] for f in flows_stab]
                    
                    ax.scatter(flows_stab, condition_numbers, label='Condition Number', s=60, alpha=0.8, color='orange')
                    ax2 = ax.twinx()
                    ax2.scatter(flows_stab, max_eigenvals, label='Max |λ|', s=60, alpha=0.8, color='purple', marker='^')
                    
                    ax.set_xlabel('Flow Index')
                    ax.set_ylabel('Condition Number', color='orange')
                    ax2.set_ylabel('Max |Eigenvalue|', color='purple')
                    ax.set_title('Flow Stability Metrics')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='upper left')
                    ax2.legend(loc='upper right')
            
            # Row 3: Enhanced temporal evolution analysis
            # Plot 1: det(G) evolution over time
            ax = axes[2, 0]
            for seq_idx in range(min(batch_size, 8)):
                ax.plot(range(n_obs), det_G_seq[seq_idx, :], 
                       color=colors[seq_idx], alpha=0.7, linewidth=1.5)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('det(G)')
            ax.set_title('Metric Determinant Evolution')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            # Plot 2: PCA trajectory evolution
            ax = axes[2, 1]
            for seq_idx in range(min(batch_size, 6)):
                traj = z_pca_seq[seq_idx, :, :]
                ax.plot(traj[:, 0], traj[:, 1], 'o-', color=colors[seq_idx], 
                       alpha=0.7, linewidth=1.5, markersize=4)
                # Mark start and end
                ax.scatter(traj[0, 0], traj[0, 1], color=colors[seq_idx], s=80, marker='s', edgecolor='black')
                ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[seq_idx], s=80, marker='*', edgecolor='black')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_title('Latent Trajectories (PCA)')
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Cumulative volume change
            ax = axes[2, 2]
            if flow_jacobians:
                cumulative_volume_change = np.zeros(len(flow_jacobians) + 1)
                for i, jac_dets in enumerate(flow_jacobians):
                    log_det_mean = np.mean(np.log(np.clip(jac_dets, 1e-12, None)))
                    cumulative_volume_change[i+1] = cumulative_volume_change[i] + log_det_mean
                
                ax.plot(range(len(cumulative_volume_change)), cumulative_volume_change, 
                       'o-', linewidth=2, markersize=6, color='red', alpha=0.8)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                ax.set_xlabel('Flow Step')
                ax.set_ylabel('Cumulative Log Volume Change')
                ax.set_title('Cumulative Volume Evolution')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save with higher quality
            filename = f'enhanced_flow_temporal_analysis_epoch_{epoch}.png'
            saved_file = self._safe_save_plt_figure(filename, dpi=200, bbox_inches='tight')
            
            if self.should_log_to_wandb() and saved_file:
                wandb.log({
                    "flow_analysis/enhanced_temporal": wandb.Image(saved_file, caption=f"Enhanced Flow Analysis - Epoch {epoch}")
                })
                
                # Log numerical metrics
                if volume_preservation:
                    avg_preservation = np.mean([v['volume_preservation_score'] for v in volume_preservation.values()])
                    wandb.log({"metrics/flow_volume_preservation": avg_preservation})
                
                if flow_stability:
                    avg_condition = np.mean([v['mean_condition_number'] for v in flow_stability.values()])
                    wandb.log({"metrics/flow_stability_condition": avg_condition})
            
            plt.close()
            print(f"✅ Enhanced flow temporal analysis created")
            
        except Exception as e:
            print(f"⚠️ Enhanced flow temporal plotting failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_enhanced_jacobian_analysis(self, det_G_seq, flow_jacobians, flow_stability, epoch):
        """Create enhanced detailed Jacobian analysis visualization."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Enhanced Flow Jacobian Analysis - Epoch {epoch}', fontsize=16)
            
            if not flow_jacobians:
                for ax in axes.flat:
                    ax.text(0.5, 0.5, 'No flow Jacobian\ndata available', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                plt.tight_layout()
                filename = f'enhanced_jacobian_analysis_epoch_{epoch}.png'
                saved_file = self._safe_save_plt_figure(filename, dpi=150, bbox_inches='tight')
                if self.should_log_to_wandb() and saved_file:
                    wandb.log({"flow_analysis/enhanced_jacobian": wandb.Image(saved_file, caption=f"Enhanced Jacobian Analysis - Epoch {epoch}")})
                plt.close()
                return
            
            # Plot 1: Jacobian determinant distributions
            ax = axes[0, 0]
            for i, jac_dets in enumerate(flow_jacobians):
                ax.hist(np.log10(np.clip(jac_dets, 1e-12, None)), bins=30, alpha=0.6, 
                       label=f'Flow {i}', density=True)
            ax.set_xlabel('log₁₀ |det(J)|')
            ax.set_ylabel('Density')
            ax.set_title('Jacobian Determinant Distributions')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Volume Preserving')
            
            # Plot 2: Flow-by-flow Jacobian comparison
            ax = axes[0, 1]
            flow_indices = list(range(len(flow_jacobians)))
            jac_means = [np.mean(jac_dets) for jac_dets in flow_jacobians]
            jac_stds = [np.std(jac_dets) for jac_dets in flow_jacobians]
            
            ax.errorbar(flow_indices, jac_means, yerr=jac_stds, 
                       marker='o', capsize=5, capthick=2, linewidth=2)
            ax.set_xlabel('Flow Index')
            ax.set_ylabel('Mean |det(J)|')
            ax.set_title('Flow Jacobian Comparison')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Stability analysis
            ax = axes[0, 2]
            if flow_stability:
                flows = list(flow_stability.keys())
                condition_nums = [flow_stability[f]['mean_condition_number'] for f in flows]
                max_eigs = [flow_stability[f]['max_eigenvalue_magnitude'] for f in flows]
                
                ax.scatter(condition_nums, max_eigs, s=80, alpha=0.8, c=flows, cmap='viridis')
                ax.set_xlabel('Mean Condition Number')
                ax.set_ylabel('Max |Eigenvalue|')
                ax.set_title('Flow Stability Analysis')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
                
                # Add stability regions
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Stable')
                ax.axvline(x=10.0, color='orange', linestyle='--', alpha=0.5, label='Well-conditioned')
                ax.legend()
            
            # Bottom row: Enhanced analysis
            # Plot 4: Cumulative Jacobian product
            ax = axes[1, 0]
            if len(flow_jacobians) > 1:
                cumulative_jac = np.ones_like(flow_jacobians[0])
                cum_products = [cumulative_jac.copy()]
                
                for jac_dets in flow_jacobians:
                    cumulative_jac *= jac_dets
                    cum_products.append(cumulative_jac.copy())
                
                for i, cum_prod in enumerate(cum_products):
                    ax.hist(np.log10(np.clip(cum_prod, 1e-12, None)), bins=20, alpha=0.6, 
                           label=f'After {i} flows', density=True)
                
                ax.set_xlabel('log₁₀ Cumulative |det(J)|')
                ax.set_ylabel('Density')
                ax.set_title('Cumulative Jacobian Evolution')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Plot 5: Volume preservation score over flows
            ax = axes[1, 1]
            if len(flow_jacobians) > 0:
                preservation_scores = []
                for jac_dets in flow_jacobians:
                    log_dets = np.log(np.clip(jac_dets, 1e-12, None))
                    score = 1.0 - np.abs(np.mean(log_dets))
                    preservation_scores.append(score)
                
                ax.plot(range(len(preservation_scores)), preservation_scores, 
                       'o-', linewidth=2, markersize=8)
                ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect Preservation')
                ax.set_xlabel('Flow Index')
                ax.set_ylabel('Volume Preservation Score')
                ax.set_title('Volume Preservation by Flow')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1.1)
            
            # Plot 6: Eigenvalue distribution (if available)
            ax = axes[1, 2]
            if flow_stability and any(flow_stability[f]['eigenvalues'] for f in flow_stability.keys()):
                all_eigenvals = []
                for flow_idx in flow_stability.keys():
                    for eigvals in flow_stability[flow_idx]['eigenvalues']:
                        all_eigenvals.extend(np.abs(eigvals))
                
                if all_eigenvals:
                    ax.hist(np.log10(np.clip(all_eigenvals, 1e-12, None)), bins=30, alpha=0.7, density=True)
                    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='|λ| = 1')
                    ax.set_xlabel('log₁₀ |Eigenvalue|')
                    ax.set_ylabel('Density')
                    ax.set_title('Flow Eigenvalue Distribution')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filename = f'enhanced_jacobian_analysis_epoch_{epoch}.png'
            saved_file = self._safe_save_plt_figure(filename, dpi=200, bbox_inches='tight')
            
            if self.should_log_to_wandb() and saved_file:
                wandb.log({
                    "flow_analysis/enhanced_jacobian": wandb.Image(saved_file, caption=f"Enhanced Jacobian Analysis - Epoch {epoch}")
                })
            
            plt.close()
            print(f"✅ Enhanced Jacobian analysis created")
            
        except Exception as e:
            print(f"⚠️ Enhanced Jacobian analysis plotting failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_flows(self):
        """Get flows from either legacy or modular model structure."""
        # Try legacy structure first
        if hasattr(self.model, 'flows') and self.model.flows is not None:
            return self.model.flows
        # Try modular structure
        elif hasattr(self.model, 'flow_manager') and hasattr(self.model.flow_manager, 'flows'):
            return self.model.flow_manager.flows
        else:
            return None
    
    def create_flow_jacobian_analysis(self, x_sample: torch.Tensor, epoch: int):
        """Create flow Jacobian analysis visualizations."""
        print(f"📊 Creating flow Jacobian analysis for epoch {epoch}")
        
        if self._get_flows() is None:
            print("⚠️ No flows available for Jacobian analysis")
            return
            
        try:
            self.model.eval()
            with torch.no_grad():
                result = self.model_forward(x_sample)
                # Handle both dict format and ModelOutput format
                if isinstance(result, dict):
                    z_seq = result.get('latent_samples', result.get('z', None))
                else:
                    # ModelOutput object
                    z_seq = result.z if hasattr(result, 'z') else None
                
                if z_seq is None:
                    print("⚠️ Could not extract latent samples from model output")
                    return
                
                # Compute det(G) evolution
                det_G_seq = self._compute_flow_evolved_det_G(z_seq)
                
                # Compute flow Jacobians
                flow_jacobians = self._compute_flow_jacobians(z_seq)
                
                # Create detailed Jacobian analysis
                self._create_detailed_jacobian_analysis(det_G_seq, flow_jacobians, epoch)
                
        except Exception as e:
            print(f"⚠️ Jacobian analysis visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.model.train()
