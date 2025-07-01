"""
Flow Analysis Visualizations Module
==================================

Flow Jacobian analysis and temporal evolution:
- Flow-based temporal plots
- Jacobian determinant evolution
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
    """Flow Jacobian and temporal evolution visualization suite."""
    
    def create_temporal_evolution(self, x_sample: torch.Tensor, epoch: int):
        """Create flow-based temporal metric evolution visualizations."""
        print(f"🌊 Creating temporal evolution analysis for epoch {epoch}")
        
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
                z_seq = result['latent_samples'] if isinstance(result, dict) else result.z
                
                batch_size, n_obs, latent_dim = z_seq.shape
                
                # Create PCA projection
                z_pca_seq, pca = self._prepare_pca_data(z_seq, n_components=2)
                
                # Compute det(G) at each timestep using flow-evolved coordinates
                det_G_seq = self._compute_flow_evolved_det_G(z_seq)
                
                # Compute flow Jacobians
                flow_jacobians = self._compute_flow_jacobians(z_seq)
                
                # Create comprehensive flow-based visualization
                self._create_flow_based_temporal_plots(z_pca_seq, det_G_seq, flow_jacobians, epoch, pca)
                
        except Exception as e:
            print(f"⚠️ Temporal evolution visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.model.train()
        
    def create_jacobian_analysis(self, x_sample: torch.Tensor, epoch: int):
        """Create flow Jacobian analysis visualizations."""
        print(f"📊 Creating flow Jacobian analysis for epoch {epoch}")
        
        if self._get_flows() is None:
            print("⚠️ No flows available for Jacobian analysis")
            return
            
        try:
            self.model.eval()
            with torch.no_grad():
                result = self.model_forward(x_sample)
                z_seq = result['latent_samples'] if isinstance(result, dict) else result.z
                
                # Compute det(G) evolution
                det_G_seq = self._compute_flow_evolved_det_G(z_seq)
                
                # Compute flow Jacobians
                flow_jacobians = self._compute_flow_jacobians(z_seq)
                
                # Create detailed Jacobian analysis
                self._create_detailed_jacobian_analysis(det_G_seq, flow_jacobians, epoch)
                
                # Create interactive animation if Plotly available
                if PLOTLY_AVAILABLE:
                    z_pca_seq, _ = self._prepare_pca_data(z_seq, n_components=2)
                    self._create_flow_interactive_animation(z_pca_seq, det_G_seq, flow_jacobians, epoch)
                
        except Exception as e:
            print(f"⚠️ Jacobian analysis visualization failed: {e}")
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
    
    def _compute_flow_jacobians(self, z_seq):
        """Compute Jacobian determinants for each flow step."""
        batch_size, n_obs, latent_dim = z_seq.shape
        flow_jacobians = []
        
        for flow_idx in range(min(n_obs - 1, len(self._get_flows() or []))):
            try:
                z_input = z_seq[:, flow_idx, :]  # Input to flow
                flows = self._get_flows()
                if flows is None:
                    break
                flow = flows[flow_idx]
                
                # Compute Jacobian determinant
                z_input.requires_grad_(True)
                flow_result = flow(z_input)
                z_output = flow_result.out
                
                # Compute Jacobian matrix and its determinant
                jac_det = torch.zeros(batch_size)
                for i in range(batch_size):
                    try:
                        jac_matrix = torch.autograd.functional.jacobian(
                            lambda x: flow(x.unsqueeze(0)).out.squeeze(0), 
                            z_input[i]
                        )
                        jac_det[i] = torch.linalg.det(jac_matrix)
                    except:
                        jac_det[i] = 1.0  # Fallback
                
                flow_jacobians.append(jac_det.detach())
                
            except Exception as e:
                print(f"⚠️ Jacobian computation failed for flow {flow_idx}: {e}")
                flow_jacobians.append(torch.ones(batch_size))
        
        return flow_jacobians
    
    def _create_flow_based_temporal_plots(self, z_pca_seq, det_G_seq, flow_jacobians, epoch, pca):
        """Create comprehensive flow-based temporal visualization - SMALLER SIZE."""
        try:
            batch_size, n_obs, _ = z_pca_seq.shape
            
            # Create smaller visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # SMALLER
            fig.suptitle(f'Flow-Based det(G) Evolution - Epoch {epoch}', fontsize=14)
            
            colors = plt.get_cmap("tab10")(np.linspace(0, 1, min(batch_size, 8)))
            timesteps = np.linspace(0, n_obs-1, min(6, n_obs), dtype=int)  # Fewer timesteps
            
            # Row 1: Flow-evolved det(G) spatial distribution
            for idx, t in enumerate(timesteps[:3]):  # Show only 3 timesteps
                ax = axes[0, idx]
                
                x_coords = z_pca_seq[:, t, 0]
                y_coords = z_pca_seq[:, t, 1]
                det_values = det_G_seq[:, t]
                
                scatter = ax.scatter(x_coords, y_coords, c=det_values, s=60,  # Smaller markers
                                   cmap='viridis', alpha=0.8, edgecolors='white',
                                   vmin=det_G_seq.min(), vmax=det_G_seq.max())
                
                # Draw trajectories up to this timestep (limit sequences)
                for seq_idx in range(min(batch_size, 6)):
                    traj = z_pca_seq[seq_idx, :t+1, :]
                    ax.plot(traj[:, 0], traj[:, 1], color=colors[seq_idx], 
                           linewidth=1.5, alpha=0.6)  # Thinner lines
                
                ax.set_title(f'Flow Step t={t}', fontsize=10)
                ax.set_xlabel('PC1', fontsize=9)
                ax.set_ylabel('PC2', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
                
                if idx == 0:
                    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('det(G)', fontsize=9)
                    cbar.ax.tick_params(labelsize=8)
            
            # Row 2: Temporal evolution plots
            
            # Plot 1: det(G) evolution over time (limit sequences)
            ax = axes[1, 0]
            timesteps_full = np.arange(n_obs)
            for seq_idx in range(min(batch_size, 6)):  # Limit sequences
                det_trajectory = det_G_seq[seq_idx, :]
                ax.plot(timesteps_full, det_trajectory, 'o-', color=colors[seq_idx], 
                       linewidth=1.5, markersize=4, alpha=0.8, label=f'Seq {seq_idx}')
            
            ax.set_xlabel('Timestep', fontsize=9)
            ax.set_ylabel('det(G)', fontsize=9)
            ax.set_title('det(G) Evolution', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=2)  # Smaller legend
            ax.set_yscale('log')
            ax.tick_params(labelsize=8)
            
            # Plot 2: Jacobian determinants over time
            ax = axes[1, 1]
            if flow_jacobians:
                jac_array = torch.stack(flow_jacobians, dim=0).cpu().numpy()
                
                for seq_idx in range(min(batch_size, 6)):  # Limit sequences
                    jac_trajectory = jac_array[:, seq_idx]
                    ax.plot(range(1, len(jac_trajectory)+1), jac_trajectory, 'o-', 
                           color=colors[seq_idx], linewidth=1.5, markersize=4, alpha=0.8)
                
                ax.set_xlabel('Flow Step', fontsize=9)
                ax.set_ylabel('|J_flow|', fontsize=9)
                ax.set_title('Flow Jacobian Evolution', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No Jacobian\nData Available', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
                ax.set_title('Flow Jacobian Evolution', fontsize=10)
            
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
            # Plot 3: Cumulative amplification
            ax = axes[1, 2]
            amplification = det_G_seq / det_G_seq[:, 0:1]
            
            mean_amp = np.mean(amplification, axis=0)
            std_amp = np.std(amplification, axis=0)
            
            ax.fill_between(timesteps_full, mean_amp - std_amp, mean_amp + std_amp, 
                           alpha=0.3, color='blue', label='±1 std')
            ax.plot(timesteps_full, mean_amp, 'o-', color='blue', linewidth=2, 
                   markersize=4, label='Mean amplification')
            
            ax.set_xlabel('Timestep', fontsize=9)
            ax.set_ylabel('det(G) Amplification', fontsize=9)
            ax.set_title('Cumulative Amplification', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_yscale('log')
            ax.tick_params(labelsize=8)
            
            plt.tight_layout()
            
            # Save visualization
            filename = f'flow_based_det_evolution_epoch_{epoch}.png'
            saved_file = self._safe_save_plt_figure(filename, dpi=150, bbox_inches='tight')
            
            # Log to WandB
            if self.should_log_to_wandb() and saved_file:
                wandb.log({
                    "flow_analysis/temporal_evolution": wandb.Image(saved_file, 
                        caption=f"Epoch {epoch} - Flow-based temporal evolution"),
                })
            
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Flow-based temporal plots failed: {e}")
    
    def _create_detailed_jacobian_analysis(self, det_G_seq, flow_jacobians, epoch):
        """Create detailed analysis of flow Jacobians - SMALLER SIZE."""
        try:
            if not flow_jacobians:
                return
                
            jac_array = torch.stack(flow_jacobians, dim=0).cpu().numpy()
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # SMALLER
            fig.suptitle(f'Flow Jacobian Analysis - Epoch {epoch}', fontsize=14)
            
            # Plot 1: Jacobian distribution at each flow step
            ax = axes[0, 0]
            for flow_idx in range(min(len(flow_jacobians), 4)):  # Limit flows shown
                jac_values = jac_array[flow_idx, :]
                ax.hist(jac_values, bins=15, alpha=0.6, label=f'Flow {flow_idx+1}', density=True)
            
            ax.set_xlabel('|J_flow|', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.set_title('Jacobian Distribution by Flow', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
            # Plot 2: Jacobian vs det(G) correlation
            ax = axes[0, 1]
            for flow_idx in range(min(len(flow_jacobians), 3)):  # Limit flows
                if flow_idx + 1 < det_G_seq.shape[1]:
                    jac_values = jac_array[flow_idx, :]
                    det_before = det_G_seq[:, flow_idx]
                    det_after = det_G_seq[:, flow_idx + 1]
                    
                    ax.scatter(jac_values, det_after / det_before, alpha=0.6, s=20,
                             label=f'Flow {flow_idx+1}')
            
            ax.set_xlabel('|J_flow|', fontsize=9)
            ax.set_ylabel('det(G) Ratio', fontsize=9)
            ax.set_title('Jacobian vs det(G) Amplification', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
            # Plot 3: Cumulative Jacobian product
            ax = axes[1, 0]
            cumulative_jac = np.ones((jac_array.shape[1], len(flow_jacobians) + 1))
            for flow_idx in range(len(flow_jacobians)):
                cumulative_jac[:, flow_idx + 1] = cumulative_jac[:, flow_idx] * jac_array[flow_idx, :]
            
            mean_cum_jac = np.mean(cumulative_jac, axis=0)
            std_cum_jac = np.std(cumulative_jac, axis=0)
            timesteps = np.arange(len(mean_cum_jac))
            
            ax.fill_between(timesteps, mean_cum_jac - std_cum_jac, mean_cum_jac + std_cum_jac,
                           alpha=0.3, color='green')
            ax.plot(timesteps, mean_cum_jac, 'o-', color='green', linewidth=2, 
                   markersize=4, label='Mean')
            
            ax.set_xlabel('Flow Step', fontsize=9)
            ax.set_ylabel('Cumulative |J| Product', fontsize=9)
            ax.set_title('Cumulative Jacobian Product', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            ax.tick_params(labelsize=8)
            
            # Plot 4: Flow impact analysis
            ax = axes[1, 1]
            impact_ratios = []
            for flow_idx in range(len(flow_jacobians)):
                jac_values = jac_array[flow_idx, :]
                impact = np.std(jac_values) / (np.mean(jac_values) + 1e-10)
                impact_ratios.append(impact)
            
            ax.bar(range(1, len(impact_ratios) + 1), impact_ratios, alpha=0.7, color='orange')
            ax.set_xlabel('Flow Step', fontsize=9)
            ax.set_ylabel('Jacobian Variability (CV)', fontsize=9)
            ax.set_title('Flow Impact Analysis', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
            plt.tight_layout()
            
            # Save Jacobian analysis
            jac_filename = f'flow_jacobian_analysis_epoch_{epoch}.png'
            saved_file = self._safe_save_plt_figure(jac_filename, dpi=150, bbox_inches='tight')
            
            # Log to WandB
            if self.should_log_to_wandb() and saved_file:
                wandb.log({
                    "flow_analysis/jacobian_analysis": wandb.Image(saved_file, 
                        caption=f"Epoch {epoch} - Flow Jacobian analysis"),
                })
            
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Detailed Jacobian analysis failed: {e}")
    
    def _create_flow_interactive_animation(self, z_pca_seq, det_G_seq, flow_jacobians, epoch):
        """Create interactive animation of flow evolution - SMALLER SIZE."""
        if not PLOTLY_AVAILABLE:
            return
            
        try:
            batch_size, n_obs, _ = z_pca_seq.shape
            
            # Create interactive animation - SMALLER SIZE
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=["🌊 Flow Spatial Evolution", "📊 det(G) Evolution",
                               "⚡ Flow Jacobians", "🔗 Cumulative Amplification"],
                horizontal_spacing=0.12,
                vertical_spacing=0.15
            )
            
            # Prepare data for animation frames
            frames = []
            colors = px.colors.qualitative.Set3[:min(batch_size, 6)]  # Limit sequences
            
            for t in range(n_obs):
                frame_data = []
                
                # Panel 1: Spatial distribution with det(G) coloring
                frame_data.append(
                    go.Scatter(
                        x=z_pca_seq[:min(batch_size, 6), t, 0],  # Limit sequences
                        y=z_pca_seq[:min(batch_size, 6), t, 1],
                        mode='markers',
                        marker=dict(
                            size=8,  # Smaller markers
                            color=det_G_seq[:min(batch_size, 6), t],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="det(G)", x=0.45, len=0.4),
                            line=dict(color='white', width=1)
                        ),
                        name="Sequences",
                        showlegend=False,
                        xaxis='x', yaxis='y'
                    )
                )
                
                # Add trajectory lines (limit sequences)
                for seq_idx in range(min(batch_size, 4)):
                    traj = z_pca_seq[seq_idx, :t+1, :]
                    frame_data.append(
                        go.Scatter(
                            x=traj[:, 0],
                            y=traj[:, 1],
                            mode='lines',
                            line=dict(color=colors[seq_idx], width=1.5),  # Thinner lines
                            name=f'Traj {seq_idx}',
                            showlegend=(t == 0 and seq_idx < 3),
                            xaxis='x', yaxis='y'
                        )
                    )
                
                # Panel 2: det(G) evolution up to timestep t
                for seq_idx in range(min(batch_size, 4)):  # Limit sequences
                    det_so_far = det_G_seq[seq_idx, :t+1]
                    timesteps_so_far = np.arange(t+1)
                    
                    frame_data.append(
                        go.Scatter(
                            x=timesteps_so_far,
                            y=det_so_far,
                            mode='lines+markers',
                            line=dict(color=colors[seq_idx], width=1.5),
                            marker=dict(size=4, color=colors[seq_idx]),
                            name=f'det(G) {seq_idx}',
                            showlegend=False,
                            xaxis='x2', yaxis='y2'
                        )
                    )
                
                # Panel 3: Flow Jacobians (if available)
                if t > 0 and t-1 < len(flow_jacobians):
                    jac_values = flow_jacobians[t-1].cpu().numpy()[:min(batch_size, 6)]  # Limit
                    
                    frame_data.append(
                        go.Bar(
                            x=list(range(len(jac_values))),
                            y=jac_values,
                            name="Jacobians",
                            showlegend=False,
                            marker_color='rgba(255, 100, 102, 0.7)',
                            xaxis='x3', yaxis='y3'
                        )
                    )
                
                # Panel 4: Cumulative amplification
                amplification = det_G_seq / det_G_seq[:, 0:1]
                mean_amp = np.mean(amplification[:min(batch_size, 6), :t+1], axis=0)  # Limit
                
                frame_data.append(
                    go.Scatter(
                        x=np.arange(t+1),
                        y=mean_amp,
                        mode='lines+markers',
                        line=dict(color='red', width=2),
                        marker=dict(size=4, color='red'),
                        name="Mean Amplification",
                        showlegend=False,
                        xaxis='x4', yaxis='y4'
                    )
                )
                
                frames.append(go.Frame(data=frame_data, name=str(t)))
            
            # Set initial frame
            fig.add_traces(frames[0].data)
            fig.frames = frames
            
            # Update layout - SMALLER SIZE
            fig.update_layout(
                title=f"🌊 Flow Evolution Animation - Epoch {epoch}",
                sliders=[{
                    "active": 0,
                    "currentvalue": {"prefix": "Flow Step: ", "visible": True},
                    "pad": {"b": 10, "t": 50},
                    "steps": [{"args": [[f], {"frame": {"duration": 400, "redraw": True}}], 
                             "label": str(t), "method": "animate"} 
                             for t, f in enumerate(frames)]
                }],
                width=1000,  # SMALLER
                height=700,  # SMALLER
                showlegend=True
            )
            
            # Save interactive HTML
            html_filename = f'flow_evolution_animation_epoch_{epoch}.html'
            html_path = self._get_output_path(html_filename, "interactive")
            fig.write_html(html_path, include_plotlyjs=True)
            print(f"💾 Saved flow evolution animation: {html_path}")
            
            # Save static version
            png_filename = f'flow_evolution_animation_epoch_{epoch}.png'
            saved_png = self._safe_write_image(fig, png_filename, width=1000, height=700)
            
            # Log to WandB
            if self.should_log_to_wandb():
                log_dict = {"flow_analysis/interactive_animation": wandb.Html(html_path)}
                if saved_png and saved_png.endswith('.png'):
                    log_dict["flow_analysis/animation_static"] = wandb.Image(saved_png)
                wandb.log(log_dict)
            
        except Exception as e:
            print(f"⚠️ Flow interactive animation failed: {e}")

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
                z_seq = result['latent_samples'] if isinstance(result, dict) else result.z
                
                # Compute det(G) evolution
                det_G_seq = self._compute_flow_evolved_det_G(z_seq)
                
                # Compute flow Jacobians
                flow_jacobians = self._compute_flow_jacobians(z_seq)
                
                # Create detailed Jacobian analysis
                self._create_detailed_jacobian_analysis(det_G_seq, flow_jacobians, epoch)
                
                # Create interactive animation if Plotly available
                if PLOTLY_AVAILABLE:
                    z_pca_seq, _ = self._prepare_pca_data(z_seq, n_components=2)
                    self._create_flow_interactive_animation(z_pca_seq, det_G_seq, flow_jacobians, epoch)
                
        except Exception as e:
            print(f"⚠️ Jacobian analysis visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.model.train()
