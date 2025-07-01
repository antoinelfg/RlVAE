"""
Interactive Visualizations Module
================================

Advanced Plotly-based interactive visualizations:
- Geodesic slider visualizations
- Fancy interactive plots
- Animated metric evolution
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import wandb
from .base import BaseVisualization

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.colors as pc
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class InteractiveVisualizations(BaseVisualization):
    """Interactive Plotly-based visualization suite."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - interactive visualizations will be skipped")
    
    def _ensure_model_on_device(self):
        """Ensure the entire model and all its components are on the correct device."""
        try:
            # First, ensure the model itself is on the device
            self.model = self.model.to(self.device)
            
            # Ensure encoder and decoder are on device
            if hasattr(self.model, 'encoder') and self.model.encoder is not None:
                self.model.encoder = self.model.encoder.to(self.device)
                
            if hasattr(self.model, 'decoder') and self.model.decoder is not None:
                self.model.decoder = self.model.decoder.to(self.device)
            
            # Ensure metric tensor G and its components are on device
            if hasattr(self.model, 'G') and self.model.G is not None:
                # Handle different types of G (function, module, tensor)
                if hasattr(self.model.G, 'to'):
                    self.model.G = self.model.G.to(self.device)
                elif hasattr(self.model.G, '__self__') and hasattr(self.model.G.__self__, 'to'):
                    # For bound methods, move the underlying object
                    self.model.G.__self__ = self.model.G.__self__.to(self.device)
            
            if hasattr(self.model, 'G_inv') and self.model.G_inv is not None:
                if hasattr(self.model.G_inv, 'to'):
                    self.model.G_inv = self.model.G_inv.to(self.device)
                elif hasattr(self.model.G_inv, '__self__') and hasattr(self.model.G_inv.__self__, 'to'):
                    self.model.G_inv.__self__ = self.model.G_inv.__self__.to(self.device)
            
            # Move flows to device if available
            if hasattr(self.model, 'flows') and self.model.flows is not None:
                if isinstance(self.model.flows, (list, nn.ModuleList)):
                    for i, flow in enumerate(self.model.flows):
                        if hasattr(flow, 'to'):
                            self.model.flows[i] = flow.to(self.device)
                elif hasattr(self.model.flows, 'to'):
                    self.model.flows = self.model.flows.to(self.device)
            elif hasattr(self.model, 'flow_manager') and hasattr(self.model.flow_manager, 'flows'):
                # Handle modular model structure
                flows = self.model.flow_manager.flows
                if isinstance(flows, (list, nn.ModuleList)):
                    for i, flow in enumerate(flows):
                        if hasattr(flow, 'to'):
                            flows[i] = flow.to(self.device)
                elif hasattr(flows, 'to'):
                    self.model.flow_manager.flows = flows.to(self.device)
            
            # Ensure any centroids/metric components are on device
            if hasattr(self.model, 'centroids') and self.model.centroids is not None:
                self.model.centroids = self.model.centroids.to(self.device)
                
            if hasattr(self.model, 'metric') and self.model.metric is not None:
                if hasattr(self.model.metric, 'to'):
                    self.model.metric = self.model.metric.to(self.device)
                if hasattr(self.model.metric, 'centroids') and self.model.metric.centroids is not None:
                    self.model.metric.centroids = self.model.metric.centroids.to(self.device)
            
            # Ensure any other tensor attributes are on device
            for attr_name in ['G_centroids', 'T', 'lbd', 'mu', 'sigma']:
                if hasattr(self.model, attr_name):
                    attr_value = getattr(self.model, attr_name)
                    if isinstance(attr_value, torch.Tensor):
                        setattr(self.model, attr_name, attr_value.to(self.device))
            
            # For RHVAE models, ensure any RHVAE-specific components are on device
            if hasattr(self.model, 'rhvae_sampler') and self.model.rhvae_sampler is not None:
                if hasattr(self.model.rhvae_sampler, 'to'):
                    self.model.rhvae_sampler = self.model.rhvae_sampler.to(self.device)
                    
            # For any model attribute that has named_parameters, ensure it's on device
            for attr_name in dir(self.model):
                if not attr_name.startswith('_'):
                    try:
                        attr = getattr(self.model, attr_name)
                        if hasattr(attr, 'named_parameters') and hasattr(attr, 'to'):
                            attr = attr.to(self.device)
                            setattr(self.model, attr_name, attr)
                    except:
                        continue
                        
        except Exception as e:
            print(f"⚠️ Warning: Could not move some model components to device: {e}")
    
    def _ensure_tensor_on_device(self, tensor):
        """Ensure a tensor is on the correct device."""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        return tensor
    
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
    
    def create_geodesic_sliders(self, x_sample: torch.Tensor, epoch: int):
        """Create interactive geodesic slider visualizations with timestep evolution."""
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - skipping geodesic sliders")
            return
            
        print(f"🎚️ Creating interactive geodesic sliders for epoch {epoch}")
        
        if not hasattr(self.model, 'G'):
            print("⚠️ No metric tensor available for geodesic visualization")
            return
            
        try:
            # Ensure entire model is on correct device
            self._ensure_model_on_device()
            
            self.model.eval()
            with torch.no_grad():
                result = self.model_forward(x_sample)
                z_seq = result['latent_samples'] if isinstance(result, dict) else result.z  # [batch_size, n_obs, latent_dim]
                
                batch_size, n_obs, latent_dim = z_seq.shape
                
                # Create PCA projection 
                z_pca_seq, pca = self._prepare_pca_data(z_seq, n_components=2)
                
                # Create smaller grid for better performance
                x_min, x_max = z_pca_seq[:, :, 0].min() - 0.5, z_pca_seq[:, :, 0].max() + 0.5
                y_min, y_max = z_pca_seq[:, :, 1].min() - 0.5, z_pca_seq[:, :, 1].max() + 0.5
                nx, ny = 30, 30  # Smaller grid for better performance
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
                
                # Create interactive slider visualization
                self._create_interactive_geodesic_slider(z_seq, z_pca_seq, xx, yy, pca, epoch)
                
        except Exception as e:
            print(f"⚠️ Geodesic slider visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.model.train()
        
    def create_fancy_geodesics(self, x_sample: torch.Tensor, epoch: int):
        """Create fancy interactive geodesic visualizations with dense trajectories."""
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - skipping fancy geodesics")
            return
            
        print(f"✨ Creating fancy interactive geodesic visualizations for epoch {epoch}")
        
        if not hasattr(self.model, 'G'):
            print("⚠️ No metric tensor available for fancy visualization")
            return
            
        try:
            # Ensure entire model is on correct device
            self._ensure_model_on_device()
            
            self.model.eval()
            with torch.no_grad():
                result = self.model_forward(x_sample)
                z_seq = result['latent_samples'] if isinstance(result, dict) else result.z  # [batch_size, n_obs, latent_dim]
                
                batch_size, n_obs, latent_dim = z_seq.shape
                
                # Generate dense trajectories with fewer interpolation points for performance
                dense_trajectories = self._generate_dense_trajectories(z_seq, n_interp_points=10)
                
                # Prepare PCA data
                z_flat = dense_trajectories.reshape(-1, latent_dim).cpu().numpy()
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                z_pca = pca.fit_transform(z_flat)
                dense_n_points = dense_trajectories.shape[1]
                z_pca_dense = z_pca.reshape(batch_size, dense_n_points, 2)
                
                # Original trajectory points in PCA space
                z_orig_flat = z_seq.reshape(-1, latent_dim).cpu().numpy()
                z_orig_pca = pca.transform(z_orig_flat).reshape(batch_size, n_obs, 2)
                
                # Create fancy interactive visualization
                self._create_fancy_interactive_plot(z_seq, z_pca_dense, z_orig_pca, pca, epoch)
                
        except Exception as e:
            print(f"⚠️ Fancy geodesic visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.model.train()
    
    def _create_interactive_geodesic_slider(self, z_seq, z_pca_seq, xx, yy, pca, epoch):
        """Create interactive slider visualization for geodesic evolution."""
        try:
            batch_size, n_obs, latent_dim = z_seq.shape
            
            # Project metric to PCA space
            V = self._ensure_tensor_on_device(torch.tensor(pca.components_, dtype=torch.float32))
            
            # Pre-compute background fields for selected timesteps (fewer for performance)
            timesteps_to_show = list(range(0, n_obs, max(1, n_obs // 4)))  # Show max 4 timesteps
            timestep_background_fields = {}
            
            print(f"📊 Computing background fields for timesteps: {timesteps_to_show}")
            
            grid_points_pca = np.column_stack([xx.ravel(), yy.ravel()])
            grid_points_latent_base = pca.inverse_transform(grid_points_pca)
            grid_tensor_base = self._ensure_tensor_on_device(torch.tensor(grid_points_latent_base, dtype=torch.float32))
            
            for t_bg in timesteps_to_show:
                try:
                    if t_bg == 0:
                        grid_tensor_t = grid_tensor_base.clone()
                    else:
                        # Apply flows to transform grid
                        grid_tensor_t = grid_tensor_base.clone()
                        for flow_idx in range(min(t_bg, len(self._get_flows()))):
                            flow = self._get_flows()[flow_idx]
                            flow_result = flow(grid_tensor_t)
                            # Handle tuple output (e.g., (tensor, log_det))
                            if isinstance(flow_result, tuple):
                                flow_result = flow_result[0]
                            # Extract tensor if ModelOutput, else pass through
                            if hasattr(flow_result, 'sample'):
                                grid_tensor_t = flow_result.sample
                            elif hasattr(flow_result, 'z'):
                                grid_tensor_t = flow_result.z
                            elif hasattr(flow_result, 'out'):
                                grid_tensor_t = flow_result.out
                            elif isinstance(flow_result, torch.Tensor):
                                grid_tensor_t = flow_result
                            else:
                                raise TypeError(f"Flow {flow_idx} did not return a tensor, ModelOutput, or tuple with tensor as first element. Got: {type(flow_result)}")
                            if not isinstance(grid_tensor_t, torch.Tensor):
                                raise TypeError(f"After extraction, flow {flow_idx} did not yield a tensor.")
                            grid_tensor_t = self._ensure_tensor_on_device(grid_tensor_t)
                    
                    # Ensure tensor is on correct device
                    grid_tensor_t = self._ensure_tensor_on_device(grid_tensor_t)
                    
                    # Compute metric at transformed grid
                    G_grid_t = self.model.G(grid_tensor_t)
                    G_grid_t = self._ensure_tensor_on_device(G_grid_t)
                    
                    # Ensure V is on correct device
                    V = self._ensure_tensor_on_device(V)
                    
                    # Handle potential dimension mismatches
                    if G_grid_t.dim() == 2:  # [latent_dim, latent_dim] -> [N, latent_dim, latent_dim]
                        G_grid_t = G_grid_t.unsqueeze(0).expand(grid_tensor_t.shape[0], -1, -1)
                    elif G_grid_t.dim() == 3 and G_grid_t.shape[0] != grid_tensor_t.shape[0]:
                        # If batch dimension mismatch, use first element for all
                        G_grid_t = G_grid_t[0:1].expand(grid_tensor_t.shape[0], -1, -1)
                    
                    # Ensure V has correct batch dimension
                    V_expanded = V.unsqueeze(0).expand(G_grid_t.shape[0], -1, -1)
                    VT_expanded = V.T.unsqueeze(0).expand(G_grid_t.shape[0], -1, -1)
                    
                    # Ensure all tensors are on same device before matrix operations
                    V_expanded = self._ensure_tensor_on_device(V_expanded)
                    VT_expanded = self._ensure_tensor_on_device(VT_expanded)
                    
                    # Compute G_pca_t = V @ G @ V.T
                    G_pca_t = torch.matmul(torch.matmul(V_expanded, G_grid_t), VT_expanded)
                    det_G_pca_t = torch.linalg.det(G_pca_t).cpu().numpy().reshape(xx.shape)
                    
                    timestep_background_fields[t_bg] = {'det_G_pca': det_G_pca_t}
                    
                except Exception as e:
                    print(f"⚠️ Background computation failed for t={t_bg}: {e}")
                    import traceback
                    traceback.print_exc()
                    timestep_background_fields[t_bg] = {'det_G_pca': np.ones(xx.shape)}
            
            # Compute metrics at flow-evolved coordinates
            timestep_geodesic_data = []
            for t in range(n_obs):
                z_t_pca = z_pca_seq[:, t, :]
                z_t_latent = z_seq[:, t, :].cpu().numpy()
                z_t_tensor = self._ensure_tensor_on_device(torch.tensor(z_t_latent, dtype=torch.float32))
                
                try:
                    # Compute metric
                    G_t = self.model.G(z_t_tensor)
                    G_t = self._ensure_tensor_on_device(G_t)
                    
                    # Ensure V is on correct device
                    V = self._ensure_tensor_on_device(V)
                    
                    # Handle potential dimension mismatches
                    if G_t.dim() == 2:  # [latent_dim, latent_dim] -> [N, latent_dim, latent_dim]
                        G_t = G_t.unsqueeze(0).expand(z_t_tensor.shape[0], -1, -1)
                    elif G_t.dim() == 3 and G_t.shape[0] != z_t_tensor.shape[0]:
                        # If batch dimension mismatch, use first element for all
                        G_t = G_t[0:1].expand(z_t_tensor.shape[0], -1, -1)
                    
                    # Ensure V has correct batch dimension  
                    V_expanded = V.unsqueeze(0).expand(G_t.shape[0], -1, -1)
                    VT_expanded = V.T.unsqueeze(0).expand(G_t.shape[0], -1, -1)
                    
                    # Ensure all tensors are on same device before matrix operations
                    V_expanded = self._ensure_tensor_on_device(V_expanded)
                    VT_expanded = self._ensure_tensor_on_device(VT_expanded)
                    
                    # Compute G_t_pca = V @ G @ V.T
                    G_t_pca = torch.matmul(torch.matmul(V_expanded, G_t), VT_expanded)
                    det_t = torch.linalg.det(G_t_pca).cpu().numpy()
                    
                    timestep_geodesic_data.append({
                        'positions': z_t_pca,
                        'det': det_t
                    })
                except Exception as e:
                    print(f"⚠️ Metric computation failed for t={t}: {e}")
                    import traceback
                    traceback.print_exc()
                    timestep_geodesic_data.append({
                        'positions': z_t_pca,
                        'det': np.ones(len(z_t_pca))
                    })
            
            # Create interactive plot - SMALLER SIZE
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["🎯 Geodesic Trajectories", "🌟 Det(G) Heatmap"],
                horizontal_spacing=0.15
            )
            
            # Create frames for each timestep
            frames = []
            colors = px.colors.qualitative.Set3[:min(batch_size, 6)]  # Limit sequences for performance
            
            for t in range(n_obs):
                frame_data = []
                geo_data = timestep_geodesic_data[t]
                
                # Get background field (use closest computed timestep)
                closest_t = min(timesteps_to_show, key=lambda x: abs(x - t))
                bg_fields = timestep_background_fields.get(closest_t, {'det_G_pca': np.ones(xx.shape)})
                det_G_pca_t = bg_fields['det_G_pca']
                
                # Panel 1: Geodesic trajectories
                frame_data.append(
                    go.Contour(
                        x=np.linspace(xx.min(), xx.max(), xx.shape[1]),
                        y=np.linspace(yy.min(), yy.max(), yy.shape[0]),
                        z=np.log10(np.clip(det_G_pca_t, 1e-10, None)),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="log₁₀(det(G))", x=0.45, len=0.8),
                        opacity=0.3,
                        name="det(G) field",
                        xaxis='x', yaxis='y'
                    )
                )
                
                # Add trajectory paths (limit sequences)
                for seq_idx in range(min(batch_size, 6)):
                    traj_segment = z_pca_seq[seq_idx, :t+1, :]
                    if len(traj_segment) > 1:
                        frame_data.append(
                            go.Scatter(
                                x=traj_segment[:, 0],
                                y=traj_segment[:, 1],
                                mode='lines+markers',
                                line=dict(color=colors[seq_idx], width=2),
                                marker=dict(size=4, color=colors[seq_idx]),
                                name=f"Path {seq_idx}",
                                showlegend=(t == 0),
                                xaxis='x', yaxis='y'
                            )
                        )
                    
                    # Current position
                    frame_data.append(
                        go.Scatter(
                            x=[geo_data['positions'][seq_idx, 0]],
                            y=[geo_data['positions'][seq_idx, 1]],
                            mode='markers',
                            marker=dict(size=8, color=colors[seq_idx], symbol='star'),
                            name=f"t={t}",
                            showlegend=False,
                            xaxis='x', yaxis='y'
                        )
                    )
                
                # Panel 2: Det(G) values at sequence positions
                frame_data.append(
                    go.Scatter(
                        x=geo_data['positions'][:, 0],
                        y=geo_data['positions'][:, 1],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=geo_data['det'],
                            colorscale='Plasma',
                            showscale=True,
                            colorbar=dict(title="det(G)", x=1.02, len=0.8),
                            line=dict(color='white', width=1)
                        ),
                        name="Sequences",
                        showlegend=False,
                        xaxis='x2', yaxis='y2'
                    )
                )
                
                frames.append(go.Frame(data=frame_data, name=str(t)))
            
            # Set initial frame
            fig.add_traces(frames[0].data)
            fig.frames = frames
            
            # Update layout - SMALLER SIZE
            fig.update_layout(
                title=f"🎚️ Interactive Geodesic Evolution - Epoch {epoch}",
                sliders=[{
                    "active": 0,
                    "currentvalue": {"prefix": "Timestep: ", "visible": True, "font": {"color": "white"}},
                    "pad": {"b": 10, "t": 50},
                    "steps": [{"args": [[f], {"frame": {"duration": 300, "redraw": True}}], 
                             "label": str(t), "method": "animate"} 
                             for t, f in enumerate(frames)]
                }],
                width=1000,  # SMALLER
                height=500,  # SMALLER
                showlegend=True,
                legend=dict(
                    bgcolor='rgba(20,20,20,0.9)',  # Dark background for visibility
                    bordercolor='white',
                    borderwidth=2,
                    font=dict(size=12, color='white')
                ),
                # Dark theme to match Wandb
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            # Save interactive HTML
            html_filename = f'geodesic_sliders_epoch_{epoch}.html'
            html_path = self._get_output_path(html_filename, "interactive")
            fig.write_html(html_path, include_plotlyjs=True)
            print(f"💾 Saved geodesic sliders: {html_path}")
            
            # Save static version
            png_filename = f'geodesic_sliders_epoch_{epoch}.png'
            saved_png = self._safe_write_image(fig, png_filename, width=1000, height=500)
            
            # Log to WandB
            if self.should_log_to_wandb():
                log_dict = {"interactive/geodesic_sliders": wandb.Html(html_path)}
                if saved_png and saved_png.endswith('.png'):
                    log_dict["interactive/geodesic_sliders_static"] = wandb.Image(saved_png)
                wandb.log(log_dict)
            
        except Exception as e:
            print(f"⚠️ Interactive geodesic slider creation failed: {e}")
    
    def _create_fancy_interactive_plot(self, z_seq, z_pca_dense, z_orig_pca, pca, epoch):
        """Create fancy interactive plot with multiple panels."""
        try:
            batch_size, dense_n_points, _ = z_pca_dense.shape
            
            # Create fancy interactive subplots - SMALLER SIZE
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=["🌀 Dense Trajectories", "🎭 Eigenvalue Field", 
                               "📊 Path Analytics", "🔥 Metric Amplification"],
                horizontal_spacing=0.12,
                vertical_spacing=0.15
            )
            
            # Limit sequences for performance
            max_seqs = min(6, batch_size)
            colors = px.colors.qualitative.Set3[:max_seqs]
            
            # Panel 1: Dense trajectory paths
            for seq_idx in range(max_seqs):
                # Dense trajectory line
                fig.add_trace(
                    go.Scatter(
                        x=z_pca_dense[seq_idx, :, 0],
                        y=z_pca_dense[seq_idx, :, 1],
                        mode='lines',
                        line=dict(color=colors[seq_idx], width=2),
                        name=f'Dense Path {seq_idx}',
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                
                # Original points
                fig.add_trace(
                    go.Scatter(
                        x=z_orig_pca[seq_idx, :, 0],
                        y=z_orig_pca[seq_idx, :, 1],
                        mode='markers',
                        marker=dict(color=colors[seq_idx], size=6, 
                                   line=dict(color='white', width=1)),
                        name=f'Original {seq_idx}',
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Panel 2: Simplified eigenvalue field
            self._add_simplified_eigenvalue_field(fig, z_pca_dense, pca, row=1, col=2)
            
            # Panel 3: Path analytics
            self._add_path_analytics(fig, z_orig_pca, row=2, col=1)
            
            # Panel 4: Simplified amplification heatmap
            self._add_simplified_amplification(fig, z_pca_dense, pca, row=2, col=2)
            
            # Global styling - SMALLER SIZE
            fig.update_layout(
                title=f"✨ Interactive Geodesic Analysis - Epoch {epoch}",
                width=1000,  # SMALLER
                height=800,  # SMALLER
                showlegend=True,
                legend=dict(
                    bgcolor='rgba(20,20,20,0.9)',  # Dark background for visibility
                    bordercolor='white',
                    borderwidth=2,
                    font=dict(size=12, color='white')
                ),
                font={'size': 10, 'color': 'white'},  # Smaller font with white color
                # Dark theme to match Wandb
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            # Save as interactive HTML
            html_filename = f'fancy_geodesic_analysis_epoch_{epoch}.html'
            html_path = self._get_output_path(html_filename, "interactive")
            fig.write_html(html_path, include_plotlyjs=True)
            print(f"💾 Saved fancy geodesic analysis: {html_path}")
            
            # Save static version
            png_filename = f'fancy_geodesic_analysis_epoch_{epoch}.png'
            saved_png = self._safe_write_image(fig, png_filename, width=1000, height=800)
            
            # Log to WandB
            if self.should_log_to_wandb():
                log_dict = {"interactive/fancy_geodesics": wandb.Html(html_path)}
                if saved_png and saved_png.endswith('.png'):
                    log_dict["interactive/fancy_geodesics_static"] = wandb.Image(saved_png)
                wandb.log(log_dict)
            
        except Exception as e:
            print(f"⚠️ Fancy interactive plot creation failed: {e}")
    
    def _generate_dense_trajectories(self, z_seq, n_interp_points=10):
        """Generate dense trajectories with interpolated points."""
        batch_size, n_obs, latent_dim = z_seq.shape
        dense_trajectories = []
        
        for seq_idx in range(batch_size):
            seq_points = []
            for t in range(n_obs - 1):
                seq_points.append(z_seq[seq_idx, t])
                
                # Add interpolated points
                start_point = z_seq[seq_idx, t]
                end_point = z_seq[seq_idx, t + 1]
                
                for i in range(1, n_interp_points):
                    alpha = i / n_interp_points
                    interp_point = (1 - alpha) * start_point + alpha * end_point
                    seq_points.append(interp_point)
            
            seq_points.append(z_seq[seq_idx, -1])
            dense_trajectories.append(torch.stack(seq_points))
        
        return torch.stack(dense_trajectories)
    
    def _add_simplified_eigenvalue_field(self, fig, z_pca_dense, pca, row, col):
        """Add simplified eigenvalue field visualization."""
        try:
            # Ensure model is on correct device
            self._ensure_model_on_device()
            
            # Sample fewer points for performance
            sample_indices = np.random.choice(z_pca_dense.shape[1], 
                                            min(100, z_pca_dense.shape[1]), replace=False)
            
            # Create sample grid
            x_coords = z_pca_dense[0, sample_indices, 0]
            y_coords = z_pca_dense[0, sample_indices, 1]
            
            # Compute simplified metric field
            V = torch.tensor(pca.components_, dtype=torch.float32, device=self.device)
            grid_latent = pca.inverse_transform(np.column_stack([x_coords, y_coords]))
            grid_tensor = torch.tensor(grid_latent, dtype=torch.float32, device=self.device)
            
            G_grid = self.model.G(grid_tensor)
            
            # Ensure all tensors are on the same device and have correct dimensions
            V = V.to(self.device)
            G_grid = G_grid.to(self.device)
            
            # Handle potential dimension mismatches
            if G_grid.dim() == 2:  # [N, latent_dim, latent_dim] -> add batch dimension
                G_grid = G_grid.unsqueeze(0)
            
            V_expanded = V.unsqueeze(0).expand(G_grid.shape[0], -1, -1)
            VT_expanded = V.T.unsqueeze(0).expand(G_grid.shape[0], -1, -1)
            
            G_pca = torch.matmul(torch.matmul(V_expanded, G_grid), VT_expanded)
            eigenvals = torch.linalg.eigvals(G_pca).real.cpu().numpy()
            mean_eigenvals = eigenvals.mean(axis=1)
            
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='markers',
                    marker=dict(size=4, color=mean_eigenvals, colorscale='Viridis'),
                    name="Eigenvalue Field",
                    showlegend=False
                ),
                row=row, col=col
            )
        except Exception as e:
            print(f"⚠️ Eigenvalue field computation failed: {e}")
    
    def _add_path_analytics(self, fig, z_orig_pca, row, col):
        """Add path analytics visualization."""
        batch_size, n_obs, _ = z_orig_pca.shape
        
        # Calculate path lengths
        path_lengths = []
        for seq_idx in range(batch_size):
            diffs = np.diff(z_orig_pca[seq_idx], axis=0)
            lengths = np.linalg.norm(diffs, axis=1)
            total_length = np.sum(lengths)
            path_lengths.append(total_length)
        
        fig.add_trace(
            go.Histogram(
                x=path_lengths,
                nbinsx=20,
                name="Path Lengths",
                showlegend=False,
                marker_color='rgba(55, 128, 191, 0.7)'
            ),
            row=row, col=col
        )
    
    def _add_simplified_amplification(self, fig, z_pca_dense, pca, row, col):
        """Add simplified amplification heatmap."""
        try:
            # Ensure model is on correct device
            self._ensure_model_on_device()
            
            # Sample grid points
            x_range = [z_pca_dense[:, :, 0].min(), z_pca_dense[:, :, 0].max()]
            y_range = [z_pca_dense[:, :, 1].min(), z_pca_dense[:, :, 1].max()]
            
            # Create small grid
            x_grid = np.linspace(x_range[0], x_range[1], 15)
            y_grid = np.linspace(y_range[0], y_range[1], 15)
            XX, YY = np.meshgrid(x_grid, y_grid)
            
            # Compute amplification
            grid_points = np.column_stack([XX.ravel(), YY.ravel()])
            grid_latent = pca.inverse_transform(grid_points)
            grid_tensor = self._ensure_tensor_on_device(torch.tensor(grid_latent, dtype=torch.float32))
            
            # Compute metric
            G_grid = self.model.G(grid_tensor)
            G_grid = self._ensure_tensor_on_device(G_grid)
            det_G = torch.linalg.det(G_grid).cpu().numpy()
            det_G_grid = det_G.reshape(XX.shape)
            
            fig.add_trace(
                go.Contour(
                    x=x_grid,
                    y=y_grid,
                    z=np.log10(np.clip(det_G_grid, 1e-10, None)),
                    colorscale='Hot',
                    showscale=False,
                    name="Amplification",
                    opacity=0.7
                ),
                row=row, col=col
            )
        except Exception as e:
            print(f"⚠️ Amplification computation failed: {e}")

    def create_metric_slider_visualization(self, x_sample: torch.Tensor, epoch: int):
        """Create interactive metric evolution slider with timestep-based heatmaps (SMALLER VERSION)."""
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - skipping metric slider")
            return
            
        try:
            print(f"🎬 Creating INTERACTIVE metric slider visualization for epoch {epoch}")
            
            # Ensure entire model is on correct device
            self._ensure_model_on_device()
            
            self.model.eval()
            with torch.no_grad():
                result = self.model_forward(x_sample)
                z_seq = result['latent_samples'] if isinstance(result, dict) else result.z  # [batch_size, n_obs, latent_dim]
                
                batch_size, n_obs, latent_dim = z_seq.shape
                
                # Apply PCA for visualization
                z_pca_seq, pca = self._prepare_pca_data(z_seq, n_components=2)
                
                # Create SMALLER grid for performance
                x_min, x_max = z_pca_seq[:, :, 0].min() - 1, z_pca_seq[:, :, 0].max() + 1
                y_min, y_max = z_pca_seq[:, :, 1].min() - 1, z_pca_seq[:, :, 1].max() + 1
                nx, ny = 25, 25  # SMALLER GRID for performance
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
                
                # Create SMALLER figure
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=["🎯 det(G) Evolution", "📊 Sequence Metrics"],
                    horizontal_spacing=0.15
                )
                
                # Pre-compute heatmaps for fewer timesteps for performance
                timesteps_to_compute = list(range(0, n_obs, max(1, n_obs // 4)))
                heatmap_cache = {}
                
                grid_points_pca = np.column_stack([xx.ravel(), yy.ravel()])
                grid_points_latent = pca.inverse_transform(grid_points_pca)
                grid_tensor = self._ensure_tensor_on_device(torch.tensor(grid_points_latent, dtype=torch.float32))
                
                for t in timesteps_to_compute:
                    try:
                        if t > 0 and hasattr(self.model, 'flows'):
                            grid_t = grid_tensor.clone()
                            for flow_idx in range(min(t, len(self._get_flows()))):
                                flow = self._get_flows()[flow_idx]
                                flow_result = flow(grid_t)
                                # Handle tuple output (e.g., (tensor, log_det))
                                if isinstance(flow_result, tuple):
                                    flow_result = flow_result[0]
                                # Extract tensor if ModelOutput, else pass through
                                if hasattr(flow_result, 'sample'):
                                    grid_t = flow_result.sample
                                elif hasattr(flow_result, 'z'):
                                    grid_t = flow_result.z
                                elif hasattr(flow_result, 'out'):
                                    grid_t = flow_result.out
                                elif isinstance(flow_result, torch.Tensor):
                                    grid_t = flow_result
                                else:
                                    raise TypeError(f"Flow {flow_idx} did not return a tensor, ModelOutput, or tuple with tensor as first element. Got: {type(flow_result)}")
                                if not isinstance(grid_t, torch.Tensor):
                                    raise TypeError(f"After extraction, flow {flow_idx} did not yield a tensor.")
                                grid_t = self._ensure_tensor_on_device(grid_t)
                        else:
                            grid_t = self._ensure_tensor_on_device(grid_tensor)
                        
                        # Compute metric
                        G_grid = self.model.G(grid_t)
                        G_grid = self._ensure_tensor_on_device(G_grid)
                        det_G = torch.linalg.det(G_grid).cpu().numpy()
                        heatmap_cache[t] = det_G.reshape(xx.shape)
                    except Exception as e:
                        print(f"⚠️ Heatmap computation failed for t={t}: {e}")
                        import traceback
                        traceback.print_exc()
                        heatmap_cache[t] = np.ones(xx.shape)
                
                # Prepare frames for each timestep
                frames = []
                colors = px.colors.qualitative.Set3[:min(batch_size, 4)]  # Limit to 4 sequences
                
                for t in range(n_obs):
                    frame_data = []
                    
                    # Use closest computed heatmap
                    closest_t = min(timesteps_to_compute, key=lambda x: abs(x - t))
                    det_heatmap = heatmap_cache.get(closest_t, np.ones(xx.shape))
                    
                    # Background heatmap
                    frame_data.append(
                        go.Contour(
                            x=np.linspace(x_min, x_max, nx),
                            y=np.linspace(y_min, y_max, ny),
                            z=np.log10(np.clip(det_heatmap, 1e-10, None)),
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="log₁₀(det(G))", x=0.4, len=0.6),
                            name="det(G) field",
                            xaxis='x', yaxis='y'
                        )
                    )
                    
                    # Sequence points (limited number)
                    for seq_idx in range(min(batch_size, 4)):
                        frame_data.append(
                            go.Scatter(
                                x=[z_pca_seq[seq_idx, t, 0]],
                                y=[z_pca_seq[seq_idx, t, 1]],
                                mode='markers',
                                marker=dict(size=10, color=colors[seq_idx], 
                                          line=dict(color='white', width=2)),
                                name=f"Seq {seq_idx}",
                                showlegend=(t == 0),
                                xaxis='x', yaxis='y'
                            )
                        )
                    
                    # Metric evolution (right panel) - simplified
                    timesteps_so_far = np.arange(t+1)
                    for seq_idx in range(min(batch_size, 4)):
                        try:
                            z_seq_so_far = z_seq[seq_idx, :t+1, :]
                            z_seq_tensor = self._ensure_tensor_on_device(z_seq_so_far)
                            G_seq = self.model.G(z_seq_tensor)
                            G_seq = self._ensure_tensor_on_device(G_seq)
                            det_seq = torch.linalg.det(G_seq).cpu().numpy()
                        except Exception as e:
                            det_seq = np.ones(t+1)
                        
                        frame_data.append(
                            go.Scatter(
                                x=timesteps_so_far, y=det_seq,
                                mode='lines+markers',
                                line=dict(color=colors[seq_idx], width=2),
                                marker=dict(size=4, color=colors[seq_idx]),
                                name=f'det(G) Seq {seq_idx}',
                                showlegend=False,
                                xaxis='x2', yaxis='y2'
                            )
                        )
                    
                    frames.append(go.Frame(data=frame_data, name=str(t)))
                
                # Set initial frame
                for trace in frames[0].data:
                    if hasattr(trace, 'xaxis') and trace.xaxis == 'x2':
                        fig.add_trace(trace, row=1, col=2)
                    else:
                        fig.add_trace(trace, row=1, col=1)
                
                fig.frames = frames
                
                # Add controls - SMALLER SIZE
                fig.update_layout(
                    title=f"🎬 Interactive Metric Evolution - Epoch {epoch}",
                    sliders=[{
                        "active": 0,
                        "yanchor": "top",
                        "xanchor": "left",
                        "currentvalue": {
                            "font": {"size": 14, "color": "white"}, 
                            "prefix": "Sequence: ", 
                            "visible": True, 
                            "xanchor": "left"
                        },
                        "transition": {"duration": 300, "easing": "cubic-in-out"},
                        "pad": {"b": 10, "t": 10},
                        "len": 0.6,  # Even shorter slider to make room for wider layout
                        "x": 0.2,    # More centered
                        "y": -0.06,  # Match play button position
                        "steps": [{"args": [[f], {"frame": {"duration": 300}}], 
                                 "label": str(t), "method": "animate"} 
                                for t, f in enumerate(frames)]
                    }],
                    showlegend=True,
                    legend=dict(
                        bgcolor='rgba(20,20,20,0.9)',  # Dark background for visibility
                        bordercolor='white',
                        borderwidth=2,
                        font=dict(size=12, color='white')
                    ),
                    # Dark theme for overall figure
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    # Better margins to accommodate larger layout
                    margin=dict(l=80, r=120, t=60, b=60)
                )
                
                # Save without opening in browser
                html_filename = f'interactive_metric_slider_epoch_{epoch}.html'
                html_path = self._get_output_path(html_filename, "interactive")
                fig.write_html(html_path, include_plotlyjs=True)
                print(f"💾 Saved interactive metric slider: {html_path}")
                
                if self.should_log_to_wandb():
                    wandb.log({f"interactive/metric_slider_epoch_{epoch}": wandb.Html(html_path)})
        except Exception as e:
            print(f"⚠️ Failed to create interactive metric slider: {e}")
            import traceback
            traceback.print_exc()

    def create_temporal_animation(self, x_sample: torch.Tensor, epoch: int):
        """Create interactive temporal animation of metric evolution (SMALLER VERSION)."""
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - skipping temporal animation")
            return
            
        try:
            print(f"🎬 Creating INTERACTIVE temporal animation for epoch {epoch}")
            
            # Ensure entire model is on correct device
            self._ensure_model_on_device()
            
            self.model.eval()
            with torch.no_grad():
                result = self.model_forward(x_sample)
                z_seq = result['latent_samples'] if isinstance(result, dict) else result.z  # [batch_size, n_obs, latent_dim]
                
                batch_size, n_obs, latent_dim = z_seq.shape
                z_pca_seq, pca = self._prepare_pca_data(z_seq, n_components=2)
                
                # Create SMALLER grid
                x_min, x_max = z_pca_seq[:, :, 0].min() - 1, z_pca_seq[:, :, 0].max() + 1
                y_min, y_max = z_pca_seq[:, :, 1].min() - 1, z_pca_seq[:, :, 1].max() + 1
                nx, ny = 20, 20  # SMALLER GRID
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
                
                # Pre-compute simplified temporal det maps
                temporal_det_maps = []
                sequence_dets = np.zeros((n_obs, min(batch_size, 4)))
                
                grid_points_pca = np.column_stack([xx.ravel(), yy.ravel()])
                grid_points_latent = pca.inverse_transform(grid_points_pca)
                grid_tensor = torch.tensor(grid_points_latent, dtype=torch.float32, device=self.device)
                
                for t in range(n_obs):
                    try:
                        # Simplified: use same grid for all timesteps (for performance)
                        # Ensure tensor is on correct device
                        grid_tensor_device = grid_tensor.to(self.device)
                        G_grid = self.model.G(grid_tensor_device)
                        det_G = torch.linalg.det(G_grid).cpu().numpy()
                        temporal_det_maps.append(det_G.reshape(xx.shape))
                        
                        # Compute sequence metrics
                        for seq_idx in range(min(batch_size, 4)):
                            z_t = z_seq[seq_idx, t:t+1, :].to(self.device)
                            G_t = self.model.G(z_t)
                            sequence_dets[t, seq_idx] = torch.linalg.det(G_t).cpu().item()
                    except Exception as e:
                        print(f"⚠️ Temporal animation computation failed for t={t}: {e}")
                        temporal_det_maps.append(np.ones(xx.shape))
                        sequence_dets[t, :] = 1.0
                
                # Create WIDER animation figure with better proportions
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=["🎬 Temporal det(G) Evolution", "📈 det(G) Along Sequences"],
                    horizontal_spacing=0.08,  # Reduced spacing for wider layout
                    column_widths=[0.6, 0.4]  # Give more space to the spatial plot
                )
                
                frames = []
                colors = px.colors.qualitative.Set3[:min(batch_size, 4)]
                
                for t in range(n_obs):
                    frame_data = []
                    
                    # Heatmap for current timestep
                    frame_data.append(
                        go.Contour(
                            x=np.linspace(x_min, x_max, nx),
                            y=np.linspace(y_min, y_max, ny),
                            z=temporal_det_maps[t],
                            colorscale='Viridis',
                            opacity=0.7,
                            showscale=True,
                            colorbar=dict(title="det(G)", x=0.52, len=0.8, thickness=15),  # Adjusted for medium layout
                            name="det(G) field",
                            xaxis='x', yaxis='y'
                        )
                    )
                    
                    # Sequence trajectories up to current timestep (limited number)
                    for seq_idx in range(min(batch_size, 4)):
                        traj_x = z_pca_seq[seq_idx, :t+1, 0]
                        traj_y = z_pca_seq[seq_idx, :t+1, 1]
                        
                        frame_data.append(
                            go.Scatter(
                                x=traj_x, y=traj_y,
                                mode='lines+markers',
                                line=dict(color=colors[seq_idx], width=2),
                                marker=dict(size=5, color=colors[seq_idx]),
                                name=f'Seq {seq_idx}',
                                showlegend=(t == 0),
                                xaxis='x', yaxis='y'
                            )
                        )
                        
                        # Current position marker
                        if t < len(traj_x):
                            frame_data.append(
                                go.Scatter(
                                    x=[traj_x[-1]], y=[traj_y[-1]],
                                    mode='markers',
                                    marker=dict(size=10, color=colors[seq_idx], symbol='star'),
                                    name=f'Current {seq_idx}',
                                    showlegend=False,
                                    xaxis='x', yaxis='y'
                                )
                            )
                    
                    # det(G) evolution plot (right panel)
                    for seq_idx in range(min(batch_size, 4)):
                        det_so_far = sequence_dets[:t+1, seq_idx]
                        timesteps_so_far = np.arange(t+1)
                        
                        frame_data.append(
                            go.Scatter(
                                x=timesteps_so_far, y=det_so_far,
                                mode='lines+markers',
                                line=dict(color=colors[seq_idx], width=2),
                                marker=dict(size=4, color=colors[seq_idx]),
                                name=f'det(G) Seq {seq_idx}',
                                showlegend=False,
                                xaxis='x2', yaxis='y2'
                            )
                        )
                    
                    frames.append(go.Frame(data=frame_data, name=str(t)))
                
                # Set initial frame
                for trace in frames[0].data:
                    if hasattr(trace, 'xaxis') and trace.xaxis == 'x2':
                        fig.add_trace(trace, row=1, col=2)
                    else:
                        fig.add_trace(trace, row=1, col=1)
                
                fig.frames = frames
                
                # Add animation controls - MEDIUM SIZE
                fig.update_layout(
                    title=f"🎬 Temporal Metric Animation - Epoch {epoch}",
                    # Remove play buttons entirely - only keep slider
                    sliders=[{
                        "active": 0,
                        "currentvalue": {"prefix": "Timestep: ", "font": {"color": "white", "size": 14}},
                        "pad": {"b": 20, "t": 20},
                        "len": 0.8,  # Longer slider since no play button
                        "x": 0.1,
                        "y": -0.08,
                        "steps": [{"args": [[f], {"frame": {"duration": 0, "redraw": True}}],  # No auto-duration
                                 "label": str(t), "method": "animate"} 
                                for t, f in enumerate(frames)]
                    }],
                    width=1200,   # Reduced from 1400 to 1200
                    height=500,   # Reduced from 600 to 500 (tinier)
                    showlegend=True,
                    legend=dict(
                        bgcolor='rgba(20,20,20,0.9)',  # Dark background for visibility
                        bordercolor='white',
                        borderwidth=2,
                        font=dict(size=12, color='white')
                    ),
                    # Dark theme to match Wandb
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    margin=dict(l=80, r=80, t=60, b=80)  # Better margins for medium layout
                )
                
                # Update axes with proper ranges to show all timesteps
                fig.update_xaxes(title_text="PC1", row=1, col=1)
                fig.update_yaxes(title_text="PC2", row=1, col=1)
                fig.update_xaxes(title_text="Timestep", range=[-0.5, n_obs-0.5], row=1, col=2)  # Ensure all timesteps visible
                fig.update_yaxes(title_text="det(G)", row=1, col=2)
                
                # Save animation
                html_filename = f'temporal_metric_animation_epoch_{epoch}.html'
                html_path = self._get_output_path(html_filename, "interactive")
                fig.write_html(html_path, include_plotlyjs=True)
                
                static_filename = f'temporal_metric_animation_epoch_{epoch}.png'
                saved_png = self._safe_write_image(fig, static_filename, width=1200, height=500)
                
                if self.should_log_to_wandb():
                    log_dict = {"interactive/temporal_animation": wandb.Html(html_path)}
                    if saved_png and saved_png.endswith('.png'):
                        log_dict["interactive/temporal_animation_static"] = wandb.Image(saved_png)
                    wandb.log(log_dict)
                
                print(f"✅ Interactive temporal animation saved: {html_filename}")
                
        except Exception as e:
            print(f"⚠️ Failed to create interactive temporal animation: {e}")
            import traceback
            traceback.print_exc()

    def create_html_latent_space(self, x_sample: torch.Tensor, epoch: int, num_sequences: int = 6):
        """Create interactive HTML latent space visualization (COMPACT VERSION)."""
        try:
            print(f"🌐 Creating interactive HTML latent space for epoch {epoch}")
            
            # Ensure entire model is on correct device
            self._ensure_model_on_device()
            
            self.model.eval()
            with torch.no_grad():
                # Use fewer sequences for performance
                n_sequences = min(num_sequences, x_sample.shape[0])
                selected_data = x_sample[:n_sequences]
                
                result = self.model_forward(selected_data)
                z_seq = result['latent_samples'] if isinstance(result, dict) else result.z  # [n_sequences, n_obs, latent_dim]
                recon_x = result['reconstruction'] if isinstance(result, dict) else result.recon_x  # [n_sequences, n_obs, 3, 64, 64]
                
                # Flatten for visualization
                all_latents = []
                all_images = []
                sequence_info = []
                
                for seq_idx in range(n_sequences):
                    for t in range(z_seq.shape[1]):
                        all_latents.append(z_seq[seq_idx, t].cpu().numpy())
                        all_images.append(recon_x[seq_idx, t].cpu().numpy())
                        sequence_info.append({
                            'seq_id': seq_idx,
                            'timestep': t,
                            'is_start': t == 0,
                            'is_end': t == z_seq.shape[1] - 1
                        })
                
                # Apply PCA
                latents_array = np.array(all_latents)
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                latents_2d = pca.fit_transform(latents_array)
                
                # Create SMALLER images directory
                import os
                images_dir = f"html_latent_images_epoch_{epoch}"
                os.makedirs(images_dir, exist_ok=True)
                
                # Save SMALLER images (downsampled for performance)
                import matplotlib.pyplot as plt
                for i, (img_array, info) in enumerate(zip(all_images, sequence_info)):
                    img_display = np.transpose(img_array, (1, 2, 0))
                    img_display = np.clip(img_display, 0, 1)
                    
                    filename = f"seq_{info['seq_id']:02d}_t_{info['timestep']:02d}.png"
                    filepath = os.path.join(images_dir, filename)
                    
                    # Save at smaller resolution for web display
                    plt.imsave(filepath, img_display, dpi=50)  # Lower DPI for smaller files
                
                # Generate COMPACT HTML
                self._generate_compact_html_file(
                    f"interactive_latent_space_epoch_{epoch}.html",
                    latents_2d, sequence_info, images_dir, pca
                )
                
                print(f"✅ Interactive HTML latent space created for epoch {epoch}")
                
        except Exception as e:
            print(f"⚠️ Failed to create interactive HTML latent space: {e}")
            import traceback
            traceback.print_exc()

    def _generate_compact_html_file(self, filename, latents_2d, sequence_info, images_dir, pca):
        """Generate a compact interactive HTML file (SMALLER VERSION)."""
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Interactive Latent Space</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 10px; }}
        .container {{ max-width: 800px; margin: 0 auto; padding: 15px; border-radius: 8px; }}
        #canvas {{ border: 2px solid #333; cursor: crosshair; display: block; margin: 10px auto; }}
        .info {{ display: flex; gap: 15px; margin-top: 15px; }}
        .point-info, .image-display {{ padding: 10px; border-radius: 5px; flex: 1; }}
        #selectedImage {{ max-width: 100%; border: 1px solid #ccc; }}
        h2 {{ text-align: center; color: #333; margin-bottom: 5px; }}
        p {{ text-align: center; color: #666; margin: 5px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>🌐 Interactive Latent Space</h2>
        <p>PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%}</p>
        
        <canvas id="canvas" width="500" height="300"></canvas>
        
        <div class="info">
            <div class="point-info">
                <h3>Point Info</h3>
                <div id="pointInfo">Click a point to see details</div>
            </div>
            <div class="image-display">
                <h3>Reconstruction</h3>
                <img id="selectedImage" src="" alt="Select a point" style="display:none;">
            </div>
        </div>
    </div>

    <script>
        const latents = {latents_2d.tolist()};
        const info = {sequence_info};
        const imagesDir = "{images_dir}";
        
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        // Find bounds
        const xValues = latents.map(p => p[0]);
        const yValues = latents.map(p => p[1]);
        const xMin = Math.min(...xValues), xMax = Math.max(...xValues);
        const yMin = Math.min(...yValues), yMax = Math.max(...yValues);
        
        const margin = 30;
        const plotWidth = canvas.width - 2 * margin;
        const plotHeight = canvas.height - 2 * margin;
        
        function scaleX(x) {{ return margin + (x - xMin) / (xMax - xMin) * plotWidth; }}
        function scaleY(y) {{ return margin + (1 - (y - yMin) / (yMax - yMin)) * plotHeight; }}
        
        function draw() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw axes
            ctx.strokeStyle = '#ddd';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(margin, margin);
            ctx.lineTo(margin, canvas.height - margin);
            ctx.lineTo(canvas.width - margin, canvas.height - margin);
            ctx.stroke();
            
            // Draw points (smaller for performance)
            latents.forEach((point, i) => {{
                const x = scaleX(point[0]);
                const y = scaleY(point[1]);
                const seqId = info[i].seq_id;
                
                ctx.fillStyle = `hsl(${{seqId * 60}}, 70%, 50%)`;
                ctx.beginPath();
                ctx.arc(x, y, info[i].is_start ? 6 : (info[i].is_end ? 4 : 3), 0, 2 * Math.PI);
                ctx.fill();
                
                if (info[i].is_start) {{
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }}
            }});
        }}
        
        canvas.addEventListener('click', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;
            
            let closest = null;
            let minDist = Infinity;
            
            latents.forEach((point, i) => {{
                const x = scaleX(point[0]);
                const y = scaleY(point[1]);
                const dist = Math.sqrt((clickX - x) ** 2 + (clickY - y) ** 2);
                
                if (dist < minDist && dist < 15) {{
                    minDist = dist;
                    closest = i;
                }}
            }});
            
            if (closest !== null) {{
                const pointInfo = info[closest];
                document.getElementById('pointInfo').innerHTML = `
                    <strong>Seq:</strong> ${{pointInfo.seq_id}}<br>
                    <strong>Time:</strong> ${{pointInfo.timestep}}<br>
                    <strong>Pos:</strong> (${{latents[closest][0].toFixed(2)}}, ${{latents[closest][1].toFixed(2)}})
                `;
                
                const imgPath = `${{imagesDir}}/seq_${{pointInfo.seq_id.toString().padStart(2, '0')}}_t_${{pointInfo.timestep.toString().padStart(2, '0')}}.png`;
                const img = document.getElementById('selectedImage');
                img.src = imgPath;
                img.style.display = 'block';
            }}
        }});
        
        draw();
    </script>
</body>
</html>"""
        
        html_path = self._get_output_path(filename, "interactive")
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        if self.should_log_to_wandb():
            wandb.log({"interactive/html_latent_space": wandb.Html(html_path)})
        
        print(f"💾 Saved compact HTML: {html_path}")

    def create_sequence_slider_visualization(self, x_sample: torch.Tensor, epoch: int):
        """
        Interactive visualization: slider to select a sequence, showing
        - Original sequence (row of images)
        - Reconstructed sequence (row of images)
        - Latent trajectory (PCA) for that sequence
        """
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - skipping sequence slider visualization")
            return
        print(f"🖼️ Creating interactive sequence slider visualization for epoch {epoch}")
        try:
            from plotly.subplots import make_subplots
            import plotly.io as pio
            self._ensure_model_on_device()
            self.model.eval()
            with torch.no_grad():
                result = self.model_forward(x_sample)
                # x_sample: [batch, n_obs, C, H, W]
                if isinstance(result, dict):
                    z_seq = result['latent_samples']  # [batch, n_obs, latent_dim]
                    # Accept both 'reconstructions' and 'reconstruction'
                    if 'reconstructions' in result:
                        recon_seq = result['reconstructions']
                    elif 'reconstruction' in result:
                        recon_seq = result['reconstruction']
                    else:
                        raise KeyError("Model output dict must contain 'reconstructions' or 'reconstruction'")
                else:
                    z_seq = result.z
                    recon_seq = result.recon_x
                x_seq = x_sample.cpu().numpy()
                recon_seq = recon_seq.cpu().numpy()
                z_seq = z_seq.cpu().numpy()
                batch_size, n_obs = x_seq.shape[0], x_seq.shape[1]
                # Limit number of sequences
                max_sequences = getattr(self.config.visualization, 'max_sequences', 8)
                n_sequences = min(batch_size, max_sequences)
                # PCA for latent trajectory
                from sklearn.decomposition import PCA
                z_flat = z_seq[:n_sequences].reshape(-1, z_seq.shape[-1])
                pca = PCA(n_components=2)
                z_pca = pca.fit_transform(z_flat).reshape(n_sequences, n_obs, 2)
                # Prepare images for plotly
                def to_img_array(img):
                    # img: [C, H, W] or [H, W, C]
                    if img.shape[0] <= 4:  # [C, H, W]
                        img = np.transpose(img, (1, 2, 0))
                    img = np.clip(img, 0, 1)
                    return (img * 255).astype(np.uint8)
                # Build frames for slider
                frames = []
                for seq_idx in range(n_sequences):
                    # Original and recon images
                    orig_imgs = [to_img_array(x_seq[seq_idx, t]) for t in range(n_obs)]
                    recon_imgs = [to_img_array(recon_seq[seq_idx, t]) for t in range(n_obs)]
                    # Latent trajectory
                    z_traj = z_pca[seq_idx]
                    # Build subplot
                    fig = make_subplots(
                        rows=3, cols=n_obs,
                        subplot_titles=None,  # We'll add custom annotations instead
                        row_heights=[0.2, 0.2, 0.6],  # Even larger trajectory plot (was 0.25, 0.25, 0.5)
                        vertical_spacing=0.02,  # Much reduced spacing (was 0.05)
                        horizontal_spacing=0.02,  # Slightly more horizontal spacing
                        specs=[[{"type": "xy"} for _ in range(n_obs)] for _ in range(2)] + 
                              [[{"type": "xy", "colspan": n_obs}] + [None] * (n_obs-1)]  # Trajectory spans ALL columns for maximum width
                    )
                    
                    # Add custom row labels (cleaner than overlapping titles)
                    fig.add_annotation(
                        text="<b>Original</b>", 
                        xref="paper", yref="paper",
                        x=-0.05, y=0.9, xanchor='right', yanchor='middle',  # Adjusted for new spacing
                        showarrow=False, font=dict(size=14, color='white')
                    )
                    fig.add_annotation(
                        text="<b>Reconstructed</b>", 
                        xref="paper", yref="paper", 
                        x=-0.05, y=0.7, xanchor='right', yanchor='middle',  # Adjusted for new spacing
                        showarrow=False, font=dict(size=14, color='white')
                    )
                    fig.add_annotation(
                        text="<b>Latent Trajectory (PCA)</b>", 
                        xref="paper", yref="paper",
                        x=-0.05, y=0.3, xanchor='right', yanchor='middle',  # Adjusted for larger plot
                        showarrow=False, font=dict(size=16, color='white', family='Arial Black')
                    )
                    
                    # Add timestep labels at the top
                    for t in range(n_obs):
                        fig.add_annotation(
                            text=f"<b>t={t}</b>",
                            xref="paper", yref="paper",
                            x=(t + 0.5) / n_obs, y=0.98,  # Top of the figure
                            xanchor='center', yanchor='top',
                            showarrow=False, font=dict(size=12, color='white')
                        )
                    
                    # Row 1: original images
                    for t in range(n_obs):
                        fig.add_trace(
                            go.Image(z=orig_imgs[t], name=f"Original t={t}"),
                            row=1, col=t+1
                        )
                    # Row 2: recon images
                    for t in range(n_obs):
                        fig.add_trace(
                            go.Image(z=recon_imgs[t], name=f"Recon t={t}"),
                            row=2, col=t+1
                        )
                    # Calculate auto-scaling bounds for PCA plot (across all sequences)
                    all_z_pca = np.array([z_pca[i] for i in range(n_sequences)])
                    x_min, x_max = all_z_pca[:, :, 0].min(), all_z_pca[:, :, 0].max()
                    y_min, y_max = all_z_pca[:, :, 1].min(), all_z_pca[:, :, 1].max()
                    # Add some padding (10%)
                    x_padding = (x_max - x_min) * 0.1
                    y_padding = (y_max - y_min) * 0.1
                    x_range = [x_min - x_padding, x_max + x_padding]
                    y_range = [y_min - y_padding, y_max + y_padding]
                    
                    # Row 3: latent trajectory (PCA) - spans multiple columns for larger display
                    fig.add_trace(
                        go.Scatter(x=z_traj[:, 0], y=z_traj[:, 1], mode='lines+markers',
                                   marker=dict(size=12, color='cyan', line=dict(width=2, color='white')),
                                   line=dict(width=4, color='cyan'),
                                   name='Trajectory'),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=[z_traj[0, 0]], y=[z_traj[0, 1]], mode='markers',
                                   marker=dict(size=16, color='lime', symbol='square', line=dict(width=2, color='white')),
                                   name='Start'),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=[z_traj[-1, 0]], y=[z_traj[-1, 1]], mode='markers',
                                   marker=dict(size=16, color='red', symbol='star', line=dict(width=2, color='white')),
                                   name='End'),
                        row=3, col=1
                    )
                    
                    # Hide axes for image rows
                    for t in range(n_obs):
                        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=t+1)
                        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=t+1)
                        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=t+1)
                        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=t+1)
                    
                    # Show proper labels and styling for trajectory plot (larger and more prominent) with auto-scaling
                    fig.update_xaxes(
                        title_text='<b>PC1</b>', 
                        showgrid=True, 
                        zeroline=True, 
                        gridcolor='rgba(255,255,255,0.3)',
                        title_font=dict(size=14, color='white'),
                        tickfont=dict(size=12, color='white'),
                        range=x_range,  # Auto-scaled range
                        row=3, col=1
                    )
                    fig.update_yaxes(
                        title_text='<b>PC2</b>', 
                        showgrid=True, 
                        zeroline=True, 
                        gridcolor='rgba(255,255,255,0.3)',
                        title_font=dict(size=14, color='white'),
                        tickfont=dict(size=12, color='white'),
                        range=y_range,  # Auto-scaled range
                        row=3, col=1
                    )
                    
                    fig.update_layout(
                        title=dict(
                            text=f"<b>Sequence {seq_idx} (Epoch {epoch})</b>",
                            x=0.5,
                            xanchor='center',
                            font=dict(size=18, color='white')
                        ),
                        height=800,  # Taller to accommodate larger trajectory plot
                        width=max(1000, 120*n_obs),  # Wider to accommodate better layout
                        margin=dict(l=80, r=120, t=60, b=40),  # More right margin for legend
                        showlegend=True,
                        legend=dict(
                            x=1.02,  # Position legend outside plot area
                            y=0.25,   # Align with trajectory plot
                            xanchor='left',
                            yanchor='middle',
                            bgcolor='rgba(20,20,20,0.9)',  # Darker background
                            bordercolor='white',
                            borderwidth=2,
                            font=dict(size=12, color='white')
                        ),
                        # Dark theme to match Wandb
                        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                        plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
                        font=dict(color='white', size=11)  # White text
                    )
                    frames.append(fig)
                # Create slider
                # Use the first frame as the initial figure
                fig = frames[0]
                # Add slider steps
                steps = []
                for i, frame in enumerate(frames):
                    steps.append(dict(
                        method="animate",
                        args=[[f"frame{i}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        label=f"Seq {i}"
                    ))
                # Add frames to the figure
                fig.frames = [go.Frame(data=frame.data, name=f"frame{i}") for i, frame in enumerate(frames)]
                fig.update_layout(
                    sliders=[{
                        "active": 0,
                        "yanchor": "top",
                        "xanchor": "left",
                        "currentvalue": {
                            "font": {"size": 14, "color": "white"}, 
                            "prefix": "Sequence: ", 
                            "visible": True, 
                            "xanchor": "left"
                        },
                        "transition": {"duration": 300, "easing": "cubic-in-out"},
                        "pad": {"b": 10, "t": 10},
                        "len": 0.6,  # Even shorter slider to make room for wider layout
                        "x": 0.2,    # More centered
                        "y": -0.06,  # Match play button position
                        "steps": steps
                    }],
                    # Dark theme for overall figure
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    # Better margins to accommodate larger layout
                    margin=dict(l=80, r=120, t=60, b=60)
                )
                # Save without opening in browser
                html_filename = f'sequence_slider_epoch_{epoch}.html'
                html_path = self._get_output_path(html_filename, "interactive")
                fig.write_html(html_path, include_plotlyjs=True)
                print(f"💾 Saved interactive sequence slider: {html_path}")
                
                if self.should_log_to_wandb():
                    wandb.log({f"interactive/sequence_slider_epoch_{epoch}": wandb.Html(html_path)})
        except Exception as e:
            print(f"⚠️ Sequence slider visualization failed: {e}")
            import traceback
            traceback.print_exc()
        self.model.train()