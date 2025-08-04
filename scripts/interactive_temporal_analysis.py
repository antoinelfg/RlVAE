#!/usr/bin/env python3
"""
Interactive Temporal Analysis for Trained RLVAE
===============================================

Rich interactive visualizations of temporal dynamics, latent trajectories,
and Riemannian metric evolution using Plotly and the existing interactive system.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from omegaconf import DictConfig
import logging
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.colors as pc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import wandb

# Setup paths
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir.parent / "src"
lib_src_dir = src_dir / "lib" / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(lib_src_dir) not in sys.path:
    sys.path.insert(0, str(lib_src_dir))

# Project imports
from models.modular_rlvae import ModularRiemannianFlowVAE
from data.cyclic_dataset import CyclicSpritesDataModule
from visualizations.interactive import InteractiveVisualizations
from visualizations.manager import VisualizationManager, VisualizationLevel
from omegaconf import DictConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InteractiveTemporalAnalyzer:
    """Create rich interactive visualizations of temporal dynamics and latent trajectories."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """Initialize interactive analyzer."""
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.model = None
        self.checkpoint = None
        self.data_module = None
        self.test_loader = None
        
        self.output_dir = Path("outputs/interactive_analysis") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Rich trajectory data storage
        self.all_trajectories = []
        self.all_determinants = []
        self.all_sequences = []
        self.all_reconstructions = []
        self.all_timestamps = []
        
        logger.info(f"🎭 Interactive temporal analyzer initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model_and_data(self) -> None:
        """Load model and data using the working approach."""
        logger.info(f"🔄 Loading model from {self.checkpoint_path}")
        
        # Load checkpoint
        self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        model_hparams = self.checkpoint['hyper_parameters']['model']
        
        # Create config
        config = DictConfig({
            'input_dim': model_hparams['input_dim'],
            'latent_dim': model_hparams['latent_dim'],
            'n_flows': model_hparams['n_flows'],
            'flow_hidden_size': model_hparams['flow_hidden_size'],
            'flow_n_blocks': model_hparams['flow_n_blocks'],
            'flow_n_hidden': model_hparams['flow_n_hidden'],
            'epsilon': model_hparams['epsilon'],
            'encoder': model_hparams['encoder'],
            'decoder': model_hparams['decoder'],
            'beta': model_hparams['beta'],
            'riemannian_beta': model_hparams['riemannian_beta'],
            'posterior': model_hparams['posterior'],
            'sampling': model_hparams['sampling'],
            'loop': model_hparams['loop'],
            'metric': model_hparams['metric'],
            'pretrained': {'encoder_path': None, 'decoder_path': None, 'metric_path': None},
            'sequence_length': model_hparams['sequence_length']
        })
        
        # Create and load model
        self.model = ModularRiemannianFlowVAE(config)
        
        # Resize metric tensor
        state_dict = self.checkpoint['state_dict']
        clean_state_dict = {k.replace('model.', '') if k.startswith('model.') else k: v 
                           for k, v in state_dict.items()}
        
        # Fix metric tensor size
        for name, param in clean_state_dict.items():
            if 'modular_metric.centroids' in name:
                self.model.modular_metric.centroids = torch.nn.Parameter(torch.zeros_like(param))
            elif 'modular_metric.metric_matrices' in name:
                self.model.modular_metric.metric_matrices = torch.nn.Parameter(torch.zeros_like(param))
        
        self.model.load_state_dict(clean_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # Setup data
        data_config = DictConfig({
            'train_path': 'data/sprites/ColoredCircles_train.pt',
            'test_path': 'data/sprites/ColoredCircles_test.pt',
            'train_meta_path': 'data/sprites/ColoredCircles_train_params.pt',
            'test_meta_path': 'data/sprites/ColoredCircles_test_params.pt',
            'sequence_length': 10,
            'image_size': [28, 28],
            'channels': 3,
            'batch_size': 1,  # Process one sequence at a time for detailed analysis
            'num_workers': 0,
            'pin_memory': False,
            'max_test_samples': 500,  # Use many sequences
            'verify_cyclicity': False
        })
        
        self.data_module = CyclicSpritesDataModule(data_config)
        self.data_module.setup('test')
        self.test_loader = self.data_module.test_dataloader()
        
        logger.info("✅ Model and data loaded successfully")
    
    def collect_rich_trajectory_data(self, n_sequences: int = 100) -> None:
        """Collect rich trajectory data from many sequences."""
        logger.info(f"🌊 Collecting rich trajectory data from {n_sequences} sequences")
        
        trajectory_count = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if trajectory_count >= n_sequences:
                    break
                
                sequences = batch.to(self.device)
                if len(sequences.shape) == 4:  # Single sequence
                    sequences = sequences.unsqueeze(0)
                
                # Full forward pass to get all outputs
                output = self.model(sequences)
                
                if isinstance(output, dict):
                    latent_seq = output['latent_samples']  # [1, seq_len, latent_dim]
                    recon_seq = output['reconstruction']   # [1, seq_len, C, H, W]
                else:
                    continue
                
                # Store trajectory data
                seq_len = latent_seq.shape[1]
                trajectory = latent_seq.squeeze(0).cpu().numpy()  # [seq_len, latent_dim]
                original = sequences.squeeze(0).cpu().numpy()      # [seq_len, C, H, W]
                recon = recon_seq.squeeze(0).cpu().numpy()         # [seq_len, C, H, W]
                
                # Compute metric determinants for each timestep
                determinants = []
                for t in range(seq_len):
                    z_t = latent_seq[0, t:t+1]  # [1, latent_dim]
                    
                    if hasattr(self.model, 'modular_metric') and self.model.modular_metric is not None:
                        # Find closest centroid
                        centroids = self.model.modular_metric.centroids
                        distances = torch.norm(z_t - centroids, dim=1)
                        closest_idx = torch.argmin(distances)
                        
                        # Get metric determinant
                        metric_matrix = self.model.modular_metric.metric_matrices[closest_idx]
                        det_g = torch.det(metric_matrix)
                        log_det_g_inv = -torch.log10(det_g + 1e-8)
                        determinants.append(log_det_g_inv.cpu().numpy())
                    else:
                        determinants.append(0.0)
                
                # Store data
                self.all_trajectories.append(trajectory)
                self.all_determinants.append(np.array(determinants))
                self.all_sequences.append(original)
                self.all_reconstructions.append(recon)
                self.all_timestamps.append(np.arange(seq_len))
                
                trajectory_count += 1
                
                if trajectory_count % 20 == 0:
                    logger.info(f"📊 Collected {trajectory_count}/{n_sequences} trajectories")
        
        logger.info(f"✅ Collected {len(self.all_trajectories)} rich trajectories")
    
    def create_interactive_trajectory_explorer(self) -> None:
        """Create comprehensive interactive trajectory explorer."""
        logger.info("🎭 Creating interactive trajectory explorer")
        
        # Combine all trajectories for global analysis
        all_points = np.vstack(self.all_trajectories)  # [total_points, latent_dim]
        all_dets = np.concatenate(self.all_determinants)
        
        # Create trajectory IDs and timesteps
        trajectory_ids = []
        timesteps = []
        for i, traj in enumerate(self.all_trajectories):
            trajectory_ids.extend([i] * len(traj))
            timesteps.extend(self.all_timestamps[i])
        
        trajectory_ids = np.array(trajectory_ids)
        timesteps = np.array(timesteps)
        
        # Apply PCA and t-SNE for different views
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(all_points)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_points)//4))
        tsne_coords = tsne.fit_transform(all_points)
        
        logger.info(f"📊 PCA explained variance: {pca.explained_variance_ratio_}")
        
        # Create comprehensive interactive plot
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "🎯 PCA Latent Space - Trajectories",
                "🌊 t-SNE Latent Space - Trajectories", 
                "⏱️ Temporal Evolution",
                "📏 Metric Determinant Evolution",
                "🎨 Trajectory Selection",
                "🔍 Detailed Trajectory View"
            ],
            specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]],
            horizontal_spacing=0.08,
            vertical_spacing=0.12
        )
        
        # 1. PCA Space with trajectories
        unique_traj_ids = np.unique(trajectory_ids)
        colors = px.colors.qualitative.Set3 * (len(unique_traj_ids) // len(px.colors.qualitative.Set3) + 1)
        
        for i, traj_id in enumerate(unique_traj_ids[:50]):  # Limit for performance
            mask = trajectory_ids == traj_id
            fig.add_trace(
                go.Scatter(
                    x=pca_coords[mask, 0],
                    y=pca_coords[mask, 1],
                    mode='markers+lines',
                    name=f'Traj {traj_id}',
                    line=dict(color=colors[i], width=2),
                    marker=dict(
                        size=8,
                        color=timesteps[mask],
                        colorscale='Viridis',
                        showscale=(i == 0),
                        colorbar=dict(title="Timestep", x=0.32) if i == 0 else None
                    ),
                    hovertemplate=f"Traj {traj_id}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>Timestep: %{{marker.color}}<extra></extra>",
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 2. t-SNE Space with determinant coloring
        fig.add_trace(
            go.Scatter(
                x=tsne_coords[:, 0],
                y=tsne_coords[:, 1],
                mode='markers',
                name='t-SNE',
                marker=dict(
                    size=6,
                    color=all_dets,
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(title="Log₁₀ det(G⁻¹)", x=0.65)
                ),
                hovertemplate="t-SNE1: %{x:.3f}<br>t-SNE2: %{y:.3f}<br>Det: %{marker.color:.3f}<br>Traj: %{customdata}<extra></extra>",
                customdata=trajectory_ids,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Temporal evolution - average trajectory
        mean_trajectory = np.mean(self.all_trajectories, axis=0)
        std_trajectory = np.std(self.all_trajectories, axis=0)
        
        for dim in range(mean_trajectory.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(mean_trajectory)),
                    y=mean_trajectory[:, dim],
                    error_y=dict(type='data', array=std_trajectory[:, dim]),
                    mode='lines+markers',
                    name=f'Latent Dim {dim+1}',
                    line=dict(width=3),
                    showlegend=False
                ),
                row=1, col=3
            )
        
        # 4. Metric determinant evolution
        mean_dets = np.mean(self.all_determinants, axis=0)
        std_dets = np.std(self.all_determinants, axis=0)
        
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(mean_dets)),
                y=mean_dets,
                error_y=dict(type='data', array=std_dets),
                mode='lines+markers',
                name='Mean Det',
                line=dict(color='red', width=3),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add individual determinant trajectories (sample)
        for i in range(min(10, len(self.all_determinants))):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(self.all_determinants[i])),
                    y=self.all_determinants[i],
                    mode='lines',
                    name=f'Traj {i}',
                    line=dict(width=1, color=colors[i]),
                    opacity=0.6,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 5. Trajectory clustering/selection view
        # Use first few timesteps for clustering
        first_timestep_mask = timesteps == 0
        fig.add_trace(
            go.Scatter(
                x=pca_coords[first_timestep_mask, 0],
                y=pca_coords[first_timestep_mask, 1],
                mode='markers',
                name='Starting Points',
                                    marker=dict(
                        size=12,
                        color=trajectory_ids[first_timestep_mask],
                        colorscale='plotly3',
                        showscale=False
                    ),
                hovertemplate="Start PC1: %{x:.3f}<br>Start PC2: %{y:.3f}<br>Trajectory: %{marker.color}<extra></extra>",
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 6. Detailed view - show metric centroids
        if hasattr(self.model, 'modular_metric') and self.model.modular_metric is not None:
            centroids = self.model.modular_metric.centroids.cpu().numpy()
            fig.add_trace(
                go.Scatter(
                    x=centroids[:, 0],
                    y=centroids[:, 1],
                    mode='markers',
                    name='Metric Centroids',
                    marker=dict(
                        size=15,
                        color='black',
                        symbol='x',
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate="Centroid: %{pointNumber}<br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<extra></extra>",
                    showlegend=False
                ),
                row=2, col=3
            )
            
            # Add data points colored by closest centroid
            distances = torch.cdist(torch.tensor(all_points), self.model.modular_metric.centroids)
            closest_centroids = torch.argmin(distances, dim=1).numpy()
            
            fig.add_trace(
                go.Scatter(
                    x=all_points[:, 0],
                    y=all_points[:, 1],
                    mode='markers',
                    name='Data Points',
                    marker=dict(
                        size=4,
                        color=closest_centroids,
                        colorscale='plotly3',
                        showscale=False,
                        opacity=0.6
                    ),
                    hovertemplate="Z1: %{x:.3f}<br>Z2: %{y:.3f}<br>Closest Centroid: %{marker.color}<extra></extra>",
                    showlegend=False
                ),
                row=2, col=3
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title=dict(
                text="🎭 Interactive Temporal Trajectory Analysis - Riemannian VAE",
                x=0.5,
                font=dict(size=20)
            ),
            showlegend=False,
            hovermode='closest'
        )
        
        # Update axes
        fig.update_xaxes(title="PC1", row=1, col=1)
        fig.update_yaxes(title="PC2", row=1, col=1)
        fig.update_xaxes(title="t-SNE 1", row=1, col=2)
        fig.update_yaxes(title="t-SNE 2", row=1, col=2)
        fig.update_xaxes(title="Timestep", row=1, col=3)
        fig.update_yaxes(title="Latent Value", row=1, col=3)
        fig.update_xaxes(title="Timestep", row=2, col=1)
        fig.update_yaxes(title="Log₁₀ det(G⁻¹)", row=2, col=1)
        fig.update_xaxes(title="PC1", row=2, col=2)
        fig.update_yaxes(title="PC2", row=2, col=2)
        fig.update_xaxes(title="Latent Dim 1", row=2, col=3)
        fig.update_yaxes(title="Latent Dim 2", row=2, col=3)
        
        # Save interactive plot
        html_path = self.output_dir / "interactive_trajectory_explorer.html"
        fig.write_html(str(html_path))
        
        logger.info(f"✅ Interactive trajectory explorer saved to {html_path}")
        
        # Log to WandB if available
        try:
            wandb.log({"interactive_trajectory_explorer": wandb.Html(str(html_path))})
        except:
            pass
        
        return fig
    
    def create_trajectory_comparison_tool(self) -> None:
        """Create interactive tool to compare different trajectories."""
        logger.info("🔍 Creating trajectory comparison tool")
        
        # Select interesting trajectories (high/low determinant variance, different patterns)
        det_variances = [np.var(dets) for dets in self.all_determinants]
        sorted_indices = np.argsort(det_variances)
        
        # Pick diverse trajectories
        selected_indices = [
            sorted_indices[0],      # Lowest variance
            sorted_indices[-1],     # Highest variance
            sorted_indices[len(sorted_indices)//4],   # Lower quartile
            sorted_indices[3*len(sorted_indices)//4], # Upper quartile
            sorted_indices[len(sorted_indices)//2],   # Median
        ]
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "📊 Trajectory Comparison - Latent Space",
                "📈 Determinant Evolution",
                "🎨 Original Sequences (t=0,2,4,6,8)",
                "🔄 Reconstructed Sequences",
                "🌊 Latent Evolution Detailed",
                "📏 Metric Analysis"
            ],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]],
            vertical_spacing=0.1
        )
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # 1. Latent space trajectories
        for i, idx in enumerate(selected_indices):
            traj = self.all_trajectories[idx]
            timesteps = self.all_timestamps[idx]
            
            fig.add_trace(
                go.Scatter(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    mode='markers+lines',
                    name=f'Trajectory {idx}',
                    line=dict(color=colors[i], width=3),
                    marker=dict(
                        size=10,
                        color=timesteps,
                        colorscale='Viridis',
                        showscale=(i == 0),
                        colorbar=dict(title="Timestep") if i == 0 else None
                    ),
                    hovertemplate=f"Traj {idx}<br>Z1: %{{x:.3f}}<br>Z2: %{{y:.3f}}<br>Time: %{{marker.color}}<extra></extra>"
                ),
                row=1, col=1
            )
        
        # 2. Determinant evolution
        for i, idx in enumerate(selected_indices):
            dets = self.all_determinants[idx]
            timesteps = self.all_timestamps[idx]
            
            fig.add_trace(
                go.Scatter(
                    x=timesteps,
                    y=dets,
                    mode='lines+markers',
                    name=f'Det {idx}',
                    line=dict(color=colors[i], width=3),
                    marker=dict(size=8),
                    hovertemplate=f"Traj {idx}<br>Time: %{{x}}<br>Det: %{{y:.3f}}<extra></extra>",
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3 & 4. Sequence visualization (showing key frames)
        for i, idx in enumerate(selected_indices[:3]):  # Limit for readability
            orig_seq = self.all_sequences[idx]
            recon_seq = self.all_reconstructions[idx]
            
            # Show frames at t=0,2,4,6,8
            for t_idx, t in enumerate([0, 2, 4, 6, 8]):
                if t < len(orig_seq):
                    # Original
                    fig.add_trace(
                        go.Scatter(
                            x=[t_idx + i*0.1],
                            y=[i],
                            mode='markers',
                            name=f'Orig {idx}',
                            marker=dict(size=15, color=colors[i]),
                            hovertemplate=f"Original Traj {idx}<br>Frame {t}<extra></extra>",
                            showlegend=False
                        ),
                        row=2, col=1
                    )
                    
                    # Reconstructed
                    fig.add_trace(
                        go.Scatter(
                            x=[t_idx + i*0.1],
                            y=[i],
                            mode='markers',
                            name=f'Recon {idx}',
                            marker=dict(size=15, color=colors[i], symbol='square'),
                            hovertemplate=f"Reconstructed Traj {idx}<br>Frame {t}<extra></extra>",
                            showlegend=False
                        ),
                        row=2, col=2
                    )
        
        # 5. Detailed latent evolution
        for i, idx in enumerate(selected_indices):
            traj = self.all_trajectories[idx]
            timesteps = self.all_timestamps[idx]
            
            for dim in range(traj.shape[1]):
                fig.add_trace(
                    go.Scatter(
                        x=timesteps,
                        y=traj[:, dim],
                        mode='lines+markers',
                        name=f'Traj {idx} Dim {dim+1}',
                        line=dict(color=colors[i], dash='solid' if dim == 0 else 'dash', width=2),
                        showlegend=(i == 0)
                    ),
                    row=3, col=1
                )
        
        # 6. Metric analysis - distance to centroids over time
        if hasattr(self.model, 'modular_metric') and self.model.modular_metric is not None:
            centroids = self.model.modular_metric.centroids
            
            for i, idx in enumerate(selected_indices[:3]):
                traj = torch.tensor(self.all_trajectories[idx])
                timesteps = self.all_timestamps[idx]
                
                # Distance to closest centroid over time
                distances = []
                for t in range(len(traj)):
                    z_t = traj[t:t+1]
                    dists = torch.norm(z_t - centroids, dim=1)
                    min_dist = torch.min(dists)
                    distances.append(min_dist.numpy())
                
                fig.add_trace(
                    go.Scatter(
                        x=timesteps,
                        y=distances,
                        mode='lines+markers',
                        name=f'Distance {idx}',
                        line=dict(color=colors[i], width=2),
                        showlegend=False
                    ),
                    row=3, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title=dict(
                text="🔍 Trajectory Comparison Tool - Selected Diverse Patterns",
                x=0.5,
                font=dict(size=18)
            ),
            showlegend=True
        )
        
        # Save
        html_path = self.output_dir / "trajectory_comparison_tool.html"
        fig.write_html(str(html_path))
        
        logger.info(f"✅ Trajectory comparison tool saved to {html_path}")
        
        # Log to WandB
        try:
            wandb.log({"trajectory_comparison_tool": wandb.Html(str(html_path))})
        except:
            pass
    
    def create_3d_trajectory_visualization(self) -> None:
        """Create 3D interactive visualization with PCA + time as third dimension."""
        logger.info("🌍 Creating 3D trajectory visualization")
        
        # Combine trajectories for PCA
        all_points = np.vstack(self.all_trajectories)
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(all_points)
        
        # Reconstruct trajectory structure with PCA coordinates
        trajectory_pca = []
        start_idx = 0
        for traj in self.all_trajectories:
            end_idx = start_idx + len(traj)
            trajectory_pca.append(pca_coords[start_idx:end_idx])
            start_idx = end_idx
        
        # Create 3D plot
        fig = go.Figure()
        
        # Add trajectories as 3D lines (PC1, PC2, Time)
        colors = px.colors.qualitative.Set3 * 10
        
        for i, (traj_pca, dets) in enumerate(zip(trajectory_pca[:30], self.all_determinants[:30])):
            timesteps = self.all_timestamps[i]
            
            fig.add_trace(
                go.Scatter3d(
                    x=traj_pca[:, 0],
                    y=traj_pca[:, 1],
                    z=timesteps,
                    mode='markers+lines',
                    name=f'Trajectory {i}',
                    line=dict(color=colors[i % len(colors)], width=4),
                    marker=dict(
                        size=8,
                        color=dets,
                        colorscale='RdYlBu_r',
                        showscale=(i == 0),
                        colorbar=dict(title="Log₁₀ det(G⁻¹)") if i == 0 else None
                    ),
                    hovertemplate=f"Traj {i}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>Time: %{{z}}<br>Det: %{{marker.color:.3f}}<extra></extra>",
                    showlegend=False
                )
            )
        
        # Add metric centroids as reference (at z=0 plane)
        if hasattr(self.model, 'modular_metric') and self.model.modular_metric is not None:
            centroids = self.model.modular_metric.centroids.cpu().numpy()
            centroids_pca = pca.transform(centroids)
            
            fig.add_trace(
                go.Scatter3d(
                    x=centroids_pca[:, 0],
                    y=centroids_pca[:, 1],
                    z=np.zeros(len(centroids)),
                    mode='markers',
                    name='Metric Centroids',
                    marker=dict(
                        size=12,
                        color='black',
                        symbol='x'
                    ),
                    hovertemplate="Centroid %{pointNumber}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
                    showlegend=True
                )
            )
        
        fig.update_layout(
            title=dict(
                text="🌍 3D Interactive Trajectory Visualization (PC1 × PC2 × Time)",
                x=0.5,
                font=dict(size=18)
            ),
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="Timestep",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=800
        )
        
        # Save
        html_path = self.output_dir / "3d_trajectory_visualization.html"
        fig.write_html(str(html_path))
        
        logger.info(f"✅ 3D trajectory visualization saved to {html_path}")
        
        # Log to WandB
        try:
            wandb.log({"3d_trajectory_visualization": wandb.Html(str(html_path))})
        except:
            pass
    
    def create_interactive_sequence_player(self) -> None:
        """Create interactive sequence player with synchronized plots."""
        logger.info("🎬 Creating interactive sequence player")
        
        # Select a few representative sequences
        selected_sequences = self.all_sequences[:10]
        selected_reconstructions = self.all_reconstructions[:10]
        selected_trajectories = self.all_trajectories[:10]
        selected_determinants = self.all_determinants[:10]
        
        # Create frames for animation
        frames = []
        n_timesteps = len(selected_trajectories[0])
        
        for t in range(n_timesteps):
            frame_data = []
            
            # Latent space positions at time t
            latent_positions = np.array([traj[t] for traj in selected_trajectories])
            determinants_t = np.array([dets[t] for dets in selected_determinants])
            
            # Add scatter plot for this timestep
            frame_data.append(
                go.Scatter(
                    x=latent_positions[:, 0],
                    y=latent_positions[:, 1],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=determinants_t,
                        colorscale='RdYlBu_r',
                        showscale=True,
                        colorbar=dict(title="Log₁₀ det(G⁻¹)")
                    ),
                    text=[f"Seq {i}" for i in range(len(latent_positions))],
                    hovertemplate="Seq: %{text}<br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<br>Det: %{marker.color:.3f}<extra></extra>",
                    name=f"t={t}"
                )
            )
            
            frames.append(go.Frame(data=frame_data, name=str(t)))
        
        # Create base figure
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        # Add play/pause controls
        fig.update_layout(
            title="🎬 Interactive Sequence Player - Latent Space Evolution",
            xaxis_title="Latent Dimension 1",
            yaxis_title="Latent Dimension 2",
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                       "fromcurrent": True, "transition": {"duration": 300}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                         "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Timestep:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], {
                            "frame": {"duration": 300, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 300}
                        }],
                        "label": f"t={f.name}",
                        "method": "animate"
                    } for f in frames
                ]
            }]
        )
        
        # Save
        html_path = self.output_dir / "interactive_sequence_player.html"
        fig.write_html(str(html_path))
        
        logger.info(f"✅ Interactive sequence player saved to {html_path}")
        
        # Log to WandB
        try:
            wandb.log({"interactive_sequence_player": wandb.Html(str(html_path))})
        except:
            pass
    
    def use_existing_interactive_system(self) -> None:
        """Use the existing interactive visualization system from the project."""
        logger.info("🎭 Using existing interactive visualization system")
        
        # Create visualization config
        viz_config = DictConfig({
            'level': 'full',
            'basic_frequency': 1,
            'manifold_frequency': 1,
            'interactive_frequency': 1,
            'enable_fancy_plots': True,
            'max_sequences': 50,
            'sequence_viz_count': 20,
            'enable_pca_plots': True,
            'enable_flow_analysis': True,
            'enable_comprehensive': True,
            'save_plots': True,
            'log_to_wandb': True
        })
        
        # Create interactive visualizer
        interactive_viz = InteractiveVisualizations(
            model=self.model,
            config=viz_config,
            device=self.device,
            output_dir=str(self.output_dir)
        )
        
        # Get a sample batch for visualization
        sample_batch = next(iter(self.test_loader))
        sample_batch = sample_batch.to(self.device)
        if len(sample_batch.shape) == 4:
            sample_batch = sample_batch.unsqueeze(0)
        
        # Create rich visualizations
        logger.info("🎚️ Creating geodesic sliders...")
        interactive_viz.create_geodesic_sliders(sample_batch, epoch=0)
        
        logger.info("🎯 Creating metric slider visualization...")
        interactive_viz.create_metric_slider_visualization(sample_batch, epoch=0)
        
        logger.info("🎬 Creating sequence slider visualization...")
        interactive_viz.create_sequence_slider_visualization(sample_batch, epoch=0)
        
        logger.info("⛰️ Creating time curvature heatmap...")
        interactive_viz.create_time_curvature_heatmap(sample_batch, epoch=0)
        
        logger.info("🌐 Creating HTML latent space...")
        interactive_viz.create_html_latent_space(sample_batch, epoch=0)
        
        logger.info("✨ Creating fancy geodesics...")
        interactive_viz.create_fancy_geodesics(sample_batch, epoch=0)
        
        logger.info("🎞️ Creating temporal animation...")
        interactive_viz.create_temporal_animation(sample_batch, epoch=0)
        
        logger.info("✅ Existing interactive system visualizations created")
    
    def run_comprehensive_analysis(self) -> None:
        """Run comprehensive interactive analysis."""
        logger.info("🚀 Starting comprehensive interactive temporal analysis")
        
        # Initialize WandB
        wandb.init(
            project="rlvae-interactive-analysis",
            name=f"interactive_temporal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "checkpoint": self.checkpoint_path,
                "device": self.device,
                "analysis_type": "comprehensive_interactive"
            }
        )
        
        # Load model and data
        self.load_model_and_data()
        
        # Collect rich trajectory data
        self.collect_rich_trajectory_data(n_sequences=100)
        
        # Create comprehensive interactive visualizations
        self.create_interactive_trajectory_explorer()
        self.create_trajectory_comparison_tool()
        self.create_3d_trajectory_visualization()
        self.create_interactive_sequence_player()
        
        # Use existing interactive system
        self.use_existing_interactive_system()
        
        logger.info("🎉 Comprehensive interactive analysis completed!")
        logger.info(f"📁 All results saved in: {self.output_dir}")
        
        wandb.finish()


def main():
    """Main execution."""
    checkpoint_path = "outputs/checkpoints/epoch=197-val_loss=6.402.ckpt"
    
    analyzer = InteractiveTemporalAnalyzer(checkpoint_path)
    analyzer.run_comprehensive_analysis()


if __name__ == "__main__":
    main() 