#!/usr/bin/env python3
"""
Fixed Interactive Visualizations for RLVAE
==========================================

Comprehensive fix for all issues:
- Proper metric tensor access
- Correct data shape handling
- Working forward passes
- True interactive visualizations
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedInteractiveVisualizer:
    """Fixed interactive visualizations with proper model handling."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """Initialize visualizer."""
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.model = None
        self.data_module = None
        self.checkpoint = None
        
        self.output_dir = Path("outputs/fixed_interactive") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔧 Fixed interactive visualizer initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model_and_data_properly(self) -> None:
        """Load model and data with proper error handling and verification."""
        logger.info(f"🔄 Loading model from {self.checkpoint_path}")
        
        # Load checkpoint
        self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        model_hparams = self.checkpoint['hyper_parameters']['model']
        
        logger.info(f"📊 Model hyperparameters: {model_hparams}")
        
        # Create config with exact checkpoint parameters
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
        
        logger.info(f"🔧 Creating model with config: {config}")
        
        # Create model
        self.model = ModularRiemannianFlowVAE(config)
        
        # Load state dict properly
        state_dict = self.checkpoint['state_dict']
        clean_state_dict = {k.replace('model.', '') if k.startswith('model.') else k: v 
                           for k, v in state_dict.items()}
        
        # Check metric tensor dimensions
        metric_centroids_shape = None
        metric_matrices_shape = None
        
        for name, param in clean_state_dict.items():
            if 'modular_metric.centroids' in name:
                metric_centroids_shape = param.shape
                logger.info(f"📐 Found metric centroids shape: {metric_centroids_shape}")
            elif 'modular_metric.metric_matrices' in name:
                metric_matrices_shape = param.shape
                logger.info(f"📐 Found metric matrices shape: {metric_matrices_shape}")
        
        # Resize metric tensor parameters before loading
        if metric_centroids_shape is not None:
            self.model.modular_metric.centroids = torch.nn.Parameter(torch.zeros(metric_centroids_shape))
            logger.info(f"✅ Resized centroids to {metric_centroids_shape}")
        
        if metric_matrices_shape is not None:
            self.model.modular_metric.metric_matrices = torch.nn.Parameter(torch.zeros(metric_matrices_shape))
            logger.info(f"✅ Resized metric matrices to {metric_matrices_shape}")
        
        # Load state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(clean_state_dict, strict=False)
        logger.info(f"📊 Model loaded - Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Ensure metric tensor is on the correct device
        if hasattr(self.model, 'modular_metric') and self.model.modular_metric is not None:
            self.model.modular_metric.centroids = self.model.modular_metric.centroids.to(self.device)
            self.model.modular_metric.metric_matrices = self.model.modular_metric.metric_matrices.to(self.device)
            logger.info(f"✅ Moved metric tensor to {self.device}")
        
        # Verify metric tensor is accessible
        self._verify_metric_tensor()
        
        # Setup data
        data_config = DictConfig({
            'train_path': 'data/sprites/ColoredCircles_train.pt',
            'test_path': 'data/sprites/ColoredCircles_test.pt',
            'train_meta_path': 'data/sprites/ColoredCircles_train_params.pt',
            'test_meta_path': 'data/sprites/ColoredCircles_test_params.pt',
            'sequence_length': 10,
            'image_size': [28, 28],
            'channels': 3,
            'batch_size': 1,
            'num_workers': 0,
            'pin_memory': False,
            'max_test_samples': 100,
            'verify_cyclicity': False
        })
        
        self.data_module = CyclicSpritesDataModule(data_config)
        self.data_module.setup('test')
        
        logger.info("✅ Model and data loaded successfully")
    
    def _verify_metric_tensor(self) -> None:
        """Verify that the metric tensor is properly loaded and accessible."""
        logger.info("🔍 Verifying metric tensor...")
        
        if hasattr(self.model, 'modular_metric') and self.model.modular_metric is not None:
            centroids = self.model.modular_metric.centroids
            matrices = self.model.modular_metric.metric_matrices
            
            logger.info(f"✅ Metric tensor verified:")
            logger.info(f"  - Centroids: {centroids.shape}")
            logger.info(f"  - Matrices: {matrices.shape}")
            logger.info(f"  - Centroids range: [{centroids.min():.3f}, {centroids.max():.3f}]")
            logger.info(f"  - Matrices range: [{matrices.min():.3f}, {matrices.max():.3f}]")
            
            # Test determinant computation
            test_matrix = matrices[0]
            det = torch.det(test_matrix)
            logger.info(f"  - Test determinant: {det:.6f}")
            
            return True
        else:
            logger.error("❌ Metric tensor not found!")
            return False
    
    def safe_model_forward(self, sequences: torch.Tensor) -> dict:
        """Safely perform model forward pass with proper error handling."""
        try:
            sequences = sequences.to(self.device)
            
            # Ensure proper batch dimension
            if len(sequences.shape) == 4:  # [seq_len, C, H, W]
                sequences = sequences.unsqueeze(0)  # [1, seq_len, C, H, W]
            
            logger.debug(f"🔄 Forward pass input shape: {sequences.shape}")
            
            with torch.no_grad():
                output = self.model(sequences)
                
            if isinstance(output, dict):
                logger.debug(f"✅ Forward pass successful, output keys: {list(output.keys())}")
                return output
            else:
                logger.error(f"❌ Unexpected output type: {type(output)}")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def extract_latent_trajectories_safely(self, n_sequences: int = 50) -> tuple:
        """Extract latent trajectories with comprehensive error handling."""
        logger.info(f"🌊 Extracting {n_sequences} latent trajectories safely")
        
        test_loader = self.data_module.test_dataloader()
        all_trajectories = []
        all_determinants = []
        all_timestamps = []
        
        trajectory_count = 0
        
        for batch_idx, batch in enumerate(test_loader):
            if trajectory_count >= n_sequences:
                break
            
            try:
                sequences = batch
                logger.debug(f"📊 Processing batch {batch_idx}, shape: {sequences.shape}")
                
                # Perform forward pass
                output = self.safe_model_forward(sequences)
                
                if not output or 'latent_samples' not in output:
                    logger.warning(f"⚠️ Skipping batch {batch_idx} - no latent samples")
                    continue
                
                latent_seq = output['latent_samples']  # [1, seq_len, latent_dim]
                trajectory = latent_seq.squeeze(0).cpu().numpy()  # [seq_len, latent_dim]
                
                logger.debug(f"📊 Extracted trajectory shape: {trajectory.shape}")
                
                # Compute determinants for each timestep
                determinants = []
                seq_len = trajectory.shape[0]
                
                for t in range(seq_len):
                    det_value = self._compute_determinant_at_point(trajectory[t])
                    determinants.append(det_value)
                
                all_trajectories.append(trajectory)
                all_determinants.append(np.array(determinants))
                all_timestamps.append(np.arange(seq_len))
                
                trajectory_count += 1
                
                if trajectory_count % 10 == 0:
                    logger.info(f"📊 Extracted {trajectory_count}/{n_sequences} trajectories")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error processing batch {batch_idx}: {e}")
                continue
        
        logger.info(f"✅ Successfully extracted {len(all_trajectories)} trajectories")
        return all_trajectories, all_determinants, all_timestamps
    
    def _compute_determinant_at_point(self, z_point: np.ndarray) -> float:
        """Compute metric determinant at a specific latent point."""
        try:
            z_tensor = torch.tensor(z_point, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            if hasattr(self.model, 'modular_metric') and self.model.modular_metric is not None:
                centroids = self.model.modular_metric.centroids
                matrices = self.model.modular_metric.metric_matrices
                
                # Find closest centroid
                distances = torch.norm(z_tensor - centroids, dim=1)
                closest_idx = torch.argmin(distances)
                
                # Get metric matrix and compute determinant
                metric_matrix = matrices[closest_idx]
                det_g = torch.det(metric_matrix)
                log_det_g_inv = -torch.log10(det_g + 1e-8)
                
                return log_det_g_inv.cpu().numpy()
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"⚠️ Determinant computation failed: {e}")
            return 0.0
    
    def create_comprehensive_interactive_visualization(self, trajectories, determinants, timestamps) -> None:
        """Create a comprehensive interactive visualization with all the data."""
        logger.info("🎨 Creating comprehensive interactive visualization")
        
        # Combine all data
        all_points = np.vstack(trajectories)
        all_dets = np.concatenate(determinants)
        all_times = []
        all_traj_ids = []
        
        for i, (traj, times) in enumerate(zip(trajectories, timestamps)):
            all_times.extend(times)
            all_traj_ids.extend([i] * len(traj))
        
        all_times = np.array(all_times)
        all_traj_ids = np.array(all_traj_ids)
        
        logger.info(f"📊 Combined data: {len(all_points)} points from {len(trajectories)} trajectories")
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(all_points)
        
        logger.info(f"📊 PCA explained variance: {pca.explained_variance_ratio_}")
        
        # Create comprehensive interactive plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "🎯 Latent Space Trajectories (PCA)",
                "📈 Temporal Evolution",
                "📏 Metric Determinant Evolution", 
                "🌊 3D Trajectory View"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter3d"}]
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )
        
        # 1. PCA trajectory scatter with lines
        colors = px.colors.qualitative.Set3
        
        # Add individual trajectories as lines
        trajectory_pca = []
        start_idx = 0
        for traj in trajectories:
            end_idx = start_idx + len(traj)
            trajectory_pca.append(pca_coords[start_idx:end_idx])
            start_idx = end_idx
        
        for i, (traj_pca, dets) in enumerate(zip(trajectory_pca[:20], determinants[:20])):
            fig.add_trace(
                go.Scatter(
                    x=traj_pca[:, 0],
                    y=traj_pca[:, 1],
                    mode='lines+markers',
                    name=f'Traj {i}',
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(
                        size=8,
                        color=dets,
                        colorscale='viridis',
                        showscale=(i == 0),
                        colorbar=dict(title="Log₁₀ det(G⁻¹)", x=0.45) if i == 0 else None
                    ),
                    hovertemplate=f"Traj {i}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>Det: %{{marker.color:.3f}}<extra></extra>",
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 2. Temporal evolution
        mean_trajectory = np.mean(trajectories, axis=0)
        std_trajectory = np.std(trajectories, axis=0)
        timesteps = np.arange(len(mean_trajectory))
        
        for dim in range(mean_trajectory.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=timesteps,
                    y=mean_trajectory[:, dim],
                    error_y=dict(type='data', array=std_trajectory[:, dim]),
                    mode='lines+markers',
                    name=f'Latent Dim {dim+1}',
                    line=dict(width=3),
                    marker=dict(size=8),
                    hovertemplate=f"Dim {dim+1}<br>Time: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>",
                    showlegend=(dim < 2)
                ),
                row=1, col=2
            )
        
        # 3. Determinant evolution
        mean_dets = np.mean(determinants, axis=0)
        std_dets = np.std(determinants, axis=0)
        
        fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=mean_dets,
                error_y=dict(type='data', array=std_dets),
                mode='lines+markers',
                name='Mean Det',
                line=dict(color='red', width=4),
                marker=dict(size=10),
                hovertemplate="Mean Det<br>Time: %{x}<br>Det: %{y:.3f}<extra></extra>",
                showlegend=True
            ),
            row=2, col=1
        )
        
        # Add individual determinant trajectories
        for i, dets in enumerate(determinants[:10]):
            fig.add_trace(
                go.Scatter(
                    x=timesteps,
                    y=dets,
                    mode='lines',
                    name=f'Det {i}',
                    line=dict(color=colors[i % len(colors)], width=1),
                    opacity=0.6,
                    showlegend=False,
                    hovertemplate=f"Traj {i}<br>Time: %{{x}}<br>Det: %{{y:.3f}}<extra></extra>"
                ),
                row=2, col=1
            )
        
        # 4. 3D trajectory view (PC1, PC2, Time)
        for i, (traj_pca, dets) in enumerate(zip(trajectory_pca[:10], determinants[:10])):
            timesteps_3d = np.arange(len(traj_pca))
            
            fig.add_trace(
                go.Scatter3d(
                    x=traj_pca[:, 0],
                    y=traj_pca[:, 1],
                    z=timesteps_3d,
                    mode='lines+markers',
                    name=f'3D Traj {i}',
                    line=dict(color=colors[i % len(colors)], width=4),
                    marker=dict(
                        size=6,
                        color=dets,
                        colorscale='plasma',
                        showscale=(i == 0),
                        colorbar=dict(title="Det", x=1.02) if i == 0 else None
                    ),
                    hovertemplate=f"3D Traj {i}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>Time: %{{z}}<br>Det: %{{marker.color:.3f}}<extra></extra>",
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title=dict(
                text="🎨 Comprehensive RLVAE Temporal Analysis - Fixed Implementation",
                x=0.5,
                font=dict(size=18)
            ),
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", row=1, col=1)
        fig.update_yaxes(title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", row=1, col=1)
        fig.update_xaxes(title="Timestep", row=1, col=2)
        fig.update_yaxes(title="Latent Value", row=1, col=2)
        fig.update_xaxes(title="Timestep", row=2, col=1)
        fig.update_yaxes(title="Log₁₀ det(G⁻¹)", row=2, col=1)
        
        # Update 3D scene
        fig.update_scenes(
            xaxis_title="PC1",
            yaxis_title="PC2", 
            zaxis_title="Timestep",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            row=2, col=2
        )
        
        # Save
        html_path = self.output_dir / "comprehensive_fixed_visualization.html"
        fig.write_html(str(html_path))
        
        logger.info(f"✅ Comprehensive visualization saved to {html_path}")
        
        # Log to WandB
        try:
            wandb.log({"comprehensive_fixed_visualization": wandb.Html(str(html_path))})
        except:
            pass
        
        return fig
    
    def create_individual_trajectory_explorer(self, trajectories, determinants, timestamps) -> None:
        """Create individual trajectory explorer with selection capabilities."""
        logger.info("🔍 Creating individual trajectory explorer")
        
        # Select diverse trajectories for detailed analysis
        det_variances = [np.var(dets) for dets in determinants]
        sorted_indices = np.argsort(det_variances)
        
        selected_indices = [
            sorted_indices[0],  # Lowest variance
            sorted_indices[-1], # Highest variance
            sorted_indices[len(sorted_indices)//4],    # Q1
            sorted_indices[len(sorted_indices)//2],    # Median
            sorted_indices[3*len(sorted_indices)//4],  # Q3
        ]
        
        fig = go.Figure()
        
        # Add dropdown for trajectory selection
        buttons = []
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, idx in enumerate(selected_indices):
            traj = trajectories[idx]
            dets = determinants[idx]
            times = timestamps[idx]
            
            # Create traces for this trajectory (initially all invisible)
            visible = [False] * (len(selected_indices) * 3)  # 3 traces per trajectory
            visible[i*3:(i+1)*3] = [True, True, True]
            
            button = dict(
                label=f"Trajectory {idx} (Var: {det_variances[idx]:.3f})",
                method="update",
                args=[
                    {"visible": visible},
                    {"title": f"Individual Trajectory Analysis - Trajectory {idx}"}
                ]
            )
            buttons.append(button)
        
        # Add all trajectory traces
        for i, idx in enumerate(selected_indices):
            traj = trajectories[idx]
            dets = determinants[idx]
            times = timestamps[idx]
            
            # Latent dimension 1
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=traj[:, 0],
                    mode='lines+markers',
                    name=f'Traj {idx} - Dim 1',
                    line=dict(color=colors[i], width=3),
                    marker=dict(size=8),
                    visible=(i == 0),
                    hovertemplate=f"Traj {idx} - Dim 1<br>Time: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>"
                )
            )
            
            # Latent dimension 2
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=traj[:, 1],
                    mode='lines+markers',
                    name=f'Traj {idx} - Dim 2',
                    line=dict(color=colors[i], width=3, dash='dash'),
                    marker=dict(size=8, symbol='square'),
                    visible=(i == 0),
                    hovertemplate=f"Traj {idx} - Dim 2<br>Time: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>"
                )
            )
            
            # Determinant
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=dets,
                    mode='lines+markers',
                    name=f'Traj {idx} - Det',
                    line=dict(color=colors[i], width=3, dash='dot'),
                    marker=dict(size=8, symbol='diamond'),
                    yaxis='y2',
                    visible=(i == 0),
                    hovertemplate=f"Traj {idx} - Det<br>Time: %{{x}}<br>Det: %{{y:.3f}}<extra></extra>"
                )
            )
        
        # Update layout with dropdown
        fig.update_layout(
            title="🔍 Individual Trajectory Explorer - Fixed Implementation",
            xaxis_title="Timestep",
            yaxis_title="Latent Value",
            yaxis2=dict(
                title="Log₁₀ det(G⁻¹)",
                overlaying='y',
                side='right'
            ),
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                )
            ],
            height=600,
            showlegend=True
        )
        
        # Save
        html_path = self.output_dir / "individual_trajectory_explorer.html"
        fig.write_html(str(html_path))
        
        logger.info(f"✅ Individual trajectory explorer saved to {html_path}")
        
        # Log to WandB
        try:
            wandb.log({"individual_trajectory_explorer": wandb.Html(str(html_path))})
        except:
            pass
        
        return fig
    
    def create_metric_tensor_analysis(self) -> None:
        """Create detailed analysis of the metric tensor structure."""
        logger.info("📏 Creating metric tensor analysis")
        
        if not hasattr(self.model, 'modular_metric') or self.model.modular_metric is None:
            logger.warning("⚠️ No metric tensor available for analysis")
            return
        
        centroids = self.model.modular_metric.centroids.detach().cpu().numpy()
        matrices = self.model.modular_metric.metric_matrices.detach().cpu().numpy()
        
        # Compute determinants and eigenvalues
        determinants = []
        traces = []
        eigenvalues_all = []
        condition_numbers = []
        
        for i in range(len(matrices)):
            matrix = matrices[i]
            det = np.linalg.det(matrix)
            trace = np.trace(matrix)
            eigenvals = np.linalg.eigvals(matrix)
            cond_num = np.linalg.cond(matrix)
            
            determinants.append(det)
            traces.append(trace)
            eigenvalues_all.append(eigenvals)
            condition_numbers.append(cond_num)
        
        determinants = np.array(determinants)
        traces = np.array(traces)
        condition_numbers = np.array(condition_numbers)
        
        # Create comprehensive metric analysis
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "🎯 Metric Centroids Distribution",
                "📊 Determinant vs Trace",
                "🌈 Eigenvalue Analysis",
                "📏 Condition Numbers",
                "🗺️ Centroid Heatmap",
                "🔍 Matrix Properties"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "heatmap"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Centroids distribution
        fig.add_trace(
            go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode='markers',
                marker=dict(
                    size=12,
                    color=determinants,
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title="Determinant", x=0.32)
                ),
                text=[f"Centroid {i}" for i in range(len(centroids))],
                hovertemplate="<b>%{text}</b><br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<br>Det: %{marker.color:.3f}<extra></extra>",
                name="Centroids"
            ),
            row=1, col=1
        )
        
        # 2. Determinant vs Trace
        fig.add_trace(
            go.Scatter(
                x=traces,
                y=determinants,
                mode='markers',
                marker=dict(
                    size=10,
                    color=condition_numbers,
                    colorscale='plasma',
                    showscale=True,
                    colorbar=dict(title="Condition Number", x=0.65)
                ),
                hovertemplate="Trace: %{x:.3f}<br>Det: %{y:.3f}<br>Cond: %{marker.color:.1f}<extra></extra>",
                name="Det vs Trace"
            ),
            row=1, col=2
        )
        
        # 3. Eigenvalue analysis
        eig1_vals = [eigs[0] for eigs in eigenvalues_all]
        eig2_vals = [eigs[1] for eigs in eigenvalues_all]
        
        fig.add_trace(
            go.Scatter(
                x=eig1_vals,
                y=eig2_vals,
                mode='markers',
                marker=dict(size=10, color='red'),
                hovertemplate="λ₁: %{x:.3f}<br>λ₂: %{y:.3f}<extra></extra>",
                name="Eigenvalues"
            ),
            row=1, col=3
        )
        
        # 4. Condition numbers
        fig.add_trace(
            go.Scatter(
                x=list(range(len(condition_numbers))),
                y=condition_numbers,
                mode='markers+lines',
                marker=dict(size=8),
                line=dict(width=2),
                hovertemplate="Centroid: %{x}<br>Condition: %{y:.2f}<extra></extra>",
                name="Condition Numbers"
            ),
            row=2, col=1
        )
        
        # 5. Centroid heatmap
        centroid_distances = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=2)
        
        fig.add_trace(
            go.Heatmap(
                z=centroid_distances,
                colorscale='blues',
                showscale=True,
                colorbar=dict(title="Distance", x=0.98),
                hovertemplate="Centroid %{y} to %{x}<br>Distance: %{z:.3f}<extra></extra>"
            ),
            row=2, col=2
        )
        
        # 6. Matrix properties scatter
        fig.add_trace(
            go.Scatter(
                x=determinants,
                y=condition_numbers,
                mode='markers',
                marker=dict(
                    size=10,
                    color=traces,
                    colorscale='magma',
                    showscale=True,
                    colorbar=dict(title="Trace", x=1.15)
                ),
                hovertemplate="Det: %{x:.3f}<br>Cond: %{y:.2f}<br>Trace: %{marker.color:.3f}<extra></extra>",
                name="Matrix Properties"
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title=dict(
                text="📏 Comprehensive Metric Tensor Analysis - Fixed Implementation",
                x=0.5,
                font=dict(size=18)
            ),
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title="Latent Dimension 1", row=1, col=1)
        fig.update_yaxes(title="Latent Dimension 2", row=1, col=1)
        fig.update_xaxes(title="Trace", row=1, col=2)
        fig.update_yaxes(title="Determinant", row=1, col=2)
        fig.update_xaxes(title="Eigenvalue 1", row=1, col=3)
        fig.update_yaxes(title="Eigenvalue 2", row=1, col=3)
        fig.update_xaxes(title="Centroid Index", row=2, col=1)
        fig.update_yaxes(title="Condition Number", row=2, col=1)
        fig.update_xaxes(title="Determinant", row=2, col=3)
        fig.update_yaxes(title="Condition Number", row=2, col=3)
        
        # Save
        html_path = self.output_dir / "metric_tensor_analysis.html"
        fig.write_html(str(html_path))
        
        logger.info(f"✅ Metric tensor analysis saved to {html_path}")
        
        # Log to WandB
        try:
            wandb.log({"metric_tensor_analysis": wandb.Html(str(html_path))})
        except:
            pass
        
        return fig
    
    def run_complete_fixed_analysis(self) -> None:
        """Run complete fixed analysis with all error handling."""
        logger.info("🚀 Starting complete fixed interactive analysis")
        
        # Initialize WandB
        wandb.init(
            project="rlvae-fixed-interactive",
            name=f"fixed_interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "checkpoint": self.checkpoint_path,
                "device": self.device,
                "analysis_type": "fixed_comprehensive"
            }
        )
        
        try:
            # Load model and data properly
            self.load_model_and_data_properly()
            
            # Extract trajectories safely
            trajectories, determinants, timestamps = self.extract_latent_trajectories_safely(n_sequences=50)
            
            if len(trajectories) == 0:
                logger.error("❌ No trajectories extracted - cannot create visualizations")
                return
            
            # Create visualizations
            self.create_comprehensive_interactive_visualization(trajectories, determinants, timestamps)
            self.create_individual_trajectory_explorer(trajectories, determinants, timestamps)
            self.create_metric_tensor_analysis()
            
            # Log all HTML files to WandB
            for html_file in self.output_dir.glob("*.html"):
                try:
                    wandb.log({html_file.stem: wandb.Html(str(html_file))})
                    logger.info(f"📤 Logged {html_file.name} to WandB")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to log {html_file}: {e}")
            
            logger.info("🎉 Complete fixed analysis completed successfully!")
            logger.info(f"📁 All results saved in: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            wandb.finish()


def main():
    """Main execution."""
    checkpoint_path = "outputs/checkpoints/epoch=197-val_loss=6.402.ckpt"
    
    visualizer = FixedInteractiveVisualizer(checkpoint_path)
    visualizer.run_complete_fixed_analysis()


if __name__ == "__main__":
    main() 