#!/usr/bin/env python3
"""
Ultimate Comprehensive RLVAE Explorer
====================================

Single file with ALL visualizations:
1. Interactive sequence slider with PCA trajectories
2. Reconstruction galleries for all sequences
3. Dense latent space with many points
4. METRIC LANDSCAPE: Mountains and valleys of latent space
5. 3D visualizations and temporal analysis

Everything in one complete interactive package!
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
import wandb
import base64
import io
from PIL import Image

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

class UltimateComprehensiveExplorer:
    """Ultimate explorer with everything in one file including metric landscape."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """Initialize explorer."""
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.model = None
        self.data_module = None
        
        self.output_dir = Path("outputs/ultimate_comprehensive") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Ultimate comprehensive explorer initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model_complete(self) -> None:
        """Load model with complete setup."""
        logger.info(f"🔄 Loading model for comprehensive analysis")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model_hparams = checkpoint['hyper_parameters']['model']
        
        # Create config
        config = DictConfig(model_hparams)
        config.pretrained = {'encoder_path': None, 'decoder_path': None, 'metric_path': None}
        
        # Create and load model
        self.model = ModularRiemannianFlowVAE(config)
        
        # Load state dict
        state_dict = checkpoint['state_dict']
        clean_state_dict = {k.replace('model.', '') if k.startswith('model.') else k: v.to(self.device) 
                           for k, v in state_dict.items()}
        
        # Resize metric tensor
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
            'sequence_length': 10, 'image_size': [28, 28], 'channels': 3,
            'batch_size': 1, 'num_workers': 0, 'pin_memory': False,
            'max_test_samples': 100, 'verify_cyclicity': False
        })
        
        self.data_module = CyclicSpritesDataModule(data_config)
        self.data_module.setup('test')
        
        logger.info("✅ Model loaded for comprehensive analysis")
    
    def _compute_metric_determinant(self, z_point: torch.Tensor) -> float:
        """Compute metric determinant."""
        try:
            if hasattr(self.model, 'modular_metric') and self.model.modular_metric is not None:
                centroids = self.model.modular_metric.centroids
                matrices = self.model.modular_metric.metric_matrices
                
                distances = torch.norm(z_point - centroids, dim=1)
                closest_idx = torch.argmin(distances)
                metric_matrix = matrices[closest_idx]
                det_g = torch.det(metric_matrix)
                return -torch.log10(det_g + 1e-8).cpu().item()
            return 0.0
        except:
            return 0.0
    
    def tensor_to_base64(self, tensor_img: torch.Tensor) -> str:
        """Convert tensor image to base64."""
        if len(tensor_img.shape) == 3:
            img_np = tensor_img.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = tensor_img.cpu().numpy()
        
        img_np = (img_np * 255).astype(np.uint8)
        
        if img_np.shape[2] == 3:
            img_pil = Image.fromarray(img_np, 'RGB')
        else:
            img_pil = Image.fromarray(img_np[:, :, 0], 'L')
        
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_base64}"
    
    def extract_comprehensive_data(self, n_sequences: int = 50) -> tuple:
        """Extract all data needed for comprehensive analysis."""
        logger.info(f"🌊 Extracting comprehensive data from {n_sequences} sequences")
        
        test_loader = self.data_module.test_dataloader()
        
        # Data containers
        trajectories = []
        determinants = []
        sequences = []
        reconstructions = []
        
        for batch_idx, batch in enumerate(test_loader):
            if len(trajectories) >= n_sequences:
                break
            
            try:
                if len(batch.shape) == 4:
                    batch = batch.unsqueeze(0)
                original_sequence = batch.clone()
                batch = batch.to(self.device)
                
                with torch.no_grad():
                    output = self.model(batch)
                
                if 'latent_samples' not in output:
                    continue
                
                # Extract latent trajectory
                latent_seq = output['latent_samples'].squeeze(0).cpu().numpy()
                trajectories.append(latent_seq)
                
                # Store original sequence
                sequences.append(original_sequence.squeeze(0))
                
                # Get reconstructions
                if 'reconstruction' in output:
                    reconstruction = output['reconstruction'].squeeze(0).cpu()
                else:
                    with torch.no_grad():
                        decoder_out = self.model.decoder(output['latent_samples'].view(-1, output['latent_samples'].shape[-1]))
                        if isinstance(decoder_out, dict) and 'reconstruction' in decoder_out:
                            reconstruction = decoder_out['reconstruction']
                        else:
                            reconstruction = decoder_out
                        reconstruction = reconstruction.view(output['latent_samples'].shape[1], *reconstruction.shape[1:]).cpu()
                
                reconstructions.append(reconstruction)
                
                # Compute determinants
                dets = []
                for t in range(latent_seq.shape[0]):
                    z_point = torch.tensor(latent_seq[t], device=self.device).unsqueeze(0)
                    det_value = self._compute_metric_determinant(z_point)
                    dets.append(det_value)
                determinants.append(np.array(dets))
                
                if len(trajectories) % 10 == 0:
                    logger.info(f"📊 Extracted {len(trajectories)}/{n_sequences} sequences")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error processing batch {batch_idx}: {e}")
                continue
        
        logger.info(f"✅ Extracted comprehensive data: {len(trajectories)} sequences")
        return trajectories, determinants, sequences, reconstructions
    
    def create_metric_landscape(self, trajectories, determinants) -> go.Figure:
        """Create the metric landscape visualization showing mountains and valleys."""
        logger.info("🏔️ Creating metric landscape visualization")
        
        # Get all points for landscape bounds
        all_points = np.vstack(trajectories)
        all_dets = np.concatenate(determinants)
        
        # Create a grid over the latent space
        x_min, x_max = all_points[:, 0].min() - 0.5, all_points[:, 0].max() + 0.5
        y_min, y_max = all_points[:, 1].min() - 0.5, all_points[:, 1].max() + 0.5
        
        # Create grid
        grid_resolution = 50
        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Compute metric determinant for each grid point
        Z = np.zeros_like(X)
        
        logger.info("Computing metric landscape...")
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                z_point = torch.tensor([X[i, j], Y[i, j]], device=self.device).unsqueeze(0)
                Z[i, j] = self._compute_metric_determinant(z_point)
            if i % 10 == 0:
                logger.info(f"Grid computation: {i}/{grid_resolution}")
        
        # Create landscape visualization
        fig = go.Figure()
        
        # Add surface plot (mountains and valleys)
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z,
                colorscale='earth',
                showscale=True,
                colorbar=dict(title="det(G⁻¹)", x=1.1),
                opacity=0.8,
                name="Metric Landscape"
            )
        )
        
        # Add trajectory paths on the landscape
        colors = px.colors.qualitative.Set3
        for i, (traj, dets) in enumerate(zip(trajectories[:10], determinants[:10])):
            fig.add_trace(
                go.Scatter3d(
                    x=traj[:, 0],
                    y=traj[:, 1], 
                    z=dets + 0.1,  # Slightly above surface
                    mode='lines+markers',
                    name=f'Trajectory {i}',
                    line=dict(color=colors[i % len(colors)], width=5),
                    marker=dict(size=4, color=colors[i % len(colors)]),
                    hovertemplate=f"Traj {i}<br>Z1: %{{x:.3f}}<br>Z2: %{{y:.3f}}<br>det(G⁻¹): %{{z:.3f}}<extra></extra>"
                )
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="🏔️ Metric Tensor Landscape - Mountains and Valleys of Latent Space",
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title="Latent Dimension 1",
                yaxis_title="Latent Dimension 2",
                zaxis_title="det(G⁻¹) - Metric Strength",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=700,
            showlegend=True
        )
        
        return fig
    
    def create_ultimate_combined_visualization(self, trajectories, determinants, sequences, reconstructions) -> None:
        """Create the ultimate combined visualization with everything."""
        logger.info("🎨 Creating ultimate combined visualization")
        
        # Apply PCA
        all_points = np.vstack(trajectories)
        pca = PCA(n_components=2)
        pca_all = pca.fit_transform(all_points)
        
        # Transform trajectories to PCA space
        trajectory_pca = []
        start_idx = 0
        for traj in trajectories:
            end_idx = start_idx + len(traj)
            trajectory_pca.append(pca_all[start_idx:end_idx])
            start_idx = end_idx
        
        # Create the metric landscape
        landscape_fig = self.create_metric_landscape(trajectories, determinants)
        
        # Create main interactive explorer
        main_fig = go.Figure()
        
        # Colors
        colors = px.colors.qualitative.Set3
        
        # Add trajectory traces for slider control
        for seq_idx in range(len(trajectories)):
            traj_pca = trajectory_pca[seq_idx]
            dets = determinants[seq_idx]
            
            main_fig.add_trace(
                go.Scatter(
                    x=traj_pca[:, 0],
                    y=traj_pca[:, 1],
                    mode='lines+markers',
                    name=f'Sequence {seq_idx}',
                    line=dict(color=colors[seq_idx % len(colors)], width=4),
                    marker=dict(
                        size=12,
                        color=dets,
                        colorscale='viridis',
                        showscale=True,
                        colorbar=dict(title="det(G⁻¹)", thickness=15, len=0.7)
                    ),
                    visible=(seq_idx == 0),
                    hovertemplate=(
                        f"<b>Sequence {seq_idx}</b><br>"
                        "PC1: %{x:.3f}<br>"
                        "PC2: %{y:.3f}<br>"
                        "det(G⁻¹): %{marker.color:.3f}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False
                )
            )
        
        # Create slider for main visualization
        steps = []
        for seq_idx in range(len(trajectories)):
            visible = [False] * len(trajectories)
            visible[seq_idx] = True
            
            step = dict(
                method="update",
                args=[
                    {"visible": visible},
                    {
                        "title": f"🎮 Sequence {seq_idx} - PCA Trajectory with Metric Determinants",
                        "annotations": [
                            dict(
                                text=f"<b>Sequence {seq_idx}</b><br>Timesteps: 0→9<br>Points colored by det(G⁻¹)<br><br>📊 PCA Variance:<br>PC1: {pca.explained_variance_ratio_[0]:.1%}<br>PC2: {pca.explained_variance_ratio_[1]:.1%}",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.02, y=0.98,
                                xanchor="left", yanchor="top",
                                bgcolor="rgba(255,255,255,0.9)",
                                bordercolor="rgba(0,0,0,0.3)",
                                borderwidth=1,
                                font=dict(size=11)
                            )
                        ]
                    }
                ],
                label=f"Seq {seq_idx}"
            )
            steps.append(step)
        
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Current Sequence: "},
            pad={"t": 80},
            steps=steps,
            x=0.1,
            xanchor="left",
            y=0,
            yanchor="top",
            len=0.8
        )]
        
        # Update main figure layout
        main_fig.update_layout(
            title=dict(
                text="🎮 Sequence 0 - PCA Trajectory with Metric Determinants",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis=dict(
                title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} explained variance)",
                showgrid=True,
                gridcolor="rgba(128,128,128,0.2)"
            ),
            yaxis=dict(
                title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} explained variance)",
                showgrid=True,
                gridcolor="rgba(128,128,128,0.2)"
            ),
            sliders=sliders,
            height=700,
            width=1000,
            plot_bgcolor="rgba(240,240,240,0.3)",
            annotations=[
                dict(
                    text=f"<b>Sequence 0</b><br>Timesteps: 0→9<br>Points colored by det(G⁻¹)<br><br>📊 PCA Variance:<br>PC1: {pca.explained_variance_ratio_[0]:.1%}<br>PC2: {pca.explained_variance_ratio_[1]:.1%}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.3)",
                    borderwidth=1,
                    font=dict(size=11)
                )
            ]
        )
        
        # Create dense points visualization
        all_dets = np.concatenate(determinants)
        dense_fig = go.Figure()
        
        dense_fig.add_trace(
            go.Scatter(
                x=pca_all[:, 0],
                y=pca_all[:, 1],
                mode='markers',
                marker=dict(
                    size=5,
                    color=all_dets,
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title="det(G⁻¹)", thickness=15),
                    line=dict(width=0.5, color='rgba(0,0,0,0.2)')
                ),
                name='All Points',
                hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>det(G⁻¹): %{marker.color:.3f}<extra></extra>",
                showlegend=False
            )
        )
        
        dense_fig.update_layout(
            title="🎯 Dense Latent Space - All Points Colored by det(G⁻¹)",
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
            height=600,
            plot_bgcolor="rgba(240,240,240,0.3)"
        )
        
        # Create reconstruction galleries
        self._create_reconstruction_galleries(sequences, reconstructions, determinants)
        
        # Save all visualizations
        main_path = self.output_dir / "sequence_explorer_with_slider.html"
        landscape_path = self.output_dir / "metric_landscape.html"
        dense_path = self.output_dir / "dense_latent_space.html"
        
        main_fig.write_html(str(main_path))
        landscape_fig.write_html(str(landscape_path))
        dense_fig.write_html(str(dense_path))
        
        logger.info(f"✅ Main explorer saved to {main_path}")
        logger.info(f"✅ Metric landscape saved to {landscape_path}")
        logger.info(f"✅ Dense space saved to {dense_path}")
        
        # Create comprehensive index
        self._create_comprehensive_index(main_path, landscape_path, dense_path, len(trajectories), pca)
    
    def _create_reconstruction_galleries(self, sequences, reconstructions, determinants) -> None:
        """Create reconstruction galleries."""
        logger.info("🖼️ Creating reconstruction galleries")
        
        reconstruction_dir = self.output_dir / "reconstructions"
        reconstruction_dir.mkdir(exist_ok=True)
        
        for seq_idx, (orig_seq, recon_seq, dets) in enumerate(zip(sequences[:20], reconstructions[:20], determinants[:20])):
            try:
                fig = make_subplots(
                    rows=2, cols=10,
                    subplot_titles=[f"T{t}" for t in range(10)] + [f"Recon T{t}" for t in range(10)],
                    vertical_spacing=0.05,
                    horizontal_spacing=0.02
                )
                
                # Add original and reconstruction images
                for t in range(10):
                    # Original
                    img_base64 = self.tensor_to_base64(orig_seq[t])
                    fig.add_layout_image(
                        dict(
                            source=img_base64,
                            xref=f"x{t+1}", yref=f"y{t+1}",
                            x=0, y=1, sizex=1, sizey=1,
                            sizing="stretch", layer="below"
                        )
                    )
                    
                    # Reconstruction
                    img_base64 = self.tensor_to_base64(recon_seq[t])
                    fig.add_layout_image(
                        dict(
                            source=img_base64,
                            xref=f"x{t+11}", yref=f"y{t+11}",
                            x=0, y=1, sizex=1, sizey=1,
                            sizing="stretch", layer="below"
                        )
                    )
                    
                    # Add invisible scatters for hover
                    fig.add_trace(
                        go.Scatter(
                            x=[0, 1], y=[0, 1], mode='markers',
                            marker=dict(opacity=0), showlegend=False,
                            hovertemplate=f"Original T{t}<br>det(G⁻¹): {dets[t]:.3f}<extra></extra>"
                        ), row=1, col=t+1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[0, 1], y=[0, 1], mode='markers',
                            marker=dict(opacity=0), showlegend=False,
                            hovertemplate=f"Reconstruction T{t}<br>det(G⁻¹): {dets[t]:.3f}<extra></extra>"
                        ), row=2, col=t+1
                    )
                
                fig.update_layout(
                    title=f"Sequence {seq_idx} - Original vs Reconstructions",
                    height=400, showlegend=False
                )
                
                fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
                fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
                
                recon_path = reconstruction_dir / f"sequence_{seq_idx}_reconstructions.html"
                fig.write_html(str(recon_path))
                
            except Exception as e:
                logger.warning(f"⚠️ Error creating reconstruction for sequence {seq_idx}: {e}")
        
        logger.info(f"✅ Created reconstruction galleries for {min(20, len(sequences))} sequences")
    
    def _create_comprehensive_index(self, main_path, landscape_path, dense_path, n_sequences, pca) -> None:
        """Create comprehensive index with everything."""
        
        index_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>🚀 Ultimate RLVAE Comprehensive Explorer</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            min-height: 100vh;
            box-shadow: 0 0 30px rgba(0,0,0,0.2);
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 3em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            animation: glow 2s ease-in-out infinite alternate;
        }}
        @keyframes glow {{
            from {{ text-shadow: 2px 2px 4px rgba(0,0,0,0.3), 0 0 20px rgba(255,255,255,0.2); }}
            to {{ text-shadow: 2px 2px 4px rgba(0,0,0,0.3), 0 0 30px rgba(255,255,255,0.4); }}
        }}
        .header p {{
            margin: 15px 0 0 0;
            font-size: 1.3em;
            opacity: 0.9;
        }}
        .main-content {{
            padding: 40px;
        }}
        .section {{
            margin: 40px 0;
            padding: 30px;
            border-radius: 15px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-left: 6px solid #667eea;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin-top: 0;
            color: #667eea;
            font-size: 1.8em;
            display: flex;
            align-items: center;
        }}
        .section h2:before {{
            content: "✨";
            margin-right: 10px;
            font-size: 1.2em;
        }}
        .btn {{
            display: inline-block;
            padding: 18px 35px;
            margin: 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 30px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .btn:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        }}
        .btn.landscape {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            box-shadow: 0 5px 20px rgba(17, 153, 142, 0.3);
        }}
        .btn.landscape:hover {{
            box-shadow: 0 8px 25px rgba(17, 153, 142, 0.5);
        }}
        .btn.primary {{
            font-size: 1.2em;
            padding: 25px 50px;
            background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
            box-shadow: 0 5px 20px rgba(255, 107, 107, 0.3);
        }}
        .btn.primary:hover {{
            box-shadow: 0 8px 25px rgba(255, 107, 107, 0.5);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 25px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }}
        .stat-label {{
            color: #666;
            margin-top: 8px;
            font-weight: 500;
        }}
        .reconstruction-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            gap: 15px;
            margin: 25px 0;
        }}
        .recon-btn {{
            padding: 12px;
            font-size: 0.95em;
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            color: #333;
        }}
        .features {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin: 30px 0;
        }}
        .features h3 {{
            margin-top: 0;
            font-size: 1.5em;
        }}
        .features ul {{
            margin: 20px 0;
            padding-left: 25px;
            line-height: 1.8;
        }}
        .features li {{
            margin: 12px 0;
        }}
        .landscape-highlight {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin: 30px 0;
            text-align: center;
        }}
        .landscape-highlight h3 {{
            margin-top: 0;
            font-size: 2em;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Ultimate RLVAE Explorer</h1>
            <p>Complete comprehensive analysis with metric landscape visualization</p>
        </div>
        
        <div class="main-content">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{n_sequences}</div>
                    <div class="stat-label">Sequences Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{n_sequences * 10}</div>
                    <div class="stat-label">Total Timesteps</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{pca.explained_variance_ratio_[0]:.1%}</div>
                    <div class="stat-label">PC1 Variance</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{pca.explained_variance_ratio_[1]:.1%}</div>
                    <div class="stat-label">PC2 Variance</div>
                </div>
            </div>
            
            <div class="landscape-highlight">
                <h3>🏔️ NEW: Metric Landscape Visualization</h3>
                <p>Explore the mountains and valleys of your latent space based on the metric tensor!</p>
                <a href="{landscape_path.name}" class="btn landscape">🚀 Explore Metric Landscape</a>
            </div>
            
            <div class="section">
                <h2>Interactive Sequence Explorer</h2>
                <p>Use the slider to select different sequences and see their PCA trajectories with metric determinants.</p>
                <a href="{main_path.name}" class="btn primary">🎮 Launch Sequence Explorer</a>
            </div>
            
            <div class="section">
                <h2>Dense Latent Space Analysis</h2>
                <p>View all {n_sequences * 10} points in latent space colored by metric determinant values.</p>
                <a href="{dense_path.name}" class="btn">🎯 View Dense Space</a>
            </div>
            
            <div class="section">
                <h2>Reconstruction Galleries</h2>
                <p>Compare original images vs reconstructions for each sequence timestep.</p>
                <div class="reconstruction-grid">
                    {self._generate_reconstruction_buttons()}
                </div>
            </div>
            
            <div class="features">
                <h3>🌟 Complete Feature Set</h3>
                <ul>
                    <li><strong>🏔️ Metric Landscape:</strong> 3D visualization of latent space mountains and valleys</li>
                    <li><strong>🎮 Interactive Slider:</strong> Select any sequence (0-{n_sequences-1}) for analysis</li>
                    <li><strong>📊 PCA Trajectories:</strong> See sequence movement through latent space</li>
                    <li><strong>🎨 Metric Coloring:</strong> All points colored by real det(G⁻¹) values</li>
                    <li><strong>🖼️ Reconstruction Galleries:</strong> Original vs reconstructed comparisons</li>
                    <li><strong>🎯 Dense Point Cloud:</strong> {n_sequences * 10} points with metric information</li>
                    <li><strong>💫 Interactive Elements:</strong> Hover details, 3D rotation, zoom capabilities</li>
                    <li><strong>📈 Real Model Data:</strong> All visualizations use actual trained model outputs</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        index_path = self.output_dir / "index.html"
        with open(index_path, 'w') as f:
            f.write(index_html)
        
        logger.info(f"✅ Comprehensive index created at {index_path}")
    
    def _generate_reconstruction_buttons(self) -> str:
        """Generate reconstruction buttons."""
        buttons = []
        recon_dir = self.output_dir / "reconstructions"
        if recon_dir.exists():
            for i in range(20):
                recon_file = recon_dir / f"sequence_{i}_reconstructions.html"
                if recon_file.exists():
                    buttons.append(f'<a href="reconstructions/sequence_{i}_reconstructions.html" class="btn recon-btn">Seq {i}</a>')
        return "\n".join(buttons)
    
    def run_ultimate_comprehensive_analysis(self) -> None:
        """Run the ultimate comprehensive analysis."""
        logger.info("🚀 Starting ULTIMATE comprehensive analysis")
        
        wandb.init(
            project="rlvae-ultimate-comprehensive",
            name=f"ultimate_comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "checkpoint": self.checkpoint_path,
                "device": self.device,
                "analysis_type": "ultimate_comprehensive_with_metric_landscape"
            }
        )
        
        try:
            # Load model
            self.load_model_complete()
            
            # Extract comprehensive data
            trajectories, determinants, sequences, reconstructions = self.extract_comprehensive_data(n_sequences=50)
            
            if len(trajectories) == 0:
                logger.error("❌ No sequences extracted")
                return
            
            # Create all visualizations
            self.create_ultimate_combined_visualization(trajectories, determinants, sequences, reconstructions)
            
            # Log to WandB
            for html_file in self.output_dir.glob("*.html"):
                try:
                    wandb.log({html_file.stem: wandb.Html(str(html_file))})
                    logger.info(f"📤 Logged {html_file.name} to WandB")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to log {html_file}: {e}")
            
            logger.info("🎉 ULTIMATE comprehensive analysis completed!")
            logger.info(f"📁 Results in: {self.output_dir}")
            logger.info(f"🌐 Open: {self.output_dir}/index.html")
            logger.info("🏔️ Don't miss the new metric landscape visualization!")
            
        except Exception as e:
            logger.error(f"❌ Ultimate analysis failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            wandb.finish()


def main():
    """Main execution."""
    checkpoint_path = "outputs/checkpoints/epoch=197-val_loss=6.402.ckpt"
    
    explorer = UltimateComprehensiveExplorer(checkpoint_path)
    explorer.run_ultimate_comprehensive_analysis()


if __name__ == "__main__":
    main() 