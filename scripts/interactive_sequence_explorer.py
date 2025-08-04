#!/usr/bin/env python3
"""
Interactive Sequence Explorer for RLVAE
======================================

Interactive visualization with:
1. Slider to select sequence
2. Reconstruction images for each timestep
3. PCA trajectory for selected sequence
4. Metric determinant evolution
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

class InteractiveSequenceExplorer:
    """Interactive sequence explorer with slider-based sequence selection."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """Initialize explorer."""
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.model = None
        self.data_module = None
        self.checkpoint = None
        
        self.output_dir = Path("outputs/interactive_explorer") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎮 Interactive sequence explorer initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model_for_interactive(self) -> None:
        """Load model for interactive exploration."""
        logger.info(f"🔄 Loading model for interactive exploration")
        
        # Load checkpoint
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
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
        
        # Load state dict with device handling
        state_dict = self.checkpoint['state_dict']
        clean_state_dict = {}
        
        for k, v in state_dict.items():
            clean_key = k.replace('model.', '') if k.startswith('model.') else k
            clean_state_dict[clean_key] = v.to(self.device)
        
        # Resize metric tensor
        for name, param in clean_state_dict.items():
            if 'modular_metric.centroids' in name:
                self.model.modular_metric.centroids = torch.nn.Parameter(torch.zeros_like(param))
            elif 'modular_metric.metric_matrices' in name:
                self.model.modular_metric.metric_matrices = torch.nn.Parameter(torch.zeros_like(param))
        
        # Load state dict
        self.model.load_state_dict(clean_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # Setup data
        self._setup_data()
        
        logger.info("✅ Model loaded for interactive exploration")
    
    def _setup_data(self) -> None:
        """Setup data module for exploration."""
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
        logger.info("✅ Data module setup for interactive exploration")
    
    def _compute_metric_determinant(self, z_point: torch.Tensor) -> float:
        """Compute metric determinant at a latent point."""
        if hasattr(self.model, 'modular_metric') and self.model.modular_metric is not None:
            centroids = self.model.modular_metric.centroids
            matrices = self.model.modular_metric.metric_matrices
            
            # Find closest centroid
            distances = torch.norm(z_point - centroids, dim=1)
            closest_idx = torch.argmin(distances)
            
            # Get metric matrix and compute determinant
            metric_matrix = matrices[closest_idx]
            det_g = torch.det(metric_matrix)
            log_det_g_inv = -torch.log10(det_g + 1e-8)
            
            return log_det_g_inv.cpu().item()
        else:
            return 0.0
    
    def tensor_to_base64(self, tensor_img: torch.Tensor) -> str:
        """Convert tensor image to base64 string for embedding in HTML."""
        # Convert from [C, H, W] to [H, W, C] and scale to 0-255
        if len(tensor_img.shape) == 3:
            img_np = tensor_img.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = tensor_img.cpu().numpy()
        
        img_np = (img_np * 255).astype(np.uint8)
        
        # Convert to PIL Image
        if img_np.shape[2] == 3:
            img_pil = Image.fromarray(img_np, 'RGB')
        else:
            img_pil = Image.fromarray(img_np[:, :, 0], 'L')
        
        # Convert to base64
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_base64}"
    
    def extract_sequences_with_reconstructions(self, n_sequences: int = 50) -> tuple:
        """Extract sequences with original images, reconstructions, and latent trajectories."""
        logger.info(f"🌊 Extracting {n_sequences} sequences with reconstructions")
        
        test_loader = self.data_module.test_dataloader()
        
        all_sequences = []
        all_reconstructions = []
        all_trajectories = []
        all_determinants = []
        
        sequence_count = 0
        
        for batch_idx, batch in enumerate(test_loader):
            if sequence_count >= n_sequences:
                break
            
            try:
                # Ensure proper batch shape and device
                if len(batch.shape) == 4:
                    batch = batch.unsqueeze(0)
                original_sequence = batch.clone()
                batch = batch.to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    output = self.model(batch)
                
                if not isinstance(output, dict) or 'latent_samples' not in output:
                    continue
                
                # Extract data
                latent_seq = output['latent_samples']  # [1, seq_len, latent_dim]
                trajectory = latent_seq.squeeze(0).cpu().numpy()  # [seq_len, latent_dim]
                
                # Get reconstructions
                if 'reconstruction' in output:
                    reconstruction = output['reconstruction'].squeeze(0).cpu()  # [seq_len, C, H, W]
                else:
                    # Try to get reconstruction through decoder
                    with torch.no_grad():
                        decoder_out = self.model.decoder(latent_seq.view(-1, latent_seq.shape[-1]))
                        if isinstance(decoder_out, dict) and 'reconstruction' in decoder_out:
                            reconstruction = decoder_out['reconstruction']
                        else:
                            reconstruction = decoder_out
                        reconstruction = reconstruction.view(latent_seq.shape[1], *reconstruction.shape[1:]).cpu()
                
                # Compute determinants
                determinants = []
                for t in range(trajectory.shape[0]):
                    z_point = torch.tensor(trajectory[t], device=self.device).unsqueeze(0)
                    det_value = self._compute_metric_determinant(z_point)
                    determinants.append(det_value)
                
                # Store data
                all_sequences.append(original_sequence.squeeze(0))  # [seq_len, C, H, W]
                all_reconstructions.append(reconstruction)  # [seq_len, C, H, W]
                all_trajectories.append(trajectory)  # [seq_len, latent_dim]
                all_determinants.append(np.array(determinants))  # [seq_len]
                
                sequence_count += 1
                
                if sequence_count % 10 == 0:
                    logger.info(f"📊 Extracted {sequence_count}/{n_sequences} sequences with reconstructions")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error processing batch {batch_idx}: {e}")
                continue
        
        logger.info(f"✅ Extracted {len(all_sequences)} sequences with reconstructions")
        return all_sequences, all_reconstructions, all_trajectories, all_determinants
    
    def create_interactive_sequence_explorer(self, sequences, reconstructions, trajectories, determinants) -> None:
        """Create interactive sequence explorer with slider and reconstructions."""
        logger.info("🎮 Creating interactive sequence explorer")
        
        # Combine all trajectories for PCA
        all_points = np.vstack(trajectories)
        pca = PCA(n_components=2)
        pca_all = pca.fit_transform(all_points)
        
        logger.info(f"📊 PCA explained variance: {pca.explained_variance_ratio_}")
        
        # Transform individual trajectories to PCA space
        trajectory_pca = []
        start_idx = 0
        for traj in trajectories:
            end_idx = start_idx + len(traj)
            trajectory_pca.append(pca_all[start_idx:end_idx])
            start_idx = end_idx
        
        # Create the main interactive figure
        fig = go.Figure()
        
        # Colors for different sequences
        colors = px.colors.qualitative.Set3
        
        # Add traces for each sequence (initially all invisible except first)
        for seq_idx in range(len(trajectories)):
            traj_pca = trajectory_pca[seq_idx]
            dets = determinants[seq_idx]
            
            # Trajectory line
            fig.add_trace(
                go.Scatter(
                    x=traj_pca[:, 0],
                    y=traj_pca[:, 1],
                    mode='lines+markers',
                    name=f'Trajectory {seq_idx}',
                    line=dict(color=colors[seq_idx % len(colors)], width=3),
                    marker=dict(
                        size=10,
                        color=dets,
                        colorscale='viridis',
                        showscale=(seq_idx == 0),
                        colorbar=dict(title="det(G⁻¹)", x=1.02) if seq_idx == 0 else None
                    ),
                    visible=(seq_idx == 0),
                    hovertemplate=f"Seq {seq_idx}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>det(G⁻¹): %{{marker.color:.3f}}<extra></extra>"
                )
            )
        
        # Create slider steps
        steps = []
        for seq_idx in range(len(trajectories)):
            # Create visibility array
            visible = [False] * len(trajectories)
            visible[seq_idx] = True
            
            step = dict(
                method="update",
                args=[
                    {"visible": visible},
                    {"title": f"Interactive Sequence Explorer - Sequence {seq_idx}<br>PC1 vs PC2 Trajectory"}
                ],
                label=f"Seq {seq_idx}"
            )
            steps.append(step)
        
        # Add slider
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Sequence: "},
            pad={"t": 50},
            steps=steps,
            x=0.1,
            xanchor="left",
            y=0,
            yanchor="top"
        )]
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="🎮 Interactive Sequence Explorer - Sequence 0<br>PC1 vs PC2 Trajectory",
                x=0.5,
                font=dict(size=18)
            ),
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
            sliders=sliders,
            height=600,
            showlegend=False
        )
        
        # Save the trajectory explorer
        trajectory_path = self.output_dir / "interactive_trajectory_explorer.html"
        fig.write_html(str(trajectory_path))
        logger.info(f"✅ Interactive trajectory explorer saved to {trajectory_path}")
        
        # Create reconstruction grid for each sequence
        self._create_reconstruction_grids(sequences, reconstructions, determinants)
        
        # Create combined HTML with both trajectory and reconstructions
        self._create_combined_explorer(trajectory_path, sequences, reconstructions, trajectories, determinants, trajectory_pca, pca)
        
        return fig
    
    def _create_reconstruction_grids(self, sequences, reconstructions, determinants) -> None:
        """Create reconstruction grids for each sequence."""
        logger.info("🖼️ Creating reconstruction grids for each sequence")
        
        reconstruction_dir = self.output_dir / "reconstructions"
        reconstruction_dir.mkdir(exist_ok=True)
        
        for seq_idx, (orig_seq, recon_seq, dets) in enumerate(zip(sequences[:10], reconstructions[:10], determinants[:10])):
            # Create subplot for this sequence
            fig = make_subplots(
                rows=2, cols=10,
                subplot_titles=[f"T{t}" for t in range(10)] + [f"Recon T{t}" for t in range(10)],
                vertical_spacing=0.05,
                horizontal_spacing=0.02
            )
            
            # Add original images (top row)
            for t in range(10):
                img_base64 = self.tensor_to_base64(orig_seq[t])
                fig.add_layout_image(
                    dict(
                        source=img_base64,
                        xref=f"x{t+1}",
                        yref=f"y{t+1}",
                        x=0, y=1,
                        sizex=1, sizey=1,
                        sizing="stretch",
                        layer="below"
                    )
                )
                
                # Add invisible scatter to maintain subplot structure
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='markers',
                        marker=dict(opacity=0),
                        showlegend=False,
                        hovertemplate=f"Original T{t}<br>det(G⁻¹): {dets[t]:.3f}<extra></extra>"
                    ),
                    row=1, col=t+1
                )
            
            # Add reconstruction images (bottom row)
            for t in range(10):
                img_base64 = self.tensor_to_base64(recon_seq[t])
                fig.add_layout_image(
                    dict(
                        source=img_base64,
                        xref=f"x{t+11}",
                        yref=f"y{t+11}",
                        x=0, y=1,
                        sizex=1, sizey=1,
                        sizing="stretch",
                        layer="below"
                    )
                )
                
                # Add invisible scatter
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='markers',
                        marker=dict(opacity=0),
                        showlegend=False,
                        hovertemplate=f"Reconstruction T{t}<br>det(G⁻¹): {dets[t]:.3f}<extra></extra>"
                    ),
                    row=2, col=t+1
                )
            
            # Update layout
            fig.update_layout(
                title=f"Sequence {seq_idx} - Original vs Reconstructions",
                height=400,
                showlegend=False
            )
            
            # Remove axes
            fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
            
            # Save
            recon_path = reconstruction_dir / f"sequence_{seq_idx}_reconstructions.html"
            fig.write_html(str(recon_path))
        
        logger.info(f"✅ Created reconstruction grids for {min(10, len(sequences))} sequences")
    
    def _create_combined_explorer(self, trajectory_path, sequences, reconstructions, trajectories, determinants, trajectory_pca, pca) -> None:
        """Create combined HTML with trajectory explorer and reconstruction viewer."""
        logger.info("🎨 Creating combined interactive explorer")
        
        # Create a comprehensive combined visualization
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "scatter", "colspan": 2}, None],
                [{"type": "scatter"}, {"type": "scatter3d"}]
            ],
            subplot_titles=[
                "🎮 Interactive PCA Trajectory (Select with slider below)",
                "📏 Determinant Evolution",
                "🌊 3D Trajectory View"
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Colors
        colors = px.colors.qualitative.Set3
        
        # Add trajectory traces for slider (top panel)
        for seq_idx in range(min(20, len(trajectories))):  # Limit to 20 for performance
            traj_pca = trajectory_pca[seq_idx]
            dets = determinants[seq_idx]
            
            # Main trajectory
            fig.add_trace(
                go.Scatter(
                    x=traj_pca[:, 0],
                    y=traj_pca[:, 1],
                    mode='lines+markers',
                    name=f'Trajectory {seq_idx}',
                    line=dict(color=colors[seq_idx % len(colors)], width=4),
                    marker=dict(
                        size=12,
                        color=dets,
                        colorscale='viridis',
                        showscale=(seq_idx == 0),
                        colorbar=dict(title="det(G⁻¹)", x=0.48, len=0.8) if seq_idx == 0 else None
                    ),
                    visible=(seq_idx == 0),
                    hovertemplate=f"Seq {seq_idx}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>det(G⁻¹): %{{marker.color:.3f}}<extra></extra>",
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Determinant evolution
            timesteps = np.arange(len(dets))
            fig.add_trace(
                go.Scatter(
                    x=timesteps,
                    y=dets,
                    mode='lines+markers',
                    name=f'Det {seq_idx}',
                    line=dict(color=colors[seq_idx % len(colors)], width=3),
                    marker=dict(size=8),
                    visible=(seq_idx == 0),
                    hovertemplate=f"Seq {seq_idx}<br>Time: %{{x}}<br>det(G⁻¹): %{{y:.3f}}<extra></extra>",
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # 3D trajectory
            fig.add_trace(
                go.Scatter3d(
                    x=traj_pca[:, 0],
                    y=traj_pca[:, 1],
                    z=timesteps,
                    mode='lines+markers',
                    name=f'3D Traj {seq_idx}',
                    line=dict(color=colors[seq_idx % len(colors)], width=4),
                    marker=dict(
                        size=6,
                        color=dets,
                        colorscale='plasma',
                        showscale=(seq_idx == 0),
                        colorbar=dict(title="det(G⁻¹)", x=1.02, len=0.4) if seq_idx == 0 else None
                    ),
                    visible=(seq_idx == 0),
                    hovertemplate=f"Seq {seq_idx}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>Time: %{{z}}<br>det(G⁻¹): %{{marker.color:.3f}}<extra></extra>",
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Create slider
        steps = []
        for seq_idx in range(min(20, len(trajectories))):
            # Create visibility array (3 traces per sequence)
            visible = [False] * (min(20, len(trajectories)) * 3)
            visible[seq_idx] = True  # Trajectory
            visible[seq_idx + min(20, len(trajectories))] = True  # Determinant
            visible[seq_idx + 2 * min(20, len(trajectories))] = True  # 3D
            
            step = dict(
                method="update",
                args=[
                    {"visible": visible},
                    {"title.text": f"🎮 Combined Interactive Explorer - Sequence {seq_idx}"}
                ],
                label=f"Seq {seq_idx}"
            )
            steps.append(step)
        
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Sequence: "},
            pad={"t": 50},
            steps=steps,
            x=0.1,
            xanchor="left",
            y=0,
            yanchor="top"
        )]
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="🎮 Combined Interactive Explorer - Sequence 0",
                x=0.5,
                font=dict(size=18)
            ),
            sliders=sliders,
            height=1000,
            showlegend=False
        )
        
        # Update axes
        fig.update_xaxes(title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", row=1, col=1)
        fig.update_yaxes(title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", row=1, col=1)
        fig.update_xaxes(title="Timestep", row=2, col=1)
        fig.update_yaxes(title="det(G⁻¹)", row=2, col=1)
        
        # Update 3D scene
        fig.update_scenes(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="Timestep",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            row=2, col=2
        )
        
        # Save combined explorer
        combined_path = self.output_dir / "combined_interactive_explorer.html"
        fig.write_html(str(combined_path))
        
        logger.info(f"✅ Combined interactive explorer saved to {combined_path}")
        
        # Create index HTML that links everything together
        self._create_index_html(combined_path)
    
    def _create_index_html(self, combined_path) -> None:
        """Create an index HTML that provides navigation to all visualizations."""
        index_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RLVAE Interactive Sequence Explorer</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }}
        .section {{
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background: #fafafa;
        }}
        .btn {{
            display: inline-block;
            padding: 12px 24px;
            margin: 10px;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
        }}
        .btn:hover {{
            background: #2980b9;
        }}
        .description {{
            color: #666;
            margin-bottom: 15px;
        }}
        .stats {{
            background: #e8f5e8;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎮 RLVAE Interactive Sequence Explorer</h1>
        
        <div class="stats">
            <strong>📊 Analysis Summary:</strong><br>
            • {len(trajectories)} sequences analyzed<br>
            • {len(trajectories) * 10} total timesteps<br>
            • PCA explained variance: {pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%}<br>
            • Metric tensor determinants computed for all points
        </div>
        
        <div class="section">
            <h2>🎯 Main Interactive Explorer</h2>
            <div class="description">
                Combined visualization with slider to select sequences. Shows PCA trajectory, 
                determinant evolution, and 3D view for each sequence.
            </div>
            <a href="{combined_path.name}" class="btn">🚀 Open Combined Explorer</a>
        </div>
        
        <div class="section">
            <h2>🖼️ Reconstruction Galleries</h2>
            <div class="description">
                View original images vs reconstructions for each sequence timestep.
            </div>
            {self._generate_reconstruction_links()}
        </div>
        
        <div class="section">
            <h2>📈 Features</h2>
            <ul>
                <li><strong>Slider Control:</strong> Select any sequence to analyze</li>
                <li><strong>PCA Trajectories:</strong> See how each sequence moves in latent space</li>
                <li><strong>Metric Determinants:</strong> Real det(G⁻¹) values colored on points</li>
                <li><strong>Reconstructions:</strong> Compare original vs reconstructed images</li>
                <li><strong>3D View:</strong> Interactive 3D trajectory visualization</li>
                <li><strong>Hover Details:</strong> Detailed information on mouseover</li>
            </ul>
        </div>
    </div>
</body>
</html>
        """
        
        index_path = self.output_dir / "index.html"
        with open(index_path, 'w') as f:
            f.write(index_html)
        
        logger.info(f"✅ Index page created at {index_path}")
    
    def _generate_reconstruction_links(self) -> str:
        """Generate HTML links for reconstruction galleries."""
        links = []
        reconstruction_dir = self.output_dir / "reconstructions"
        if reconstruction_dir.exists():
            for i in range(min(10, len(list(reconstruction_dir.glob("*.html"))))):
                links.append(f'<a href="reconstructions/sequence_{i}_reconstructions.html" class="btn">Seq {i} Reconstructions</a>')
        return "\n".join(links)
    
    def run_interactive_exploration(self) -> None:
        """Run complete interactive exploration."""
        logger.info("🚀 Starting interactive sequence exploration")
        
        # Initialize WandB
        wandb.init(
            project="rlvae-interactive-explorer",
            name=f"interactive_explorer_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "checkpoint": self.checkpoint_path,
                "device": self.device,
                "analysis_type": "interactive_sequence_explorer_with_reconstructions"
            }
        )
        
        try:
            # Load model
            self.load_model_for_interactive()
            
            # Extract sequences with reconstructions
            sequences, reconstructions, trajectories, determinants = self.extract_sequences_with_reconstructions(n_sequences=50)
            
            if len(sequences) == 0:
                logger.error("❌ No sequences extracted")
                return
            
            # Create interactive explorer
            self.create_interactive_sequence_explorer(sequences, reconstructions, trajectories, determinants)
            
            # Log results to WandB
            for html_file in self.output_dir.glob("*.html"):
                try:
                    wandb.log({html_file.stem: wandb.Html(str(html_file))})
                    logger.info(f"📤 Logged {html_file.name} to WandB")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to log {html_file}: {e}")
            
            logger.info("🎉 Interactive exploration completed successfully!")
            logger.info(f"📁 Results in: {self.output_dir}")
            logger.info(f"🌐 Open: {self.output_dir}/index.html")
            
        except Exception as e:
            logger.error(f"❌ Interactive exploration failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            wandb.finish()


def main():
    """Main execution."""
    checkpoint_path = "outputs/checkpoints/epoch=197-val_loss=6.402.ckpt"
    
    explorer = InteractiveSequenceExplorer(checkpoint_path)
    explorer.run_interactive_exploration()


if __name__ == "__main__":
    main() 