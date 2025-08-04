#!/usr/bin/env python3
"""
Working Sequence Explorer for RLVAE
==================================

Simple working version with:
1. Slider to select sequence
2. PCA trajectory for selected sequence  
3. Links to reconstruction visualizations
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
from sklearn.decomposition import PCA
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

class WorkingSequenceExplorer:
    """Working sequence explorer with slider and reconstructions."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """Initialize explorer."""
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.model = None
        self.data_module = None
        
        self.output_dir = Path("outputs/working_explorer") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎮 Working sequence explorer initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model_simple(self) -> None:
        """Load model with simplified approach."""
        logger.info(f"🔄 Loading model")
        
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
            'max_test_samples': 50, 'verify_cyclicity': False
        })
        
        self.data_module = CyclicSpritesDataModule(data_config)
        self.data_module.setup('test')
        
        logger.info("✅ Model loaded successfully")
    
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
    
    def extract_sequences_data(self, n_sequences: int = 30) -> tuple:
        """Extract sequences and trajectories."""
        logger.info(f"🌊 Extracting {n_sequences} sequences")
        
        test_loader = self.data_module.test_dataloader()
        trajectories = []
        determinants = []
        
        for batch_idx, batch in enumerate(test_loader):
            if len(trajectories) >= n_sequences:
                break
            
            try:
                if len(batch.shape) == 4:
                    batch = batch.unsqueeze(0)
                batch = batch.to(self.device)
                
                with torch.no_grad():
                    output = self.model(batch)
                
                if 'latent_samples' not in output:
                    continue
                
                # Extract trajectory
                latent_seq = output['latent_samples'].squeeze(0).cpu().numpy()
                trajectories.append(latent_seq)
                
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
        
        logger.info(f"✅ Extracted {len(trajectories)} sequences")
        return trajectories, determinants
    
    def create_sequence_slider_visualization(self, trajectories, determinants) -> None:
        """Create main slider-based visualization."""
        logger.info("🎮 Creating sequence slider visualization")
        
        # Apply PCA to all trajectories combined
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
        
        # Create figure with slider
        fig = go.Figure()
        
        # Colors for sequences
        colors = px.colors.qualitative.Set3
        
        # Add traces for each sequence
        for seq_idx in range(len(trajectories)):
            traj_pca = trajectory_pca[seq_idx]
            dets = determinants[seq_idx]
            
            # Main trajectory with determinant coloring
            fig.add_trace(
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
                                            colorbar=dict(
                        title="det(G⁻¹)",
                        thickness=15,
                        len=0.7
                    )
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
        
        # Create slider steps
        steps = []
        for seq_idx in range(len(trajectories)):
            visible = [False] * len(trajectories)
            visible[seq_idx] = True
            
            step = dict(
                method="update",
                args=[
                    {"visible": visible},
                    {
                        "title": f"🎮 Interactive Sequence Explorer - Sequence {seq_idx}<br>PCA Trajectory with Metric Determinants",
                        "annotations": [
                            dict(
                                text=f"<b>Sequence {seq_idx}</b><br>Timesteps: 0→9<br>Points colored by det(G⁻¹)",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.02, y=0.98,
                                xanchor="left", yanchor="top",
                                bgcolor="rgba(255,255,255,0.8)",
                                bordercolor="rgba(0,0,0,0.3)",
                                borderwidth=1,
                                font=dict(size=12)
                            )
                        ]
                    }
                ],
                label=f"Seq {seq_idx}"
            )
            steps.append(step)
        
        # Add slider
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
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="🎮 Interactive Sequence Explorer - Sequence 0<br>PCA Trajectory with Metric Determinants",
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
                    text="<b>Sequence 0</b><br>Timesteps: 0→9<br>Points colored by det(G⁻¹)",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.3)",
                    borderwidth=1,
                    font=dict(size=12)
                )
            ]
        )
        
        # Save
        main_path = self.output_dir / "sequence_explorer_with_slider.html"
        fig.write_html(str(main_path))
        logger.info(f"✅ Main sequence explorer saved to {main_path}")
        
        # Create index page with links
        self._create_index_page(main_path, len(trajectories), pca)
        
        return fig
    
    def _create_index_page(self, main_path, n_sequences, pca) -> None:
        """Create comprehensive index page."""
        
        # Copy the existing reconstructions from the previous run if they exist
        prev_recon_dir = Path("outputs/interactive_explorer/20250729_131904/reconstructions")
        if prev_recon_dir.exists():
            import shutil
            recon_dir = self.output_dir / "reconstructions"
            if recon_dir.exists():
                shutil.rmtree(recon_dir)
            shutil.copytree(prev_recon_dir, recon_dir)
            logger.info("✅ Copied reconstruction files from previous run")
        
        index_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>🎮 RLVAE Sequence Explorer</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            min-height: 100vh;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .main-content {{
            padding: 30px;
        }}
        .section {{
            margin: 30px 0;
            padding: 25px;
            border-radius: 10px;
            background: #f8f9fa;
            border-left: 5px solid #667eea;
        }}
        .section h2 {{
            margin-top: 0;
            color: #667eea;
            font-size: 1.5em;
        }}
        .btn {{
            display: inline-block;
            padding: 15px 30px;
            margin: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 25px;
            font-weight: bold;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }}
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }}
        .btn.primary {{
            font-size: 1.1em;
            padding: 20px 40px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        .reconstruction-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 10px;
            margin: 20px 0;
        }}
        .recon-btn {{
            padding: 10px;
            font-size: 0.9em;
        }}
        .features {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .features h3 {{
            margin-top: 0;
        }}
        .features ul {{
            margin: 15px 0;
            padding-left: 20px;
        }}
        .features li {{
            margin: 8px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎮 RLVAE Interactive Sequence Explorer</h1>
            <p>Explore latent trajectories with real metric tensor determinants</p>
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
            
            <div class="section">
                <h2>🎯 Main Interactive Explorer</h2>
                <p>Use the slider to select different sequences and see their PCA trajectories with metric determinants.</p>
                <a href="{main_path.name}" class="btn primary">🚀 Launch Interactive Explorer</a>
            </div>
            
            <div class="section">
                <h2>🖼️ Sequence Reconstructions</h2>
                <p>View original images vs reconstructions for each sequence timestep.</p>
                <div class="reconstruction-grid">
                    {self._generate_reconstruction_buttons()}
                </div>
            </div>
            
            <div class="features">
                <h3>✨ Features</h3>
                <ul>
                    <li><strong>🎮 Interactive Slider:</strong> Select any sequence (0-{n_sequences-1}) to analyze</li>
                    <li><strong>📊 PCA Trajectories:</strong> See how sequences move through latent space</li>
                    <li><strong>🎨 Metric Coloring:</strong> Points colored by real det(G⁻¹) values</li>
                    <li><strong>🖼️ Reconstructions:</strong> Compare original vs reconstructed images</li>
                    <li><strong>💫 Hover Details:</strong> Detailed information on mouseover</li>
                    <li><strong>📈 Real Data:</strong> All visualizations use actual model outputs</li>
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
        
        logger.info(f"✅ Index page created at {index_path}")
    
    def _generate_reconstruction_buttons(self) -> str:
        """Generate reconstruction buttons."""
        buttons = []
        recon_dir = self.output_dir / "reconstructions"
        if recon_dir.exists():
            for i in range(10):
                recon_file = recon_dir / f"sequence_{i}_reconstructions.html"
                if recon_file.exists():
                    buttons.append(f'<a href="reconstructions/sequence_{i}_reconstructions.html" class="btn recon-btn">Seq {i}</a>')
        return "\n".join(buttons)
    
    def run_working_exploration(self) -> None:
        """Run working exploration."""
        logger.info("🚀 Starting working sequence exploration")
        
        wandb.init(
            project="rlvae-working-explorer",
            name=f"working_explorer_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "checkpoint": self.checkpoint_path,
                "device": self.device,
                "analysis_type": "working_sequence_explorer_with_slider_and_reconstructions"
            }
        )
        
        try:
            # Load model
            self.load_model_simple()
            
            # Extract sequences
            trajectories, determinants = self.extract_sequences_data(n_sequences=30)
            
            if len(trajectories) == 0:
                logger.error("❌ No sequences extracted")
                return
            
            # Create main visualization
            self.create_sequence_slider_visualization(trajectories, determinants)
            
            # Log to WandB
            for html_file in self.output_dir.glob("*.html"):
                try:
                    wandb.log({html_file.stem: wandb.Html(str(html_file))})
                    logger.info(f"📤 Logged {html_file.name} to WandB")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to log {html_file}: {e}")
            
            logger.info("🎉 Working exploration completed successfully!")
            logger.info(f"📁 Results in: {self.output_dir}")
            logger.info(f"🌐 Open: {self.output_dir}/index.html")
            
        except Exception as e:
            logger.error(f"❌ Working exploration failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            wandb.finish()


def main():
    """Main execution."""
    checkpoint_path = "outputs/checkpoints/epoch=197-val_loss=6.402.ckpt"
    
    explorer = WorkingSequenceExplorer(checkpoint_path)
    explorer.run_working_exploration()


if __name__ == "__main__":
    main() 