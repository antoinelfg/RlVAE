#!/usr/bin/env python3
"""
Simple Interactive Visualizations for Trained RLVAE
===================================================

Create working interactive visualizations focusing on trajectory analysis.
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

class SimpleInteractiveVizualizer:
    """Simple interactive visualizations of temporal trajectories."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """Initialize visualizer."""
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.model = None
        self.data_module = None
        
        self.output_dir = Path("outputs/simple_interactive") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎨 Simple interactive visualizer initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model_and_data(self) -> None:
        """Load model and data."""
        logger.info(f"🔄 Loading model from {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        model_hparams = checkpoint['hyper_parameters']['model']
        
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
        state_dict = checkpoint['state_dict']
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
            'batch_size': 1,
            'num_workers': 0,
            'pin_memory': False,
            'max_test_samples': 200,
            'verify_cyclicity': False
        })
        
        self.data_module = CyclicSpritesDataModule(data_config)
        self.data_module.setup('test')
        
        logger.info("✅ Model and data loaded successfully")
    
    def collect_trajectories(self, n_sequences: int = 50) -> tuple:
        """Collect trajectory data from sequences."""
        logger.info(f"🌊 Collecting trajectory data from {n_sequences} sequences")
        
        test_loader = self.data_module.test_dataloader()
        all_trajectories = []
        all_determinants = []
        
        trajectory_count = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if trajectory_count >= n_sequences:
                    break
                
                sequences = batch.to(self.device)
                if len(sequences.shape) == 4:
                    sequences = sequences.unsqueeze(0)
                
                # Forward pass
                output = self.model(sequences)
                
                if isinstance(output, dict) and 'latent_samples' in output:
                    latent_seq = output['latent_samples']  # [1, seq_len, latent_dim]
                    trajectory = latent_seq.squeeze(0).cpu().numpy()  # [seq_len, latent_dim]
                    
                    # Compute metric determinants for each timestep
                    determinants = []
                    seq_len = latent_seq.shape[1]
                    
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
                    
                    all_trajectories.append(trajectory)
                    all_determinants.append(np.array(determinants))
                    trajectory_count += 1
                    
                    if trajectory_count % 10 == 0:
                        logger.info(f"📊 Collected {trajectory_count}/{n_sequences} trajectories")
        
        logger.info(f"✅ Collected {len(all_trajectories)} trajectories")
        return all_trajectories, all_determinants
    
    def create_trajectory_scatter_plot(self, trajectories, determinants) -> None:
        """Create interactive scatter plot of trajectories."""
        logger.info("🎯 Creating trajectory scatter plot")
        
        # Combine all points for PCA
        all_points = np.vstack(trajectories)
        all_dets = np.concatenate(determinants)
        
        # Create trajectory and timestep labels
        trajectory_ids = []
        timesteps = []
        for i, traj in enumerate(trajectories):
            trajectory_ids.extend([i] * len(traj))
            timesteps.extend(list(range(len(traj))))
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(all_points)
        
        # Create scatter plot
        fig = go.Figure()
        
        # Color by determinant values
        fig.add_trace(go.Scatter(
            x=pca_coords[:, 0],
            y=pca_coords[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=all_dets,
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title="Log₁₀ det(G⁻¹)")
            ),
            text=[f"Traj {tid}, t={ts}" for tid, ts in zip(trajectory_ids, timesteps)],
            hovertemplate="<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>Det: %{marker.color:.3f}<extra></extra>",
            name="Latent Points"
        ))
        
        fig.update_layout(
            title="🎯 Interactive Trajectory Scatter Plot - PCA Space",
            xaxis_title=f"PC1 (explains {pca.explained_variance_ratio_[0]:.1%} variance)",
            yaxis_title=f"PC2 (explains {pca.explained_variance_ratio_[1]:.1%} variance)",
            hovermode='closest',
            width=800,
            height=600
        )
        
        # Save
        html_path = self.output_dir / "trajectory_scatter_plot.html"
        fig.write_html(str(html_path))
        
        logger.info(f"✅ Trajectory scatter plot saved to {html_path}")
        return fig
    
    def create_trajectory_lines_plot(self, trajectories, determinants) -> None:
        """Create interactive plot showing trajectory lines."""
        logger.info("🌊 Creating trajectory lines plot")
        
        # Apply PCA to all trajectories together
        all_points = np.vstack(trajectories)
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(all_points)
        
        # Reconstruct trajectories in PCA space
        start_idx = 0
        trajectory_pca = []
        for traj in trajectories:
            end_idx = start_idx + len(traj)
            trajectory_pca.append(pca_coords[start_idx:end_idx])
            start_idx = end_idx
        
        fig = go.Figure()
        
        # Add each trajectory as a line
        colors = px.colors.qualitative.Plotly
        
        for i, (traj_pca, dets) in enumerate(zip(trajectory_pca[:20], determinants[:20])):  # Limit for performance
            timesteps = list(range(len(traj_pca)))
            
            fig.add_trace(go.Scatter(
                x=traj_pca[:, 0],
                y=traj_pca[:, 1],
                mode='lines+markers',
                name=f'Trajectory {i}',
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(
                    size=6,
                    color=dets,
                    colorscale='plasma',
                    showscale=(i == 0),
                    colorbar=dict(title="Log₁₀ det(G⁻¹)") if i == 0 else None
                ),
                text=[f"t={t}" for t in timesteps],
                hovertemplate=f"<b>Trajectory {i}</b><br>%{{text}}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>Det: %{{marker.color:.3f}}<extra></extra>",
                showlegend=False
            ))
        
        fig.update_layout(
            title="🌊 Interactive Trajectory Lines - Temporal Evolution in PCA Space",
            xaxis_title=f"PC1 (explains {pca.explained_variance_ratio_[0]:.1%} variance)",
            yaxis_title=f"PC2 (explains {pca.explained_variance_ratio_[1]:.1%} variance)",
            hovermode='closest',
            width=900,
            height=700
        )
        
        # Save
        html_path = self.output_dir / "trajectory_lines_plot.html"
        fig.write_html(str(html_path))
        
        logger.info(f"✅ Trajectory lines plot saved to {html_path}")
        return fig
    
    def create_temporal_evolution_plot(self, trajectories, determinants) -> None:
        """Create plot showing temporal evolution of latent dimensions."""
        logger.info("⏱️ Creating temporal evolution plot")
        
        fig = go.Figure()
        
        # Plot individual trajectories
        colors = px.colors.qualitative.Set1
        
        for i, (traj, dets) in enumerate(zip(trajectories[:15], determinants[:15])):  # Limit for clarity
            timesteps = list(range(len(traj)))
            
            # Latent dimension 1
            fig.add_trace(go.Scatter(
                x=timesteps,
                y=traj[:, 0],
                mode='lines+markers',
                name=f'Traj {i} - Dim 1',
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6),
                opacity=0.7,
                showlegend=False,
                hovertemplate=f"<b>Trajectory {i} - Dimension 1</b><br>Time: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>"
            ))
            
            # Latent dimension 2
            fig.add_trace(go.Scatter(
                x=timesteps,
                y=traj[:, 1],
                mode='lines+markers',
                name=f'Traj {i} - Dim 2',
                line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                marker=dict(size=6, symbol='square'),
                opacity=0.7,
                showlegend=False,
                hovertemplate=f"<b>Trajectory {i} - Dimension 2</b><br>Time: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>"
            ))
        
        # Add mean trajectory
        mean_traj = np.mean(trajectories, axis=0)
        timesteps = list(range(len(mean_traj)))
        
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=mean_traj[:, 0],
            mode='lines+markers',
            name='Mean - Dim 1',
            line=dict(color='black', width=4),
            marker=dict(size=10),
            hovertemplate="<b>Mean Trajectory - Dimension 1</b><br>Time: %{x}<br>Value: %{y:.3f}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=mean_traj[:, 1],
            mode='lines+markers',
            name='Mean - Dim 2',
            line=dict(color='red', width=4, dash='dash'),
            marker=dict(size=10, symbol='square'),
            hovertemplate="<b>Mean Trajectory - Dimension 2</b><br>Time: %{x}<br>Value: %{y:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="⏱️ Temporal Evolution of Latent Dimensions",
            xaxis_title="Timestep",
            yaxis_title="Latent Value",
            hovermode='closest',
            width=900,
            height=600,
            showlegend=True
        )
        
        # Save
        html_path = self.output_dir / "temporal_evolution_plot.html"
        fig.write_html(str(html_path))
        
        logger.info(f"✅ Temporal evolution plot saved to {html_path}")
        return fig
    
    def create_determinant_evolution_plot(self, determinants) -> None:
        """Create plot showing metric determinant evolution."""
        logger.info("📏 Creating determinant evolution plot")
        
        fig = go.Figure()
        
        # Plot individual determinant trajectories
        colors = px.colors.qualitative.Pastel
        
        for i, dets in enumerate(determinants[:20]):  # Limit for performance
            timesteps = list(range(len(dets)))
            
            fig.add_trace(go.Scatter(
                x=timesteps,
                y=dets,
                mode='lines+markers',
                name=f'Trajectory {i}',
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6),
                opacity=0.6,
                showlegend=False,
                hovertemplate=f"<b>Trajectory {i}</b><br>Time: %{{x}}<br>Det: %{{y:.3f}}<extra></extra>"
            ))
        
        # Add mean and std
        mean_dets = np.mean(determinants, axis=0)
        std_dets = np.std(determinants, axis=0)
        timesteps = list(range(len(mean_dets)))
        
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=mean_dets,
            mode='lines+markers',
            name='Mean Determinant',
            line=dict(color='black', width=4),
            marker=dict(size=10),
            hovertemplate="<b>Mean Determinant</b><br>Time: %{x}<br>Det: %{y:.3f}<extra></extra>"
        ))
        
        # Add error bands
        fig.add_trace(go.Scatter(
            x=timesteps + timesteps[::-1],
            y=(mean_dets + std_dets).tolist() + (mean_dets - std_dets).tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±1 Std',
            showlegend=True
        ))
        
        fig.update_layout(
            title="📏 Metric Determinant Evolution Over Time",
            xaxis_title="Timestep",
            yaxis_title="Log₁₀ det(G⁻¹)",
            hovermode='closest',
            width=800,
            height=600,
            showlegend=True
        )
        
        # Save
        html_path = self.output_dir / "determinant_evolution_plot.html"
        fig.write_html(str(html_path))
        
        logger.info(f"✅ Determinant evolution plot saved to {html_path}")
        return fig
    
    def run_analysis(self) -> None:
        """Run complete simple interactive analysis."""
        logger.info("🚀 Starting simple interactive analysis")
        
        # Initialize WandB
        wandb.init(
            project="rlvae-simple-interactive",
            name=f"simple_interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "checkpoint": self.checkpoint_path,
                "device": self.device,
                "analysis_type": "simple_interactive"
            }
        )
        
        # Load model and data
        self.load_model_and_data()
        
        # Collect trajectory data
        trajectories, determinants = self.collect_trajectories(n_sequences=50)
        
        # Create visualizations
        self.create_trajectory_scatter_plot(trajectories, determinants)
        self.create_trajectory_lines_plot(trajectories, determinants)
        self.create_temporal_evolution_plot(trajectories, determinants)
        self.create_determinant_evolution_plot(determinants)
        
        # Log to WandB
        for html_file in self.output_dir.glob("*.html"):
            try:
                wandb.log({html_file.stem: wandb.Html(str(html_file))})
            except:
                pass
        
        logger.info("🎉 Simple interactive analysis completed!")
        logger.info(f"📁 All results saved in: {self.output_dir}")
        
        wandb.finish()


def main():
    """Main execution."""
    checkpoint_path = "outputs/checkpoints/epoch=197-val_loss=6.402.ckpt"
    
    visualizer = SimpleInteractiveVizualizer(checkpoint_path)
    visualizer.run_analysis()


if __name__ == "__main__":
    main() 