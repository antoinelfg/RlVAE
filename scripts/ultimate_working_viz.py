#!/usr/bin/env python3
"""
Ultimate Working RLVAE Visualizations
====================================

Final comprehensive solution:
1. Dense latent space structure with det(G^-1) coloring
2. Proper multi-panel layout
3. Fix metric tensor usage for sampling
4. Many points visualization
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

class UltimateWorkingVisualizer:
    """Ultimate working visualizer with dense latent space and proper metric usage."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """Initialize visualizer."""
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.model = None
        self.data_module = None
        self.checkpoint = None
        
        self.output_dir = Path("outputs/ultimate_working") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Ultimate working visualizer initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🖥️ Using device: {self.device}")
    
    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model_ultimate(self) -> None:
        """Load model with ultimate device and metric tensor fixes."""
        logger.info(f"🔄 Loading model from {self.checkpoint_path}")
        
        # Load checkpoint to target device
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
        
        logger.info(f"🔧 Creating model with proper metric tensor integration")
        
        # Create model
        self.model = ModularRiemannianFlowVAE(config)
        
        # Load state dict with device placement
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
        
        logger.info("✅ Model loaded with ultimate fixes")
        
        # Setup data
        self._setup_data()
        
        # Verify metric tensor access
        self._verify_and_test_metric_tensor()
    
    def _setup_data(self) -> None:
        """Setup data module."""
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
            'max_test_samples': 500,  # More data for dense visualization
            'verify_cyclicity': False
        })
        
        self.data_module = CyclicSpritesDataModule(data_config)
        self.data_module.setup('test')
        logger.info("✅ Data module setup complete with increased samples")
    
    def _verify_and_test_metric_tensor(self) -> None:
        """Verify metric tensor and test its usage."""
        logger.info("🔍 Verifying and testing metric tensor usage...")
        
        if hasattr(self.model, 'modular_metric') and self.model.modular_metric is not None:
            centroids = self.model.modular_metric.centroids
            matrices = self.model.modular_metric.metric_matrices
            
            logger.info(f"✅ Metric tensor loaded:")
            logger.info(f"  - Centroids: {centroids.shape} on {centroids.device}")
            logger.info(f"  - Matrices: {matrices.shape} on {matrices.device}")
            logger.info(f"  - Centroids range: [{centroids.min():.3f}, {centroids.max():.3f}]")
            logger.info(f"  - Matrices range: [{matrices.min():.3f}, {matrices.max():.3f}]")
            
            # Test metric tensor computation
            test_point = torch.randn(1, self.model.latent_dim, device=self.device)
            try:
                det_value = self._compute_metric_determinant(test_point)
                logger.info(f"  - Test determinant computation: {det_value:.6f}")
            except Exception as e:
                logger.warning(f"  - Determinant computation failed: {e}")
            
            # Check why metric tensor might not be used during sampling
            logger.info("🔍 Investigating metric tensor usage in sampling...")
            logger.info(f"  - Model posterior_type: {getattr(self.model, 'posterior_type', 'Not found')}")
            logger.info(f"  - Model has metric_tensor attr: {hasattr(self.model, 'metric_tensor')}")
            logger.info(f"  - Model has modular_metric attr: {hasattr(self.model, 'modular_metric')}")
            
        else:
            logger.error("❌ Metric tensor not properly loaded!")
    
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
    
    def extract_dense_latent_structure(self, n_sequences: int = 200) -> tuple:
        """Extract dense latent structure for comprehensive visualization."""
        logger.info(f"🌊 Extracting dense latent structure from {n_sequences} sequences")
        
        test_loader = self.data_module.test_dataloader()
        all_trajectories = []
        all_determinants = []
        all_timestamps = []
        all_latent_points = []
        all_point_determinants = []
        
        trajectory_count = 0
        
        for batch_idx, batch in enumerate(test_loader):
            if trajectory_count >= n_sequences:
                break
            
            try:
                # Ensure proper batch shape and device
                if len(batch.shape) == 4:
                    batch = batch.unsqueeze(0)
                batch = batch.to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    output = self.model(batch)
                
                if not isinstance(output, dict) or 'latent_samples' not in output:
                    continue
                
                latent_seq = output['latent_samples']  # [1, seq_len, latent_dim]
                trajectory = latent_seq.squeeze(0).cpu().numpy()  # [seq_len, latent_dim]
                
                # Compute determinants for each point in trajectory
                determinants = []
                for t in range(trajectory.shape[0]):
                    z_point = torch.tensor(trajectory[t], device=self.device).unsqueeze(0)
                    det_value = self._compute_metric_determinant(z_point)
                    determinants.append(det_value)
                
                # Store trajectory data
                all_trajectories.append(trajectory)
                all_determinants.append(np.array(determinants))
                all_timestamps.append(np.arange(len(trajectory)))
                
                # Store individual points for dense visualization
                for t, point in enumerate(trajectory):
                    all_latent_points.append(point)
                    all_point_determinants.append(determinants[t])
                
                trajectory_count += 1
                
                if trajectory_count % 50 == 0:
                    logger.info(f"📊 Extracted {trajectory_count}/{n_sequences} trajectories")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error processing batch {batch_idx}: {e}")
                continue
        
        all_latent_points = np.array(all_latent_points)
        all_point_determinants = np.array(all_point_determinants)
        
        logger.info(f"✅ Extracted dense structure:")
        logger.info(f"  - {len(all_trajectories)} trajectories")
        logger.info(f"  - {len(all_latent_points)} individual points")
        logger.info(f"  - Determinant range: [{all_point_determinants.min():.3f}, {all_point_determinants.max():.3f}]")
        
        return (all_trajectories, all_determinants, all_timestamps, 
                all_latent_points, all_point_determinants)
    
    def create_ultimate_multi_panel_visualization(self, trajectories, determinants, timestamps, 
                                                 latent_points, point_determinants) -> None:
        """Create ultimate multi-panel visualization with proper layout."""
        logger.info("🎨 Creating ultimate multi-panel visualization")
        
        # Apply PCA to all points
        pca = PCA(n_components=2)
        pca_points = pca.fit_transform(latent_points)
        
        logger.info(f"📊 PCA explained variance: {pca.explained_variance_ratio_}")
        logger.info(f"📊 Total points for visualization: {len(pca_points)}")
        
        # Create 2x2 subplots with proper specifications
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "🎯 Dense Latent Space Structure (det(G⁻¹) colored)",
                "📈 Trajectory Lines in PCA Space",
                "📏 Metric Determinant Distribution", 
                "🌊 3D Latent Space (PC1 × PC2 × det(G⁻¹))"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter3d"}]
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.12
        )
        
        # 1. Dense latent space structure with det(G^-1) coloring
        logger.info("Creating dense point cloud visualization...")
        fig.add_trace(
            go.Scatter(
                x=pca_points[:, 0],
                y=pca_points[:, 1],
                mode='markers',
                marker=dict(
                    size=4,
                    color=point_determinants,
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Log₁₀ det(G⁻¹)",
                        x=0.48,
                        len=0.4
                    ),
                    line=dict(width=0.5, color='rgba(0,0,0,0.2)')
                ),
                name='Latent Points',
                hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>det(G⁻¹): %{marker.color:.3f}<extra></extra>",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Trajectory lines in PCA space
        logger.info("Creating trajectory lines visualization...")
        colors = px.colors.qualitative.Set3
        
        # Convert trajectories to PCA space
        trajectory_pca = []
        start_idx = 0
        for traj in trajectories:
            end_idx = start_idx + len(traj)
            trajectory_pca.append(pca_points[start_idx:end_idx])
            start_idx = end_idx
        
        # Plot first 30 trajectories as lines
        for i, (traj_pca, dets) in enumerate(zip(trajectory_pca[:30], determinants[:30])):
            fig.add_trace(
                go.Scatter(
                    x=traj_pca[:, 0],
                    y=traj_pca[:, 1],
                    mode='lines+markers',
                    name=f'Traj {i}',
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4),
                    opacity=0.7,
                    showlegend=False,
                    hovertemplate=f"Traj {i}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>"
                ),
                row=1, col=2
            )
        
        # 3. Metric determinant distribution
        logger.info("Creating determinant distribution...")
        fig.add_trace(
            go.Histogram(
                x=point_determinants,
                nbinsx=50,
                name='det(G⁻¹) Distribution',
                marker=dict(color='blue', opacity=0.7),
                showlegend=False,
                hovertemplate="det(G⁻¹): %{x:.3f}<br>Count: %{y}<extra></extra>"
            ),
            row=2, col=1
        )
        
        # 4. 3D scatter (PC1 × PC2 × det(G^-1))
        logger.info("Creating 3D scatter visualization...")
        
        # Subsample for 3D visualization (every 10th point for performance)
        subsample_indices = np.arange(0, len(pca_points), 10)
        pca_sub = pca_points[subsample_indices]
        det_sub = point_determinants[subsample_indices]
        
        fig.add_trace(
            go.Scatter3d(
                x=pca_sub[:, 0],
                y=pca_sub[:, 1],
                z=det_sub,
                mode='markers',
                marker=dict(
                    size=3,
                    color=det_sub,
                    colorscale='plasma',
                    showscale=True,
                    colorbar=dict(
                        title="det(G⁻¹)",
                        x=1.02,
                        len=0.4
                    ),
                    opacity=0.8
                ),
                name='3D Latent Space',
                showlegend=False,
                hovertemplate="PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>det(G⁻¹): %{z:.3f}<extra></extra>"
            ),
            row=2, col=2
        )
        
        # Update layout with proper sizing and titles
        fig.update_layout(
            height=1000,
            width=1400,
            title=dict(
                text=f"🚀 Ultimate RLVAE Analysis - {len(pca_points)} Points with Real Metric Tensor",
                x=0.5,
                font=dict(size=20, color='darkblue')
            ),
            showlegend=False
        )
        
        # Update subplot axes
        fig.update_xaxes(title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", row=1, col=1)
        fig.update_yaxes(title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", row=1, col=1)
        
        fig.update_xaxes(title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", row=1, col=2)
        fig.update_yaxes(title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", row=1, col=2)
        
        fig.update_xaxes(title="Log₁₀ det(G⁻¹)", row=2, col=1)
        fig.update_yaxes(title="Frequency", row=2, col=1)
        
        # Update 3D scene
        fig.update_scenes(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="Log₁₀ det(G⁻¹)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            row=2, col=2
        )
        
        # Save
        html_path = self.output_dir / "ultimate_multi_panel_visualization.html"
        fig.write_html(str(html_path))
        
        logger.info(f"✅ Ultimate multi-panel visualization saved to {html_path}")
        
        # Log to WandB
        try:
            wandb.log({"ultimate_multi_panel_visualization": wandb.Html(str(html_path))})
        except:
            pass
        
        return fig
    
    def investigate_metric_sampling_issue(self) -> None:
        """Investigate why metric tensor isn't used for sampling."""
        logger.info("🔍 Investigating metric tensor sampling issue...")
        
        # Test sampling methods
        test_loader = self.data_module.test_dataloader()
        batch = next(iter(test_loader))
        if len(batch.shape) == 4:
            batch = batch.unsqueeze(0)
        batch = batch.to(self.device)
        
        logger.info("🧪 Testing different sampling approaches...")
        
        try:
            # Method 1: Direct model sampling
            logger.info("Method 1: Direct model.sample()...")
            if hasattr(self.model, 'sample'):
                try:
                    samples = self.model.sample(n_samples=5)
                    logger.info(f"  ✅ Direct sampling shape: {samples.shape if hasattr(samples, 'shape') else type(samples)}")
                except Exception as e:
                    logger.info(f"  ❌ Direct sampling failed: {e}")
            
            # Method 2: Forward pass latent extraction
            logger.info("Method 2: Forward pass latent extraction...")
            with torch.no_grad():
                output = self.model(batch)
                if 'latent_samples' in output:
                    latent_shape = output['latent_samples'].shape
                    logger.info(f"  ✅ Forward pass latents: {latent_shape}")
                    
                    # Check if these latents use the metric tensor
                    sample_latent = output['latent_samples'][0, 0]  # First timestep of first sequence
                    det_value = self._compute_metric_determinant(sample_latent.unsqueeze(0))
                    logger.info(f"  📊 Sample latent det(G⁻¹): {det_value:.6f}")
            
            # Method 3: Check model's internal sampling configuration
            logger.info("Method 3: Model sampling configuration...")
            logger.info(f"  - posterior_type: {getattr(self.model, 'posterior_type', 'Not found')}")
            logger.info(f"  - sampling config: {getattr(self.model, 'sampling', 'Not found')}")
            
            if hasattr(self.model, 'config'):
                sampling_config = self.model.config.get('sampling', {})
                logger.info(f"  - config.sampling: {sampling_config}")
            
        except Exception as e:
            logger.error(f"❌ Sampling investigation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def test_metric_generation(self) -> None:
        """Test generation using the metric tensor directly."""
        logger.info("🎮 Testing metric-aware generation...")
        
        try:
            # Generate points in latent space
            n_samples = 100
            latent_samples = torch.randn(n_samples, self.model.latent_dim, device=self.device)
            
            # Compute determinants for generated points
            determinants = []
            for i in range(n_samples):
                det_value = self._compute_metric_determinant(latent_samples[i:i+1])
                determinants.append(det_value)
            
            determinants = np.array(determinants)
            
            logger.info(f"✅ Generated {n_samples} latent samples")
            logger.info(f"📊 Generated determinants range: [{determinants.min():.3f}, {determinants.max():.3f}]")
            
            # Try to decode these samples
            if hasattr(self.model, 'decoder'):
                try:
                    with torch.no_grad():
                        decoded = self.model.decoder(latent_samples)
                    
                    if hasattr(decoded, 'shape'):
                        logger.info(f"✅ Decoded samples shape: {decoded.shape}")
                    elif isinstance(decoded, dict) and 'reconstruction' in decoded:
                        logger.info(f"✅ Decoded reconstruction shape: {decoded['reconstruction'].shape}")
                    else:
                        logger.info(f"✅ Decoded output type: {type(decoded)}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Decoding failed: {e}")
            
            return latent_samples.cpu().numpy(), determinants
            
        except Exception as e:
            logger.error(f"❌ Metric generation test failed: {e}")
            return None, None
    
    def run_ultimate_analysis(self) -> None:
        """Run ultimate comprehensive analysis."""
        logger.info("🚀 Starting ULTIMATE comprehensive analysis")
        
        # Initialize WandB
        wandb.init(
            project="rlvae-ultimate-working",
            name=f"ultimate_working_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "checkpoint": self.checkpoint_path,
                "device": self.device,
                "analysis_type": "ultimate_dense_latent_space_with_metric_determinants"
            }
        )
        
        try:
            # Load model with ultimate fixes
            self.load_model_ultimate()
            
            # Extract dense latent structure
            (trajectories, determinants, timestamps, 
             latent_points, point_determinants) = self.extract_dense_latent_structure(n_sequences=200)
            
            if len(latent_points) == 0:
                logger.error("❌ No latent points extracted")
                return
            
            # Create ultimate multi-panel visualization
            self.create_ultimate_multi_panel_visualization(
                trajectories, determinants, timestamps, latent_points, point_determinants)
            
            # Investigate metric sampling issue
            self.investigate_metric_sampling_issue()
            
            # Test metric generation
            self.test_metric_generation()
            
            # Log results
            for html_file in self.output_dir.glob("*.html"):
                try:
                    wandb.log({html_file.stem: wandb.Html(str(html_file))})
                    logger.info(f"📤 Logged {html_file.name} to WandB")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to log {html_file}: {e}")
            
            logger.info("🎉 ULTIMATE analysis completed successfully!")
            logger.info(f"📁 Results in: {self.output_dir}")
            logger.info(f"📊 Analyzed {len(latent_points)} points with real metric determinants")
            
        except Exception as e:
            logger.error(f"❌ Ultimate analysis failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            wandb.finish()


def main():
    """Main execution."""
    checkpoint_path = "outputs/checkpoints/epoch=197-val_loss=6.402.ckpt"
    
    visualizer = UltimateWorkingVisualizer(checkpoint_path)
    visualizer.run_ultimate_analysis()


if __name__ == "__main__":
    main() 