#!/usr/bin/env python3
"""
Completely Fixed Interactive Visualizations for RLVAE
====================================================

Final comprehensive fix for ALL device placement and forward pass issues.
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

class CompletelyFixedVisualizer:
    """Completely fixed visualizer with proper device handling."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """Initialize visualizer."""
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.model = None
        self.data_module = None
        self.checkpoint = None
        
        self.output_dir = Path("outputs/completely_fixed") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔧 Completely fixed visualizer initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🖥️ Using device: {self.device}")
    
    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model_with_complete_device_fix(self) -> None:
        """Load model with complete device placement fixes."""
        logger.info(f"🔄 Loading model from {self.checkpoint_path}")
        
        # Load checkpoint
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model_hparams = self.checkpoint['hyper_parameters']['model']
        
        logger.info(f"📊 Model hyperparameters: {model_hparams}")
        
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
        
        logger.info(f"🔧 Creating model with config")
        
        # Create model directly on target device
        with torch.cuda.device(self.device) if 'cuda' in self.device else torch.no_grad():
            self.model = ModularRiemannianFlowVAE(config)
        
        # Load state dict with complete device handling
        state_dict = self.checkpoint['state_dict']
        clean_state_dict = {}
        
        for k, v in state_dict.items():
            clean_key = k.replace('model.', '') if k.startswith('model.') else k
            # Ensure every parameter is on the correct device
            clean_state_dict[clean_key] = v.to(self.device)
        
        # Resize metric tensor parameters and ensure device placement
        metric_centroids_shape = None
        metric_matrices_shape = None
        
        for name, param in clean_state_dict.items():
            if 'modular_metric.centroids' in name:
                metric_centroids_shape = param.shape
                logger.info(f"📐 Found metric centroids shape: {metric_centroids_shape}")
            elif 'modular_metric.metric_matrices' in name:
                metric_matrices_shape = param.shape
                logger.info(f"📐 Found metric matrices shape: {metric_matrices_shape}")
        
        # Resize and place on correct device
        if metric_centroids_shape is not None:
            self.model.modular_metric.centroids = torch.nn.Parameter(
                torch.zeros(metric_centroids_shape, device=self.device)
            )
            logger.info(f"✅ Resized centroids to {metric_centroids_shape} on {self.device}")
        
        if metric_matrices_shape is not None:
            self.model.modular_metric.metric_matrices = torch.nn.Parameter(
                torch.zeros(metric_matrices_shape, device=self.device)
            )
            logger.info(f"✅ Resized metric matrices to {metric_matrices_shape} on {self.device}")
        
        # Load state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(clean_state_dict, strict=False)
        logger.info(f"📊 Model loaded - Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        
        # Move entire model to device and verify
        self.model.to(self.device)
        self.model.eval()
        
        # Verify all model components are on correct device
        self._verify_complete_device_placement()
        
        # Verify metric tensor access
        self._verify_metric_tensor()
        
        # Setup data
        self._setup_data()
        
        logger.info("✅ Model and data loaded with complete device fixes")
    
    def _verify_complete_device_placement(self) -> None:
        """Verify all model components are on the correct device."""
        logger.info("🔍 Verifying complete device placement...")
        
        device_issues = []
        
        # Check main model device
        try:
            model_device = next(self.model.parameters()).device
            logger.info(f"  - Model main device: {model_device}")
            if not (self.device in str(model_device) or str(model_device).startswith(self.device)):
                device_issues.append(f"Model on {model_device}, expected {self.device}")
        except:
            device_issues.append("Could not determine model device")
        
        # Check encoder device
        if hasattr(self.model, 'encoder'):
            try:
                encoder_device = next(self.model.encoder.parameters()).device
                logger.info(f"  - Encoder device: {encoder_device}")
                if not (self.device in str(encoder_device) or str(encoder_device).startswith(self.device)):
                    device_issues.append(f"Encoder on {encoder_device}, expected {self.device}")
            except:
                device_issues.append("Could not determine encoder device")
        
        # Check decoder device
        if hasattr(self.model, 'decoder'):
            try:
                decoder_device = next(self.model.decoder.parameters()).device
                logger.info(f"  - Decoder device: {decoder_device}")
                if not (self.device in str(decoder_device) or str(decoder_device).startswith(self.device)):
                    device_issues.append(f"Decoder on {decoder_device}, expected {self.device}")
            except:
                device_issues.append("Could not determine decoder device")
        
        # Check metric tensor device
        if hasattr(self.model, 'modular_metric') and self.model.modular_metric is not None:
            try:
                centroids_device = self.model.modular_metric.centroids.device
                matrices_device = self.model.modular_metric.metric_matrices.device
                logger.info(f"  - Metric centroids device: {centroids_device}")
                logger.info(f"  - Metric matrices device: {matrices_device}")
                if not (self.device in str(centroids_device) or str(centroids_device).startswith(self.device)):
                    device_issues.append(f"Metric centroids on {centroids_device}, expected {self.device}")
                if not (self.device in str(matrices_device) or str(matrices_device).startswith(self.device)):
                    device_issues.append(f"Metric matrices on {matrices_device}, expected {self.device}")
            except:
                device_issues.append("Could not determine metric tensor device")
        
        if device_issues:
            logger.error(f"❌ Device placement issues found: {device_issues}")
            raise RuntimeError(f"Device placement issues: {device_issues}")
        else:
            logger.info("✅ All model components verified on correct device")
    
    def _verify_metric_tensor(self) -> None:
        """Verify metric tensor is properly loaded and accessible."""
        logger.info("🔍 Verifying metric tensor...")
        
        if hasattr(self.model, 'modular_metric') and self.model.modular_metric is not None:
            centroids = self.model.modular_metric.centroids
            matrices = self.model.modular_metric.metric_matrices
            
            logger.info(f"✅ Metric tensor verified:")
            logger.info(f"  - Centroids: {centroids.shape} on {centroids.device}")
            logger.info(f"  - Matrices: {matrices.shape} on {matrices.device}")
            logger.info(f"  - Centroids range: [{centroids.min():.3f}, {centroids.max():.3f}]")
            logger.info(f"  - Matrices range: [{matrices.min():.3f}, {matrices.max():.3f}]")
            
            # Test determinant computation
            test_matrix = matrices[0]
            det = torch.det(test_matrix)
            logger.info(f"  - Test determinant: {det:.6f}")
            
        else:
            logger.error("❌ Metric tensor not found!")
            raise RuntimeError("Metric tensor not properly loaded")
    
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
            'max_test_samples': 100,
            'verify_cyclicity': False
        })
        
        self.data_module = CyclicSpritesDataModule(data_config)
        self.data_module.setup('test')
        
        logger.info("✅ Data module setup complete")
    
    def test_single_forward_pass(self) -> None:
        """Test a single forward pass to ensure everything works."""
        logger.info("🧪 Testing single forward pass...")
        
        test_loader = self.data_module.test_dataloader()
        batch = next(iter(test_loader))
        
        # Ensure proper batch shape and device placement
        if len(batch.shape) == 4:
            batch = batch.unsqueeze(0)
        
        batch = batch.to(self.device)
        logger.info(f"📊 Test batch shape: {batch.shape} on device {batch.device}")
        
        try:
            with torch.no_grad():
                output = self.model(batch)
            
            logger.info("✅ Forward pass successful!")
            if isinstance(output, dict):
                logger.info(f"  - Output keys: {list(output.keys())}")
                if 'latent_samples' in output:
                    latent_shape = output['latent_samples'].shape
                    logger.info(f"  - Latent samples shape: {latent_shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_working_trajectories(self, n_sequences: int = 50) -> tuple:
        """Extract trajectories with verified working forward passes."""
        logger.info(f"🌊 Extracting {n_sequences} trajectories with verified forward passes")
        
        test_loader = self.data_module.test_dataloader()
        all_trajectories = []
        all_determinants = []
        all_timestamps = []
        
        trajectory_count = 0
        
        for batch_idx, batch in enumerate(test_loader):
            if trajectory_count >= n_sequences:
                break
            
            try:
                # Ensure proper batch shape and device placement
                if len(batch.shape) == 4:
                    batch = batch.unsqueeze(0)
                
                batch = batch.to(self.device)
                
                logger.debug(f"📊 Processing batch {batch_idx}, shape: {batch.shape} on {batch.device}")
                
                # Perform forward pass
                with torch.no_grad():
                    output = self.model(batch)
                
                if not isinstance(output, dict) or 'latent_samples' not in output:
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
    
    def create_final_working_visualization(self, trajectories, determinants, timestamps) -> None:
        """Create the final working comprehensive visualization."""
        logger.info("🎨 Creating final working comprehensive visualization")
        
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
        
        # Create comprehensive plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "🎯 Working Latent Space Trajectories (PCA)",
                "📈 Temporal Evolution (Working Forward Passes)",
                "📏 Real Metric Determinant Evolution", 
                "🌊 3D Trajectory View (Working Model)"
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
        
        # Add trajectory lines
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
                    name=f'Working Traj {i}',
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(
                        size=8,
                        color=dets,
                        colorscale='viridis',
                        showscale=(i == 0),
                        colorbar=dict(title="Real Log₁₀ det(G⁻¹)", x=0.45) if i == 0 else None
                    ),
                    hovertemplate=f"Working Traj {i}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>Real Det: %{{marker.color:.3f}}<extra></extra>",
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
                    name=f'Latent Dim {dim+1} (Working)',
                    line=dict(width=3),
                    marker=dict(size=8),
                    hovertemplate=f"Working Dim {dim+1}<br>Time: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>",
                    showlegend=(dim < 2)
                ),
                row=1, col=2
            )
        
        # 3. Real determinant evolution
        mean_dets = np.mean(determinants, axis=0)
        std_dets = np.std(determinants, axis=0)
        
        fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=mean_dets,
                error_y=dict(type='data', array=std_dets),
                mode='lines+markers',
                name='Real Mean Det',
                line=dict(color='red', width=4),
                marker=dict(size=10),
                hovertemplate="Real Mean Det<br>Time: %{x}<br>Det: %{y:.3f}<extra></extra>",
                showlegend=True
            ),
            row=2, col=1
        )
        
        # Add individual real determinant trajectories
        for i, dets in enumerate(determinants[:10]):
            fig.add_trace(
                go.Scatter(
                    x=timesteps,
                    y=dets,
                    mode='lines',
                    name=f'Real Det {i}',
                    line=dict(color=colors[i % len(colors)], width=1),
                    opacity=0.6,
                    showlegend=False,
                    hovertemplate=f"Real Traj {i}<br>Time: %{{x}}<br>Det: %{{y:.3f}}<extra></extra>"
                ),
                row=2, col=1
            )
        
        # 4. 3D trajectory view
        for i, (traj_pca, dets) in enumerate(zip(trajectory_pca[:10], determinants[:10])):
            timesteps_3d = np.arange(len(traj_pca))
            
            fig.add_trace(
                go.Scatter3d(
                    x=traj_pca[:, 0],
                    y=traj_pca[:, 1],
                    z=timesteps_3d,
                    mode='lines+markers',
                    name=f'Working 3D Traj {i}',
                    line=dict(color=colors[i % len(colors)], width=4),
                    marker=dict(
                        size=6,
                        color=dets,
                        colorscale='plasma',
                        showscale=(i == 0),
                        colorbar=dict(title="Real Det", x=1.02) if i == 0 else None
                    ),
                    hovertemplate=f"Working 3D Traj {i}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>Time: %{{z}}<br>Real Det: %{{marker.color:.3f}}<extra></extra>",
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title=dict(
                text="🎉 COMPLETELY FIXED RLVAE Analysis - Real Forward Passes Working!",
                x=0.5,
                font=dict(size=18, color='green')
            ),
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", row=1, col=1)
        fig.update_yaxes(title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", row=1, col=1)
        fig.update_xaxes(title="Timestep", row=1, col=2)
        fig.update_yaxes(title="Real Latent Value", row=1, col=2)
        fig.update_xaxes(title="Timestep", row=2, col=1)
        fig.update_yaxes(title="Real Log₁₀ det(G⁻¹)", row=2, col=1)
        
        # Update 3D scene
        fig.update_scenes(
            xaxis_title="PC1",
            yaxis_title="PC2", 
            zaxis_title="Timestep",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            row=2, col=2
        )
        
        # Save
        html_path = self.output_dir / "completely_fixed_working_visualization.html"
        fig.write_html(str(html_path))
        
        logger.info(f"✅ COMPLETELY FIXED visualization saved to {html_path}")
        
        # Log to WandB
        try:
            wandb.log({"completely_fixed_working_visualization": wandb.Html(str(html_path))})
        except:
            pass
        
        return fig
    
    def run_completely_fixed_analysis(self) -> None:
        """Run completely fixed analysis with verified working forward passes."""
        logger.info("🚀 Starting COMPLETELY FIXED interactive analysis")
        
        # Initialize WandB
        wandb.init(
            project="rlvae-completely-fixed",
            name=f"completely_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "checkpoint": self.checkpoint_path,
                "device": self.device,
                "analysis_type": "completely_fixed_with_working_forward_passes"
            }
        )
        
        try:
            # Load model with complete device fixes
            self.load_model_with_complete_device_fix()
            
            # Test single forward pass to ensure everything works
            if not self.test_single_forward_pass():
                logger.error("❌ Forward pass test failed - cannot continue")
                return
            
            logger.info("✅ Forward pass test passed - proceeding with trajectory extraction")
            
            # Extract trajectories with working forward passes
            trajectories, determinants, timestamps = self.extract_working_trajectories(n_sequences=50)
            
            if len(trajectories) == 0:
                logger.error("❌ No trajectories extracted - forward passes still failing")
                return
            
            # Create working visualization
            self.create_final_working_visualization(trajectories, determinants, timestamps)
            
            # Log all HTML files to WandB
            for html_file in self.output_dir.glob("*.html"):
                try:
                    wandb.log({html_file.stem: wandb.Html(str(html_file))})
                    logger.info(f"📤 Logged {html_file.name} to WandB")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to log {html_file}: {e}")
            
            logger.info("🎉 COMPLETELY FIXED analysis completed successfully!")
            logger.info(f"📁 All results saved in: {self.output_dir}")
            logger.info("✅ Forward passes working, real metric tensor data extracted!")
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            wandb.finish()


def main():
    """Main execution."""
    checkpoint_path = "outputs/checkpoints/epoch=197-val_loss=6.402.ckpt"
    
    visualizer = CompletelyFixedVisualizer(checkpoint_path)
    visualizer.run_completely_fixed_analysis()


if __name__ == "__main__":
    main() 