#!/usr/bin/env python3
"""
Post-Training Comprehensive Visualizations for RLVAE
====================================================

This script loads a trained RLVAE checkpoint and generates comprehensive visualizations
that might be too time-consuming during training, including:
- Metric tensor visualizations and heatmaps
- Detailed latent space analysis
- Flow dynamics and geodesic analysis
- Comprehensive evaluation metrics
- Interactive visualizations
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from omegaconf import DictConfig
import warnings
warnings.filterwarnings('ignore')

# Add src to path like run_experiment.py does
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
from visualizations.manager import VisualizationManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PostTrainingVisualizer:
    """Comprehensive post-training visualization manager for RLVAE models."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """
        Initialize the post-training visualizer.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.model = None
        self.data_module = None
        self.test_loader = None
        
        # Create output directory for visualizations
        self.output_dir = Path("outputs/post_training_viz") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📊 Post-training visualizer initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🖥️ Device: {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup and return the appropriate device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model(self) -> None:
        """Load the trained model from checkpoint."""
        logger.info(f"🔄 Loading model from {self.checkpoint_path}")
        
        try:
            # Load checkpoint (set weights_only=False for compatibility)
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            
            # Extract model state and hyperparameters
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Remove lightning module prefix if present
                state_dict = {k.replace('model.', '') if k.startswith('model.') else k: v 
                            for k, v in state_dict.items()}
            else:
                state_dict = checkpoint
            
            # Create model with appropriate configuration
            model_config = self._extract_model_config(checkpoint)
            self.model = ModularRiemannianFlowVAE(model_config)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"📐 Model: {type(self.model).__name__}")
            logger.info(f"🎯 Latent dim: {self.model.latent_dim}")
            logger.info(f"🌊 Flows: {self.model.n_flows}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _extract_model_config(self, checkpoint: Dict[str, Any]) -> DictConfig:
        """Extract model configuration from checkpoint."""
        from omegaconf import DictConfig
        
        # Default configuration based on the training config we found
        config_dict = {
            'input_dim': [3, 28, 28],  # Based on ColoredCircles dataset
            'latent_dim': 2,
            'n_flows': 8,
            'flow_hidden_size': 256,
            'flow_n_blocks': 2,
            'flow_n_hidden': 1,
            'epsilon': 1e-6,
            'encoder': {'architecture': 'mlp'},
            'decoder': {'architecture': 'mlp'},
            'beta': 1.0,
            'riemannian_beta': 8.0,
            'posterior': {'type': 'riemannian_metric'},
            'sampling': {'method': 'geodesic', 'use_riemannian': True},
            'loop': {'mode': 'open', 'penalty': 5.0},
            'metric': {'path': 'metric_T0.7_scaled.pt', 'temperature_override': 3.0},
            'pretrained': {
                'encoder_path': 'data/pretrained/encoder.pt',
                'decoder_path': 'data/pretrained/decoder.pt',
                'metric_path': 'data/pretrained/metric_T0.7_scaled.pt'
            }
        }
        
        # Try to extract from checkpoint if available
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
            for key in config_dict.keys():
                if key in hparams:
                    config_dict[key] = hparams[key]
        
        return DictConfig(config_dict)
    
    def setup_data(self) -> None:
        """Setup the data module and test loader."""
        logger.info("🔄 Setting up data module")
        
        # Create data module with same config as training
        from omegaconf import DictConfig
        
        data_config = DictConfig({
            'train_path': 'data/sprites/ColoredCircles_train.pt',
            'test_path': 'data/sprites/ColoredCircles_test.pt',
            'train_meta_path': 'data/sprites/ColoredCircles_train_params.pt',
            'test_meta_path': 'data/sprites/ColoredCircles_test_params.pt',
            'sequence_length': 10,
            'image_size': [28, 28],
            'channels': 3,
            'batch_size': 32,  # Smaller batch for visualization
            'num_workers': 2,
            'pin_memory': True,
            'max_train_samples': 1000,  # Limit for visualization
            'max_test_samples': 1000,
            'verify_cyclicity': False,
            'cyclicity_threshold': 0.01
        })
        
        self.data_module = CyclicSpritesDataModule(data_config)
        self.data_module.setup('test')
        self.test_loader = self.data_module.test_dataloader()
        
        logger.info(f"✅ Data module setup complete")
        logger.info(f"📊 Test batches: {len(self.test_loader)}")
    
    def visualize_metric_tensor(self) -> None:
        """Create comprehensive metric tensor visualizations."""
        logger.info("🎨 Creating metric tensor visualizations")
        
        if not hasattr(self.model, 'metric_tensor') or self.model.metric_tensor is None:
            logger.warning("⚠️ No metric tensor found in model")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Riemannian Metric Tensor Analysis', fontsize=16, fontweight='bold')
        
        try:
            # Get a batch of data for metric computation
            batch = next(iter(self.test_loader))
            sequences, _ = batch
            sequences = sequences.to(self.device)
            batch_size, seq_len = sequences.shape[:2]
            
            # Flatten sequences for encoding
            x_flat = sequences.view(-1, *sequences.shape[2:])
            
            with torch.no_grad():
                # Encode to latent space
                mu, logvar = self.model.encoder(x_flat)
                z = self.model.reparameterize(mu, logvar)
                
                # Compute metric at different points
                metrics = []
                latent_points = []
                
                # Sample points across latent space
                n_points = 100
                for i in range(n_points):
                    idx = torch.randint(0, len(z), (1,))
                    z_point = z[idx]
                    latent_points.append(z_point.cpu().numpy())
                    
                    # Compute metric at this point
                    if hasattr(self.model.metric_tensor, 'compute_metric'):
                        metric = self.model.metric_tensor.compute_metric(z_point)
                        metrics.append(metric.cpu().numpy())
            
            if metrics:
                metrics = np.array(metrics)
                latent_points = np.array(latent_points).squeeze()
                
                # Plot 1: Metric determinant heatmap
                if metrics.shape[-1] == 4:  # 2x2 metric for 2D latent space
                    dets = np.linalg.det(metrics.reshape(-1, 2, 2))
                    scatter = axes[0, 0].scatter(latent_points[:, 0], latent_points[:, 1], 
                                              c=dets, cmap='viridis', alpha=0.7)
                    axes[0, 0].set_title('Metric Determinant')
                    axes[0, 0].set_xlabel('Latent Dim 1')
                    axes[0, 0].set_ylabel('Latent Dim 2')
                    plt.colorbar(scatter, ax=axes[0, 0])
                
                # Plot 2: Metric trace heatmap
                if metrics.shape[-1] == 4:
                    traces = np.trace(metrics.reshape(-1, 2, 2), axis1=1, axis2=2)
                    scatter = axes[0, 1].scatter(latent_points[:, 0], latent_points[:, 1], 
                                              c=traces, cmap='plasma', alpha=0.7)
                    axes[0, 1].set_title('Metric Trace')
                    axes[0, 1].set_xlabel('Latent Dim 1')
                    axes[0, 1].set_ylabel('Latent Dim 2')
                    plt.colorbar(scatter, ax=axes[0, 1])
                
                # Plot 3: Average metric eigenvalues
                if metrics.shape[-1] == 4:
                    eigenvals = np.linalg.eigvals(metrics.reshape(-1, 2, 2))
                    axes[1, 0].hist(eigenvals[:, 0], alpha=0.7, label='λ1', bins=30)
                    axes[1, 0].hist(eigenvals[:, 1], alpha=0.7, label='λ2', bins=30)
                    axes[1, 0].set_title('Metric Eigenvalue Distribution')
                    axes[1, 0].set_xlabel('Eigenvalue')
                    axes[1, 0].set_ylabel('Frequency')
                    axes[1, 0].legend()
                
                # Plot 4: Condition number
                if metrics.shape[-1] == 4:
                    cond_numbers = []
                    for i in range(len(metrics)):
                        metric_mat = metrics[i].reshape(2, 2)
                        cond = np.linalg.cond(metric_mat)
                        if np.isfinite(cond):
                            cond_numbers.append(cond)
                    
                    if cond_numbers:
                        axes[1, 1].hist(cond_numbers, bins=30, alpha=0.7)
                        axes[1, 1].set_title('Metric Condition Number Distribution')
                        axes[1, 1].set_xlabel('Condition Number')
                        axes[1, 1].set_ylabel('Frequency')
                        axes[1, 1].set_yscale('log')
        
        except Exception as e:
            logger.warning(f"⚠️ Error in metric visualization: {e}")
            # Create placeholder plots
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'Metric visualization\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Metric Analysis')
        
        plt.tight_layout()
        
        # Save and log
        metric_path = self.output_dir / 'metric_tensor_analysis.png'
        plt.savefig(metric_path, dpi=300, bbox_inches='tight')
        
        if wandb.run:
            wandb.log({"post_training/metric_tensor_analysis": wandb.Image(str(metric_path))})
        
        plt.close()
        logger.info(f"✅ Metric tensor visualization saved to {metric_path}")
    
    def visualize_latent_space(self) -> None:
        """Create comprehensive latent space visualizations."""
        logger.info("🎨 Creating latent space visualizations")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Latent Space Analysis', fontsize=16, fontweight='bold')
        
        try:
            # Collect latent representations
            latent_points = []
            reconstructions = []
            originals = []
            
            with torch.no_grad():
                for i, batch in enumerate(self.test_loader):
                    if i >= 10:  # Limit to first 10 batches for speed
                        break
                    
                    sequences, _ = batch
                    sequences = sequences.to(self.device)
                    
                    # Flatten for encoding
                    x_flat = sequences.view(-1, *sequences.shape[2:])
                    
                    # Encode
                    mu, logvar = self.model.encoder(x_flat)
                    z = self.model.reparameterize(mu, logvar)
                    
                    # Decode
                    x_recon = self.model.decoder(z)
                    
                    latent_points.append(z.cpu().numpy())
                    reconstructions.append(x_recon.cpu().numpy())
                    originals.append(x_flat.cpu().numpy())
            
            latent_points = np.concatenate(latent_points, axis=0)
            reconstructions = np.concatenate(reconstructions, axis=0)
            originals = np.concatenate(originals, axis=0)
            
            # Plot 1: Latent space scatter (2D)
            if latent_points.shape[1] >= 2:
                axes[0, 0].scatter(latent_points[:, 0], latent_points[:, 1], 
                                 alpha=0.6, s=1)
                axes[0, 0].set_title('Latent Space Distribution')
                axes[0, 0].set_xlabel('Latent Dim 1')
                axes[0, 0].set_ylabel('Latent Dim 2')
                axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Latent space density
            if latent_points.shape[1] >= 2:
                axes[0, 1].hist2d(latent_points[:, 0], latent_points[:, 1], 
                                bins=50, cmap='Blues')
                axes[0, 1].set_title('Latent Space Density')
                axes[0, 1].set_xlabel('Latent Dim 1')
                axes[0, 1].set_ylabel('Latent Dim 2')
            
            # Plot 3: Dimension-wise distributions
            n_dims = min(latent_points.shape[1], 4)
            for dim in range(n_dims):
                axes[0, 2].hist(latent_points[:, dim], alpha=0.7, 
                              label=f'Dim {dim}', bins=50)
            axes[0, 2].set_title('Latent Dimension Distributions')
            axes[0, 2].set_xlabel('Value')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].legend()
            
            # Plot 4: Reconstruction quality scatter
            mse_scores = np.mean((originals - reconstructions) ** 2, axis=(1, 2, 3))
            axes[1, 0].scatter(latent_points[:, 0], latent_points[:, 1], 
                             c=mse_scores, cmap='viridis', alpha=0.6)
            axes[1, 0].set_title('Reconstruction Quality in Latent Space')
            axes[1, 0].set_xlabel('Latent Dim 1')
            axes[1, 0].set_ylabel('Latent Dim 2')
            
            # Plot 5: Latent space covariance
            cov_matrix = np.cov(latent_points.T)
            im = axes[1, 1].imshow(cov_matrix, cmap='RdBu', aspect='auto')
            axes[1, 1].set_title('Latent Space Covariance Matrix')
            plt.colorbar(im, ax=axes[1, 1])
            
            # Plot 6: Example reconstructions
            n_examples = 8
            example_indices = np.random.choice(len(originals), n_examples, replace=False)
            
            # Create a grid of original vs reconstruction
            grid = np.zeros((2 * 28, n_examples * 28, 3))
            for i, idx in enumerate(example_indices):
                orig = originals[idx].transpose(1, 2, 0)
                recon = reconstructions[idx].transpose(1, 2, 0)
                
                # Normalize to [0, 1]
                orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
                recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
                
                grid[:28, i*28:(i+1)*28] = orig
                grid[28:, i*28:(i+1)*28] = recon
            
            axes[1, 2].imshow(grid)
            axes[1, 2].set_title('Original (top) vs Reconstruction (bottom)')
            axes[1, 2].axis('off')
            
        except Exception as e:
            logger.warning(f"⚠️ Error in latent space visualization: {e}")
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'Visualization\nerror', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        # Save and log
        latent_path = self.output_dir / 'latent_space_analysis.png'
        plt.savefig(latent_path, dpi=300, bbox_inches='tight')
        
        if wandb.run:
            wandb.log({"post_training/latent_space_analysis": wandb.Image(str(latent_path))})
        
        plt.close()
        logger.info(f"✅ Latent space visualization saved to {latent_path}")
    
    def visualize_flow_dynamics(self) -> None:
        """Visualize normalizing flow dynamics and transformations."""
        logger.info("🎨 Creating flow dynamics visualizations")
        
        if not hasattr(self.model, 'flows') or self.model.flows is None:
            logger.warning("⚠️ No flows found in model")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Normalizing Flow Dynamics Analysis', fontsize=16, fontweight='bold')
        
        try:
            # Get a sample batch
            batch = next(iter(self.test_loader))
            sequences, _ = batch
            sequences = sequences.to(self.device)
            x_flat = sequences.view(-1, *sequences.shape[2:])[:100]  # Limit for speed
            
            with torch.no_grad():
                # Encode to get initial latent
                mu, logvar = self.model.encoder(x_flat)
                z0 = self.model.reparameterize(mu, logvar)
                
                # Apply flows step by step
                flow_states = [z0.cpu().numpy()]
                z_current = z0
                
                for i, flow in enumerate(self.model.flows):
                    z_current, _ = flow(z_current)
                    flow_states.append(z_current.cpu().numpy())
                    if i >= 5:  # Limit number of flows to visualize
                        break
            
            # Plot flow evolution
            n_flows = min(len(flow_states), 6)
            
            for i in range(n_flows):
                row = i // 3
                col = i % 3
                
                if row < 2 and col < 3:
                    z_state = flow_states[i]
                    if z_state.shape[1] >= 2:
                        axes[row, col].scatter(z_state[:, 0], z_state[:, 1], 
                                             alpha=0.6, s=2)
                        title = f'Flow Step {i}' if i > 0 else 'Initial Distribution'
                        axes[row, col].set_title(title)
                        axes[row, col].set_xlabel('Latent Dim 1')
                        axes[row, col].set_ylabel('Latent Dim 2')
                        axes[row, col].grid(True, alpha=0.3)
        
        except Exception as e:
            logger.warning(f"⚠️ Error in flow visualization: {e}")
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'Flow visualization\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Flow Analysis')
        
        plt.tight_layout()
        
        # Save and log
        flow_path = self.output_dir / 'flow_dynamics_analysis.png'
        plt.savefig(flow_path, dpi=300, bbox_inches='tight')
        
        if wandb.run:
            wandb.log({"post_training/flow_dynamics_analysis": wandb.Image(str(flow_path))})
        
        plt.close()
        logger.info(f"✅ Flow dynamics visualization saved to {flow_path}")
    
    def run_comprehensive_evaluation(self) -> Dict[str, float]:
        """Run comprehensive evaluation metrics."""
        logger.info("📊 Running comprehensive evaluation")
        
        metrics = {}
        
        try:
            # Simple reconstruction evaluation
            recon_losses = []
            kl_losses = []
            total_losses = []
            
            with torch.no_grad():
                for i, batch in enumerate(self.test_loader):
                    if i >= 20:  # Limit for speed
                        break
                    
                    sequences, _ = batch
                    sequences = sequences.to(self.device)
                    x_flat = sequences.view(-1, *sequences.shape[2:])
                    
                    # Forward pass
                    mu, logvar = self.model.encoder(x_flat)
                    z = self.model.reparameterize(mu, logvar)
                    x_recon = self.model.decoder(z)
                    
                    # Compute losses
                    recon_loss = torch.nn.functional.mse_loss(x_recon, x_flat, reduction='mean')
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(x_flat)
                    total_loss = recon_loss + self.model.beta * kl_loss
                    
                    recon_losses.append(recon_loss.item())
                    kl_losses.append(kl_loss.item())
                    total_losses.append(total_loss.item())
            
            # Compute average metrics
            metrics['avg_reconstruction_loss'] = np.mean(recon_losses)
            metrics['avg_kl_loss'] = np.mean(kl_losses)
            metrics['avg_total_loss'] = np.mean(total_losses)
            metrics['std_reconstruction_loss'] = np.std(recon_losses)
            metrics['std_kl_loss'] = np.std(kl_losses)
            
            # Model-specific metrics
            metrics['model_latent_dim'] = self.model.latent_dim
            metrics['model_n_flows'] = self.model.n_flows
            metrics['model_beta'] = self.model.beta
            if hasattr(self.model, 'riemannian_beta'):
                metrics['model_riemannian_beta'] = self.model.riemannian_beta
            
            logger.info("✅ Basic evaluation completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Error in evaluation: {e}")
            metrics['evaluation_error'] = 1.0
        
        return metrics
    
    def create_summary_report(self, metrics: Dict[str, float]) -> None:
        """Create a summary report of all analyses."""
        logger.info("📋 Creating summary report")
        
        # Create summary figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('RLVAE Model Analysis Summary', fontsize=16, fontweight='bold')
        
        # Metrics bar chart
        if metrics:
            metric_names = list(metrics.keys())[:10]  # Top 10 metrics
            metric_values = [metrics[name] for name in metric_names]
            
            ax1.barh(range(len(metric_names)), metric_values)
            ax1.set_yticks(range(len(metric_names)))
            ax1.set_yticklabels(metric_names)
            ax1.set_title('Key Performance Metrics')
            ax1.set_xlabel('Value')
        
        # Model architecture summary
        arch_info = [
            f"Model: {type(self.model).__name__}",
            f"Latent Dim: {self.model.latent_dim}",
            f"Flows: {self.model.n_flows}",
            f"Beta: {self.model.beta}",
            f"Riemannian Beta: {getattr(self.model, 'riemannian_beta', 'N/A')}",
            f"Device: {self.device}",
            f"Checkpoint: {Path(self.checkpoint_path).name}"
        ]
        
        ax2.text(0.1, 0.9, '\n'.join(arch_info), transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax2.set_title('Model Configuration')
        ax2.axis('off')
        
        # Training progress (if available)
        ax3.text(0.5, 0.5, 'Training progress\nvisualization\n(if available)', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Training Progress')
        
        # Generated samples grid (if available)
        ax4.text(0.5, 0.5, 'Generated samples\ngrid\n(if available)', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Generated Samples')
        
        plt.tight_layout()
        
        # Save summary
        summary_path = self.output_dir / 'analysis_summary.png'
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        
        if wandb.run:
            wandb.log({"post_training/analysis_summary": wandb.Image(str(summary_path))})
        
        plt.close()
        
        # Save metrics to file
        metrics_path = self.output_dir / 'evaluation_metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write("RLVAE Post-Training Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Device: {self.device}\n\n")
            
            f.write("Model Configuration:\n")
            f.write("-" * 20 + "\n")
            for info in arch_info:
                f.write(f"{info}\n")
            f.write("\n")
            
            f.write("Evaluation Metrics:\n")
            f.write("-" * 20 + "\n")
            for name, value in metrics.items():
                f.write(f"{name}: {value:.4f}\n")
        
        logger.info(f"✅ Summary report saved to {summary_path}")
        logger.info(f"📊 Metrics saved to {metrics_path}")
    
    def run_all_visualizations(self) -> None:
        """Run all post-training visualizations and analyses."""
        logger.info("🚀 Starting comprehensive post-training analysis")
        
        # Initialize WandB for logging (optional)
        try:
            wandb.init(
                project="rlvae-post-training-analysis",
                name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "checkpoint_path": self.checkpoint_path,
                    "device": self.device,
                    "analysis_type": "comprehensive_post_training"
                }
            )
        except Exception as e:
            logger.warning(f"⚠️ WandB initialization failed: {e}")
        
        # Load model and data
        self.load_model()
        self.setup_data()
        
        # Run all visualizations
        self.visualize_metric_tensor()
        self.visualize_latent_space()
        self.visualize_flow_dynamics()
        
        # Run comprehensive evaluation
        metrics = self.run_comprehensive_evaluation()
        
        # Create summary report
        self.create_summary_report(metrics)
        
        logger.info("🎉 All post-training analyses completed successfully!")
        logger.info(f"📁 Results saved in: {self.output_dir}")
        
        if wandb.run:
            wandb.finish()


def main():
    """Main execution function."""
    # Configuration
    checkpoint_path = "outputs/checkpoints/epoch=197-val_loss=6.402.ckpt"
    device = 'auto'
    
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        logger.info("🔍 Available checkpoints:")
        checkpoint_dir = Path(checkpoint_path).parent
        if checkpoint_dir.exists():
            for ckpt in sorted(checkpoint_dir.glob("*.ckpt")):
                logger.info(f"  - {ckpt.name}")
        return
    
    # Run comprehensive analysis
    visualizer = PostTrainingVisualizer(checkpoint_path, device)
    visualizer.run_all_visualizations()


if __name__ == "__main__":
    main() 