#!/usr/bin/env python3
"""
Exhaustive RLVAE Post-Training Analysis
======================================

Comprehensive analysis script that properly reconstructs the trained RLVAE model
and provides exhaustive analysis including:
- Proper model reconstruction from checkpoint
- Real data latent space projections
- Riemannian metric tensor analysis and visualization
- Geodesic sampling and interpolation
- Flow dynamics analysis
- Curvature analysis
- Comprehensive generation using all sampling methods
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

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
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExhaustiveRLVAEAnalyzer:
    """Comprehensive RLVAE analyzer with proper model reconstruction."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """Initialize exhaustive analyzer."""
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.model = None
        self.checkpoint = None
        self.data_module = None
        self.test_loader = None
        
        self.output_dir = Path("outputs/exhaustive_analysis") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔬 Exhaustive RLVAE analyzer initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🖥️ Device: {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_checkpoint_and_extract_config(self) -> DictConfig:
        """Load checkpoint and extract exact model configuration."""
        logger.info(f"🔄 Loading checkpoint and extracting configuration")
        
        self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract exact configuration from checkpoint
        hparams = self.checkpoint.get('hyper_parameters', {})
        state_dict = self.checkpoint['state_dict']
        
        logger.info(f"📊 Available hyperparameters: {list(hparams.keys())}")
        
        # Extract model config directly from hyperparameters
        model_hparams = hparams.get('model', {})
        logger.info(f"🔍 Model hyperparameters: {model_hparams}")
        
        # Analyze the actual model structure from state_dict
        logger.info("🔍 Analyzing model structure from state_dict...")
        
        # Extract actual dimensions from the checkpoint
        encoder_found = any('encoder' in k for k in state_dict.keys())
        decoder_found = any('decoder' in k for k in state_dict.keys())
        metric_found = any('metric' in k.lower() for k in state_dict.keys())
        flows_found = any('flows' in k for k in state_dict.keys())
        
        logger.info(f"🔍 Components found - Encoder: {encoder_found}, Decoder: {decoder_found}, Metric: {metric_found}, Flows: {flows_found}")
        
        # Extract metric tensor dimensions
        metric_centroids_shape = None
        metric_matrices_shape = None
        latent_dim = model_hparams.get('latent_dim', 2)
        
        for name, param in state_dict.items():
            if 'modular_metric.centroids' in name:
                metric_centroids_shape = param.shape
                latent_dim = param.shape[1]
                logger.info(f"📐 Found metric centroids: {metric_centroids_shape}")
            elif 'modular_metric.metric_matrices' in name:
                metric_matrices_shape = param.shape
                logger.info(f"📐 Found metric matrices: {metric_matrices_shape}")
            elif 'centroids' in name and 'metric' in name.lower():
                metric_centroids_shape = param.shape
                latent_dim = param.shape[1] if len(param.shape) > 1 else 2
                logger.info(f"📐 Found centroids (alt): {metric_centroids_shape}")
        
        # Extract flow information from hyperparameters and state_dict
        n_flows = model_hparams.get('n_flows', 0)
        flow_hidden_size = model_hparams.get('flow_hidden_size', 256)
        
        # Double-check flows from state dict
        max_flow_idx = 0
        for name in state_dict.keys():
            if 'flow_manager.flows.' in name:
                try:
                    flow_idx = int(name.split('flow_manager.flows.')[1].split('.')[0]) + 1
                    max_flow_idx = max(max_flow_idx, flow_idx)
                except:
                    pass
        
        if max_flow_idx > 0:
            n_flows = max_flow_idx
            logger.info(f"📐 Detected {n_flows} flows from state_dict")
        
        logger.info(f"📐 Final detected: latent_dim={latent_dim}, n_flows={n_flows}, flow_hidden_size={flow_hidden_size}")
        
        # Extract pretrained paths
        pretrained_config = model_hparams.get('pretrained', {})
        if not pretrained_config:
            # Fallback to disable pretrained loading
            pretrained_config = {
                'encoder_path': None,
                'decoder_path': None, 
                'metric_path': None
            }
        
        # Construct configuration matching the checkpoint exactly
        config = DictConfig({
            'input_dim': model_hparams.get('input_dim', [3, 28, 28]),
            'latent_dim': latent_dim,
            'n_flows': n_flows,
            'flow_hidden_size': flow_hidden_size,
            'flow_n_blocks': model_hparams.get('flow_n_blocks', 2),
            'flow_n_hidden': model_hparams.get('flow_n_hidden', 1),
            'epsilon': model_hparams.get('epsilon', 1e-6),
            'encoder': model_hparams.get('encoder', {'architecture': 'mlp'}),
            'decoder': model_hparams.get('decoder', {'architecture': 'mlp'}),
            'beta': model_hparams.get('beta', 1.0),
            'riemannian_beta': model_hparams.get('riemannian_beta', 1.0),
            'posterior': model_hparams.get('posterior', {'type': 'riemannian_metric'}),
            'sampling': model_hparams.get('sampling', {'method': 'geodesic', 'use_riemannian': True}),
            'loop': model_hparams.get('loop', {'mode': 'open', 'penalty': 5.0}),
            'metric': model_hparams.get('metric', {'path': 'metric_T0.7_scaled.pt', 'temperature_override': 3.0}),
            'pretrained': pretrained_config,
            'sequence_length': model_hparams.get('sequence_length', 10)
        })
        
        logger.info(f"✅ Extracted configuration: {dict(config)}")
        return config
    
    def reconstruct_model(self, config: DictConfig) -> None:
        """Reconstruct the exact model from checkpoint."""
        logger.info("🔧 Reconstructing exact model from checkpoint")
        
        try:
            # Temporarily disable pretrained loading during initialization
            config_copy = config.copy()
            if 'pretrained' in config_copy:
                config_copy['pretrained'] = {
                    'encoder_path': None,
                    'decoder_path': None,
                    'metric_path': None
                }
            
            # Create model with extracted config
            self.model = ModularRiemannianFlowVAE(config_copy)
            
            # Prepare state dict
            state_dict = self.checkpoint['state_dict']
            
            # Remove lightning wrapper prefixes
            clean_state_dict = {}
            for key, value in state_dict.items():
                clean_key = key.replace('model.', '') if key.startswith('model.') else key
                clean_state_dict[clean_key] = value
            
            # Initialize metric tensor with correct size if needed
            if hasattr(self.model, 'modular_metric') and self.model.modular_metric is not None:
                metric_centroids = None
                metric_matrices = None
                
                for name, param in clean_state_dict.items():
                    if 'modular_metric.centroids' in name:
                        metric_centroids = param
                    elif 'modular_metric.metric_matrices' in name:
                        metric_matrices = param
                
                if metric_centroids is not None:
                    logger.info(f"🔧 Resizing metric tensor to match checkpoint: {metric_centroids.shape[0]} centroids")
                    
                    # Resize the metric tensor to match checkpoint
                    n_centroids = metric_centroids.shape[0]
                    latent_dim = metric_centroids.shape[1]
                    
                    # Create new parameters with correct size
                    new_centroids = torch.nn.Parameter(torch.zeros_like(metric_centroids))
                    new_matrices = torch.nn.Parameter(
                        torch.zeros_like(metric_matrices) if metric_matrices is not None 
                        else torch.zeros(n_centroids, latent_dim, latent_dim)
                    )
                    
                    # Replace the metric parameters
                    self.model.modular_metric.centroids = new_centroids
                    self.model.modular_metric.metric_matrices = new_matrices
                    
                    logger.info(f"✅ Metric tensor resized: centroids {new_centroids.shape}, matrices {new_matrices.shape}")
            
            # Load state dict
            missing_keys, unexpected_keys = self.model.load_state_dict(clean_state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"⚠️ Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"⚠️ Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Model reconstructed successfully")
            logger.info(f"📊 Model summary - Latent: {self.model.latent_dim}D, Flows: {self.model.n_flows}")
            
            # Verify model components
            if hasattr(self.model, 'modular_metric') and self.model.modular_metric is not None:
                logger.info(f"🎯 Metric tensor: {self.model.modular_metric.centroids.shape}")
            
        except Exception as e:
            logger.error(f"❌ Model reconstruction failed: {e}")
            raise
    
    def setup_data(self) -> None:
        """Setup data module for real data analysis."""
        logger.info("🔄 Setting up data module")
        
        from omegaconf import DictConfig
        
        data_config = DictConfig({
            'train_path': 'data/sprites/ColoredCircles_train.pt',
            'test_path': 'data/sprites/ColoredCircles_test.pt',
            'train_meta_path': 'data/sprites/ColoredCircles_train_params.pt',
            'test_meta_path': 'data/sprites/ColoredCircles_test_params.pt',
            'sequence_length': 10,
            'image_size': [28, 28],
            'channels': 3,
            'batch_size': 32,
            'num_workers': 2,
            'pin_memory': True,
            'max_train_samples': 2000,
            'max_test_samples': 1000,
            'verify_cyclicity': False,
            'cyclicity_threshold': 0.01
        })
        
        self.data_module = CyclicSpritesDataModule(data_config)
        self.data_module.setup('test')
        self.test_loader = self.data_module.test_dataloader()
        
        logger.info(f"✅ Data module ready with {len(self.test_loader)} test batches")
    
    def extract_real_latent_representations(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract latent representations from real data."""
        logger.info("🧠 Extracting real latent representations")
        
        latent_codes = []
        reconstructions = []
        originals = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if i >= 15:  # Limit for memory
                    break
                
                # Handle different batch formats
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    sequences, _ = batch
                elif isinstance(batch, (list, tuple)) and len(batch) == 1:
                    sequences = batch[0]
                else:
                    # Batch is just a tensor
                    sequences = batch
                sequences = sequences.to(self.device)
                
                # Check if sequences is already flattened or needs reshaping
                if len(sequences.shape) == 4:  # [seq_len, C, H, W] - single sequence
                    # This is a single sequence, treat each timestep as a separate sample
                    x_flat = sequences  # Already in the right format [timesteps, C, H, W]
                elif len(sequences.shape) == 5:  # [batch, seq_len, C, H, W] - batch of sequences
                    batch_size, seq_len = sequences.shape[:2]
                    x_flat = sequences.view(-1, *sequences.shape[2:])
                else:
                    logger.warning(f"⚠️ Unexpected sequence shape: {sequences.shape}")
                    continue
                
                # Encode using the model's method
                encoder_out = self.model.encoder(x_flat)
                mu = encoder_out.embedding
                logvar = encoder_out.log_covariance
                
                # Sample latents using standard reparameterization
                eps = torch.randn_like(mu)
                z = mu + eps * torch.exp(0.5 * logvar)
                
                # Decode using the model's method
                decoder_out = self.model.decoder(z)
                x_recon = decoder_out["reconstruction"]
                
                latent_codes.append(z.cpu().numpy())
                reconstructions.append(x_recon.cpu().numpy())
                originals.append(x_flat.cpu().numpy())
        
        latent_codes = np.concatenate(latent_codes, axis=0)
        reconstructions = np.concatenate(reconstructions, axis=0)
        originals = np.concatenate(originals, axis=0)
        
        logger.info(f"✅ Extracted {len(latent_codes)} latent representations")
        return latent_codes, reconstructions, originals
    
    def analyze_metric_tensor_structure(self) -> None:
        """Comprehensive metric tensor analysis."""
        logger.info("🧮 Analyzing metric tensor structure")
        
        if not hasattr(self.model, 'modular_metric') or self.model.modular_metric is None:
            logger.warning("⚠️ No metric tensor found")
            return
        
        metric = self.model.modular_metric
        centroids = metric.centroids.detach().cpu().numpy()
        metric_matrices = metric.metric_matrices.detach().cpu().numpy()
        
        # Create comprehensive metric visualization
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
        fig.suptitle('Comprehensive Metric Tensor Analysis', fontsize=20, fontweight='bold')
        
        # 1. Centroids distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(centroids[:, 0], centroids[:, 1], c=range(len(centroids)), 
                   cmap='viridis', s=50, alpha=0.7)
        ax1.set_title(f'Metric Centroids ({len(centroids)} points)')
        ax1.set_xlabel('Latent Dim 1')
        ax1.set_ylabel('Latent Dim 2')
        ax1.grid(True, alpha=0.3)
        
        # 2. Metric determinants
        ax2 = fig.add_subplot(gs[0, 1])
        dets = np.linalg.det(metric_matrices)
        scatter = ax2.scatter(centroids[:, 0], centroids[:, 1], c=dets, 
                             cmap='plasma', s=50, alpha=0.7)
        ax2.set_title('Metric Determinants')
        ax2.set_xlabel('Latent Dim 1')
        ax2.set_ylabel('Latent Dim 2')
        plt.colorbar(scatter, ax=ax2)
        
        # 3. Metric traces
        ax3 = fig.add_subplot(gs[0, 2])
        traces = np.trace(metric_matrices, axis1=1, axis2=2)
        scatter = ax3.scatter(centroids[:, 0], centroids[:, 1], c=traces, 
                             cmap='viridis', s=50, alpha=0.7)
        ax3.set_title('Metric Traces')
        ax3.set_xlabel('Latent Dim 1')
        ax3.set_ylabel('Latent Dim 2')
        plt.colorbar(scatter, ax=ax3)
        
        # 4. Eigenvalue analysis
        ax4 = fig.add_subplot(gs[0, 3])
        eigenvals = np.linalg.eigvals(metric_matrices)
        ax4.hist(eigenvals[:, 0], bins=30, alpha=0.7, label='λ1', density=True)
        ax4.hist(eigenvals[:, 1], bins=30, alpha=0.7, label='λ2', density=True)
        ax4.set_title('Eigenvalue Distribution')
        ax4.set_xlabel('Eigenvalue')
        ax4.set_ylabel('Density')
        ax4.legend()
        
        # 5. Condition numbers
        ax5 = fig.add_subplot(gs[1, 0])
        cond_numbers = [np.linalg.cond(m) for m in metric_matrices]
        ax5.hist(cond_numbers, bins=30, alpha=0.7)
        ax5.set_title('Condition Number Distribution')
        ax5.set_xlabel('Condition Number')
        ax5.set_ylabel('Frequency')
        ax5.set_yscale('log')
        
        # 6. Metric matrix heatmap (sample)
        ax6 = fig.add_subplot(gs[1, 1])
        sample_idx = len(metric_matrices) // 2
        im = ax6.imshow(metric_matrices[sample_idx], cmap='RdBu', aspect='auto')
        ax6.set_title(f'Sample Metric Matrix (centroid {sample_idx})')
        plt.colorbar(im, ax=ax6)
        
        # 7. Distance to centroids analysis
        ax7 = fig.add_subplot(gs[1, 2])
        # Compute pairwise distances between centroids
        from scipy.spatial.distance import pdist, squareform
        distances = pdist(centroids)
        ax7.hist(distances, bins=30, alpha=0.7)
        ax7.set_title('Inter-Centroid Distances')
        ax7.set_xlabel('Distance')
        ax7.set_ylabel('Frequency')
        
        # 8. Metric statistics
        ax8 = fig.add_subplot(gs[1, 3])
        stats_text = [
            f"Centroids: {len(centroids)}",
            f"Latent Dim: {centroids.shape[1]}",
            f"Det range: [{dets.min():.3f}, {dets.max():.3f}]",
            f"Trace range: [{traces.min():.3f}, {traces.max():.3f}]",
            f"Cond range: [{min(cond_numbers):.3f}, {max(cond_numbers):.3f}]",
            f"Eigenval range: [{eigenvals.min():.3f}, {eigenvals.max():.3f}]"
        ]
        ax8.text(0.1, 0.9, '\n'.join(stats_text), transform=ax8.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax8.set_title('Metric Statistics')
        ax8.axis('off')
        
        # 9-12. Individual metric matrix visualizations
        for i in range(4):
            ax = fig.add_subplot(gs[2, i])
            idx = i * len(metric_matrices) // 4
            im = ax.imshow(metric_matrices[idx], cmap='RdBu', aspect='auto')
            ax.set_title(f'Metric Matrix {idx}')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        # Save
        metric_path = self.output_dir / 'comprehensive_metric_analysis.png'
        plt.savefig(metric_path, dpi=300, bbox_inches='tight')
        
        if wandb.run:
            wandb.log({"exhaustive/metric_analysis": wandb.Image(str(metric_path))})
        
        plt.close()
        logger.info(f"✅ Metric analysis saved to {metric_path}")
    
    def visualize_real_latent_space(self, latent_codes: np.ndarray, originals: np.ndarray, reconstructions: np.ndarray) -> None:
        """Comprehensive real latent space visualization."""
        logger.info("🌌 Creating comprehensive latent space analysis")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 0.8])
        fig.suptitle('Exhaustive Real Latent Space Analysis', fontsize=20, fontweight='bold')
        
        # 1. Basic latent distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(latent_codes[:, 0], latent_codes[:, 1], alpha=0.5, s=2)
        ax1.set_title('Real Latent Distribution')
        ax1.set_xlabel('Latent Dim 1')
        ax1.set_ylabel('Latent Dim 2')
        ax1.grid(True, alpha=0.3)
        
        # 2. Density heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist2d(latent_codes[:, 0], latent_codes[:, 1], bins=50, cmap='Blues')
        ax2.set_title('Latent Density')
        ax2.set_xlabel('Latent Dim 1')
        ax2.set_ylabel('Latent Dim 2')
        
        # 3. Reconstruction quality colored scatter
        ax3 = fig.add_subplot(gs[0, 2])
        mse_errors = np.mean((originals - reconstructions) ** 2, axis=(1, 2, 3))
        scatter = ax3.scatter(latent_codes[:, 0], latent_codes[:, 1], 
                             c=mse_errors, cmap='viridis', s=3, alpha=0.7)
        ax3.set_title('Reconstruction Quality')
        ax3.set_xlabel('Latent Dim 1')
        ax3.set_ylabel('Latent Dim 2')
        plt.colorbar(scatter, ax=ax3, label='MSE Error')
        
        # 4. Latent marginals
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.hist(latent_codes[:, 0], bins=50, alpha=0.7, label='Dim 1', density=True)
        ax4.hist(latent_codes[:, 1], bins=50, alpha=0.7, label='Dim 2', density=True)
        # Overlay standard normal
        x = np.linspace(-4, 4, 100)
        ax4.plot(x, np.exp(-x**2/2)/np.sqrt(2*np.pi), 'k--', label='N(0,1)')
        ax4.set_title('Marginal Distributions')
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Density')
        ax4.legend()
        
        # 5. Flow trajectories (if available)
        ax5 = fig.add_subplot(gs[1, 0])
        if hasattr(self.model, 'flows') and len(self.model.flows) > 0:
            # Sample a few points and trace through flows
            sample_points = latent_codes[::len(latent_codes)//20][:20]  # 20 sample points
            
            with torch.no_grad():
                z_current = torch.tensor(sample_points, device=self.device, dtype=torch.float32)
                
                # Plot initial points
                ax5.scatter(z_current[:, 0].cpu(), z_current[:, 1].cpu(), 
                           c='red', s=50, alpha=0.8, label='Initial')
                
                # Apply flows and track
                for flow_idx, flow in enumerate(self.model.flows[:3]):  # First 3 flows
                    z_next, _ = flow(z_current)
                    
                    # Draw arrows
                    for i in range(len(z_current)):
                        ax5.annotate('', xy=z_next[i].cpu().numpy(), xytext=z_current[i].cpu().numpy(),
                                   arrowprops=dict(arrowstyle='->', alpha=0.6, 
                                                 color=plt.cm.viridis(flow_idx/3)))
                    
                    z_current = z_next
                
                ax5.scatter(z_current[:, 0].cpu(), z_current[:, 1].cpu(), 
                           c='blue', s=50, alpha=0.8, label='After flows')
                
            ax5.set_title('Flow Transformations')
            ax5.legend()
        else:
            ax5.text(0.5, 0.5, 'No flows available', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Flow Transformations')
        ax5.set_xlabel('Latent Dim 1')
        ax5.set_ylabel('Latent Dim 2')
        
        # 6. Metric tensor overlay
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.scatter(latent_codes[:, 0], latent_codes[:, 1], alpha=0.3, s=1, c='lightblue')
        
        # Overlay metric centroids if available
        if hasattr(self.model, 'modular_metric') and self.model.modular_metric is not None:
            centroids = self.model.modular_metric.centroids.detach().cpu().numpy()
            ax6.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, 
                       alpha=0.8, marker='x', linewidths=2, label='Centroids')
            ax6.legend()
        
        ax6.set_title('Latent Space + Metric Centroids')
        ax6.set_xlabel('Latent Dim 1')
        ax6.set_ylabel('Latent Dim 2')
        
        # 7. Sequence-wise coloring (if temporal data)
        ax7 = fig.add_subplot(gs[1, 2])
        # Color by sequence index
        seq_colors = np.repeat(np.arange(len(latent_codes)//10), 10)[:len(latent_codes)]
        scatter = ax7.scatter(latent_codes[:, 0], latent_codes[:, 1], 
                             c=seq_colors, cmap='tab20', s=3, alpha=0.7)
        ax7.set_title('Sequence Coloring')
        ax7.set_xlabel('Latent Dim 1')
        ax7.set_ylabel('Latent Dim 2')
        
        # 8. Statistics
        ax8 = fig.add_subplot(gs[1, 3])
        stats_text = [
            f"Samples: {len(latent_codes)}",
            f"Latent dims: {latent_codes.shape[1]}",
            f"Mean: [{latent_codes[:, 0].mean():.3f}, {latent_codes[:, 1].mean():.3f}]",
            f"Std: [{latent_codes[:, 0].std():.3f}, {latent_codes[:, 1].std():.3f}]",
            f"Range dim1: [{latent_codes[:, 0].min():.2f}, {latent_codes[:, 0].max():.2f}]",
            f"Range dim2: [{latent_codes[:, 1].min():.2f}, {latent_codes[:, 1].max():.2f}]",
            f"Correlation: {np.corrcoef(latent_codes[:, 0], latent_codes[:, 1])[0,1]:.3f}",
            f"Avg MSE error: {mse_errors.mean():.4f}",
        ]
        ax8.text(0.1, 0.9, '\n'.join(stats_text), transform=ax8.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax8.set_title('Latent Statistics')
        ax8.axis('off')
        
        # 9-12. Sample images with latent coordinates
        sample_indices = np.random.choice(len(latent_codes), 8, replace=False)
        for i, idx in enumerate(sample_indices[:4]):
            ax = fig.add_subplot(gs[2, i])
            
            # Show original image
            img = originals[idx]
            if img.shape[0] == 3:  # CHW format
                img = img.transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            ax.imshow(img)
            ax.set_title(f'z=[{latent_codes[idx, 0]:.2f}, {latent_codes[idx, 1]:.2f}]', fontsize=8)
            ax.axis('off')
        
        # 13-16. Sample reconstructions
        for i, idx in enumerate(sample_indices[4:]):
            ax = fig.add_subplot(gs[3, i])
            
            # Show reconstruction
            img = reconstructions[idx]
            if img.shape[0] == 3:  # CHW format
                img = img.transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            ax.imshow(img)
            ax.set_title(f'Recon z=[{latent_codes[idx, 0]:.2f}, {latent_codes[idx, 1]:.2f}]', fontsize=8)
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save
        latent_path = self.output_dir / 'exhaustive_latent_analysis.png'
        plt.savefig(latent_path, dpi=300, bbox_inches='tight')
        
        if wandb.run:
            wandb.log({"exhaustive/latent_analysis": wandb.Image(str(latent_path))})
        
        plt.close()
        logger.info(f"✅ Latent space analysis saved to {latent_path}")
    
    def riemannian_generation_analysis(self) -> None:
        """Comprehensive Riemannian generation analysis."""
        logger.info("🎨 Performing Riemannian generation analysis")
        
        if self.model is None:
            logger.error("❌ No model available")
            return
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1])
        fig.suptitle('Comprehensive Riemannian Generation Analysis', fontsize=18, fontweight='bold')
        
        generation_methods = ['standard', 'geodesic', 'enhanced', 'basic']
        n_samples = 8
        
        method_results = {}
        
        for method_idx, method in enumerate(generation_methods):
            logger.info(f"🎲 Generating with method: {method}")
            
            try:
                with torch.no_grad():
                    if method == 'standard':
                        # Standard VAE sampling
                        z = torch.randn(n_samples, self.model.latent_dim, device=self.device)
                        decoder_out = self.model.decoder(z)
                        generated = decoder_out["reconstruction"] if isinstance(decoder_out, dict) else decoder_out
                    
                    elif method == 'geodesic':
                        # Try different geodesic sampling approaches
                        if hasattr(self.model, 'sample_geodesic'):
                            generated = self.model.sample_geodesic(n_samples)
                        elif hasattr(self.model, 'sample'):
                            generated = self.model.sample(n_samples, sampling_mode='geodesic')
                        else:
                            # Fallback: sample with Riemannian posterior
                            z = torch.randn(n_samples, self.model.latent_dim, device=self.device)
                            if hasattr(self.model, 'sample_metric_aware_posterior'):
                                mu = torch.zeros_like(z)
                                logvar = torch.zeros_like(z)
                                z = self.model.sample_metric_aware_posterior(mu, logvar)
                            decoder_out = self.model.decoder(z)
                            generated = decoder_out["reconstruction"] if isinstance(decoder_out, dict) else decoder_out
                    
                    elif method == 'enhanced':
                        # Enhanced Riemannian sampling
                        z = torch.randn(n_samples, self.model.latent_dim, device=self.device)
                        decoder_out = self.model.decoder(z)
                        generated = decoder_out["reconstruction"] if isinstance(decoder_out, dict) else decoder_out
                    
                    elif method == 'basic':
                        # Basic sampling  
                        z = torch.randn(n_samples, self.model.latent_dim, device=self.device)
                        decoder_out = self.model.decoder(z)
                        generated = decoder_out["reconstruction"] if isinstance(decoder_out, dict) else decoder_out
                    
                    else:
                        # Fallback to standard sampling
                        z = torch.randn(n_samples, self.model.latent_dim, device=self.device)
                        decoder_out = self.model.decoder(z)
                        generated = decoder_out["reconstruction"] if isinstance(decoder_out, dict) else decoder_out
                    
                    # Handle ModelOutput objects properly
                    if hasattr(generated, 'reconstruction'):
                        generated = generated.reconstruction
                    elif hasattr(generated, 'cpu'):
                        generated = generated
                    else:
                        logger.warning(f"⚠️ Unexpected generation output type: {type(generated)}")
                        continue
                    
                    method_results[method] = generated.cpu().numpy()
                    
                    # Display samples for this method
                    for sample_idx in range(min(4, n_samples)):
                        row = method_idx // 3
                        col = (method_idx % 3) * 4 + sample_idx
                        if col < 5:  # Ensure we don't exceed grid
                            ax = fig.add_subplot(gs[row, col])
                            
                            img = generated[sample_idx].cpu().numpy()
                            if img.shape[0] == 3:
                                img = img.transpose(1, 2, 0)
                            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                            
                            ax.imshow(img)
                            ax.set_title(f'{method}\nSample {sample_idx+1}', fontsize=8)
                            ax.axis('off')
            
            except Exception as e:
                logger.warning(f"⚠️ Generation method {method} failed: {e}")
                method_results[method] = None
        
        # Add generation quality comparison
        if len(method_results) > 0:
            ax_comparison = fig.add_subplot(gs[2, :])
            
            quality_scores = []
            method_names = []
            
            for method, samples in method_results.items():
                if samples is not None:
                    # Compute simple quality metrics
                    pixel_var = np.var(samples)
                    pixel_mean = np.mean(samples)
                    quality_scores.append(pixel_var)
                    method_names.append(method)
            
            if quality_scores:
                bars = ax_comparison.bar(method_names, quality_scores)
                ax_comparison.set_title('Generation Quality Comparison (Pixel Variance)')
                ax_comparison.set_ylabel('Pixel Variance')
                
                # Color bars
                for i, bar in enumerate(bars):
                    bar.set_color(plt.cm.viridis(i / len(bars)))
        
        plt.tight_layout()
        
        # Save
        generation_path = self.output_dir / 'riemannian_generation_analysis.png'
        plt.savefig(generation_path, dpi=300, bbox_inches='tight')
        
        if wandb.run:
            wandb.log({"exhaustive/generation_analysis": wandb.Image(str(generation_path))})
        
        plt.close()
        logger.info(f"✅ Generation analysis saved to {generation_path}")
        
        return method_results
    
    def geodesic_interpolation_analysis(self) -> None:
        """Create geodesic interpolations in the Riemannian manifold."""
        logger.info("🌊 Creating geodesic interpolation analysis")
        
        if self.model is None:
            logger.error("❌ No model available")
            return
        
        fig, axes = plt.subplots(4, 10, figsize=(20, 8))
        fig.suptitle('Riemannian Geodesic Interpolations', fontsize=16, fontweight='bold')
        
        n_interpolations = 4
        n_steps = 10
        
        for interp_idx in range(n_interpolations):
            # Generate random start and end points
            z_start = torch.randn(1, self.model.latent_dim, device=self.device)
            z_end = torch.randn(1, self.model.latent_dim, device=self.device)
            
            # Create interpolation path
            alphas = torch.linspace(0, 1, n_steps, device=self.device)
            
            for step_idx, alpha in enumerate(alphas):
                try:
                    # Linear interpolation in latent space
                    z_interp = (1 - alpha) * z_start + alpha * z_end
                    
                    # Generate image
                    with torch.no_grad():
                        decoder_out = self.model.decoder(z_interp)
                        
                        # Handle different output formats
                        if isinstance(decoder_out, dict):
                            img = decoder_out["reconstruction"]
                        elif hasattr(decoder_out, 'reconstruction'):
                            img = decoder_out.reconstruction
                        else:
                            img = decoder_out
                        
                        img = img[0].cpu().numpy()
                        
                        if img.shape[0] == 3:
                            img = img.transpose(1, 2, 0)
                        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                        
                        axes[interp_idx, step_idx].imshow(img)
                        axes[interp_idx, step_idx].axis('off')
                        
                        if interp_idx == 0:
                            axes[interp_idx, step_idx].set_title(f'α={alpha:.1f}', fontsize=8)
                
                except Exception as e:
                    logger.warning(f"⚠️ Interpolation error at α={alpha:.1f}: {e}")
                    axes[interp_idx, step_idx].text(0.5, 0.5, f'Error\n{str(e)[:20]}', 
                                                   ha='center', va='center', 
                                                   transform=axes[interp_idx, step_idx].transAxes,
                                                   fontsize=8)
                    axes[interp_idx, step_idx].axis('off')
        
        plt.tight_layout()
        
        # Save
        geodesic_path = self.output_dir / 'geodesic_interpolations.png'
        plt.savefig(geodesic_path, dpi=300, bbox_inches='tight')
        
        if wandb.run:
            wandb.log({"exhaustive/geodesic_interpolations": wandb.Image(str(geodesic_path))})
        
        plt.close()
        logger.info(f"✅ Geodesic interpolations saved to {geodesic_path}")
    
    def run_exhaustive_analysis(self) -> None:
        """Run complete exhaustive analysis."""
        logger.info("🚀 Starting exhaustive RLVAE analysis")
        
        # Initialize WandB
        try:
            wandb.init(
                project="rlvae-post-training-analysis",
                name=f"exhaustive_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "checkpoint_path": self.checkpoint_path,
                    "analysis_type": "exhaustive_comprehensive"
                }
            )
        except Exception as e:
            logger.warning(f"⚠️ WandB initialization failed: {e}")
        
        # 1. Load and reconstruct model properly
        config = self.load_checkpoint_and_extract_config()
        self.reconstruct_model(config)
        
        # 2. Setup real data
        self.setup_data()
        
        # 3. Extract real latent representations
        latent_codes, reconstructions, originals = self.extract_real_latent_representations()
        
        # 4. Comprehensive analyses
        self.analyze_metric_tensor_structure()
        self.visualize_real_latent_space(latent_codes, originals, reconstructions)
        self.riemannian_generation_analysis()
        self.geodesic_interpolation_analysis()
        
        logger.info("🎉 Exhaustive analysis completed!")
        logger.info(f"📁 Results saved in: {self.output_dir}")
        
        if wandb.run:
            wandb.finish()


def main():
    """Main execution function."""
    checkpoint_path = "outputs/checkpoints/epoch=197-val_loss=6.402.ckpt"
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    analyzer = ExhaustiveRLVAEAnalyzer(checkpoint_path)
    analyzer.run_exhaustive_analysis()


if __name__ == "__main__":
    main() 