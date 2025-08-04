#!/usr/bin/env python3
"""
Investigate Model Outputs and Understand Training Purpose
========================================================

Script to understand what the trained RLVAE model was actually supposed to do,
based on the PC1/PC2 plots with determinant tracking over time steps.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import logging
from datetime import datetime

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

class ModelOutputInvestigator:
    """Investigate what the model outputs and how it should be analyzed."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """Initialize investigator."""
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.model = None
        self.checkpoint = None
        self.data_module = None
        self.test_loader = None
        
        self.output_dir = Path("outputs/model_investigation") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔍 Model output investigator initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model_quick(self) -> None:
        """Load model using the working approach from before."""
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
        
        logger.info("✅ Model loaded for investigation")
    
    def setup_data_quick(self) -> None:
        """Setup data module quickly."""
        data_config = DictConfig({
            'train_path': 'data/sprites/ColoredCircles_train.pt',
            'test_path': 'data/sprites/ColoredCircles_test.pt',
            'train_meta_path': 'data/sprites/ColoredCircles_train_params.pt',
            'test_meta_path': 'data/sprites/ColoredCircles_test_params.pt',
            'sequence_length': 10,
            'image_size': [28, 28],
            'channels': 3,
            'batch_size': 4,
            'num_workers': 0,
            'pin_memory': False,
            'max_test_samples': 40,
            'verify_cyclicity': False
        })
        
        self.data_module = CyclicSpritesDataModule(data_config)
        self.data_module.setup('test')
        self.test_loader = self.data_module.test_dataloader()
        
        logger.info("✅ Data module ready")
    
    def investigate_model_forward_pass(self) -> None:
        """Investigate what happens during a forward pass."""
        logger.info("🔍 Investigating model forward pass")
        
        # Get a batch of data
        batch = next(iter(self.test_loader))
        sequences = batch  # Shape should be [seq_len, C, H, W] or [batch, seq_len, C, H, W]
        sequences = sequences.to(self.device)
        
        logger.info(f"📊 Input sequences shape: {sequences.shape}")
        
        try:
            with torch.no_grad():
                # Try the full forward pass
                if len(sequences.shape) == 4:  # Single sequence
                    # Add batch dimension
                    sequences = sequences.unsqueeze(0)
                
                logger.info(f"📊 Processed sequences shape: {sequences.shape}")
                
                # Forward pass
                output = self.model(sequences)
                
                logger.info(f"✅ Forward pass successful!")
                logger.info(f"📊 Output type: {type(output)}")
                
                if isinstance(output, dict):
                    logger.info(f"📊 Output keys: {list(output.keys())}")
                    for key, value in output.items():
                        if hasattr(value, 'shape'):
                            logger.info(f"  {key}: {value.shape}")
                        else:
                            logger.info(f"  {key}: {type(value)} = {value}")
                elif hasattr(output, '__dict__'):
                    logger.info(f"📊 Output attributes: {list(output.__dict__.keys())}")
                    for attr in output.__dict__.keys():
                        value = getattr(output, attr)
                        if hasattr(value, 'shape'):
                            logger.info(f"  {attr}: {value.shape}")
                        else:
                            logger.info(f"  {attr}: {type(value)}")
                
                return output
                
        except Exception as e:
            logger.error(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def investigate_individual_components(self) -> None:
        """Investigate encoder, decoder, and metric components individually."""
        logger.info("🔍 Investigating individual model components")
        
        # Get a sample
        batch = next(iter(self.test_loader))
        sequences = batch.to(self.device)
        
        if len(sequences.shape) == 4:
            x_sample = sequences[0]  # First timestep
        else:
            x_sample = sequences[0, 0]  # First batch, first timestep
        
        logger.info(f"📊 Sample input shape: {x_sample.shape}")
        
        try:
            with torch.no_grad():
                # Test encoder
                logger.info("🧠 Testing encoder...")
                encoder_out = self.model.encoder(x_sample.unsqueeze(0))
                logger.info(f"✅ Encoder output type: {type(encoder_out)}")
                
                if hasattr(encoder_out, 'embedding'):
                    logger.info(f"  embedding shape: {encoder_out.embedding.shape}")
                    logger.info(f"  log_covariance shape: {encoder_out.log_covariance.shape}")
                    mu = encoder_out.embedding
                    logvar = encoder_out.log_covariance
                elif isinstance(encoder_out, dict):
                    logger.info(f"  encoder dict keys: {list(encoder_out.keys())}")
                    mu = encoder_out.get('embedding', encoder_out.get('mu'))
                    logvar = encoder_out.get('log_covariance', encoder_out.get('logvar'))
                else:
                    logger.warning(f"  Unexpected encoder output: {encoder_out}")
                    return
                
                # Test sampling
                logger.info("🎲 Testing latent sampling...")
                eps = torch.randn_like(mu)
                z = mu + eps * torch.exp(0.5 * logvar)
                logger.info(f"✅ Latent sample shape: {z.shape}")
                logger.info(f"  Latent range: [{z.min():.3f}, {z.max():.3f}]")
                
                # Test decoder
                logger.info("🎨 Testing decoder...")
                decoder_out = self.model.decoder(z)
                logger.info(f"✅ Decoder output type: {type(decoder_out)}")
                
                if isinstance(decoder_out, dict):
                    logger.info(f"  decoder dict keys: {list(decoder_out.keys())}")
                    if 'reconstruction' in decoder_out:
                        recon = decoder_out['reconstruction']
                        logger.info(f"  reconstruction shape: {recon.shape}")
                        logger.info(f"  reconstruction range: [{recon.min():.3f}, {recon.max():.3f}]")
                elif hasattr(decoder_out, 'reconstruction'):
                    recon = decoder_out.reconstruction
                    logger.info(f"  reconstruction shape: {recon.shape}")
                else:
                    logger.info(f"  decoder output shape: {decoder_out.shape}")
                    recon = decoder_out
                
                # Test metric tensor
                logger.info("📏 Testing metric tensor...")
                if hasattr(self.model, 'modular_metric') and self.model.modular_metric is not None:
                    centroids = self.model.modular_metric.centroids
                    matrices = self.model.modular_metric.metric_matrices
                    logger.info(f"✅ Metric centroids shape: {centroids.shape}")
                    logger.info(f"✅ Metric matrices shape: {matrices.shape}")
                else:
                    logger.warning("⚠️ No metric tensor found")
                
        except Exception as e:
            logger.error(f"❌ Component investigation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def investigate_what_model_should_do(self) -> None:
        """Based on the PC1/PC2 plots, figure out what analysis we should be doing."""
        logger.info("🔍 Investigating what the model was supposed to analyze")
        
        # The user's image shows:
        # - PC1 vs PC2 plots
        # - Log10 det(G^-1) at different time steps t=0,1,2,3
        # - Colored trajectories
        
        logger.info("📊 Based on your image, the model should be doing:")
        logger.info("  1. Principal Component Analysis (PCA) of latent representations")
        logger.info("  2. Computing determinant of inverse metric tensor G^-1")
        logger.info("  3. Tracking this over time steps t=0,1,2,3")
        logger.info("  4. Visualizing trajectories in PC space")
        
        # Let's try to recreate this analysis
        self.analyze_temporal_latent_dynamics()
    
    def analyze_temporal_latent_dynamics(self) -> None:
        """Analyze temporal dynamics in latent space like the PC plots show."""
        logger.info("🌊 Analyzing temporal latent dynamics")
        
        # Collect data from multiple sequences
        all_latents = []
        all_determinants = []
        all_timesteps = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if i >= 5:  # Limit batches
                    break
                
                sequences = batch.to(self.device)
                
                if len(sequences.shape) == 4:  # Single sequence [seq_len, C, H, W]
                    seq_len = sequences.shape[0]
                    for t in range(seq_len):
                        x_t = sequences[t].unsqueeze(0)
                        
                        # Encode
                        encoder_out = self.model.encoder(x_t)
                        mu = encoder_out.embedding
                        logvar = encoder_out.log_covariance
                        
                        # Sample
                        eps = torch.randn_like(mu)
                        z_t = mu + eps * torch.exp(0.5 * logvar)
                        
                        all_latents.append(z_t.cpu().numpy())
                        all_timesteps.append(t)
                        
                        # Try to compute determinant of metric
                        if hasattr(self.model, 'modular_metric') and self.model.modular_metric is not None:
                            # Find closest centroid
                            centroids = self.model.modular_metric.centroids
                            distances = torch.norm(z_t - centroids, dim=1)
                            closest_idx = torch.argmin(distances)
                            
                            # Get corresponding metric matrix
                            metric_matrix = self.model.modular_metric.metric_matrices[closest_idx]
                            det_g = torch.det(metric_matrix)
                            log_det_g_inv = -torch.log10(det_g + 1e-8)  # log10(det(G^-1)) = -log10(det(G))
                            
                            all_determinants.append(log_det_g_inv.cpu().numpy())
                        else:
                            all_determinants.append(0.0)
        
        all_latents = np.array(all_latents).squeeze()
        all_determinants = np.array(all_determinants)
        all_timesteps = np.array(all_timesteps)
        
        logger.info(f"📊 Collected {len(all_latents)} latent points")
        logger.info(f"📊 Latent shape: {all_latents.shape}")
        logger.info(f"📊 Timestep range: [{all_timesteps.min()}, {all_timesteps.max()}]")
        
        # Create the PC analysis plots like in the user's image
        self.create_pc_analysis_plots(all_latents, all_determinants, all_timesteps)
    
    def create_pc_analysis_plots(self, latents, determinants, timesteps):
        """Create PC1/PC2 plots with determinant coloring like the user's image."""
        logger.info("🎨 Creating PC analysis plots")
        
        # Perform PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pc_coords = pca.fit_transform(latents)
        
        logger.info(f"📊 PCA explained variance ratio: {pca.explained_variance_ratio_}")
        
        # Create plots for different timesteps (like t=0,1,2,3 in the image)
        unique_timesteps = np.unique(timesteps)
        n_timesteps = min(4, len(unique_timesteps))
        
        fig, axes = plt.subplots(1, n_timesteps, figsize=(4*n_timesteps, 4))
        if n_timesteps == 1:
            axes = [axes]
        
        fig.suptitle('PC Analysis with Log₁₀ det(G⁻¹) - Recreating Your Model Output', fontsize=14, fontweight='bold')
        
        for i, t in enumerate(unique_timesteps[:n_timesteps]):
            mask = timesteps == t
            
            if np.sum(mask) > 0:
                pc1_t = pc_coords[mask, 0]
                pc2_t = pc_coords[mask, 1]
                det_t = determinants[mask]
                
                scatter = axes[i].scatter(pc1_t, pc2_t, c=det_t, cmap='viridis', 
                                        s=50, alpha=0.7)
                axes[i].set_title(f'Log₁₀ det(G⁻¹) t={t}')
                axes[i].set_xlabel('PC1')
                axes[i].set_ylabel('PC2')
                axes[i].grid(True, alpha=0.3)
                
                # Add colorbar
                plt.colorbar(scatter, ax=axes[i])
        
        plt.tight_layout()
        
        # Save
        pc_path = self.output_dir / 'pc_analysis_recreation.png'
        plt.savefig(pc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ PC analysis plots saved to {pc_path}")
        logger.info("📊 This should match the style of your training output!")
    
    def run_investigation(self) -> None:
        """Run full investigation."""
        logger.info("🚀 Starting model output investigation")
        
        self.load_model_quick()
        self.setup_data_quick()
        
        # Investigate what the model does
        self.investigate_individual_components()
        output = self.investigate_model_forward_pass()
        
        # Try to understand what it should be doing
        self.investigate_what_model_should_do()
        
        logger.info("🎉 Investigation completed!")
        logger.info(f"📁 Results saved in: {self.output_dir}")


def main():
    """Main execution."""
    checkpoint_path = "outputs/checkpoints/epoch=197-val_loss=6.402.ckpt"
    
    investigator = ModelOutputInvestigator(checkpoint_path)
    investigator.run_investigation()


if __name__ == "__main__":
    main() 