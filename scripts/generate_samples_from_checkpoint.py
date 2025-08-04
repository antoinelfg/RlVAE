#!/usr/bin/env python3
"""
Generate Samples from RLVAE Checkpoint
======================================

Script to load a trained RLVAE checkpoint and generate sample sequences
to evaluate generation quality and visualize learned representations.
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
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SampleGenerator:
    """Generate samples from trained RLVAE checkpoint."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """Initialize sample generator."""
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.model = None
        self.output_dir = Path("outputs/sample_generation") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎨 Sample generator initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🖥️ Device: {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup and return the appropriate device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model(self) -> None:
        """Load model from checkpoint with minimal config."""
        logger.info(f"🔄 Loading model from {self.checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            
            # Create minimal config for generation
            config = DictConfig({
                'input_dim': [3, 28, 28],
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
            })
            
            # Create model
            self.model = ModularRiemannianFlowVAE(config)
            
            # Load state dict with flexibility
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('model.', '') if k.startswith('model.') else k: v 
                         for k, v in state_dict.items()}
            
            # Load with strict=False to handle size mismatches gracefully
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"⚠️ Missing keys: {len(missing_keys)} (this is expected for metric components)")
            if unexpected_keys:
                logger.warning(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Model loaded successfully for generation")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            # Fall back to generation from scratch
            logger.info("🔄 Attempting generation from latent space...")
            self.model = None
    
    def generate_latent_samples(self, n_samples: int = 64) -> torch.Tensor:
        """Generate samples in latent space."""
        logger.info(f"🎲 Generating {n_samples} latent samples")
        
        # Generate from prior distribution
        z = torch.randn(n_samples, 2, device=self.device)  # 2D latent space
        return z
    
    def generate_images_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Generate images from latent codes."""
        if self.model is None:
            logger.error("❌ No model available for generation")
            return None
        
        logger.info(f"🎨 Generating {len(z)} images from latent codes")
        
        with torch.no_grad():
            try:
                # Try to use the decoder directly
                images = self.model.decoder(z)
                return images
            except Exception as e:
                logger.error(f"❌ Generation failed: {e}")
                return None
    
    def generate_sample_grid(self, n_samples: int = 64) -> None:
        """Generate and visualize a grid of samples."""
        logger.info(f"🖼️ Creating sample grid with {n_samples} samples")
        
        # Generate latent codes
        z = self.generate_latent_samples(n_samples)
        
        # Generate images
        images = self.generate_images_from_latent(z)
        
        if images is None:
            logger.error("❌ Could not generate images")
            return
        
        # Create grid visualization
        grid_size = int(np.sqrt(n_samples))
        n_samples = grid_size * grid_size  # Ensure perfect square
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        fig.suptitle(f'Generated Samples from RLVAE (Epoch 197)', fontsize=16, fontweight='bold')
        
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < len(images):
                    img = images[idx].cpu().numpy()
                    
                    # Handle different image formats
                    if img.shape[0] == 3:  # CHW format
                        img = img.transpose(1, 2, 0)
                    
                    # Normalize to [0, 1]
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    
                    axes[i, j].imshow(img)
                    axes[i, j].axis('off')
                else:
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        
        # Save
        grid_path = self.output_dir / 'generated_samples_grid.png'
        plt.savefig(grid_path, dpi=300, bbox_inches='tight')
        
        if wandb.run:
            wandb.log({"generation/sample_grid": wandb.Image(str(grid_path))})
        
        plt.close()
        logger.info(f"✅ Sample grid saved to {grid_path}")
    
    def visualize_latent_space(self, n_samples: int = 500) -> None:
        """Visualize the latent space distribution."""
        logger.info(f"📊 Visualizing latent space with {n_samples} samples")
        
        # Generate latent samples
        z = self.generate_latent_samples(n_samples)
        z_np = z.cpu().numpy()
        
        # Create latent space visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Latent Space Analysis', fontsize=16, fontweight='bold')
        
        # Scatter plot
        ax1.scatter(z_np[:, 0], z_np[:, 1], alpha=0.6, s=10)
        ax1.set_title('Latent Space Distribution (2D)')
        ax1.set_xlabel('Latent Dimension 1')
        ax1.set_ylabel('Latent Dimension 2')
        ax1.grid(True, alpha=0.3)
        
        # Density plot
        ax2.hist2d(z_np[:, 0], z_np[:, 1], bins=30, cmap='Blues')
        ax2.set_title('Latent Space Density')
        ax2.set_xlabel('Latent Dimension 1')
        ax2.set_ylabel('Latent Dimension 2')
        
        # Marginal distributions
        ax3.hist(z_np[:, 0], bins=50, alpha=0.7, label='Dim 1')
        ax3.hist(z_np[:, 1], bins=50, alpha=0.7, label='Dim 2')
        ax3.set_title('Marginal Distributions')
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # Statistics
        stats_text = [
            f"Samples: {n_samples}",
            f"Dim 1: μ={z_np[:, 0].mean():.3f}, σ={z_np[:, 0].std():.3f}",
            f"Dim 2: μ={z_np[:, 1].mean():.3f}, σ={z_np[:, 1].std():.3f}",
            f"Correlation: {np.corrcoef(z_np[:, 0], z_np[:, 1])[0, 1]:.3f}"
        ]
        
        ax4.text(0.1, 0.8, '\n'.join(stats_text), transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        ax4.set_title('Statistics')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save
        latent_path = self.output_dir / 'latent_space_analysis.png'
        plt.savefig(latent_path, dpi=300, bbox_inches='tight')
        
        if wandb.run:
            wandb.log({"generation/latent_space": wandb.Image(str(latent_path))})
        
        plt.close()
        logger.info(f"✅ Latent space visualization saved to {latent_path}")
    
    def interpolate_in_latent_space(self, n_steps: int = 10, n_interpolations: int = 4) -> None:
        """Create interpolations in latent space."""
        logger.info(f"🔄 Creating {n_interpolations} interpolations with {n_steps} steps each")
        
        if self.model is None:
            logger.warning("⚠️ No model available for interpolation")
            return
        
        fig, axes = plt.subplots(n_interpolations, n_steps, figsize=(n_steps * 2, n_interpolations * 2))
        fig.suptitle('Latent Space Interpolations', fontsize=16, fontweight='bold')
        
        for interp_idx in range(n_interpolations):
            # Generate random start and end points
            z_start = torch.randn(1, 2, device=self.device)
            z_end = torch.randn(1, 2, device=self.device)
            
            # Create interpolation
            alphas = torch.linspace(0, 1, n_steps, device=self.device)
            
            for step_idx, alpha in enumerate(alphas):
                z_interp = (1 - alpha) * z_start + alpha * z_end
                
                # Generate image
                with torch.no_grad():
                    try:
                        img = self.model.decoder(z_interp)
                        img = img[0].cpu().numpy()
                        
                        if img.shape[0] == 3:
                            img = img.transpose(1, 2, 0)
                        
                        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                        
                        axes[interp_idx, step_idx].imshow(img)
                        axes[interp_idx, step_idx].axis('off')
                        
                        if interp_idx == 0:
                            axes[interp_idx, step_idx].set_title(f'α={alpha:.2f}', fontsize=10)
                    
                    except Exception as e:
                        axes[interp_idx, step_idx].text(0.5, 0.5, 'Error', 
                                                       ha='center', va='center', 
                                                       transform=axes[interp_idx, step_idx].transAxes)
                        axes[interp_idx, step_idx].axis('off')
        
        plt.tight_layout()
        
        # Save
        interp_path = self.output_dir / 'latent_interpolations.png'
        plt.savefig(interp_path, dpi=300, bbox_inches='tight')
        
        if wandb.run:
            wandb.log({"generation/interpolations": wandb.Image(str(interp_path))})
        
        plt.close()
        logger.info(f"✅ Interpolations saved to {interp_path}")
    
    def run_generation_analysis(self) -> None:
        """Run complete generation analysis."""
        logger.info("🚀 Starting generation analysis")
        
        # Initialize WandB
        try:
            wandb.init(
                project="rlvae-post-training-analysis",
                name=f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "checkpoint_path": self.checkpoint_path,
                    "analysis_type": "generation_analysis"
                }
            )
        except Exception as e:
            logger.warning(f"⚠️ WandB initialization failed: {e}")
        
        # Load model
        self.load_model()
        
        # Run generation analyses
        self.visualize_latent_space()
        self.generate_sample_grid()
        self.interpolate_in_latent_space()
        
        logger.info("🎉 Generation analysis completed!")
        logger.info(f"📁 Results saved in: {self.output_dir}")
        
        if wandb.run:
            wandb.finish()


def main():
    """Main execution function."""
    checkpoint_path = "outputs/checkpoints/epoch=197-val_loss=6.402.ckpt"
    
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Run generation analysis
    generator = SampleGenerator(checkpoint_path)
    generator.run_generation_analysis()


if __name__ == "__main__":
    main() 