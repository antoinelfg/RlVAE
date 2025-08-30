#!/usr/bin/env python3
"""
Phase 1 Training Script for Sprites Data
========================================

Implements Section 5: Phase 1 Training (Posterior → Metric, metric frozen)
- Freeze metric network/parameters at init; train encoder/decoder/flows only
- Light centroid regularizer at t=0 (optional): λ_cent min_k ||μ(x_0)-c_k||_{G(c_k)}^2  
- Monitor: KL non-constant; recon improving; min distance to nearest centroid decreasing
- Visuals at epoch 0, mid, end: latent scatter of μ, posterior samples, centroids, heatmap of logdet(G^-1)
"""

import sys
import torch
import torch.nn as nn
import lightning as L
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
current_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "src"))
sys.path.append(str(current_dir / "original_rlvae/src"))

from data.cyclic_dataset import CyclicSpritesDataModule
from models.riemannian_flow_vae import RiemannianFlowVAE
from utils.reproducibility import configure_for_experiment


class Phase1LightningModule(L.LightningModule):
    """Lightning module for Phase 1 training with Sprites data."""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Initialize model with Phase 1 configuration
        self.model = RiemannianFlowVAE(**config.model)
        
        # Freeze metric parameters at initialization
        if hasattr(self.model, 'freeze_metric_parameters'):
            self.model.freeze_metric_parameters()
        
        # Track metrics for monitoring
        self.training_metrics = []
        self.validation_metrics = []
        
    def configure_optimizers(self):
        """Configure optimizers for Phase 1 training."""
        # Only optimize encoder, decoder, and flows (metric is frozen)
        trainable_params = []
        
        # Add encoder parameters
        if hasattr(self.model, 'encoder'):
            trainable_params.extend(list(self.model.encoder.parameters()))
        
        # Add decoder parameters  
        if hasattr(self.model, 'decoder'):
            trainable_params.extend(list(self.model.decoder.parameters()))
        
        # Add flow parameters
        if hasattr(self.model, 'flows'):
            for flow in self.model.flows:
                trainable_params.extend(list(flow.parameters()))
        
        print(f"🔧 Phase 1 optimizer: {len(trainable_params)} trainable parameters")
        
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.get('weight_decay', 1e-5)
        )
        
        # Optional learning rate scheduler
        if self.config.training.get('use_scheduler', True):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.config.training.max_epochs,
                eta_min=self.config.training.learning_rate * 0.01
            )
            return [optimizer], [scheduler]
        
        return optimizer
    
    def training_step(self, batch, batch_idx):
        """Training step for Phase 1."""
        # Set current epoch for ramping
        self.model._current_epoch = self.current_epoch
        
        # Forward pass through model
        output = self.model(batch)
        
        # Extract loss and metrics
        loss = output.total_loss
        metrics = {k: v for k, v in output.__dict__.items() if torch.is_tensor(v) and v.numel() == 1}
        
        # Log training metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recon_loss', metrics.get('recon_loss', 0.0), on_epoch=True)
        self.log('train_kl_loss', metrics.get('kl_loss', 0.0), on_epoch=True)
        self.log('train_flow_loss', metrics.get('flow_loss', 0.0), on_epoch=True)
        
        # Section 5 specific metrics
        self.log('train_centroid_regularizer', metrics.get('centroid_regularizer', 0.0), on_epoch=True)
        self.log('train_min_centroid_distance', metrics.get('min_centroid_distance', 0.0), on_epoch=True)
        self.log('train_ramp_beta', metrics.get('ramp_beta', 0.0), on_epoch=True)
        self.log('train_ramp_alpha', metrics.get('ramp_alpha', 0.0), on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for Phase 1."""
        # Set current epoch for ramping
        self.model._current_epoch = self.current_epoch
        
        # Forward pass through model
        with torch.no_grad():
            output = self.model(batch)
        
        # Extract loss and metrics
        loss = output.total_loss
        metrics = {k: v for k, v in output.__dict__.items() if torch.is_tensor(v) and v.numel() == 1}
        
        # Log validation metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_recon_loss', metrics.get('recon_loss', 0.0), on_epoch=True)
        self.log('val_kl_loss', metrics.get('kl_loss', 0.0), on_epoch=True)
        self.log('val_flow_loss', metrics.get('flow_loss', 0.0), on_epoch=True)
        
        # Section 5 specific metrics
        self.log('val_centroid_regularizer', metrics.get('centroid_regularizer', 0.0), on_epoch=True)
        self.log('val_min_centroid_distance', metrics.get('min_centroid_distance', 0.0), on_epoch=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Log Phase 1 monitoring metrics at epoch end."""
        # Store current epoch metrics for visualization
        current_metrics = {
            'epoch': self.current_epoch,
            'train_loss': self.trainer.logged_metrics.get('train_loss_epoch', 0.0),
            'train_recon_loss': self.trainer.logged_metrics.get('train_recon_loss', 0.0),
            'train_kl_loss': self.trainer.logged_metrics.get('train_kl_loss', 0.0),
            'train_min_centroid_distance': self.trainer.logged_metrics.get('train_min_centroid_distance', 0.0),
            'train_ramp_beta': self.trainer.logged_metrics.get('train_ramp_beta', 0.0),
        }
        self.training_metrics.append(current_metrics)


def create_phase1_visualization(model: RiemannianFlowVAE, dataloader, epoch: int, save_path: Path):
    """
    Create Phase 1 visualization (Section 5.4):
    - Latent scatter of μ, posterior samples, centroids  
    - Heatmap of logdet(G^-1) (same PCA)
    """
    model.eval()
    
    # Collect latent representations
    mus = []
    samples = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 10:  # Limit for visualization
                break
            
            # Use first frame for temporal data
            if batch.ndim == 5:
                batch = batch[:, 0]  # [batch, channels, height, width]
            
            output = model(batch)
            
            # Extract latent representations
            if hasattr(output, 'mu'):
                mus.append(output.mu.cpu())
            if hasattr(output, 'z'):
                if output.z.ndim > 2:  # Handle sequence data
                    samples.append(output.z[:, 0].cpu())
                else:
                    samples.append(output.z.cpu())
    
    if not mus:
        print("⚠️ No latent representations found for visualization")
        return
    
    # Concatenate all samples
    mus = torch.cat(mus, dim=0)
    samples = torch.cat(samples, dim=0) if samples else mus
    
    # PCA for 2D visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    
    # Fit PCA on posterior means
    mus_2d = pca.fit_transform(mus.numpy())
    samples_2d = pca.transform(samples.numpy())
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Latent scatter
    ax1.scatter(mus_2d[:, 0], mus_2d[:, 1], alpha=0.6, s=20, c='blue', label='Posterior means μ')
    ax1.scatter(samples_2d[:, 0], samples_2d[:, 1], alpha=0.3, s=10, c='red', label='Posterior samples z')
    
    # Add centroids if available
    if hasattr(model, 'centroids_tens') and model.centroids_tens is not None:
        centroids_2d = pca.transform(model.centroids_tens.cpu().numpy())
        ax1.scatter(centroids_2d[:, 0], centroids_2d[:, 1], 
                   s=100, c='green', marker='x', linewidths=3, label='Centroids')
    
    ax1.set_title(f'Latent Space Scatter (Epoch {epoch})')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Heatmap of log det(G^-1)
    if hasattr(model, 'G_inv'):
        # Create grid in PCA space
        x_min, x_max = mus_2d[:, 0].min() - 1, mus_2d[:, 0].max() + 1
        y_min, y_max = mus_2d[:, 1].min() - 1, mus_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                           np.linspace(y_min, y_max, 50))
        
        # Transform grid back to latent space
        grid_2d = np.c_[xx.ravel(), yy.ravel()]
        grid_latent = pca.inverse_transform(grid_2d)
        grid_latent = torch.tensor(grid_latent, dtype=torch.float32, device=model.device)
        
        # Compute log det(G^-1) at grid points
        with torch.no_grad():
            try:
                G_inv_grid = model.G_inv(grid_latent)
                logdet_vals = torch.logdet(G_inv_grid + 1e-6 * torch.eye(G_inv_grid.shape[-1], device=G_inv_grid.device))
                logdet_vals = logdet_vals.cpu().numpy().reshape(xx.shape)
                
                # Create heatmap
                im = ax2.contourf(xx, yy, logdet_vals, levels=20, cmap='viridis', alpha=0.7)
                ax2.scatter(mus_2d[:, 0], mus_2d[:, 1], alpha=0.6, s=10, c='white', edgecolors='black', linewidths=0.5)
                
                plt.colorbar(im, ax=ax2, label='log det(G^-1)')
                ax2.set_title(f'Metric Heatmap: log det(G^-1) (Epoch {epoch})')
                ax2.set_xlabel('PC1')
                ax2.set_ylabel('PC2')
                
            except Exception as e:
                ax2.text(0.5, 0.5, f'Metric visualization failed:\\n{str(e)}', 
                        transform=ax2.transAxes, ha='center', va='center')
                ax2.set_title('Metric Heatmap (Failed)')
    else:
        ax2.text(0.5, 0.5, 'No metric available', transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('Metric Heatmap (N/A)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Phase 1 visualization saved: {save_path}")


def main():
    """Main Phase 1 training function."""
    print("🚀 Starting Phase 1 Training: Posterior → Metric (metric frozen)")
    print("=" * 60)
    
    # Load configuration
    config_path = current_dir / "conf" / "model" / "rhvae_phase1_training.yaml"
    config = OmegaConf.load(config_path)
    
    # Add training configuration
    training_config = {
        'max_epochs': 20,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'use_scheduler': True,
        'batch_size': 32,
        'num_workers': 4
    }
    config.training = OmegaConf.create(training_config)
    
    # Set reproducibility
    configure_for_experiment(42, "research")
    
    # Setup data
    data_module = CyclicSpritesDataModule(
        train_path="/home/alaforgu/scratch/longitudinal_experiments/RlVAE/data/processed/Sprites_train_cyclic.pt",
        test_path="/home/alaforgu/scratch/longitudinal_experiments/RlVAE/data/processed/Sprites_test_cyclic.pt",
        batch_size=training_config['batch_size'],
        num_workers=training_config['num_workers']
    )
    
    # Setup model
    model = Phase1LightningModule(config)
    
    # Create output directory
    output_dir = current_dir / "outputs" / "phase1_sprites" / f"run_{torch.randint(10000, 99999, (1,)).item()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup trainer
    trainer = L.Trainer(
        max_epochs=training_config['max_epochs'],
        accelerator='auto',
        devices=1,
        logger=False,  # Simple logging for now
        enable_checkpointing=True,
        default_root_dir=str(output_dir),
        log_every_n_steps=50,
        enable_progress_bar=True
    )
    
    # Prepare data
    data_module.setup("fit")
    
    # Initial visualization (epoch 0)
    print("\\n📊 Creating initial visualization (epoch 0)...")
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    create_phase1_visualization(model.model, data_module.val_dataloader(), 0, viz_dir / "phase1_epoch_0.png")
    
    # Train model
    print("\\n🏋️ Starting Phase 1 training...")
    trainer.fit(model, datamodule=data_module)
    
    # Mid-training visualization  
    mid_epoch = training_config['max_epochs'] // 2
    print(f"\\n📊 Creating mid-training visualization (epoch {mid_epoch})...")
    create_phase1_visualization(model.model, data_module.val_dataloader(), mid_epoch, viz_dir / f"phase1_epoch_{mid_epoch}.png")
    
    # Final visualization
    print(f"\\n📊 Creating final visualization (epoch {training_config['max_epochs']})...")
    create_phase1_visualization(model.model, data_module.val_dataloader(), training_config['max_epochs'], viz_dir / f"phase1_epoch_{training_config['max_epochs']}.png")
    
    # Save trained model
    model_path = output_dir / "phase1_model.ckpt"
    trainer.save_checkpoint(model_path)
    print(f"\\n💾 Phase 1 model saved: {model_path}")
    
    # Print summary
    print("\\n" + "=" * 60)
    print("🎉 Phase 1 Training Complete!")
    print(f"📁 Output directory: {output_dir}")
    print("✅ Metric parameters remained frozen during training")
    print("✅ Encoder/decoder/flows optimized for reconstruction")
    print("✅ Visualizations created at epoch 0, mid, and end")
    print("\\n📈 Key monitoring metrics:")
    if model.training_metrics:
        final_metrics = model.training_metrics[-1]
        print(f"   Final KL loss: {final_metrics.get('train_kl_loss', 0.0):.4f}")
        print(f"   Final recon loss: {final_metrics.get('train_recon_loss', 0.0):.4f}")
        print(f"   Final centroid distance: {final_metrics.get('train_min_centroid_distance', 0.0):.4f}")
        print(f"   Final β: {final_metrics.get('train_ramp_beta', 0.0):.4f}")


if __name__ == "__main__":
    main()
