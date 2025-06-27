"""
Lightning Trainer Module for RlVAE
==================================

PyTorch Lightning wrapper for ModularRiemannianFlowVAE with integrated visualizations.
"""

import torch
import torch.nn as nn
import lightning as L
from typing import Dict, Any, Optional
from omegaconf import DictConfig
import wandb

from models.modular_rlvae import ModularRiemannianFlowVAE
from visualizations.manager import VisualizationManager, VisualizationLevel, VisualizationConfig


class LightningRlVAETrainer(L.LightningModule):
    """Lightning module for RiemannianFlowVAE training."""
    
    def __init__(self, config: DictConfig, data_module=None):
        super().__init__()
        
        self.config = config
        self.data_module = data_module
        
        # Create model
        self.model = ModularRiemannianFlowVAE(config.model)
        
        # Setup visualizations
        self._setup_visualizations()
        
        # Save hyperparameters
        self.save_hyperparameters(config)
        
        print(f"⚡ Lightning trainer initialized")
        print(f"   Model: {self.model.model_name}")
        print(f"   Visualization level: {self.config.visualization.level}")
    
    def _setup_visualizations(self):
        """Setup visualization manager."""
        try:
            # Create visualization config - handle level as string
            level_str = self.config.visualization.level
            if isinstance(level_str, str):
                level_enum = VisualizationLevel(level_str)
            else:
                level_enum = level_str
                
            viz_config = VisualizationConfig.from_level(level_enum)
            
            # Override with specific config values
            for key, value in self.config.visualization.items():
                if hasattr(viz_config, key) and key != 'level':
                    setattr(viz_config, key, value)
            
            self.viz_manager = VisualizationManager(
                model=self.model,
                config=self.config,
                device=self.device,
                viz_config=viz_config
            )
            
            self.enable_visualizations = True
            print(f"🎨 Visualizations enabled: {viz_config.level.value}")
            
        except Exception as e:
            print(f"⚠️ Visualization setup failed: {e}")
            self.viz_manager = None
            self.enable_visualizations = False
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x, compute_metrics=True)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x = batch  # [B, T, C, H, W]
        
        # Forward pass
        output = self(x)
        
        # Extract losses
        total_loss = output['total_loss']
        recon_loss = output['reconstruction_loss']
        kl_loss = output['kl_divergence']
        
        # Log losses
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_recon_loss', recon_loss)
        self.log('train_kl_loss', kl_loss)
        
        # Log additional metrics if available
        if 'riemannian_kl' in output:
            self.log('train_riemannian_kl', output['riemannian_kl'])
        
        if 'cyclicity_error' in output:
            self.log('train_cyclicity_error', output['cyclicity_error'])
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x = batch
        
        # Forward pass
        output = self(x)
        
        # Extract losses
        total_loss = output['total_loss']
        recon_loss = output['reconstruction_loss']
        kl_loss = output['kl_divergence']
        
        # Log losses
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_recon_loss', recon_loss)
        self.log('val_kl_loss', kl_loss)
        
        # Log additional metrics
        if 'riemannian_kl' in output:
            self.log('val_riemannian_kl', output['riemannian_kl'])
        
        if 'cyclicity_error' in output:
            self.log('val_cyclicity_error', output['cyclicity_error'])
        
        # Store for visualization
        if batch_idx == 0:  # Only store first batch for efficiency
            self.validation_batch = x.detach().cpu()
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        x = batch
        
        # Forward pass
        output = self(x)
        
        # Extract losses
        total_loss = output['total_loss']
        recon_loss = output['reconstruction_loss']
        kl_loss = output['kl_divergence']
        
        # Create comprehensive test metrics
        metrics = {
            'test_loss': total_loss,
            'test_recon_loss': recon_loss,
            'test_kl_loss': kl_loss,
        }
        
        # Add additional metrics
        if 'riemannian_kl' in output:
            metrics['test_riemannian_kl'] = output['riemannian_kl']
        
        if 'cyclicity_error' in output:
            metrics['test_cyclicity_error'] = output['cyclicity_error']
        
        # Add Riemannian-specific metrics
        if 'metric_conditioning' in output:
            metrics['test_metric_conditioning'] = output['metric_conditioning']
        
        if 'manifold_regularity' in output:
            metrics['test_manifold_regularity'] = output['manifold_regularity']
        
        # Log all metrics
        for key, value in metrics.items():
            self.log(key, value)
        
        return metrics
    
    def on_validation_epoch_end(self):
        """Create visualizations at end of validation epoch."""
        if not self.enable_visualizations or self.viz_manager is None:
            return
        
        # Only create visualizations at specified frequency
        if self.current_epoch % self.config.visualization.frequency != 0:
            return
        
        try:
            # Get sample data
            if hasattr(self, 'validation_batch'):
                x_sample = self.validation_batch.to(self.device)
            elif self.data_module and hasattr(self.data_module, 'get_sample_batch'):
                x_sample = self.data_module.get_sample_batch('val').to(self.device)
            else:
                print("⚠️ No sample data available for visualization")
                return
            
            # Create visualizations
            print(f"🎨 Creating visualizations for epoch {self.current_epoch}")
            self.viz_manager.create_visualizations(
                x_sample=x_sample,
                epoch=self.current_epoch
            )
            
        except Exception as e:
            print(f"⚠️ Visualization error at epoch {self.current_epoch}: {e}")
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        
        # Create optimizer using new config structure
        optimizer_config = self.config.training.optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
        )
        
        # Create scheduler if specified
        if hasattr(self.config.training, 'scheduler'):
            scheduler_config = self.config.training.scheduler
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.mode,
                factor=scheduler_config.factor,
                patience=scheduler_config.patience,
                min_lr=scheduler_config.min_lr
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.config.training.logging.monitor,
                    "frequency": 1
                }
            }
        
        return optimizer
    
    def setup(self, stage=None):
        """Setup method called by Lightning."""
        super().setup(stage)
        
        # Ensure the entire model is properly on device
        self._ensure_model_on_device()
        
        # Update visualization manager device
        if self.viz_manager is not None:
            self.viz_manager.device = self.device
            # Update device for all visualization modules
            for viz_name, viz_module in self.viz_manager.modules.items():
                viz_module.device = self.device
    
    def _ensure_model_on_device(self):
        """Ensure all model components are on the correct device."""
        device = self.device
        
        # Move main model
        self.model = self.model.to(device)
        
        # Ensure encoder/decoder are on device
        if hasattr(self.model, 'encoder') and self.model.encoder is not None:
            self.model.encoder = self.model.encoder.to(device)
        if hasattr(self.model, 'decoder') and self.model.decoder is not None:
            self.model.decoder = self.model.decoder.to(device)
        
        # Ensure metric components are on device
        for attr_name in ['G', 'G_inv', 'centroids', 'flows']:
            if hasattr(self.model, attr_name):
                attr_value = getattr(self.model, attr_name)
                if attr_value is not None:
                    if hasattr(attr_value, 'to'):
                        setattr(self.model, attr_name, attr_value.to(device))
                    elif isinstance(attr_value, (list, nn.ModuleList)):
                        for i, item in enumerate(attr_value):
                            if hasattr(item, 'to'):
                                attr_value[i] = item.to(device)
        
        print(f"✅ Ensured model is on device: {device}")
    
    def on_train_start(self):
        """Log model summary at start of training."""
        # Ensure model is on device again at train start
        self._ensure_model_on_device()
        
        if wandb.run is not None:
            try:
                summary = self.model.get_model_summary()
                
                # Convert ListConfig objects to regular lists for JSON serialization
                def convert_config_to_dict(obj):
                    if hasattr(obj, '_content'):
                        # This is a DictConfig or ListConfig
                        if isinstance(obj._content, dict):
                            return {k: convert_config_to_dict(v) for k, v in obj._content.items()}
                        elif isinstance(obj._content, list):
                            return [convert_config_to_dict(v) for v in obj._content]
                        else:
                            return obj._content
                    elif isinstance(obj, dict):
                        return {k: convert_config_to_dict(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_config_to_dict(v) for v in obj]
                    elif hasattr(obj, '__dict__'):
                        # Handle objects with __dict__
                        return str(obj)
                    else:
                        return obj
                
                summary_serializable = convert_config_to_dict(summary)
                wandb.log({"model_summary": summary_serializable})
                
                # Log data statistics if available
                if self.data_module and hasattr(self.data_module, 'get_data_stats'):
                    data_stats = self.data_module.get_data_stats()
                    data_stats_serializable = convert_config_to_dict(data_stats)
                    wandb.log({"data_stats": data_stats_serializable})
                    
            except Exception as e:
                print(f"⚠️ Failed to log model summary to wandb: {e}")
        
        print(f"🚀 Starting training for {self.config.training.trainer.max_epochs} epochs")
        print(f"   Model parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"   Visualization frequency: every {self.config.visualization.frequency} epochs") 