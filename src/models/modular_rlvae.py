"""
Modular Riemannian Flow VAE
==========================

Enhanced version of RiemannianFlowVAE with:
- Hydra configuration support
- Modular architecture for easy comparisons
- Better experiment tracking and analysis
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path
from omegaconf import DictConfig
import wandb

from .riemannian_flow_vae import RiemannianFlowVAE
from pythae.models.base.base_utils import ModelOutput


class ModularRiemannianFlowVAE(RiemannianFlowVAE):
    """
    Modular version of RiemannianFlowVAE with Hydra support.
    
    Key improvements:
    - Configuration-driven initialization
    - Standardized metrics tracking
    - Comparison-friendly interface
    - Enhanced experiment logging
    """
    
    def __init__(self, config: DictConfig):
        """Initialize from Hydra configuration."""
        
        # Extract core parameters
        super().__init__(
            input_dim=tuple(config.input_dim),
            latent_dim=config.latent_dim,
            n_flows=config.n_flows,
            flow_hidden_size=config.flow_hidden_size,
            flow_n_blocks=config.flow_n_blocks,
            flow_n_hidden=config.flow_n_hidden,
            epsilon=config.epsilon,
            beta=config.beta,
            riemannian_beta=config.get('riemannian_beta', config.beta),
            posterior_type=config.posterior.type,
            loop_mode=config.loop.mode
        )
        
        # Store config for later use
        self.config = config
        self.model_name = config.get('_target_', 'ModularRiemannianFlowVAE').split('.')[-1]
        
        # Setup components based on config
        self._setup_from_config()
        
        # Initialize metrics tracking
        self._setup_metrics_tracking()
        
    def _setup_from_config(self):
        """Setup model components from configuration."""
        
        # Configure loop mode
        if hasattr(self, 'set_loop_mode'):
            self.set_loop_mode(
                self.config.loop.mode, 
                self.config.loop.penalty
            )
        
        # Load pretrained components if specified
        if self.config.pretrained.encoder_path:
            self.load_pretrained_components(
                encoder_path=self.config.pretrained.encoder_path,
                decoder_path=self.config.pretrained.decoder_path,
                metric_path=self.config.pretrained.metric_path,
                temperature_override=self.config.metric.get('temperature_override')
            )
        
        # Configure Riemannian sampling
        if self.config.sampling.use_riemannian:
            method = "custom" if self.config.sampling.method in ['geodesic', 'enhanced', 'basic'] else self.config.sampling.method
            self.enable_pure_rhvae(enable=True, method=method)
            self._riemannian_method = self.config.sampling.method
        else:
            self.enable_pure_rhvae(enable=False)
    
    def _setup_metrics_tracking(self):
        """Initialize comprehensive metrics tracking."""
        self.metrics_history = {
            'reconstruction_loss': [],
            'kl_divergence': [],
            'cyclicity_error': [],
            'riemannian_kl': [],
            'total_loss': []
        }
        
        # Model-specific metrics
        if self.config.sampling.use_riemannian:
            self.metrics_history.update({
                'geodesic_preservation': [],
                'metric_conditioning': [],
                'manifold_regularity': []
            })
    
    def forward(self, x: torch.Tensor, compute_metrics: bool = False) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with standardized output format.
        
        Args:
            x: Input tensor [B, T, C, H, W]
            compute_metrics: Whether to compute additional metrics
            
        Returns:
            Dictionary with standardized keys for easy comparison
        """
        output = super().forward(x)

        result = {
            'reconstruction': output.recon_x,
            'latent_samples': output.z,
            'reconstruction_loss': output.recon_loss,
            'kl_divergence': output.kld_loss,
            'total_loss': output.loss
        }

        if hasattr(output, 'riemannian_kl'):
            result['riemannian_kl'] = output.riemannian_kl

        if compute_metrics:
            result.update(self._compute_additional_metrics(x, result))

        return result
    
    def _compute_additional_metrics(self, x: torch.Tensor, output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute additional metrics for analysis."""
        metrics = {}
        
        # Cyclicity error (for temporal sequences)
        if len(x.shape) == 5:  # [B, T, C, H, W]
            with torch.no_grad():
                recon = output['reconstruction']
                cyclicity_error = torch.mean((recon[:, 0] - recon[:, -1]) ** 2)
                metrics['cyclicity_error'] = cyclicity_error
        
        # Latent space metrics
        z = output['latent_samples']
        metrics['latent_norm'] = torch.mean(torch.norm(z, dim=-1))
        metrics['latent_variance'] = torch.var(z)
        
        # Riemannian-specific metrics
        if self.config.sampling.use_riemannian and hasattr(self, 'G_inv'):
            metrics.update(self._compute_riemannian_metrics(z))
        
        return metrics
    
    def _compute_riemannian_metrics(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute Riemannian geometry metrics."""
        with torch.no_grad():
            try:
                # Metric conditioning
                G_inv = self.G_inv(z)
                eigenvals = torch.linalg.eigvals(G_inv)
                condition_number = torch.mean(torch.max(eigenvals.real, dim=-1)[0] / torch.min(eigenvals.real, dim=-1)[0])
                
                # Manifold regularity (determinant stability)
                det_G_inv = torch.det(G_inv)
                regularity = torch.std(torch.log(det_G_inv + 1e-8))
                
                return {
                    'metric_conditioning': condition_number,
                    'manifold_regularity': regularity,
                    'metric_determinant': torch.mean(det_G_inv)
                }
            except:
                return {
                    'metric_conditioning': torch.tensor(0.0),
                    'manifold_regularity': torch.tensor(0.0),
                    'metric_determinant': torch.tensor(1.0)
                }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary for comparison."""
        summary = {
            'model_name': self.model_name,
            'architecture': {
                'latent_dim': self.latent_dim,
                'n_flows': self.config.n_flows,
                'input_dim': self.config.input_dim
            },
            'configuration': {
                'posterior_type': self.config.posterior.type,
                'sampling_method': self.config.sampling.method,
                'loop_mode': self.config.loop.mode,
                'uses_riemannian': self.config.sampling.use_riemannian
            },
            'hyperparameters': {
                'beta': self.config.beta,
                'riemannian_beta': self.config.get('riemannian_beta', 0.0),
                'cycle_penalty': self.config.loop.penalty
            }
        }
        
        # Add parameter count
        summary['parameters'] = {
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        
        return summary
    
    def log_to_wandb(self, metrics: Dict[str, torch.Tensor], epoch: int, prefix: str = ""):
        """Log metrics to wandb with standardized naming."""
        if wandb.run is None:
            return
        
        log_dict = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.detach().cpu().numpy()
            log_dict[f"{prefix}{key}"] = value
        
        log_dict['epoch'] = epoch
        wandb.log(log_dict)
    
    @classmethod
    def create_comparison_models(cls, base_config: DictConfig, model_variants: List[str]) -> Dict[str, 'ModularRiemannianFlowVAE']:
        """
        Create multiple model variants for comparison.
        
        Args:
            base_config: Base configuration
            model_variants: List of model configuration names
            
        Returns:
            Dictionary of model_name -> model instances
        """
        models = {}
        
        for variant in model_variants:
            # Load variant config (this would be implemented with Hydra compose API)
            variant_config = base_config.copy()  # Simplified for now
            
            # Create model with variant config
            models[variant] = cls(variant_config)
        
        return models


class ModelFactory:
    """Factory for creating models from configurations."""
    
    @staticmethod
    def create_model(config: DictConfig) -> ModularRiemannianFlowVAE:
        """Create model from configuration."""
        return ModularRiemannianFlowVAE(config)
    
    @staticmethod
    def create_comparison_suite(config: DictConfig) -> Dict[str, ModularRiemannianFlowVAE]:
        """Create a suite of models for comparison."""
        if hasattr(config.experiment, 'models'):
            models = {}
            for model_name in config.experiment.models:
                # Create config for this model variant
                model_config = config.model.copy()
                
                # Apply model-specific overrides
                if model_name == 'vanilla_vae':
                    model_config.n_flows = 0
                    model_config.riemannian_beta = 0.0
                    model_config.posterior.type = 'gaussian'
                    model_config.sampling.use_riemannian = False
                    model_config.sampling.method = 'standard'
                    model_config.loop.mode = 'open'
                    model_config.loop.penalty = 0.0
                
                models[model_name] = ModularRiemannianFlowVAE(model_config)
            
            return models
        else:
            return {'main': ModularRiemannianFlowVAE(config.model)}


class MetricsCollector:
    """Collect and analyze metrics across different models."""
    
    def __init__(self):
        self.model_metrics = {}
    
    def add_model_metrics(self, model_name: str, metrics: Dict[str, float]):
        """Add metrics for a specific model."""
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = []
        self.model_metrics[model_name].append(metrics)
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """Get summary comparing all models."""
        summary = {}
        
        for model_name, metrics_list in self.model_metrics.items():
            if not metrics_list:
                continue
                
            # Aggregate metrics
            aggregated = {}
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list if key in m]
                if values:
                    aggregated[f"{key}_mean"] = sum(values) / len(values)
                    aggregated[f"{key}_std"] = (sum((x - aggregated[f"{key}_mean"])**2 for x in values) / len(values))**0.5
                    aggregated[f"{key}_final"] = values[-1]
            
            summary[model_name] = aggregated
        
        return summary
    
    def log_comparison_to_wandb(self):
        """Log comparison results to wandb."""
        if wandb.run is None:
            return
        
        summary = self.get_comparison_summary()
        
        # Create comparison table
        table_data = []
        for model_name, metrics in summary.items():
            row = {'model': model_name}
            row.update(metrics)
            table_data.append(row)
        
        wandb.log({"model_comparison": wandb.Table(data=table_data, columns=list(table_data[0].keys()))}) 