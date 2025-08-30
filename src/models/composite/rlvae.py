"""
Riemannian Latent VAE (RLVAE) composite model.

This model wires together all components for Riemannian VAE training.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from omegaconf import DictConfig, OmegaConf

from ..base import register
from ..base.mixins import LoggingMixin, DeviceMixin
from ..components.encoders import MLPEncoder, CNNEncoder
from ..components.decoders import MLPDecoder, CNNDecoder
from ..components.metric import LearnedMetric, IdentityMetric, FixedMetric
from ..components.posteriors import LocalRiemannianPosterior, EuclideanGaussianPosterior
from ..components.losses import (
    GaussianReconstructionLoss, BernoulliReconstructionLoss,
    KLVolumePriorLoss, KLEuclideanLoss, ELBOLoss
)


@register("rlvae")
class RLVAE(nn.Module, LoggingMixin, DeviceMixin):
    """Riemannian Latent VAE composite model."""
    
    def __init__(
        self,
        input_dim: Tuple[int, ...],
        latent_dim: int,
        encoder: DictConfig,
        decoder: DictConfig,
        metric: DictConfig,
        posterior: DictConfig,
        reconstruction_loss: DictConfig,
        kl_loss: DictConfig,
        elbo_loss: DictConfig,
        **kwargs
    ):
        super().__init__()
        LoggingMixin.__init__(self)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build components
        self.encoder = self._build_component(encoder, input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = self._build_component(decoder, input_dim=input_dim, latent_dim=latent_dim)
        self.metric = self._build_component(metric, latent_dim=latent_dim)
        self.posterior = self._build_component(posterior, latent_dim=latent_dim)
        
        # Build losses in order
        self.reconstruction_loss = self._build_component(reconstruction_loss)
        self.kl_loss = self._build_component(kl_loss, latent_dim=latent_dim)
        
        # Build ELBO loss with the other losses
        if elbo_loss is None:
            # Create default ELBO loss
            self.elbo_loss = ELBOLoss(
                reconstruction_loss=self.reconstruction_loss,
                kl_loss=self.kl_loss,
                flow_loss_weight=1.0,
                loop_penalty_weight=0.0
            )
        elif isinstance(elbo_loss, nn.Module):
            # If it's already built, we need to rebuild it with the correct losses
            elbo_config = OmegaConf.create({
                "_target_": "src.models.components.losses.elbo.ELBOLoss",
                "flow_loss_weight": 1.0,
                "loop_penalty_weight": 0.0
            })
            self.elbo_loss = self._build_component(elbo_config)
            # Replace the losses
            self.elbo_loss.reconstruction_loss = self.reconstruction_loss
            self.elbo_loss.kl_loss = self.kl_loss
        else:
            # Build ELBO loss with the other losses
            elbo_config = dict(elbo_loss)
            elbo_config['reconstruction_loss'] = self.reconstruction_loss
            elbo_config['kl_loss'] = self.kl_loss
            self.elbo_loss = self._build_component(OmegaConf.create(elbo_config))
        
        # Store additional parameters
        self.kwargs = kwargs
    
    def _build_component(self, config, **kwargs) -> nn.Module:
        """Build a component from config or return already built component."""
        # If config is already a module, return it
        if isinstance(config, nn.Module):
            return config
        
        # If config is a DictConfig, build it
        if hasattr(config, '_target_'):
            import hydra
            # Pass kwargs as additional parameters
            return hydra.utils.instantiate(config, **kwargs)
        else:
            # Fallback to registry
            from ..base import build_component
            return build_component(config)
    
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def sample_posterior(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample from posterior."""
        return self.posterior.sample(mu, log_var, metric=self.metric)
    
    def compute_losses(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        z: torch.Tensor,
        log_var: Optional[torch.Tensor] = None,
        flow_log_det: Optional[torch.Tensor] = None,
        loop_penalty: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        return self.elbo_loss(
            x=x,
            x_recon=x_recon,
            mu=mu,
            z=z,
            log_var=log_var,
            flow_log_det=flow_log_det,
            loop_penalty=loop_penalty,
            metric=self.metric,
            **kwargs
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing all outputs and losses
        """
        # Encode
        encoder_output = self.encode(x)
        mu = encoder_output['mu']
        log_var = encoder_output.get('log_var', torch.zeros_like(mu))
        
        # Sample from posterior
        z = self.sample_posterior(mu, log_var)
        
        # Decode
        x_recon = self.decode(z)
        
        # Compute losses
        losses = self.compute_losses(
            x=x,
            x_recon=x_recon,
            mu=mu,
            z=z,
            log_var=log_var,
            **kwargs
        )
        
        # Collect all outputs
        outputs = {
            'reconstruction': x_recon,
            'latent_samples': z,
            'mu': mu,
            'log_var': log_var,
            **losses
        }
        
        # Add component logs
        for component_name, component in [
            ('encoder', self.encoder),
            ('decoder', self.decoder),
            ('metric', self.metric),
            ('posterior', self.posterior)
        ]:
            if hasattr(component, 'get_logs'):
                outputs[f'{component_name}_logs'] = component.get_logs()
        
        return outputs
    
    def get_metric(self, z: torch.Tensor) -> torch.Tensor:
        """Get metric tensor at given points."""
        return self.metric.G(z)
    
    def get_metric_inv(self, z: torch.Tensor) -> torch.Tensor:
        """Get inverse metric tensor at given points."""
        return self.metric.G_inv(z)
