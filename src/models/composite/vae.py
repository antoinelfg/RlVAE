"""
Vanilla VAE composite model.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from omegaconf import DictConfig

from ..base import register
from ..base.mixins import LoggingMixin, DeviceMixin
from ..components.encoders import MLPEncoder, CNNEncoder
from ..components.decoders import MLPDecoder, CNNDecoder
from ..components.posteriors import EuclideanGaussianPosterior
from ..components.losses import (
    GaussianReconstructionLoss, BernoulliReconstructionLoss,
    KLEuclideanLoss, ELBOLoss
)


@register("vae")
class VAE(nn.Module, LoggingMixin, DeviceMixin):
    """Vanilla VAE composite model."""
    
    def __init__(
        self,
        input_dim: Tuple[int, ...],
        latent_dim: int,
        encoder: DictConfig,
        decoder: DictConfig,
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
        self.posterior = self._build_component(posterior, latent_dim=latent_dim)
        
        # Build losses
        self.reconstruction_loss = self._build_component(reconstruction_loss)
        self.kl_loss = self._build_component(kl_loss, latent_dim=latent_dim)
        self.elbo_loss = self._build_component(
            elbo_loss,
            reconstruction_loss=self.reconstruction_loss,
            kl_loss=self.kl_loss
        )
        
        # Store additional parameters
        self.kwargs = kwargs
    
    def _build_component(self, config: DictConfig, **kwargs) -> nn.Module:
        """Build a component from config."""
        # Merge kwargs with config
        config_dict = dict(config)
        config_dict.update(kwargs)
        
        # Create component
        if hasattr(config, '_target_'):
            import hydra
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
        return self.posterior.sample(mu, log_var)
    
    def compute_losses(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        z: torch.Tensor,
        log_var: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        return self.elbo_loss(
            x=x,
            x_recon=x_recon,
            mu=mu,
            z=z,
            log_var=log_var,
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
            ('posterior', self.posterior)
        ]:
            if hasattr(component, 'get_logs'):
                outputs[f'{component_name}_logs'] = component.get_logs()
        
        return outputs
