"""
Reparameterization Sampler implementation.
"""

import torch
from typing import Optional

from ...base import Sampler, register
from ...base.mixins import LoggingMixin, DeviceMixin


@register("reparameterization_sampler")
class ReparameterizationSampler(Sampler, LoggingMixin, DeviceMixin):
    """Reparameterization trick sampler."""
    
    def __init__(self):
        super().__init__()
        LoggingMixin.__init__(self)
    
    def sample(self, n: int, mu: torch.Tensor, log_var: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Sample using reparameterization trick.
        
        Args:
            n: Number of samples
            mu: Mean of shape (B, D)
            log_var: Log variance of shape (B, D)
            **kwargs: Additional arguments
            
        Returns:
            Samples of shape (n, D)
        """
        # Standard reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu)
        samples = mu + std * eps
        
        # Log statistics
        self.log("reparam_samples_mean", samples.mean().item())
        self.log("reparam_samples_std", samples.std().item())
        
        return samples
