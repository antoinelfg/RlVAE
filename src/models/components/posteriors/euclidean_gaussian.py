"""
Euclidean Gaussian Posterior implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from ...base import Posterior, register
from ...base.mixins import LoggingMixin, DeviceMixin


@register("euclidean_gaussian_posterior")
class EuclideanGaussianPosterior(Posterior, LoggingMixin, DeviceMixin):
    """Standard Euclidean Gaussian posterior."""
    
    def __init__(self, latent_dim: int):
        super().__init__()
        LoggingMixin.__init__(self)
        
        self.latent_dim = latent_dim
    
    def sample(self, mu: torch.Tensor, log_var: torch.Tensor, metric=None) -> torch.Tensor:
        """
        Sample from Euclidean Gaussian posterior.
        
        Args:
            mu: Mean of shape (B, D)
            log_var: Log variance of shape (B, D)
            metric: Not used for Euclidean posterior
            
        Returns:
            Samples of shape (B, D)
        """
        # Standard reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu)
        z = mu + std * eps
        
        # Log statistics
        self.log("posterior_samples_mean", z.mean().item())
        self.log("posterior_samples_std", z.std().item())
        
        return z
    
    def log_prob(self, z: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, metric=None) -> torch.Tensor:
        """
        Compute log probability of samples.
        
        Args:
            z: Samples of shape (B, D)
            mu: Mean of shape (B, D)
            log_var: Log variance of shape (B, D)
            metric: Not used for Euclidean posterior
            
        Returns:
            Log probabilities of shape (B,)
        """
        # Standard Gaussian log probability
        std = torch.exp(0.5 * log_var)
        log_prob = -0.5 * torch.sum(((z - mu) / std) ** 2 + 2 * log_var, dim=1)
        
        return log_prob
    
    def extra_logs(self) -> Dict[str, Any]:
        """Return extra logging information."""
        return {
            "posterior_type": "euclidean_gaussian"
        }
