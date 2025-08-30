"""
Standard Gaussian Prior implementation.
"""

import torch
import torch.nn as nn
from typing import Optional

from ...base import Prior, register
from ...base.mixins import LoggingMixin, DeviceMixin


@register("standard_gaussian_prior")
class StandardGaussianPrior(Prior, LoggingMixin, DeviceMixin):
    """Standard Gaussian prior p(z) = N(0, I)."""
    
    def __init__(self, latent_dim: int):
        super().__init__()
        LoggingMixin.__init__(self)
        
        self.latent_dim = latent_dim
    
    def log_prob(self, z: torch.Tensor, metric: Optional[object] = None) -> torch.Tensor:
        """
        Compute log probability of latent points.
        
        Args:
            z: Latent points of shape (B, D)
            metric: Not used for standard Gaussian
            
        Returns:
            Log probabilities of shape (B,)
        """
        # Standard Gaussian log probability
        log_prob = -0.5 * torch.sum(z ** 2, dim=1)
        
        # Log statistics
        self.log("standard_gaussian_log_prob_mean", log_prob.mean().item())
        self.log("standard_gaussian_log_prob_std", log_prob.std().item())
        
        return log_prob
    
    def sample(self, n: int, metric: Optional[object] = None) -> torch.Tensor:
        """
        Sample from standard Gaussian prior.
        
        Args:
            n: Number of samples
            metric: Not used for standard Gaussian
            
        Returns:
            Samples of shape (n, D)
        """
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        return torch.randn(n, self.latent_dim, device=device)
