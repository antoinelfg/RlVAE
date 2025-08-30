"""
Volume Prior implementation.
"""

import torch
import torch.nn as nn
from typing import Optional

from ...base import Prior, register
from ...base.mixins import LoggingMixin, DeviceMixin, NumericalStabilityMixin


@register("volume_prior")
class VolumePrior(Prior, LoggingMixin, DeviceMixin, NumericalStabilityMixin):
    """Volume prior p(z) ∝ sqrt(det G^{-1}(z))."""
    
    def __init__(self, latent_dim: int):
        super().__init__()
        LoggingMixin.__init__(self)
        NumericalStabilityMixin.__init__(self)
        
        self.latent_dim = latent_dim
    
    def log_prob(self, z: torch.Tensor, metric: Optional[object] = None) -> torch.Tensor:
        """
        Compute log probability of latent points.
        
        Args:
            z: Latent points of shape (B, D)
            metric: Metric object with G_inv() method
            
        Returns:
            Log probabilities of shape (B,)
        """
        if metric is None:
            # Fallback to uniform prior
            return torch.zeros(z.shape[0], device=z.device)
        
        # Compute inverse metric at points
        G_inv_z = metric.G_inv(z)  # (B, D, D)
        
        # Volume prior: log p(z) = 0.5 * log det(G^{-1}(z))
        log_det_G_inv = self.safe_logdet(G_inv_z)
        log_prob = 0.5 * log_det_G_inv
        
        # Log statistics
        self.log("volume_prior_log_prob_mean", log_prob.mean().item())
        self.log("volume_prior_log_prob_std", log_prob.std().item())
        
        return log_prob
    
    def sample(self, n: int, metric: Optional[object] = None) -> torch.Tensor:
        """
        Sample from prior (approximate).
        
        Args:
            n: Number of samples
            metric: Not used for sampling
            
        Returns:
            Samples of shape (n, D)
        """
        # Note: Exact sampling from volume prior is complex
        # This is a placeholder implementation using standard normal
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        return torch.randn(n, self.latent_dim, device=device)
