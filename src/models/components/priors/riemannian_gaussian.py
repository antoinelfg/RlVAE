"""
Riemannian Gaussian Prior implementation.
"""

import torch
import torch.nn as nn
from typing import Optional

from ...base import Prior, register
from ...base.mixins import LoggingMixin, DeviceMixin, NumericalStabilityMixin


@register("riemannian_gaussian_prior")
class RiemannianGaussianPrior(Prior, LoggingMixin, DeviceMixin, NumericalStabilityMixin):
    """Riemannian Gaussian prior p(z) ∝ sqrt(det G(z)) exp(-½ zᵀGz)."""
    
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
            metric: Metric object with G() method
            
        Returns:
            Log probabilities of shape (B,)
        """
        if metric is None:
            # Fallback to standard Gaussian
            return -0.5 * torch.sum(z ** 2, dim=1)
        
        # Compute metric at points
        G_z = metric.G(z)  # (B, D, D)
        
        # Riemannian Gaussian: log p(z) = 0.5 * log det(G(z)) - 0.5 * z^T G(z) z
        log_det_G = self.safe_logdet(G_z)
        quad_form = torch.sum(z.unsqueeze(1) * torch.bmm(G_z, z.unsqueeze(-1)).squeeze(-1), dim=1)
        
        log_prob = 0.5 * log_det_G - 0.5 * quad_form
        
        # Log statistics
        self.log("riemannian_gaussian_log_prob_mean", log_prob.mean().item())
        self.log("riemannian_gaussian_log_prob_std", log_prob.std().item())
        
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
        # Note: Exact sampling from Riemannian Gaussian is complex
        # This is a placeholder implementation using standard normal
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        return torch.randn(n, self.latent_dim, device=device)
