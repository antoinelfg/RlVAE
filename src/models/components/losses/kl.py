"""
KL divergence loss implementations.
"""

import torch
import torch.nn as nn
from typing import Optional

from ...base import KLLoss, register
from ...base.mixins import LoggingMixin, DeviceMixin, NumericalStabilityMixin


@register("kl_volume_prior")
class KLVolumePriorLoss(KLLoss, LoggingMixin, DeviceMixin, NumericalStabilityMixin):
    """KL divergence with volume prior."""
    
    def __init__(
        self,
        latent_dim: int,
        beta: float = 1.0,
        amp_safe: bool = True
    ):
        super().__init__()
        LoggingMixin.__init__(self)
        NumericalStabilityMixin.__init__(self)
        
        self.latent_dim = latent_dim
        self.beta = beta
        self.amp_safe = amp_safe
    
    def forward(
        self, 
        mu: torch.Tensor, 
        z: torch.Tensor, 
        metric: Optional[object] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute KL divergence with volume prior.
        
        Args:
            mu: Posterior mean of shape (B, D)
            z: Sampled latents of shape (B, D)
            metric: Metric object with G() and G_inv() methods
            **kwargs: Additional arguments
            
        Returns:
            KL loss scalar
        """
        if metric is None:
            # Fallback to Euclidean KL
            return self._euclidean_kl(mu, z)
        
        # Compute metric at sampled points
        G_z = metric.G(z)  # (B, D, D)
        G_inv_z = metric.G_inv(z)  # (B, D, D)
        
        # Volume prior log probability: log p(z) = 0.5 * log det(G^{-1}(z))
        log_det_G_inv = self.safe_logdet(G_inv_z)
        log_prior = 0.5 * log_det_G_inv
        
        # Posterior log probability (assuming local Riemannian posterior)
        diff = z - mu  # (B, D)
        quad_form = torch.sum(diff.unsqueeze(1) * torch.bmm(G_inv_z, diff.unsqueeze(-1)).squeeze(-1), dim=1)
        log_posterior = -0.5 * quad_form
        

        
        # KL divergence: KL(q||p) = E_q[log q - log p]
        # Both should be (B,) tensors
        # Sum over the second dimension if log_posterior has extra dimensions
        if log_posterior.dim() > 1:
            log_posterior = log_posterior.sum(dim=1)
        kl = log_posterior - log_prior
        
        # Apply beta scaling
        kl = self.beta * kl.mean()
        
        # Log statistics
        self.log("kl_loss", kl.item())
        self.log("kl_beta", self.beta)
        
        return kl
    
    def _euclidean_kl(self, mu: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Fallback to Euclidean KL divergence."""
        # Standard KL divergence for Gaussian
        kl = 0.5 * torch.sum(mu ** 2 + z ** 2 - 1 - torch.log(z ** 2 + 1e-8), dim=1)
        return self.beta * kl.mean()


@register("kl_euclidean")
class KLEuclideanLoss(KLLoss, LoggingMixin, DeviceMixin):
    """Standard Euclidean KL divergence."""
    
    def __init__(self, latent_dim: int = None, beta: float = 1.0):
        super().__init__()
        LoggingMixin.__init__(self)
        
        self.beta = beta
    
    def forward(
        self, 
        mu: torch.Tensor, 
        z: torch.Tensor, 
        metric: Optional[object] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute Euclidean KL divergence.
        
        Args:
            mu: Posterior mean of shape (B, D)
            z: Sampled latents of shape (B, D)
            metric: Not used for Euclidean KL
            **kwargs: Additional arguments
            
        Returns:
            KL loss scalar
        """
        # Standard KL divergence for Gaussian
        kl = 0.5 * torch.sum(mu ** 2 + z ** 2 - 1 - torch.log(z ** 2 + 1e-8), dim=1)
        kl = self.beta * kl.mean()
        
        # Log statistics
        self.log("kl_loss", kl.item())
        self.log("kl_beta", self.beta)
        
        return kl
