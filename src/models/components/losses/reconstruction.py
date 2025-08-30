"""
Reconstruction loss implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ...base import ReconstructionLoss, register
from ...base.mixins import LoggingMixin, DeviceMixin


@register("gaussian_reconstruction")
class GaussianReconstructionLoss(ReconstructionLoss, LoggingMixin, DeviceMixin):
    """Gaussian reconstruction loss."""
    
    def __init__(self, sigma: float = 0.1):
        super().__init__()
        LoggingMixin.__init__(self)
        
        self.sigma = sigma
        self.sigma_squared = sigma ** 2
    
    def forward(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian reconstruction loss.
        
        Args:
            x: Original input of shape (B, ...)
            x_recon: Reconstructed input of shape (B, ...)
            
        Returns:
            Reconstruction loss scalar
        """
        # MSE loss with Gaussian assumption
        mse = F.mse_loss(x_recon, x, reduction='none')
        loss = 0.5 * torch.sum(mse / self.sigma_squared, dim=tuple(range(1, mse.dim())))
        
        # Log statistics
        self.log("recon_loss_mean", loss.mean().item())
        self.log("recon_loss_std", loss.std().item())
        
        return loss.mean()


@register("bernoulli_reconstruction")
class BernoulliReconstructionLoss(ReconstructionLoss, LoggingMixin, DeviceMixin):
    """Bernoulli reconstruction loss (binary cross entropy)."""
    
    def __init__(self):
        super().__init__()
        LoggingMixin.__init__(self)
    
    def forward(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """
        Compute Bernoulli reconstruction loss.
        
        Args:
            x: Original input of shape (B, ...) in [0, 1]
            x_recon: Reconstructed input of shape (B, ...) in [0, 1]
            
        Returns:
            Reconstruction loss scalar
        """
        # Binary cross entropy loss
        loss = F.binary_cross_entropy(x_recon, x, reduction='none')
        loss = torch.sum(loss, dim=tuple(range(1, loss.dim())))
        
        # Log statistics
        self.log("recon_loss_mean", loss.mean().item())
        self.log("recon_loss_std", loss.std().item())
        
        return loss.mean()
