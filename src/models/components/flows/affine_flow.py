"""
Affine Flow implementation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from ...base import Flow, register
from ...base.mixins import LoggingMixin, DeviceMixin


@register("affine_flow")
class AffineFlow(Flow, LoggingMixin, DeviceMixin):
    """Simple affine flow transformation."""
    
    def __init__(self, latent_dim: int):
        super().__init__()
        LoggingMixin.__init__(self)
        
        self.latent_dim = latent_dim
        
        # Learnable parameters
        self.scale = nn.Parameter(torch.ones(latent_dim))
        self.shift = nn.Parameter(torch.zeros(latent_dim))
    
    def forward(self, z: torch.Tensor, t: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply affine transformation.
        
        Args:
            z: Input tensor of shape (B, D)
            t: Not used for affine flow
            
        Returns:
            Tuple of (transformed_z, log_det_jacobian)
        """
        # Apply affine transformation
        z_transformed = z * self.scale + self.shift
        
        # Log determinant is sum of log scales
        log_det = torch.sum(torch.log(torch.abs(self.scale)))
        
        # Log statistics
        self.log("flow_scale_mean", self.scale.mean().item())
        self.log("flow_shift_mean", self.shift.mean().item())
        self.log("flow_log_det", log_det.item())
        
        return z_transformed, log_det
    
    def inverse(self, z: torch.Tensor, t: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply inverse affine transformation.
        
        Args:
            z: Input tensor of shape (B, D)
            t: Not used for affine flow
            
        Returns:
            Tuple of (inverse_transformed_z, log_det_jacobian)
        """
        # Inverse transformation
        z_inverse = (z - self.shift) / self.scale
        
        # Log determinant is negative of forward
        log_det = -torch.sum(torch.log(torch.abs(self.scale)))
        
        return z_inverse, log_det
