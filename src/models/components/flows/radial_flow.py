"""
Radial Flow implementation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from ...base import Flow, register
from ...base.mixins import LoggingMixin, DeviceMixin


@register("radial_flow")
class RadialFlow(Flow, LoggingMixin, DeviceMixin):
    """Radial flow transformation."""
    
    def __init__(self, latent_dim: int):
        super().__init__()
        LoggingMixin.__init__(self)
        
        self.latent_dim = latent_dim
        
        # Learnable parameters
        self.z0 = nn.Parameter(torch.randn(latent_dim))
        self.alpha = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(1))
    
    def forward(self, z: torch.Tensor, t: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply radial transformation.
        
        Args:
            z: Input tensor of shape (B, D)
            t: Not used for radial flow
            
        Returns:
            Tuple of (transformed_z, log_det_jacobian)
        """
        # Radial transformation
        r = torch.norm(z - self.z0, dim=1, keepdim=True)
        h = 1 / (self.alpha + r)
        beta_h = self.beta * h
        
        f_z = z + beta_h * (z - self.z0)
        
        # Log determinant
        log_det = (self.latent_dim - 1) * torch.log(1 + beta_h) + torch.log(1 + beta_h + self.beta * r * h ** 2)
        
        # Log statistics
        self.log("radial_z0_norm", torch.norm(self.z0).item())
        self.log("radial_alpha", self.alpha.item())
        self.log("radial_beta", self.beta.item())
        self.log("radial_log_det_mean", log_det.mean().item())
        
        return f_z, log_det
    
    def inverse(self, z: torch.Tensor, t: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply inverse radial transformation (approximate).
        
        Args:
            z: Input tensor of shape (B, D)
            t: Not used for radial flow
            
        Returns:
            Tuple of (inverse_transformed_z, log_det_jacobian)
        """
        # Note: Exact inverse is not available for radial flows
        # This is a placeholder implementation
        raise NotImplementedError("Exact inverse not available for radial flows")
