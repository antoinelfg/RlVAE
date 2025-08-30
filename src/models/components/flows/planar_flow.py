"""
Planar Flow implementation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from ...base import Flow, register
from ...base.mixins import LoggingMixin, DeviceMixin


@register("planar_flow")
class PlanarFlow(Flow, LoggingMixin, DeviceMixin):
    """Planar flow transformation."""
    
    def __init__(self, latent_dim: int):
        super().__init__()
        LoggingMixin.__init__(self)
        
        self.latent_dim = latent_dim
        
        # Learnable parameters
        self.w = nn.Parameter(torch.randn(latent_dim))
        self.u = nn.Parameter(torch.randn(latent_dim))
        self.b = nn.Parameter(torch.randn(1))
    
    def forward(self, z: torch.Tensor, t: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply planar transformation.
        
        Args:
            z: Input tensor of shape (B, D)
            t: Not used for planar flow
            
        Returns:
            Tuple of (transformed_z, log_det_jacobian)
        """
        # Planar transformation
        zwb = torch.sum(z * self.w, dim=1, keepdim=True) + self.b
        f_z = z + self.u * torch.tanh(zwb)
        
        # Log determinant
        psi = (1 - torch.tanh(zwb) ** 2) * self.w.unsqueeze(0)
        log_det = torch.log(torch.abs(1 + torch.sum(psi * self.u, dim=1)))
        
        # Log statistics
        self.log("planar_w_norm", torch.norm(self.w).item())
        self.log("planar_u_norm", torch.norm(self.u).item())
        self.log("planar_log_det_mean", log_det.mean().item())
        
        return f_z, log_det
    
    def inverse(self, z: torch.Tensor, t: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply inverse planar transformation (approximate).
        
        Args:
            z: Input tensor of shape (B, D)
            t: Not used for planar flow
            
        Returns:
            Tuple of (inverse_transformed_z, log_det_jacobian)
        """
        # Note: Exact inverse is not available for planar flows
        # This is a placeholder implementation
        raise NotImplementedError("Exact inverse not available for planar flows")
