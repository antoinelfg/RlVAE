"""
Identity Metric implementation.
"""

import torch
import torch.nn as nn

from ...base import Metric, register
from ...base.mixins import LoggingMixin, DeviceMixin


@register("identity_metric")
class IdentityMetric(Metric, LoggingMixin, DeviceMixin):
    """Identity metric G(z) = I for Euclidean geometry."""
    
    def __init__(self, latent_dim: int):
        super().__init__()
        LoggingMixin.__init__(self)
        
        self.latent_dim = latent_dim
    
    def G(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute identity metric tensor G(z) = I.
        
        Args:
            z: Latent points of shape (B, D)
            
        Returns:
            Identity matrix of shape (B, D, D)
        """
        batch_size = z.shape[0]
        device = z.device
        
        # Create identity matrices
        G = torch.eye(self.latent_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Log statistics
        self.log("metric_identity", True)
        
        return G
    
    def G_inv(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse identity metric G^{-1}(z) = I.
        
        Args:
            z: Latent points of shape (B, D)
            
        Returns:
            Identity matrix of shape (B, D, D)
        """
        return self.G(z)
