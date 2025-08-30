"""
ELBO loss implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from ...base import ELBOLoss, register
from ...base.mixins import LoggingMixin, DeviceMixin


@register("elbo_loss")
class ELBOLoss(ELBOLoss, LoggingMixin, DeviceMixin):
    """Evidence Lower BOund loss."""
    
    def __init__(
        self,
        reconstruction_loss: nn.Module,
        kl_loss: nn.Module,
        flow_loss_weight: float = 1.0,
        loop_penalty_weight: float = 0.0
    ):
        super().__init__()
        LoggingMixin.__init__(self)
        
        self.reconstruction_loss = reconstruction_loss
        self.kl_loss = kl_loss
        self.flow_loss_weight = flow_loss_weight
        self.loop_penalty_weight = loop_penalty_weight
    
    def forward(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        z: torch.Tensor,
        log_var: Optional[torch.Tensor] = None,
        flow_log_det: Optional[torch.Tensor] = None,
        loop_penalty: Optional[torch.Tensor] = None,
        metric: Optional[object] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ELBO loss.
        
        Args:
            x: Original input
            x_recon: Reconstructed input
            mu: Posterior mean
            z: Sampled latents
            log_var: Posterior log variance
            flow_log_det: Flow log determinant
            loop_penalty: Loop penalty term
            metric: Metric object for KL computation
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing all loss components
        """
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(x, x_recon)
        
        # KL divergence loss
        kl_loss = self.kl_loss(mu, z, metric=metric, **kwargs)
        
        # Flow loss (if provided)
        flow_loss = torch.tensor(0.0, device=x.device)
        if flow_log_det is not None:
            flow_loss = -self.flow_loss_weight * flow_log_det.mean()
        
        # Loop penalty (if provided)
        loop_loss = torch.tensor(0.0, device=x.device)
        if loop_penalty is not None:
            loop_loss = self.loop_penalty_weight * loop_penalty.mean()
        
        # Total loss
        total_loss = recon_loss + kl_loss + flow_loss + loop_loss
        
        # Log all components
        self.log("total_loss", total_loss.item())
        self.log("reconstruction_loss", recon_loss.item())
        self.log("kl_loss", kl_loss.item())
        self.log("flow_loss", flow_loss.item())
        self.log("loop_loss", loop_loss.item())
        
        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
            "flow_loss": flow_loss,
            "loop_penalty": loop_loss
        }
