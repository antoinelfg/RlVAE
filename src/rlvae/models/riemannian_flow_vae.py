"""Canonical import wrapper for RiemannianFlowVAE.

This keeps a stable import path `rlvae.models.riemannian_flow_vae.RiemannianFlowVAE`
while reusing the implementation in `rlvae.models.base.riemannian_flow_vae`.
"""

# Import from the base implementation
from .base.riemannian_flow_vae import (
    RiemannianFlowVAE,
    WorkingRiemannianSampler,
    OfficialRHVAESampler,
)

__all__ = ['RiemannianFlowVAE', 'WorkingRiemannianSampler', 'OfficialRHVAESampler']
