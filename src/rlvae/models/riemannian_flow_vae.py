"""Canonical import wrapper for RiemannianFlowVAE.

This keeps a stable import path `rlvae.models.riemannian_flow_vae.RiemannianFlowVAE`
while reusing the implementation in `src.models.riemannian_flow_vae`.
"""

# Import from the local implementation
from models.riemannian_flow_vae import (
    RiemannianFlowVAE,
    WorkingRiemannianSampler,
    OfficialRHVAESampler,
)

__all__ = ['RiemannianFlowVAE', 'WorkingRiemannianSampler', 'OfficialRHVAESampler']
