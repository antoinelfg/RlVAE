"""Canonical import wrapper for the original RiemannianFlowVAE.

This keeps a stable import path `rlvae.models.riemannian_flow_vae.RiemannianFlowVAE`
while reusing the implementation in `original_rlvae.src.models.riemannian_flow_vae`.
"""

try:
    # Prefer the original implementation and re-export helper samplers
    from original_rlvae.src.models.riemannian_flow_vae import (  # type: ignore
        RiemannianFlowVAE,
        WorkingRiemannianSampler,
        OfficialRHVAESampler,
    )
except Exception as e:
    # Fallback: if original path not available, raise a helpful error
    raise ImportError(
        "Failed to import original RiemannianFlowVAE. Ensure `original_rlvae` is present "
        "and on PYTHONPATH."
    ) from e
