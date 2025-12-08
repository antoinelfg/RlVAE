"""
Legacy model components module.

This module has been deprecated. Use rlvae.models.components instead.
"""

# Re-export from rlvae.models.components for backwards compatibility
try:
    from rlvae.models.components.encoder_manager import EncoderManager
    from rlvae.models.components.decoder_manager import DecoderManager
    from rlvae.models.components.flow_manager import FlowManager
    from rlvae.models.components.loss_manager import LossManager
    from rlvae.models.components.metric_tensor import MetricTensor
    from rlvae.models.components.metric_loader import MetricLoader
except ImportError:
    pass

__all__ = [
    "EncoderManager",
    "DecoderManager", 
    "FlowManager",
    "LossManager",
    "MetricTensor",
    "MetricLoader",
]
