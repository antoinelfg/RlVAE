"""
RiemannianFlowVAE Components Package

This package contains modular components extracted from the monolithic
riemannian_flow_vae.py implementation.

Components:
- metric_tensor: Riemannian metric tensor computations
- metric_loader: Pretrained metric loading utilities  
- flow_manager: Normalizing flow sequence management
- loss_manager: Loss computation coordination
- encoder_manager: Modular encoder architectures
- decoder_manager: Modular decoder architectures
"""

from .metric_tensor import MetricTensor
from .metric_loader import MetricLoader
from .flow_manager import FlowManager
from .loss_manager import LossManager
from .encoder_manager import EncoderManager
from .decoder_manager import DecoderManager

__all__ = [
    'MetricTensor',
    'MetricLoader',
    'FlowManager',
    'LossManager',
    'EncoderManager',
    'DecoderManager',
] 