"""
RiemannianFlowVAE Components Package

This package contains modular components extracted from the monolithic
riemannian_flow_vae.py implementation.

Components:
- metric_tensor: Riemannian metric tensor computations
- metric_loader: Pretrained metric loading utilities  
- flow_manager: Normalizing flow sequence management
- loss_manager: Loss computation coordination
"""

from .metric_tensor import MetricTensor
from .metric_loader import MetricLoader

__all__ = [
    'MetricTensor',
    'MetricLoader',
] 