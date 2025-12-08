"""
Base module for model components.

This module provides abstract base classes and registry functionality
for all model components (encoders, decoders, metrics, flows, etc.).
"""

from .interfaces import (
    Encoder,
    Decoder,
    Metric,
    Flow,
    Prior,
    Posterior,
    Sampler,
    KLLoss,
    ReconstructionLoss,
    ELBOLoss
)

from .registry import (
    register,
    build_component,
    get_component,
    list_components
)

__all__ = [
    # Interfaces
    "Encoder",
    "Decoder", 
    "Metric",
    "Flow",
    "Prior",
    "Posterior",
    "Sampler",
    "KLLoss",
    "ReconstructionLoss",
    "ELBOLoss",
    # Registry
    "register",
    "build_component", 
    "get_component",
    "list_components"
]
