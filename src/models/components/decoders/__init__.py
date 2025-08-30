"""
Decoder components.

This module contains decoder implementations for various architectures.
"""

from .mlp_decoder import MLPDecoder
from .cnn_decoder import CNNDecoder

__all__ = [
    "MLPDecoder",
    "CNNDecoder",
]
