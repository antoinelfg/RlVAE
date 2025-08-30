"""
Composite models.

This module contains composite models that wire together components.
"""

from .rlvae import RLVAE
from .vae import VAE

__all__ = [
    "RLVAE",
    "VAE",
]
