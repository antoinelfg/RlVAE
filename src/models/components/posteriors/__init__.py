"""
Posterior components.

This module contains posterior sampling implementations.
"""

from .local_riemannian import LocalRiemannianPosterior
from .euclidean_gaussian import EuclideanGaussianPosterior

__all__ = [
    "LocalRiemannianPosterior",
    "EuclideanGaussianPosterior",
]
