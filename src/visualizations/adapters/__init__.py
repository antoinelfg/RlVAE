"""
Adapters for integrating external libraries with the visualization system.
"""

try:
    from .geodesic_metric_adapter import RLVAEGeodesicAdapter, GEODESIC_TOOLBOX_AVAILABLE
except ImportError:
    RLVAEGeodesicAdapter = None
    GEODESIC_TOOLBOX_AVAILABLE = False

__all__ = [
    "RLVAEGeodesicAdapter",
    "GEODESIC_TOOLBOX_AVAILABLE",
]
