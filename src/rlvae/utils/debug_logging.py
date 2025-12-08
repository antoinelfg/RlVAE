"""
Re-export DebugPlotLogger so code can import from either ``rlvae.utils`` or
legacy ``utils`` package locations.
"""
from utils.debug_logging import DebugPlotLogger

__all__ = ["DebugPlotLogger"]
