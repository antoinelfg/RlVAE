"""
RlVAE Visualization System
=========================

Modular visualization system for RiemannianFlowVAE training.

Categories:
- basic: Essential visualizations (cyclicity, trajectories, reconstruction)
- manifold: Metric tensor and manifold analysis
- interactive: Advanced Plotly-based interactive visualizations  
- flow_analysis: Flow Jacobian and temporal evolution analysis
- manager: Central coordinator for visualization execution
"""

from .manager import VisualizationManager
from .basic import BasicVisualizations
from .manifold import ManifoldVisualizations
from .interactive import InteractiveVisualizations
from .flow_analysis import FlowAnalysisVisualizations

__all__ = [
    'VisualizationManager',
    'BasicVisualizations',
    'ManifoldVisualizations', 
    'InteractiveVisualizations',
    'FlowAnalysisVisualizations'
] 