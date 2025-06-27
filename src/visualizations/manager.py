"""
Visualization Manager
====================

Central coordinator for all visualization modules.
Provides configurable, performance-aware visualization execution.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch

from .base import BaseVisualization
from .basic import BasicVisualizations
from .manifold import ManifoldVisualizations
from .interactive import InteractiveVisualizations
from .flow_analysis import FlowAnalysisVisualizations


class VisualizationLevel(Enum):
    """Predefined visualization complexity levels."""
    MINIMAL = "minimal"      # Only basic cyclicity analysis
    BASIC = "basic"          # Essential visualizations
    STANDARD = "standard"    # Most common visualizations
    ADVANCED = "advanced"    # Includes interactive elements
    FULL = "full"           # All visualizations


@dataclass
class VisualizationConfig:
    """Configuration for visualization execution."""
    
    # Execution levels
    level: VisualizationLevel = VisualizationLevel.STANDARD
    
    # Category toggles
    enable_basic: bool = True
    enable_manifold: bool = True
    enable_interactive: bool = False
    enable_flow_analysis: bool = False
    
    # Frequency controls (every N epochs)
    basic_frequency: int = 1
    manifold_frequency: int = 2
    interactive_frequency: int = 9
    flow_frequency: int = 3
    
    # Advanced controls
    disable_curvature: bool = True
    max_sequences: int = 8
    enable_fancy_plots: bool = False
    
    @classmethod
    def from_level(cls, level: VisualizationLevel) -> 'VisualizationConfig':
        """Create config from predefined level."""
        configs = {
            VisualizationLevel.MINIMAL: cls(
                level=level,
                enable_basic=True,
                enable_manifold=False,
                enable_interactive=False,
                enable_flow_analysis=False,
                basic_frequency=5
            ),
            VisualizationLevel.BASIC: cls(
                level=level,
                enable_basic=True,
                enable_manifold=True,
                enable_interactive=False,
                enable_flow_analysis=False,
                manifold_frequency=5
            ),
            VisualizationLevel.STANDARD: cls(
                level=level,
                enable_basic=True,
                enable_manifold=True,
                enable_interactive=False,
                enable_flow_analysis=True,
                flow_frequency=5
            ),
            VisualizationLevel.ADVANCED: cls(
                level=level,
                enable_basic=True,
                enable_manifold=True,
                enable_interactive=True,
                enable_flow_analysis=True,
                enable_fancy_plots=True,
                interactive_frequency=1  # Run interactive every epoch for advanced
            ),
            VisualizationLevel.FULL: cls(
                level=level,
                enable_basic=True,
                enable_manifold=True,
                enable_interactive=True,
                enable_flow_analysis=True,
                enable_fancy_plots=True,
                disable_curvature=False,
                basic_frequency=1,
                manifold_frequency=1,
                interactive_frequency=3,
                flow_frequency=1
            )
        }
        return configs[level]


class VisualizationManager:
    """Central manager for coordinating all visualizations."""
    
    def __init__(self, model, device, config, viz_config: Optional[VisualizationConfig] = None):
        self.model = model
        self.device = device
        self.config = config
        
        # Use provided viz config or create standard one
        self.viz_config = viz_config or VisualizationConfig.from_level(VisualizationLevel.STANDARD)
        
        # Initialize visualization modules
        should_log = getattr(config, 'wandb_only', False) or True
        
        self.modules = {}
        if self.viz_config.enable_basic:
            self.modules['basic'] = BasicVisualizations(model, device, config, should_log)
            
        if self.viz_config.enable_manifold:
            self.modules['manifold'] = ManifoldVisualizations(model, device, config, should_log)
            
        if self.viz_config.enable_interactive:
            self.modules['interactive'] = InteractiveVisualizations(model, device, config, should_log)
            
        if self.viz_config.enable_flow_analysis:
            self.modules['flow_analysis'] = FlowAnalysisVisualizations(model, device, config, should_log)
    
    def create_visualizations(self, x_sample: torch.Tensor, epoch: int, val_loader=None):
        """
        Create visualizations based on configuration and epoch.
        
        Args:
            x_sample: Sample data for visualization
            epoch: Current training epoch
            val_loader: Validation data loader (optional)
        """
        print(f"🎨 Creating visualizations for epoch {epoch} (level: {self.viz_config.level.value})")
        
        try:
            # Basic visualizations (always run if enabled)
            if (self.viz_config.enable_basic and 
                epoch % self.viz_config.basic_frequency == 0):
                self._run_basic_visualizations(x_sample, epoch)
            
            # Manifold visualizations
            if (self.viz_config.enable_manifold and 
                epoch % self.viz_config.manifold_frequency == 0):
                self._run_manifold_visualizations(x_sample, epoch)
            
            # Interactive visualizations
            if (self.viz_config.enable_interactive and 
                epoch % self.viz_config.interactive_frequency == 0):
                self._run_interactive_visualizations(x_sample, epoch)
            
            # Flow analysis visualizations
            if (self.viz_config.enable_flow_analysis and 
                epoch % self.viz_config.flow_frequency == 0):
                self._run_flow_visualizations(x_sample, epoch)
                
        except Exception as e:
            print(f"⚠️ Visualization error at epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
    
    def _run_basic_visualizations(self, x_sample: torch.Tensor, epoch: int):
        """Run basic visualization suite."""
        if 'basic' not in self.modules:
            return
            
        basic = self.modules['basic']
        basic.create_cyclicity_analysis(x_sample, epoch)
        basic.create_sequence_trajectories(x_sample, epoch)
        
        # Reconstruction analysis less frequently
        if epoch % (self.viz_config.basic_frequency * 2) == 0:
            basic.create_reconstruction_analysis(x_sample, epoch)
    
    def _run_manifold_visualizations(self, x_sample: torch.Tensor, epoch: int):
        """Run manifold and metric visualizations."""
        if 'manifold' not in self.modules:
            return
            
        manifold = self.modules['manifold']
        manifold.create_metric_heatmaps(x_sample, epoch)
        manifold.create_pca_analysis(x_sample, epoch)
        
        # Enhanced analysis less frequently
        if epoch % (self.viz_config.manifold_frequency * 2) == 0:
            manifold.create_temporal_analysis(x_sample, epoch)
    
    def _run_interactive_visualizations(self, x_sample: torch.Tensor, epoch: int):
        """Run interactive Plotly visualizations."""
        if 'interactive' not in self.modules:
            print("⚠️ Interactive module not available")
            return
            
        print(f"🎭 Running interactive visualizations for epoch {epoch}")
        interactive = self.modules['interactive']
        
        # Core interactive visualizations
        print("🎚️ Creating geodesic sliders...")
        interactive.create_geodesic_sliders(x_sample, epoch)
        
        print("🎬 Creating metric slider visualization...")
        interactive.create_metric_slider_visualization(x_sample, epoch)
        
        # Advanced interactive features
        if self.viz_config.enable_fancy_plots:
            interactive.create_fancy_geodesics(x_sample, epoch)
            interactive.create_temporal_animation(x_sample, epoch)
        
        # HTML latent space for full level only
        if self.viz_config.level == VisualizationLevel.FULL:
            interactive.create_html_latent_space(x_sample, epoch)
    
    def _run_flow_visualizations(self, x_sample: torch.Tensor, epoch: int):
        """Run flow-based analysis visualizations."""
        if 'flow_analysis' not in self.modules:
            return
            
        flow = self.modules['flow_analysis']
        flow.create_temporal_evolution(x_sample, epoch)
        flow.create_jacobian_analysis(x_sample, epoch)
    
    def set_level(self, level: VisualizationLevel):
        """Change visualization level dynamically."""
        self.viz_config = VisualizationConfig.from_level(level)
        print(f"📊 Visualization level changed to: {level.value}")
    
    def enable_module(self, module_name: str):
        """Enable a specific visualization module."""
        if module_name == 'basic':
            self.viz_config.enable_basic = True
        elif module_name == 'manifold':
            self.viz_config.enable_manifold = True
        elif module_name == 'interactive':
            self.viz_config.enable_interactive = True
        elif module_name == 'flow_analysis':
            self.viz_config.enable_flow_analysis = True
        print(f"✅ Enabled {module_name} visualizations")
    
    def disable_module(self, module_name: str):
        """Disable a specific visualization module."""
        if module_name == 'basic':
            self.viz_config.enable_basic = False
        elif module_name == 'manifold':
            self.viz_config.enable_manifold = False
        elif module_name == 'interactive':
            self.viz_config.enable_interactive = False
        elif module_name == 'flow_analysis':
            self.viz_config.enable_flow_analysis = False
        print(f"❌ Disabled {module_name} visualizations")
    
    def get_summary(self) -> Dict:
        """Get summary of current visualization configuration."""
        return {
            'level': self.viz_config.level.value,
            'enabled_modules': [name for name, enabled in {
                'basic': self.viz_config.enable_basic,
                'manifold': self.viz_config.enable_manifold,
                'interactive': self.viz_config.enable_interactive,
                'flow_analysis': self.viz_config.enable_flow_analysis
            }.items() if enabled],
            'frequencies': {
                'basic': self.viz_config.basic_frequency,
                'manifold': self.viz_config.manifold_frequency,
                'interactive': self.viz_config.interactive_frequency,
                'flow': self.viz_config.flow_frequency
            }
        } 