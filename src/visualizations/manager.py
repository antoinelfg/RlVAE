"""
Visualization Manager
====================

Central coordinator for all visualization modules.
Provides configurable, performance-aware visualization execution.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch

from .base import BaseVisualization
from .basic import BasicVisualizations
from .manifold import ManifoldVisualizations
from .interactive import InteractiveVisualizations
from .flow_analysis import FlowAnalysisVisualizations
from .latent_dynamics import LatentDynamicsVisualizations
from .manifold_evolution import ManifoldEvolutionVisualizations

from omegaconf import DictConfig


class VisualizationLevel(Enum):
    """Predefined visualization complexity levels."""
    MINIMAL = "minimal"      # Only basic cyclicity analysis
    BASIC = "basic"          # Essential visualizations
    STANDARD = "standard"    # Most common visualizations
    ADVANCED = "advanced"    # Includes interactive elements
    DYNAMICS = "dynamics"    # Includes advanced dynamics analysis
    FULL = "full"           # All visualizations
    BETWEEN = "between"     # Only sequence slider and det G


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
    enable_dynamics: bool = False
    
    # Frequency controls (every N epochs)
    basic_frequency: int = 1
    manifold_frequency: int = 2
    interactive_frequency: int = 9
    flow_frequency: int = 3
    dynamics_frequency: int = 5
    
    # Advanced controls
    disable_curvature: bool = True
    max_sequences: int = 8
    enable_fancy_plots: bool = False
    
    # Store arbitrary extra fields for extensibility
    extra: dict = field(default_factory=dict)
    
    def __post_init__(self):
        # Set any extra fields as attributes for direct access
        for k, v in self.extra.items():
            setattr(self, k, v)
    
    @classmethod
    def from_dict(cls, d):
        # Accepts a dict (e.g., from Hydra config) and sets known fields, others go to extra
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        base = {}
        extra = {}
        for k, v in d.items():
            if k in known_fields:
                base[k] = v
            else:
                extra[k] = v
        base['extra'] = extra
        return cls(**base)
    
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
                enable_dynamics=False,
                basic_frequency=5
            ),
            VisualizationLevel.BASIC: cls(
                level=level,
                enable_basic=True,
                enable_manifold=True,
                enable_interactive=False,
                enable_flow_analysis=False,
                enable_dynamics=False,
                manifold_frequency=5
            ),
            VisualizationLevel.STANDARD: cls(
                level=level,
                enable_basic=True,
                enable_manifold=True,
                enable_interactive=False,
                enable_flow_analysis=True,
                enable_dynamics=False,
                flow_frequency=5
            ),
            VisualizationLevel.ADVANCED: cls(
                level=level,
                enable_basic=True,
                enable_manifold=True,
                enable_interactive=True,
                enable_flow_analysis=True,
                enable_dynamics=False,
                enable_fancy_plots=True,
                interactive_frequency=1  # Run interactive every epoch for advanced
            ),
            VisualizationLevel.DYNAMICS: cls(
                level=level,
                enable_basic=True,
                enable_manifold=True,
                enable_interactive=False,
                enable_flow_analysis=True,
                enable_dynamics=True,
                dynamics_frequency=3,
                flow_frequency=3
            ),
            VisualizationLevel.FULL: cls(
                level=level,
                enable_basic=True,
                enable_manifold=True,
                enable_interactive=True,
                enable_flow_analysis=True,
                enable_dynamics=True,
                enable_fancy_plots=True,
                disable_curvature=False,
                basic_frequency=1,
                manifold_frequency=1,
                interactive_frequency=3,
                flow_frequency=1,
                dynamics_frequency=2
            ),
            VisualizationLevel.BETWEEN: cls(
                level=level,
                enable_basic=True,  # Enable minimal visualizations
                enable_manifold=True,  # Only for det G
                enable_interactive=True,  # Only for sequence slider
                enable_flow_analysis=False,
                enable_dynamics=False,
                interactive_frequency=1,
                manifold_frequency=1
            ),
        }
        return configs[level]


class VisualizationManager:
    """Central manager for coordinating all visualizations."""
    
    def __init__(self, model, device, config, viz_config: Optional[VisualizationConfig] = None):
        print(f"[DEBUG] VisualizationManager __init__ called. config.visualization: {getattr(config, 'visualization', None)}")
        self.model = model
        self.device = device
        self.config = config
        
        # Use provided viz config or create standard one
        if hasattr(config, 'visualization') and isinstance(config.visualization, (dict, DictConfig)):
            # Build from dict, using level if present
            level = config.visualization.get('level', VisualizationLevel.STANDARD)
            if isinstance(level, str):
                level = VisualizationLevel(level)
            base_viz_config = VisualizationConfig.from_level(level)
            print(f"[DEBUG] from_level({level}) returned: enable_interactive={base_viz_config.enable_interactive}, enable_basic={base_viz_config.enable_basic}, enable_manifold={base_viz_config.enable_manifold}")
            # Update with any extra fields from config.visualization, but do NOT override module enables
            for k, v in config.visualization.items():
                if k not in [
                    'enable_basic', 'enable_manifold', 'enable_interactive', 'enable_flow_analysis', 'enable_dynamics', 'level'
                ]:
                    setattr(base_viz_config, k, v)
            self.viz_config = base_viz_config
        else:
            print("[DEBUG] Using viz_config or default STANDARD in VisualizationManager")
            self.viz_config = viz_config or VisualizationConfig.from_level(VisualizationLevel.STANDARD)
        
        # DEBUG PRINT: Show visualization config at init
        print(f"[DEBUG] Visualization config at init: level={self.viz_config.level}, enable_interactive={self.viz_config.enable_interactive}, enable_basic={self.viz_config.enable_basic}, enable_manifold={self.viz_config.enable_manifold}")
        
        # DEBUG PRINT: Show FINAL visualization config before module init
        print(f"[DEBUG] FINAL Visualization config before module init: level={self.viz_config.level}, enable_interactive={self.viz_config.enable_interactive}, enable_basic={self.viz_config.enable_basic}, enable_manifold={self.viz_config.enable_manifold}")
        
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
            
        if self.viz_config.enable_dynamics:
            self.modules['dynamics'] = LatentDynamicsVisualizations(model, device, config, should_log)
            
        # Always enable manifold evolution for adaptive centroid training
        if hasattr(config, 'adaptive_centroids') and config.adaptive_centroids.get('enabled', False):
            self.modules['manifold_evolution'] = ManifoldEvolutionVisualizations(model, device, config, should_log)
    
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
            # Special handling for 'between' level: only sequence slider and det G
            if self.viz_config.level == VisualizationLevel.BETWEEN:
                # Minimal (basic) visualizations
                if 'basic' in self.modules and epoch % self.viz_config.basic_frequency == 0:
                    basic = self.modules['basic']
                    basic.create_cyclicity_analysis(x_sample, epoch)
                    basic.create_sequence_trajectories(x_sample, epoch)
                    basic.create_reconstruction_analysis(x_sample, epoch)  # Added: third minimal plot
                # Sequence slider (from interactive)
                if 'interactive' in self.modules and epoch % self.viz_config.interactive_frequency == 0:
                    print("[DEBUG] Calling interactive sequence slider visualization...")
                    interactive = self.modules['interactive']
                    interactive.create_sequence_slider_visualization(x_sample, epoch)
                    print("[DEBUG] Calling static metric heatmap visualization...")
                    interactive.create_static_metric_heatmap(x_sample, epoch)
                    print("[DEBUG] Calling static metric heatmap timesteps visualization...")
                    interactive.create_static_metric_heatmap_timesteps(x_sample, epoch)
                    print("[DEBUG] Calling interactive det G heatmap visualization...")
                    interactive.create_time_curvature_heatmap(x_sample, epoch)
                # det G plot (from manifold)
                if 'manifold' in self.modules and epoch % self.viz_config.manifold_frequency == 0:
                    manifold = self.modules['manifold']
                    manifold.create_metric_heatmaps(x_sample, epoch)
                return
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
            
            # Dynamics analysis visualizations
            if (self.viz_config.enable_dynamics and 
                epoch % self.viz_config.dynamics_frequency == 0):
                self._run_dynamics_visualizations(x_sample, epoch)
                
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
        print("⛰️ Creating time curvature heatmap...")
        interactive.create_time_curvature_heatmap(x_sample, epoch)
        print("🎯 Creating 2D-focused curvature heatmap...")
        interactive.create_time_curvature_heatmap_2d_focused(x_sample, epoch)
        # Advanced interactive features
        if self.viz_config.enable_fancy_plots:
            interactive.create_fancy_geodesics(x_sample, epoch)
            interactive.create_temporal_animation(x_sample, epoch)
        # HTML latent space for full level only
        if self.viz_config.level == VisualizationLevel.FULL:
            interactive.create_html_latent_space(x_sample, epoch)
        # Always run the sequence slider visualization and safely set sequence count
        try:
            # Try to set sequence_viz_count in the config
            sequence_count = min(x_sample.shape[0], 16)
            
            if hasattr(self.config, 'visualization'):
                # If it's a dict-like object, set directly
                if isinstance(self.config.visualization, dict):
                    self.config.visualization['sequence_viz_count'] = sequence_count
                else:
                    # Try OmegaConf struct_mode temporarily disabled
                    from omegaconf import OmegaConf
                    try:
                        # Save current struct mode
                        was_struct = OmegaConf.is_struct(self.config.visualization)
                        # Temporarily disable struct mode
                        OmegaConf.set_struct(self.config.visualization, False)
                        # Set the value
                        self.config.visualization.sequence_viz_count = sequence_count
                        # Restore struct mode
                        OmegaConf.set_struct(self.config.visualization, was_struct)
                    except Exception:
                        # Fallback: set on viz_config
                        self.viz_config.sequence_viz_count = sequence_count
            elif hasattr(self.config, 'sequence_viz_count'):
                self.config.sequence_viz_count = sequence_count
            else:
                # Set on the viz_config instead
                self.viz_config.sequence_viz_count = sequence_count
        except Exception as e:
            print(f"⚠️ Could not set sequence_viz_count in config: {e}")
            # Set on the viz_config as fallback
            self.viz_config.sequence_viz_count = min(x_sample.shape[0], 16)
            
        print("🎞️ Creating sequence slider visualization...")
        interactive.create_sequence_slider_visualization(x_sample, epoch)
    
    def _run_flow_visualizations(self, x_sample: torch.Tensor, epoch: int):
        """Run flow-based analysis visualizations."""
        if 'flow_analysis' not in self.modules:
            return
            
        flow = self.modules['flow_analysis']
        flow.create_temporal_evolution(x_sample, epoch)
        flow.create_jacobian_analysis(x_sample, epoch)
    
    def _run_dynamics_visualizations(self, x_sample: torch.Tensor, epoch: int):
        """Run latent dynamics analysis visualizations."""
        if 'dynamics' not in self.modules:
            return
            
        print(f"🌀 Running dynamics visualizations for epoch {epoch}")
        dynamics = self.modules['dynamics']
        
        # Core dynamics analyses
        dynamics.create_phase_portrait_analysis(x_sample, epoch)
        dynamics.create_velocity_field_analysis(x_sample, epoch)
        
        # Advanced dynamics analyses less frequently
        if epoch % (self.viz_config.dynamics_frequency * 2) == 0:
            dynamics.create_energy_landscape_analysis(x_sample, epoch)
            dynamics.create_attractor_analysis(x_sample, epoch)
    
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
        elif module_name == 'dynamics':
            self.viz_config.enable_dynamics = True
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
        elif module_name == 'dynamics':
            self.viz_config.enable_dynamics = False
        print(f"❌ Disabled {module_name} visualizations")
    
    def get_summary(self) -> Dict:
        """Get summary of current visualization configuration."""
        return {
            'level': self.viz_config.level.value,
            'enabled_modules': [name for name, enabled in {
                'basic': self.viz_config.enable_basic,
                'manifold': self.viz_config.enable_manifold,
                'interactive': self.viz_config.enable_interactive,
                'flow_analysis': self.viz_config.enable_flow_analysis,
                'dynamics': self.viz_config.enable_dynamics
            }.items() if enabled],
            'frequencies': {
                'basic': self.viz_config.basic_frequency,
                'manifold': self.viz_config.manifold_frequency,
                'interactive': self.viz_config.interactive_frequency,
                'flow': self.viz_config.flow_frequency,
                'dynamics': self.viz_config.dynamics_frequency
            }
        } 

    def log_final_visualizations_to_wandb(self, epoch: int):
        """
        Log the latest generation and inference visualizations to wandb at the end of the run.
        This collects the most recent saved images from each visualization module (if available)
        and logs them to wandb with appropriate captions.
        """
        import wandb
        logged = False
        for module_name, module in self.modules.items():
            # Check for common visualization attributes
            for attr, label in [
                ("last_reconstruction_path", "final/reconstruction"),
                ("last_sequence_trajectories_path", "final/sequence_trajectories"),
                ("last_cyclicity_path", "final/cyclicity"),
                ("last_generation_path", "final/generation"),
                ("last_pca_analysis_path", "final/pca_analysis"),
                ("last_heatmaps_path", "final/heatmaps"),
                ("last_temporal_metric_path", "final/temporal_metric")
            ]:
                if hasattr(module, attr):
                    path = getattr(module, attr)
                    if path:
                        wandb.log({label: wandb.Image(path, caption=f"{label.replace('final/', '').replace('_', ' ').title()} (Epoch {epoch})")})
                        logged = True
        if not logged:
            print("[VisualizationManager] No final visualizations found to log to wandb.") 