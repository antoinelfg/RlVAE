"""
RlVAE Main Presentation
========================

Main orchestrator for the complete RlVAE presentation.
"""

from manim import *
from manim_slides import Slide
from components.persistent_scheme import PersistentRlVAEScheme
from components.color_scheme import COLOR_SCHEME

class RlVAEMainPresentation(Slide):
    """Main presentation orchestrator for RlVAE."""
    
    def __init__(self):
        super().__init__()
        self.scheme = PersistentRlVAEScheme()
        self.scheme.position_in_corner()
    
    def construct(self):
        """Main construction sequence."""
        
        # Add persistent scheme to all slides
        self.add(self.scheme)
        
        # Scene 1: VAE Fundamentals & Metric Extraction
        self.scheme.highlight_section("vanilla_vae", stage=1)
        self.vae_metric_scene()
        self.next_slide()
        
        # Scene 2: Riemannian Geometry
        self.scheme.highlight_section("riemannian_prior", stage=2)
        self.riemannian_geometry_scene()
        self.next_slide()
        
        # Scene 3: RlVAE Architecture
        self.scheme.highlight_section("flow_sequence", stage=2)
        self.rlvae_architecture_scene()
        self.next_slide()
        
        # Scene 4: Flow Sequence Progression
        self.scheme.highlight_flow_progression("z₀")
        self.flow_progression_scene()
        self.next_slide()
        
        # Scene 5: Training Process
        self.scheme.highlight_section("geometric_loss", stage=4)
        self.training_process_scene()
        self.next_slide()
        
        # Scene 6: Results & Evaluation
        self.scheme.highlight_section("sequence_quality", stage=4)
        self.results_evaluation_scene()
        self.next_slide()
    
    def vae_metric_scene(self):
        """Scene 1: VAE Fundamentals & Metric Extraction."""
        # Import and run the VAE metric extraction scene
        from scenes.vae_metric_extraction import VAEMetricExtraction
        vae_scene = VAEMetricExtraction()
        vae_scene.construct()
    
    def riemannian_geometry_scene(self):
        """Scene 2: Riemannian Geometry Introduction."""
        from scenes.riemannian_geometry import RiemannianGeometry
        riemannian_scene = RiemannianGeometry()
        riemannian_scene.construct()
    
    def rlvae_architecture_scene(self):
        """Scene 3: RlVAE Architecture Overview."""
        from scenes.rlvae_architecture import RlVAEArchitecture
        arch_scene = RlVAEArchitecture()
        arch_scene.construct()
    
    def flow_progression_scene(self):
        """Scene 4: Flow-Based Sequence Progression."""
        from scenes.flow_sequence_progression import FlowSequenceProgression
        flow_scene = FlowSequenceProgression()
        flow_scene.construct()
    
    def training_process_scene(self):
        """Scene 5: Training Process Visualization."""
        from scenes.training_process import TrainingProcess
        training_scene = TrainingProcess()
        training_scene.construct()
    
    def results_evaluation_scene(self):
        """Scene 6: Results & Evaluation."""
        from scenes.results_evaluation import ResultsEvaluation
        results_scene = ResultsEvaluation()
        results_scene.construct()

if __name__ == "__main__":
    # Run the complete presentation
    presentation = RlVAEMainPresentation()
    presentation.render()
