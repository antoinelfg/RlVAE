"""
RlVAE Scene Runner
==================

Script to run individual scenes for the RlVAE presentation.
"""

import sys
import os
from manim import config

def run_scene(scene_name):
    """Run a specific scene by name."""
    
    # Set up manim configuration
    config.background_color = "#0d1117"
    config.frame_rate = 30
    config.pixel_height = 1080
    config.pixel_width = 1920
    config.quality = "high_quality"
    config.disable_caching = True
    
    # Import and run the specific scene
    if scene_name == "vae_metric":
        from scenes.vae_metric_extraction import VAEMetricExtraction
        scene = VAEMetricExtraction()
        scene.render()
        
    elif scene_name == "riemannian":
        from scenes.riemannian_geometry import RiemannianGeometry
        scene = RiemannianGeometry()
        scene.render()
        
    elif scene_name == "architecture":
        from scenes.rlvae_architecture import RlVAEArchitecture
        scene = RlVAEArchitecture()
        scene.render()
        
    elif scene_name == "flow_progression":
        from scenes.flow_sequence_progression import FlowSequenceProgression
        scene = FlowSequenceProgression()
        scene.render()
        
    elif scene_name == "training":
        from scenes.training_process import TrainingProcess
        scene = TrainingProcess()
        scene.render()
        
    elif scene_name == "results":
        from scenes.results_evaluation import ResultsEvaluation
        scene = ResultsEvaluation()
        scene.render()
        
    else:
        print(f"Unknown scene: {scene_name}")
        print("Available scenes: vae_metric, riemannian, architecture, flow_progression, training, results")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_scene.py <scene_name>")
        print("Available scenes: vae_metric, riemannian, architecture, flow_progression, training, results")
        sys.exit(1)
    
    scene_name = sys.argv[1]
    run_scene(scene_name)
