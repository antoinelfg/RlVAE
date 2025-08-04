#!/usr/bin/env python3
"""
Advanced Interactive Visualizations Using Existing System
=========================================================

Leverage the project's existing interactive visualization system
to create rich interactive plots with the trained RLVAE model.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from omegaconf import DictConfig
import logging
from datetime import datetime
import wandb

# Setup paths
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir.parent / "src"
lib_src_dir = src_dir / "lib" / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(lib_src_dir) not in sys.path:
    sys.path.insert(0, str(lib_src_dir))

# Project imports
from models.modular_rlvae import ModularRiemannianFlowVAE
from data.cyclic_dataset import CyclicSpritesDataModule
from visualizations.interactive import InteractiveVisualizations
from visualizations.manager import VisualizationManager, VisualizationLevel, VisualizationConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedInteractiveAnalyzer:
    """Use the existing advanced interactive visualization system."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """Initialize analyzer."""
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.model = None
        self.data_module = None
        
        self.output_dir = Path("outputs/advanced_interactive") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎭 Advanced interactive analyzer initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model_and_data(self) -> None:
        """Load model and data."""
        logger.info(f"🔄 Loading model from {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        model_hparams = checkpoint['hyper_parameters']['model']
        
        # Create config
        config = DictConfig({
            'input_dim': model_hparams['input_dim'],
            'latent_dim': model_hparams['latent_dim'],
            'n_flows': model_hparams['n_flows'],
            'flow_hidden_size': model_hparams['flow_hidden_size'],
            'flow_n_blocks': model_hparams['flow_n_blocks'],
            'flow_n_hidden': model_hparams['flow_n_hidden'],
            'epsilon': model_hparams['epsilon'],
            'encoder': model_hparams['encoder'],
            'decoder': model_hparams['decoder'],
            'beta': model_hparams['beta'],
            'riemannian_beta': model_hparams['riemannian_beta'],
            'posterior': model_hparams['posterior'],
            'sampling': model_hparams['sampling'],
            'loop': model_hparams['loop'],
            'metric': model_hparams['metric'],
            'pretrained': {'encoder_path': None, 'decoder_path': None, 'metric_path': None},
            'sequence_length': model_hparams['sequence_length']
        })
        
        # Create and load model
        self.model = ModularRiemannianFlowVAE(config)
        
        # Resize metric tensor
        state_dict = checkpoint['state_dict']
        clean_state_dict = {k.replace('model.', '') if k.startswith('model.') else k: v 
                           for k, v in state_dict.items()}
        
        # Fix metric tensor size
        for name, param in clean_state_dict.items():
            if 'modular_metric.centroids' in name:
                self.model.modular_metric.centroids = torch.nn.Parameter(torch.zeros_like(param))
            elif 'modular_metric.metric_matrices' in name:
                self.model.modular_metric.metric_matrices = torch.nn.Parameter(torch.zeros_like(param))
        
        self.model.load_state_dict(clean_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # Setup data with larger batch size for richer visualizations
        data_config = DictConfig({
            'train_path': 'data/sprites/ColoredCircles_train.pt',
            'test_path': 'data/sprites/ColoredCircles_test.pt',
            'train_meta_path': 'data/sprites/ColoredCircles_train_params.pt',
            'test_meta_path': 'data/sprites/ColoredCircles_test_params.pt',
            'sequence_length': 10,
            'image_size': [28, 28],
            'channels': 3,
            'batch_size': 8,  # Larger batch for richer visualizations
            'num_workers': 0,
            'pin_memory': False,
            'max_test_samples': 500,
            'verify_cyclicity': False
        })
        
        self.data_module = CyclicSpritesDataModule(data_config)
        self.data_module.setup('test')
        
        logger.info("✅ Model and data loaded successfully")
    
    def create_comprehensive_interactive_visualizations(self) -> None:
        """Create comprehensive interactive visualizations using the existing system."""
        logger.info("🎭 Creating comprehensive interactive visualizations")
        
        # Create rich visualization config
        viz_config = DictConfig({
            'level': 'full',
            'basic_frequency': 1,
            'manifold_frequency': 1,
            'interactive_frequency': 1,
            'enable_fancy_plots': True,
            'max_sequences': 100,  # Many sequences for rich data
            'sequence_viz_count': 30,  # Many sequences in visualizations
            'enable_pca_plots': True,
            'enable_flow_analysis': True,
            'enable_comprehensive': True,
            'save_plots': True,
            'log_to_wandb': True,
            'output_dir': str(self.output_dir)
        })
        
        # Create interactive visualizer with the trained model
        interactive_viz = InteractiveVisualizations(
            model=self.model,
            config=viz_config,
            device=self.device
        )
        
        # Get sample batches for visualization
        test_loader = self.data_module.test_dataloader()
        sample_batch = next(iter(test_loader))
        sample_batch = sample_batch.to(self.device)
        
        logger.info(f"📊 Sample batch shape: {sample_batch.shape}")
        
        # Create comprehensive interactive visualizations
        epoch = 0  # Using epoch 0 for post-training analysis
        
        logger.info("🎚️ Creating geodesic sliders...")
        try:
            interactive_viz.create_geodesic_sliders(sample_batch, epoch)
            logger.info("✅ Geodesic sliders created")
        except Exception as e:
            logger.warning(f"⚠️ Geodesic sliders failed: {e}")
        
        logger.info("🎯 Creating metric slider visualization...")
        try:
            interactive_viz.create_metric_slider_visualization(sample_batch, epoch)
            logger.info("✅ Metric slider visualization created")
        except Exception as e:
            logger.warning(f"⚠️ Metric slider visualization failed: {e}")
        
        logger.info("🎬 Creating sequence slider visualization...")
        try:
            interactive_viz.create_sequence_slider_visualization(sample_batch, epoch)
            logger.info("✅ Sequence slider visualization created")
        except Exception as e:
            logger.warning(f"⚠️ Sequence slider visualization failed: {e}")
        
        logger.info("⛰️ Creating time curvature heatmap...")
        try:
            interactive_viz.create_time_curvature_heatmap(sample_batch, epoch)
            logger.info("✅ Time curvature heatmap created")
        except Exception as e:
            logger.warning(f"⚠️ Time curvature heatmap failed: {e}")
        
        logger.info("🎯 Creating 2D-focused curvature heatmap...")
        try:
            interactive_viz.create_time_curvature_heatmap_2d_focused(sample_batch, epoch)
            logger.info("✅ 2D-focused curvature heatmap created")
        except Exception as e:
            logger.warning(f"⚠️ 2D-focused curvature heatmap failed: {e}")
        
        logger.info("🌐 Creating HTML latent space...")
        try:
            interactive_viz.create_html_latent_space(sample_batch, epoch)
            logger.info("✅ HTML latent space created")
        except Exception as e:
            logger.warning(f"⚠️ HTML latent space failed: {e}")
        
        logger.info("✨ Creating fancy geodesics...")
        try:
            interactive_viz.create_fancy_geodesics(sample_batch, epoch)
            logger.info("✅ Fancy geodesics created")
        except Exception as e:
            logger.warning(f"⚠️ Fancy geodesics failed: {e}")
        
        logger.info("🎞️ Creating temporal animation...")
        try:
            interactive_viz.create_temporal_animation(sample_batch, epoch)
            logger.info("✅ Temporal animation created")
        except Exception as e:
            logger.warning(f"⚠️ Temporal animation failed: {e}")
        
        logger.info("📊 Creating static metric heatmap...")
        try:
            interactive_viz.create_static_metric_heatmap(sample_batch, epoch)
            logger.info("✅ Static metric heatmap created")
        except Exception as e:
            logger.warning(f"⚠️ Static metric heatmap failed: {e}")
        
        logger.info("📈 Creating static metric heatmap timesteps...")
        try:
            interactive_viz.create_static_metric_heatmap_timesteps(sample_batch, epoch)
            logger.info("✅ Static metric heatmap timesteps created")
        except Exception as e:
            logger.warning(f"⚠️ Static metric heatmap timesteps failed: {e}")
        
        logger.info("✅ Comprehensive interactive visualizations completed")
    
    def create_multiple_batch_visualizations(self) -> None:
        """Create visualizations using multiple batches for richer data."""
        logger.info("🌊 Creating multiple batch visualizations")
        
        test_loader = self.data_module.test_dataloader()
        
        # Create visualizations for multiple batches
        viz_config = DictConfig({
            'level': 'full',
            'max_sequences': 50,
            'sequence_viz_count': 20,
            'enable_fancy_plots': True,
            'enable_pca_plots': True,
            'save_plots': True,
            'log_to_wandb': True,
            'output_dir': str(self.output_dir)
        })
        
        interactive_viz = InteractiveVisualizations(
            model=self.model,
            config=viz_config,
            device=self.device
        )
        
        batch_count = 0
        for batch in test_loader:
            if batch_count >= 5:  # Process 5 batches
                break
            
            batch = batch.to(self.device)
            
            logger.info(f"📊 Processing batch {batch_count+1}/5, shape: {batch.shape}")
            
            try:
                # Create sequence-specific visualizations
                interactive_viz.create_sequence_slider_visualization(batch, batch_count)
                logger.info(f"✅ Sequence slider for batch {batch_count+1} created")
            except Exception as e:
                logger.warning(f"⚠️ Batch {batch_count+1} visualization failed: {e}")
            
            batch_count += 1
        
        logger.info("✅ Multiple batch visualizations completed")
    
    def run_advanced_analysis(self) -> None:
        """Run complete advanced interactive analysis."""
        logger.info("🚀 Starting advanced interactive analysis")
        
        # Initialize WandB
        wandb.init(
            project="rlvae-advanced-interactive",
            name=f"advanced_interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "checkpoint": self.checkpoint_path,
                "device": self.device,
                "analysis_type": "advanced_interactive",
                "max_sequences": 100,
                "enable_fancy_plots": True
            }
        )
        
        # Load model and data
        self.load_model_and_data()
        
        # Create comprehensive interactive visualizations
        self.create_comprehensive_interactive_visualizations()
        
        # Create multiple batch visualizations
        self.create_multiple_batch_visualizations()
        
        # Log all HTML files to WandB
        for html_file in self.output_dir.glob("**/*.html"):
            try:
                relative_path = html_file.relative_to(self.output_dir)
                log_name = str(relative_path).replace('/', '_').replace('.html', '')
                wandb.log({log_name: wandb.Html(str(html_file))})
                logger.info(f"📤 Logged {log_name} to WandB")
            except Exception as e:
                logger.warning(f"⚠️ Failed to log {html_file}: {e}")
        
        logger.info("🎉 Advanced interactive analysis completed!")
        logger.info(f"📁 All results saved in: {self.output_dir}")
        
        wandb.finish()


def main():
    """Main execution."""
    checkpoint_path = "outputs/checkpoints/epoch=197-val_loss=6.402.ckpt"
    
    analyzer = AdvancedInteractiveAnalyzer(checkpoint_path)
    analyzer.run_advanced_analysis()


if __name__ == "__main__":
    main() 