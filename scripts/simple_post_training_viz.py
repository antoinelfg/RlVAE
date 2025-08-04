#!/usr/bin/env python3
"""
Simple Post-Training Visualizations for RLVAE
==============================================

Simplified script to generate key visualizations from a trained RLVAE checkpoint
without complex model reconstruction. Focuses on:
- Metric tensor analysis from checkpoint data
- Latent space visualizations using saved data
- Key performance metrics
- Model parameter analysis
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

# Setup paths
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplePostTrainingVisualizer:
    """Simple post-training visualization manager focused on key insights."""
    
    def __init__(self, checkpoint_path: str):
        """Initialize with checkpoint path."""
        self.checkpoint_path = checkpoint_path
        self.checkpoint = None
        self.output_dir = Path("outputs/simple_post_viz") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📊 Simple post-training visualizer initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def load_checkpoint(self) -> None:
        """Load and analyze checkpoint."""
        logger.info(f"🔄 Loading checkpoint from {self.checkpoint_path}")
        
        try:
            self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
            logger.info(f"✅ Checkpoint loaded successfully")
            
            # Log basic info
            logger.info(f"📊 Checkpoint keys: {list(self.checkpoint.keys())}")
            if 'epoch' in self.checkpoint:
                logger.info(f"🎯 Epoch: {self.checkpoint['epoch']}")
            if 'hyper_parameters' in self.checkpoint:
                hparams = self.checkpoint['hyper_parameters']
                logger.info(f"🔧 Model: latent_dim={hparams.get('latent_dim', 'N/A')}, n_flows={hparams.get('n_flows', 'N/A')}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            raise
    
    def analyze_model_parameters(self) -> Dict[str, Any]:
        """Analyze model parameters from checkpoint."""
        logger.info("🔍 Analyzing model parameters")
        
        analysis = {}
        
        if 'state_dict' not in self.checkpoint:
            logger.warning("⚠️ No state_dict found in checkpoint")
            return analysis
        
        state_dict = self.checkpoint['state_dict']
        
        # Count parameters by component
        param_counts = {}
        param_sizes = {}
        
        for name, param in state_dict.items():
            component = name.split('.')[0] if '.' in name else 'other'
            
            if component not in param_counts:
                param_counts[component] = 0
                param_sizes[component] = 0
            
            param_counts[component] += 1
            param_sizes[component] += param.numel()
        
        analysis['param_counts'] = param_counts
        analysis['param_sizes'] = param_sizes
        analysis['total_params'] = sum(param_sizes.values())
        
        # Analyze metric tensor if available
        metric_params = {k: v for k, v in state_dict.items() if 'metric' in k.lower()}
        if metric_params:
            analysis['metric_params'] = {}
            for name, param in metric_params.items():
                analysis['metric_params'][name] = {
                    'shape': list(param.shape),
                    'size': param.numel(),
                    'dtype': str(param.dtype)
                }
        
        logger.info(f"✅ Parameter analysis complete: {analysis['total_params']} total parameters")
        return analysis
    
    def visualize_metric_tensor_data(self) -> None:
        """Visualize metric tensor data from checkpoint."""
        logger.info("🎨 Visualizing metric tensor data")
        
        if 'state_dict' not in self.checkpoint:
            logger.warning("⚠️ No state_dict found")
            return
        
        state_dict = self.checkpoint['state_dict']
        
        # Find metric-related parameters
        metric_data = {}
        for name, param in state_dict.items():
            if 'metric' in name.lower():
                metric_data[name] = param.cpu().numpy()
        
        if not metric_data:
            logger.warning("⚠️ No metric data found in checkpoint")
            return
        
        # Create visualization
        n_plots = len(metric_data)
        if n_plots == 0:
            return
        
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        fig.suptitle('Metric Tensor Analysis from Checkpoint', fontsize=16, fontweight='bold')
        
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, (name, data) in enumerate(metric_data.items()):
            ax = axes[i] if i < len(axes) else axes[-1]
            
            try:
                if len(data.shape) == 2:
                    # 2D data - show as heatmap
                    im = ax.imshow(data, cmap='viridis', aspect='auto')
                    ax.set_title(f'{name}\n{data.shape}')
                    plt.colorbar(im, ax=ax)
                    
                elif len(data.shape) == 3:
                    # 3D data - show statistics
                    mean_data = np.mean(data, axis=0)
                    im = ax.imshow(mean_data, cmap='viridis', aspect='auto')
                    ax.set_title(f'{name} (mean across axis 0)\n{data.shape}')
                    plt.colorbar(im, ax=ax)
                    
                elif len(data.shape) == 1:
                    # 1D data - histogram
                    ax.hist(data, bins=50, alpha=0.7)
                    ax.set_title(f'{name}\n{data.shape}')
                    ax.set_ylabel('Frequency')
                    
                else:
                    # Other shapes - show flattened histogram
                    ax.hist(data.flatten(), bins=50, alpha=0.7)
                    ax.set_title(f'{name} (flattened)\n{data.shape}')
                    ax.set_ylabel('Frequency')
                    
            except Exception as e:
                ax.text(0.5, 0.5, f'Error visualizing\n{name}:\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{name} (error)')
        
        # Hide unused subplots
        for i in range(len(metric_data), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save
        metric_path = self.output_dir / 'metric_tensor_data.png'
        plt.savefig(metric_path, dpi=300, bbox_inches='tight')
        
        if wandb.run:
            wandb.log({"simple_viz/metric_tensor_data": wandb.Image(str(metric_path))})
        
        plt.close()
        logger.info(f"✅ Metric tensor visualization saved to {metric_path}")
    
    def visualize_parameter_distribution(self, analysis: Dict[str, Any]) -> None:
        """Visualize parameter distribution analysis."""
        logger.info("🎨 Creating parameter distribution visualization")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Parameter Analysis', fontsize=16, fontweight='bold')
        
        # Parameter counts by component
        if 'param_counts' in analysis:
            components = list(analysis['param_counts'].keys())
            counts = list(analysis['param_counts'].values())
            
            ax1.bar(components, counts)
            ax1.set_title('Parameter Count by Component')
            ax1.set_ylabel('Number of Parameters')
            ax1.tick_params(axis='x', rotation=45)
        
        # Parameter sizes by component
        if 'param_sizes' in analysis:
            components = list(analysis['param_sizes'].keys())
            sizes = list(analysis['param_sizes'].values())
            
            ax2.bar(components, sizes)
            ax2.set_title('Parameter Size by Component')
            ax2.set_ylabel('Total Parameters')
            ax2.tick_params(axis='x', rotation=45)
            ax2.set_yscale('log')
        
        # Total parameter breakdown pie chart
        if 'param_sizes' in analysis:
            ax3.pie(sizes, labels=components, autopct='%1.1f%%')
            ax3.set_title('Parameter Distribution')
        
        # Model configuration text
        if 'hyper_parameters' in self.checkpoint:
            hparams = self.checkpoint['hyper_parameters']
            config_text = []
            
            for key in ['latent_dim', 'n_flows', 'beta', 'riemannian_beta', 'flow_hidden_size']:
                if key in hparams:
                    config_text.append(f"{key}: {hparams[key]}")
            
            if 'total_params' in analysis:
                config_text.append(f"total_params: {analysis['total_params']:,}")
            
            ax4.text(0.1, 0.9, '\n'.join(config_text), transform=ax4.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            ax4.set_title('Model Configuration')
            ax4.axis('off')
        
        plt.tight_layout()
        
        # Save
        param_path = self.output_dir / 'parameter_analysis.png'
        plt.savefig(param_path, dpi=300, bbox_inches='tight')
        
        if wandb.run:
            wandb.log({"simple_viz/parameter_analysis": wandb.Image(str(param_path))})
        
        plt.close()
        logger.info(f"✅ Parameter analysis visualization saved to {param_path}")
    
    def analyze_training_history(self) -> None:
        """Analyze training history if available."""
        logger.info("📈 Analyzing training history")
        
        # Look for training metrics in checkpoint
        training_metrics = {}
        
        if 'lr_schedulers' in self.checkpoint:
            training_metrics['lr_schedulers'] = self.checkpoint['lr_schedulers']
        
        if 'optimizer_states' in self.checkpoint:
            training_metrics['optimizer_states'] = len(self.checkpoint['optimizer_states'])
        
        if 'epoch' in self.checkpoint:
            training_metrics['final_epoch'] = self.checkpoint['epoch']
        
        # Create training summary
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('Training Summary', fontsize=16, fontweight='bold')
        
        summary_text = []
        summary_text.append(f"Checkpoint: {Path(self.checkpoint_path).name}")
        
        if 'epoch' in self.checkpoint:
            summary_text.append(f"Final Epoch: {self.checkpoint['epoch']}")
        
        if 'hyper_parameters' in self.checkpoint:
            hparams = self.checkpoint['hyper_parameters']
            summary_text.append(f"Latent Dim: {hparams.get('latent_dim', 'N/A')}")
            summary_text.append(f"Flows: {hparams.get('n_flows', 'N/A')}")
            summary_text.append(f"Beta: {hparams.get('beta', 'N/A')}")
            summary_text.append(f"Riemannian Beta: {hparams.get('riemannian_beta', 'N/A')}")
        
        # Add checkpoint file info
        checkpoint_size = os.path.getsize(self.checkpoint_path) / (1024 * 1024)  # MB
        summary_text.append(f"Checkpoint Size: {checkpoint_size:.1f} MB")
        
        ax.text(0.1, 0.9, '\n'.join(summary_text), transform=ax.transAxes, 
               fontsize=12, verticalalignment='top', fontfamily='monospace')
        ax.set_title('Training Information')
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save
        history_path = self.output_dir / 'training_summary.png'
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        
        if wandb.run:
            wandb.log({"simple_viz/training_summary": wandb.Image(str(history_path))})
        
        plt.close()
        logger.info(f"✅ Training summary saved to {history_path}")
    
    def create_comprehensive_summary(self, analysis: Dict[str, Any]) -> None:
        """Create a comprehensive summary report."""
        logger.info("📋 Creating comprehensive summary")
        
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('RLVAE Model Analysis Summary', fontsize=20, fontweight='bold')
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
        
        # Model info (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        model_info = []
        model_info.append("🔧 Model Configuration")
        model_info.append("-" * 25)
        
        if 'hyper_parameters' in self.checkpoint:
            hparams = self.checkpoint['hyper_parameters']
            model_info.append(f"Latent Dim: {hparams.get('latent_dim', 'N/A')}")
            model_info.append(f"Flows: {hparams.get('n_flows', 'N/A')}")
            model_info.append(f"Beta: {hparams.get('beta', 'N/A')}")
            model_info.append(f"R-Beta: {hparams.get('riemannian_beta', 'N/A')}")
        
        if 'total_params' in analysis:
            model_info.append(f"Total Params: {analysis['total_params']:,}")
        
        ax1.text(0.05, 0.95, '\n'.join(model_info), transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax1.set_title('Model Configuration')
        ax1.axis('off')
        
        # Training info (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        training_info = []
        training_info.append("📈 Training Information")
        training_info.append("-" * 25)
        training_info.append(f"Checkpoint: {Path(self.checkpoint_path).name}")
        
        if 'epoch' in self.checkpoint:
            training_info.append(f"Final Epoch: {self.checkpoint['epoch']}")
        
        checkpoint_size = os.path.getsize(self.checkpoint_path) / (1024 * 1024)
        training_info.append(f"Size: {checkpoint_size:.1f} MB")
        training_info.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        
        ax2.text(0.05, 0.95, '\n'.join(training_info), transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax2.set_title('Training Summary')
        ax2.axis('off')
        
        # Key findings (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        findings = []
        findings.append("🎯 Key Findings")
        findings.append("-" * 25)
        findings.append("✅ Model loaded successfully")
        findings.append("✅ Checkpoint analysis complete")
        
        if 'metric_params' in analysis:
            findings.append(f"✅ {len(analysis['metric_params'])} metric components")
        
        if 'param_counts' in analysis:
            findings.append(f"✅ {len(analysis['param_counts'])} model components")
        
        ax3.text(0.05, 0.95, '\n'.join(findings), transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax3.set_title('Analysis Status')
        ax3.axis('off')
        
        # Parameter distribution (middle row)
        if 'param_sizes' in analysis:
            ax4 = fig.add_subplot(gs[1, :])
            components = list(analysis['param_sizes'].keys())
            sizes = list(analysis['param_sizes'].values())
            
            bars = ax4.bar(components, sizes, color=plt.cm.viridis(np.linspace(0, 1, len(components))))
            ax4.set_title('Parameter Distribution by Component')
            ax4.set_ylabel('Number of Parameters')
            ax4.set_yscale('log')
            
            # Add value labels on bars
            for bar, size in zip(bars, sizes):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{size:,}', ha='center', va='bottom', rotation=90)
        
        # Analysis metadata (bottom)
        ax5 = fig.add_subplot(gs[2, :])
        metadata = []
        metadata.append(f"📊 Analysis Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        metadata.append(f"📁 Output Directory: {self.output_dir}")
        metadata.append(f"🔗 WandB Project: rlvae-post-training-analysis")
        metadata.append("🎨 Visualizations: Metric tensor data, parameter analysis, training summary")
        
        ax5.text(0.05, 0.8, '\n'.join(metadata), transform=ax5.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax5.set_title('Analysis Metadata')
        ax5.axis('off')
        
        plt.tight_layout()
        
        # Save
        summary_path = self.output_dir / 'comprehensive_summary.png'
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        
        if wandb.run:
            wandb.log({"simple_viz/comprehensive_summary": wandb.Image(str(summary_path))})
        
        plt.close()
        logger.info(f"✅ Comprehensive summary saved to {summary_path}")
    
    def run_all_visualizations(self) -> None:
        """Run all simple visualizations."""
        logger.info("🚀 Starting simple post-training analysis")
        
        # Initialize WandB
        try:
            wandb.init(
                project="rlvae-post-training-analysis",
                name=f"simple_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "checkpoint_path": self.checkpoint_path,
                    "analysis_type": "simple_checkpoint_analysis"
                }
            )
        except Exception as e:
            logger.warning(f"⚠️ WandB initialization failed: {e}")
        
        # Load checkpoint
        self.load_checkpoint()
        
        # Run analyses
        analysis = self.analyze_model_parameters()
        
        # Create visualizations
        self.visualize_metric_tensor_data()
        self.visualize_parameter_distribution(analysis)
        self.analyze_training_history()
        self.create_comprehensive_summary(analysis)
        
        logger.info("🎉 Simple post-training analysis completed!")
        logger.info(f"📁 Results saved in: {self.output_dir}")
        
        if wandb.run:
            wandb.finish()


def main():
    """Main execution function."""
    checkpoint_path = "outputs/checkpoints/epoch=197-val_loss=6.402.ckpt"
    
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        logger.info("🔍 Available checkpoints:")
        checkpoint_dir = Path(checkpoint_path).parent
        if checkpoint_dir.exists():
            for ckpt in sorted(checkpoint_dir.glob("*.ckpt")):
                logger.info(f"  - {ckpt.name}")
        return
    
    # Run analysis
    visualizer = SimplePostTrainingVisualizer(checkpoint_path)
    visualizer.run_all_visualizations()


if __name__ == "__main__":
    main() 