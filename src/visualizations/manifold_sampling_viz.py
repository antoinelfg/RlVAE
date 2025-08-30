#!/usr/bin/env python3
"""
Manifold Sampling Visualization Component
=========================================

Specialized visualization component for manifold sampling analysis.
Integrates with the RlVAE visualization system and provides WandB logging.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..models.components.manifold_sampler import ManifoldSampler
from ..models.components.native_inverse_metric import NativeInverseMetricTensor


class ManifoldSamplingVisualizer:
    """
    Manifold sampling visualization component.
    
    Provides comprehensive visualization and logging capabilities for 
    manifold sampling analysis within the RlVAE pipeline.
    """
    
    def __init__(
        self,
        manifold_sampler: ManifoldSampler,
        enable_wandb: bool = True,
        save_local: bool = True,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the manifold sampling visualizer.
        
        Args:
            manifold_sampler: ManifoldSampler instance
            enable_wandb: Whether to log to WandB
            save_local: Whether to save plots locally
            output_dir: Local output directory for plots
        """
        self.manifold_sampler = manifold_sampler
        self.enable_wandb = enable_wandb and WANDB_AVAILABLE
        self.save_local = save_local
        self.output_dir = Path(output_dir) if output_dir else Path("./outputs/manifold_sampling")
        
        if self.save_local:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎨 ManifoldSamplingVisualizer initialized")
        print(f"   WandB enabled: {self.enable_wandb}")
        print(f"   Local save: {self.save_local}")
        if self.save_local:
            print(f"   Output dir: {self.output_dir}")
    
    def create_stage1_analysis(
        self,
        model,
        latent_data: torch.Tensor,
        epoch: int,
        stage: str = "vanilla_vae"
    ) -> Dict[str, Any]:
        """
        Create manifold sampling analysis for Stage 1 (Vanilla VAE).
        
        Args:
            model: Trained VAE model
            latent_data: Latent representations from the model
            epoch: Current epoch number
            stage: Training stage identifier
            
        Returns:
            Dictionary with analysis results and paths
        """
        print(f"🔍 Creating Stage 1 manifold sampling analysis (epoch {epoch})")
        
        # Create native inverse metric from model
        native_metric = NativeInverseMetricTensor.from_model_data(
            model=model,
            latent_data=latent_data,
            n_centroids=50,
            temperature=2.0,
            regularization=1e-4
        )
        
        # Create manifold sampler with native metric
        stage1_sampler = ManifoldSampler(
            metric_tensor=native_metric,
            method="combined",
            step_size_base=0.25,
            exploration_ratio=0.6
        )
        
        # Generate samples
        samples = stage1_sampler.sample(method="combined")
        
        # Create visualization
        title = f"Stage 1: Manifold Sampling Analysis (Epoch {epoch})\nNative G⁻¹ Metric from Vanilla VAE"
        
        save_path = None
        if self.save_local:
            save_path = self.output_dir / f"stage1_manifold_analysis_epoch_{epoch}.png"
        
        fig = stage1_sampler.create_visualization(
            samples=samples,
            latent_data=latent_data,
            title=title,
            save_path=str(save_path) if save_path else None
        )
        
        # Prepare results
        results = {
            'samples': samples,
            'native_metric': native_metric,
            'figure': fig,
            'save_path': save_path
        }
        
        # Log to WandB
        if self.enable_wandb and wandb.run is not None:
            self._log_to_wandb(samples, fig, epoch, f"{stage}/manifold_sampling")
        
        plt.close(fig)  # Clean up
        
        print(f"✅ Stage 1 manifold analysis completed for epoch {epoch}")
        return results
    
    def create_stage2_evolution(
        self,
        model,
        latent_data: torch.Tensor,
        epoch: int,
        stage: str = "rlvae_training"
    ) -> Dict[str, Any]:
        """
        Create manifold sampling evolution analysis for Stage 2 (RlVAE training).
        
        Args:
            model: RlVAE model with native G⁻¹ metric
            latent_data: Current latent representations
            epoch: Current epoch number
            stage: Training stage identifier
            
        Returns:
            Dictionary with analysis results and paths
        """
        print(f"🔍 Creating Stage 2 manifold evolution analysis (epoch {epoch})")
        
        # Extract native metric from model (assuming it has been integrated)
        if hasattr(model, 'metric_tensor') and hasattr(model.metric_tensor, 'centroids'):
            native_metric = model.metric_tensor
        else:
            # Fallback: create from current model state
            native_metric = NativeInverseMetricTensor.from_model_data(
                model=model,
                latent_data=latent_data,
                n_centroids=50,
                temperature=2.0,
                regularization=1e-4
            )
        
        # Create manifold sampler with current metric
        stage2_sampler = ManifoldSampler(
            metric_tensor=native_metric,
            method="combined",
            step_size_base=0.25,
            exploration_ratio=0.6
        )
        
        # Generate samples
        samples = stage2_sampler.sample(method="combined")
        
        # Create evolution visualization
        title = f"Stage 2: Manifold Evolution (Epoch {epoch})\nRlVAE Training with Native G⁻¹"
        
        save_path = None
        if self.save_local:
            save_path = self.output_dir / f"stage2_evolution_epoch_{epoch}.png"
        
        fig = stage2_sampler.create_visualization(
            samples=samples,
            latent_data=latent_data,
            title=title,
            save_path=str(save_path) if save_path else None
        )
        
        # Compute evolution metrics
        evolution_metrics = self._compute_evolution_metrics(samples, native_metric)
        
        # Prepare results
        results = {
            'samples': samples,
            'native_metric': native_metric,
            'figure': fig,
            'save_path': save_path,
            'evolution_metrics': evolution_metrics
        }
        
        # Log to WandB with evolution metrics
        if self.enable_wandb and wandb.run is not None:
            self._log_evolution_to_wandb(samples, fig, evolution_metrics, epoch, f"{stage}/manifold_evolution")
        
        plt.close(fig)  # Clean up
        
        print(f"✅ Stage 2 evolution analysis completed for epoch {epoch}")
        return results
    
    def _compute_evolution_metrics(
        self, 
        samples: Dict[str, torch.Tensor], 
        native_metric: NativeInverseMetricTensor
    ) -> Dict[str, float]:
        """Compute quantitative metrics for manifold evolution."""
        metrics = {}
        
        # Combine all samples
        all_samples = []
        for key, sample_tensor in samples.items():
            if sample_tensor is not None:
                all_samples.append(sample_tensor)
        
        if all_samples:
            combined_samples = torch.cat(all_samples, dim=0)
            
            # Compute metric determinant statistics
            with torch.no_grad():
                _, log_det_values = native_metric(combined_samples)
                det_values = torch.exp(log_det_values)
            
            metrics['mean_determinant'] = det_values.mean().item()
            metrics['std_determinant'] = det_values.std().item()
            metrics['min_determinant'] = det_values.min().item()
            metrics['max_determinant'] = det_values.max().item()
            
            # Compute spatial coverage
            metrics['spatial_extent_x'] = combined_samples[:, 0].max().item() - combined_samples[:, 0].min().item()
            metrics['spatial_extent_y'] = combined_samples[:, 1].max().item() - combined_samples[:, 1].min().item()
            metrics['total_samples'] = len(combined_samples)
            
            # Compute sample type distribution
            for key, sample_tensor in samples.items():
                if sample_tensor is not None:
                    metrics[f'{key}_count'] = len(sample_tensor)
        
        return metrics
    
    def _log_to_wandb(
        self, 
        samples: Dict[str, torch.Tensor], 
        fig: plt.Figure, 
        epoch: int, 
        prefix: str
    ):
        """Log basic manifold sampling to WandB."""
        log_dict = {}
        
        # Log figure
        log_dict[f"{prefix}/visualization"] = wandb.Image(fig)
        
        # Log sample counts
        for key, sample_tensor in samples.items():
            if sample_tensor is not None:
                log_dict[f"{prefix}/{key}_count"] = len(sample_tensor)
                
                # Log sample distribution statistics
                samples_np = sample_tensor.detach().cpu().numpy()
                log_dict[f"{prefix}/{key}_mean_x"] = np.mean(samples_np[:, 0])
                log_dict[f"{prefix}/{key}_mean_y"] = np.mean(samples_np[:, 1])
                log_dict[f"{prefix}/{key}_std_x"] = np.std(samples_np[:, 0])
                log_dict[f"{prefix}/{key}_std_y"] = np.std(samples_np[:, 1])
        
        wandb.log(log_dict, step=epoch)
    
    def _log_evolution_to_wandb(
        self, 
        samples: Dict[str, torch.Tensor], 
        fig: plt.Figure, 
        evolution_metrics: Dict[str, float],
        epoch: int, 
        prefix: str
    ):
        """Log manifold evolution with metrics to WandB."""
        log_dict = {}
        
        # Log figure
        log_dict[f"{prefix}/visualization"] = wandb.Image(fig)
        
        # Log evolution metrics
        for key, value in evolution_metrics.items():
            log_dict[f"{prefix}/metrics/{key}"] = value
        
        # Log sample-specific metrics (same as basic logging)
        for key, sample_tensor in samples.items():
            if sample_tensor is not None:
                log_dict[f"{prefix}/{key}_count"] = len(sample_tensor)
                
                samples_np = sample_tensor.detach().cpu().numpy()
                log_dict[f"{prefix}/{key}_mean_x"] = np.mean(samples_np[:, 0])
                log_dict[f"{prefix}/{key}_mean_y"] = np.mean(samples_np[:, 1])
                log_dict[f"{prefix}/{key}_std_x"] = np.std(samples_np[:, 0])
                log_dict[f"{prefix}/{key}_std_y"] = np.std(samples_np[:, 1])
        
        wandb.log(log_dict, step=epoch)
    
    def create_comparison_analysis(
        self,
        stage1_results: Dict[str, Any],
        stage2_results: Dict[str, Any],
        final_epoch: int
    ) -> Dict[str, Any]:
        """
        Create comparative analysis between Stage 1 and Stage 2.
        
        Args:
            stage1_results: Results from Stage 1 analysis
            stage2_results: Results from Stage 2 analysis
            final_epoch: Final epoch number
            
        Returns:
            Dictionary with comparison results
        """
        print(f"🔍 Creating Stage 1 vs Stage 2 comparison analysis")
        
        # Create side-by-side comparison figure
        fig, axes = plt.subplots(2, 6, figsize=(30, 20))
        fig.suptitle(f"Manifold Sampling Evolution: Stage 1 → Stage 2 (Final Epoch {final_epoch})", 
                     fontsize=20, fontweight='bold')
        
        # Extract samples and metrics
        stage1_samples = stage1_results['samples']
        stage2_samples = stage2_results['samples']
        stage1_metric = stage1_results['native_metric']
        stage2_metric = stage2_results['native_metric']
        
        # Plot Stage 1 results in top row
        self._plot_comparison_row(axes[0], stage1_samples, stage1_metric, "Stage 1: Vanilla VAE")
        
        # Plot Stage 2 results in bottom row
        self._plot_comparison_row(axes[1], stage2_samples, stage2_metric, "Stage 2: RlVAE Training")
        
        plt.tight_layout()
        
        # Save comparison
        save_path = None
        if self.save_local:
            save_path = self.output_dir / f"stage_comparison_final_epoch_{final_epoch}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Log to WandB
        if self.enable_wandb and wandb.run is not None:
            wandb.log({
                "final_analysis/stage_comparison": wandb.Image(fig),
                "final_analysis/final_epoch": final_epoch
            })
        
        plt.close(fig)
        
        results = {
            'comparison_figure': fig,
            'save_path': save_path,
            'stage1_samples': stage1_samples,
            'stage2_samples': stage2_samples
        }
        
        print(f"✅ Comparison analysis completed")
        return results
    
    def _plot_comparison_row(
        self, 
        axes, 
        samples: Dict[str, torch.Tensor], 
        metric: NativeInverseMetricTensor, 
        title_prefix: str
    ):
        """Plot a single row of the comparison analysis."""
        # Get determinant grid
        sampler = ManifoldSampler(metric_tensor=metric)
        X, Y, det_grid = sampler.compute_determinant_grid()
        
        # Extract sample data
        guided_paths = samples.get('guided_paths')
        explorations = samples.get('explorations') 
        connections = samples.get('connections')
        
        # Convert to numpy
        if guided_paths is not None:
            guided_paths = guided_paths.detach().cpu().numpy()
        if explorations is not None:
            explorations = explorations.detach().cpu().numpy()
        if connections is not None:
            connections = connections.detach().cpu().numpy()
        
        centroids = metric.centroids.detach().cpu().numpy()
        
        # Plot 1: Guided paths
        ax1 = axes[0]
        contour1 = ax1.contourf(X, Y, det_grid, levels=20, cmap='viridis', alpha=0.7)
        ax1.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=100, zorder=5)
        if guided_paths is not None:
            ax1.scatter(guided_paths[:, 0], guided_paths[:, 1], c='green', alpha=0.5, s=8)
        ax1.set_title(f"{title_prefix}\nGuided Paths")
        plt.colorbar(contour1, ax=ax1)
        
        # Plot 2: Explorations
        ax2 = axes[1]
        contour2 = ax2.contourf(X, Y, det_grid, levels=20, cmap='viridis', alpha=0.7)
        ax2.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=100, zorder=5)
        if explorations is not None:
            ax2.scatter(explorations[:, 0], explorations[:, 1], c='blue', alpha=0.5, s=8)
        ax2.set_title(f"{title_prefix}\nExplorations")
        plt.colorbar(contour2, ax=ax2)
        
        # Plot 3: Connections
        ax3 = axes[2]
        contour3 = ax3.contourf(X, Y, det_grid, levels=20, cmap='viridis', alpha=0.7)
        ax3.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=100, zorder=5)
        if connections is not None:
            ax3.scatter(connections[:, 0], connections[:, 1], c='purple', alpha=0.5, s=8)
        ax3.set_title(f"{title_prefix}\nConnections")
        plt.colorbar(contour3, ax=ax3)
        
        # Plot 4: Level lines
        ax4 = axes[3]
        contour4 = ax4.contour(X, Y, det_grid, levels=15, colors='black', alpha=0.6)
        ax4.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=100, zorder=5)
        if guided_paths is not None:
            ax4.scatter(guided_paths[:, 0], guided_paths[:, 1], c='green', alpha=0.3, s=4)
        if explorations is not None:
            ax4.scatter(explorations[:, 0], explorations[:, 1], c='blue', alpha=0.3, s=4)
        if connections is not None:
            ax4.scatter(connections[:, 0], connections[:, 1], c='purple', alpha=0.3, s=4)
        ax4.set_title(f"{title_prefix}\nLevel Lines")
        
        # Plot 5: Density
        ax5 = axes[4]
        contour5 = ax5.contourf(X, Y, det_grid, levels=20, cmap='viridis', alpha=0.7)
        all_samples = []
        if guided_paths is not None:
            all_samples.append(guided_paths)
        if explorations is not None:
            all_samples.append(explorations)
        if connections is not None:
            all_samples.append(connections)
        if all_samples:
            combined_samples = np.vstack(all_samples)
            ax5.hist2d(combined_samples[:, 0], combined_samples[:, 1], bins=30, cmap='Reds', alpha=0.8)
        ax5.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=100, zorder=5)
        ax5.set_title(f"{title_prefix}\nSampling Density")
        plt.colorbar(contour5, ax=ax5)
        
        # Plot 6: Combined
        ax6 = axes[5]
        contour6 = ax6.contourf(X, Y, det_grid, levels=20, cmap='viridis', alpha=0.6)
        ax6.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=100, zorder=5)
        if guided_paths is not None:
            ax6.scatter(guided_paths[:, 0], guided_paths[:, 1], c='green', alpha=0.3, s=4, label='Guided')
        if explorations is not None:
            ax6.scatter(explorations[:, 0], explorations[:, 1], c='blue', alpha=0.3, s=4, label='Exploration')
        if connections is not None:
            ax6.scatter(connections[:, 0], connections[:, 1], c='purple', alpha=0.3, s=4, label='Connections')
        ax6.set_title(f"{title_prefix}\nCombined")
        ax6.legend()
        plt.colorbar(contour6, ax=ax6)