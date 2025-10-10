"""
Optimized WandB logging utilities for three-stage pipeline.
Provides clean, organized logging with stage-specific prefixes and reduced noise.
"""

import wandb
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import matplotlib.pyplot as plt
from omegaconf import DictConfig


class ThreeStageWandBLogger:
    """Optimized WandB logger for three-stage pipeline experiments."""
    
    def __init__(self, config: DictConfig, stage: str = None):
        """
        Initialize the logger with stage-specific settings.
        
        Args:
            config: Hydra configuration
            stage: Current stage ('stage_a', 'stage_b', 'stage_c', or None for pipeline-level)
        """
        self.config = config
        self.current_stage = stage
        self.step_counters = {'stage_a': 0, 'stage_b': 0, 'stage_c': 0, 'pipeline': 0}
        
        # Get logging settings
        self.use_stage_prefixes = getattr(config.wandb, 'use_stage_prefixes', True)
        self.reduce_frequency = getattr(config.wandb, 'reduce_logging_frequency', True)
        
        # Essential metrics only
        self.essential_metrics = {
            'stage_a': ['train_loss', 'val_loss', 'reconstruction_mse', 'kl_divergence'],
            'stage_b': ['metric_eigenvalues', 'centroids_quality', 'temperature_stability'],
            'stage_c': ['riemannian_kl', 'flow_loss', 'total_loss', 'convergence_rate']
        }
        
    def set_stage(self, stage: str):
        """Set the current stage for logging."""
        self.current_stage = stage
        
    def get_prefix(self) -> str:
        """Get the appropriate prefix for current stage."""
        if not self.use_stage_prefixes or not self.current_stage:
            return ""
        
        stage_prefixes = {
            'stage_a': 'stageA',
            'stage_b': 'stageB', 
            'stage_c': 'stageC'
        }
        return stage_prefixes.get(self.current_stage, "")
        
    def should_log(self, metric_name: str, epoch: int = None) -> bool:
        """Determine if a metric should be logged based on optimization settings."""
        if not self.reduce_frequency:
            return True
            
        # Always log essential metrics
        if self.current_stage and metric_name in self.essential_metrics.get(self.current_stage, []):
            return True
            
        # Reduce frequency for non-essential metrics
        if epoch is not None:
            # Log every 5 epochs for non-essential metrics
            return epoch % 5 == 0
            
        return True
        
    def log_metrics(self, metrics: Dict[str, Any], epoch: int = None, step: int = None):
        """Log metrics with stage-specific prefixes and filtering."""
        if not wandb.run:
            return
            
        prefix = self.get_prefix()
        filtered_metrics = {}
        
        for key, value in metrics.items():
            # Skip if we shouldn't log this metric
            if not self.should_log(key, epoch):
                continue
                
            # Add prefix if configured
            log_key = f"{prefix}/{key}" if prefix else key
            
            # Handle different value types
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    filtered_metrics[log_key] = value.item()
                else:
                    # Log as histogram for multi-element tensors
                    filtered_metrics[f"{log_key}_hist"] = wandb.Histogram(value.detach().cpu().numpy())
            elif isinstance(value, (int, float, np.number)):
                filtered_metrics[log_key] = float(value)
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    filtered_metrics[log_key] = float(value.item())
                else:
                    filtered_metrics[f"{log_key}_hist"] = wandb.Histogram(value)
            else:
                filtered_metrics[log_key] = value
                
        # Use appropriate step counter
        if step is None:
            step = self.step_counters.get(self.current_stage, 0)
            self.step_counters[self.current_stage] = step + 1
            
        if filtered_metrics:
            wandb.log(filtered_metrics, step=step)
            
    def log_image(self, image_path: str, caption: str, key: str, epoch: int = None):
        """Log an image with stage-specific prefix."""
        if not wandb.run or not Path(image_path).exists():
            return
            
        # Check if we should log this image
        if not self.should_log(key, epoch):
            return
            
        prefix = self.get_prefix()
        log_key = f"{prefix}/{key}" if prefix else key
        
        wandb.log({
            log_key: wandb.Image(image_path, caption=caption)
        }, step=self.step_counters.get(self.current_stage, 0))
        
    def log_stage_summary(self, stage: str, summary_metrics: Dict[str, Any]):
        """Log a summary at the end of each stage."""
        if not wandb.run:
            return
            
        prefix = f"summary/{stage}"
        summary_data = {}
        
        for key, value in summary_metrics.items():
            summary_data[f"{prefix}/{key}"] = value
            
        wandb.log(summary_data)
        
    def log_pipeline_progress(self, stage: str, progress: float, message: str = ""):
        """Log overall pipeline progress."""
        if not wandb.run:
            return
            
        wandb.log({
            "pipeline/current_stage": stage,
            "pipeline/progress": progress,
            "pipeline/status": message
        })
        
    def create_stage_artifact(self, artifact_path: str, artifact_name: str, 
                            artifact_type: str, description: str = ""):
        """Create and log a stage-specific artifact."""
        if not wandb.run or not Path(artifact_path).exists():
            return None
            
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            description=description
        )
        
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)
        
        return artifact
        
    def finalize_stage(self, stage: str, final_metrics: Dict[str, Any]):
        """Finalize logging for a stage."""
        # Log final summary
        self.log_stage_summary(stage, final_metrics)
        
        # Update pipeline progress
        progress_map = {'stage_a': 0.33, 'stage_b': 0.66, 'stage_c': 1.0}
        self.log_pipeline_progress(
            stage, 
            progress_map.get(stage, 0.0),
            f"Completed {stage}"
        )


def create_optimized_wandb_config(base_config: DictConfig) -> Dict[str, Any]:
    """Create an optimized WandB configuration for three-stage pipeline."""
    
    # Extract key information
    experiment_name = base_config.experiment.name
    data_name = base_config.data.name
    latent_dim = base_config.model.latent_dim
    
    # Create clean run name
    run_name = f"3stage_{experiment_name}_{data_name}_ld{latent_dim}"
    
    # Optimized config
    wandb_config = {
        'project': getattr(base_config.wandb, 'project', 'rlvae-three-stage-clean'),
        'name': run_name,
        'group': getattr(base_config.wandb, 'group', 'three_stage_experiments'),
        'job_type': 'pipeline',
        'tags': [
            'three-stage-pipeline',
            f'data-{data_name}',
            f'latent-{latent_dim}d',
            'optimized-logging'
        ],
        'config': {
            # Essential config only
            'experiment_type': 'three_stage_pipeline',
            'data_name': data_name,
            'latent_dim': latent_dim,
            'stage_a_epochs': base_config.experiment.stage_a.epochs,
            'stage_b_implementation': base_config.experiment.stage_b.implementation,
            'stage_c_epochs': base_config.experiment.stage_c.epochs,
        }
    }
    
    return wandb_config


def log_essential_training_metrics(logger: ThreeStageWandBLogger, 
                                 stage: str, epoch: int, 
                                 train_loss: float, val_loss: float = None,
                                 additional_metrics: Dict[str, Any] = None):
    """Log essential training metrics with reduced noise."""
    
    logger.set_stage(stage)
    
    metrics = {
        'train_loss': train_loss,
        'epoch': epoch
    }
    
    if val_loss is not None:
        metrics['val_loss'] = val_loss
        
    if additional_metrics:
        # Only include essential additional metrics
        essential_keys = logger.essential_metrics.get(stage, [])
        for key, value in additional_metrics.items():
            if key in essential_keys:
                metrics[key] = value
                
    logger.log_metrics(metrics, epoch=epoch)
