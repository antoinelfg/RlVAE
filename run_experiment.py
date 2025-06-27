#!/usr/bin/env python3
"""
RlVAE Experiment Runner
======================

Hydra-powered experiment runner for systematic comparison of Riemannian VAE variants.

Usage Examples:
--------------

1. Single experiment:
   python run_experiment.py

2. Quick development test:
   python run_experiment.py training=quick visualization=minimal

3. Compare models:
   python run_experiment.py experiment=comparison_study

4. Hyperparameter sweep:
   python run_experiment.py experiment=hyperparameter_sweep -m

5. Custom configuration:
   python run_experiment.py model=vanilla_vae training.n_epochs=50 wandb.mode=offline

6. Override specific parameters:
   python run_experiment.py model.riemannian_beta=10.0 training.learning_rate=1e-3
"""

import os
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir / "src"
lib_src_dir = src_dir / "lib" / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(lib_src_dir) not in sys.path:
    sys.path.insert(0, str(lib_src_dir))

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import wandb
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import datetime
from typing import Dict, Any, List, Optional

# Local imports
from models.modular_rlvae import ModularRiemannianFlowVAE, ModelFactory, MetricsCollector
from data.cyclic_dataset import CyclicSpritesDataModule
from training.lightning_trainer import LightningRlVAETrainer
from visualizations.manager import VisualizationManager


class ExperimentRunner:
    """Main experiment runner with Hydra configuration."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.results = {}
        
        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        print(f"🚀 RlVAE Experiment Runner")
        print(f"📅 Experiment: {config.experiment_name}")
        print(f"💻 Device: {self.device}")
        print(f"📁 Output: {self.output_dir}")
        
        # Set random seed for reproducibility
        if config.get('seed'):
            L.seed_everything(config.seed)
            print(f"🎲 Seed: {config.seed}")
    
    def run(self):
        """Run the experiment based on configuration."""
        experiment_type = self.config.experiment.type
        
        print(f"\n🧪 Running {experiment_type} experiment: {self.config.experiment.name}")
        
        if experiment_type == "single":
            self.run_single_experiment()
        elif experiment_type == "comparison":
            self.run_comparison_study()
        elif experiment_type == "sweep":
            self.run_hyperparameter_sweep()
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    def run_single_experiment(self):
        """Run a single experiment with current configuration."""
        
        # Initialize wandb
        wandb_logger = self._setup_wandb("single_run")
        
        # Create data module
        data_module = CyclicSpritesDataModule(self.config.data)
        data_module.setup("fit", self.config.training)
        
        # Create model
        model_wrapper = LightningRlVAETrainer(
            self.config,
            data_module=data_module
        )
        
        # Setup trainer
        trainer = self._create_trainer(wandb_logger)
        
        # Train
        print(f"🚀 Starting training...")
        trainer.fit(model_wrapper, data_module)
        
        # Test
        print(f"🧪 Running test...")
        test_results = trainer.test(model_wrapper, data_module)
        
        # Save results
        self.results = {
            'test_results': test_results[0] if test_results else {},
            'model_summary': model_wrapper.model.get_model_summary()
        }
        
        print(f"✅ Single experiment completed!")
        self._save_results()
    
    def run_comparison_study(self):
        """Run comparison between multiple model variants."""
        
        print(f"🔬 Running comparison study...")
        models_to_compare = self.config.experiment.models
        comparison_metrics = self.config.experiment.comparison_metrics
        
        metrics_collector = MetricsCollector()
        all_results = {}
        
        for model_name in models_to_compare:
            print(f"\n🎯 Training model: {model_name}")
            
            # Create model-specific config
            model_config = self._create_model_config(model_name)
            
            # Setup wandb for this model
            wandb_logger = self._setup_wandb(f"comparison_{model_name}")
            
            # Create data module
            data_module = CyclicSpritesDataModule(model_config.data)
            data_module.setup("fit", model_config.training)
            
            # Create model wrapper
            model_wrapper = LightningRlVAETrainer(
                model_config,
                data_module=data_module
            )
            
            # Setup trainer
            trainer = self._create_trainer(wandb_logger)
            
            # Train
            trainer.fit(model_wrapper, data_module)
            
            # Test
            test_results = trainer.test(model_wrapper, data_module)
            
            # Collect metrics
            if test_results:
                metrics = self._extract_comparison_metrics(test_results[0], comparison_metrics)
                metrics_collector.add_model_metrics(model_name, metrics)
            
            # Store results
            all_results[model_name] = {
                'test_results': test_results[0] if test_results else {},
                'model_summary': model_wrapper.model.get_model_summary()
            }
            
            # Finish this wandb run
            wandb.finish()
        
        # Create comparison analysis
        print(f"\n📊 Analyzing comparison results...")
        comparison_summary = metrics_collector.get_comparison_summary()
        
        # Start final wandb run for comparison
        wandb_logger = self._setup_wandb("comparison_analysis")
        metrics_collector.log_comparison_to_wandb()
        
        self.results = {
            'comparison_summary': comparison_summary,
            'individual_results': all_results,
            'comparison_metrics': comparison_metrics
        }
        
        print(f"✅ Comparison study completed!")
        self._save_results()
        self._print_comparison_summary(comparison_summary)
    
    def run_hyperparameter_sweep(self):
        """Run hyperparameter sweep (placeholder for Hydra sweep)."""
        print(f"🌊 Hyperparameter sweep mode")
        print(f"⚠️ This should be run with Hydra multirun (-m flag)")
        print(f"Example: python run_experiment.py experiment=hyperparameter_sweep -m")
        
        # For individual sweep runs, just run single experiment
        self.run_single_experiment()
    
    def _create_model_config(self, model_name: str) -> DictConfig:
        """Create configuration for a specific model variant."""
        config = OmegaConf.structured(self.config)
        
        # Apply model-specific overrides
        if model_name == 'vanilla_vae':
            config.model.n_flows = 0
            config.model.riemannian_beta = 0.0
            config.model.posterior.type = 'gaussian'
            config.model.sampling.use_riemannian = False
            config.model.sampling.method = 'standard'
            config.model.loop.mode = 'open'
            config.model.loop.penalty = 0.0
        elif model_name == 'riemannian_flow_vae':
            # Use default Riemannian configuration
            pass
        
        # Apply experiment overrides
        if hasattr(self.config.experiment, 'training_override'):
            config.training.update(self.config.experiment.training_override)
        
        if hasattr(self.config.experiment, 'visualization_override'):
            config.visualization.update(self.config.experiment.visualization_override)
        
        return config
    
    def _setup_wandb(self, run_name: str) -> Optional[WandbLogger]:
        """Setup Weights & Biases logging."""
        if self.config.wandb.mode == "disabled":
            return None
        
        # Create unique run name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        full_run_name = f"{run_name}_{timestamp}"
        
        wandb_logger = WandbLogger(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=full_run_name,
            mode=self.config.wandb.mode,
            tags=self.config.wandb.get('tags', []),
            config=OmegaConf.to_container(self.config, resolve=True)
        )
        
        return wandb_logger
    
    def _create_trainer(self, wandb_logger) -> L.Trainer:
        """Create Lightning trainer."""
        callbacks = []
        
        # Early stopping
        if hasattr(self.config.training, 'early_stopping'):
            early_stop = EarlyStopping(
                monitor=self.config.training.early_stopping.monitor,
                patience=self.config.training.early_stopping.patience,
                mode=self.config.training.early_stopping.mode
            )
            callbacks.append(early_stop)
        
        # Model checkpointing
        checkpoint = ModelCheckpoint(
            monitor=self.config.training.logging.monitor,
            save_top_k=self.config.training.logging.save_top_k,
            mode=self.config.training.logging.mode,
            dirpath=self.output_dir / "checkpoints",
            filename="{epoch:02d}-{val_loss:.3f}"
        )
        callbacks.append(checkpoint)
        
        # Create trainer with new config structure
        trainer_kwargs = {
            'max_epochs': self.config.training.trainer.max_epochs,
            'accelerator': self.config.training.trainer.accelerator,
            'devices': self.config.training.trainer.devices,
            'strategy': self.config.training.trainer.strategy,
            'precision': self.config.training.trainer.precision,
            'log_every_n_steps': self.config.training.trainer.log_every_n_steps,
            'val_check_interval': self.config.training.trainer.val_check_interval,
            'num_sanity_val_steps': self.config.training.trainer.num_sanity_val_steps,
            'enable_progress_bar': self.config.training.trainer.enable_progress_bar,
            'enable_model_summary': self.config.training.trainer.enable_model_summary,
            'deterministic': self.config.training.trainer.deterministic,
            'logger': wandb_logger,
            'callbacks': callbacks,
        }
        
        trainer = L.Trainer(**trainer_kwargs)
        
        return trainer
    
    def _extract_comparison_metrics(self, test_results: Dict, metric_names: List[str]) -> Dict[str, float]:
        """Extract specific metrics for comparison."""
        extracted = {}
        
        for metric_name in metric_names:
            if metric_name in test_results:
                value = test_results[metric_name]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                extracted[metric_name] = value
        
        return extracted
    
    def _save_results(self):
        """Save experiment results."""
        results_path = self.output_dir / "results.yaml"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            OmegaConf.save(OmegaConf.create(self.results), f)
        
        print(f"💾 Results saved to: {results_path}")
    
    def _print_comparison_summary(self, summary: Dict[str, Any]):
        """Print comparison summary to console."""
        print(f"\n📊 COMPARISON SUMMARY")
        print("=" * 60)
        
        for model_name, metrics in summary.items():
            print(f"\n🎯 {model_name.upper()}:")
            for metric_name, value in metrics.items():
                if metric_name.endswith('_final'):
                    clean_name = metric_name.replace('_final', '')
                    print(f"   {clean_name}: {value:.4f}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    
    print("🧠 RlVAE Experiment Runner with Hydra")
    print("=" * 50)
    
    # Print configuration
    print("📋 Configuration:")
    print(OmegaConf.to_yaml(config))
    
    # Run experiment
    runner = ExperimentRunner(config)
    runner.run()
    
    print("\n🏁 Experiment completed successfully!")


if __name__ == "__main__":
    main() 