"""
Streamlit Experiment Runner
==========================

Bridges the Hydra-based experiment system with the Streamlit interface.
"""

import torch
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from omegaconf import DictConfig, OmegaConf
import threading
import time
import queue
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import streamlit as st

# Add src to path for imports
current_dir = Path(__file__).parent.parent.parent.absolute()
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from models.modular_rlvae import ModularRiemannianFlowVAE
from data.cyclic_dataset import CyclicSpritesDataModule
from training.lightning_trainer import LightningRlVAETrainer


class StreamlitExperimentRunner:
    """
    Experiment runner designed for Streamlit integration.
    
    Provides real-time training with live metrics updates and experiment management.
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_experiment: Optional[Dict[str, Any]] = None
        self.training_thread: Optional[threading.Thread] = None
        self.metrics_queue = queue.Queue()
        self.stop_training = threading.Event()
        self.is_training = False
        
        # Experiment storage
        self.experiment_history: List[Dict[str, Any]] = []
        
        print(f"🚀 StreamlitExperimentRunner initialized on {self.device}")
    
    def create_experiment_config(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        data_config: Dict[str, Any],
        experiment_name: str = "streamlit_experiment"
    ) -> DictConfig:
        """Create a complete experiment configuration."""
        
        # Set default pretrained paths
        pretrained_paths = {
            'encoder_path': str(current_dir / "data" / "pretrained" / "encoder.pt"),
            'decoder_path': str(current_dir / "data" / "pretrained" / "decoder.pt"),
            'metric_path': str(current_dir / "data" / "pretrained" / "metric_T0.7_scaled.pt")
        }
        
        # Check which files exist
        for key, path in pretrained_paths.items():
            if not Path(path).exists():
                pretrained_paths[key] = None
        
        # Create complete configuration
        config_dict = {
            'experiment_name': experiment_name,
            'seed': 42,
            'device': "auto",
            'output_dir': str(current_dir / "outputs" / "streamlit_experiments"),
            
            # Model configuration
            'model': {
                'input_dim': model_config.get('input_dim', [3, 64, 64]),
                'latent_dim': model_config.get('latent_dim', 10),
                'n_flows': model_config.get('n_flows', 4),
                'flow_hidden_size': model_config.get('flow_hidden_size', 128),
                'flow_n_blocks': model_config.get('flow_n_blocks', 3),
                'flow_n_hidden': model_config.get('flow_n_hidden', 128),
                'epsilon': model_config.get('epsilon', 1e-6),
                'beta': model_config.get('beta', 1.0),
                'riemannian_beta': model_config.get('riemannian_beta', 5.0),
                
                'encoder': model_config.get('encoder', {'architecture': 'mlp'}),
                'decoder': model_config.get('decoder', {'architecture': 'mlp'}),
                
                'posterior': {
                    'type': model_config.get('posterior_type', 'riemannian_metric')
                },
                
                'sampling': {
                    'use_riemannian': model_config.get('use_riemannian', True),
                    'method': model_config.get('sampling_method', 'enhanced')
                },
                
                'loop': {
                    'mode': model_config.get('loop_mode', 'closed'),
                    'penalty': model_config.get('loop_penalty', 1.0)
                },
                
                'metric': {
                    'temperature_override': model_config.get('temperature_override', 0.7),
                    'regularization_override': None
                },
                
                'pretrained': pretrained_paths
            },
            
            # Training configuration
            'training': {
                'optimizer': {
                    'lr': training_config.get('learning_rate', 1e-3),
                    'weight_decay': training_config.get('weight_decay', 1e-5)
                },
                
                'scheduler': {
                    'mode': 'min',
                    'factor': 0.5,
                    'patience': 10,
                    'min_lr': 1e-6
                },
                
                'trainer': {
                    'max_epochs': training_config.get('max_epochs', 50),
                    'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
                    'devices': 1,
                    'strategy': 'auto',
                    'precision': '32-true',
                    'log_every_n_steps': 10,
                    'val_check_interval': 1.0,
                    'num_sanity_val_steps': 2,
                    'enable_progress_bar': False,  # Disable for Streamlit
                    'enable_model_summary': True,
                    'deterministic': False
                },
                
                'data': {
                    'batch_size': training_config.get('batch_size', 8),
                    'num_workers': 0,  # Disable for Streamlit
                    'pin_memory': False
                },
                
                'data_splits': {
                    'train': 0.7,
                    'val': 0.3
                },
                
                'early_stopping': {
                    'monitor': 'val_loss',
                    'patience': 15,
                    'mode': 'min'
                },
                
                'logging': {
                    'monitor': 'val_loss',
                    'save_top_k': 3,
                    'mode': 'min'
                }
            },
            
            # Data configuration
            'data': {
                'train_path': str(current_dir / "src" / "datasprites" / "Sprites_train.pt"),
                'test_path': str(current_dir / "src" / "datasprites" / "Sprites_test.pt"),
                'num_workers': 0,
                'pin_memory': False,
                'verify_cyclicity': True,
                'cyclicity_threshold': 0.01
            },
            
            # Visualization configuration
            'visualization': {
                'level': 'full',
                'frequency': 5,
                'save_visualizations': True,
                'n_samples': 8
            },
            
            # Wandb configuration (disabled for Streamlit)
            'wandb': {
                'project': 'streamlit-rlvae',
                'entity': None,
                'mode': 'disabled',
                'tags': ['streamlit', 'interactive']
            }
        }
        
        return OmegaConf.create(config_dict)
    
    def start_training(
        self,
        config: DictConfig,
        progress_callback: Optional[Callable] = None,
        metrics_callback: Optional[Callable] = None
    ) -> bool:
        """Start training in a separate thread."""
        
        if self.is_training:
            print("⚠️ Training already in progress")
            return False
        
        # Reset state
        self.stop_training.clear()
        self.metrics_queue = queue.Queue()
        
        # Create training thread
        self.training_thread = threading.Thread(
            target=self._training_worker,
            args=(config, progress_callback, metrics_callback)
        )
        
        self.training_thread.start()
        self.is_training = True
        
        print(f"🚀 Training started with config: {config.experiment_name}")
        return True
    
    def _training_worker(
        self,
        config: DictConfig,
        progress_callback: Optional[Callable] = None,
        metrics_callback: Optional[Callable] = None
    ):
        """Worker function that runs training in separate thread."""
        
        try:
            # Set random seed
            L.seed_everything(config.get('seed', 42))
            
            # Create data module
            data_module = CyclicSpritesDataModule(config.data)
            data_module.setup("fit", config.training)
            
            # Create model wrapper
            model_wrapper = LightningRlVAETrainer(config, data_module=data_module)
            
            # Create custom callback for Streamlit updates
            streamlit_callback = StreamlitProgressCallback(
                metrics_queue=self.metrics_queue,
                stop_event=self.stop_training,
                progress_callback=progress_callback,
                metrics_callback=metrics_callback
            )
            
            # Setup trainer
            callbacks = [streamlit_callback]
            
            # Add early stopping if configured
            if config.training.get('early_stopping'):
                early_stop = EarlyStopping(
                    monitor=config.training.early_stopping.monitor,
                    patience=config.training.early_stopping.patience,
                    mode=config.training.early_stopping.mode
                )
                callbacks.append(early_stop)
            
            # Add model checkpointing
            checkpoint_dir = Path(config.output_dir) / config.experiment_name / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = ModelCheckpoint(
                monitor=config.training.logging.monitor,
                save_top_k=config.training.logging.save_top_k,
                mode=config.training.logging.mode,
                dirpath=checkpoint_dir,
                filename="{epoch:02d}-{val_loss:.3f}"
            )
            callbacks.append(checkpoint)
            
            # Create trainer
            trainer = L.Trainer(
                max_epochs=config.training.trainer.max_epochs,
                accelerator=config.training.trainer.accelerator,
                devices=config.training.trainer.devices,
                strategy=config.training.trainer.strategy,
                precision=config.training.trainer.precision,
                log_every_n_steps=config.training.trainer.log_every_n_steps,
                val_check_interval=config.training.trainer.val_check_interval,
                num_sanity_val_steps=config.training.trainer.num_sanity_val_steps,
                enable_progress_bar=config.training.trainer.enable_progress_bar,
                enable_model_summary=config.training.trainer.enable_model_summary,
                deterministic=config.training.trainer.deterministic,
                callbacks=callbacks
            )
            
            # Start training
            print(f"🚀 Starting training for {config.training.trainer.max_epochs} epochs")
            trainer.fit(model_wrapper, data_module)
            
            # Test the model
            if not self.stop_training.is_set():
                print(f"🧪 Running test evaluation...")
                test_results = trainer.test(model_wrapper, data_module)
                
                # Store results
                self.current_experiment = {
                    'config': config,
                    'model': model_wrapper.model,
                    'test_results': test_results[0] if test_results else {},
                    'checkpoint_path': checkpoint.best_model_path,
                    'training_completed': True
                }
                
                # Add to history
                self.experiment_history.append(self.current_experiment.copy())
                
                print(f"✅ Training completed successfully!")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            
            # Store error state
            self.current_experiment = {
                'config': config,
                'error': str(e),
                'training_completed': False
            }
            
        finally:
            self.is_training = False
    
    def stop_training_process(self):
        """Stop the current training process."""
        if self.is_training and self.training_thread is not None:
            print("⏹️ Stopping training...")
            self.stop_training.set()
            
            # Wait for thread to finish (with timeout)
            self.training_thread.join(timeout=30)
            
            if self.training_thread.is_alive():
                print("⚠️ Training thread did not stop gracefully")
            else:
                print("✅ Training stopped successfully")
            
            self.is_training = False
    
    def get_training_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the latest training metrics from the queue."""
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_current_experiment(self) -> Optional[Dict[str, Any]]:
        """Get the current experiment details."""
        return self.current_experiment
    
    def get_experiment_history(self) -> List[Dict[str, Any]]:
        """Get the history of all experiments."""
        return self.experiment_history
    
    def save_experiment(self, experiment_name: str) -> bool:
        """Save the current experiment to disk."""
        if self.current_experiment is None:
            return False
        
        try:
            save_dir = Path(current_dir) / "outputs" / "streamlit_experiments" / experiment_name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            config_path = save_dir / "config.yaml"
            with open(config_path, 'w') as f:
                OmegaConf.save(self.current_experiment['config'], f)
            
            # Save model if available
            if 'model' in self.current_experiment:
                model_path = save_dir / "model.pt"
                torch.save(self.current_experiment['model'].state_dict(), model_path)
            
            # Save results
            results_path = save_dir / "results.yaml"
            if 'test_results' in self.current_experiment:
                with open(results_path, 'w') as f:
                    OmegaConf.save(OmegaConf.create(self.current_experiment['test_results']), f)
            
            print(f"💾 Experiment saved to: {save_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save experiment: {e}")
            return False


class StreamlitProgressCallback(L.Callback):
    """Custom Lightning callback for Streamlit progress updates."""
    
    def __init__(
        self,
        metrics_queue: queue.Queue,
        stop_event: threading.Event,
        progress_callback: Optional[Callable] = None,
        metrics_callback: Optional[Callable] = None
    ):
        super().__init__()
        self.metrics_queue = metrics_queue
        self.stop_event = stop_event
        self.progress_callback = progress_callback
        self.metrics_callback = metrics_callback
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch."""
        
        # Check if we should stop
        if self.stop_event.is_set():
            trainer.should_stop = True
            return
        
        # Get current metrics
        metrics = {
            'epoch': trainer.current_epoch,
            'train_loss': trainer.callback_metrics.get('train_loss', 0.0),
            'val_loss': trainer.callback_metrics.get('val_loss', 0.0),
            'train_recon_loss': trainer.callback_metrics.get('train_recon_loss', 0.0),
            'train_kl_loss': trainer.callback_metrics.get('train_kl_loss', 0.0),
            'val_recon_loss': trainer.callback_metrics.get('val_recon_loss', 0.0),
            'val_kl_loss': trainer.callback_metrics.get('val_kl_loss', 0.0),
            'learning_rate': trainer.optimizers[0].param_groups[0]['lr']
        }
        
        # Convert tensors to floats
        for key, value in metrics.items():
            if torch.is_tensor(value):
                metrics[key] = value.item()
        
        # Add to queue for Streamlit
        try:
            self.metrics_queue.put_nowait(metrics)
        except queue.Full:
            pass  # Skip if queue is full
        
        # Call external callbacks
        if self.progress_callback:
            progress = (trainer.current_epoch + 1) / trainer.max_epochs
            self.progress_callback(progress)
        
        if self.metrics_callback:
            self.metrics_callback(metrics)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of each validation epoch."""
        
        # Check if we should stop
        if self.stop_event.is_set():
            trainer.should_stop = True