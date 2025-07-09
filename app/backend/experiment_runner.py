"""
Enhanced Experiment Runner Backend
=================================

Real experiment execution backend with GPU/CPU integration, WandB logging,
and comprehensive progress tracking for the RlVAE pipeline.
"""

import os
import sys
import subprocess
import threading
import time
import json
import yaml
import torch
import wandb
import hydra
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime
import queue
import logging
from omegaconf import DictConfig, OmegaConf

# Add src to path
current_dir = Path(__file__).parent.parent.parent.absolute()
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from models.modular_rlvae import ModularRiemannianFlowVAE, ModelFactory, MetricsCollector
    from data.cyclic_dataset import CyclicSpritesDataModule
    from training.lightning_trainer import LightningRlVAETrainer
    from visualizations.manager import VisualizationManager
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    print(f"⚠️ Backend not available: {e}")


@dataclass
class ExperimentStatus:
    """Experiment status tracking."""
    status: str  # 'idle', 'running', 'completed', 'failed'
    current_stage: str  # 'stage1', 'stage2', 'testing', 'visualization'
    progress: float  # 0.0 to 1.0
    current_epoch: int
    total_epochs: int
    current_loss: float
    best_loss: float
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    error_message: Optional[str]
    wandb_run_id: Optional[str]
    gpu_utilization: float
    memory_usage: float


class StreamlitExperimentRunner:
    """Enhanced experiment runner with real backend integration."""
    
    def __init__(self):
        self.current_experiment = None
        self.status = ExperimentStatus(
            status='idle',
            current_stage='',
            progress=0.0,
            current_epoch=0,
            total_epochs=0,
            current_loss=0.0,
            best_loss=float('inf'),
            start_time=None,
            end_time=None,
            error_message=None,
            wandb_run_id=None,
            gpu_utilization=0.0,
            memory_usage=0.0
        )
        self.metrics_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        info = {
            'device': 'cpu',
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': None,
            'gpu_name': None,
            'gpu_memory_gb': 0.0,
            'gpu_memory_used_gb': 0.0,
            'gpu_memory_free_gb': 0.0,
            'gpu_utilization': 0.0,
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'num_cpus': os.cpu_count(),
            'total_memory_gb': 0.0,
            'available_memory_gb': 0.0
        }
        
        if torch.cuda.is_available():
            info['device'] = 'cuda'
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info['gpu_memory_used_gb'] = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9
            info['gpu_memory_free_gb'] = torch.cuda.memory_reserved(0) / 1e9
            
            # Try to get GPU utilization
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                info['gpu_utilization'] = util.gpu
            except:
                pass
        
        # Get system memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            info['total_memory_gb'] = memory.total / 1e9
            info['available_memory_gb'] = memory.available / 1e9
        except:
            pass
        
        return info
    
    def validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate experiment configuration."""
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ['experiment_name', 'model_type', 'latent_dim', 'n_epochs']
        for field in required_fields:
            if not config.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate model parameters
        if config.get('latent_dim', 0) <= 0:
            errors.append("Latent dimension must be positive")
        
        if config.get('n_epochs', 0) <= 0:
            errors.append("Number of epochs must be positive")
        
        # Check device availability
        device_info = self.get_device_info()
        if config.get('device') == 'cuda' and not device_info['cuda_available']:
            errors.append("CUDA requested but not available")
        
        # Check memory requirements
        if config.get('device') == 'cuda':
            required_memory = config.get('batch_size', 32) * config.get('latent_dim', 16) * 4  # Rough estimate
            if device_info['gpu_memory_free_gb'] < required_memory / 1e9:
                warnings.append(f"GPU memory might be insufficient for batch size {config.get('batch_size')}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'device_info': device_info
        }
    
    def create_hydra_config(self, config: Dict[str, Any]) -> DictConfig:
        """Create Hydra configuration from Streamlit config."""
        
        # Base configuration
        hydra_config = {
            'experiment_name': config['experiment_name'],
            'project_name': config.get('project_name', 'rlvae-streamlit'),
            'description': config.get('description', ''),
            'seed': config.get('seed', 42),
            'device': config.get('device', 'auto'),
            'output_dir': 'outputs',
            
            'wandb': {
                'project': config.get('project_name', 'rlvae-streamlit'),
                'entity': config.get('wandb_entity'),
                'mode': config.get('wandb_mode', 'online'),
                'tags': config.get('tags', []).split(',') if config.get('tags') else []
            },
            
            'model': {
                'type': config['model_type'],
                'latent_dim': config['latent_dim'],
                'input_dim': eval(config.get('input_dim', '(3, 64, 64)')),
                'n_flows': config.get('n_flows', 5),
                'beta': config.get('beta', 1.0),
                'riemannian_beta': config.get('riemannian_beta', 1.0),
                'posterior_type': config.get('posterior_type', 'gaussian'),
                'encoder_architecture': config.get('encoder_architecture', 'cnn'),
                'decoder_architecture': config.get('decoder_architecture', 'cnn'),
                'hidden_dims': config.get('hidden_dims', [32, 64, 128, 256]),
                'sampling_method': config.get('sampling_method', 'geodesic'),
                'use_riemannian': config.get('use_riemannian', True)
            },
            
            'training': {
                'n_epochs': config['n_epochs'],
                'batch_size': config.get('batch_size', 32),
                'learning_rate': config.get('learning_rate', 1e-3),
                'weight_decay': config.get('weight_decay', 1e-5),
                'optimizer': config.get('optimizer', 'adam'),
                'scheduler': config.get('scheduler', 'cosine'),
                'early_stopping_patience': config.get('early_stopping_patience', 10),
                'gradient_clip_val': config.get('gradient_clip_val', 1.0),
                'accumulate_grad_batches': config.get('accumulate_grad_batches', 1),
                'precision': config.get('precision', 16),
                'num_workers': config.get('num_workers', 4)
            },
            
            'data': {
                'dataset': config.get('dataset', 'cyclic_sprites'),
                'data_dir': config.get('data_dir', 'data'),
                'train_split': config.get('train_split', 0.8),
                'val_split': config.get('val_split', 0.1),
                'test_split': config.get('test_split', 0.1),
                'num_workers': config.get('num_workers', 4),
                'pin_memory': config.get('pin_memory', True)
            },
            
            'visualization': {
                'level': config.get('visualization_level', 'standard'),
                'save_frequency': config.get('save_frequency', 5),
                'max_sequences': config.get('max_sequences', 100),
                'interactive': config.get('interactive_plots', True),
                'wandb_logging': config.get('wandb_logging', True)
            },
            
            'experiment': {
                'type': config.get('experiment_type', 'single'),
                'name': config['experiment_name'],
                'stages': config.get('stages', ['stage1', 'stage2']),
                'comparison_metrics': ['val_loss', 'reconstruction_loss', 'kl_loss', 'riemannian_kl']
            },
            
            'hydra': {
                'run': {
                    'dir': '${output_dir}/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}'
                }
            }
        }
        
        return OmegaConf.create(hydra_config)
    
    def start_experiment(self, config: Dict[str, Any], 
                        progress_callback: Optional[Callable] = None,
                        metrics_callback: Optional[Callable] = None) -> bool:
        """Start experiment execution in a separate thread."""
        
        # Validate configuration
        validation = self.validate_configuration(config)
        if not validation['valid']:
            self.status.error_message = f"Configuration errors: {validation['errors']}"
            return False
        
        # Reset status
        self.status = ExperimentStatus(
            status='running',
            current_stage='initializing',
            progress=0.0,
            current_epoch=0,
            total_epochs=config['n_epochs'],
            current_loss=0.0,
            best_loss=float('inf'),
            start_time=datetime.now(),
            end_time=None,
            error_message=None,
            wandb_run_id=None,
            gpu_utilization=0.0,
            memory_usage=0.0
        )
        
        # Create Hydra config
        hydra_config = self.create_hydra_config(config)
        
        # Start experiment thread
        self.stop_event.clear()
        experiment_thread = threading.Thread(
            target=self._run_experiment_thread,
            args=(hydra_config, progress_callback, metrics_callback)
        )
        experiment_thread.daemon = True
        experiment_thread.start()
        
        return True
    
    def _run_experiment_thread(self, config: DictConfig, 
                              progress_callback: Optional[Callable],
                              metrics_callback: Optional[Callable]):
        """Run experiment in separate thread."""
        
        try:
            if not BACKEND_AVAILABLE:
                self._run_simulation_experiment(config, progress_callback, metrics_callback)
            else:
                self._run_real_experiment(config, progress_callback, metrics_callback)
                
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            self.status.status = 'failed'
            self.status.error_message = str(e)
            self.status.end_time = datetime.now()
    
    def _run_simulation_experiment(self, config: DictConfig, 
                                  progress_callback: Optional[Callable],
                                  metrics_callback: Optional[Callable]):
        """Run simulated experiment for testing."""
        
        self.status.current_stage = 'simulation'
        
        # Simulate training progress
        for epoch in range(config.training.n_epochs):
            if self.stop_event.is_set():
                break
                
            # Simulate epoch progress
            for step in range(10):  # 10 steps per epoch
                if self.stop_event.is_set():
                    break
                    
                progress = (epoch * 10 + step) / (config.training.n_epochs * 10)
                self.status.progress = progress
                self.status.current_epoch = epoch + 1
                self.status.current_loss = 1.0 - progress + 0.1  # Simulate decreasing loss
                
                if progress_callback:
                    progress_callback(self.status)
                
                time.sleep(0.1)  # Simulate training time
            
            # Simulate metrics
            if metrics_callback:
                metrics = {
                    'epoch': epoch + 1,
                    'train_loss': 1.0 - progress + 0.1,
                    'val_loss': 1.0 - progress + 0.15,
                    'reconstruction_loss': 0.5 - progress * 0.3,
                    'kl_loss': 0.3 - progress * 0.2,
                    'riemannian_kl': 0.2 - progress * 0.1 if config.model.use_riemannian else 0.0
                }
                metrics_callback(metrics)
        
        self.status.status = 'completed'
        self.status.progress = 1.0
        self.status.end_time = datetime.now()
    
    def _run_real_experiment(self, config: DictConfig,
                            progress_callback: Optional[Callable],
                            metrics_callback: Optional[Callable]):
        """Run real experiment with backend integration."""
        
        try:
            # Initialize WandB
            wandb_logger = self._setup_wandb(config)
            self.status.wandb_run_id = wandb.run.id if wandb.run else None
            
            # Create data module
            self.status.current_stage = 'data_loading'
            data_module = CyclicSpritesDataModule(config.data)
            data_module.setup("fit", config.training)
            
            # Create model wrapper
            self.status.current_stage = 'model_creation'
            model_wrapper = LightningRlVAETrainer(
                config,
                data_module=data_module,
                progress_callback=progress_callback,
                metrics_callback=metrics_callback
            )
            
            # Setup trainer
            trainer = self._create_trainer(config, wandb_logger)
            
            # Train
            self.status.current_stage = 'training'
            trainer.fit(model_wrapper, data_module)
            
            # Test
            self.status.current_stage = 'testing'
            test_results = trainer.test(model_wrapper, data_module)
            
            # Visualization
            self.status.current_stage = 'visualization'
            self._run_visualizations(config, model_wrapper, data_module)
            
            self.status.status = 'completed'
            self.status.progress = 1.0
            self.status.end_time = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Real experiment failed: {e}")
            self.status.status = 'failed'
            self.status.error_message = str(e)
            self.status.end_time = datetime.now()
            raise
    
    def _setup_wandb(self, config: DictConfig):
        """Setup WandB logging."""
        if config.wandb.mode == 'disabled':
            return None
            
        wandb_config = {
            'experiment_name': config.experiment_name,
            'model_type': config.model.type,
            'latent_dim': config.model.latent_dim,
            'n_epochs': config.training.n_epochs,
            'batch_size': config.training.batch_size,
            'learning_rate': config.training.learning_rate,
            'use_riemannian': config.model.use_riemannian,
            'n_flows': config.model.n_flows,
            'beta': config.model.beta,
            'riemannian_beta': config.model.riemannian_beta
        }
        
        return wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=wandb_config,
            tags=config.wandb.tags,
            mode=config.wandb.mode
        )
    
    def _create_trainer(self, config: DictConfig, wandb_logger):
        """Create PyTorch Lightning trainer."""
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from lightning.pytorch.loggers import WandbLogger
        
        callbacks = []
        
        # Early stopping
        if config.training.early_stopping_patience > 0:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=config.training.early_stopping_patience,
                mode='min'
            ))
        
        # Model checkpointing
        callbacks.append(ModelCheckpoint(
            dirpath=f"{config.output_dir}/{config.experiment_name}/checkpoints",
            filename='best-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3
        ))
        
        # Create trainer
        trainer_kwargs = {
            'max_epochs': config.training.n_epochs,
            'accelerator': 'auto',
            'devices': 1,
            'precision': config.training.precision,
            'gradient_clip_val': config.training.gradient_clip_val,
            'accumulate_grad_batches': config.training.accumulate_grad_batches,
            'callbacks': callbacks,
            'enable_progress_bar': False,  # We handle progress ourselves
            'enable_model_summary': False,
            'enable_checkpointing': True,
            'log_every_n_steps': 10
        }
        
        if wandb_logger:
            trainer_kwargs['logger'] = WandbLogger()
        
        return L.Trainer(**trainer_kwargs)
    
    def _run_visualizations(self, config: DictConfig, model_wrapper, data_module):
        """Run visualization pipeline."""
        try:
            viz_manager = VisualizationManager(config.visualization)
            viz_manager.create_all_visualizations(model_wrapper, data_module)
        except Exception as e:
            self.logger.warning(f"Visualization failed: {e}")
    
    def stop_experiment(self):
        """Stop current experiment."""
        self.stop_event.set()
        self.status.status = 'stopping'
    
    def get_status(self) -> ExperimentStatus:
        """Get current experiment status."""
        return self.status
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get collected metrics."""
        metrics = []
        while not self.metrics_queue.empty():
            try:
                metrics.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        return metrics
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_experiment()
        if wandb.run:
            wandb.finish()