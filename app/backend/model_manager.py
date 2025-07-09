"""
Enhanced Model Manager
=====================

Comprehensive model management system for RlVAE experiments including
model loading, saving, comparison, and analysis.
"""

import os
import sys
import torch
import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Add src to path
current_dir = Path(__file__).parent.parent.parent.absolute()
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from models.modular_rlvae import ModularRiemannianFlowVAE, ModelFactory
    from models.modular_vanilla_vae import ModularVanillaVAE
    from models.hybrid_rlvae import HybridRiemannianFlowVAE
    from models.riemannian_flow_vae import RiemannianFlowVAE
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    print(f"⚠️ Backend not available: {e}")


@dataclass
class ModelInfo:
    """Model information and metadata."""
    name: str
    type: str
    path: Path
    config: Dict[str, Any]
    metrics: Dict[str, float]
    created_at: datetime
    size_mb: float
    architecture: str
    latent_dim: int
    n_flows: int
    beta: float
    riemannian_beta: float
    training_epochs: int
    final_loss: float
    best_val_loss: float
    wandb_run_id: Optional[str]
    tags: List[str]


class ModelManager:
    """Enhanced model management system."""
    
    def __init__(self, models_dir: str = "outputs/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Model registry
        self.model_registry: Dict[str, ModelInfo] = {}
        self.load_model_registry()
    
    def load_model_registry(self):
        """Load the model registry from disk."""
        registry_path = self.models_dir / "registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    registry_data = json.load(f)
                
                for model_id, model_data in registry_data.items():
                    model_data['path'] = Path(model_data['path'])
                    model_data['created_at'] = datetime.fromisoformat(model_data['created_at'])
                    self.model_registry[model_id] = ModelInfo(**model_data)
                    
                self.logger.info(f"Loaded {len(self.model_registry)} models from registry")
            except Exception as e:
                self.logger.error(f"Failed to load model registry: {e}")
    
    def save_model_registry(self):
        """Save the model registry to disk."""
        registry_path = self.models_dir / "registry.json"
        try:
            registry_data = {}
            for model_id, model_info in self.model_registry.items():
                model_data = {
                    'name': model_info.name,
                    'type': model_info.type,
                    'path': str(model_info.path),
                    'config': model_info.config,
                    'metrics': model_info.metrics,
                    'created_at': model_info.created_at.isoformat(),
                    'size_mb': model_info.size_mb,
                    'architecture': model_info.architecture,
                    'latent_dim': model_info.latent_dim,
                    'n_flows': model_info.n_flows,
                    'beta': model_info.beta,
                    'riemannian_beta': model_info.riemannian_beta,
                    'training_epochs': model_info.training_epochs,
                    'final_loss': model_info.final_loss,
                    'best_val_loss': model_info.best_val_loss,
                    'wandb_run_id': model_info.wandb_run_id,
                    'tags': model_info.tags
                }
                registry_data[model_id] = model_data
            
            with open(registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save model registry: {e}")
    
    def save_model(self, model, config: Dict[str, Any], metrics: Dict[str, float], 
                   experiment_name: str, wandb_run_id: Optional[str] = None) -> str:
        """Save a trained model with metadata."""
        
        if not BACKEND_AVAILABLE:
            return self._save_simulation_model(config, metrics, experiment_name, wandb_run_id)
        
        try:
            # Create model directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = self.models_dir / f"{experiment_name}_{timestamp}"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model state
            model_path = model_dir / "model.pt"
            torch.save(model.state_dict(), model_path)
            
            # Save configuration
            config_path = model_dir / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Save metrics
            metrics_path = model_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Calculate model size
            model_size = model_path.stat().st_size / (1024 * 1024)  # MB
            
            # Create model info
            model_info = ModelInfo(
                name=experiment_name,
                type=config.get('model_type', 'unknown'),
                path=model_dir,
                config=config,
                metrics=metrics,
                created_at=datetime.now(),
                size_mb=model_size,
                architecture=config.get('encoder_architecture', 'unknown'),
                latent_dim=config.get('latent_dim', 0),
                n_flows=config.get('n_flows', 0),
                beta=config.get('beta', 0.0),
                riemannian_beta=config.get('riemannian_beta', 0.0),
                training_epochs=config.get('n_epochs', 0),
                final_loss=metrics.get('final_loss', 0.0),
                best_val_loss=metrics.get('best_val_loss', 0.0),
                wandb_run_id=wandb_run_id,
                tags=config.get('tags', [])
            )
            
            # Add to registry
            model_id = f"{experiment_name}_{timestamp}"
            self.model_registry[model_id] = model_info
            self.save_model_registry()
            
            self.logger.info(f"Model saved: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    def _save_simulation_model(self, config: Dict[str, Any], metrics: Dict[str, float],
                              experiment_name: str, wandb_run_id: Optional[str] = None) -> str:
        """Save a simulation model for testing."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.models_dir / f"sim_{experiment_name}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration and metrics
        config_path = model_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        metrics_path = model_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create model info
        model_info = ModelInfo(
            name=f"sim_{experiment_name}",
            type=config.get('model_type', 'simulation'),
            path=model_dir,
            config=config,
            metrics=metrics,
            created_at=datetime.now(),
            size_mb=0.1,  # Small size for simulation
            architecture=config.get('encoder_architecture', 'simulation'),
            latent_dim=config.get('latent_dim', 0),
            n_flows=config.get('n_flows', 0),
            beta=config.get('beta', 0.0),
            riemannian_beta=config.get('riemannian_beta', 0.0),
            training_epochs=config.get('n_epochs', 0),
            final_loss=metrics.get('final_loss', 0.0),
            best_val_loss=metrics.get('best_val_loss', 0.0),
            wandb_run_id=wandb_run_id,
            tags=['simulation'] + config.get('tags', [])
        )
        
        model_id = f"sim_{experiment_name}_{timestamp}"
        self.model_registry[model_id] = model_info
        self.save_model_registry()
        
        return model_id
    
    def load_model(self, model_id: str) -> Tuple[Any, Dict[str, Any]]:
        """Load a model by ID."""
        
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_info = self.model_registry[model_id]
        
        if not BACKEND_AVAILABLE:
            return self._load_simulation_model(model_info)
        
        try:
            # Load model state
            model_path = model_info.path / "model.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Create model instance
            model = self._create_model_instance(model_info.config)
            
            # Load state dict
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            
            # Load configuration
            config_path = model_info.path / "config.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            return model, config
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    def _load_simulation_model(self, model_info: ModelInfo) -> Tuple[Any, Dict[str, Any]]:
        """Load a simulation model."""
        # Return a dummy model for simulation
        class DummyModel:
            def __init__(self, config):
                self.config = config
                self.latent_dim = config.get('latent_dim', 16)
            
            def to(self, device):
                return self
            
            def eval(self):
                return self
            
            def __call__(self, x):
                # Return dummy output
                batch_size = x.shape[0] if hasattr(x, 'shape') else 1
                return {
                    'reconstruction': torch.randn(batch_size, *self.config.get('input_dim', [3, 64, 64])),
                    'mu': torch.randn(batch_size, self.latent_dim),
                    'logvar': torch.randn(batch_size, self.latent_dim)
                }
        
        return DummyModel(model_info.config), model_info.config
    
    def _create_model_instance(self, config: Dict[str, Any]) -> Any:
        """Create a model instance based on configuration."""
        
        model_type = config.get('model_type', 'modular_rlvae')
        
        if model_type == 'modular_rlvae':
            return ModularRiemannianFlowVAE(
                input_dim=config.get('input_dim', [3, 64, 64]),
                latent_dim=config.get('latent_dim', 16),
                n_flows=config.get('n_flows', 5),
                beta=config.get('beta', 1.0),
                riemannian_beta=config.get('riemannian_beta', 1.0),
                encoder_architecture=config.get('encoder_architecture', 'cnn'),
                decoder_architecture=config.get('decoder_architecture', 'cnn'),
                hidden_dims=config.get('hidden_dims', [32, 64, 128, 256]),
                posterior_type=config.get('posterior_type', 'gaussian'),
                sampling_method=config.get('sampling_method', 'geodesic'),
                use_riemannian=config.get('use_riemannian', True)
            )
        
        elif model_type == 'vanilla_vae':
            return ModularVanillaVAE(
                input_dim=config.get('input_dim', [3, 64, 64]),
                latent_dim=config.get('latent_dim', 16),
                encoder_architecture=config.get('encoder_architecture', 'cnn'),
                decoder_architecture=config.get('decoder_architecture', 'cnn'),
                hidden_dims=config.get('hidden_dims', [32, 64, 128, 256]),
                beta=config.get('beta', 1.0)
            )
        
        elif model_type == 'hybrid_rlvae':
            return HybridRiemannianFlowVAE(
                input_dim=config.get('input_dim', [3, 64, 64]),
                latent_dim=config.get('latent_dim', 16),
                n_flows=config.get('n_flows', 5),
                beta=config.get('beta', 1.0),
                riemannian_beta=config.get('riemannian_beta', 1.0),
                encoder_architecture=config.get('encoder_architecture', 'cnn'),
                decoder_architecture=config.get('decoder_architecture', 'cnn'),
                hidden_dims=config.get('hidden_dims', [32, 64, 128, 256])
            )
        
        elif model_type == 'riemannian_flow_vae':
            return RiemannianFlowVAE(
                input_dim=config.get('input_dim', [3, 64, 64]),
                latent_dim=config.get('latent_dim', 16),
                n_flows=config.get('n_flows', 5),
                beta=config.get('beta', 1.0),
                riemannian_beta=config.get('riemannian_beta', 1.0)
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def list_models(self, filters: Optional[Dict[str, Any]] = None) -> List[ModelInfo]:
        """List all models with optional filtering."""
        
        models = list(self.model_registry.values())
        
        if filters:
            filtered_models = []
            for model in models:
                include = True
                
                for key, value in filters.items():
                    if hasattr(model, key):
                        model_value = getattr(model, key)
                        if isinstance(value, (list, tuple)):
                            if model_value not in value:
                                include = False
                                break
                        else:
                            if model_value != value:
                                include = False
                                break
                    else:
                        include = False
                        break
                
                if include:
                    filtered_models.append(model)
            
            return filtered_models
        
        return models
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model from the registry and disk."""
        
        if model_id not in self.model_registry:
            return False
        
        try:
            model_info = self.model_registry[model_id]
            
            # Remove from disk
            if model_info.path.exists():
                import shutil
                shutil.rmtree(model_info.path)
            
            # Remove from registry
            del self.model_registry[model_id]
            self.save_model_registry()
            
            self.logger.info(f"Model deleted: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple models."""
        
        if len(model_ids) < 2:
            raise ValueError("Need at least 2 models to compare")
        
        comparison = {
            'models': {},
            'metrics_comparison': {},
            'config_comparison': {},
            'summary': {}
        }
        
        # Collect model information
        for model_id in model_ids:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found")
            
            model_info = self.model_registry[model_id]
            comparison['models'][model_id] = {
                'name': model_info.name,
                'type': model_info.type,
                'architecture': model_info.architecture,
                'latent_dim': model_info.latent_dim,
                'n_flows': model_info.n_flows,
                'beta': model_info.beta,
                'riemannian_beta': model_info.riemannian_beta,
                'training_epochs': model_info.training_epochs,
                'final_loss': model_info.final_loss,
                'best_val_loss': model_info.best_val_loss,
                'size_mb': model_info.size_mb,
                'created_at': model_info.created_at.isoformat(),
                'tags': model_info.tags
            }
        
        # Compare metrics
        metric_names = set()
        for model_info in [self.model_registry[mid] for mid in model_ids]:
            metric_names.update(model_info.metrics.keys())
        
        for metric in metric_names:
            comparison['metrics_comparison'][metric] = {}
            for model_id in model_ids:
                model_info = self.model_registry[model_id]
                comparison['metrics_comparison'][metric][model_id] = model_info.metrics.get(metric, None)
        
        # Compare configurations
        config_keys = set()
        for model_info in [self.model_registry[mid] for mid in model_ids]:
            config_keys.update(model_info.config.keys())
        
        for key in config_keys:
            comparison['config_comparison'][key] = {}
            for model_id in model_ids:
                model_info = self.model_registry[model_id]
                comparison['config_comparison'][key][model_id] = model_info.config.get(key, None)
        
        # Generate summary
        comparison['summary'] = {
            'best_model': min(model_ids, key=lambda mid: self.model_registry[mid].best_val_loss),
            'largest_model': max(model_ids, key=lambda mid: self.model_registry[mid].size_mb),
            'newest_model': max(model_ids, key=lambda mid: self.model_registry[mid].created_at),
            'total_models': len(model_ids)
        }
        
        return comparison
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about all models."""
        
        if not self.model_registry:
            return {}
        
        models = list(self.model_registry.values())
        
        # Model types
        model_types = {}
        for model in models:
            model_types[model.type] = model_types.get(model.type, 0) + 1
        
        # Architecture distribution
        architectures = {}
        for model in models:
            architectures[model.architecture] = architectures.get(model.architecture, 0) + 1
        
        # Latent dimension distribution
        latent_dims = [model.latent_dim for model in models]
        
        # Performance statistics
        final_losses = [model.final_loss for model in models]
        best_val_losses = [model.best_val_loss for model in models]
        
        # Size statistics
        sizes = [model.size_mb for model in models]
        
        return {
            'total_models': len(models),
            'model_types': model_types,
            'architectures': architectures,
            'latent_dimensions': {
                'min': min(latent_dims) if latent_dims else 0,
                'max': max(latent_dims) if latent_dims else 0,
                'mean': np.mean(latent_dims) if latent_dims else 0,
                'median': np.median(latent_dims) if latent_dims else 0
            },
            'performance': {
                'final_loss': {
                    'min': min(final_losses) if final_losses else 0,
                    'max': max(final_losses) if final_losses else 0,
                    'mean': np.mean(final_losses) if final_losses else 0,
                    'median': np.median(final_losses) if final_losses else 0
                },
                'best_val_loss': {
                    'min': min(best_val_losses) if best_val_losses else 0,
                    'max': max(best_val_losses) if best_val_losses else 0,
                    'mean': np.mean(best_val_losses) if best_val_losses else 0,
                    'median': np.median(best_val_losses) if best_val_losses else 0
                }
            },
            'storage': {
                'total_size_mb': sum(sizes),
                'average_size_mb': np.mean(sizes) if sizes else 0,
                'largest_model_mb': max(sizes) if sizes else 0,
                'smallest_model_mb': min(sizes) if sizes else 0
            },
            'timeline': {
                'oldest': min(model.created_at for model in models).isoformat(),
                'newest': max(model.created_at for model in models).isoformat()
            }
        }