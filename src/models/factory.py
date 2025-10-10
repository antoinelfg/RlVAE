"""
Model Factory
=============

Centralized model creation with automatic interface wrapping and configuration normalization.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Type
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import warnings

from .base.unified_rlvae import UnifiedRLVAEInterface, create_unified_interface


class ModelFactory:
    """
    Factory for creating RLVAE models with unified interfaces.
    
    This factory handles:
    - Model type detection and instantiation
    - Configuration normalization
    - Automatic interface wrapping
    - Fallback mechanisms for legacy models
    """
    
    # Registry of known model classes and their instantiation methods
    MODEL_REGISTRY = {
        'modrlvae': {
            'module': 'rlvae.models.modrlvae',
            'class': 'ModRLVAE',
            'config_style': 'direct'  # Pass config directly to constructor
        },
        'modular_rlvae': {
            'module': 'rlvae.models.modular_rlvae',
            'class': 'ModularRiemannianFlowVAE',
            'config_style': 'direct'
        },
        'riemannian_flow_vae': {
            'module': 'models.riemannian_flow_vae',
            'class': 'RiemannianFlowVAE',
            'config_style': 'kwargs'  # Pass config fields as kwargs
        },
        'rhvae_working': {
            'module': 'models.rhvae_working',
            'class': 'RiemannianFlowVAE',
            'config_style': 'kwargs'
        },
        'composite_rlvae': {
            'module': 'models.composite.rlvae',
            'class': 'RLVAE',
            'config_style': 'hydra'  # Use Hydra instantiate
        },
        'composite_vae': {
            'module': 'models.composite.vae',
            'class': 'VAE',
            'config_style': 'hydra'
        },
        'vanilla_vae': {
            'module': 'models.modular_vanilla_vae',
            'class': 'ModularVanillaVAE',
            'config_style': 'direct'
        }
    }
    
    @classmethod
    def create_model(cls, config: Union[DictConfig, Dict[str, Any]], 
                    force_unified: bool = True) -> Union[nn.Module, UnifiedRLVAEInterface]:
        """
        Create a model from configuration with optional unified interface.
        
        Args:
            config: Model configuration
            force_unified: Whether to wrap with unified interface
            
        Returns:
            Model instance (wrapped or unwrapped)
        """
        # Normalize configuration
        config = cls._normalize_config(config)
        
        # Detect model type
        model_type = cls._detect_model_type(config)
        
        # Create model instance
        model = cls._instantiate_model(config, model_type)
        
        # Wrap with unified interface if requested
        if force_unified:
            return create_unified_interface(model, model_type)
        else:
            return model
    
    @classmethod
    def _normalize_config(cls, config: Union[DictConfig, Dict[str, Any]]) -> DictConfig:
        """Normalize configuration to DictConfig format."""
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        elif not isinstance(config, DictConfig):
            # Try to convert to dict first
            try:
                config_dict = dict(config)
                config = OmegaConf.create(config_dict)
            except Exception:
                raise ValueError(f"Cannot convert config of type {type(config)} to DictConfig")
        
        return config
    
    @classmethod
    def _detect_model_type(cls, config: DictConfig) -> str:
        """Detect model type from configuration."""
        # Check for explicit _target_ specification
        if '_target_' in config:
            target = config._target_
            
            # Map common targets to our registry
            if 'ModRLVAE' in target:
                return 'modrlvae'
            elif 'ModularRiemannianFlowVAE' in target:
                return 'modular_rlvae'
            elif 'RiemannianFlowVAE' in target:
                if 'rhvae_working' in target:
                    return 'rhvae_working'
                else:
                    return 'riemannian_flow_vae'
            elif 'composite.rlvae' in target or target.endswith('RLVAE'):
                return 'composite_rlvae'
            elif 'composite.vae' in target or target.endswith('VAE'):
                return 'composite_vae'
            elif 'ModularVanillaVAE' in target:
                return 'vanilla_vae'
        
        # Check for model type field
        if 'model_type' in config:
            return config.model_type
        
        # Check for architecture hints
        if 'architecture' in config:
            arch = config.architecture
            if 'modular' in arch.lower():
                return 'modular_rlvae'
            elif 'rhvae' in arch.lower():
                return 'rhvae_working'
            elif 'rlvae' in arch.lower():
                return 'riemannian_flow_vae'
        
        # Default fallback
        warnings.warn("Could not detect model type from config, defaulting to 'modular_rlvae'")
        return 'modular_rlvae'
    
    @classmethod
    def _instantiate_model(cls, config: DictConfig, model_type: str) -> nn.Module:
        """Instantiate model based on type and configuration."""
        # Fast-path: modular model expects the structured DictConfig directly
        if model_type == 'modular_rlvae':
            model_class = cls._import_model_class('rlvae.models.modular_rlvae', 'ModularRiemannianFlowVAE')
            return model_class(config)
        if model_type not in cls.MODEL_REGISTRY:
            # Fallback to Hydra instantiate
            return cls._instantiate_with_hydra(config)
        
        registry_entry = cls.MODEL_REGISTRY[model_type]
        
        try:
            # Import the model class
            model_class = cls._import_model_class(registry_entry['module'], registry_entry['class'])
            
            # Instantiate based on config style
            config_style = registry_entry['config_style']
            
            if config_style == 'direct':
                # Pass config directly to constructor (for modular models)
                if model_type in ['modrlvae', 'modular_rlvae', 'vanilla_vae']:
                    # These models expect a structured config
                    modular_config = cls._create_modular_config(config)
                    return model_class(modular_config)
                else:
                    return model_class(config)
            elif config_style == 'kwargs':
                # Convert config to kwargs
                kwargs = cls._config_to_kwargs(config)
                return model_class(**kwargs)
            elif config_style == 'hydra':
                # Use Hydra instantiate
                return instantiate(config)
            else:
                raise ValueError(f"Unknown config style: {config_style}")
                
        except Exception as e:
            warnings.warn(f"Failed to instantiate {model_type} with registry method: {e}")
            return cls._instantiate_with_hydra(config)
    
    @classmethod
    def _import_model_class(cls, module_name: str, class_name: str) -> Type[nn.Module]:
        """Dynamically import model class."""
        try:
            # Try absolute import first
            module = __import__(f"src.{module_name}", fromlist=[class_name])
            return getattr(module, class_name)
        except ImportError:
            try:
                # Try relative import
                module = __import__(module_name, fromlist=[class_name])
                return getattr(module, class_name)
            except ImportError as e:
                raise ImportError(f"Could not import {class_name} from {module_name}: {e}")
    
    @classmethod
    def _config_to_kwargs(cls, config: DictConfig) -> Dict[str, Any]:
        """Convert DictConfig to kwargs for model instantiation."""
        kwargs = {}
        
        # Standard parameters
        if 'input_dim' in config:
            kwargs['input_dim'] = tuple(config.input_dim) if isinstance(config.input_dim, (list, tuple)) else config.input_dim
        if 'latent_dim' in config:
            kwargs['latent_dim'] = config.latent_dim
        if 'n_flows' in config:
            kwargs['n_flows'] = config.n_flows
        if 'flow_hidden_size' in config:
            kwargs['flow_hidden_size'] = config.flow_hidden_size
        if 'beta' in config:
            kwargs['beta'] = config.beta
        if 'riemannian_beta' in config:
            kwargs['riemannian_beta'] = config.riemannian_beta
        if 'posterior_type' in config:
            kwargs['posterior_type'] = config.posterior_type
        if 'loop_mode' in config:
            kwargs['loop_mode'] = config.loop_mode
        
        # Add any other fields that don't conflict
        for key, value in config.items():
            if key not in kwargs and not key.startswith('_'):
                kwargs[key] = value
        
        return kwargs
    
    @classmethod
    def _create_modular_config(cls, config: DictConfig) -> DictConfig:
        """Create config suitable for modular models that expect full config object."""
        # Create a complete config structure for modular models
        modular_config = OmegaConf.create({
            'input_dim': config.get('input_dim', [3, 64, 64]),
            'latent_dim': config.get('latent_dim', 2),
            'n_flows': config.get('n_flows', 8),
            'flow_hidden_size': config.get('flow_hidden_size', 256),
            'flow_n_blocks': config.get('flow_n_blocks', 2),
            'flow_n_hidden': config.get('flow_n_hidden', 1),
            'beta': config.get('beta', 1.0),
            'riemannian_beta': config.get('riemannian_beta', config.get('beta', 1.0)),
            'posterior_type': config.get('posterior_type', 'riemannian_metric'),
            'epsilon': config.get('epsilon', 1e-6),
            'temperature': config.get('temperature', 0.1),
            'regularization': config.get('regularization', 0.01),
            'architecture': config.get('architecture', 'mlp'),
            
            # Loop configuration
            'loop': {
                'mode': config.get('loop_mode', 'open'),
                'penalty': config.get('cycle_penalty', 1.0)
            },
            
            # Flow configuration
            'flow': {
                'n_flows': config.get('n_flows', 8),
                'hidden_size': config.get('flow_hidden_size', 256),
                'n_blocks': config.get('flow_n_blocks', 2),
                'n_hidden': config.get('flow_n_hidden', 1)
            },
            
            # Metric configuration
            'metric': {
                'temperature': config.get('temperature', 0.1),
                'regularization': config.get('regularization', 0.01)
            },
            
            # Posterior configuration
            'posterior': {
                'type': config.get('posterior_type', 'riemannian_metric')
            },
            
            # Encoder/Decoder configuration
            'encoder': {
                'architecture': config.get('architecture', 'mlp')
            },
            'decoder': {
                'architecture': config.get('architecture', 'mlp')
            }
        })
        
        # Add any additional fields from original config
        for key, value in config.items():
            if key not in modular_config and not key.startswith('_'):
                modular_config[key] = value
        
        return modular_config
    
    @classmethod
    def _instantiate_with_hydra(cls, config: DictConfig) -> nn.Module:
        """Fallback instantiation using Hydra."""
        try:
            return instantiate(config)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate model with Hydra: {e}")
    
    @classmethod
    def register_model(cls, name: str, module: str, class_name: str, config_style: str = 'direct'):
        """Register a new model type."""
        cls.MODEL_REGISTRY[name] = {
            'module': module,
            'class': class_name,
            'config_style': config_style
        }
    
    @classmethod
    def list_available_models(cls) -> list:
        """List all available model types."""
        return list(cls.MODEL_REGISTRY.keys())


def create_model_from_config(config: Union[DictConfig, Dict[str, Any]], 
                           unified: bool = True) -> Union[nn.Module, UnifiedRLVAEInterface]:
    """
    Convenience function to create a model from configuration.
    
    Args:
        config: Model configuration
        unified: Whether to use unified interface
        
    Returns:
        Model instance
    """
    return ModelFactory.create_model(config, force_unified=unified)


def auto_detect_and_wrap_model(model: nn.Module) -> UnifiedRLVAEInterface:
    """
    Auto-detect model type and wrap with unified interface.
    
    Args:
        model: Existing model instance
        
    Returns:
        Wrapped model with unified interface
    """
    model_type = type(model).__name__
    return create_unified_interface(model, model_type)
