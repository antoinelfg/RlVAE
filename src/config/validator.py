"""
Configuration Validator
======================

Validates and normalizes configurations before model creation to prevent
configuration sync errors and ensure consistent parameter handling.
"""

import torch
from typing import Dict, Any, Optional, Union, List, Tuple
from omegaconf import DictConfig, OmegaConf
import warnings
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    config: DictConfig
    errors: List[str]
    warnings: List[str]
    
    def __bool__(self):
        return self.is_valid


class ConfigValidator:
    """
    Validates and normalizes RLVAE model configurations.
    
    This validator ensures:
    - Required parameters are present
    - Parameter types are correct
    - Cross-parameter consistency
    - Sensible defaults for missing parameters
    """
    
    # Required parameters for different model types
    REQUIRED_PARAMS = {
        'base': ['latent_dim'],
        'rlvae': ['latent_dim', 'input_dim'],
        'flow_vae': ['latent_dim', 'input_dim', 'n_flows'],
        'modular': ['latent_dim', 'input_dim']
    }
    
    # Default values for common parameters
    DEFAULTS = {
        'latent_dim': 2,
        'input_dim': (3, 64, 64),
        'n_flows': 8,
        'flow_hidden_size': 256,
        'beta': 1.0,
        'riemannian_beta': None,  # Will default to beta if not set
        'posterior_type': 'riemannian_metric',
        'loop_mode': 'open',
        'epsilon': 1e-6,
        'temperature': 0.1,
        'regularization': 0.01,
        'architecture': 'mlp'
    }
    
    # Parameter type specifications
    PARAM_TYPES = {
        'latent_dim': int,
        'n_flows': int,
        'flow_hidden_size': int,
        'beta': float,
        'riemannian_beta': (float, type(None)),
        'epsilon': float,
        'temperature': float,
        'regularization': float,
        'posterior_type': str,
        'loop_mode': str,
        'architecture': str
    }
    
    # Valid values for enum-like parameters
    VALID_VALUES = {
        'posterior_type': ['gaussian', 'riemannian_metric', 'iaf', 'riemannian_rhmc'],
        'loop_mode': ['open', 'closed'],
        'architecture': ['mlp', 'cnn', 'resnet', 'rhvae_rgb']
    }
    
    @classmethod
    def validate(cls, config: Union[DictConfig, Dict[str, Any]], 
                model_type: str = 'rlvae') -> ValidationResult:
        """
        Validate and normalize a model configuration.
        
        Args:
            config: Configuration to validate
            model_type: Type of model ('base', 'rlvae', 'flow_vae', 'modular')
            
        Returns:
            ValidationResult with validation status and normalized config
        """
        errors = []
        warnings_list = []
        
        # Convert to DictConfig if needed
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        elif not isinstance(config, DictConfig):
            try:
                config = OmegaConf.create(dict(config))
            except Exception as e:
                errors.append(f"Cannot convert config to DictConfig: {e}")
                return ValidationResult(False, config, errors, warnings_list)
        
        # Make a copy to avoid modifying original
        config = OmegaConf.create(config)
        
        # Check required parameters
        required = cls.REQUIRED_PARAMS.get(model_type, cls.REQUIRED_PARAMS['base'])
        for param in required:
            if param not in config or config[param] is None:
                errors.append(f"Required parameter '{param}' is missing")
        
        # Add defaults for missing parameters
        for param, default_value in cls.DEFAULTS.items():
            if param not in config or config[param] is None:
                # Handle struct mode for parameter assignment
                original_struct = OmegaConf.is_struct(config)
                OmegaConf.set_struct(config, False)
                config[param] = default_value
                OmegaConf.set_struct(config, original_struct)
                warnings_list.append(f"Using default value for '{param}': {default_value}")
        
        # Validate parameter types
        for param, expected_type in cls.PARAM_TYPES.items():
            if param in config and config[param] is not None:
                if not isinstance(config[param], expected_type):
                    try:
                        # Try to convert
                        if expected_type == int:
                            config[param] = int(config[param])
                        elif expected_type == float:
                            config[param] = float(config[param])
                        elif expected_type == str:
                            config[param] = str(config[param])
                        else:
                            errors.append(f"Parameter '{param}' has wrong type. Expected {expected_type}, got {type(config[param])}")
                    except (ValueError, TypeError):
                        errors.append(f"Cannot convert parameter '{param}' to {expected_type}")
        
        # Validate enum values
        for param, valid_values in cls.VALID_VALUES.items():
            if param in config and config[param] not in valid_values:
                errors.append(f"Parameter '{param}' has invalid value '{config[param]}'. Valid values: {valid_values}")
        
        # Cross-parameter validation
        errors.extend(cls._validate_cross_params(config))
        warnings_list.extend(cls._check_cross_param_warnings(config))
        
        # Normalize specific parameters
        config = cls._normalize_parameters(config)
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, config, errors, warnings_list)
    
    @classmethod
    def _validate_cross_params(cls, config: DictConfig) -> List[str]:
        """Validate cross-parameter consistency."""
        errors = []
        
        # Check input_dim format
        if 'input_dim' in config:
            input_dim = config.input_dim
            if isinstance(input_dim, (list, tuple)):
                if len(input_dim) not in [1, 3]:  # 1D or 3D (C, H, W)
                    errors.append(f"input_dim must be 1D or 3D, got {len(input_dim)}D: {input_dim}")
                if any(not isinstance(d, int) or d <= 0 for d in input_dim):
                    errors.append(f"All input_dim values must be positive integers: {input_dim}")
            elif not isinstance(input_dim, int) or input_dim <= 0:
                errors.append(f"input_dim must be positive integer or tuple of positive integers: {input_dim}")
        
        # Check latent_dim
        if 'latent_dim' in config:
            if config.latent_dim <= 0:
                errors.append(f"latent_dim must be positive: {config.latent_dim}")
        
        # Check n_flows vs sequence_length consistency
        if 'n_flows' in config and 'sequence_length' in config:
            if config.n_flows != config.sequence_length - 1:
                errors.append(f"n_flows ({config.n_flows}) should equal sequence_length - 1 ({config.sequence_length - 1})")
        
        # Check beta values
        if 'beta' in config and config.beta <= 0:
            errors.append(f"beta must be positive: {config.beta}")
        
        if 'riemannian_beta' in config and config.riemannian_beta is not None and config.riemannian_beta <= 0:
            errors.append(f"riemannian_beta must be positive: {config.riemannian_beta}")
        
        return errors
    
    @classmethod
    def _check_cross_param_warnings(cls, config: DictConfig) -> List[str]:
        """Check for potential issues that are warnings, not errors."""
        warnings_list = []
        
        # Warn about riemannian_beta defaulting to beta
        if 'riemannian_beta' not in config or config.riemannian_beta is None:
            if 'beta' in config:
                config.riemannian_beta = config.beta
                warnings_list.append(f"riemannian_beta not set, defaulting to beta value: {config.beta}")
        
        # Warn about potential architecture mismatches
        if 'input_dim' in config and 'architecture' in config:
            input_dim = config.input_dim
            arch = config.architecture
            
            if isinstance(input_dim, (list, tuple)) and len(input_dim) == 3:
                # 3D input (images)
                if arch == 'mlp':
                    warnings_list.append("Using MLP architecture with 3D input - consider CNN architecture")
            elif isinstance(input_dim, int) or (isinstance(input_dim, (list, tuple)) and len(input_dim) == 1):
                # 1D input
                if arch in ['cnn', 'resnet']:
                    warnings_list.append("Using CNN/ResNet architecture with 1D input - consider MLP architecture")
        
        # Warn about high latent dimensions with complex architectures
        if 'latent_dim' in config and config.latent_dim > 50:
            warnings_list.append(f"High latent dimension ({config.latent_dim}) may cause training instability")
        
        return warnings_list
    
    @classmethod
    def _normalize_parameters(cls, config: DictConfig) -> DictConfig:
        """Normalize parameter formats."""
        # Ensure input_dim is tuple
        if 'input_dim' in config:
            if isinstance(config.input_dim, list):
                config.input_dim = tuple(config.input_dim)
            elif isinstance(config.input_dim, int):
                config.input_dim = (config.input_dim,)
        
        # Ensure posterior configuration is properly structured
        if 'posterior_type' in config and 'posterior' not in config:
            config.posterior = {'type': config.posterior_type}
        elif 'posterior' in config and isinstance(config.posterior, str):
            config.posterior = {'type': config.posterior}
        
        # Ensure metric configuration exists for Riemannian models
        if config.get('posterior_type') in ['riemannian_metric', 'riemannian_rhmc']:
            if 'metric' not in config:
                config.metric = {
                    'temperature': config.get('temperature', 0.1),
                    'regularization': config.get('regularization', 0.01)
                }
        
        return config


def validate_model_config(config: Union[DictConfig, Dict[str, Any]], 
                         model_type: str = 'rlvae') -> ValidationResult:
    """
    Convenience function to validate a model configuration.
    
    Args:
        config: Configuration to validate
        model_type: Type of model
        
    Returns:
        ValidationResult
    """
    return ConfigValidator.validate(config, model_type)


def ensure_valid_config(config: Union[DictConfig, Dict[str, Any]], 
                       model_type: str = 'rlvae') -> DictConfig:
    """
    Validate config and return normalized version, raising exception if invalid.
    
    Args:
        config: Configuration to validate
        model_type: Type of model
        
    Returns:
        Validated and normalized config
        
    Raises:
        ValueError: If configuration is invalid
    """
    result = validate_model_config(config, model_type)
    
    if not result.is_valid:
        error_msg = "Configuration validation failed:\n" + "\n".join(result.errors)
        raise ValueError(error_msg)
    
    # Print warnings if any
    for warning in result.warnings:
        warnings.warn(warning)
    
    return result.config
