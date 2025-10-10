"""
Unified RLVAE Interface
======================

This module provides a unified interface for all RLVAE model variants,
ensuring consistent method signatures and output formats across different
model architectures.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union, Protocol, runtime_checkable
from abc import ABC, abstractmethod
from omegaconf import DictConfig
import warnings


@runtime_checkable
class RLVAEProtocol(Protocol):
    """Protocol defining the expected interface for RLVAE models."""
    
    def forward(self, x: torch.Tensor) -> Union[Dict[str, torch.Tensor], Any]:
        """Forward pass through the model."""
        ...
    
    def encode(self, x: torch.Tensor) -> Union[Dict[str, torch.Tensor], Any]:
        """Encode input to latent space."""
        ...
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent codes to reconstructions."""
        ...


class StandardModelOutput:
    """Standardized model output format."""
    
    def __init__(self, **kwargs):
        # Core outputs (always present)
        self.z = kwargs.get('z', kwargs.get('latent_samples', None))
        self.mu = kwargs.get('mu', kwargs.get('means', None))
        self.log_var = kwargs.get('log_var', kwargs.get('log_covariance', kwargs.get('logvar', None)))
        self.reconstruction = kwargs.get('reconstruction', kwargs.get('recon_x', kwargs.get('x_recon', None)))
        
        # Loss components (optional)
        self.total_loss = kwargs.get('total_loss', kwargs.get('loss', None))
        self.recon_loss = kwargs.get('recon_loss', kwargs.get('reconstruction_loss', None))
        self.kl_loss = kwargs.get('kl_loss', kwargs.get('kld_loss', None))
        self.flow_loss = kwargs.get('flow_loss', None)
        self.loop_penalty = kwargs.get('loop_penalty', None)
        
        # Riemannian-specific outputs (optional)
        self.riemannian_kl = kwargs.get('riemannian_kl', None)
        self.metric_reg = kwargs.get('metric_reg', None)
        
        # Store all original outputs for backward compatibility
        self._raw_outputs = kwargs
    
    def __getitem__(self, key: str):
        """Allow dict-like access for backward compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        return self._raw_outputs.get(key)
    
    def __contains__(self, key: str):
        """Support 'in' operator."""
        return hasattr(self, key) or key in self._raw_outputs
    
    def get(self, key: str, default=None):
        """Dict-like get method."""
        try:
            return self[key]
        except KeyError:
            return default
    
    def keys(self):
        """Return all available keys."""
        attrs = [attr for attr in dir(self) if not attr.startswith('_') and not callable(getattr(self, attr))]
        return set(attrs) | set(self._raw_outputs.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {}
        for key in self.keys():
            value = self[key]
            if value is not None:
                result[key] = value
        return result


class UnifiedRLVAEInterface(nn.Module):
    """
    Unified interface wrapper for all RLVAE model variants.
    
    This class wraps any RLVAE model to provide a consistent interface
    with standardized method signatures and output formats.
    """
    
    def __init__(self, model: nn.Module, model_type: str = "unknown"):
        super().__init__()
        # Set wrapped_model first to avoid recursion issues
        object.__setattr__(self, 'wrapped_model', model)
        object.__setattr__(self, 'model_type', model_type)
        object.__setattr__(self, '_device', None)
        
        # Detect model capabilities
        self._detect_capabilities()
    
    def _detect_capabilities(self):
        """Detect what methods and attributes the wrapped model has."""
        self.has_encode = hasattr(self.wrapped_model, 'encode') or hasattr(self.wrapped_model, 'encoder')
        self.has_decode = hasattr(self.wrapped_model, 'decode') or hasattr(self.wrapped_model, 'decoder')
        self.has_metric = (hasattr(self.wrapped_model, 'G') or 
                          hasattr(self.wrapped_model, 'metric') or
                          hasattr(self.wrapped_model, 'get_metric'))
        self.has_sample_posterior = hasattr(self.wrapped_model, 'sample_posterior')
        
        # Check for different attribute names for common properties
        self.latent_dim = self._get_attr_safe(['latent_dim', 'latent_size', 'z_dim'], default=2)
        self.input_dim = self._get_attr_safe(['input_dim', 'input_size'], default=(3, 64, 64))
        
    def _get_attr_safe(self, attr_names: list, default=None):
        """Safely get attribute with multiple possible names."""
        for name in attr_names:
            if hasattr(self.wrapped_model, name):
                return getattr(self.wrapped_model, name)
        return default
    
    @property
    def device(self):
        """Get model device."""
        if self._device is None:
            try:
                self._device = next(self.wrapped_model.parameters()).device
            except (StopIteration, AttributeError):
                self._device = torch.device('cpu')
        return self._device
    
    def forward(self, x: torch.Tensor) -> StandardModelOutput:
        """
        Unified forward pass with standardized output format.
        
        Args:
            x: Input tensor
            
        Returns:
            StandardModelOutput with consistent field names
        """
        # Call the wrapped model's forward method
        raw_output = self.wrapped_model(x)
        
        # Convert to standardized format
        return self._standardize_output(raw_output)
    
    def _standardize_output(self, raw_output) -> StandardModelOutput:
        """Convert any model output to standardized format."""
        if isinstance(raw_output, dict):
            return StandardModelOutput(**raw_output)
        elif hasattr(raw_output, '__dict__'):
            # Handle objects with attributes (like ModelOutput)
            output_dict = {}
            for key, value in raw_output.__dict__.items():
                if not key.startswith('_'):
                    output_dict[key] = value
            return StandardModelOutput(**output_dict)
        elif hasattr(raw_output, 'recon_x') and hasattr(raw_output, 'z'):
            # Handle pythae ModelOutput objects
            output_dict = {
                'reconstruction': raw_output.recon_x,
                'z': raw_output.z,
                'mu': getattr(raw_output, 'mu', None),
                'log_var': getattr(raw_output, 'log_var', None),
                'loss': getattr(raw_output, 'loss', None),
            }
            return StandardModelOutput(**output_dict)
        else:
            # Fallback: assume it's a reconstruction tensor
            warnings.warn(f"Unknown output format from {self.model_type}, assuming reconstruction tensor")
            return StandardModelOutput(reconstruction=raw_output)
    
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode input to latent space with standardized output.
        
        Args:
            x: Input tensor
            
        Returns:
            Dict with 'mu', 'log_var', and optionally 'z' keys
        """
        if hasattr(self.wrapped_model, 'encode'):
            result = self.wrapped_model.encode(x)
            
            # Standardize encoder output
            if isinstance(result, dict):
                return {
                    'mu': result.get('mu', result.get('embedding', result.get('mean', None))),
                    'log_var': result.get('log_var', result.get('log_covariance', result.get('logvar', None))),
                    'z': result.get('z', result.get('latent', None))
                }
            elif hasattr(result, 'embedding'):
                # Handle encoder objects with attributes
                return {
                    'mu': result.embedding,
                    'log_var': getattr(result, 'log_covariance', None),
                    'z': getattr(result, 'z', None)
                }
            else:
                # Assume it's just the mean
                return {'mu': result, 'log_var': None, 'z': None}
                
        elif hasattr(self.wrapped_model, 'encoder'):
            # Use encoder directly
            encoder_output = self.wrapped_model.encoder(x)
            if hasattr(encoder_output, 'embedding'):
                return {
                    'mu': encoder_output.embedding,
                    'log_var': getattr(encoder_output, 'log_covariance', None),
                    'z': None
                }
            else:
                return {'mu': encoder_output, 'log_var': None, 'z': None}
        else:
            # Fallback: use forward pass and extract latent info
            output = self.forward(x)
            return {
                'mu': output.mu,
                'log_var': output.log_var,
                'z': output.z
            }
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes to reconstructions.
        
        Args:
            z: Latent codes
            
        Returns:
            Reconstructed data
        """
        if hasattr(self.wrapped_model, 'decode'):
            return self.wrapped_model.decode(z)
        elif hasattr(self.wrapped_model, 'decoder'):
            return self.wrapped_model.decoder(z)
        else:
            raise NotImplementedError(f"Model {self.model_type} does not support decoding")
    
    def get_metric(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get Riemannian metric tensor at given points.
        
        Args:
            z: Latent points
            
        Returns:
            Metric tensor G(z) or None if not available
        """
        if hasattr(self.wrapped_model, 'G'):
            return self.wrapped_model.G(z)
        elif hasattr(self.wrapped_model, 'get_metric'):
            return self.wrapped_model.get_metric(z)
        elif hasattr(self.wrapped_model, 'metric') and hasattr(self.wrapped_model.metric, 'G'):
            return self.wrapped_model.metric.G(z)
        else:
            return None
    
    def get_metric_inv(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get inverse Riemannian metric tensor at given points.
        
        Args:
            z: Latent points
            
        Returns:
            Inverse metric tensor G^{-1}(z) or None if not available
        """
        if hasattr(self.wrapped_model, 'G_inv'):
            return self.wrapped_model.G_inv(z)
        elif hasattr(self.wrapped_model, 'get_metric_inv'):
            return self.wrapped_model.get_metric_inv(z)
        elif hasattr(self.wrapped_model, 'metric') and hasattr(self.wrapped_model.metric, 'G_inv'):
            return self.wrapped_model.metric.G_inv(z)
        else:
            # Try to compute from metric if available
            G = self.get_metric(z)
            if G is not None:
                try:
                    return torch.linalg.inv(G + 1e-6 * torch.eye(G.shape[-1], device=G.device).unsqueeze(0).expand_as(G))
                except Exception:
                    return None
            return None
    
    def sample_posterior(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Sample from posterior distribution.
        
        Args:
            mu: Posterior mean
            log_var: Posterior log variance
            
        Returns:
            Sampled latent codes
        """
        if hasattr(self.wrapped_model, 'sample_posterior'):
            return self.wrapped_model.sample_posterior(mu, log_var)
        else:
            # Standard reparameterization trick
            std = torch.exp(0.5 * log_var) if log_var is not None else torch.ones_like(mu)
            eps = torch.randn_like(mu)
            return mu + eps * std
    
    def parameters(self):
        """Return model parameters (required for device detection)."""
        return self.wrapped_model.parameters()
    
    def eval(self):
        """Set model to evaluation mode."""
        self.wrapped_model.eval()
        return self
    
    def train(self, mode: bool = True):
        """Set model to training mode."""
        self.wrapped_model.train(mode)
        return self
    
    def to(self, device):
        """Move model to device."""
        self.wrapped_model.to(device)
        self._device = device
        return self
    
    def __getattr__(self, name):
        """Delegate unknown attributes to wrapped model."""
        # Avoid infinite recursion by checking if we're accessing wrapped_model itself
        if name == 'wrapped_model':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        try:
            return getattr(self.wrapped_model, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def create_unified_interface(model: nn.Module, model_type: str = None) -> UnifiedRLVAEInterface:
    """
    Create a unified interface for any RLVAE model.
    
    Args:
        model: The model to wrap
        model_type: Optional model type identifier
        
    Returns:
        UnifiedRLVAEInterface wrapper
    """
    if model_type is None:
        model_type = type(model).__name__
    
    return UnifiedRLVAEInterface(model, model_type)


def is_unified_interface(model) -> bool:
    """Check if a model is already using the unified interface."""
    return isinstance(model, UnifiedRLVAEInterface)


def ensure_unified_interface(model: nn.Module) -> UnifiedRLVAEInterface:
    """Ensure a model uses the unified interface, wrapping if necessary."""
    if is_unified_interface(model):
        return model
    else:
        return create_unified_interface(model)
