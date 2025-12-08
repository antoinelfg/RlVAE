"""
Common mixins for model components.

This module provides reusable mixins for logging, metrics, and other
shared functionality across model components.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class LoggingMixin:
    """Mixin for logging functionality."""
    
    def __init__(self):
        self._logs: Dict[str, Any] = {}
    
    def log(self, key: str, value: Any) -> None:
        """Log a value."""
        self._logs[key] = value
    
    def get_logs(self) -> Dict[str, Any]:
        """Get all logged values."""
        return self._logs.copy()
    
    def clear_logs(self) -> None:
        """Clear all logged values."""
        self._logs.clear()


class KLMetricsMixin:
    """Mixin for KL divergence metrics."""
    
    def __init__(self):
        self._kl_metrics: Dict[str, torch.Tensor] = {}
    
    def log_kl_metric(self, key: str, value: torch.Tensor) -> None:
        """Log a KL-related metric."""
        self._kl_metrics[key] = value.detach()
    
    def get_kl_metrics(self) -> Dict[str, torch.Tensor]:
        """Get all KL metrics."""
        return self._kl_metrics.copy()
    
    def clear_kl_metrics(self) -> None:
        """Clear all KL metrics."""
        self._kl_metrics.clear()


class NumericalStabilityMixin:
    """Mixin for numerical stability helpers."""
    
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
    
    def safe_logdet(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compute log determinant safely."""
        try:
            return torch.logdet(matrix)
        except RuntimeError:
            # Add small diagonal term for stability
            matrix_stable = matrix + self.eps * torch.eye(
                matrix.shape[-1], device=matrix.device, dtype=matrix.dtype
            )
            return torch.logdet(matrix_stable)
    
    def safe_cholesky(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compute Cholesky decomposition safely."""
        try:
            return torch.linalg.cholesky(matrix)
        except RuntimeError:
            # Add small diagonal term for stability
            matrix_stable = matrix + self.eps * torch.eye(
                matrix.shape[-1], device=matrix.device, dtype=matrix.dtype
            )
            return torch.linalg.cholesky(matrix_stable)
    
    def safe_inverse(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compute matrix inverse safely."""
        try:
            return torch.linalg.inv(matrix)
        except RuntimeError:
            # Use pseudo-inverse as fallback
            return torch.linalg.pinv(matrix)
    
    def clamp_eigenvalues(self, matrix: torch.Tensor, min_eig: float = 1e-6) -> torch.Tensor:
        """Clamp eigenvalues to ensure positive definiteness."""
        # Compute eigendecomposition
        eigenvals, eigenvecs = torch.linalg.eigh(matrix)
        
        # Clamp eigenvalues
        eigenvals_clamped = torch.clamp(eigenvals, min=min_eig)
        
        # Reconstruct matrix
        return eigenvecs @ torch.diag(eigenvals_clamped) @ eigenvecs.T


class DeviceMixin:
    """Mixin for device management."""
    
    @property
    def device(self) -> torch.device:
        """Get the device of the first parameter."""
        if hasattr(self, 'parameters'):
            params = list(self.parameters())
            if params:
                return params[0].device
        return torch.device('cpu')
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the same device as this module."""
        return tensor.to(self.device)


class ConfigMixin:
    """Mixin for configuration management."""
    
    def __init__(self):
        self._config: Optional[Dict[str, Any]] = None
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set configuration."""
        self._config = config
    
    def get_config(self) -> Optional[Dict[str, Any]]:
        """Get configuration."""
        return self._config
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        if self._config is None:
            return default
        return self._config.get(key, default)
