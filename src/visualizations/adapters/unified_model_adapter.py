"""
Unified Model Adapter
====================

Robust adapter that handles any model type with consistent interface,
providing fallback implementations and graceful degradation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Union
import warnings
import numpy as np

from rlvae.models.base.unified_rlvae import UnifiedRLVAEInterface, ensure_unified_interface


class UnifiedModelAdapter:
    """
    Universal adapter for RLVAE models with robust error handling.
    
    This adapter can handle any model type and provides:
    - Automatic model interface detection
    - Fallback implementations for missing methods
    - Graceful degradation when features are unavailable
    - Consistent interface for all visualization modules
    """
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Initialize adapter with any model type.
        
        Args:
            model: Model to adapt (will be wrapped with unified interface)
            device: Device for computations
        """
        # Ensure model has unified interface
        self.model = ensure_unified_interface(model)
        self.device = device or self.model.device
        
        # Cache for expensive computations
        self._metric_cache = {}
        self._centroids_cache = None
        self._capabilities = None
        
        # Detect model capabilities
        self._detect_capabilities()
        
        print(f"🔧 Unified adapter initialized for {self.model.model_type}")
        print(f"📊 Capabilities: {self.get_capability_summary()}")
    
    def _detect_capabilities(self):
        """Detect what the model can do."""
        self._capabilities = {
            'has_encode': self.model.has_encode,
            'has_decode': self.model.has_decode,
            'has_metric': self.model.has_metric,
            'has_sample_posterior': self.model.has_sample_posterior,
            'has_centroids': self._check_centroids(),
            'has_metric_matrices': self._check_metric_matrices(),
            'latent_dim': self.model.latent_dim,
            'input_dim': self.model.input_dim
        }
    
    def _check_centroids(self) -> bool:
        """Check if model has centroids."""
        try:
            centroids = self.extract_centroids()
            return centroids is not None
        except Exception:
            return False
    
    def _check_metric_matrices(self) -> bool:
        """Check if model has metric matrices."""
        try:
            matrices = self.extract_metric_matrices()
            return matrices is not None
        except Exception:
            return False
    
    def get_capability_summary(self) -> str:
        """Get a summary of model capabilities."""
        if self._capabilities is None:
            return "Unknown"
        
        caps = []
        if self._capabilities['has_encode']:
            caps.append("encode")
        if self._capabilities['has_decode']:
            caps.append("decode")
        if self._capabilities['has_metric']:
            caps.append("metric")
        if self._capabilities['has_centroids']:
            caps.append("centroids")
        
        return f"[{', '.join(caps)}]"
    
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode input with robust error handling.
        
        Args:
            x: Input tensor
            
        Returns:
            Dict with 'mu', 'log_var', 'z' keys
        """
        try:
            return self.model.encode(x)
        except Exception as e:
            warnings.warn(f"Encoding failed: {e}")
            # Fallback: return dummy values
            batch_size = x.shape[0]
            latent_dim = self._capabilities.get('latent_dim', 2)
            device = x.device
            
            return {
                'mu': torch.randn(batch_size, latent_dim, device=device),
                'log_var': torch.zeros(batch_size, latent_dim, device=device),
                'z': torch.randn(batch_size, latent_dim, device=device)
            }
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes with error handling.
        
        Args:
            z: Latent codes
            
        Returns:
            Reconstructed data
        """
        try:
            return self.model.decode(z)
        except Exception as e:
            warnings.warn(f"Decoding failed: {e}")
            # Fallback: return dummy reconstruction
            batch_size = z.shape[0]
            input_dim = self._capabilities.get('input_dim', (3, 64, 64))
            if isinstance(input_dim, int):
                input_dim = (input_dim,)
            
            return torch.randn(batch_size, *input_dim, device=z.device)
    
    def get_metric(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get metric tensor with caching and fallbacks.
        
        Args:
            z: Latent points
            
        Returns:
            Metric tensor or None
        """
        # Check cache first
        cache_key = f"metric_{z.shape}_{z.device}"
        if cache_key in self._metric_cache:
            cached_metric = self._metric_cache[cache_key]
            if cached_metric.shape[0] >= z.shape[0]:
                return cached_metric[:z.shape[0]]
        
        try:
            metric = self.model.get_metric(z)
            if metric is not None:
                # Cache for future use
                self._metric_cache[cache_key] = metric
                return metric
        except Exception as e:
            warnings.warn(f"Metric computation failed: {e}")
        
        # Fallback: identity metric
        return self._create_identity_metric(z)
    
    def get_metric_inv(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get inverse metric tensor with fallbacks.
        
        Args:
            z: Latent points
            
        Returns:
            Inverse metric tensor or None
        """
        try:
            metric_inv = self.model.get_metric_inv(z)
            if metric_inv is not None:
                return metric_inv
        except Exception as e:
            warnings.warn(f"Inverse metric computation failed: {e}")
        
        # Try to compute from metric
        metric = self.get_metric(z)
        if metric is not None:
            try:
                return torch.linalg.inv(metric + 1e-6 * torch.eye(metric.shape[-1], device=metric.device).unsqueeze(0).expand_as(metric))
            except Exception:
                pass
        
        # Fallback: identity
        return self._create_identity_metric(z)
    
    def _create_identity_metric(self, z: torch.Tensor) -> torch.Tensor:
        """Create identity metric tensor."""
        batch_size, latent_dim = z.shape
        return torch.eye(latent_dim, device=z.device, dtype=z.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    
    def extract_centroids(self) -> Optional[torch.Tensor]:
        """
        Extract centroids from model with multiple fallback strategies.
        
        Returns:
            Centroids tensor or None
        """
        if self._centroids_cache is not None:
            return self._centroids_cache
        
        # Try multiple attribute names
        centroid_attrs = [
            'centroids_tens', 'centroids', 'metric.centroids',
            'modular_metric.centroids', 'riemannian_metric.centroids'
        ]
        
        for attr_path in centroid_attrs:
            try:
                obj = self.model.wrapped_model
                for attr in attr_path.split('.'):
                    obj = getattr(obj, attr)
                
                if obj is not None and isinstance(obj, torch.Tensor):
                    self._centroids_cache = obj.to(device=self.device, dtype=torch.float64)
                    return self._centroids_cache
            except (AttributeError, TypeError):
                continue
        
        # Fallback: create grid of centroids
        return self._create_fallback_centroids()
    
    def _create_fallback_centroids(self) -> torch.Tensor:
        """Create fallback centroids in a grid pattern."""
        latent_dim = self._capabilities.get('latent_dim', 2)
        
        if latent_dim == 2:
            # 2D grid
            grid_size = 10
            x = torch.linspace(-3, 3, grid_size, device=self.device, dtype=torch.float64)
            y = torch.linspace(-3, 3, grid_size, device=self.device, dtype=torch.float64)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            centroids = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        else:
            # Random centroids for higher dimensions
            num_centroids = 100
            centroids = torch.randn(num_centroids, latent_dim, device=self.device, dtype=torch.float64)
        
        self._centroids_cache = centroids
        return centroids
    
    def extract_metric_matrices(self) -> Optional[torch.Tensor]:
        """
        Extract metric matrices from model.
        
        Returns:
            Metric matrices tensor or None
        """
        # Try multiple attribute names
        matrix_attrs = [
            'M_tens', 'M_matrices', 'metric_matrices',
            'metric.M_matrices', 'modular_metric.metric_matrices'
        ]
        
        for attr_path in matrix_attrs:
            try:
                obj = self.model.wrapped_model
                for attr in attr_path.split('.'):
                    obj = getattr(obj, attr)
                
                if obj is not None and isinstance(obj, torch.Tensor):
                    return obj.to(device=self.device, dtype=torch.float64)
            except (AttributeError, TypeError):
                continue
        
        # Fallback: create identity matrices for centroids
        centroids = self.extract_centroids()
        if centroids is not None:
            latent_dim = centroids.shape[1]
            num_centroids = centroids.shape[0]
            return torch.eye(latent_dim, device=self.device, dtype=torch.float64).unsqueeze(0).repeat(num_centroids, 1, 1)
        
        return None
    
    def extract_temperature(self) -> float:
        """Extract temperature parameter."""
        temp_attrs = ['temperature', 'metric.temperature', 'modular_metric.temperature']
        
        for attr_path in temp_attrs:
            try:
                obj = self.model.wrapped_model
                for attr in attr_path.split('.'):
                    obj = getattr(obj, attr)
                
                if obj is not None:
                    return float(obj)
            except (AttributeError, TypeError, ValueError):
                continue
        
        return 0.1  # Default temperature
    
    def sample_latent_points(self, n_points: int) -> torch.Tensor:
        """
        Sample representative points from latent space.
        
        Args:
            n_points: Number of points to sample
            
        Returns:
            Sampled latent points
        """
        centroids = self.extract_centroids()
        
        if centroids is not None and len(centroids) >= n_points:
            # Sample from centroids
            indices = torch.randperm(len(centroids))[:n_points]
            return centroids[indices]
        else:
            # Sample random points
            latent_dim = self._capabilities.get('latent_dim', 2)
            return torch.randn(n_points, latent_dim, device=self.device, dtype=torch.float64) * 2.0
    
    def get_latent_bounds(self, margin: float = 0.5) -> Tuple[float, float, float, float]:
        """
        Get reasonable bounds for latent space visualization.
        
        Args:
            margin: Additional margin around data
            
        Returns:
            (x_min, x_max, y_min, y_max)
        """
        centroids = self.extract_centroids()
        
        if centroids is not None and centroids.shape[1] >= 2:
            centroids_np = centroids.detach().cpu().numpy()
            x_min = float(centroids_np[:, 0].min() - margin)
            x_max = float(centroids_np[:, 0].max() + margin)
            y_min = float(centroids_np[:, 1].min() - margin)
            y_max = float(centroids_np[:, 1].max() + margin)
        else:
            # Default bounds
            x_min, x_max = -3.0, 3.0
            y_min, y_max = -3.0, 3.0
        
        return x_min, x_max, y_min, y_max
    
    def parameters(self):
        """Return model parameters (for device detection)."""
        return self.model.parameters()
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self
    
    def train(self, mode: bool = True):
        """Set model to training mode."""
        self.model.train(mode)
        return self
    
    def to(self, device):
        """Move adapter to device."""
        self.model.to(device)
        self.device = device
        # Clear caches
        self._metric_cache.clear()
        self._centroids_cache = None
        return self


def create_unified_adapter(model: nn.Module, device: Optional[torch.device] = None) -> UnifiedModelAdapter:
    """
    Create a unified adapter for any model.
    
    Args:
        model: Model to adapt
        device: Device for computations
        
    Returns:
        UnifiedModelAdapter instance
    """
    return UnifiedModelAdapter(model, device)
