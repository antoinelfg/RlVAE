"""
Adapter to bridge RLVAE metric format to geodesic_toolbox CoMetric interface.

This module provides the necessary adapter classes to use trained RLVAE metrics
with the geodesic_toolbox for computing geodesic trajectories.
"""

import torch
import numpy as np
from typing import Optional, Union, Tuple

# Optional geodesic_toolbox import
try:
    from geodesic_toolbox import CoMetric, CentroidsCometric, IdentityCoMetric
    GEODESIC_TOOLBOX_AVAILABLE = True
except ImportError:
    GEODESIC_TOOLBOX_AVAILABLE = False
    CoMetric = None
    CentroidsCometric = None
    IdentityCoMetric = None

from .unified_model_adapter import UnifiedModelAdapter


class RLVAEGeodesicAdapter:
    """
    Adapter class to convert RLVAE model metrics to geodesic_toolbox compatible format.
    
    This class extracts the learned Riemannian metric from a trained RLVAE model
    and creates a geodesic_toolbox CoMetric object that can be used for geodesic
    computation.
    """
    
    def __init__(self, model, device: Optional[torch.device] = None):
        """
        Initialize the adapter with a trained RLVAE model.
        
        Args:
            model: Trained RLVAE model with learned metric
            device: Device to use for computations
        """
        # Use unified adapter for robust model handling
        self.unified_adapter = UnifiedModelAdapter(model, device)
        self.model = self.unified_adapter.model
        self.device = self.unified_adapter.device
        self._cometric = None
        
    def extract_metric_components(self) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Extract metric components from the RLVAE model using unified adapter.
        
        Returns:
            Tuple of (centroids, metric_matrices, temperature)
        """
        try:
            # Use unified adapter for robust extraction
            centroids = self.unified_adapter.extract_centroids()
            metric_matrices = self.unified_adapter.extract_metric_matrices()
            temperature = self.unified_adapter.extract_temperature()
            
            if centroids is not None and metric_matrices is not None:
                print(f"✅ Successfully extracted metric with {centroids.shape[0]} centroids")
                return centroids, metric_matrices, temperature
            else:
                raise AttributeError(f"Missing metric components: centroids={centroids is not None}, matrices={metric_matrices is not None}")
                
        except Exception as e:
            print(f"⚠️ Failed to extract metric components: {e}")
            print("🔄 Falling back to identity metric")
            return self._create_fallback_metric()
    
    def _create_fallback_metric(self) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Create a fallback identity metric when extraction fails.
        
        Returns:
            Tuple of (centroids, metric_matrices, temperature) for identity metric
        """
        # Use unified adapter's fallback centroids
        centroids = self.unified_adapter._create_fallback_centroids()
        latent_dim = centroids.shape[1]
        num_centroids = centroids.shape[0]
        
        # Identity matrices for each centroid
        metric_matrices = torch.eye(latent_dim, device=self.device, dtype=torch.float64).unsqueeze(0).repeat(num_centroids, 1, 1)
        temperature = 0.1
        
        return centroids, metric_matrices, temperature
    
    def create_cometric(self) -> CoMetric:
        """
        Create a geodesic_toolbox compatible CoMetric object.
        
        Returns:
            CoMetric object that can be used with geodesic solvers
        """
        if self._cometric is not None:
            return self._cometric
            
        try:
            centroids, metric_matrices, temperature = self.extract_metric_components()
            
            # Create base cometric (identity)
            base_cometric = IdentityCoMetric()
            
            # Compute cometric values at centroids (inverse of metric matrices)
            cometric_centroids = torch.inverse(metric_matrices)
            
            # Create CentroidsCometric
            self._cometric = CentroidsCometric(
                centroids=centroids,
                cometric_centroids=cometric_centroids,
                temperature=temperature
            )
            
            print(f"✅ Created geodesic cometric with {centroids.shape[0]} centroids, temperature={temperature}")
            return self._cometric
            
        except Exception as e:
            print(f"⚠️ Failed to create cometric: {e}")
            print("🔄 Using identity cometric as fallback")
            self._cometric = IdentityCoMetric()
            return self._cometric
    
    def get_latent_bounds(self, margin: float = 0.5) -> Tuple[float, float, float, float]:
        """
        Get reasonable bounds for the latent space based on centroids.
        
        Args:
            margin: Additional margin around the centroids
            
        Returns:
            Tuple of (x_min, x_max, y_min, y_max)
        """
        return self.unified_adapter.get_latent_bounds(margin)
    
    def sample_latent_points(self, n_points: int = 10) -> torch.Tensor:
        """
        Sample representative points from the latent space for geodesic computation.
        
        Args:
            n_points: Number of points to sample
            
        Returns:
            Tensor of shape (n_points, latent_dim) with sampled points
        """
        return self.unified_adapter.sample_latent_points(n_points)
