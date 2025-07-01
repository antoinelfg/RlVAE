"""
Base Riemannian Sampler
======================

Abstract base class for all Riemannian sampling strategies.
"""

import torch
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseRiemannianSampler(ABC):
    """
    Abstract base class for Riemannian sampling strategies.
    
    All samplers should inherit from this class and implement the required methods.
    """
    
    def __init__(self, model):
        """
        Initialize the sampler with a reference to the model.
        
        Args:
            model: The Riemannian VAE model that provides metric tensor functions
        """
        self.model = model
        self.device = next(model.parameters()).device
        
    @abstractmethod
    def sample_riemannian_latents(self, mu: torch.Tensor, log_var: torch.Tensor, 
                                 method: str = 'enhanced') -> torch.Tensor:
        """
        Sample latent codes using Riemannian geometry.
        
        Args:
            mu: Posterior mean [batch_size, latent_dim]
            log_var: Posterior log variance [batch_size, latent_dim]
            method: Sampling method to use
            
        Returns:
            Sampled latent codes [batch_size, latent_dim]
        """
        pass
    
    @abstractmethod
    def sample_prior(self, num_samples: int, method: str = 'geodesic') -> torch.Tensor:
        """
        Sample from the Riemannian prior.
        
        Args:
            num_samples: Number of samples to generate
            method: Prior sampling method to use
            
        Returns:
            Prior samples [num_samples, latent_dim]
        """
        pass
    
    def validate_metric_availability(self) -> bool:
        """
        Check if the model has the required metric tensor components.
        
        Returns:
            True if metric components are available, False otherwise
        """
        required_attrs = ['centroids_tens', 'M_tens', 'G', 'G_inv']
        return all(hasattr(self.model, attr) for attr in required_attrs)
    
    def get_sampling_methods(self) -> Dict[str, str]:
        """
        Get available sampling methods for this sampler.
        
        Returns:
            Dictionary mapping method names to descriptions
        """
        return {
            'enhanced': 'Enhanced Riemannian sampling with centroid influence',
            'geodesic': 'Geodesic-aware sampling along manifold paths',
            'basic': 'Basic metric-aware sampling',
            'standard': 'Standard reparameterization (no Riemannian)'
        }
    
    def get_sampler_info(self) -> Dict[str, Any]:
        """
        Get information about this sampler.
        
        Returns:
            Dictionary with sampler information
        """
        return {
            'sampler_type': self.__class__.__name__,
            'available_methods': list(self.get_sampling_methods().keys()),
            'metric_available': self.validate_metric_availability(),
            'device': str(self.device)
        } 