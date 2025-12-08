"""
Abstract base classes for model components.

This module defines the interfaces that all model components must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union
import torch
import torch.nn as nn


class Encoder(ABC, nn.Module):
    """Abstract base class for encoders."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (B, ...)
            
        Returns:
            Dictionary containing encoded representation, typically:
            - 'embedding': (B, D) or (B, T, D) for sequences
            - 'mu': (B, D) mean of posterior
            - 'log_var': (B, D) log variance of posterior
        """
        pass


class Decoder(ABC, nn.Module):
    """Abstract base class for decoders."""
    
    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.
        
        Args:
            z: Latent tensor of shape (B, D) or (B, T, D)
            
        Returns:
            Reconstructed tensor of shape (B, ...)
        """
        pass


class Metric(ABC, nn.Module):
    """Abstract base class for Riemannian metrics."""
    
    @abstractmethod
    def G(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute metric tensor G(z).
        
        Args:
            z: Latent points of shape (B, D)
            
        Returns:
            Metric tensor of shape (B, D, D)
        """
        pass
    
    @abstractmethod
    def G_inv(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse metric tensor G^{-1}(z).
        
        Args:
            z: Latent points of shape (B, D)
            
        Returns:
            Inverse metric tensor of shape (B, D, D)
        """
        pass


class Flow(ABC, nn.Module):
    """Abstract base class for normalizing flows."""
    
    @abstractmethod
    def forward(self, z: torch.Tensor, t: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply flow transformation.
        
        Args:
            z: Input tensor of shape (B, D)
            t: Optional timestep for temporal flows
            
        Returns:
            Tuple of (transformed_z, log_det_jacobian)
        """
        pass
    
    @abstractmethod
    def inverse(self, z: torch.Tensor, t: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply inverse flow transformation.
        
        Args:
            z: Input tensor of shape (B, D)
            t: Optional timestep for temporal flows
            
        Returns:
            Tuple of (inverse_transformed_z, log_det_jacobian)
        """
        pass


class Prior(ABC, nn.Module):
    """Abstract base class for priors."""
    
    @abstractmethod
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of latent points.
        
        Args:
            z: Latent points of shape (B, D)
            
        Returns:
            Log probabilities of shape (B,)
        """
        pass
    
    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        """
        Sample from prior.
        
        Args:
            n: Number of samples
            
        Returns:
            Samples of shape (n, D)
        """
        pass


class Posterior(ABC, nn.Module):
    """Abstract base class for posteriors."""
    
    @abstractmethod
    def sample(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Sample from posterior.
        
        Args:
            mu: Mean of shape (B, D)
            log_var: Log variance of shape (B, D)
            
        Returns:
            Samples of shape (B, D)
        """
        pass
    
    @abstractmethod
    def log_prob(self, z: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of samples.
        
        Args:
            z: Samples of shape (B, D)
            mu: Mean of shape (B, D)
            log_var: Log variance of shape (B, D)
            
        Returns:
            Log probabilities of shape (B,)
        """
        pass
    
    def extra_logs(self) -> Dict[str, Any]:
        """
        Return extra logging information.
        
        Returns:
            Dictionary of extra information to log
        """
        return {}


class Sampler(ABC):
    """Abstract base class for samplers."""
    
    @abstractmethod
    def sample(self, n: int, **kwargs) -> torch.Tensor:
        """
        Generate samples.
        
        Args:
            n: Number of samples
            **kwargs: Additional sampling parameters
            
        Returns:
            Samples of shape (n, D)
        """
        pass


class KLLoss(ABC, nn.Module):
    """Abstract base class for KL divergence losses."""
    
    @abstractmethod
    def forward(
        self, 
        mu: torch.Tensor, 
        z: torch.Tensor, 
        metric: Optional[Metric] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute KL divergence loss.
        
        Args:
            mu: Posterior mean of shape (B, D)
            z: Sampled latents of shape (B, D)
            metric: Optional metric for Riemannian KL
            **kwargs: Additional arguments
            
        Returns:
            KL loss scalar
        """
        pass


class ReconstructionLoss(ABC, nn.Module):
    """Abstract base class for reconstruction losses."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            x: Original input of shape (B, ...)
            x_recon: Reconstructed input of shape (B, ...)
            
        Returns:
            Reconstruction loss scalar
        """
        pass


class ELBOLoss(ABC, nn.Module):
    """Abstract base class for ELBO losses."""
    
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        z: torch.Tensor,
        log_var: Optional[torch.Tensor] = None,
        flow_log_det: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ELBO loss.
        
        Args:
            x: Original input
            x_recon: Reconstructed input
            mu: Posterior mean
            z: Sampled latents
            log_var: Posterior log variance
            flow_log_det: Flow log determinant
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing all loss components
        """
        pass
