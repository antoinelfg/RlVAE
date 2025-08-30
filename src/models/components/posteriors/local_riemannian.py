"""
Local Riemannian Posterior implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from ...base import Posterior, register
from ...base.mixins import LoggingMixin, DeviceMixin, NumericalStabilityMixin


@register("local_riemannian_posterior")
class LocalRiemannianPosterior(Posterior, LoggingMixin, DeviceMixin, NumericalStabilityMixin):
    """Local Riemannian posterior with Σ = α G(μ)."""
    
    def __init__(
        self,
        latent_dim: int,
        alpha: float = 0.5,
        eps_chol: float = 1e-6
    ):
        super().__init__()
        LoggingMixin.__init__(self)
        NumericalStabilityMixin.__init__(self, eps=eps_chol)
        
        self.latent_dim = latent_dim
        self.alpha = alpha
    
    def sample(self, mu: torch.Tensor, log_var: torch.Tensor, metric=None) -> torch.Tensor:
        """
        Sample from local Riemannian posterior.
        
        Args:
            mu: Mean of shape (B, D)
            log_var: Log variance of shape (B, D) (not used for Riemannian)
            metric: Metric object with G() method
            
        Returns:
            Samples of shape (B, D)
        """
        batch_size = mu.shape[0]
        device = mu.device
        
        if metric is None:
            # Fallback to standard Gaussian
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(mu)
            return mu + std * eps
        
        # Compute metric at mean
        G_mu = metric.G(mu)  # (B, D, D)
        
        # Compute covariance Σ = α G(μ)
        Sigma = self.alpha * G_mu
        
        # Cholesky decomposition for sampling
        L = self.safe_cholesky(Sigma)  # (B, D, D)
        
        # Sample from standard normal
        eps = torch.randn(batch_size, self.latent_dim, device=device)
        
        # Transform to posterior samples: z = μ + L ε
        z = mu + torch.bmm(L, eps.unsqueeze(-1)).squeeze(-1)
        
        # Log statistics
        self.log("posterior_alpha", self.alpha)
        self.log("posterior_samples_mean", z.mean().item())
        self.log("posterior_samples_std", z.std().item())
        
        return z
    
    def log_prob(self, z: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, metric=None) -> torch.Tensor:
        """
        Compute log probability of samples.
        
        Args:
            z: Samples of shape (B, D)
            mu: Mean of shape (B, D)
            log_var: Log variance of shape (B, D) (not used for Riemannian)
            metric: Metric object with G() method
            
        Returns:
            Log probabilities of shape (B,)
        """
        if metric is None:
            # Fallback to standard Gaussian
            std = torch.exp(0.5 * log_var)
            return -0.5 * torch.sum(((z - mu) / std) ** 2 + 2 * log_var, dim=1)
        
        # Compute metric at mean
        G_mu = metric.G(mu)  # (B, D, D)
        
        # Compute covariance Σ = α G(μ)
        Sigma = self.alpha * G_mu
        
        # Compute log determinant
        log_det_Sigma = self.safe_logdet(Sigma)
        
        # Compute quadratic form
        diff = z - mu  # (B, D)
        Sigma_inv = self.safe_inverse(Sigma)  # (B, D, D)
        quad_form = torch.sum(diff.unsqueeze(1) * torch.bmm(Sigma_inv, diff.unsqueeze(-1)).squeeze(-1), dim=1)
        
        # Log probability
        log_prob = -0.5 * (self.latent_dim * torch.log(2 * torch.pi) + log_det_Sigma + quad_form)
        
        return log_prob
    
    def extra_logs(self) -> Dict[str, Any]:
        """Return extra logging information."""
        return {
            "posterior_type": "local_riemannian",
            "alpha": self.alpha
        }
