"""
Learned Riemannian Metric implementation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from ...base import Metric, register
from ...base.mixins import LoggingMixin, DeviceMixin, NumericalStabilityMixin


@register("learned_metric")
class LearnedMetric(Metric, LoggingMixin, DeviceMixin, NumericalStabilityMixin):
    """Learned Riemannian metric using MLP network."""
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list = [256, 256],
        temperature: float = 0.1,
        regularization: float = 0.01,
        normalize_for_kl: str = "geomean",  # "geomean", "trace", "none"
        eps_chol: float = 1e-6,
        activation: str = "relu"
    ):
        super().__init__()
        LoggingMixin.__init__(self)
        NumericalStabilityMixin.__init__(self, eps=eps_chol)
        
        self.latent_dim = latent_dim
        self.temperature = torch.tensor(temperature)
        self.regularization = torch.tensor(regularization)
        self.normalize_for_kl = normalize_for_kl
        
        # Build MLP for metric computation
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer for metric parameters
        # We output the lower triangular part of the metric matrix
        self.mlp = nn.Sequential(*layers)
        self.metric_head = nn.Linear(prev_dim, latent_dim * (latent_dim + 1) // 2)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "leaky_relu":
            return nn.LeakyReLU()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _build_metric_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """Build positive definite metric matrix from network output."""
        batch_size = z.shape[0]
        
        # Get network output
        hidden = self.mlp(z)
        metric_params = self.metric_head(hidden)
        
        # Build lower triangular matrix
        L = torch.zeros(batch_size, self.latent_dim, self.latent_dim, device=z.device)
        
        # Fill lower triangular part
        idx = 0
        for i in range(self.latent_dim):
            for j in range(i + 1):
                L[:, i, j] = metric_params[:, idx]
                idx += 1
        
        # Ensure diagonal is positive (for positive definiteness)
        L[:, torch.arange(self.latent_dim), torch.arange(self.latent_dim)] = torch.exp(
            L[:, torch.arange(self.latent_dim), torch.arange(self.latent_dim)]
        )
        
        # Compute metric matrix G = L L^T
        G = torch.bmm(L, L.transpose(1, 2))
        
        # Add regularization term
        G = G + self.regularization * torch.eye(self.latent_dim, device=z.device).unsqueeze(0)
        
        return G
    
    def _normalize_metric(self, G: torch.Tensor) -> torch.Tensor:
        """Normalize metric for KL computation."""
        if self.normalize_for_kl == "geomean":
            # Normalize by geometric mean of eigenvalues
            eigenvals = torch.linalg.eigvalsh(G)
            geomean = torch.exp(torch.mean(torch.log(eigenvals + self.eps), dim=1))
            G_normalized = G / geomean.unsqueeze(-1).unsqueeze(-1)
        elif self.normalize_for_kl == "trace":
            # Normalize by trace
            trace = torch.diagonal(G, dim1=1, dim2=2).sum(dim=1)
            G_normalized = G / trace.unsqueeze(-1).unsqueeze(-1)
        elif self.normalize_for_kl == "none":
            G_normalized = G
        else:
            raise ValueError(f"Unknown normalization: {self.normalize_for_kl}")
        
        return G_normalized
    
    def G(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute metric tensor G(z).
        
        Args:
            z: Latent points of shape (B, D)
            
        Returns:
            Metric tensor of shape (B, D, D)
        """
        G = self._build_metric_matrix(z)
        G_normalized = self._normalize_metric(G)
        
        # Log statistics
        self.log("metric_eigenvals_min", torch.linalg.eigvalsh(G_normalized).min().item())
        self.log("metric_eigenvals_max", torch.linalg.eigvalsh(G_normalized).max().item())
        self.log("metric_det", torch.det(G_normalized).mean().item())
        
        return G_normalized
    
    def G_inv(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse metric tensor G^{-1}(z).
        
        Args:
            z: Latent points of shape (B, D)
            
        Returns:
            Inverse metric tensor of shape (B, D, D)
        """
        G = self.G(z)
        G_inv = self.safe_inverse(G)
        
        # Log statistics
        self.log("metric_inv_eigenvals_min", torch.linalg.eigvalsh(G_inv).min().item())
        self.log("metric_inv_eigenvals_max", torch.linalg.eigvalsh(G_inv).max().item())
        
        return G_inv
