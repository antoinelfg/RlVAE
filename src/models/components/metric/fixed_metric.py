"""
Fixed Metric implementation for pretrained metrics.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from ...base import Metric, register
from ...base.mixins import LoggingMixin, DeviceMixin, NumericalStabilityMixin


@register("fixed_metric")
class FixedMetric(Metric, LoggingMixin, DeviceMixin, NumericalStabilityMixin):
    """Fixed metric loaded from pretrained components."""
    
    def __init__(
        self,
        latent_dim: int,
        metric_path: Optional[str] = None,
        temperature: float = 0.1,
        regularization: float = 0.01,
        normalize_for_kl: str = "geomean"
    ):
        super().__init__()
        LoggingMixin.__init__(self)
        NumericalStabilityMixin.__init__(self)
        
        self.latent_dim = latent_dim
        self.temperature = torch.tensor(temperature)
        self.regularization = torch.tensor(regularization)
        self.normalize_for_kl = normalize_for_kl
        
        # Initialize metric components
        self.centroids = None
        self.M_matrices = None
        
        # Load metric if path provided
        if metric_path:
            self.load_metric(metric_path)
    
    def load_metric(self, metric_path: str):
        """Load pretrained metric components."""
        print(f"🔧 Loading fixed metric from: {metric_path}")
        
        metric_data = torch.load(metric_path, map_location='cpu', weights_only=False)
        
        # Extract components
        self.centroids = metric_data.get("centroids", metric_data.get("metric_centroids", None))
        if self.centroids is None:
            raise ValueError("No centroids found in metric data")
        
        self.M_matrices = metric_data.get("M_matrices", metric_data.get("metric_vars", None))
        if self.M_matrices is None and "M_i_flat" in metric_data:
            M_flat = metric_data["M_i_flat"]
            self.M_matrices = torch.diag_embed(M_flat)
        if self.M_matrices is None:
            raise ValueError("No metric matrices found")
        
        print(f"✅ Loaded fixed metric: {len(self.centroids)} centroids")
    
    def _compute_metric(self, z: torch.Tensor) -> torch.Tensor:
        """Compute metric using RBF interpolation."""
        if self.centroids is None or self.M_matrices is None:
            # Fallback to identity
            batch_size = z.shape[0]
            return torch.eye(self.latent_dim, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Move components to device
        centroids = self.centroids.to(z.device)
        M_matrices = self.M_matrices.to(z.device)
        temperature = self.temperature.to(z.device)
        regularization = self.regularization.to(z.device)
        
        # Compute distances and weights
        diff = centroids.unsqueeze(0) - z.unsqueeze(1)  # (B, K, D)
        distances = torch.norm(diff, dim=-1) ** 2
        weights = torch.exp(-distances / (temperature ** 2))
        
        # Weighted combination of metric matrices
        weighted_M = M_matrices.unsqueeze(0) * weights.unsqueeze(-1).unsqueeze(-1)
        G = weighted_M.sum(dim=1) + regularization * torch.eye(self.latent_dim, device=z.device)
        
        return G
    
    def _normalize_metric(self, G: torch.Tensor) -> torch.Tensor:
        """Normalize metric for KL computation."""
        if self.normalize_for_kl == "geomean":
            eigenvals = torch.linalg.eigvalsh(G)
            geomean = torch.exp(torch.mean(torch.log(eigenvals + self.eps), dim=1))
            G_normalized = G / geomean.unsqueeze(-1).unsqueeze(-1)
        elif self.normalize_for_kl == "trace":
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
        G = self._compute_metric(z)
        G_normalized = self._normalize_metric(G)
        
        # Log statistics
        self.log("fixed_metric_loaded", self.centroids is not None)
        if self.centroids is not None:
            self.log("metric_eigenvals_min", torch.linalg.eigvalsh(G_normalized).min().item())
            self.log("metric_eigenvals_max", torch.linalg.eigvalsh(G_normalized).max().item())
        
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
        
        return G_inv
