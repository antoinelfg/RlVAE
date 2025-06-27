"""
Riemannian Metric Tensor Module

This module provides clean, modular implementations of Riemannian metric tensor
computations, extracted from the monolithic riemannian_flow_vae.py.

Key Features:
- Metric tensor G(z) and inverse G^{-1}(z) computations
- Temperature-controlled centroid-based metrics
- Efficient batch processing
- Device handling
- Comprehensive error handling and diagnostics
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import warnings


class MetricTensor(nn.Module):
    """
    Riemannian metric tensor computation module.
    
    Implements the centroid-based metric tensor:
    G^{-1}(z) = Σ_k M_k * exp(-||z - c_k||² / T²) + λI
    G(z) = [G^{-1}(z)]^{-1}
    
    Where:
    - z: latent coordinates [batch_size, latent_dim]
    - c_k: centroids [n_centroids, latent_dim]  
    - M_k: metric matrices [n_centroids, latent_dim, latent_dim]
    - T: temperature parameter (scalar)
    - λ: regularization parameter (scalar)
    """
    
    def __init__(
        self,
        latent_dim: int,
        temperature: float = 0.1,
        regularization: float = 0.01,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Metric parameters (will be populated by load_pretrained)
        self.register_buffer('centroids', torch.empty(0, latent_dim))
        self.register_buffer('metric_matrices', torch.empty(0, latent_dim, latent_dim))
        self.register_buffer('temperature', torch.tensor(temperature))
        self.register_buffer('regularization', torch.tensor(regularization))
        
        # Internal state
        self._is_loaded = False
        self._diagnostic_counter = 0
        
    def load_pretrained(
        self,
        centroids: torch.Tensor,
        metric_matrices: torch.Tensor,
        temperature: Optional[float] = None,
        regularization: Optional[float] = None
    ) -> None:
        """
        Load pretrained metric parameters.
        
        Args:
            centroids: Centroid positions [n_centroids, latent_dim]
            metric_matrices: Metric matrices [n_centroids, latent_dim, latent_dim]
            temperature: Temperature override (optional)
            regularization: Regularization override (optional)
        """
        # Validate inputs
        if centroids.shape[1] != self.latent_dim:
            raise ValueError(f"Centroids dimension {centroids.shape[1]} != latent_dim {self.latent_dim}")
        
        if metric_matrices.shape[0] != centroids.shape[0]:
            raise ValueError(f"Number of metric matrices {metric_matrices.shape[0]} != number of centroids {centroids.shape[0]}")
            
        if metric_matrices.shape[1:] != (self.latent_dim, self.latent_dim):
            raise ValueError(f"Metric matrix shape {metric_matrices.shape[1:]} != ({self.latent_dim}, {self.latent_dim})")
        
        # Load parameters using proper buffer registration
        self.register_buffer('centroids', centroids.to(self.device))
        self.register_buffer('metric_matrices', metric_matrices.to(self.device))
        
        if temperature is not None:
            self.register_buffer('temperature', torch.tensor(temperature, device=self.device))
        if regularization is not None:
            self.register_buffer('regularization', torch.tensor(regularization, device=self.device))
            
        self._is_loaded = True
        
        print(f"✅ MetricTensor loaded: {len(centroids)} centroids, T={self.temperature.item():.3f}, λ={self.regularization.item():.3f}")
        
    def compute_inverse_metric(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse metric tensor G^{-1}(z).
        
        Args:
            z: Latent coordinates [batch_size, latent_dim]
            
        Returns:
            G_inv: Inverse metric tensor [batch_size, latent_dim, latent_dim]
        """
        if not self._is_loaded:
            raise RuntimeError("Metric tensor not loaded. Call load_pretrained() first.")
            
        batch_size = z.shape[0]
        
        # Compute distances to all centroids
        # z: [batch_size, latent_dim], centroids: [n_centroids, latent_dim]
        diff = self.centroids.unsqueeze(0) - z.unsqueeze(1)  # [batch_size, n_centroids, latent_dim]
        distances_sq = torch.sum(diff ** 2, dim=-1)  # [batch_size, n_centroids]
        
        # Compute centroid weights
        weights = torch.exp(-distances_sq / (self.temperature ** 2))  # [batch_size, n_centroids]
        
        # Weighted sum of metric matrices
        # weights: [batch_size, n_centroids] -> [batch_size, n_centroids, 1, 1]
        # metric_matrices: [n_centroids, latent_dim, latent_dim] -> [1, n_centroids, latent_dim, latent_dim]
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)
        matrices_expanded = self.metric_matrices.unsqueeze(0)
        
        weighted_matrices = weights_expanded * matrices_expanded  # [batch_size, n_centroids, latent_dim, latent_dim]
        G_inv = weighted_matrices.sum(dim=1)  # [batch_size, latent_dim, latent_dim]
        
        # Add regularization
        regularization_matrix = self.regularization * torch.eye(
            self.latent_dim, device=z.device, dtype=z.dtype
        ).unsqueeze(0).expand(batch_size, -1, -1)
        
        G_inv = G_inv + regularization_matrix
        
        return G_inv
    
    def compute_metric(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute metric tensor G(z) = [G^{-1}(z)]^{-1}.
        
        Args:
            z: Latent coordinates [batch_size, latent_dim]
            
        Returns:
            G: Metric tensor [batch_size, latent_dim, latent_dim]
        """
        G_inv = self.compute_inverse_metric(z)
        
        try:
            G = torch.linalg.inv(G_inv)
        except torch.linalg.LinAlgError as e:
            warnings.warn(f"Metric tensor inversion failed: {e}. Adding regularization.")
            # Add small regularization and retry
            eps = 1e-6
            G_inv_reg = G_inv + eps * torch.eye(self.latent_dim, device=z.device, dtype=z.dtype).unsqueeze(0)
            G = torch.linalg.inv(G_inv_reg)
            
        return G
    
    def compute_log_det_metric(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log determinant of metric tensor log|G(z)|.
        
        Args:
            z: Latent coordinates [batch_size, latent_dim]
            
        Returns:
            log_det: Log determinant [batch_size]
        """
        G = self.compute_metric(z)
        
        try:
            log_det = torch.linalg.slogdet(G).logabsdet
        except torch.linalg.LinAlgError:
            # Fallback: compute via inverse metric
            G_inv = self.compute_inverse_metric(z)
            log_det_inv = torch.linalg.slogdet(G_inv).logabsdet
            log_det = -log_det_inv
            
        return log_det
    
    def compute_riemannian_distance_squared(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute squared Riemannian distance between points.
        
        For points close together, this approximates:
        d²(z1, z2) ≈ (z1 - z2)ᵀ G((z1+z2)/2) (z1 - z2)
        
        Args:
            z1, z2: Latent coordinates [batch_size, latent_dim]
            
        Returns:
            distance_sq: Squared Riemannian distance [batch_size]
        """
        # Compute metric at midpoint
        z_mid = 0.5 * (z1 + z2)
        G_mid = self.compute_metric(z_mid)
        
        # Compute difference vector
        diff = z1 - z2  # [batch_size, latent_dim]
        
        # Compute quadratic form: diff^T G diff
        distance_sq = torch.einsum('bi,bij,bj->b', diff, G_mid, diff)
        
        return distance_sq
    
    def diagnose_metric_properties(self, z: torch.Tensor, verbose: bool = False) -> Dict[str, Any]:
        """
        Analyze metric tensor properties for debugging.
        
        Args:
            z: Sample points [batch_size, latent_dim]
            verbose: Whether to print diagnostic information
            
        Returns:
            diagnostics: Dictionary of metric properties
        """
        with torch.no_grad():
            G = self.compute_metric(z)
            G_inv = self.compute_inverse_metric(z)
            
            # Compute eigenvalues for first sample
            eigenvals_G = torch.linalg.eigvals(G[0]).real
            eigenvals_G_inv = torch.linalg.eigvals(G_inv[0]).real
            
            # Compute determinants and traces
            det_G = torch.linalg.det(G)
            det_G_inv = torch.linalg.det(G_inv)
            trace_G = torch.diagonal(G, dim1=-2, dim2=-1).sum(-1)
            trace_G_inv = torch.diagonal(G_inv, dim1=-2, dim2=-1).sum(-1)
            
            diagnostics = {
                'eigenvals_G_min': eigenvals_G.min().item(),
                'eigenvals_G_max': eigenvals_G.max().item(),
                'eigenvals_G_mean': eigenvals_G.mean().item(),
                'eigenvals_G_inv_min': eigenvals_G_inv.min().item(),
                'eigenvals_G_inv_max': eigenvals_G_inv.max().item(),
                'eigenvals_G_inv_mean': eigenvals_G_inv.mean().item(),
                'condition_number_G': (eigenvals_G.max() / (eigenvals_G.min() + 1e-8)).item(),
                'condition_number_G_inv': (eigenvals_G_inv.max() / (eigenvals_G_inv.min() + 1e-8)).item(),
                'det_G_mean': det_G.mean().item(),
                'det_G_inv_mean': det_G_inv.mean().item(),
                'trace_G_mean': trace_G.mean().item(),
                'trace_G_inv_mean': trace_G_inv.mean().item(),
                'batch_size': z.shape[0],
                'n_centroids': len(self.centroids),
                'temperature': self.temperature.item(),
                'regularization': self.regularization.item(),
            }
            
            if verbose:
                print(f"🔍 METRIC DIAGNOSTICS:")
                print(f"   G eigenvalues: min={diagnostics['eigenvals_G_min']:.3e}, max={diagnostics['eigenvals_G_max']:.3e}, mean={diagnostics['eigenvals_G_mean']:.3e}")
                print(f"   G condition number: {diagnostics['condition_number_G']:.2e}")
                print(f"   det(G): mean={diagnostics['det_G_mean']:.3e}")
                print(f"   trace(G): mean={diagnostics['trace_G_mean']:.3e}")
                print(f"   Batch size: {diagnostics['batch_size']}, Centroids: {diagnostics['n_centroids']}")
                
            return diagnostics
    
    def is_loaded(self) -> bool:
        """Check if metric parameters are loaded."""
        return self._is_loaded
        
    def get_config(self) -> Dict[str, Any]:
        """Get metric tensor configuration."""
        return {
            'latent_dim': self.latent_dim,
            'temperature': self.temperature.item() if self._is_loaded else None,
            'regularization': self.regularization.item() if self._is_loaded else None,
            'n_centroids': len(self.centroids) if self._is_loaded else 0,
            'is_loaded': self._is_loaded,
        } 