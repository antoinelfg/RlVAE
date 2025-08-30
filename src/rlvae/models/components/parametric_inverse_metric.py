#!/usr/bin/env python3
"""
Parametric Inverse Metric Tensor
===============================

This module implements a parametric inverse metric tensor where the lower triangular
matrices Lψᵢ are parametrized using neural networks.

The metric follows the formula:
G⁻¹(z) = Σᵢ Lψᵢ Lψᵢᵀ exp(-||z - cᵢ||² / T²) + λI_d

where:
- Lψᵢ: lower triangular matrices parametrized using neural networks
- T: temperature to smooth the metric
- cᵢ: centroids
- λ: regularization factor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import math


class LowerTriangularNetwork(nn.Module):
    """
    Neural network that outputs the parameters for a lower triangular matrix.
    
    For a d×d lower triangular matrix, we need d(d+1)/2 parameters.
    The network outputs these parameters and we construct the matrix.
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_params = latent_dim * (latent_dim + 1) // 2  # d(d+1)/2 for lower triangular
        
        # Neural network to output the parameters
        layers = []
        input_dim = latent_dim  # Input is the centroid position
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, self.num_params))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to ensure positive diagonal elements."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.out_features == self.num_params:
                    # Output layer - initialize to produce identity-like matrices
                    nn.init.normal_(module.weight, mean=0.0, std=0.1)
                    # Initialize bias to produce positive diagonal elements
                    bias = torch.zeros(module.bias.shape)
                    idx = 0
                    for i in range(self.latent_dim):
                        bias[idx] = 1.0  # Ensure positive diagonal
                        idx += i + 1
                    module.bias.data = bias
                else:
                    # Hidden layers
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
    
    def forward(self, centroid: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate lower triangular matrix parameters.
        
        Args:
            centroid: Centroid position [latent_dim]
            
        Returns:
            L: Lower triangular matrix [latent_dim, latent_dim]
        """
        # Get parameters from network
        params = self.network(centroid)  # [num_params]
        
        # Construct lower triangular matrix
        L = torch.zeros(self.latent_dim, self.latent_dim, device=centroid.device)
        
        # Fill the lower triangular part
        idx = 0
        for i in range(self.latent_dim):
            for j in range(i + 1):  # Lower triangular
                if i == j:  # Diagonal element (should be positive)
                    L[i, j] = F.softplus(params[idx]) + 1e-6  # Ensure positive
                else:  # Off-diagonal element
                    L[i, j] = params[idx]
                idx += 1
        
        # Scale the matrix to have reasonable determinant
        det_L = torch.det(L)
        if det_L > 0:
            # Scale to have determinant around 1
            scale_factor = (1.0 / det_L) ** (1.0 / self.latent_dim)
            L = L * scale_factor
        
        # Debug: Check if matrix is valid
        if torch.det(L) == 0:
            print(f"⚠️  Warning: L matrix has zero determinant!")
            print(f"   Diagonal elements: {torch.diag(L)}")
            print(f"   Params range: [{params.min():.3f}, {params.max():.3f}]")
        
        return L


class ParametricInverseMetricTensor(nn.Module):
    """
    Parametric Inverse Metric Tensor using neural networks.
    
    The metric is constructed as:
    G⁻¹(z) = Σᵢ Lψᵢ Lψᵢᵀ exp(-||z - cᵢ||² / T²) + λI_d
    
    where Lψᵢ are lower triangular matrices parametrized by neural networks.
    """
    
    def __init__(
        self,
        latent_dim: int,
        n_centroids: int,
        temperature: float = 1.0,
        regularization: float = 1e-6,  # Reduced from 1e-4
        hidden_dim: int = 64,
        num_layers: int = 2,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_centroids = n_centroids
        self.temperature = temperature
        self.regularization = regularization
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize centroids randomly
        self.centroids = nn.Parameter(
            torch.randn(n_centroids, latent_dim, device=self.device) * 2.0
        )
        
        # Create neural networks for each centroid
        self.l_triangular_nets = nn.ModuleList([
            LowerTriangularNetwork(latent_dim, hidden_dim, num_layers)
            for _ in range(n_centroids)
        ])
        
        # Move to device
        self.to(self.device)
        
        print(f"🔧 ParametricInverseMetricTensor initialized")
        print(f"   - Latent dimension: {latent_dim}")
        print(f"   - Number of centroids: {n_centroids}")
        print(f"   - Temperature: {temperature}")
        print(f"   - Regularization: {regularization}")
        print(f"   - Hidden dimension: {hidden_dim}")
        print(f"   - Number of layers: {num_layers}")
    
    def get_lower_triangular_matrices(self) -> torch.Tensor:
        """
        Get all lower triangular matrices Lψᵢ for current centroids.
        
        Returns:
            L_matrices: [n_centroids, latent_dim, latent_dim]
        """
        L_matrices = []
        for i in range(self.n_centroids):
            L = self.l_triangular_nets[i](self.centroids[i])
            L_matrices.append(L)
        
        return torch.stack(L_matrices)  # [n_centroids, latent_dim, latent_dim]
    
    def compute_weights(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute weights wᵢ(z) = exp(-||z - cᵢ||² / T²)
        
        Args:
            z: Query points [batch_size, latent_dim]
            
        Returns:
            weights: [batch_size, n_centroids]
        """
        # Compute squared distances
        # z: [batch_size, latent_dim]
        # centroids: [n_centroids, latent_dim]
        # distances: [batch_size, n_centroids]
        distances = torch.cdist(z, self.centroids, p=2) ** 2
        
        # Compute weights using Gaussian kernel
        weights = torch.exp(-distances / (self.temperature ** 2))
        
        # Normalize weights to sum to 1 for each point
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        return weights
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the inverse metric tensor G⁻¹(z).
        
        Args:
            z: Query points [batch_size, latent_dim]
            
        Returns:
            G_inv: Inverse metric tensor [batch_size, latent_dim, latent_dim]
            log_det_G_inv: Log determinant of G⁻¹ [batch_size]
        """
        batch_size = z.shape[0]
        
        # Get lower triangular matrices
        L_matrices = self.get_lower_triangular_matrices()  # [n_centroids, latent_dim, latent_dim]
        
        # Compute weights
        weights = self.compute_weights(z)  # [batch_size, n_centroids]
        
        # Initialize G⁻¹ with regularization
        G_inv = torch.eye(self.latent_dim, device=self.device).unsqueeze(0).expand(
            batch_size, -1, -1
        ) * self.regularization
        
        # Add weighted contributions
        for i in range(self.n_centroids):
            L = L_matrices[i]  # [latent_dim, latent_dim]
            L_LT = L @ L.T  # [latent_dim, latent_dim]
            
            # Add contribution for each batch element
            for b in range(batch_size):
                weight = weights[b, i]
                G_inv[b] += weight * L_LT
        
        # Add regularization term
        G_inv += torch.eye(self.latent_dim, device=self.device).unsqueeze(0) * self.regularization
        
        # Compute log determinant
        log_det_G_inv = torch.logdet(G_inv)
        
        return G_inv, log_det_G_inv
    
    def get_metric_info(self) -> dict:
        """Get information about the current metric state."""
        L_matrices = self.get_lower_triangular_matrices()
        
        # Compute determinants of L matrices
        dets_L = torch.det(L_matrices)
        
        # Compute determinants of L L^T matrices
        L_LT_matrices = torch.matmul(L_matrices, L_matrices.transpose(-2, -1))
        dets_L_LT = torch.det(L_LT_matrices)
        
        return {
            'centroids': self.centroids.data.clone(),
            'L_matrices': L_matrices.data.clone(),
            'L_LT_matrices': L_LT_matrices.data.clone(),
            'dets_L': dets_L.data.clone(),
            'dets_L_LT': dets_L_LT.data.clone(),
            'temperature': self.temperature,
            'regularization': self.regularization
        }
    
    @classmethod
    def from_model_data(
        cls,
        model,
        latent_data: torch.Tensor,
        n_centroids: int = 25,
        temperature: float = 1.0,
        regularization: float = 1e-4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        device: Optional[torch.device] = None
    ) -> 'ParametricInverseMetricTensor':
        """
        Create a parametric metric from model data.
        
        Args:
            model: The model (not used in parametric version)
            latent_data: Latent representations [n_samples, latent_dim]
            n_centroids: Number of centroids
            temperature: Temperature parameter
            regularization: Regularization parameter
            hidden_dim: Hidden dimension for neural networks
            num_layers: Number of layers in neural networks
            device: Device to use
            
        Returns:
            ParametricInverseMetricTensor instance
        """
        latent_dim = latent_data.shape[1]
        device = device or latent_data.device
        
        # Create metric
        metric = cls(
            latent_dim=latent_dim,
            n_centroids=n_centroids,
            temperature=temperature,
            regularization=regularization,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            device=device
        )
        
        # Initialize centroids using k-means or random selection
        if n_centroids <= len(latent_data):
            # Use random selection for simplicity
            indices = torch.randperm(len(latent_data))[:n_centroids]
            metric.centroids.data = latent_data[indices].clone()
        else:
            # Use random initialization
            metric.centroids.data = torch.randn(n_centroids, latent_dim, device=device) * 2.0
        
        print(f"✅ Created parametric metric with {n_centroids} centroids")
        print(f"   Temperature: {temperature}, Regularization: {regularization}")
        
        return metric


def test_parametric_metric():
    """Test the parametric metric implementation."""
    print("🧪 TESTING PARAMETRIC METRIC")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 16
    n_centroids = 5
    batch_size = 10
    
    # Create test data
    latent_data = torch.randn(100, latent_dim, device=device)
    
    # Create metric
    metric = ParametricInverseMetricTensor(
        latent_dim=latent_dim,
        n_centroids=n_centroids,
        temperature=1.0,
        regularization=1e-4,
        hidden_dim=32,
        num_layers=2,
        device=device
    )
    
    # Test forward pass
    z = torch.randn(batch_size, latent_dim, device=device)
    G_inv, log_det = metric(z)
    
    print(f"✅ Forward pass successful")
    print(f"   G⁻¹ shape: {G_inv.shape}")
    print(f"   log_det shape: {log_det.shape}")
    print(f"   det(G⁻¹) range: [{torch.exp(log_det).min():.1f}, {torch.exp(log_det).max():.1f}]")
    
    # Test weight computation
    weights = metric.compute_weights(z)
    print(f"✅ Weight computation successful")
    print(f"   Weights shape: {weights.shape}")
    print(f"   Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    
    # Test L matrices
    L_matrices = metric.get_lower_triangular_matrices()
    print(f"✅ L matrices computation successful")
    print(f"   L matrices shape: {L_matrices.shape}")
    print(f"   L det range: [{torch.det(L_matrices).min():.1f}, {torch.det(L_matrices).max():.1f}]")
    
    # Test from_model_data
    metric2 = ParametricInverseMetricTensor.from_model_data(
        None, latent_data, n_centroids=3, device=device
    )
    print(f"✅ from_model_data successful")
    
    print(f"\n🎉 All tests passed!")


if __name__ == "__main__":
    test_parametric_metric() 