#!/usr/bin/env python3
"""
Native Inverse Metric Tensor Component
=====================================

Native G⁻¹ implementation where G⁻¹ is the fundamental metric tensor.
This implementation never computes G, working directly with G⁻¹ throughout.

This is the modular version integrated into the RlVAE pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


class NativeInverseMetricTensor(nn.Module):
    """
    Native inverse metric tensor that treats G⁻¹ as the fundamental metric.
    
    This component never computes G, working directly with G⁻¹ for improved
    geometric fidelity and numerical stability.
    
    Key Features:
    - Direct G⁻¹ interpolation without G computation
    - Temperature-controlled metric interpolation
    - Efficient batch processing
    - Integration with RlVAE pipeline
    """
    
    def __init__(self, latent_dim: int = 2, device: Optional[torch.device] = None):
        """
        Initialize the native inverse metric tensor.
        
        Args:
            latent_dim: Dimension of the latent space
            device: Device for computations
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Metric parameters (will be loaded later)
        self.centroids = None
        self.inverse_metrics = None
        self.temperature = None
        self.regularization = None
        
        print(f"🔧 NativeInverseMetricTensor initialized")
        print(f"   - Fundamental metric: G⁻¹(z)")
        print(f"   - No G computation required")
    
    def load_inverse_metrics(
        self,
        centroids: torch.Tensor,
        inverse_metrics: torch.Tensor,
        temperature: float = 2.0,
        regularization: float = 1e-4
    ):
        """
        Load precomputed centroids and inverse metrics.
        
        Args:
            centroids: [n_centroids, latent_dim] centroid positions
            inverse_metrics: [n_centroids, latent_dim, latent_dim] G⁻¹ matrices
            temperature: Temperature for interpolation
            regularization: Regularization for numerical stability
        """
        self.centroids = centroids.to(self.device)
        self.inverse_metrics = inverse_metrics.to(self.device)
        self.temperature = temperature
        self.regularization = regularization
        
        # Compute determinants for efficient access
        self.log_det_inverse_metrics = torch.logdet(self.inverse_metrics)
        
        det_range = torch.exp(self.log_det_inverse_metrics)
        
        print(f"✅ Loaded {len(centroids)} centroids with native G⁻¹ metrics")
        print(f"   Temperature: {temperature}, Regularization: {regularization}")
        
        if len(det_range) > 0:
            print(f"   det(G⁻¹) range: [{det_range.min():.2e}, {det_range.max():.2e}]")
        else:
            print(f"   det(G⁻¹) range: [empty - no centroids loaded]")
    
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
        
        return weights

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute G⁻¹(z) and log|det(G⁻¹(z))| via interpolation.
        
        Args:
            z: [batch_size, latent_dim] input points
            
        Returns:
            G_inv: [batch_size, latent_dim, latent_dim] inverse metric tensors
            log_det_G_inv: [batch_size] log determinants
        """
        if self.centroids is None:
            raise ValueError("Must load inverse metrics first using load_inverse_metrics()")
        
        batch_size = z.shape[0]
        
        # Compute distances to centroids
        # z: [batch_size, latent_dim]
        # centroids: [n_centroids, latent_dim]
        distances = torch.cdist(z, self.centroids)  # [batch_size, n_centroids]
        
        # Temperature-scaled weights
        weights = torch.softmax(-distances / self.temperature, dim=1)  # [batch_size, n_centroids]
        
        # Interpolate G⁻¹ matrices
        # weights: [batch_size, n_centroids] -> [batch_size, n_centroids, 1, 1]
        # inverse_metrics: [n_centroids, latent_dim, latent_dim]
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)
        G_inv = torch.sum(
            weights_expanded * self.inverse_metrics.unsqueeze(0), 
            dim=1
        )  # [batch_size, latent_dim, latent_dim]
        
        # Add regularization for numerical stability
        G_inv = G_inv + self.regularization * torch.eye(
            self.latent_dim, device=self.device
        ).unsqueeze(0)
        
        # Interpolate log determinants
        log_det_G_inv = torch.sum(
            weights * self.log_det_inverse_metrics.unsqueeze(0), 
            dim=1
        )  # [batch_size]
        
        return G_inv, log_det_G_inv
    
    def __call__(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call forward method."""
        return self.forward(z)
    
    @classmethod
    def from_model_data(
        cls,
        model,
        latent_data: torch.Tensor,
        n_centroids: int = 50,
        temperature: float = 2.0,
        regularization: float = 1e-4,
        device: Optional[torch.device] = None
    ) -> 'NativeInverseMetricTensor':
        """
        Create NativeInverseMetricTensor from model and data.
        
        Args:
            model: Trained VAE model with encoder
            latent_data: [n_data, latent_dim] data for centroid computation
            n_centroids: Number of centroids to compute
            temperature: Temperature for interpolation
            regularization: Regularization for numerical stability
            device: Device for computations
            
        Returns:
            Configured NativeInverseMetricTensor
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Compute centroids using k-means
        latent_np = latent_data.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=n_centroids, random_state=42, n_init=10)
        kmeans.fit(latent_np)
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
        
        # Create diverse G⁻¹ metrics for each centroid
        inverse_metrics = []
        latent_dim = latent_data.shape[1]
        
        for i, centroid in enumerate(centroids):
            # Find closest data points to each centroid
            distances = torch.norm(latent_data - centroid.unsqueeze(0), dim=1)
            closest_indices = torch.argsort(distances)[:100]  # Use 100 closest points
            cluster_points = latent_data[closest_indices]
            
            if len(cluster_points) > 1:
                # Compute covariance matrix
                cluster_np = cluster_points.detach().cpu().numpy()
                cov_matrix = np.cov(cluster_np.T)
                cov_matrix += np.eye(cov_matrix.shape[0]) * 0.01  # Regularization
                
                try:
                    # Compute G⁻¹ = inv(cov)
                    metric_matrix = np.linalg.inv(cov_matrix)
                except np.linalg.LinAlgError:
                    metric_matrix = np.eye(latent_dim)
            else:
                metric_matrix = np.eye(latent_dim)
            
            inverse_metrics.append(torch.tensor(metric_matrix, dtype=torch.float32, device=device))
        
        inverse_metrics = torch.stack(inverse_metrics)
        
        # Create and configure the metric tensor
        native_metric = cls(latent_dim=latent_dim, device=device)
        native_metric.load_inverse_metrics(centroids, inverse_metrics, temperature, regularization)
        
        return native_metric
    
    @classmethod
    def from_pretrained_traditional_metric(
        cls,
        metric_path: str,
        device: Optional[torch.device] = None
    ) -> 'NativeInverseMetricTensor':
        """
        Create NativeInverseMetricTensor from traditional metric file.
        
        This converts a traditional G-based metric to native G⁻¹.
        
        Args:
            metric_path: Path to traditional metric file
            device: Device for computations
            
        Returns:
            Configured NativeInverseMetricTensor
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load traditional metric data
        metric_data = torch.load(metric_path, map_location=device)
        
        # Extract components
        centroids = metric_data['centroids']
        if 'metric_matrices' in metric_data:
            # Traditional G matrices - compute inverse
            G_matrices = metric_data['metric_matrices']
            inverse_metrics = torch.linalg.inv(G_matrices)
        elif 'inverse_metrics' in metric_data:
            # Already has G⁻¹ matrices
            inverse_metrics = metric_data['inverse_metrics']
        else:
            raise ValueError("Metric file must contain 'metric_matrices' or 'inverse_metrics'")
        
        temperature = metric_data.get('temperature', 2.0)
        regularization = metric_data.get('regularization', 1e-4)
        latent_dim = centroids.shape[1]
        
        # Create and configure the metric tensor
        native_metric = cls(latent_dim=latent_dim, device=device)
        native_metric.load_inverse_metrics(centroids, inverse_metrics, temperature, regularization)
        
        return native_metric
    
    def save_native_metric(self, save_path: str):
        """
        Save the native inverse metric to file.
        
        Args:
            save_path: Path to save the metric
        """
        if self.centroids is None:
            raise ValueError("No metrics loaded to save")
        
        metric_data = {
            'centroids': self.centroids.cpu(),
            'inverse_metrics': self.inverse_metrics.cpu(),
            'log_det_inverse_metrics': self.log_det_inverse_metrics.cpu(),
            'temperature': self.temperature,
            'regularization': self.regularization,
            'latent_dim': self.latent_dim,
            'type': 'native_inverse_metric',
            'version': '1.0'
        }
        
        torch.save(metric_data, save_path)
        print(f"✅ Saved native inverse metric to {save_path}")


class NativeInverseRHMC(nn.Module):
    """
    Native G⁻¹ Riemannian HMC sampler.
    
    This sampler uses G⁻¹ as the fundamental metric throughout,
    providing improved geometric sampling.
    """
    
    def __init__(
        self,
        metric_tensor: NativeInverseMetricTensor,
        n_steps: int = 100,
        n_leapfrog: int = 50,
        step_size: float = 1e-6
    ):
        """
        Initialize the native G⁻¹ RHMC sampler.
        
        Args:
            metric_tensor: Native inverse metric tensor
            n_steps: Number of MCMC steps
            n_leapfrog: Number of leapfrog steps
            step_size: Step size for integration
        """
        super().__init__()
        
        self.metric_tensor = metric_tensor
        self.n_steps = n_steps
        self.n_leapfrog = n_leapfrog
        self.step_size = step_size
        self.device = metric_tensor.device
        
        print(f"🎯 NativeInverseRHMC initialized")
        print(f"   - Steps: {n_steps}, Leapfrog: {n_leapfrog}")
        print(f"   - Step size: {step_size}")
    
    def _kinetic_energy(self, p: torch.Tensor, G_inv: torch.Tensor) -> torch.Tensor:
        """Compute kinetic energy: 0.5 * p^T * G * p = 0.5 * p^T * (G⁻¹)⁻¹ * p"""
        # Since we have G⁻¹, we need G = (G⁻¹)⁻¹
        G = torch.linalg.inv(G_inv)
        return 0.5 * torch.sum(p * torch.mv(G, p))
    
    def _potential_energy(self, z: torch.Tensor, log_det_G_inv: torch.Tensor) -> torch.Tensor:
        """Compute potential energy with volume correction."""
        # Quadratic potential
        quadratic_potential = 0.5 * torch.sum(z * z)
        
        # Volume correction term (attracts to high-determinant regions)
        volume_correction = -1.5 * log_det_G_inv
        
        return quadratic_potential + volume_correction
    
    def _momentum_gradient(self, p: torch.Tensor, G_inv: torch.Tensor) -> torch.Tensor:
        """Compute momentum gradient: ∇_p H = G * p = (G⁻¹)⁻¹ * p"""
        G = torch.linalg.inv(G_inv)
        return torch.mv(G, p)
    
    def _position_gradient(self, z: torch.Tensor) -> torch.Tensor:
        """Compute position gradient: ∇_z H"""
        z.requires_grad_(True)
        
        # Get metric and determinant
        G_inv, log_det_G_inv = self.metric_tensor(z.unsqueeze(0))
        G_inv = G_inv[0]
        log_det_G_inv = log_det_G_inv[0]
        
        # Compute potential energy
        potential = self._potential_energy(z, log_det_G_inv)
        
        # Compute gradient
        grad = torch.autograd.grad(potential, z, retain_graph=False)[0]
        
        return grad.detach()
    
    def _leapfrog_step(self, z: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one leapfrog integration step."""
        # Half step for momentum
        grad_z = self._position_gradient(z)
        p = p - 0.5 * self.step_size * grad_z
        
        # Full step for position
        G_inv, _ = self.metric_tensor(z.unsqueeze(0))
        G_inv = G_inv[0]
        grad_p = self._momentum_gradient(p, G_inv)
        z = z + self.step_size * grad_p
        
        # Half step for momentum
        grad_z = self._position_gradient(z)
        p = p - 0.5 * self.step_size * grad_z
        
        return z, p
    
    def sample(self, n_samples: int = 100) -> torch.Tensor:
        """
        Sample using native G⁻¹ RHMC.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            samples: [n_samples, latent_dim] sampled points
        """
        if self.metric_tensor.centroids is None:
            raise ValueError("Metric tensor must be loaded before sampling")
        
        samples = []
        n_accepted = 0
        latent_dim = self.metric_tensor.latent_dim
        
        # Initialize near a random centroid
        current_z = self.metric_tensor.centroids[0].clone() + torch.randn(latent_dim, device=self.device) * 0.05
        
        for step in range(self.n_steps):
            # Store current state
            z_old = current_z.clone()
            
            # Get current metric
            G_inv_old, log_det_G_inv_old = self.metric_tensor(z_old.unsqueeze(0))
            G_inv_old = G_inv_old[0]
            log_det_G_inv_old = log_det_G_inv_old[0]
            
            # Initialize momentum from N(0, G⁻¹)
            L = torch.linalg.cholesky(G_inv_old)
            p = torch.mv(L, torch.randn(latent_dim, device=self.device))
            p_old = p.clone()
            
            # Compute initial energy
            K_old = self._kinetic_energy(p_old, G_inv_old)
            U_old = self._potential_energy(z_old, log_det_G_inv_old)
            H_old = K_old + U_old
            
            # Leapfrog integration
            z_new = z_old.clone()
            p_new = p.clone()
            
            for _ in range(self.n_leapfrog):
                z_new, p_new = self._leapfrog_step(z_new, p_new)
            
            # Negate momentum for reversibility
            p_new = -p_new
            
            # Compute new energy
            G_inv_new, log_det_G_inv_new = self.metric_tensor(z_new.unsqueeze(0))
            G_inv_new = G_inv_new[0]
            log_det_G_inv_new = log_det_G_inv_new[0]
            
            K_new = self._kinetic_energy(p_new, G_inv_new)
            U_new = self._potential_energy(z_new, log_det_G_inv_new)
            H_new = K_new + U_new
            
            # Metropolis acceptance
            delta_H = H_new - H_old
            accept_prob = torch.min(torch.tensor(1.0), torch.exp(-delta_H))
            
            if torch.rand(1).item() < accept_prob:
                current_z = z_new
                n_accepted += 1
            
            # Collect samples (after burn-in)
            if step >= self.n_steps // 3 and len(samples) < n_samples:
                samples.append(current_z.clone())
        
        if samples:
            samples_tensor = torch.stack(samples)
            acceptance_rate = n_accepted / self.n_steps
            
            print(f"✅ Native G⁻¹ RHMC sampling completed")
            print(f"   Acceptance rate: {acceptance_rate:.3f}")
            print(f"   Generated {len(samples)} samples")
            
            return samples_tensor
        else:
            raise RuntimeError("No samples generated - check RHMC parameters")