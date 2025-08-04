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
import os


# === Trainable Metric Architectures ===
class MetricMLP(nn.Module):
    """MLP for mapping z to a symmetric positive-definite matrix."""
    def __init__(self, latent_dim, hidden_dim=64, n_layers=3):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, latent_dim * latent_dim))
        self.net = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, z):
        # z: [batch, latent_dim]
        G_flat = self.net(z)  # [batch, latent_dim * latent_dim]
        G = G_flat.view(-1, self.latent_dim, self.latent_dim)
        # Symmetrize
        G = 0.5 * (G + G.transpose(-1, -2))
        # Softplus on diagonal for positive-definiteness
        diag = torch.diagonal(G, dim1=-2, dim2=-1)
        diag = torch.nn.functional.softplus(diag) + 1e-3
        G = G.clone()
        G.diagonal(dim1=-2, dim2=-1).copy_(diag)
        return G

class MetricResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return self.relu(out + x)

class MetricResNet(nn.Module):
    """ResNet-style MLP for mapping z to a symmetric positive-definite matrix."""
    def __init__(self, latent_dim, hidden_dim=64, n_blocks=3):
        super().__init__()
        self.fc_in = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([MetricResNetBlock(hidden_dim) for _ in range(n_blocks)])
        self.fc_out = nn.Linear(hidden_dim, latent_dim * latent_dim)
        self.latent_dim = latent_dim
    def forward(self, z):
        x = self.fc_in(z)
        for block in self.blocks:
            x = block(x)
        G_flat = self.fc_out(x)
        G = G_flat.view(-1, self.latent_dim, self.latent_dim)
        G = 0.5 * (G + G.transpose(-1, -2))
        diag = torch.diagonal(G, dim1=-2, dim2=-1)
        diag = torch.nn.functional.softplus(diag) + 1e-3
        G = G.clone()
        G.diagonal(dim1=-2, dim2=-1).copy_(diag)
        return G

class MetricTransformer(nn.Module):
    """Transformer-based metric network for mapping z to a symmetric positive-definite matrix."""
    def __init__(self, latent_dim, n_heads=2, n_layers=2, hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(hidden_dim, latent_dim * latent_dim)
    def forward(self, z):
        # z: [batch, latent_dim] -> [batch, 1, latent_dim]
        x = self.input_proj(z).unsqueeze(1)  # [batch, 1, hidden_dim]
        x = self.transformer(x)  # [batch, 1, hidden_dim]
        x = x.squeeze(1)
        G_flat = self.fc_out(x)
        G = G_flat.view(-1, self.latent_dim, self.latent_dim)
        G = 0.5 * (G + G.transpose(-1, -2))
        diag = torch.diagonal(G, dim1=-2, dim2=-1)
        diag = torch.nn.functional.softplus(diag) + 1e-3
        G = G.clone()
        G.diagonal(dim1=-2, dim2=-1).copy_(diag)
        return G

# Factory for metric architectures
def get_metric_architecture(arch, latent_dim, **kwargs):
    if arch == 'mlp':
        return MetricMLP(latent_dim, hidden_dim=kwargs.get('hidden_dim', 64), n_layers=kwargs.get('n_layers', 3))
    elif arch == 'resnet':
        return MetricResNet(latent_dim, hidden_dim=kwargs.get('hidden_dim', 64), n_blocks=kwargs.get('n_blocks', 3))
    elif arch == 'transformer':
        return MetricTransformer(latent_dim, n_heads=kwargs.get('n_heads', 2), n_layers=kwargs.get('n_layers', 2), hidden_dim=kwargs.get('hidden_dim', 64))
    else:
        raise ValueError(f"Unknown metric architecture: {arch}")


class MetricTensor(nn.Module):
    """
    Riemannian metric tensor computation module.
    Now supports trainable neural metric with optional initialization from a fixed metric.
    """
    def __init__(
        self,
        latent_dim: int,
        temperature: float = 0.1,
        regularization: float = 0.01,
        device: Optional[torch.device] = None,
        trainable: bool = False,
        architecture: str = 'mlp',
        arch_kwargs: Optional[dict] = None,
        init_from_fixed: bool = False,
        fixed_metric_path: Optional[str] = None,
        fit_steps: int = 1000,
        fit_lr: float = 1e-3,
        fit_batch_size: int = 128,
        fit_print_every: int = 200
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainable = trainable
        self.architecture = architecture
        self.arch_kwargs = arch_kwargs or {}
        self._is_loaded = False
        self._diagnostic_counter = 0
        if trainable:
            self.metric_net = get_metric_architecture(architecture, latent_dim, **self.arch_kwargs).to(self.device)
            print("[DEBUG] MetricTensor initialized in trainable mode.")
            first_param = list(self.metric_net.parameters())[0].detach().cpu().numpy().copy()
            print(f"[DEBUG] Initial metric_net first param (mean): {first_param.mean():.6f}")
            if init_from_fixed and fixed_metric_path is not None and os.path.exists(fixed_metric_path):
                print(f"[MetricTensor] Initializing trainable metric from fixed metric at {fixed_metric_path}")
                self.fit_to_fixed_metric(fixed_metric_path, fit_steps, fit_lr, fit_batch_size, fit_print_every)
                first_param = list(self.metric_net.parameters())[0].detach().cpu().numpy().copy()
                print(f"[DEBUG] Post-fit metric_net first param (mean): {first_param.mean():.6f}")
        else:
            self.register_buffer('centroids', torch.empty(0, latent_dim))
            self.register_buffer('metric_matrices', torch.empty(0, latent_dim, latent_dim))
            self.register_buffer('temperature', torch.tensor(temperature))
            self.register_buffer('regularization', torch.tensor(regularization))

    def fit_to_fixed_metric(self, fixed_metric_path, fit_steps=1000, fit_lr=1e-3, fit_batch_size=128, print_every=200):
        """
        Fit the trainable metric network to match a fixed metric loaded from file.
        Uses MSE loss between G_net(z) and G_fixed(z) for random z.
        """
        # Load fixed metric
        import torch
        state = torch.load(fixed_metric_path, map_location=self.device, weights_only=False)
        centroids = state.get('centroids')
        metric_matrices = state.get('metric_matrices')
        if metric_matrices is None:
            metric_matrices = state.get('M_matrices')
        temperature = state.get('temperature', 0.1)
        regularization = state.get('regularization', 0.01)
        fixed_metric = MetricTensor(
            latent_dim=self.latent_dim,
            temperature=temperature,
            regularization=regularization,
            device=self.device
        )
        fixed_metric.load_pretrained(centroids, metric_matrices, temperature, regularization)
        fixed_metric.eval()
        # Fit loop
        optimizer = torch.optim.Adam(self.metric_net.parameters(), lr=fit_lr)
        for step in range(fit_steps):
            z = torch.randn(fit_batch_size, self.latent_dim, device=self.device)
            with torch.no_grad():
                G_fixed = fixed_metric.compute_metric(z)
            G_pred = self.metric_net(z)
            loss = torch.nn.functional.mse_loss(G_pred, G_fixed)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step+1) % print_every == 0 or step == 0:
                print(f"[MetricTensor] Fit step {step+1}/{fit_steps}, MSE loss: {loss.item():.6f}")
        print("[MetricTensor] Trainable metric initialized to match fixed metric.")

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

        # Check for NaN/Inf
        if not torch.isfinite(G_inv).all():
            warnings.warn("G_inv contains non-finite values! Clamping to safe values.")
            G_inv = torch.nan_to_num(G_inv, nan=1.0, posinf=1.0, neginf=1.0)
        return G_inv
    
    def compute_metric(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute metric tensor G(z) = [G^{-1}(z)]^{-1}.
        
        Args:
            z: Latent coordinates [batch_size, latent_dim]
            
        Returns:
            G: Metric tensor [batch_size, latent_dim, latent_dim]
        """
        if self.trainable:
            print("[DEBUG] Using trainable metric_net in compute_metric.")
            first_param = list(self.metric_net.parameters())[0].detach().cpu().numpy().copy()
            print(f"[DEBUG] metric_net first param (mean): {first_param.mean():.6f}")
            return self.metric_net(z)
        else:
            G_inv = self.compute_inverse_metric(z)
            
            try:
                G = torch.linalg.inv(G_inv)
            except torch.linalg.LinAlgError as e:
                warnings.warn(f"Metric tensor inversion failed: {e}. Adding regularization.")
                # Add small regularization and retry
                eps = 1e-6
                G_inv_reg = G_inv + eps * torch.eye(self.latent_dim, device=z.device, dtype=z.dtype).unsqueeze(0)
                G = torch.linalg.inv(G_inv_reg)
                
            # Check for NaN/Inf
            if not torch.isfinite(G).all():
                warnings.warn("G contains non-finite values! Clamping to safe values.")
                G = torch.nan_to_num(G, nan=1.0, posinf=1.0, neginf=1.0)
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
            
            # If you clamp eigenvalues for stability, do it like this:
            eigenvals_G = torch.clamp(eigenvals_G, min=1e-6)
            eigenvals_G_inv = torch.clamp(eigenvals_G_inv, min=1e-6)

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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Make MetricTensor callable: returns G(z) for input z.
        """
        return self.compute_metric(z) 