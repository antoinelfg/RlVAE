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

from .metric_utils import compute_metric_weights, normalize_metric_atoms

_INVERSE_FALLBACK_COUNTER = 0


def inverse_fallback_count() -> int:
    """Return how many times the inverse fallback returned identity."""
    return _INVERSE_FALLBACK_COUNTER


# === Numerical Stability Utilities ===
def _cholesky_spd(M: torch.Tensor, jitter: float = 1e-6, max_tries: int = 6) -> torch.Tensor:
    """
    Robust Cholesky decomposition with jitter and retry logic.
    
    Args:
        M: Symmetric positive definite matrix [batch, d, d]
        jitter: Initial jitter value
        max_tries: Maximum number of retry attempts
        
    Returns:
        L: Lower triangular Cholesky factor [batch, d, d]
    """
    batch_size, d = M.shape[0], M.shape[-1]
    device = M.device
    dtype = M.dtype
    
    # Ensure symmetry
    M = 0.5 * (M + M.transpose(-1, -2))
    
    base_jitter = float(max(jitter, 0.0))
    current_jitter = 0.0
    for attempt in range(max_tries):
        try:
            # Add jitter to diagonal
            jitter_matrix = current_jitter * torch.eye(d, device=device, dtype=dtype).unsqueeze(0)
            M_reg = M + jitter_matrix
            
            # Try Cholesky decomposition
            if hasattr(torch.linalg, 'cholesky_ex'):
                L, info = torch.linalg.cholesky_ex(M_reg)
                if info.eq(0).all():
                    return L
            else:
                L = torch.linalg.cholesky(M_reg)
                return L
                
        except (torch.linalg.LinAlgError, RuntimeError):
            pass
            
        # Increase jitter for next attempt
        if current_jitter == 0.0:
            current_jitter = base_jitter if base_jitter > 0 else 1e-6
        else:
            current_jitter *= 10.0
    
    # Final fallback with large jitter
    M_final = M + 1e-3 * torch.eye(d, device=device, dtype=dtype).unsqueeze(0)
    try:
        if hasattr(torch.linalg, 'cholesky_ex'):
            L, info = torch.linalg.cholesky_ex(M_final)
            return L
        else:
            return torch.linalg.cholesky(M_final)
    except (torch.linalg.LinAlgError, RuntimeError):
        # Ultimate fallback: return identity
        warnings.warn("Cholesky decomposition failed even with large jitter. Using identity.")
        return torch.eye(d, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)


def _robust_inverse_from_cholesky(G: torch.Tensor) -> torch.Tensor:
    """
    Compute inverse using Cholesky decomposition for numerical stability.
    
    Args:
        G: Symmetric positive definite matrix [batch, d, d]
        
    Returns:
        G_inv: Inverse matrix [batch, d, d]
    """
    batch_size, d = G.shape[:2]
    device = G.device
    dtype = G.dtype
    
    # Handle low precision dtypes by promoting to float32
    if dtype in [torch.float16, torch.bfloat16]:
        G_work = G.float()
        promote_back = True
    else:
        G_work = G
        promote_back = False

    # Default to Cholesky-based inversion with jitter escalation.
    try:
        L = _cholesky_spd(G_work)
        # Solve L L^T X = I for X = G^{-1}
        I = torch.eye(d, device=device, dtype=G_work.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Forward substitution: L Y = I
        Y = torch.linalg.solve_triangular(L, I, upper=False)
        # Backward substitution: L^T G_inv = Y
        G_inv = torch.linalg.solve_triangular(L.transpose(-1, -2), Y, upper=True)
        
        return G_inv.to(dtype) if promote_back else G_inv
        
    except Exception as e:
        warnings.warn(f"Cholesky-based inversion failed: {e}. Using identity fallback.")
        global _INVERSE_FALLBACK_COUNTER
        _INVERSE_FALLBACK_COUNTER += 1
        return torch.eye(d, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)


def _robust_logdet_from_cholesky(G: torch.Tensor) -> torch.Tensor:
    """
    Compute log determinant using Cholesky decomposition for numerical stability.
    
    Args:
        G: Symmetric positive definite matrix [batch, d, d]
        
    Returns:
        log_det: Log determinant [batch]
    """
    # Handle low precision dtypes by promoting to float32
    if G.dtype in [torch.float16, torch.bfloat16]:
        G_work = G.float()
        promote_back = True
    else:
        G_work = G
        promote_back = False
    
    try:
        L = _cholesky_spd(G_work)
        # log|G| = 2 * sum(log(diag(L))) - guard against negative tiny values
        diagL = torch.diagonal(L, dim1=-2, dim2=-1).abs() + 1e-18
        log_det = 2.0 * torch.log(diagL).sum(dim=-1)
        return log_det.to(G.dtype) if promote_back else log_det
        
    except Exception as e:
        warnings.warn(f"Cholesky-based logdet failed: {e}. Using slogdet fallback.")
        try:
            sign, log_det = torch.linalg.slogdet(G_work)
            return log_det.to(G.dtype) if promote_back else log_det
        except torch.linalg.LinAlgError:
            warnings.warn("All logdet methods failed. Returning zero.")
            return torch.zeros(G.shape[0], device=G.device, dtype=G.dtype)


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
        regularization: float = 1e-2,
        device: Optional[torch.device] = None,
        trainable: bool = False,
        architecture: str = 'mlp',
        arch_kwargs: Optional[dict] = None,
        init_from_fixed: bool = False,
        fixed_metric_path: Optional[str] = None,
        fit_steps: int = 1000,
        fit_lr: float = 1e-3,
        fit_batch_size: int = 128,
        fit_print_every: int = 200,
        normalize_weight_sum: bool = False,
        weight_kernel: str = 'mahalanobis_normed',
        weight_metric_normalization: str = 'trace',
        topk_weights: Optional[int] = 0,
        regularization_mode: str = 'precision',
        use_background_identity: Optional[bool] = True,
        normalize_atoms_unit_det: bool = False,
    ):
        super().__init__()
        # ---- core config
        self.latent_dim = int(latent_dim)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainable = bool(trainable)
        self.architecture = str(architecture)
        self.arch_kwargs = dict(arch_kwargs or {})
        self._is_loaded = False
        self._diagnostic_counter = 0

        # Representation (atoms are precision by construction)
        self.atoms_are = 'ginv'

        # ---- weighting / mixing (defaults emulate the original working code)
        self.normalize_weight_sum = bool(normalize_weight_sum)
        self.weight_kernel = str(weight_kernel).lower()
        self.weight_metric_normalization = str(weight_metric_normalization).lower()
        self.topk_weights = int(topk_weights) if topk_weights is not None else None
        if self.topk_weights is not None and self.topk_weights <= 0:
            self.topk_weights = None

        # ---- regularization placement (precision vs metric)
        self.regularization_mode = str(regularization_mode).lower()
        if self.regularization_mode not in {'precision', 'metric'}:
            raise ValueError("regularization_mode must be 'precision' or 'metric'")

        self.normalize_atoms_unit_det = bool(normalize_atoms_unit_det)

        # ---- background identity mixing (now enabled by default)
        self.use_background_identity = bool(use_background_identity)
        self.bg_strength = 1e-3
        self.bg_radius = None
        self.bg_floor = 0.0

        self.eig_floor_abs = 1e-4
        self.eig_ceiling = 1e3

        # ---- trainable vs fixed metric
        self.temperature_fallback = float(max(temperature, 1e-6))
        self.regularization_fallback = float(max(regularization, 0.0))

        if self.trainable:
            # Build neural metric that outputs G(z) directly
            self.metric_net = get_metric_architecture(self.architecture, self.latent_dim, **self.arch_kwargs).to(self.device)
            if init_from_fixed and fixed_metric_path:
                print(f"[MetricTensor] Initializing trainable metric from fixed: {fixed_metric_path}")
                self.fit_to_fixed_metric(
                    fixed_metric_path,
                    fit_steps=fit_steps,
                    fit_lr=fit_lr,
                    fit_batch_size=fit_batch_size,
                    fit_print_every=fit_print_every,
                )
        else:
            # Register fixed-atom buffers (centroids + precision atoms)
            self.register_buffer('centroids', torch.empty(0, self.latent_dim))
            self.register_buffer('metric_matrices', torch.empty(0, self.latent_dim, self.latent_dim))
            self.register_buffer('temperature', torch.tensor(self.temperature_fallback))
            self.register_buffer('regularization', torch.tensor(self.regularization_fallback))

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
        C = centroids.to(self.device)
        # Auto-fix common shape issues: expect [K, D]. If we get [D, K], transpose.
        if C.dim() == 2 and C.shape[1] != self.latent_dim and C.shape[0] == self.latent_dim:
            C = C.transpose(0, 1)
        self.register_buffer('centroids', C)
        
        # Enforce SPD atoms before registering
        M = metric_matrices.to(self.device)
        # Expect [K, D, D]. If we get [D, D, K], permute.
        if M.dim() == 3 and M.shape[0] == self.latent_dim and M.shape[1] == self.latent_dim and M.shape[2] != self.latent_dim:
            M = M.permute(2, 0, 1)
        M = 0.5 * (M + M.transpose(-1, -2))
        diag = torch.diagonal(M, dim1=-2, dim2=-1)
        diag = torch.nn.functional.softplus(diag) + 1e-6
        M = M.clone()
        M.diagonal(dim1=-2, dim2=-1).copy_(diag)
        
        if self.normalize_atoms_unit_det:
            with torch.no_grad():
                L = _cholesky_spd(M)
                logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1).abs() + 1e-18).sum(-1)
                scales = torch.exp(-logdet / M.shape[-1])
                M = M * scales.view(-1, 1, 1)
        
        self.register_buffer('metric_matrices', M)
        
        if temperature is not None:
            T = max(float(temperature), 1e-6)
            self.register_buffer('temperature', torch.tensor(T, device=self.device))
        if regularization is not None:
            self.register_buffer('regularization', torch.tensor(regularization, device=self.device))
            
        self._is_loaded = True
        
        print(f"✅ MetricTensor loaded: {len(centroids)} centroids, T={self.temperature.item():.3f}, λ={self.regularization.item():.3f}")
        
    def _compute_precision_components(
        self,
        z: torch.Tensor,
        *,
        return_metric: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Internal helper to assemble the mixed precision tensor (and optionally the metric).
        """
        if not self._is_loaded:
            raise RuntimeError("Metric tensor not loaded. Call load_pretrained() first.")

        B, d = z.shape[0], self.latent_dim
        device = z.device
        z64 = z.to(device=device, dtype=torch.float64)
        centroids = self.centroids.to(device=device, dtype=torch.float64)
        atoms_raw = self.metric_matrices.to(device=device, dtype=torch.float64)

        if self.weight_kernel == 'isotropic':
            weight_atoms = atoms_raw
        elif self.weight_kernel in {'mahalanobis', 'mahalanobis_normed'}:
            weight_atoms = normalize_metric_atoms(atoms_raw, mode=self.weight_metric_normalization)
        else:
            raise ValueError(f"Unknown weight kernel '{self.weight_kernel}'")

        temperature = (
            self.temperature.to(device=device, dtype=torch.float64)
            if hasattr(self, 'temperature')
            else torch.tensor(0.2, device=device, dtype=torch.float64)
        )
        weights = compute_metric_weights(
            z64,
            centroids,
            weight_atoms,
            temperature,
            kernel=self.weight_kernel,
            normalize=self.normalize_weight_sum,
            topk=self.topk_weights,
            stabilize=self.normalize_weight_sum,
        )

        G_inv_mix = torch.einsum('bk,kij->bij', weights, atoms_raw)
        G_inv_mix = 0.5 * (G_inv_mix + G_inv_mix.transpose(-1, -2))

        I = torch.eye(d, device=device, dtype=torch.float64).unsqueeze(0)
        lambda_reg = (
            self.regularization.to(device=device, dtype=torch.float64)
            if isinstance(self.regularization, torch.Tensor)
            else torch.tensor(float(self.regularization), device=device, dtype=torch.float64)
        )
        lambda_reg = lambda_reg.clamp_min(0.0)
        if lambda_reg.item() > 0:
            G_inv_mix = G_inv_mix + lambda_reg * I

        diff = z64.unsqueeze(1) - centroids.unsqueeze(0)
        if self.weight_kernel == 'isotropic':
            d2 = torch.sum(diff * diff, dim=-1)
        else:
            tmp = torch.einsum('bkd,kde->bke', diff, weight_atoms)
            d2 = torch.sum(tmp * diff, dim=-1)
        d2min = d2.min(dim=1).values
        temp_scalar = float(temperature)
        thresh = (4.0 * temp_scalar) ** 2
        far = torch.sigmoid((d2min - thresh) / (0.25 * thresh + 1e-12))
        if self.use_background_identity:
            G_inv_mix = G_inv_mix + self.bg_strength * far.view(-1, 1, 1) * I

        evals, evecs = torch.linalg.eigh(G_inv_mix)
        floor = torch.tensor(self.eig_floor_abs, device=device, dtype=torch.float64)
        evals = torch.clamp(evals, min=floor)
        if self.eig_ceiling is not None:
            ceil = torch.tensor(self.eig_ceiling, device=device, dtype=torch.float64)
            evals = torch.clamp(evals, max=ceil)
        G_inv64 = evecs @ (evals.unsqueeze(-1) * evecs.transpose(-1, -2))

        if not torch.isfinite(G_inv64).all():
            warnings.warn("G_inv contained non-finite values! Clamping.")
            G_inv64 = torch.nan_to_num(G_inv64, nan=1.0, posinf=1e3, neginf=1e-3)

        target_dtype = z.dtype if z.dtype.is_floating_point else torch.float32
        if target_dtype in (torch.float16, torch.bfloat16):
            target_dtype = torch.float32
        G_inv = G_inv64.to(target_dtype)

        G_metric: Optional[torch.Tensor] = None
        if return_metric:
            G_metric64 = _robust_inverse_from_cholesky(G_inv64)
            G_metric = G_metric64.to(target_dtype)
            if not torch.isfinite(G_metric).all():
                warnings.warn("G contained non-finite values! Clamping.")
                G_metric = torch.nan_to_num(G_metric, nan=1.0, posinf=1e3, neginf=1e-3)

        if os.getenv("RLVAE_METRIC_DEBUG", "0") == "1":
            with torch.no_grad():
                G_dbg = G_metric if G_metric is not None else _robust_inverse_from_cholesky(G_inv64).to(target_dtype)
                prod = G_dbg @ G_inv.float()
                err = (prod - torch.eye(d, device=z.device)).norm(dim=(1, 2)).mean().item()
                logdet = self.compute_log_det_inverse_metric(z).mean().item()
                sum_w = weights.sum(dim=-1)
                print(f"[MetricDebug] ||G·G⁻¹ - I|| ≈ {err:.3e}, mean log|det(G⁻¹)| ≈ {logdet:.3f}, "
                      f"sum_w mean={sum_w.mean().item():.3f} min={sum_w.min().item():.3f} max={sum_w.max().item():.3f}, "
                      f"dmin mean={d2min.mean().item():.3f}, T={float(temperature):.3f}")

        if return_metric:
            return G_inv, G_metric
        return G_inv, None

    def compute_inverse_metric(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse metric tensor G^{-1}(z).
        
        Args:
            z: Latent coordinates [batch_size, latent_dim]
            
        Returns:
            G_inv: Inverse metric tensor [batch_size, latent_dim, latent_dim]
        """
        if self.trainable:
            # Trainable mode: compute G first, then invert using Cholesky
            G = self.compute_metric(z)
            G_inv = _robust_inverse_from_cholesky(G)
            return G_inv
        else:
            G_inv, _ = self._compute_precision_components(z, return_metric=False)
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
            # Trainable mode: network outputs G directly
            G = self.metric_net(z)
            
            # Enforce symmetry
            G = 0.5 * (G + G.transpose(-1, -2))
            
            # Enforce SPD via softplus on diagonal
            diag = torch.diagonal(G, dim1=-2, dim2=-1)
            diag = torch.nn.functional.softplus(diag) + 1e-6
            G = G.clone()
            G.diagonal(dim1=-2, dim2=-1).copy_(diag)
            
            return G
        else:
            G_inv, G = self._compute_precision_components(z, return_metric=True)
            return G
    
    def compute_log_det_inverse_metric(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log|det(G^{-1}(z))| directly from the precision matrices.
        """
        G_inv = self.compute_inverse_metric(z)
        try:
            # Prefer Cholesky factorization; inverse metric is SPD after spectral clamps
            L = _cholesky_spd(G_inv)
            diagL = torch.diagonal(L, dim1=-2, dim2=-1).abs() + 1e-18
            return 2.0 * torch.log(diagL).sum(dim=-1)
        except Exception:
            # Fallback to slogdet for pathological batches
            sign, logabs = torch.linalg.slogdet(G_inv)
            return logabs
    
    def compute_log_det_metric(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log|det(G(z))|. Use compute_log_det_inverse_metric for log|det(G^{-1})|.
        """
        G = self.compute_metric(z)
        
        # Use robust Cholesky-based logdet computation
        log_det = _robust_logdet_from_cholesky(G)
        
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
            
            # Compute eigenvalues for first sample (use eigvalsh for symmetric matrices)
            eigenvals_G = torch.linalg.eigvalsh(G[0])
            eigenvals_G_inv = torch.linalg.eigvalsh(G_inv[0])
            
            # If you clamp eigenvalues for stability, do it like this:
            eigenvals_G = torch.clamp(eigenvals_G, min=1e-6)
            eigenvals_G_inv = torch.clamp(eigenvals_G_inv, min=1e-6)

            # Compute determinants and traces
            det_G = torch.linalg.det(G)
            det_G_inv = torch.linalg.det(G_inv)
            trace_G = torch.diagonal(G, dim1=-2, dim2=-1).sum(-1)
            trace_G_inv = torch.diagonal(G_inv, dim1=-2, dim2=-1).sum(-1)
            
            # Guard for trainable mode where buffers may not exist
            n_centroids = int(self.centroids.shape[0]) if hasattr(self, "centroids") else 0
            temp_val = float(self.temperature.item()) if hasattr(self, "temperature") else float('nan')
            reg_val = float(self.regularization.item()) if hasattr(self, "regularization") else float('nan')
            
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
                'lambda_min_Ginv_batch_min': torch.linalg.eigvalsh(G_inv).min().item(),
                'batch_size': z.shape[0],
                'n_centroids': n_centroids,
                'temperature': temp_val,
                'regularization': reg_val,
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
