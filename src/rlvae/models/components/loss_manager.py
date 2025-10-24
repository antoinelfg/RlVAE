"""
LossManager: Modular Loss Computation Handler
============================================

Handles all loss computations for Riemannian VAE models including:
- Reconstruction loss
- KL divergence loss (standard and Riemannian)
- Flow loss
- Loop penalty loss
- Combined loss computation
"""

import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metric_utils import half_logdet_volume as _global_half_logdet_volume
from .metric_tensor import inverse_fallback_count

class LossManager(nn.Module):
    def __init__(
        self,
        beta: float = 1.0,
        riemannian_beta: Optional[float] = None,
        loop_penalty_weight: float = 1.0,
        device: Optional[torch.device] = None,
        metric_reg_weight: float = 2.0,  # NEW: weight for metric regularization
        metric_reg_type: str = 'none',   # NEW: type: 'none', 'determinant', 'condition', 'smoothness'
        metric_reg_target: float = 2.0,  # NEW: target value for regularization (e.g., logdet target)
        # μ anchoring
        mu_l2_weight: float = 0.0,
        recon_scale: float = 100.0,
        kl_monitor_baseline_tau: float = 0.98,
        metric_representation: str = "ginv",
        # Prior mode for KL: 'uniform' (default, cancels volume) or 'volume_gaussian'
        kl_prior_mode: str = 'uniform',
        # Optional amplification of volume terms / gradients
        volume_bias_weight: float = 1.0,
        volume_grad_scale: float = 1.0,
        # Riemannian KL options (to mirror original model behavior)
        kl_use_metric_normalization: bool = True,
        kl_metric_norm_mode: str = 'geomean',   # 'geomean' | 'trace' | 'none'
        kl_amp_safe: bool = True,
        kl_metric_eval_point: str = 'z',  # 'z' or 'mu' (curvature correction if 'z')
        # RHMC KL switches
        rhmc_kl_mode: str = 'mc',          # {'mc','jac','bound'}
        rhmc_kl_source: str = 'z0',        # {'z0','zk'}
        rhmc_kl_jacobian: bool = False,
    ):
        super().__init__()
        self.beta = beta
        self.riemannian_beta = riemannian_beta if riemannian_beta is not None else beta
        self.loop_penalty_weight = loop_penalty_weight
        self.device = device or torch.device('cpu')
        self.metric_reg_weight = metric_reg_weight
        self.metric_reg_type = metric_reg_type
        self.metric_reg_target = metric_reg_target
        self.mu_l2_weight = float(mu_l2_weight)
        self.recon_scale = float(recon_scale)
        self.kl_monitor_baseline_tau = float(kl_monitor_baseline_tau)
        self._kl_monitor_baseline: Optional[torch.Tensor] = None
        self.metric_representation = str(metric_representation).lower()
        if self.metric_representation not in {"g", "ginv"}:
            raise ValueError("LossManager: metric_representation must be 'G' or 'Ginv'")
        self.kl_prior_mode = str(kl_prior_mode)
        self.volume_bias_weight = float(volume_bias_weight)
        self.volume_grad_scale = float(volume_grad_scale)
        self.kl_use_metric_normalization = kl_use_metric_normalization
        self.kl_metric_norm_mode = kl_metric_norm_mode
        self.kl_amp_safe = kl_amp_safe
        self.kl_metric_eval_point = kl_metric_eval_point
        # RHMC KL toggles
        self.rhmc_kl_mode = str(rhmc_kl_mode)
        self.rhmc_kl_source = str(rhmc_kl_source)
        self.rhmc_kl_jacobian = bool(rhmc_kl_jacobian)
        self.to(self.device)
        
        # Loss tracking
        self.loss_history = {
            'reconstruction': [],
            'kl_divergence': [],
            'riemannian_kl': [],
            'flow_loss': [],
            'loop_penalty': [],
            'total': [],
            'metric_reg': [],  # NEW
            'mu_l2': []
        }
        self._inverse_fallbacks_seen = inverse_fallback_count()
        self._debug_prev_sigma_stats: Dict[str, Dict[str, float]] = {}
        self._latent_debug_step = 0
        self._latent_debug_header_written = False

    def _monitor_inverse_fallbacks(self) -> None:
        current = inverse_fallback_count()
        previous = getattr(self, "_inverse_fallbacks_seen", 0)
        if current > previous:
            delta = current - previous
            print(f"[WARN] Metric inverse fallback triggered {delta} additional time(s); total={current}.")
            self._inverse_fallbacks_seen = current
    
    def compute_reconstruction_loss(self, x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss (MSE for continuous data).
        
        Args:
            x_recon: Reconstructed input [batch_size, *input_shape]
            x: Original input [batch_size, *input_shape]
            
        Returns:
            Reconstruction loss scalar
        """
        loss = F.mse_loss(x_recon, x, reduction='mean')
        if self.recon_scale != 1.0:
            loss = loss * x.new_tensor(self.recon_scale)
        
        if not torch.isfinite(loss):
            if os.environ.get("RLVAE_DEBUG") == "1":
                print("⚠️ Reconstruction loss is not finite! Clamping to 1.0.")
            loss = torch.tensor(1.0, device=loss.device)
        return loss
    
    def compute_standard_kl_loss(
        self, 
        mu: torch.Tensor, 
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute standard KL divergence loss.
        
        Args:
            mu: Posterior mean [batch_size, latent_dim]
            log_var: Posterior log variance [batch_size, latent_dim]
            
        Returns:
            KL divergence loss scalar
        """
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return kl_loss / mu.shape[0]  # Average over batch
    
    def compute_riemannian_kl_loss(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        z_samples: torch.Tensor,
        metric_tensor: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Local surrogate KL for the Riemannian Gaussian posterior (Stage-B bound).

        This penalises samples or means under G(z) but does not implement the Stage-C
        Monte-Carlo KL with flows. Use only for ablations or when RHMC is disabled.

        Args:
            mu: Posterior mean [batch_size, latent_dim]
            log_var: Posterior log variance (not used)
            z_samples: Samples from posterior [batch_size, latent_dim]
            metric_tensor: Metric tensor component (required)
        Returns:
            KL divergence (scalar)
        """
        debug = os.environ.get('RLVAE_DEBUG', '0') == '1'
        # DEBUG: Print KL computation details
        if debug and not hasattr(self, '_kl_debug_printed'):
            print(f"[KL DEBUG] kl_metric_eval_point: {self.kl_metric_eval_point}")
            print(f"[KL DEBUG] mu shape: {mu.shape}, mean: {mu.mean().item():.4f}, std: {mu.std().item():.4f}")
            print(f"[KL DEBUG] z_samples shape: {z_samples.shape}, mean: {z_samples.mean().item():.4f}, std: {z_samples.std().item():.4f}")
            self._kl_debug_printed = True
        if metric_tensor is None:
            # Fallback to standard KL (silenced unless debug)
            if debug:
                print("⚠️ DEBUG: Falling back to standard KL")
            if os.environ.get('RLVAE_STRICT', '0') == '1':
                raise RuntimeError("LossManager: metric_tensor is None under strict Riemannian mode")
            try:
                if os.environ.get('RLVAE_TRACE', '0') == '1':
                    print(f"TRACE KL source: fallback to standard KL; z_samples shape={tuple(z_samples.shape)}")
            except Exception:
                pass
            log_var_clamped = torch.clamp(log_var, -10.0, 10.0)
            return -0.5 * torch.sum(1 + log_var_clamped - mu.pow(2) - log_var_clamped.exp(), dim=1).mean()
        try:
            try:
                if os.environ.get('RLVAE_TRACE', '0') == '1':
                    print(f"TRACE KL source: using rhmc_z0=True shape={tuple(z_samples.shape)}, using log_q from q_Riem")
            except Exception:
                pass
            # Choose evaluation point for metric: 'z' (samples) or 'mu' (means)
            eval_pts = z_samples if self.kl_metric_eval_point == 'z' else mu
            eval_pts_metric = eval_pts.float() if self.kl_amp_safe else eval_pts
            if debug:
                print(
                    f"[KL DEBUG] Using eval_pts: {self.kl_metric_eval_point} -> "
                    f"shape {eval_pts.shape}, mean {eval_pts.mean().item():.4f}, std {eval_pts.std().item():.4f}"
                )
            # Compute metric/precision tensor at chosen points
            G_eval_raw, rep_eval = self._evaluate_metric(eval_pts_metric, metric_tensor, None, with_rep=True)
            if G_eval_raw is None or rep_eval is None:
                raise RuntimeError("Metric tensor not available for Riemannian KL computation.")
            rep_eval = rep_eval.lower()

            def _as_metric(tensor: torch.Tensor, rep: str) -> torch.Tensor:
                if rep == "g":
                    return tensor
                if rep == "ginv":
                    chol, _ = self._cholesky_spd(tensor, jitter=1e-6)
                    d_local = tensor.shape[-1]
                    eye = torch.eye(d_local, device=tensor.device, dtype=tensor.dtype).unsqueeze(0).expand_as(tensor)
                    return torch.cholesky_solve(eye, chol)
                raise ValueError(f"Unknown metric representation '{rep}'.")

            tensor_for_quad = G_eval_raw
            rep_for_quad = rep_eval

            # Optional normalization (only meaningful when actual metric is provided)
            if self.kl_use_metric_normalization and rep_eval == "g":
                d = tensor_for_quad.shape[-1]
                if self.kl_metric_norm_mode == 'geomean':
                    sign, logabsdet = torch.slogdet(tensor_for_quad)
                    s = torch.exp(logabsdet / d).unsqueeze(-1).unsqueeze(-1)
                    tensor_for_quad = tensor_for_quad / (s + 1e-12)
                elif self.kl_metric_norm_mode == 'trace':
                    s = (torch.einsum('bii->b', tensor_for_quad) / d).unsqueeze(-1).unsqueeze(-1)
                    tensor_for_quad = tensor_for_quad / (s + 1e-12)

            # AMP-safe float32 compute
            mu_f32 = mu.float() if self.kl_amp_safe else mu
            z_f32 = z_samples.float() if self.kl_amp_safe else z_samples
            tensor_quad_f32 = tensor_for_quad.float() if self.kl_amp_safe else tensor_for_quad

            # Choose what to penalize based on evaluation point
            if self.kl_metric_eval_point == 'mu':
                # Penalize encoder means directly: (μ - 0)ᵀ G(μ) (μ - 0)
                # This pulls encoder means toward the prior center (0)
                diff = mu_f32 - 0.0  # [B, D] - distance from prior center
                if debug:
                    print(f"[KL DEBUG] Penalizing encoder means with G: diff shape {diff.shape}, mean {diff.mean().item():.4f}, std {diff.std().item():.4f}")
            else:
                # Penalize distance between samples and means
                diff = z_f32 - mu_f32  # [B, D]
                if debug:
                    print(f"[KL DEBUG] Penalizing sample-mean distance with G: diff shape {diff.shape}, mean {diff.mean().item():.4f}, std {diff.std().item():.4f}")
            
            quadratic_form = self._quad_with_G(diff, tensor_quad_f32, rep_for_quad)
            kl_terms = 0.5 * quadratic_form

            # Optional volume-corrected Gaussian prior at μ: 0.5||μ||^2 - 0.5 log|G(μ)|
            if self.kl_prior_mode == 'volume_gaussian' and self.kl_metric_eval_point == 'mu':
                try:
                    mu_metric = mu_f32 if self.kl_amp_safe else mu
                    G_mu_tensor, rep_mu = self._evaluate_metric(mu_metric, metric_tensor, None, with_rep=True)
                    if G_mu_tensor is None or rep_mu is None:
                        raise RuntimeError('Metric tensor not available for volume prior term')
                    rep_mu = rep_mu.lower()
                    G_mu_metric = _as_metric(G_mu_tensor, rep_mu)
                    G_mu_metric = G_mu_metric.float() if self.kl_amp_safe else G_mu_metric
                    mu_quad = self._quad_with_G(mu_f32, G_mu_metric, "g")
                    half_logdet_volume = self._half_logdet_volume(G_mu_tensor, rep_mu)
                    logdet_G = -2.0 * half_logdet_volume
                    prior_term = 0.5 * mu_quad - 0.5 * logdet_G
                    # Debug: report prior term statistics
                    if debug:
                        print(f"[KL DEBUG] volume prior term stats: mean={prior_term.mean().item():.6f}, min={prior_term.min().item():.6f}, max={prior_term.max().item():.6f}")
                    kl_terms = kl_terms + prior_term
                    if debug:
                        print(f"[KL DEBUG] Added volume prior term: mean={prior_term.mean().item():.6f}")
                except Exception as e:
                    if debug:
                        print(f"[KL DEBUG] Volume prior term failed: {e}")

            kl_divergence = kl_terms.mean()
            if debug:
                print(f"[KL DEBUG] Final KL: {kl_divergence.item():.6f}, quadratic_form mean: {quadratic_form.mean().item():.6f}")
            return kl_divergence.to(mu.dtype)
        except Exception as e:
            if os.environ.get('RLVAE_STRICT', '0') == '1':
                raise RuntimeError(f"LossManager: Riemannian KL failed under strict mode: {e}")
            if debug:
                print(f"⚠️ Riemannian KL computation failed: {e}, using standard KL")
            try:
                if os.environ.get('RLVAE_TRACE', '0') == '1':
                    print("TRACE KL source: error path -> standard KL")
            except Exception:
                pass
            log_var_clamped = torch.clamp(log_var, -10.0, 10.0)
            return -0.5 * torch.sum(1 + log_var_clamped - mu.pow(2) - log_var_clamped.exp(), dim=1).mean()
    
    def compute_flow_loss(
        self,
        log_det_jacobians: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Small L2 regularizer on per-step log|det J| (only used when Stage-C uniform is OFF).
        """
        if not log_det_jacobians:
            return torch.tensor(0.0, device=self.device)

        # stack per-step [B] → [T-1, B] and L2
        per_step = []
        for t, term in enumerate(log_det_jacobians):
            if isinstance(term, torch.Tensor):
                v = torch.nan_to_num(term.to(device=self.device), nan=0.0, posinf=0.0, neginf=0.0)
                if v.ndim == 0:
                    v = v.expand(1)  # best-effort
                per_step.append(v.reshape(-1))
        if not per_step:
            return torch.tensor(0.0, device=self.device)

        M = torch.stack(per_step, dim=0)  # [T-1, B]
        loss = (M ** 2).mean() * 1e-4     # tiny weight; keep gradients sane
        return loss

    def _sum_logdet_jacobians(
        self,
        log_det_jacobians: Optional[list],
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Aggregate per-sample log|det J_t| from each flow into a per-sample sum.
        Each item must contribute [B]; never clamp or take abs here.
        """
        if not log_det_jacobians:
            return torch.zeros(batch_size, device=device, dtype=dtype)

        total = torch.zeros(batch_size, device=device, dtype=dtype)
        for i, term in enumerate(log_det_jacobians):
            if not isinstance(term, torch.Tensor):
                continue
            t = term.to(device=device, dtype=dtype)

            # Normalize to per-sample [B]
            if t.ndim == 1 and t.shape[0] == batch_size:
                contrib = t
            elif t.ndim == 2 and t.shape[0] == batch_size:
                contrib = t.sum(dim=1)
            elif t.ndim > 2:
                contrib = t.reshape(batch_size, -1).sum(dim=1)
            elif t.ndim == 0:
                # Scalar logdet is suspicious; treat as "pre-summed" and spread evenly as zeros
                print(f"[FLOW WARN] scalar logdet at flow {i}; replacing with zeros [B].")
                contrib = torch.zeros(batch_size, device=device, dtype=dtype)
            else:
                # Last-resort reshape/crop
                print(f"[FLOW WARN] unexpected logdet shape {tuple(t.shape)} at flow {i}; trying reshape.")
                contrib = t.reshape(-1)[:batch_size]
                if contrib.numel() < batch_size:
                    pad = torch.zeros(batch_size - contrib.numel(), device=device, dtype=dtype)
                    contrib = torch.cat([contrib, pad], dim=0)

            # Light runtime guard: very large means often implicate wrong sign/direction upstream
            mean_abs = contrib.abs().mean().item()
            if mean_abs > 5.0:
                print(f"[WARN] Large flow logdet contribution (flow {i}): mean={mean_abs:.2f}")

            total += contrib

        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
        return total

        
    def _evaluate_metric(
        self,
        z: Optional[torch.Tensor],
        metric_tensor: Optional[Any],
        rhmc_posterior: Optional[Any],
        *,
        with_rep: bool = False,
    ) -> Union[Optional[torch.Tensor], Tuple[Optional[torch.Tensor], Optional[str]]]:
        """
        Evaluate the metric tensor at ``z`` using whichever component is available.

        Returns a symmetrized SPD tensor together with the representation tag
        (``'g'`` for metric, ``'ginv'`` for precision) or ``(None, None)`` if no
        metric information is available.
        """
        if z is None:
            return None, None

        preferred = self.metric_representation.lower()

        def _resolve(component: Any) -> tuple[Optional[torch.Tensor], Optional[str]]:
            if component is None:
                return None, None
            try:
                if preferred == "ginv" and hasattr(component, "compute_inverse_metric"):
                    return component.compute_inverse_metric(z), "ginv"
                if preferred == "g" and hasattr(component, "compute_metric"):
                    return component.compute_metric(z), "g"
            except Exception:
                pass
            try:
                if hasattr(component, "compute_metric"):
                    return component.compute_metric(z), "g"
                if hasattr(component, "compute_inverse_metric"):
                    return component.compute_inverse_metric(z), "ginv"
                if callable(component):
                    tensor = component(z)
                    if isinstance(tensor, torch.Tensor):
                        return tensor, self.metric_representation
            except Exception:
                return None, None
            if hasattr(component, "G"):
                try:
                    return component.G(z), "g"
                except Exception:
                    return None, None
            if hasattr(component, "G_inv"):
                try:
                    return component.G_inv(z), "ginv"
                except Exception:
                    return None, None
            return None, None

        tensor, rep = _resolve(metric_tensor)

        if tensor is None or rep is None:
            try:
                model = getattr(rhmc_posterior, "_ctx", {}).get("model", None) if rhmc_posterior is not None else None
            except Exception:
                model = None
            if model is not None:
                tensor, rep = _resolve(model)

        if tensor is None or rep is None:
            self._monitor_inverse_fallbacks()
            return (None, None) if with_rep else None

        tensor = 0.5 * (tensor + tensor.transpose(-1, -2))
        self._monitor_inverse_fallbacks()
        return (tensor, rep) if with_rep else tensor

    def _debug_log_sigma(self, label: str, sigma: torch.Tensor, *, info: str = "") -> None:
        if os.environ.get("RLVAE_DEBUG", "0") != "1":
            return
        with torch.no_grad():
            sigma32 = sigma.detach().float()
            try:
                eigvals = torch.linalg.eigvalsh(sigma32)
                min_eig = eigvals.min().item()
                max_eig = eigvals.max().item()
                mean_eig = eigvals.mean().item()
            except RuntimeError:
                min_eig = max_eig = mean_eig = float('nan')
            trace = torch.einsum('bii->b', sigma32).mean().item()
            logdet_vals = self._slogdet_spd(sigma32).mean().item()
            prev = self._debug_prev_sigma_stats.get(label)
            delta_trace = delta_logdet = ""
            if prev is not None:
                delta_trace = f" (Δ{trace - prev['trace']:+.4e})"
                delta_logdet = f" (Δ{logdet_vals - prev['logdet']:+.4e})"
            context = f" - {info}" if info else ""
            print(f"[SIGMA DEBUG] {label}{context}: trace={trace:.4f}{delta_trace}, "
                  f"logdet={logdet_vals:.4f}{delta_logdet}, "
                  f"min_eig={min_eig:.4e}, max_eig={max_eig:.4e}, mean_eig={mean_eig:.4e}")
            self._debug_prev_sigma_stats[label] = {
                "trace": trace,
                "logdet": logdet_vals,
                "min_eig": min_eig,
                "max_eig": max_eig,
            }

    def _symmetrize(self, matrix: torch.Tensor) -> torch.Tensor:
        """Return symmetric part of a batch of matrices."""
        return 0.5 * (matrix + matrix.transpose(-1, -2))

    def _sanitize_covariance(self, matrix: torch.Tensor, *, floor: float) -> torch.Tensor:
        """
        Project a covariance onto the SPD cone with minimum eigenvalue ``floor``.
        Always operates in float32 and removes NaNs/Infs.
        """
        if not isinstance(matrix, torch.Tensor):
            raise TypeError("Expected tensor for covariance sanitization.")
        floor = float(max(floor, 0.0))
        mat32 = matrix.to(dtype=torch.float32)
        mat32 = self._symmetrize(torch.nan_to_num(mat32, nan=0.0, posinf=0.0, neginf=0.0))
        try:
            evals, evecs = torch.linalg.eigh(mat32)
            evals = torch.clamp(evals, min=floor if floor > 0 else 1e-6)
            mat32 = evecs @ (evals.unsqueeze(-1) * evecs.transpose(-1, -2))
        except Exception:
            eye = torch.eye(mat32.shape[-1], device=mat32.device, dtype=mat32.dtype).unsqueeze(0)
            mat32 = mat32 + max(floor, 1e-6) * eye
        mat32 = self._symmetrize(torch.nan_to_num(mat32, nan=0.0, posinf=0.0, neginf=0.0))
        return mat32

    def _cholesky_spd(
        self,
        matrix: torch.Tensor,
        *,
        jitter: float = 1e-6,
        max_tries: int = 6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute a stable Cholesky factor for an SPD batch with jitter fallback.
        """
        d = matrix.shape[-1]
        eye = torch.eye(d, device=matrix.device, dtype=matrix.dtype).unsqueeze(0)
        jitter_local = float(jitter)
        chol = None
        stabilized = None
        for _ in range(max_tries):
            trial = self._symmetrize(matrix) + jitter_local * eye
            trial32 = trial.float() if trial.dtype in (torch.float16, torch.bfloat16) else trial
            if hasattr(torch.linalg, "cholesky_ex"):
                chol32, info = torch.linalg.cholesky_ex(trial32)
                if isinstance(info, torch.Tensor) and info.numel() > 0 and (info > 0).any():
                    jitter_local = max(jitter_local * 10.0, 1e-5)
                    continue
                chol = chol32
                stabilized = trial32
                break
            try:
                chol32 = torch.linalg.cholesky(trial32)
                chol = chol32
                stabilized = trial32
                break
            except RuntimeError:
                jitter_local = max(jitter_local * 10.0, 1e-5)
        if chol is None or stabilized is None:
            # Final fallback: add large jitter and attempt once more
            jitter_local = max(jitter_local, 1e-3)
            trial = self._symmetrize(matrix) + jitter_local * eye
            trial32 = trial.float() if trial.dtype in (torch.float16, torch.bfloat16) else trial
            chol = torch.linalg.cholesky(trial32)
            stabilized = trial32
        return chol, stabilized

    def _slogdet_spd(self, matrix: torch.Tensor, *, jitter: float = 1e-6) -> torch.Tensor:
        """
        Compute log det of an SPD batch with jitter-guarded Cholesky.
        """
        chol, stabilized = self._cholesky_spd(matrix, jitter=jitter)
        diag = torch.diagonal(chol, dim1=-2, dim2=-1)
        logdet = 2.0 * torch.log(diag + 1e-18).sum(dim=-1)
        return logdet.to(matrix.dtype if matrix.dtype.is_floating_point else torch.float32)

    def _stable_half_logdet(self, G: torch.Tensor, *, jitter: float = 1e-6) -> torch.Tensor:
        """
        Compute ½ log det G with SPD regularisation.
        """
        logdet = self._slogdet_spd(G, jitter=jitter)
        return 0.5 * torch.nan_to_num(logdet.to(G.dtype), nan=0.0, posinf=0.0, neginf=0.0)

    def _log_gaussian_density(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        Sigma: torch.Tensor,
        *,
        jitter: float = 1e-6,
    ) -> torch.Tensor:
        """
        Evaluate log N(z | mu, Sigma) with SPD covariance using Cholesky solve.
        """
        chol, stabilized = self._cholesky_spd(Sigma, jitter=jitter)
        diff = (z - mu).unsqueeze(-1)
        diff32 = diff.float() if diff.dtype != chol.dtype else diff
        sol = torch.cholesky_solve(diff32, chol)
        quad = torch.einsum('bij,bij->b', diff32, sol)
        diag = torch.diagonal(chol, dim1=-2, dim2=-1)
        logdet = 2.0 * torch.log(diag + 1e-18).sum(dim=-1)
        const = z.shape[-1] * math.log(2 * math.pi)
        log_prob = -0.5 * quad - 0.5 * logdet - 0.5 * const
        return log_prob.to(z.dtype if z.dtype.is_floating_point else stabilized.dtype)

    def _half_logdet_volume(
        self,
        G_or_Ginv: torch.Tensor,
        rep: Optional[str] = None,
        *,
        jitter: float = 1e-6,
    ) -> torch.Tensor:
        """
        Return +½ log det G^{-1} regardless of representation supplied.
        """
        rep_effective = (rep or self.metric_representation).lower()
        return _global_half_logdet_volume(G_or_Ginv, rep_effective, jitter=jitter)

    def _tensor_to_metric(
        self,
        tensor: torch.Tensor,
        rep: str,
        *,
        jitter: float = 1e-6,
    ) -> torch.Tensor:
        rep = (rep or self.metric_representation).lower()
        if rep == "g":
            return 0.5 * (tensor + tensor.transpose(-1, -2))
        if rep == "ginv":
            chol, _ = self._cholesky_spd(tensor, jitter=jitter)
            d = tensor.shape[-1]
            eye = torch.eye(d, device=tensor.device, dtype=tensor.dtype).unsqueeze(0).expand_as(tensor)
            metric = torch.cholesky_solve(eye, chol)
            return 0.5 * (metric + metric.transpose(-1, -2))
        raise ValueError(f"Unknown representation '{rep}'.")

    def _debug_metric_stats(
        self,
        label: str,
        tensor: torch.Tensor,
        rep: str,
        *,
        jitter: float = 1e-6,
    ) -> None:
        if os.environ.get("RLVAE_DEBUG", "0") != "1":
            return
        try:
            with torch.no_grad():
                metric = self._tensor_to_metric(tensor, rep, jitter=jitter).double()
                precision = tensor.double() if rep.lower() == "ginv" else torch.linalg.inv(metric)
                eig_metric = torch.linalg.eigvalsh(metric)
                eig_precision = torch.linalg.eigvalsh(precision)
                min_eig = eig_metric.min().item()
                max_eig = eig_metric.max().item()
                cond_metric = float(max_eig / max(min_eig, 1e-12))
                min_eig_inv = eig_precision.min().item()
                max_eig_inv = eig_precision.max().item()
                cond_precision = float(max_eig_inv / max(min_eig_inv, 1e-12))
                eye = torch.eye(metric.shape[-1], device=metric.device, dtype=metric.dtype).unsqueeze(0)
                prod_err = torch.linalg.norm(metric @ precision - eye, dim=(1, 2)).mean().item()
                logdet_metric = torch.log(torch.clamp(eig_metric, min=1e-18)).sum(-1).mean().item()
                logdet_precision = torch.log(torch.clamp(eig_precision, min=1e-18)).sum(-1).mean().item()
                print(
                    f"[METRIC DEBUG] {label}: rep={rep.lower()} | "
                    f"eig_min={min_eig:.3e}, eig_max={max_eig:.3e}, cond(G)={cond_metric:.3e} | "
                    f"eig_min_inv={min_eig_inv:.3e}, eig_max_inv={max_eig_inv:.3e}, cond(G⁻¹)={cond_precision:.3e} | "
                    f"log|G|={logdet_metric:.3e}, log|G⁻¹|={logdet_precision:.3e} | "
                    f"||G·G⁻¹-I||_F={prod_err:.3e}"
                )
        except Exception as exc:
            print(f"[METRIC DEBUG] {label}: failed to gather stats ({exc})")

    def _debug_latent_anisotropy(
        self,
        label: str,
        samples: Optional[torch.Tensor],
        *,
        reference: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, float]]:
        if os.environ.get("RLVAE_DEBUG", "0") != "1":
            return None
        if not isinstance(samples, torch.Tensor) or samples.ndim != 2 or samples.shape[0] < 2:
            return None
        try:
            with torch.no_grad():
                data = samples
                if isinstance(reference, torch.Tensor):
                    data = samples - reference
                centered = data - data.mean(dim=0, keepdim=True)
                denom = max(centered.shape[0] - 1, 1)
                cov = centered.t().matmul(centered) / denom
                evals = torch.linalg.eigvalsh(cov.double()).clamp_min(0.0)
                total = evals.sum().item()
                if total <= 0:
                    print(f"[LATENT DEBUG] {label}: degenerate covariance (total variance ≈ 0)")
                    return {
                        "pc1_ratio": float("nan"),
                        "pc2_ratio": float("nan"),
                        "eig_min": 0.0,
                        "eig_max": 0.0,
                        "trace": 0.0,
                    }
                sorted_evals, _ = torch.sort(evals, descending=True)
                ratios = sorted_evals / total
                pc1 = ratios[0].item()
                pc2 = ratios[1].item() if ratios.numel() > 1 else float("nan")
                stats = {
                    "pc1_ratio": pc1,
                    "pc2_ratio": pc2,
                    "eig_min": evals.min().item(),
                    "eig_max": evals.max().item(),
                    "trace": total,
                }
                print(
                    f"[LATENT DEBUG] {label}: pc1_ratio={pc1:.3f}, "
                    f"pc2_ratio={pc2:.3f}, eig_min={evals.min().item():.3e}, "
                    f"eig_max={evals.max().item():.3e}, trace={total:.3e}"
                )
                return stats
        except Exception as exc:
            print(f"[LATENT DEBUG] {label}: PCA computation failed ({exc})")
        return None

    def _log_latent_debug_row(self, row: Dict[str, float]) -> None:
        try:
            path = Path("outputs/probes")
            path.mkdir(parents=True, exist_ok=True)
            file = path / "latent_diagnostics.csv"
            keys = list(row.keys())
            with file.open("a", encoding="utf-8") as fh:
                if not self._latent_debug_header_written:
                    fh.write(",".join(keys) + "\n")
                    self._latent_debug_header_written = True
                fh.write(",".join(str(row[k]) for k in keys) + "\n")
        except Exception as exc:
            print(f"[LATENT DEBUG] failed to log diagnostics ({exc})")

    def _quad_with_G(
        self,
        v: torch.Tensor,
        G_or_Ginv: torch.Tensor,
        rep: Optional[str] = None,
        *,
        jitter: float = 1e-6,
    ) -> torch.Tensor:
        """
        Compute v^T G v regardless of metric representation.
        """
        rep_effective = (rep or self.metric_representation).lower()
        rep = rep_effective
        if rep == "ginv":
            chol, _ = self._cholesky_spd(G_or_Ginv, jitter=jitter)
            v_col = v.unsqueeze(-1).float()
            x = torch.cholesky_solve(v_col, chol)
            quad = torch.einsum('bij,bij->b', v_col, x)
            return quad.to(v.dtype if v.dtype.is_floating_point else torch.float32)
        elif rep == "g":
            v32 = v.float()
            G32 = G_or_Ginv.float()
            quad = torch.einsum('bi,bij,bj->b', v32, G32, v32)
            return quad.to(v.dtype if v.dtype.is_floating_point else torch.float32)
        else:
            raise ValueError(f"Unknown metric representation '{rep}' in quad computation.")

    def _resolve_sigma_mu(
        self,
        mu: torch.Tensor,
        Sigma_mu: Optional[torch.Tensor],
        metric_tensor: Optional[Any],
        rhmc_posterior: Optional[Any],
        rhmc_traj_info: Optional[dict],
        *,
        jitter: float = 1e-6,
    ) -> Optional[torch.Tensor]:
        """
        Build Σ_μ = α G^{-1}(μ) + ε I consistent with ``metric_representation``.
        """
        # Prefer caller-provided covariance only during training to avoid
        # validation/test runs inheriting stale caches.
        sigma_candidates = []
        if self.training and isinstance(Sigma_mu, torch.Tensor):
            sigma_candidates.append(Sigma_mu)
        if self.training and isinstance(rhmc_traj_info, dict):
            sigma_from_traj = rhmc_traj_info.get('Sigma_mu', None)
            if isinstance(sigma_from_traj, torch.Tensor):
                sigma_candidates.append(sigma_from_traj)
        if self.training and rhmc_posterior is not None:
            cached = getattr(rhmc_posterior, '_last_sigma_mu', None)
            if isinstance(cached, torch.Tensor):
                sigma_candidates.append(cached)

        for candidate in sigma_candidates:
            Sigma = candidate.to(device=mu.device, dtype=torch.float32)
            Sigma = self._sanitize_covariance(Sigma, floor=1e-6)
            target_dtype = mu.dtype if mu.dtype in (torch.float32, torch.float64) else torch.float32
            self._debug_log_sigma("Sigma_mu_cached", Sigma, info="candidate")
            return Sigma.to(target_dtype).detach()

        G_eval, rep = self._evaluate_metric(mu, metric_tensor, rhmc_posterior, with_rep=True)
        if G_eval is None or rep is None:
            return None

        rep = rep.lower()
        if rep == "ginv":
            Ginv_mu = G_eval.to(dtype=torch.float32)
        elif rep == "g":
            chol, _ = self._cholesky_spd(G_eval, jitter=jitter)
            d = G_eval.shape[-1]
            eye = torch.eye(d, device=G_eval.device, dtype=G_eval.dtype).unsqueeze(0)
            Ginv_mu = torch.cholesky_solve(eye, chol)
            Ginv_mu = Ginv_mu.to(dtype=torch.float32)
        else:
            raise ValueError(f"Unknown metric representation '{rep}' when building Σ_μ.")
        self._debug_log_sigma("Ginv_mu", Ginv_mu, info=f"rep={rep}")

        alpha = 1.0
        eps_reg = float(jitter)
        if isinstance(rhmc_traj_info, dict):
            try_alpha = rhmc_traj_info.get('alpha', None)
            if try_alpha is not None:
                try:
                    alpha = float(try_alpha)
                except Exception:
                    pass
            try_eps = rhmc_traj_info.get('eps_reg', None)
            if try_eps is not None:
                try:
                    eps_reg = float(try_eps)
                except Exception:
                    pass
        if rhmc_posterior is not None:
            try:
                alpha = float(getattr(rhmc_posterior, 'rhmc_alpha', alpha))
            except Exception:
                pass
            try:
                eps_reg = float(getattr(rhmc_posterior, 'eps_reg', eps_reg))
            except Exception:
                pass

        alpha = float(alpha) if math.isfinite(alpha) and alpha > 0.0 else 1.0
        eps_reg = float(eps_reg) if math.isfinite(eps_reg) and eps_reg >= 0.0 else 1e-6
        alpha = max(alpha, 1e-3)
        eps_reg = max(eps_reg, 1e-6)
        d = Ginv_mu.shape[-1]
        eye = torch.eye(d, device=Ginv_mu.device, dtype=Ginv_mu.dtype).unsqueeze(0)
        Sigma = alpha * Ginv_mu + eps_reg * eye
        Sigma = self._symmetrize(torch.nan_to_num(Sigma, nan=0.0, posinf=0.0, neginf=0.0))
        Sigma = self._sanitize_covariance(Sigma, floor=max(eps_reg, 1e-6))
        target_dtype = mu.dtype if mu.dtype in (torch.float32, torch.float64) else torch.float32
        self._debug_log_sigma("Sigma_mu", Sigma, info=f"alpha={alpha:.4f}, eps={eps_reg:.2e}")
        return Sigma.to(target_dtype).detach()

    def _log_kinetic_density(
        self,
        rho: torch.Tensor,
        z: torch.Tensor,
        metric_tensor: Optional[Any],
        rhmc_posterior: Optional[Any],
        *,
        jitter: float = 1e-6,
    ) -> torch.Tensor:
        """
        log π_kin(ρ | z) with π_kin(ρ|z) = N(0, G(z)).
        """
        G_or_Ginv, rep = self._evaluate_metric(z, metric_tensor, rhmc_posterior, with_rep=True)
        if G_or_Ginv is None or rep is None:
            raise RuntimeError("Cannot evaluate kinetic density without metric.")
        d = G_or_Ginv.shape[-1]
        const = 0.5 * d * math.log(2 * math.pi)
        rep = rep.lower()
        if rep == "ginv":
            rho32 = rho.float()
            quad = torch.einsum('bi,bij,bj->b', rho32, G_or_Ginv.float(), rho32)
            half_logdet = self._half_logdet_volume(G_or_Ginv, 'ginv', jitter=jitter)
            return (-0.5 * quad + half_logdet - const).to(rho.dtype)
        if rep == "g":
            chol, _ = self._cholesky_spd(G_or_Ginv, jitter=jitter)
            rho_col = rho.unsqueeze(-1).float()
            sol = torch.cholesky_solve(rho_col, chol)
            quad = torch.einsum('bij,bij->b', rho_col, sol)
            half_logdet = self._half_logdet_volume(G_or_Ginv, 'g', jitter=jitter)
            return (-0.5 * quad + half_logdet - const).to(rho.dtype)
        raise ValueError(f"Unknown metric representation '{rep}' for kinetic density.")
    
    def _batch_jacobian(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute batched Jacobian d(outputs)/d(inputs) for vector-valued outputs.
        
        Args:
            inputs: [B, D] tensor with requires_grad=True
            outputs: [B, D] tensor produced from inputs
        Returns:
            jac: [B, D, D] Jacobian for each batch element
        """
        batch, dim = inputs.shape
        jac = torch.zeros(batch, dim, dim, device=inputs.device, dtype=inputs.dtype)
        for k in range(dim):
            grads = torch.autograd.grad(
                outputs[:, k].sum(),
                inputs,
                retain_graph=True,
                create_graph=False,
                allow_unused=False
            )[0]
            jac[:, k, :] = grads
        return jac
    
    def _pushforward_metric_via_flows(
        self,
        z0: torch.Tensor,
        flow_manager: Optional[Any],
        metric_tensor: Optional[Any],
        rhmc_posterior: Optional[Any],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Push forward the Stage-B metric through the flow stack.

        Returns:
            GT_sel: transported metric matching current self.metric_representation ('g' or 'ginv')
            min_sv: min singular value of J (diagnostic)
            half_logdet_push_g:     -1/2 log det G'     (always returned)
            half_logdet_push_ginv:  +1/2 log det G'^{-1} (always returned)
        """
        # 0) Get base tensor in configured representation
        Gbase, base_rep = self._evaluate_metric(z0, metric_tensor, rhmc_posterior, with_rep=True)
        if Gbase is None or base_rep is None:
            return (None, None), None, None, None

        B, D = z0.shape
        rep  = base_rep.lower()

        def _spd_inverse(A: torch.Tensor) -> torch.Tensor:
            chol, _ = self._cholesky_spd(A)
            eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype).unsqueeze(0).expand_as(A)
            return torch.cholesky_solve(eye, chol)

        if rep == "ginv":
            Ginv0 = Gbase
            G0 = _spd_inverse(Ginv0)
        elif rep == "g":
            G0 = Gbase
            Ginv0 = _spd_inverse(G0)
        else:
            raise ValueError(f"Unknown metric representation '{rep}' for pushforward.")

        # 2) No-flow case (J = I) — compute both volumes correctly
        if flow_manager is None or getattr(flow_manager, 'n_flows', 0) == 0:
            # -1/2 log det G' = -1/2 log det G0
            half_logdet_push_g    = self._half_logdet_volume(G0, 'g', jitter=1e-6)
            half_logdet_push_ginv = self._half_logdet_volume(Ginv0, 'ginv', jitter=1e-6)
            GT_sel = G0 if rep == "g" else Ginv0
            return (
                (GT_sel.to(z0.dtype).detach(), rep),
                torch.full((B,), float('nan'), device=G0.device, dtype=G0.dtype),
                half_logdet_push_g.to(z0.dtype),
                half_logdet_push_ginv.to(z0.dtype),
            )

        # 3) Build full Jacobian with autograd
        from contextlib import nullcontext
        grad_ctx = nullcontext()
        if not torch.is_grad_enabled():
            grad_ctx = torch.enable_grad()

        with grad_ctx:
            z = z0.detach().clone().requires_grad_(True)
            eye = torch.eye(D, device=z.device, dtype=z.dtype).unsqueeze(0)
            J_total = eye.repeat(B, 1, 1)
            for flow in flow_manager.flows:
                out_struct = flow(z)
                z_next = out_struct.out
                if not z_next.requires_grad:
                    # Fall back to no-flow formulas if graph is broken
                    half_logdet_push_g    = self._half_logdet_volume(G0, 'g', jitter=1e-6)
                    half_logdet_push_ginv = self._half_logdet_volume(Ginv0, 'ginv', jitter=1e-6)
                    GT_sel = G0 if rep == "g" else Ginv0
                    return (
                        (GT_sel.to(z0.dtype).detach(), rep),
                        torch.full((B,), float('nan'), device=z.device, dtype=z.dtype),
                        half_logdet_push_g.to(z0.dtype),
                        half_logdet_push_ginv.to(z0.dtype),
                    )
                jac = self._batch_jacobian(z, z_next)  # [B, D, D] of ∂z_next/∂z
                J_total = torch.bmm(jac, J_total)
                z = z_next.detach().clone().requires_grad_(True)

        # 4) Transport both objects with correct formulas
        J64   = J_total.double()
        G064  = G0.double()
        Ginv64 = Ginv0.double()

        # Diagnostics
        sv = torch.linalg.svdvals(J64)
        min_sv = sv.min(dim=-1).values

        # G-transport:    G' = J^{-T} G J^{-1}
        eye64 = torch.eye(D, device=J64.device, dtype=J64.dtype).unsqueeze(0).expand(B, -1, -1)
        J_inv = torch.linalg.solve(J64, eye64)
        GT_g  = torch.bmm(J_inv.transpose(1, 2), torch.bmm(G064, J_inv))
        GT_g  = 0.5 * (GT_g + GT_g.transpose(1, 2))

        # G^{-1}-transport:  G'^{-1} = J G^{-1} J^T
        GT_ginv = torch.bmm(J64, torch.bmm(Ginv64, J64.transpose(1, 2)))
        GT_ginv = 0.5 * (GT_ginv + GT_ginv.transpose(1, 2))

        # 5) Volumes (always return both canonical scalars)
        half_logdet_push_g    = self._half_logdet_volume(GT_g, 'g', jitter=1e-6)
        half_logdet_push_ginv = self._half_logdet_volume(GT_ginv, 'ginv', jitter=1e-6)

        # 6) Select the transported tensor for downstream, matching configured rep
        GT_sel = GT_g if rep == "g" else GT_ginv

        if os.environ.get("RLVAE_DEBUG", "0") == "1":
            try:
                with torch.no_grad():
                    eig_metric = torch.linalg.eigvalsh(GT_g.double())
                    eig_precision = torch.linalg.eigvalsh(GT_ginv.double())
                    cond_metric = float(eig_metric.max().item() / max(eig_metric.min().item(), 1e-12))
                    cond_precision = float(eig_precision.max().item() / max(eig_precision.min().item(), 1e-12))
                    min_sv_val = float(min_sv.min().item()) if min_sv is not None and torch.isfinite(min_sv).any() else float('nan')
                    prod_err = torch.linalg.norm(GT_g.double() @ GT_ginv.double() - torch.eye(D, device=GT_g.device, dtype=torch.float64), dim=(1, 2)).mean().item()
                    print(
                        "[PUSH DEBUG] transported metric stats: "
                        f"eig_min={eig_metric.min().item():.3e}, eig_max={eig_metric.max().item():.3e}, "
                        f"cond(G')={cond_metric:.3e}, cond(G'^{-1})={cond_precision:.3e}, "
                        f"min_sv(J)={min_sv_val:.3e}, ||G'·G'^{-1}-I||_F={prod_err:.3e}"
                    )
            except Exception as exc:
                print(f"[PUSH DEBUG] failed to gather stats ({exc})")

        return (
            (GT_sel.to(z0.dtype).detach(), rep),
            min_sv.to(z0.dtype),
            half_logdet_push_g.to(z0.dtype),
            half_logdet_push_ginv.to(z0.dtype),
        )
        
    def compute_loop_penalty(
        self, 
        z_seq: list, 
        loop_mode: str = "open"
    ) -> torch.Tensor:
        """
        Compute loop penalty for temporal consistency.
        
        Args:
            z_seq: List of latent tensors [n_timesteps]
            loop_mode: "open" or "closed" loop
            
        Returns:
            Loop penalty scalar
        """
        if loop_mode == "open" or len(z_seq) < 2:
            return torch.tensor(0.0, device=self.device)
        
        if loop_mode == "closed":
            # Penalize difference between first and last latent
            z_first = z_seq[0]
            z_last = z_seq[-1]
            penalty = F.mse_loss(z_first, z_last, reduction='mean')
            if not torch.isfinite(penalty):
                print("⚠️ Loop penalty is not finite! Clamping to 0.0.")
                penalty = torch.tensor(0.0, device=self.device)
            return penalty * self.loop_penalty_weight
        
        return torch.tensor(0.0, device=self.device)
    
    def compute_total_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        z_samples: torch.Tensor,
        *,
        log_det_jacobians: Optional[list] = None,
        z_seq: Optional[list] = None,
        flow_manager: Optional[Any] = None,
        loop_mode: str = "open",
        metric_tensor: Optional[Any] = None,
        use_riemannian_kl: bool = True,
        # Explicit Stage-C latent states (preferred over legacy args)
        z0: Optional[torch.Tensor] = None,
        zS: Optional[torch.Tensor] = None,
        zF: Optional[torch.Tensor] = None,
        Sigma_mu: Optional[torch.Tensor] = None,
        sum_logdet_flow: Optional[torch.Tensor] = None,
        # RHMC posterior pathway (MC KL on pushforward)
        rhmc_z0: Optional[torch.Tensor] = None,
        rhmc_zK: Optional[torch.Tensor] = None,
        rhmc_log_q: Optional[torch.Tensor] = None,
        rhmc_traj_info: Optional[dict] = None,
        rhmc_posterior: Optional[Any] = None,
        rhmc_kl_mode: Optional[str] = None,
        rhmc_kl_source: Optional[str] = None,
        rhmc_kl_jacobian: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss combining all components.
        
        Args:
            x: Original input [batch_size, *input_shape]
            x_recon: Reconstructed input [batch_size, *input_shape]
            mu: Posterior mean [batch_size, latent_dim]
            log_var: Posterior log variance [batch_size, latent_dim]
            z_samples: Sampled latent variables [batch_size, latent_dim]
            log_det_jacobians: List of log determinants for flows
            z_seq: List of latent tensors for loop penalty
            loop_mode: Loop mode for temporal consistency
            metric_tensor: Metric tensor component
            use_riemannian_kl: Whether to use Riemannian KL
            
        Returns:
            Dictionary containing all loss components and total
        """
        # Ensure metric_reg is always defined for safe returns
        metric_reg = torch.tensor(0.0, device=x.device)
        batch_size = mu.shape[0]

        # Compute individual loss components
        recon_loss = self.compute_reconstruction_loss(x_recon, x)
        if not torch.isfinite(recon_loss) and os.environ.get("RLVAE_DEBUG") == "1":
            print("[DEBUG] recon_loss is not finite!", recon_loss)
        
        # Decide KL path
        kl_weight = self.beta
        kl_loss: torch.Tensor

        # Prefer RHMC Monte-Carlo path when posterior extras are provided
        use_rhmc_mc = (
            rhmc_posterior is not None and
            (rhmc_zK is not None or z_samples is not None)
        )

        # Pre-compute flow Jacobian sum for potential MC KL usage
        if sum_logdet_flow is not None:
            sum_logdet_flow = sum_logdet_flow.to(device=mu.device, dtype=mu.dtype)
        else:
            if os.environ.get("RLVAE_DEBUG", "0") == "1":
                print("\n[DEBUG] --- FLOW JACOBIAN SUMMARY ---")
                for i, term in enumerate(log_det_jacobians or []):
                    if not isinstance(term, torch.Tensor):
                        continue
                    t_term = term.to(device=mu.device, dtype=mu.dtype)
                    summary = (
                        f"[DEBUG FLOW {i}] shape={tuple(t_term.shape)}, "
                        f"min={t_term.min().item():.4f}, max={t_term.max().item():.4f}, "
                        f"mean={t_term.mean().item():.4f}, std={t_term.std().item():.4f}"
                    )
                    print(summary)
            sum_logdet_flow = self._sum_logdet_jacobians(
                log_det_jacobians,
                batch_size,
                device=mu.device,
                dtype=mu.dtype,
            )

        # Resolve runtime switches (kwargs override ctor defaults)
        _mode = (rhmc_kl_mode or self.rhmc_kl_mode).lower()
        _src = (rhmc_kl_source or self.rhmc_kl_source).lower()
        _with_jac = bool(self.rhmc_kl_jacobian if rhmc_kl_jacobian is None else rhmc_kl_jacobian)

        kl_aux_metrics: Dict[str, torch.Tensor] = {}
        kl_components_sample: Dict[str, torch.Tensor] = {}
        stage_c_uniform = False

        if use_rhmc_mc and _mode in {"mc", "jac"}:
            if os.environ.get('RLVAE_DEBUG', '0') == '1':
                print(f"[KL ROUTE] RHMC Monte-Carlo branch (prior_mode={self.kl_prior_mode})")
            if not hasattr(self, "_loss_cfg_printed"):
                print(f"[LOSS CONFIG] rhmc_kl_mode={_mode}, source={_src}, jacobian={str(_with_jac).lower()}")
                self._loss_cfg_printed = True

            # Resolve latent states
            stage_zS = zS if zS is not None else (rhmc_zK if rhmc_zK is not None else z_samples)
            if stage_zS is None:
                raise RuntimeError("Stage-C KL requires post-RHMC samples (zS).")
            stage_z0 = z0 if z0 is not None else (rhmc_z0 if rhmc_z0 is not None else z_samples)
            stage_zF = zF
            if stage_zF is None:
                if isinstance(z_seq, list):
                    for candidate in reversed(z_seq):
                        if isinstance(candidate, torch.Tensor):
                            stage_zF = candidate
                            break
                if stage_zF is None:
                    stage_zF = stage_zS

            if self.kl_prior_mode == "uniform" and _mode == "mc":
                stage_c_uniform = True
                eps_reg = float(getattr(rhmc_posterior, 'eps_reg', 1e-6)) if rhmc_posterior is not None else 1e-6
                zS_effective = stage_zS

                if _src == "z0":
                    if rhmc_log_q is not None:
                        log_q = rhmc_log_q.to(mu.dtype)
                    else:
                        Sigma_mu = self._resolve_sigma_mu(
                            mu,
                            Sigma_mu,
                            metric_tensor,
                            rhmc_posterior,
                            rhmc_traj_info,
                            jitter=eps_reg,
                        )
                        if Sigma_mu is not None:
                            log_q = self._log_gaussian_density(stage_z0, mu, Sigma_mu, jitter=eps_reg).to(mu.dtype)
                        else:
                            try:
                                log_q = rhmc_posterior._compute_log_riemannian_gaussian(stage_z0, mu, log_var).to(mu.dtype)
                            except Exception:
                                diff = stage_z0 - mu
                                d = diff.shape[-1]
                            log_q = (-0.5 * torch.sum(diff ** 2, dim=-1) - 0.5 * d * np.log(2 * np.pi)).to(mu.dtype)
                else:
                    try:
                        log_q = rhmc_posterior._compute_log_riemannian_gaussian(zS_effective, mu, log_var).to(mu.dtype)
                    except Exception:
                        diff = zS_effective - mu
                        d = diff.shape[-1]
                        log_q = (-0.5 * torch.sum(diff ** 2, dim=-1) - 0.5 * d * np.log(2 * np.pi)).to(mu.dtype)

                G_source, rep_source = self._evaluate_metric(stage_z0, metric_tensor, rhmc_posterior, with_rep=True)
                if G_source is None or rep_source is None:
                    raise RuntimeError("Stage C uniform prior requires metric evaluation at z0.")
                half_logdet_source_ginv = self._half_logdet_volume(G_source, rep_source.lower(), jitter=eps_reg).to(mu.dtype)
                debug_mode = os.environ.get("RLVAE_DEBUG", "0") == "1"
                if debug_mode:
                    self._debug_metric_stats("G(z0)", G_source, rep_source, jitter=eps_reg)
                    metric_source = self._tensor_to_metric(G_source, rep_source, jitter=eps_reg).double()
                else:
                    metric_source = None

                G_target, rep_target = self._evaluate_metric(zS_effective, metric_tensor, rhmc_posterior, with_rep=True)
                if G_target is None or rep_target is None:
                    raise RuntimeError("Stage C uniform prior requires metric evaluation at zS.")
                half_logdet_target_ginv = self._half_logdet_volume(G_target, rep_target.lower(), jitter=eps_reg).to(mu.dtype)
                if debug_mode:
                    self._debug_metric_stats("G(zS)", G_target, rep_target, jitter=eps_reg)
                    metric_target = self._tensor_to_metric(G_target, rep_target, jitter=eps_reg).double()
                else:
                    metric_target = None

                flow_term = sum_logdet_flow.to(mu.dtype)
                flow_term = torch.nan_to_num(flow_term, nan=0.0, posinf=0.0, neginf=0.0)

                delta_kin = torch.zeros_like(flow_term)
                delta_vol = torch.zeros_like(flow_term)
                if isinstance(rhmc_traj_info, dict):
                    dk = rhmc_traj_info.get('delta_kin', None)
                    if isinstance(dk, torch.Tensor):
                        delta_kin = dk.to(mu.dtype)
                    else:
                        traj = rhmc_traj_info.get('trajectory', None)
                        if isinstance(traj, list) and len(traj) > 1:
                            rho0 = traj[0].get('rho', None)
                            rhoS = traj[-1].get('rho', None)
                            z_traj0 = traj[0].get('z', stage_z0)
                            z_trajS = traj[-1].get('z', zS_effective)
                            if isinstance(rho0, torch.Tensor) and isinstance(rhoS, torch.Tensor):
                                delta_kin = (
                                    self._log_kinetic_density(rho0, z_traj0, metric_tensor, rhmc_posterior)
                                    - self._log_kinetic_density(rhoS, z_trajS, metric_tensor, rhmc_posterior)
                                ).to(mu.dtype)
                    dv = rhmc_traj_info.get('delta_vol', None)
                    if isinstance(dv, torch.Tensor):
                        delta_vol = dv.to(mu.dtype)

                kl_terms = (
                    log_q.to(x.dtype)
                    - (self.volume_bias_weight * half_logdet_target_ginv).to(x.dtype)
                    - flow_term.to(x.dtype)           # ✅ fixed sign
                    + (delta_kin.to(x.dtype) - delta_vol.to(x.dtype))
                )
                kl_loss = kl_terms.mean().to(x.dtype)
                kl_weight = self.riemannian_beta
                flow_loss = torch.zeros((), device=x.device, dtype=x.dtype)

                if os.environ.get("RLVAE_DEBUG", "0") == "1":
                    with torch.no_grad():
                        rhs = half_logdet_source_ginv + flow_term
                        vol_residual = (half_logdet_target_ginv - rhs).abs()
                        diff_z_mu = torch.norm(zS_effective - mu, dim=-1)
                        diff_z0_mu = torch.norm(stage_z0 - mu, dim=-1)
                        diff_z = torch.norm(zS_effective - stage_z0, dim=-1)
                        mu_norm = torch.norm(mu, dim=-1)
                        corr_value = float("nan")
                        max_eig_mean = float("nan")
                        if metric_target is not None and metric_target.ndim == 3:
                            try:
                                eig_target = torch.linalg.eigvalsh(metric_target)
                                max_eig = eig_target[:, -1]
                                max_eig_mean = max_eig.mean().item()
                                if diff_z_mu.numel() > 1:
                                    stacked = torch.stack([diff_z_mu.double(), max_eig.double()])
                                    corr_matrix = torch.corrcoef(stacked)
                                    corr_value = corr_matrix[0, 1].item()
                                print(
                                    f"[KL DEBUG] corr(||zS-mu||, max_eig(G(zS)))={corr_value:.3e}, "
                                    f"max_eig_mean={max_eig_mean:.3e}"
                                )
                            except Exception as exc:
                                print(f"[KL DEBUG] correlation computation failed ({exc})")
                                max_eig_mean = float("nan")
                        else:
                            max_eig_mean = float("nan")
                        stats_zs = self._debug_latent_anisotropy("StageC (zS - mu)", zS_effective, reference=mu)
                        stats_mu = self._debug_latent_anisotropy("StageC mu", mu)
                        if metric_target is not None and metric_target.ndim == 3:
                            try:
                                eig_target = torch.linalg.eigvalsh(metric_target)
                                max_eig = eig_target[:, -1]
                                if diff_z_mu.numel() > 1:
                                    stacked = torch.stack([diff_z_mu.double(), max_eig.double()])
                                    corr = torch.corrcoef(stacked)[0, 1].item()
                                else:
                                    corr = float("nan")
                                print(
                                    f"[KL DEBUG] corr(||zS-mu||, max_eig(G(zS)))={corr:.3e}, "
                                    f"max_eig_mean={max_eig.mean().item():.3e}"
                                )
                            except Exception as exc:
                                print(f"[KL DEBUG] correlation computation failed ({exc})")
                        self._debug_latent_anisotropy("StageC (zS - mu)", zS_effective, reference=mu)
                        self._debug_latent_anisotropy("StageC mu", mu)
                        print(
                            "[KL DEBUG] latent norms: "
                            f"||mu|| mean={torch.norm(mu, dim=-1).mean().item():.3e}, "
                            f"||z0-mu|| mean={diff_z0_mu.mean().item():.3e}, "
                            f"||zS-mu|| mean={diff_z_mu.mean().item():.3e}, "
                            f"||zS-z0|| mean={diff_z.mean().item():.3e}"
                        )
                        print(
                            "[KL DEBUG] flow stats: "
                            f"log_q mean={log_q.mean().item():.3e}, "
                            f"sum_logdet_flow mean={flow_term.mean().item():.3e}, "
                            f"delta_kin mean={delta_kin.mean().item():.3e}, "
                            f"delta_vol mean={delta_vol.mean().item():.3e}"
                        )
                        print(
                            "[KL DEBUG] volume identity residual: "
                            f"mean={vol_residual.mean().item():.3e}, "
                            f"max={vol_residual.max().item():.3e}"
                        )
                        kin_start = kin_end = None
                        if isinstance(rhmc_traj_info, dict):
                            traj = rhmc_traj_info.get('trajectory', None)
                            if isinstance(traj, list) and len(traj) > 1:
                                rho0 = traj[0].get('rho', None)
                                rhoS = traj[-1].get('rho', None)
                                z_traj0 = traj[0].get('z', stage_z0)
                                z_trajS = traj[-1].get('z', zS_effective)
                                if isinstance(rho0, torch.Tensor) and isinstance(rhoS, torch.Tensor):
                                    try:
                                        kin_start = self._log_kinetic_density(rho0, z_traj0, metric_tensor, rhmc_posterior)
                                        kin_end = self._log_kinetic_density(rhoS, z_trajS, metric_tensor, rhmc_posterior)
                                    except Exception as exc:
                                        print(f"[KL DEBUG] kinetic density computation failed ({exc})")
                        kin_residual = float("nan")
                        if kin_start is not None and kin_end is not None:
                            kin_diff = kin_start - kin_end
                            kin_residual = (delta_kin - kin_diff.to(delta_kin.dtype)).abs().mean().item()
                            print(
                                "[KL DEBUG] kinetic density: "
                                f"start_mean={kin_start.mean().item():.3e}, "
                                f"end_mean={kin_end.mean().item():.3e}, "
                                f"diff_mean={kin_diff.mean().item():.3e}, "
                                f"delta_kin_residual={kin_residual:.3e}"
                            )

                        row = {
                            "step": float(self._latent_debug_step),
                            "mu_norm_mean": float(mu_norm.mean().item()),
                            "z0_mu_norm_mean": float(diff_z0_mu.mean().item()),
                            "zS_mu_norm_mean": float(diff_z_mu.mean().item()),
                            "zS_z0_norm_mean": float(diff_z.mean().item()),
                            "max_eig_mean": float(max_eig_mean),
                            "corr_zS_mu_max_eig": float(corr_value),
                            "volume_residual_mean": float(vol_residual.mean().item()),
                            "volume_residual_max": float(vol_residual.max().item()),
                            "delta_kin_mean": float(delta_kin.mean().item()),
                            "delta_kin_residual": float(kin_residual),
                            "pc1_zS_minus_mu": float(stats_zs.get("pc1_ratio", float("nan")) if stats_zs else float("nan")),
                            "pc1_mu": float(stats_mu.get("pc1_ratio", float("nan")) if stats_mu else float("nan")),
                            "trace_zS_minus_mu": float(stats_zs.get("trace", float("nan")) if stats_zs else float("nan")),
                            "trace_mu": float(stats_mu.get("trace", float("nan")) if stats_mu else float("nan")),
                        }
                        self._log_latent_debug_row(row)
                        self._latent_debug_step += 1

                log_q_key = 'log_q0' if _src == "z0" else 'log_qS'
                volume_key = 'half_logdet_ginv_source'
                kl_aux_metrics = {
                    'loss/KL_uniform_mc': kl_loss.detach(),
                    'loss/log_q_source_mean': log_q.mean().detach(),
                    f'loss/{volume_key}_mean': half_logdet_source_ginv.mean().detach(),
                    'loss/half_logdet_ginv_target_mean': half_logdet_target_ginv.mean().detach(),
                    'loss/half_logdet_ginv_source_mean': half_logdet_source_ginv.mean().detach(),
                    'loss/sum_logdet_flow_mean': flow_term.mean().detach(),
                    'rhmc/delta_kin_mean': delta_kin.mean().detach(),
                    'rhmc/delta_vol_mean': delta_vol.mean().detach(),
                    'routing/kl_prior_mode_uniform': torch.tensor(1.0, device=x.device, dtype=x.dtype),
                    'routing/rhmc_kl_mode_mc': torch.tensor(1.0, device=x.device, dtype=x.dtype),
                }

                consistency_sample = None
                push_target_gap_sample = None
                try:
                    ((G_pushforward, rep_push),
                    min_sv,
                    half_logdet_push_g,
                    half_logdet_push_ginv) = self._pushforward_metric_via_flows(stage_z0, flow_manager, metric_tensor, rhmc_posterior)

                    if G_pushforward is not None and rep_push is not None:
                        # We expect:  +1/2 logdet(G'^{-1}) = +1/2 logdet(G^{-1}) + log|det J|
                        target_rhs = (half_logdet_source_ginv + flow_term).to(mu.dtype)

                        # Compare residuals but honour configured representation
                        res_g    = (half_logdet_push_g    - target_rhs).abs().mean()
                        res_ginv = (half_logdet_push_ginv - target_rhs).abs().mean()

                        rep_push = rep_push.lower()
                        half_logdet_push = half_logdet_push_ginv if rep_push == "ginv" else half_logdet_push_g
                        consistency = (half_logdet_push - target_rhs)

                        if os.environ.get("RLVAE_DEBUG", "0") == "1":
                            self._debug_metric_stats("Pushforward G'", G_pushforward, rep_push, jitter=eps_reg)

                        if not hasattr(self, "_vol_triplet_printed"):
                            print(f"[VOL CHECK] rep_push={rep_push} mean(source)={half_logdet_source_ginv.mean().item():.3f}, "
                                f"mean(flow)={flow_term.mean().item():.3f}, "
                                f"mean(push)={half_logdet_push.mean().item():.3f}, "
                                f"mean(target)={half_logdet_target_ginv.mean().item():.3f}")
                            self._vol_triplet_printed = True

                        kl_aux_metrics['diagnostics/pushforward_consistency_mean'] = consistency.abs().mean().detach()
                        kl_aux_metrics['diagnostics/pushforward_vs_target_mean']   = (half_logdet_push - half_logdet_target_ginv).abs().mean().detach()
                        consistency_sample = consistency.detach()
                        push_target_gap_sample = (half_logdet_push - half_logdet_target_ginv).detach()

                        fits_ginv_better = bool(res_ginv <= res_g)
                        config_pref_is_ginv = (self.metric_representation == "ginv")
                        if config_pref_is_ginv != fits_ginv_better and not hasattr(self, "_push_rep_warned"):
                            print(f"[WARN] Push-forward identity residual favours "
                                f"{'G^{-1}' if fits_ginv_better else 'G'} (res_g={res_g.item():.3f}, res_ginv={res_ginv.item():.3f}) "
                                f"while metric_representation='{self.metric_representation}'.")
                            self._push_rep_warned = True

                        if min_sv is not None:
                            finite_mask = torch.isfinite(min_sv)
                            if finite_mask.any():
                                kl_aux_metrics['diagnostics/jacobian_min_singular_mean'] = min_sv[finite_mask].mean().detach()
                except Exception:
                    pass

                if not hasattr(self, '_half_logdet_check_printed'):
                    with torch.no_grad():
                        print(
                            "[CHECK] mean half_logdet_ginv(source) = "
                            f"{half_logdet_source_ginv.mean().item():.3f}, "
                            f"target = {half_logdet_target_ginv.mean().item():.3f}"
                        )
                    self._half_logdet_check_printed = True

                kl_components_sample = {
                    log_q_key: log_q.detach(),
                    volume_key: half_logdet_source_ginv.detach(),
                    'half_logdet_ginv_target': half_logdet_target_ginv.detach(),
                    'half_logdet_ginv_source': half_logdet_source_ginv.detach(),
                    'sum_logdet_flow': flow_term.detach(),
                    'delta_kin': delta_kin.detach(),
                    'delta_vol': delta_vol.detach(),
                    'kl_terms': kl_terms.detach(),
                }
                if consistency_sample is not None:
                    kl_components_sample['pushforward_consistency'] = consistency_sample
                if push_target_gap_sample is not None:
                    kl_components_sample['pushforward_vs_target'] = push_target_gap_sample
                if os.environ.get("RLVAE_DEBUG", "0") == "1" and consistency_sample is not None:
                    with torch.no_grad():
                        print(
                            "[KL DEBUG] pushforward consistency: "
                            f"mean={consistency_sample.mean().item():.3e}, "
                            f"max={consistency_sample.abs().max().item():.3e}"
                        )
            else:
                # Legacy RHMC KL path (volume Gaussian / diagnostics)
                log_q = None
                if _src == "z0" and rhmc_log_q is not None:
                    log_q = rhmc_log_q
                else:
                    try:
                        log_q = rhmc_posterior._compute_log_riemannian_gaussian(stage_zS, mu, log_var)
                    except Exception as e:
                        if os.environ.get("RLVAE_DEBUG") == "1":
                            print(f"[KL DEBUG] Fallback log_q failed: {e}; using isotropic Gaussian")
                        diff = stage_zS - mu
                        d = stage_zS.shape[-1]
                        log_q = -0.5 * torch.sum(diff ** 2, dim=-1) - 0.5 * d * np.log(2 * np.pi)

                try:
                    log_p = rhmc_posterior._compute_log_prior(stage_zS)
                except Exception:
                    d = stage_zS.shape[-1]
                    log_p = -0.5 * torch.sum(stage_zS ** 2, dim=-1) - 0.5 * d * np.log(2 * np.pi)

                jac_correction = 0.0
                if _mode == "jac" and _with_jac and isinstance(rhmc_traj_info, dict):
                    j = rhmc_traj_info.get('jac_logdet', None)
                    if isinstance(j, torch.Tensor):
                        jac_correction = j
                    elif j is not None:
                        try:
                            jac_correction = torch.as_tensor(j, device=stage_zS.device, dtype=log_q.dtype)
                        except Exception:
                            jac_correction = 0.0

                jac_correction_tensor = jac_correction if isinstance(jac_correction, torch.Tensor) else torch.tensor(float(jac_correction), device=log_q.device, dtype=log_q.dtype)
                kl_terms = log_q - log_p - jac_correction_tensor
                kl_loss = kl_terms.mean().to(x.dtype)
                kl_weight = self.riemannian_beta
                kl_aux_metrics = {
                    'loss/KL_rhmc_legacy': kl_loss.detach(),
                    'routing/rhmc_kl_mode_legacy': torch.tensor(1.0, device=x.device, dtype=x.dtype),
                }
                kl_components_sample = {
                    'log_q': log_q.detach(),
                    'log_p': log_p.detach(),
                    'jac_correction': jac_correction_tensor.detach(),
                }
        elif use_riemannian_kl and metric_tensor is not None:
            if os.environ.get('RLVAE_DEBUG', '0') == '1':
                print(f"[KL ROUTE] compute_riemannian_kl_loss (prior_mode={self.kl_prior_mode})")
            # Geodesic bound / classical Riemannian KL (bound mode)
            kl_loss = self.compute_riemannian_kl_loss(mu, log_var, z_samples, metric_tensor)
            kl_weight = self.riemannian_beta
        else:
            if os.environ.get("RLVAE_DEBUG") == "1":
                print("⚠️ DEBUG: Falling back to STANDARD KL")
            kl_loss = self.compute_standard_kl_loss(mu, log_var)
            kl_weight = self.beta
        if not torch.isfinite(kl_loss) and os.environ.get("RLVAE_DEBUG") == "1":
            print("[DEBUG] kl_loss is not finite!", kl_loss)

        # Flow loss: for uniform MC path the Jacobian term is absorbed into KL
        if stage_c_uniform:
            flow_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        else:
            flow_loss = self.compute_flow_loss(log_det_jacobians)
        if not torch.isfinite(flow_loss) and os.environ.get("RLVAE_DEBUG") == "1":
            print("[DEBUG] flow_loss is not finite!", flow_loss)
        
        loop_penalty = self.compute_loop_penalty(z_seq, loop_mode)
        if not torch.isfinite(loop_penalty) and os.environ.get("RLVAE_DEBUG") == "1":
            print("[DEBUG] loop_penalty is not finite!", loop_penalty)

        kl_detached = kl_loss.detach()
        kl_monitor = torch.zeros_like(kl_detached)
        if torch.isfinite(kl_detached).all():
            needs_reset = (
                self._kl_monitor_baseline is None
                or self._kl_monitor_baseline.device != kl_detached.device
                or not torch.isfinite(self._kl_monitor_baseline).all()
            )
            if needs_reset:
                self._kl_monitor_baseline = kl_detached.clone()
            else:
                tau = self.kl_monitor_baseline_tau
                baseline = (tau * self._kl_monitor_baseline) + ((1.0 - tau) * kl_detached)
                if not torch.isfinite(baseline).all():
                    if os.environ.get("RLVAE_DEBUG") == "1":
                        print("[DEBUG] KL baseline update produced non-finite value; resetting to current KL.")
                    baseline = kl_detached.clone()
                self._kl_monitor_baseline = baseline
            kl_monitor = (kl_detached - self._kl_monitor_baseline).abs()
        else:
            if os.environ.get("RLVAE_DEBUG") == "1":
                print(f"[DEBUG] kl_loss detached contained non-finite values: {kl_detached}")
            self._kl_monitor_baseline = None
        kl_monitor = torch.nan_to_num(kl_monitor, nan=0.0, posinf=0.0, neginf=0.0)
        if self._kl_monitor_baseline is None:
            self._kl_monitor_baseline = torch.nan_to_num(kl_detached.clone(), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            self._kl_monitor_baseline = torch.nan_to_num(self._kl_monitor_baseline, nan=kl_detached.new_tensor(0.0), posinf=kl_detached.new_tensor(0.0), neginf=kl_detached.new_tensor(0.0))
        kl_aux_metrics['monitor/kl_centered_abs'] = kl_monitor.detach()
        kl_aux_metrics['monitor/kl_baseline'] = self._kl_monitor_baseline.detach()
        
        # Route debugging (one-time): show how total loss is assembled
        if not hasattr(self, "_route_debug_printed"):
            try:
                print(
                    f"[LOSS ROUTE] recon={float(recon_loss):.4f} | kl={float(kl_loss):.4f} (w={float(kl_weight):.2f}) | "
                    f"flow={float(flow_loss):.4f} | loop={float(loop_penalty):.4f} | use_riem_kl={bool(use_riemannian_kl and metric_tensor is not None)}"
                )
            except Exception:
                pass
            self._route_debug_printed = True
        
        # Combine losses
        total_loss = recon_loss + kl_weight * kl_loss + flow_loss + loop_penalty
        # Auxiliary μ L2 anchor (encourages μ toward 0 to align with prior/centroids)
        mu_l2_pen = torch.tensor(0.0, device=x.device)
        if self.mu_l2_weight > 0.0 and isinstance(mu, torch.Tensor):
            mu_l2_pen = (mu.pow(2).sum(dim=-1)).mean() * self.mu_l2_weight
            total_loss = total_loss + mu_l2_pen
            self.loss_history['mu_l2'].append(mu_l2_pen.item())
            kl_aux_metrics['loss/mu_l2_penalty'] = mu_l2_pen.detach()
            if os.environ.get('RLVAE_DEBUG') == '1' and not hasattr(self, '_mu_l2_debugged'):
                print(f"[MU L2] weight={self.mu_l2_weight:.3f}, penalty={mu_l2_pen.item():.6f}")
                self._mu_l2_debugged = True
        if not torch.isfinite(total_loss) and os.environ.get("RLVAE_DEBUG") == "1":
            print("[DEBUG] total_loss is not finite!", total_loss)
            print(f"[DEBUG] recon_loss: {recon_loss}, kl_loss: {kl_loss}, flow_loss: {flow_loss}, loop_penalty: {loop_penalty}")
        
        # If metric_tensor is used, print stats
        if metric_tensor is not None and z_samples is not None and os.environ.get("RLVAE_DEBUG") == "1":
            try:
                G_eval_debug, rep_debug = self._evaluate_metric(z_samples, metric_tensor, rhmc_posterior, with_rep=True)
                if G_eval_debug is not None and rep_debug is not None:
                    rep_debug = rep_debug.lower()
                    if rep_debug == "ginv":
                        chol_dbg, _ = self._cholesky_spd(G_eval_debug, jitter=1e-6)
                        d_dbg = G_eval_debug.shape[-1]
                        eye_dbg = torch.eye(d_dbg, device=G_eval_debug.device, dtype=G_eval_debug.dtype).unsqueeze(0).expand_as(G_eval_debug)
                        G_eval_debug = torch.cholesky_solve(eye_dbg, chol_dbg)
                    eigvals = torch.linalg.eigvalsh(G_eval_debug.float())
                    print(
                        "[DEBUG] Metric eigvals: min",
                        eigvals.min().item(),
                        "max",
                        eigvals.max().item(),
                        "mean",
                        eigvals.mean().item(),
                    )
            except Exception as e:
                print(f"[DEBUG] Error computing metric tensor stats: {e}")
        
        # If latent z_samples is available, print stats
        if z_samples is not None and os.environ.get("RLVAE_DEBUG") == "1":
            try:
                print("[DEBUG] Latent z stats: min", z_samples.min().item(), "max", z_samples.max().item(), "mean", z_samples.mean().item(), "std", z_samples.std().item())
            except Exception as e:
                print(f"[DEBUG] Error computing latent z stats: {e}")
        
        # Store in history
        self.loss_history['reconstruction'].append(recon_loss.item())
        self.loss_history['kl_divergence'].append(kl_loss.item())
        self.loss_history['flow_loss'].append(flow_loss.item())
        self.loss_history['loop_penalty'].append(loop_penalty.item())
        self.loss_history['total'].append(total_loss.item())
        
        # Metric regularization (only if trainable)
        if metric_tensor is not None and getattr(metric_tensor, 'trainable', False) and self.metric_reg_weight > 0.0:
            z_reg = z_samples.detach() if z_samples is not None else torch.randn(32, mu.shape[-1], device=x.device)
            if self.metric_reg_type == 'determinant':
                G = metric_tensor.compute_metric(z_reg)
                logdet = torch.logdet(G)
                metric_reg = ((logdet - self.metric_reg_target) ** 2).mean() * self.metric_reg_weight
            elif self.metric_reg_type == 'condition':
                G = metric_tensor.compute_metric(z_reg)
                eigvals = torch.linalg.eigvals(G)
                cond = (eigvals.abs().max(dim=-1).values / eigvals.abs().min(dim=-1).values).mean()
                metric_reg = ((cond - self.metric_reg_target) ** 2) * self.metric_reg_weight
            elif self.metric_reg_type == 'smoothness':
                # Smoothness: penalize large changes in G(z) for nearby z
                z2 = z_reg + 0.01 * torch.randn_like(z_reg)
                G1 = metric_tensor.compute_metric(z_reg)
                G2 = metric_tensor.compute_metric(z2)
                metric_reg = ((G1 - G2) ** 2).mean() * self.metric_reg_weight
            self.loss_history['metric_reg'].append(metric_reg.item())
            total_loss = total_loss + metric_reg

        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_divergence_loss': kl_loss,
            'flow_loss': flow_loss,
            'loop_penalty': loop_penalty,
            'mu_l2_penalty': mu_l2_pen,
            'loss_weights': {
                'beta': self.beta,
                'riemannian_beta': self.riemannian_beta,
                'loop_penalty_weight': self.loop_penalty_weight
            },
            'metric_reg': metric_reg, # NEW
            'loss_details': {
                **kl_aux_metrics,
                **({
                    f'preview/{k}': (v[:5] if isinstance(v, torch.Tensor) and v.dim() > 0 else v)
                    for k, v in kl_components_sample.items()
                } if kl_components_sample else {}),
            }
        }
    
    def get_loss_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of loss history.
        
        Returns:
            Dictionary with loss statistics
        """
        summary = {}
        for loss_name, history in self.loss_history.items():
            if len(history) > 0:
                summary[f'{loss_name}_mean'] = np.mean(history)
                summary[f'{loss_name}_std'] = np.std(history)
                summary[f'{loss_name}_min'] = np.min(history)
                summary[f'{loss_name}_max'] = np.max(history)
                summary[f'{loss_name}_recent'] = history[-10:] if len(history) >= 10 else history
        
        return summary
    
    def reset_history(self):
        """Reset loss history."""
        for key in self.loss_history:
            self.loss_history[key] = []
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get loss manager configuration.
        
        Returns:
            Dictionary of configuration parameters
        """
        return {
            'beta': self.beta,
            'riemannian_beta': self.riemannian_beta,
            'loop_penalty_weight': self.loop_penalty_weight,
            'device': str(self.device),
            'metric_reg_weight': self.metric_reg_weight,
            'metric_reg_type': self.metric_reg_type,
            'metric_reg_target': self.metric_reg_target,
            'mu_l2_weight': self.mu_l2_weight,
            'recon_scale': self.recon_scale,
            'kl_monitor_baseline_tau': self.kl_monitor_baseline_tau,
            'metric_representation': self.metric_representation,
            'kl_prior_mode': self.kl_prior_mode,
            'kl_use_metric_normalization': self.kl_use_metric_normalization,
            'kl_metric_norm_mode': self.kl_metric_norm_mode,
            'kl_amp_safe': self.kl_amp_safe,
        }
