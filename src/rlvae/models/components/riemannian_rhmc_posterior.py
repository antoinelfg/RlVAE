"""
Baseline Riemannian RHMC Posterior - Simplified Version
======================================================

Minimal implementation without complex constraints or stability checks.
"""
import math
import os
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metric_utils import half_logdet_volume
from ...utils.grad_debug import volume_grad_sanity
from ...utils.stagec_debugger import stagec_debugger

_HALF_DTYPES = (torch.float16, torch.bfloat16)


def _to_float32(t: torch.Tensor) -> torch.Tensor:
    """Promote low-precision tensors to float32 for linalg ops."""
    return t.float() if t.dtype in _HALF_DTYPES else t


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    """Return the symmetric part of a batch of matrices."""
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _safe_cholesky(matrix: torch.Tensor, jitter: float) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Compute a stable Cholesky factor with AMP safety (upcast to float32).

    Prefer torch.linalg.cholesky_ex when available to detect failures and
    retry with a small diagonal jitter for numerical stability.
    
    Returns:
        chol: Cholesky factor
        stabilized: Stabilized matrix (original or with jitter added)
        was_stabilized: True if jitter was added
    """
    def _chol(mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mat32 = mat.float() if mat.dtype in (torch.float16, torch.bfloat16) else mat
        if hasattr(torch.linalg, "cholesky_ex"):
            chol32, info = torch.linalg.cholesky_ex(mat32)
            return chol32, info
        chol32 = torch.linalg.cholesky(mat32)
        return chol32, torch.zeros((), dtype=torch.int64, device=mat32.device)

    try:
        chol, info = _chol(matrix)
        # If cholesky_ex reports non‑SPD, trigger jitter path
        if isinstance(info, torch.Tensor) and info.numel() > 0 and (info > 0).any():
            raise RuntimeError("cholesky_ex reported non‑SPD")
        stagec_debugger.log_event(
            "safe_cholesky",
            {
                "shape": list(matrix.shape),
                "jitter": float(jitter),
                "stabilized": False,
            },
        )
        return chol, matrix, False  # No stabilization needed
    except RuntimeError:
        d = matrix.shape[-1]
        eye = torch.eye(d, device=matrix.device, dtype=matrix.dtype).unsqueeze(0)
        stabilized = matrix + jitter * eye
        chol, _ = _chol(stabilized)
        stagec_debugger.log_fallback(
            "safe_cholesky",
            reason="jitter_applied",
            payload={
                "shape": list(matrix.shape),
                "jitter": float(jitter),
                "d": d,
            },
        )
        return chol, stabilized, True  # Stabilization applied


def _log_kinetic_density(
    model: nn.Module,
    z: torch.Tensor,
    rho: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """log π_kin(ρ|z) with kinetic metric G(z).
       pi_kin = N(0, G(z))
    """
    if not hasattr(model, "G"):
        raise RuntimeError("Model missing metric tensor G for kinetic density computation.")
    
    G = model.G(z)
    G = _symmetrize(G)
    d = G.shape[-1]
    eye = torch.eye(d, device=G.device, dtype=G.dtype).unsqueeze(0)
    jitter = float(max(eps, 1e-8))
    
    # Appliquer le jitter pour la stabilité
    G_jittered = G + jitter * eye
    
    # MODIFICATION 1: Calcule G_inv pour le terme quadratique
    # L'énergie est 1/2 * rho^T * G_inv * rho
    G_inv = torch.linalg.inv(G_jittered.float())
    quad = torch.einsum('bi,bij,bj->b', rho.float(), G_inv, rho.float())
    
    # MODIFICATION 2: Le terme volume pour N(0, G) est 1/2 * log(det(G^-1))
    # (ce que half_logdet_volume fait par défaut)
    half_logdet = half_logdet_volume(G_jittered, 'g', jitter=jitter)
    
    if torch.isnan(half_logdet).any():
        # Branche de secours
        jitter = max(jitter * 10, 1e-5)
        G_jittered_retry = G + jitter * eye
        
        # Recalculer les deux termes
        G_inv_retry = torch.linalg.inv(G_jittered_retry.float())
        quad = torch.einsum('bi,bij,bj->b', rho.float(), G_inv_retry, rho.float())
        half_logdet = half_logdet_volume(G_jittered_retry, 'g', jitter=jitter)

    const = z.shape[-1] * math.log(2 * math.pi)
    
    # Formule pour log N(rho | 0, G)
    return (-0.5 * quad + half_logdet - 0.5 * const).to(z.dtype)


def log_q_riem(
    z: torch.Tensor,
    mu: torch.Tensor,
    Sigma: torch.Tensor,
    *,
    min_eig: float,
) -> torch.Tensor:
    """
    Compute log-density log N_Riem(z | μ, Σ) with Σ assumed SPD.
    """
    if stagec_debugger.enabled:
        stagec_debugger.log_event(
            "log_q_riem_call",
            {
                "z_shape": list(z.shape),
                "mu_shape": list(mu.shape),
                "Sigma_shape": list(Sigma.shape),
                "min_eig": float(min_eig),
            },
        )
    # Accept inputs with arbitrary leading batch dims [..., D] and [..., D, D]
    # Broadcast mu and Sigma to z's leading shape if necessary
    D = z.shape[-1]
    lead_z = z.shape[:-1]

    # Make mu broadcastable to z
    if mu.shape[:-1] != lead_z:
        ndiff = len(lead_z) - (mu.dim() - 1)
        if ndiff > 0:
            mu = mu.view(*mu.shape[:-1], *([1] * ndiff), mu.shape[-1])
        try:
            mu = mu.expand(*lead_z, D)
        except RuntimeError as e:
            # Handle dimension mismatch by reshaping
            if mu.dim() > len(lead_z) + 1:
                # Too many dimensions, reshape to match
                mu = mu.reshape(-1, D)
                if mu.shape[0] == 1:
                    mu = mu.expand(lead_z[0], D)
                else:
                    mu = mu[:lead_z[0]]
            else:
                raise e

    # Make Sigma broadcastable to z
    if Sigma.shape[:-2] != lead_z:
        ndiffS = len(lead_z) - (Sigma.dim() - 2)
        if ndiffS > 0:
            Sigma = Sigma.view(*Sigma.shape[:-2], *([1] * ndiffS), *Sigma.shape[-2:])
        try:
            Sigma = Sigma.expand(*lead_z, D, D)
        except RuntimeError as e:
            # Handle dimension mismatch by reshaping
            if Sigma.dim() > len(lead_z) + 2:
                # Too many dimensions, reshape to match
                Sigma = Sigma.reshape(-1, D, D)
                if Sigma.shape[0] == 1:
                    Sigma = Sigma.expand(lead_z[0], D, D)
                else:
                    Sigma = Sigma[:lead_z[0]]
            else:
                raise e

    z_flat = z.reshape(-1, D)
    mu_flat = mu.reshape(-1, D)
    Sigma_flat = Sigma.reshape(-1, D, D)
    Sigma_flat = _symmetrize(Sigma_flat)
    chol, stabilized_sigma, was_stabilized = _safe_cholesky(Sigma_flat, min_eig)
    
    # DIAGNOSTIC: Log stabilization details
    if os.environ.get("RLVAE_DEBUG", "0") == "1":
        Sigma_dbg = _to_float32(Sigma_flat)
        stab_dbg = _to_float32(stabilized_sigma)
        try:
            eigvals_orig = torch.linalg.eigvalsh(Sigma_dbg)
        except Exception as e:
            eigvals_orig = None
            print(f"[LOG_Q_RIEM STABILIZATION] eigvals_orig failed: {e}")
        try:
            eigvals_stab = torch.linalg.eigvalsh(stab_dbg)
        except Exception as e:
            eigvals_stab = None
            print(f"[LOG_Q_RIEM STABILIZATION] eigvals_stab failed: {e}")
        try:
            logdet_orig = torch.linalg.slogdet(Sigma_dbg)[1]
        except Exception as e:
            logdet_orig = None
            print(f"[LOG_Q_RIEM STABILIZATION] logdet_orig failed: {e}")
        try:
            logdet_stab = torch.linalg.slogdet(stab_dbg)[1]
        except Exception as e:
            logdet_stab = None
            print(f"[LOG_Q_RIEM STABILIZATION] logdet_stab failed: {e}")

        print(f"[LOG_Q_RIEM STABILIZATION]")
        print(f"  min_eig (jitter):     {min_eig:.6f}")
        print(f"  was_stabilized:       {was_stabilized}")
        print(f"  Original Σ:")
        if eigvals_orig is not None:
            print(f"    eigenvalues:        min={eigvals_orig.min().item():.6f}, max={eigvals_orig.max().item():.6f}")
        if logdet_orig is not None:
            print(f"    log|Σ|:             {logdet_orig.mean().item():.6f}")
        if was_stabilized:
            print(f"  Stabilized Σ:")
            if eigvals_stab is not None:
                print(f"    eigenvalues:        min={eigvals_stab.min().item():.6f}, max={eigvals_stab.max().item():.6f}")
            if logdet_stab is not None:
                print(f"    log|Σ|:             {logdet_stab.mean().item():.6f}")
            if logdet_orig is not None and logdet_stab is not None:
                print(f"    Δ log|Σ|:           {(logdet_stab - logdet_orig).mean().item():+.6f}")
    
    diff = (z_flat - mu_flat).unsqueeze(-1)
    diff32 = diff.float() if diff.dtype != chol.dtype else diff
    sol32 = torch.cholesky_solve(diff32, chol)
    quad_form = torch.einsum('...ij,...ij->...', diff32, sol32)
    log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-18).sum(dim=-1)
    const = 0.5 * D * math.log(2 * math.pi)
    
    # DIAGNOSTIC: Decompose log_q into its components
    if os.environ.get("RLVAE_DEBUG", "0") == "1":
        import numpy as np
        quad_term = -0.5 * quad_form
        vol_term = -0.5 * log_det
        const_term = -const
        
        print(f"\n{'='*80}")
        if os.environ.get("RLVAE_DEBUG", "0") == "1":
            try:
                eigvals_sigma, eigvecs_sigma = torch.linalg.eigh(stab_dbg)
                diff_flat = (z_flat - mu_flat).unsqueeze(-1)
                y = torch.einsum('bij,bj->bi', eigvecs_sigma.transpose(-1, -2), diff_flat.squeeze(-1))
                contrib_per_eig = (y ** 2) / eigvals_sigma.clamp(min=1e-12)
                mahal_sq = contrib_per_eig.sum(dim=-1)
                euclidean_sq = torch.norm(diff_flat.squeeze(-1), dim=-1) ** 2

                print(f"\n[MAHALANOBIS EIGENBASIS DECOMPOSITION]")
                print(f"  Euclidean ||z-μ||²:    mean={euclidean_sq.mean().item():.4f}")
                print(f"  Mahalanobis (z-μ)ᵀΣ⁻¹(z-μ): mean={mahal_sq.mean().item():.4f}")
            except Exception as e:
                print(f"[LOG_Q_RIEM DIAGNOSE] failed: {e}")
    
    out = -0.5 * quad_form - 0.5 * log_det - const
    out = out.to(mu.dtype)
    return out.reshape(z.shape[:-1])


def log_q_riem_from_G(
    z: torch.Tensor,
    mu: torch.Tensor,
    G_mu: torch.Tensor,
    *,
    alpha: float,
    eps_reg: float,
    min_eig: float,
) -> torch.Tensor:
    """Compute log N_Riem at μ using G(μ).

    Builds Σ(μ) = α · G(μ)^{-1} + ε I then delegates to ``log_q_riem``.
    AMP-safe and numerically stabilized with ``min_eig``.
    """
    G_mu = _symmetrize(G_mu)
    # Invert in float32 for stability under AMP
    G_mu32 = G_mu.float() if G_mu.dtype in (torch.float16, torch.bfloat16) else G_mu
    G_inv_mu = torch.linalg.inv(G_mu32)
    d = G_inv_mu.shape[-1]
    eye = torch.eye(d, device=G_inv_mu.device, dtype=G_inv_mu.dtype).unsqueeze(0)
    Sigma32 = alpha * _symmetrize(G_inv_mu) + float(eps_reg) * eye
    return log_q_riem(z, mu, Sigma32.to(mu.dtype), min_eig=float(min_eig))


class RiemannianRHMCPosterior(nn.Module):
    """
    Baseline posterior sampler combining Riemannian initial sampling with RHMC exploration.
    
    Simplified version without complex constraints.
    """
    
    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        import weakref
        self._ctx = {'model': weakref.proxy(model)}
        self.device = getattr(model, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Normalise config for attribute lookups
        if hasattr(config, 'items'):
            self.config = dict(config)
        else:
            self.config = config or {}

        # Resolve hyperparameters from config (fallback to safe defaults)
        def _cfg_get(key, default):
            try:
                return self.config.get(key, default)
            except Exception:
                return default
        # RHMC integration defaults (tuned for Stage C)
        self.rhmc_steps = int(_cfg_get('rhmc_steps', 4))
        self.rhmc_step_size = float(_cfg_get('rhmc_step_size', 0.05))
        # Persist the configured step size so unexpected overrides can be detected and corrected
        self._configured_step_size = self.rhmc_step_size
        # DEBUG: Log what config was passed
        import os
        if os.environ.get("RLVAE_DEBUG", "0") == "1":
            print(f"[RHMC INIT DEBUG] config['rhmc_step_size'] = {self.config.get('rhmc_step_size', 'NOT FOUND')}")
            print(f"[RHMC INIT DEBUG] self.rhmc_step_size = {self.rhmc_step_size}")
        # Resolve alpha with hard override support (env or model attribute)
        env_alpha = os.environ.get("RLVAE_ALPHA", None)
        if env_alpha is not None:
            try:
                raw_alpha = float(env_alpha)
            except Exception:
                raw_alpha = float(_cfg_get('rhmc_alpha', 1.))
        else:
            raw_alpha = float(_cfg_get('rhmc_alpha', 1.)) ####sfsd
        if not math.isfinite(raw_alpha) or raw_alpha <= 0.0:
            fallback_alpha = getattr(model, 'posterior_local_alpha', 1)
            self.rhmc_alpha = float(fallback_alpha)
            print(f"[RHMC INIT] α<=0 detected ({raw_alpha}); using fallback {self.rhmc_alpha}")
        else:
            self.rhmc_alpha = raw_alpha
        # Propagate resolved alpha back to model for downstream consumers
        try:
            setattr(model, 'rhmc_alpha', float(self.rhmc_alpha))
        except Exception:
            pass
        try:
            setattr(model, 'posterior_local_alpha', float(self.rhmc_alpha))
        except Exception:
            pass
        self.eps_reg = float(_cfg_get('rhmc_eps_reg', _cfg_get('eps_regularization', 1e-3)))

        # Numerical guards
        self.min_cov_eig = float(_cfg_get('min_cov_eig', 1e-3))  # ensure >= eps_reg
        if self.min_cov_eig < self.eps_reg:
            self.min_cov_eig = self.eps_reg
        self.metric_eig_ceiling = float(_cfg_get('metric_eig_ceiling', float('inf')))
        if not math.isfinite(self.metric_eig_ceiling) or self.metric_eig_ceiling <= 0:
            self.metric_eig_ceiling = float('inf')
        self.max_cov_eig = float(_cfg_get('max_cov_eig', 1.0))
        if not math.isfinite(self.max_cov_eig) or self.max_cov_eig <= 0:
            self.max_cov_eig = float('inf')
        self.max_alpha_eff = float(_cfg_get('max_alpha_eff', 50.0))
        if not math.isfinite(self.max_alpha_eff) or self.max_alpha_eff <= 0:
            self.max_alpha_eff = float('inf')
        self.max_momentum_norm = float(_cfg_get('max_momentum_norm', 3.0))
        self.max_velocity_norm = float(_cfg_get('max_velocity_norm', 2.0))
        self.max_position_step = float(_cfg_get('max_position_step', 1.0))
        self.max_position_norm = float(_cfg_get('max_position_norm', 0.0))
        # Keep factorized path disabled by default
        self.use_factorized_G_mu = False
        # Prior mode (affects log p computation under Monte-Carlo KL)
        prior_mode_cfg = _cfg_get('kl_prior_mode', _cfg_get('riemannian_prior_mode', 'uniform'))
        try:
            self.kl_prior_mode = str(prior_mode_cfg).lower() if prior_mode_cfg is not None else 'volume_gaussian'
        except Exception:
            self.kl_prior_mode = 'volume_gaussian'
        if self.kl_prior_mode not in {'volume_gaussian', 'uniform', 'gaussian'}:
            print(f"[RHMC INIT] ⚠️ Unknown kl_prior_mode='{self.kl_prior_mode}', falling back to 'volume_gaussian'")
            self.kl_prior_mode = 'volume_gaussian'
        self.uniform_prior_log_norm = float(_cfg_get('uniform_prior_log_norm', 0.0))
        self.volume_bias_weight = float(_cfg_get('volume_bias_weight', 2.0))
        self.volume_grad_scale = float(_cfg_get('volume_grad_scale', 1.0))
        # New stability/geometry options
        self.soft_position_norm = float(_cfg_get('soft_position_norm', 5.5))
        self.kinetic_grad_enabled = bool(_cfg_get('kinetic_grad_enabled', True))
        self.kinetic_grad_weight = float(_cfg_get('kinetic_grad_weight', 1.0))
        self.projection_step_scale = float(_cfg_get('projection_step_scale', 0.05))
        self.initial_max_norm = float(_cfg_get('initial_max_norm', 1.5))
        # Stabilization for volume-gradient path
        self.volume_grad_jitter = float(_cfg_get('volume_grad_jitter', 1e-4))
        self.volume_grad_eig_floor = float(_cfg_get('volume_grad_eig_floor', 1e-2))
        # Control how the volume force is computed (diagnostics/toggles)
        self.volume_force_representation = str(_cfg_get('volume_force_representation', 'g')).lower()  # 'g' or 'ginv'
        self.volume_force_sign = float(_cfg_get('volume_force_sign', 1.0))  # +1 or -1
        self._volume_force_logged = False  # For one-time debug logging
        self._step_scaling_logged = False  # For one-time debug logging
        
        # FIX: Covariance normalization and target radius (critical for correct KL)
        self.sigma_normalization_mode = str(_cfg_get('sigma_normalization_mode', 'none')).lower()
        self.initial_target_radius = float(_cfg_get('initial_target_radius', 0.))
        # Volume acceptance defaults: disabled unless explicitly requested
        self.initial_volume_tolerance = float(_cfg_get('initial_volume_tolerance', 0.))
        self.initial_max_retries = int(_cfg_get('initial_max_retries', 0))
        self.initial_volume_logdet_tol = float(_cfg_get('initial_volume_logdet_tol', 0.0))
        self.initial_volume_max_adjustments = int(_cfg_get('initial_volume_max_adjustments', 4))
        self.initial_volume_shrink_factor = float(_cfg_get('initial_volume_shrink_factor', 0.5))
        self.initial_volume_grow_factor = float(_cfg_get('initial_volume_grow_factor', 2.0))
        self.initial_volume_min_alpha = float(_cfg_get('initial_volume_min_alpha', 1e-4))
        self.initial_volume_max_alpha = float(_cfg_get('initial_volume_max_alpha', 10.0))
        self.initial_volume_warmup_steps = int(_cfg_get('initial_volume_warmup_steps', 0))
        self.initial_volume_warmup_step_size = float(_cfg_get('initial_volume_warmup_step_size', 0.05))
        self.initial_volume_warmup_max_step = float(_cfg_get('initial_volume_warmup_max_step', 0.0))
        if not (0.0 < self.initial_volume_shrink_factor < 1.0):
            self.initial_volume_shrink_factor = 0.5
        if not math.isfinite(self.initial_volume_grow_factor) or self.initial_volume_grow_factor <= 1.0:
            self.initial_volume_grow_factor = 2.0
        if self.initial_volume_min_alpha <= 0.0:
            self.initial_volume_min_alpha = 1e-4
        if not math.isfinite(self.initial_volume_max_alpha) or self.initial_volume_max_alpha <= self.initial_volume_min_alpha:
            self.initial_volume_max_alpha = max(self.initial_volume_min_alpha * 10.0, 1.0)
        self.max_quadratic_growth = float(_cfg_get('max_quadratic_growth', 12.0))
        if not math.isfinite(self.max_quadratic_growth) or self.max_quadratic_growth <= 0:
            self.max_quadratic_growth = 0.0

        # Default return behaviour
        self.default_return_initial = True
        self.default_return_log_prob = True
        self.default_return_traj = False
        self.default_with_jacobian = False

        # Debug: confirm safety params are set
        print(
            "⚙️ [RHMC INIT] Safety bounds:"
            f" momentum={self.max_momentum_norm},"
            f" velocity={self.max_velocity_norm},"
            f" step={self.max_position_step},"
            f" norm={self.max_position_norm}"
        )
        print(f"⚙️ [RHMC INIT] Step params: rhmc_steps={self.rhmc_steps}, rhmc_step_size={self.rhmc_step_size}, rhmc_alpha={self.rhmc_alpha}")
        try:
            print(f"[RHMC INIT] Params: rhmc_steps={int(self.rhmc_steps)}, rhmc_step_size={float(self.rhmc_step_size):.6g}, rhmc_alpha={float(self.rhmc_alpha):.6g}, rhmc_eps_reg={float(self.eps_reg):.6g}")
            print(f"[RHMC INIT] Covariance: sigma_normalization_mode='{self.sigma_normalization_mode}', initial_target_radius={self.initial_target_radius:.6g}")
        except Exception:
            pass
        # TRACE-once guard and flags
        self._trace_printed = False
        self._first_forward_done = False
        self._warned_no_grad = False
        
    def sample_riemannian_rhmc_posterior(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        *,
        return_log_prob: Optional[bool] = None,
        return_traj: Optional[bool] = None,
        return_initial: Optional[bool] = None,
        with_jacobian: Optional[bool] = None,
        alpha: Optional[float] = None,
        eps_reg: Optional[float] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Sample from the RHMC posterior with configurable return extras.

        Args:
            mu: Encoder mean [B, D]
            log_var: Encoder log variance [B, D] (unused, kept for interface compatibility)
            return_log_prob: Whether to return log q (defaults to class config)
            return_traj: Whether to return trajectory diagnostics
            return_initial: Whether to return initial z₀ samples
            with_jacobian: Whether to request Jacobian accumulation (placeholder)
        """
        grad_ctx = torch.enable_grad() if not torch.is_grad_enabled() else nullcontext()
        amp_ctx = nullcontext()
        if torch.cuda.is_available() and hasattr(torch.cuda, "amp"):
            amp_ctx = torch.cuda.amp.autocast(enabled=False)

        def _run() -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
            if not torch.is_grad_enabled():
                raise RuntimeError(
                    "RiemannianRHMCPosterior requires gradients; wrap call in torch.enable_grad() before sampling."
                )
            # Guard against accidental external overwrites of the step size
            try:
                cfg_eps = getattr(self, "_configured_step_size", None)
                if cfg_eps is not None:
                    if (self.rhmc_step_size < 1e-3) or (abs(self.rhmc_step_size - cfg_eps) > 1e-6):
                        if os.environ.get("RLVAE_DEBUG", "0") == "1":
                            print(f"[RHMC STEP GUARD] restoring rhmc_step_size from {self.rhmc_step_size} to configured {cfg_eps}")
                        self.rhmc_step_size = cfg_eps
            except Exception:
                pass
            # Route tracing: RHMC parameters (once)
            if not hasattr(self, '_rhmc_traced'):
                print(f"[ROUTE] RHMC: steps={self.rhmc_steps}, step_size={self.rhmc_step_size}, alpha={self.rhmc_alpha}, eps_reg={self.eps_reg}")
                self._rhmc_traced = True

            # Resolve behaviour toggles with fallback to constructor defaults
            log_prob_flag = self.default_return_log_prob if return_log_prob is None else bool(return_log_prob)
            traj_flag = self.default_return_traj if return_traj is None else bool(return_traj)
            initial_flag = self.default_return_initial if return_initial is None else bool(return_initial)
            jac_flag = self.default_with_jacobian if with_jacobian is None else bool(with_jacobian)

            # Step 1: Riemannian initial sampling
            alpha_eff = float(alpha) if (alpha is not None) else self._resolve_alpha()
            eps_backup = self.eps_reg
            if eps_reg is not None:
                try:
                    self.eps_reg = float(eps_reg)
                except Exception:
                    pass
            if os.environ.get("RLVAE_DEBUG", "0") == "1":
                print("[RHMC DEBUG] entering _sample_initial_riemannian")
            # Always rebuild Σμ from the current μ; never reuse stale cache
            self._last_sigma_mu = None
            z0, Sigma_mu = self._sample_initial_riemannian(mu, log_var, alpha_eff)
            self._last_sigma_mu = Sigma_mu.detach().clone()

            # Step 2: RHMC exploration (differentiable pushforward)
            z_final, traj_states, delta_logdet = self._rhmc_exploration(z0, record_traj=traj_flag, mu=mu)

            debug_mode = os.environ.get("RLVAE_DEBUG", "0") == "1"
            if debug_mode:
                with torch.no_grad():
                    model = self._ctx['model']

                    def _metric_stats(label: str, pts: torch.Tensor) -> None:
                        try:
                            if hasattr(model, "G"):
                                G = _symmetrize(model.G(pts))
                            elif hasattr(model, "G_inv"):
                                G_inv = _symmetrize(model.G_inv(pts))
                                G = torch.linalg.inv(G_inv.double()).to(G_inv.dtype)
                            else:
                                print(f"[RHMC DEBUG] {label}: model missing G/G_inv.")
                                return
                            eigvals = torch.linalg.eigvalsh(G.double())
                            min_eig = eigvals.min().item()
                            max_eig = eigvals.max().item()
                            cond = float(max_eig / max(min_eig, 1e-12))
                            logdet = torch.log(torch.clamp(eigvals, min=1e-18)).sum(-1).mean().item()
                            print(
                                f"[RHMC DEBUG] {label}: eig_min={min_eig:.3e}, eig_max={max_eig:.3e}, "
                                f"cond={cond:.3e}, log|G|={logdet:.3e}"
                            )
                        except Exception as exc:
                            print(f"[RHMC DEBUG] {label}: metric stats failed ({exc})")

                    _metric_stats("G(z0)", z0.detach())
                    _metric_stats("G(zK)", z_final.detach())

                    diff_mu = torch.norm(z_final - mu, dim=-1)
                    diff_start = torch.norm(z_final - z0, dim=-1)
                    mu_norm = torch.norm(mu, dim=-1)
                    print(
                        "[RHMC DEBUG] latent drift: "
                        f"||mu|| mean={mu_norm.mean().item():.3e}, "
                        f"||zK-mu|| mean={diff_mu.mean().item():.3e} (max={diff_mu.max().item():.3e}), "
                        f"||zK-z0|| mean={diff_start.mean().item():.3e} (max={diff_start.max().item():.3e})"
                    )
                    if traj_states:
                        rho0 = traj_states[0]['rho']
                        rhoS = traj_states[-1]['rho']
                        rho0_norm = torch.norm(rho0, dim=-1)
                        rhoS_norm = torch.norm(rhoS, dim=-1)
                        print(
                            "[RHMC DEBUG] momentum norms: "
                            f"initial mean={rho0_norm.mean().item():.3e}, "
                            f"final mean={rhoS_norm.mean().item():.3e}"
                        )

            # TRACE: invariants and dtype diagnostics (first batch only)
            try:
                if os.environ.get('RLVAE_TRACE', '0') == '1' and not self._trace_printed:
                    z0_norm = torch.norm(z0, dim=1)
                    zK_norm = torch.norm(z_final, dim=1)
                    max_diff = (z_final - z0).abs().max().item()
                    evals = torch.linalg.eigvalsh(Sigma_mu.float())
                    eigmin = evals.min().item()
                    eigmed = evals.median().item()
                    cond = (evals.max() / evals.min().clamp_min(1e-12)).item()
                    logdet_S = torch.logdet(Sigma_mu.float()).median().item()
                    mu_dt = str(mu.dtype)
                    try:
                        Ginv_mu = self._get_inverse_metric(mu)
                        Ginv_dt = str(Ginv_mu.dtype)
                    except Exception:
                        Ginv_dt = 'n/a'
                    Sigma_dt = str(Sigma_mu.dtype)
                    print(f"TRACE RHMC dtype: mu={mu_dt}, Ginv={Ginv_dt}, Sigma={Sigma_dt}")
                    print(f"TRACE RHMC Σ(μ): eigmin={eigmin:.3e}, median={eigmed:.3e}, cond={cond:.3e}, logdet_med={logdet_S:.3e}")
                    print(f"TRACE RHMC alpha, eps: alpha={float(alpha_eff):.6g}, eps_reg={float(self.eps_reg):.3g}, autocast={torch.is_autocast_enabled()}")
                    if int(self.rhmc_steps) == 0:
                        print(
                            f"TRACE RHMC z0,zK norms: ||z0|| mean={z0_norm.mean().item():.4g} std={z0_norm.std().item():.4g}, "
                            f"||zK|| mean={zK_norm.mean().item():.4g} std={zK_norm.std().item():.4g}, max|zK-z0|={max_diff:.3e}"
                        )
                    else:
                        print(
                            f"TRACE RHMC z0,zK norms: ||z0|| mean={z0_norm.mean().item():.4g} std={z0_norm.std().item():.4g}, "
                            f"||zK|| mean={zK_norm.mean().item():.4g} std={zK_norm.std().item():.4g} (steps={int(self.rhmc_steps)})"
                        )
                    self._trace_printed = True
            except Exception:
                pass

            # Compute log-density if requested
            log_q = None
            if log_prob_flag:
                base_points = z0 if initial_flag else z_final
                log_q = self._compute_log_riemannian_gaussian(
                    base_points,
                    mu,
                    log_var,
                    covariance=Sigma_mu,
                )

            # Prepare trajectory info payload
            traj_info = None
            if traj_flag:
                traj_info = {
                    'with_jacobian': jac_flag,
                    'rhmc_steps': int(self.rhmc_steps),
                    'step_size': float(self.rhmc_step_size),
                    'alpha': float(alpha_eff),
                    'eps_reg': float(self.eps_reg),
                    'jac_logdet': None if jac_flag else None,
                    'trajectory': traj_states,
                    'Sigma_mu': Sigma_mu.detach(),
                    'delta_vol': delta_logdet,
                    'delta_kin': (
                        _log_kinetic_density(self._ctx['model'], traj_states[0]['z'], traj_states[0]['rho'], self.eps_reg)
                        - _log_kinetic_density(self._ctx['model'], traj_states[-1]['z'], traj_states[-1]['rho'], self.eps_reg)
                    ) if len(traj_states) > 0 else torch.zeros(
                        z_final.shape[0],
                        device=z_final.device,
                        dtype=z_final.dtype,
                    ),
                }

            outputs = [z_final]
            if log_prob_flag:
                outputs.append(log_q)
            if initial_flag:
                outputs.append(z0)
            if traj_flag:
                outputs.append(traj_info)

            self.eps_reg = eps_backup

            if len(outputs) == 1:
                return outputs[0]
            return tuple(outputs)

        with grad_ctx:
            with amp_ctx:
                return _run()
    
    def _sample_initial_once(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        alpha: float,
        *,
        target_radius: Optional[float] = None,
        diagnostics_enabled: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a single draw z₀ ~ N_Riem(μ, Σ(μ)) with Σ = α·Ĝ^{-1}(μ) + εI.
        Returns both the samples and the covariance used (for log-density).
        """
        batch_size, latent_dim = mu.shape
        stagec_debugger.log_event(
            "sample_initial_once_start",
            {
                "batch_size": batch_size,
                "latent_dim": latent_dim,
                "alpha": float(alpha),
                "target_radius": float(target_radius or 0.0),
                "use_factorized_G_mu": bool(self.use_factorized_G_mu),
                "diagnostics_enabled": bool(diagnostics_enabled),
            },
        )

        try:
            if self.use_factorized_G_mu:
                # Efficient sampling using factorization of G(μ):
                # z0 = μ + C^{-T}·sqrt(α)·ξ1 + sqrt(ε)·ξ2 where G(μ) = C Cᵀ
                # Build covariance explicitly for return: Σ = α·G^{-1}(μ) + εI
                if os.environ.get("RLVAE_DEBUG", "0") == "1":
                    print("[RHMC DEBUG] factorized path -> _make_covariance")
                # 1) Factorize G(μ)
                d = mu.shape[-1]
                I = torch.eye(d, device=mu.device, dtype=mu.dtype).unsqueeze(0)
                G_mu = self._ctx['model'].G(mu)
                G_mu = _symmetrize(G_mu)
                if diagnostics_enabled and os.environ.get("RLVAE_DEBUG", "0") == "1":
                    grad_counter = getattr(self, "_mu_gradient_diag_count", 0)
                    if grad_counter < 5:
                        try:
                            with torch.enable_grad():
                                mu_dbg = mu.detach().clone().requires_grad_(True)
                                G_mu_dbg = self._ctx['model'].G(mu_dbg)
                                G_mu_dbg = _symmetrize(G_mu_dbg)
                                half_logdet_mu_dbg = half_logdet_volume(G_mu_dbg.float(), 'g', jitter=self.eps_reg)
                                logdet_mu_dbg = -half_logdet_mu_dbg
                                grad_mu_dbg = torch.autograd.grad(
                                    logdet_mu_dbg.sum(),
                                    mu_dbg,
                                    retain_graph=False,
                                    create_graph=False,
                                    allow_unused=True,
                                )[0]
                            grad_mu_dbg = torch.zeros_like(mu) if grad_mu_dbg is None else grad_mu_dbg.detach()
                            grad_norm = torch.norm(grad_mu_dbg, dim=-1)
                            print("\n[MU GRADIENT DEBUG]")
                            print(f"  log|G(μ)| mean={logdet_mu_dbg.mean().item():+.4f}")
                            print(f"  ||∇ log|G(μ)|| mean={grad_norm.mean().item():.4f}, max={grad_norm.max().item():.4f}")
                            centroids = getattr(self._ctx['model'], 'centroids_tens', None)
                            if centroids is not None:
                                centroids = centroids.to(mu.device, mu.dtype)
                                dists = torch.cdist(mu.detach(), centroids)
                                min_dists, min_idx = dists.min(dim=1)
                                nearest = centroids[min_idx]
                                dir_to_centroid = torch.nn.functional.normalize(nearest - mu.detach(), dim=-1)
                                grad_dir = torch.nn.functional.normalize(grad_mu_dbg, dim=-1)
                                align = (dir_to_centroid * grad_dir).sum(dim=-1)
                                print(
                                    f"  Nearest-centroid dist mean={min_dists.mean().item():.4f}, "
                                    f"grad alignment mean={align.mean().item():.4f}"
                                )
                            else:
                                print("  Centroids unavailable; skipping alignment diagnostic.")
                        except Exception as grad_exc:
                            print(f"[MU GRADIENT DEBUG] failed: {grad_exc}")
                        self._mu_gradient_diag_count = grad_counter + 1
                C, _, _ = _safe_cholesky(G_mu + self.eps_reg * I, self.min_cov_eig)
                # 2) Multi-try sampling
                if C.ndim == 2:
                    C = C.unsqueeze(0)
                B, D = mu.shape
                K = int(getattr(self, 'initial_n_candidates', 1))
                K = max(1, K)
                xi1 = torch.randn(B, K, D, device=mu.device, dtype=C.dtype)
                xi2 = torch.randn(B, K, D, device=mu.device, dtype=mu.dtype)
                Ct = C.transpose(-1, -2).unsqueeze(1)  # [B,1,D,D]
                y = torch.linalg.solve_triangular(Ct, xi1.unsqueeze(-1), upper=True).squeeze(-1)
                z_cand = mu.float().unsqueeze(1) + (alpha ** 0.5) * y + (self.eps_reg ** 0.5) * xi2.float()
                with torch.no_grad():
                    z_eval = z_cand.reshape(B * K, D)
                    Gz = self._ctx['model'].G(z_eval)
                    # Selection score: invert sign to favor lower metric volume (smaller log|G^{-1}|).
                    h = -half_logdet_volume(Gz, 'g', jitter=self.eps_reg).reshape(B, K)
                best = torch.argmax(h, dim=1)
                
                # DIAGNOSTIC: Analyze candidates before selection
                # Compute G^{-1}(μ) and Sigma early for diagnostics
                G_inv_mu_diag = torch.linalg.inv(G_mu.float())
                Sigma_diag = self._make_covariance(
                    G_inv_mu_diag.to(mu.dtype),
                    alpha,
                    target_radius=target_radius,
                )
                if diagnostics_enabled and os.environ.get("RLVAE_DEBUG", "0") == "1":
                    batch_counter = getattr(self, '_debug_batch_counter', 0)
                    self._diagnose_candidates(z_cand.to(mu.dtype), mu, Sigma_diag, h, best, batch_counter)
                    self._debug_batch_counter = batch_counter + 1
                
                z0 = z_cand[torch.arange(B, device=mu.device), best].to(mu.dtype)
                # Optional radial cap on initial displacement
                try:
                    max_norm = float(getattr(self, 'initial_max_norm', 0.0))
                except Exception:
                    max_norm = 0.0
                if max_norm and max_norm > 0:
                    delta = z0 - mu
                    dnorm = torch.norm(delta, dim=-1, keepdim=True)
                    scale = torch.clamp(max_norm / (dnorm + 1e-12), max=1.0)
                    z0 = mu + scale * delta
                # 5) Prepare Σ for return (already computed for diagnostics)
                Sigma = Sigma_diag
                # Save z_selected (after multi-try, before volume) and z_base for stage tracking
                z_selected = z0.clone() if diagnostics_enabled and os.environ.get("RLVAE_DEBUG", "0") == "1" else None
                # Optional manifold-aware acceptance: require non-decrease in log|G^{-1}|
                z0 = self._initial_accept_volume(z0, mu, Sigma)
                stagec_debugger.log_event(
                    "sample_initial_once_factorized",
                    {
                        "candidate_stats": {
                            "mean_norm": float(torch.norm(z0 - mu, dim=-1).mean().item()),
                            "Sigma_logdet": float(torch.linalg.slogdet(Sigma.float())[1].mean().item()),
                        },
                        "target_radius": float(target_radius or 0.0),
                    },
                )
                
                # DIAGNOSTIC: Comprehensive initial sampling analysis with stage comparison
                if diagnostics_enabled and os.environ.get("RLVAE_DEBUG", "0") == "1":
                    self._diagnose_initial_sample(z_selected, z0, mu, Sigma, alpha)
                    self._diagnose_metric_at_samples(z0, mu, z_selected)
                
                return z0, Sigma
            else:
                # Standard covariance-based path with multi-try sampling
                # BUGFIX: _make_covariance expects G^{-1}(μ). We were incorrectly
                # passing inv(G^{-1}(μ)) = G(μ), which flips the intended precision/covariance.
                # Use the inverse metric directly.
                if os.environ.get("RLVAE_DEBUG", "0") == "1":
                    print("[RHMC DEBUG] standard path -> _make_covariance")
                G_inv_mu = self._get_inverse_metric(mu)
                Sigma = self._make_covariance(
                    G_inv_mu,
                    alpha,
                    target_radius=target_radius,
                )
            debug_mode = os.environ.get("RLVAE_DEBUG", "0") == "1"
            chol, Sigma, chol_was_stabilized = _safe_cholesky(Sigma, self.min_cov_eig)
            if debug_mode and diagnostics_enabled:
                grad_counter = getattr(self, "_mu_gradient_diag_count", 0)
                try:
                    with torch.enable_grad():
                        mu_dbg = mu.detach().clone().requires_grad_(True)
                        G_mu_dbg = self._ctx['model'].G(mu_dbg)
                        G_mu_dbg = _symmetrize(G_mu_dbg)
                        half_logdet_mu_dbg = half_logdet_volume(G_mu_dbg.float(), 'g', jitter=self.eps_reg)
                        logdet_mu_dbg = half_logdet_mu_dbg
                        grad_mu_dbg = torch.autograd.grad(
                            logdet_mu_dbg.sum(),
                            mu_dbg,
                            retain_graph=False,
                            create_graph=False,
                            allow_unused=True,
                        )[0]
                    grad_mu_dbg = torch.zeros_like(mu) if grad_mu_dbg is None else grad_mu_dbg.detach()
                    grad_norm = torch.norm(grad_mu_dbg, dim=-1)
                    print("\n[MU GRADIENT DEBUG]")
                    print(f"  log|G(μ)| mean={logdet_mu_dbg.mean().item():+.4f}")
                    print(f"  ||∇ log|G(μ)|| mean={grad_norm.mean().item():.4f}, max={grad_norm.max().item():.4f}")
                    centroids = getattr(self._ctx['model'], 'centroids_tens', None)
                    if centroids is not None:
                        centroids = centroids.to(mu.device, mu.dtype)
                        dists = torch.cdist(mu.detach(), centroids)
                        min_dists, min_idx = dists.min(dim=1)
                        nearest = centroids[min_idx]
                        dir_to_centroid = torch.nn.functional.normalize(nearest - mu.detach(), dim=-1)
                        grad_dir = torch.nn.functional.normalize(grad_mu_dbg, dim=-1)
                        align = (dir_to_centroid * grad_dir).sum(dim=-1)
                        print(
                            f"  Nearest-centroid dist mean={min_dists.mean().item():.4f}, "
                            f"grad alignment mean={align.mean().item():.4f}"
                        )
                    else:
                        print("  Centroids unavailable; skipping alignment diagnostic.")
                except Exception as grad_exc:
                    print(f"[MU GRADIENT DEBUG] failed: {grad_exc}")
                self._mu_gradient_diag_count = grad_counter + 1

            B, D = mu.shape
            K = int(getattr(self, 'initial_n_candidates', 1))
            K = max(1, K)

            # Ensure chol has the correct shape [B, D, D]
            if chol.ndim == 2:
                chol = chol.unsqueeze(0)  # [1, D, D] -> [B, D, D]
            elif chol.ndim > 3:
                # Handle unexpected extra dimensions by reshaping
                total_elements = chol.numel()
                expected_elements = B * D * D
                if total_elements == expected_elements:
                    chol = chol.reshape(B, D, D)
                else:
                    last_two_dims = chol.shape[-2:]
                    if last_two_dims[0] == last_two_dims[1]:  # Square matrix
                        inferred_D = last_two_dims[0]
                        inferred_B = total_elements // (inferred_D * inferred_D)
                        if inferred_B == B:
                            chol = chol.reshape(B, inferred_D, inferred_D)
                            D = inferred_D
                        else:
                            chol = chol.reshape(-1, inferred_D, inferred_D)
                            if chol.shape[0] != B:
                                chol = chol[:B] if chol.shape[0] > B else chol.expand(B, -1, -1)
                            D = inferred_D
                    else:
                        raise RuntimeError(
                            f"Cannot reshape chol tensor of shape {chol.shape} to [B={B}, D={D}, D={D}]. "
                            f"Total elements: {total_elements}, expected: {B * D * D}"
                        )
            elif chol.shape[0] != B:
                # Handle batch size mismatch
                if chol.shape[0] == 1:
                    chol = chol.expand(B, -1, -1)
                else:
                    chol = chol[:B]
                
            eps = torch.randn(B, K, D, device=mu.device, dtype=chol.dtype)

            # Use batch matrix multiplication for efficiency
            # chol: [B, D, D], eps: [B, K, D]
            # We want: chol @ eps.transpose(-1, -2) -> [B, D, K]
            # Then transpose to get [B, K, D]
            transformed = torch.bmm(chol, eps.transpose(-1, -2)).transpose(-1, -2)  # [B, K, D]
            z_cand = mu.float().unsqueeze(1) + transformed
            try:
                Sigma_inv = torch.cholesky_inverse(chol.float())
            except RuntimeError:
                Sigma_inv = torch.pinverse(Sigma.float())
            diff = (z_cand - mu.unsqueeze(1)).float()
            mahal = torch.einsum('bkd,bij,bkj->bk', diff, Sigma_inv, diff)
            # Evaluate 0.5 log|G^{-1}| at candidates and pick best per batch
            with torch.no_grad():
                z_eval = z_cand.reshape(B * K, D)
                Gz = self._ctx['model'].G(z_eval)
                # Invert sign to favor lower metric volume (smaller log|G^{-1}|).
                h = -half_logdet_volume(Gz, 'g', jitter=self.eps_reg).reshape(B, K)
            best_idx = torch.argmax(h, dim=1)

            # DIAGNOSTIC: Analyze candidates before selection
            if diagnostics_enabled and os.environ.get("RLVAE_DEBUG", "0") == "1":
                batch_counter = getattr(self, '_debug_batch_counter', 0)
                self._diagnose_candidates(z_cand.to(mu.dtype), mu, Sigma, h, best_idx, batch_counter)
                self._debug_batch_counter = batch_counter + 1
            stagec_debugger.log_event(
                "sample_initial_once_standard_candidates",
                {
                    "B": B,
                    "K": K,
                    "chol_was_stabilized": bool(chol_was_stabilized),
                    "candidate_logdet_mean": float(h.mean().item()),
                    "candidate_mahal_mean": float(mahal.mean().item()),
                },
            )
            z0 = z_cand[torch.arange(B, device=mu.device), best_idx]
            z0 = z0.to(mu.dtype)
            # Optional radial cap on initial displacement
            try:
                max_norm = float(getattr(self, 'initial_max_norm', 0.0))
            except Exception:
                max_norm = 0.0
            if max_norm and max_norm > 0:
                delta = z0 - mu
                dnorm = torch.norm(delta, dim=-1, keepdim=True)
                scale = torch.clamp(max_norm / (dnorm + 1e-12), max=1.0)
                z0 = mu + scale * delta
            # Save z_selected (after multi-try, before volume) for stage tracking
            z_selected = z0.clone() if diagnostics_enabled and os.environ.get("RLVAE_DEBUG", "0") == "1" else None
            z0 = self._initial_accept_volume(z0, mu, Sigma)
            
            # DIAGNOSTIC: Comprehensive initial sampling analysis with stage comparison
            if diagnostics_enabled and os.environ.get("RLVAE_DEBUG", "0") == "1":
                self._diagnose_initial_sample(z_selected, z0, mu, Sigma, alpha)
                self._diagnose_metric_at_samples(z0, mu, z_selected)
            stagec_debugger.log_event(
                "sample_initial_once_standard_result",
                {
                    "mean_norm": float(torch.norm(z0 - mu, dim=-1).mean().item()),
                    "sigma_logdet": float(torch.linalg.slogdet(Sigma.float())[1].mean().item()),
                },
            )
            return z0, Sigma
        except Exception as exc:
            if getattr(self, '_ctx', None) and getattr(self._ctx.get('model', None), 'riemannian_strict', False) or os.environ.get('RLVAE_STRICT', '0') == '1':
                raise RuntimeError(f"RiemannianRHMCPosterior: initial sampling failed under strict mode: {exc}")
            print(f"⚠️ Riemannian sampling failed: {exc}, using Gaussian fallback")
            eps = torch.randn_like(mu)
            z0 = mu + eps * torch.exp(0.5 * log_var)
            Sigma_diag = torch.exp(log_var)
            Sigma = torch.diag_embed(Sigma_diag) + self.eps_reg * torch.eye(
                latent_dim, device=mu.device, dtype=mu.dtype
            ).unsqueeze(0)
            return z0, Sigma

    def _sample_initial_riemannian(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        alpha: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive initial draw with optional shrinkage toward high-volume regions
        and a light volume-gradient warm-up.
        """
        shrink_steps = max(0, int(self.initial_volume_max_adjustments))
        volume_tol = float(max(0.0, self.initial_volume_logdet_tol))
        shrink_factor = float(min(max(self.initial_volume_shrink_factor, 1e-3), 0.999))
        grow_factor = float(max(self.initial_volume_grow_factor, 1.001))
        min_alpha = float(max(self.initial_volume_min_alpha, 1e-6))
        max_alpha = float(max(self.initial_volume_max_alpha, min_alpha * 1.01))
        if volume_tol <= 0.0:
            shrink_steps = 0

        alpha_current = max(float(alpha), min_alpha)
        if self.initial_target_radius > 0:
            target_radius_current = float(self.initial_target_radius)
        else:
            target_radius_current = 0.0
        debug_mode = os.environ.get("RLVAE_DEBUG", "0") == "1"

        with torch.no_grad():
            G_inv_mu_dbg = _to_float32(self._get_inverse_metric(mu))
            logdet_mu = torch.linalg.slogdet(G_inv_mu_dbg)[1]
        logdet_mu_mean = logdet_mu.mean().item()
        stagec_debugger.log_event(
            "sample_initial_riemannian_start",
            {
                "alpha": float(alpha),
                "shrink_steps": shrink_steps,
                "volume_tol": volume_tol,
                "initial_target_radius": float(getattr(self, "initial_target_radius", 0.0)),
                "logdet_mu_mean": logdet_mu_mean,
            },
        )

        last_z = None
        last_sigma = None
        last_delta = None

        for attempt in range(shrink_steps + 1):
            diagnostics_enabled = (attempt == shrink_steps)
            z0, Sigma = self._sample_initial_once(
                mu,
                log_var,
                alpha_current,
                target_radius=target_radius_current,
                diagnostics_enabled=diagnostics_enabled,
            )
            if os.environ.get("RLVAE_DEBUG", "0") == "1":
                try:
                    logdet_sigma = torch.linalg.slogdet(Sigma.float())[1].mean().item()
                    print(f"[RHMC DEBUG] using _make_covariance for attempt {attempt} log|Σ|={logdet_sigma:+.4f}")
                except Exception:
                    print(f"[RHMC DEBUG] using _make_covariance for attempt {attempt}")
            with torch.no_grad():
                G_inv_z0_dbg = _to_float32(self._get_inverse_metric(z0))
                logdet_z0 = torch.linalg.slogdet(G_inv_z0_dbg)[1]
            logdet_z0_mean = logdet_z0.mean().item()
            delta = logdet_z0_mean - logdet_mu_mean
            if delta < -volume_tol:
                action_hint = "grow"
            elif delta > volume_tol:
                action_hint = "shrink"
            else:
                action_hint = "accept"
            stagec_debugger.log_event(
                "sample_initial_riemannian_attempt",
                {
                    "attempt": attempt,
                    "alpha": alpha_current,
                    "target_radius": target_radius_current,
                    "delta_logdet": delta,
                    "action": action_hint,
                },
            )

            if debug_mode:
                print(
                    f"[INITIAL SHRINK] attempt={attempt} "
                    f"alpha={alpha_current:.4g} radius={target_radius_current:.4g} "
                    f"logdet(z0)={logdet_z0_mean:+.4f} Δ={delta:+.4f}"
                )

            last_z, last_sigma, last_delta = z0, Sigma, delta

            if abs(delta) <= volume_tol:
                break

            if attempt >= shrink_steps:
                break

            if delta < -volume_tol:
                alpha_current = min(alpha_current * grow_factor, max_alpha)
                stagec_debugger.log_event(
                    "sample_initial_riemannian_adjust",
                    {"attempt": attempt, "action": "grow", "alpha": alpha_current, "delta": delta},
                )
                continue

            if delta > volume_tol:
                alpha_current = max(alpha_current * shrink_factor, min_alpha)
                if target_radius_current > 0:
                    target_radius_current = max(
                        target_radius_current * math.sqrt(shrink_factor),
                        1e-6,
                    )
                stagec_debugger.log_event(
                    "sample_initial_riemannian_adjust",
                    {"attempt": attempt, "action": "shrink", "alpha": alpha_current, "delta": delta},
                )
                continue

        if last_z is None or last_sigma is None:
            raise RuntimeError("Adaptive initial sampling failed to produce a sample.")

        if self.initial_volume_warmup_steps > 0 and self.initial_volume_warmup_step_size > 0:
            last_z = self._apply_volume_warmup(
                last_z,
                steps=self.initial_volume_warmup_steps,
                step_size=self.initial_volume_warmup_step_size,
                max_step_norm=self.initial_volume_warmup_max_step,
            )
            if debug_mode:
                with torch.no_grad():
                    warmed_logdet = torch.linalg.slogdet(_to_float32(self._get_inverse_metric(last_z)))[1]
                print(
                    f"[INITIAL WARMUP] steps={self.initial_volume_warmup_steps} "
                    f"logdet(z_warm)={warmed_logdet.mean().item():+.4f} "
                    f"Δ={warmed_logdet.mean().item() - logdet_mu_mean:+.4f}"
                )
            stagec_debugger.log_event(
                "sample_initial_riemannian_warmup",
                {
                    "steps": self.initial_volume_warmup_steps,
                    "step_size": self.initial_volume_warmup_step_size,
                    "max_step_norm": self.initial_volume_warmup_max_step,
                },
            )

        stagec_debugger.log_event(
            "sample_initial_riemannian_result",
              {
                  "final_mean_norm": float(torch.norm(last_z - mu, dim=-1).mean().item()),
                  "final_sigma_logdet": float(torch.linalg.slogdet(last_sigma.float())[1].mean().item()),
              },
        )
        return last_z, last_sigma

    def _apply_volume_warmup(
        self,
        z: torch.Tensor,
        *,
        steps: int,
        step_size: float,
        max_step_norm: float,
    ) -> torch.Tensor:
        """Small gradient-ascent warm-up on log|G⁻¹(z)| to nudge samples toward high-volume regions."""
        if steps <= 0 or step_size <= 0:
            return z
        stagec_debugger.log_event(
            "volume_warmup_start",
            {
                "steps": steps,
                "step_size": step_size,
                "max_step_norm": max_step_norm,
            },
        )
        debug_mode = os.environ.get("RLVAE_DEBUG", "0") == "1"
        z_cur = z
        for idx in range(int(steps)):
            z_cur = z_cur.detach().requires_grad_(True)
            G_z = self._ctx['model'].G(z_cur)
            G_z = torch.linalg.inv(_symmetrize(G_z))
            print('okokok')
            # For rep='g', half_logdet is -½ log|G| = +½ log|G^{-1}|. Ascend this directly.
            half_logdet = half_logdet_volume(G_z.float(), 'g', jitter=self.eps_reg)
            logdet = half_logdet
            grad = torch.autograd.grad(
                logdet.sum(),
                z_cur,
                create_graph=True,
                retain_graph=False,
                allow_unused=False,
            )[0]
            if max_step_norm > 0:
                grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                scale = torch.clamp(max_step_norm / grad_norm, max=1.0)
                grad = grad * scale
            z_next = z_cur + step_size * grad
            z_cur = z_next
            if debug_mode:
                with torch.no_grad():
                    current_logdet = torch.linalg.slogdet(_to_float32(self._get_inverse_metric(z_cur)))[1]
                print(
                    f"[VOLUME WARMUP] step={idx} "
                    f"logdet={current_logdet.mean().item():+.4f}"
                )
        return z_cur
    def _make_covariance(
        self,
        G_inv: torch.Tensor,
        alpha: float,
        *,
        target_radius: Optional[float] = None,
    ) -> torch.Tensor:
        # Ensure symmetry
        Ginv = _symmetrize(G_inv)
        d = Ginv.shape[-1]
        eye = torch.eye(d, device=Ginv.device, dtype=Ginv.dtype).unsqueeze(0)

        # --- FIX: Avoid eigh() on isotropic matrices (NaN gradient risk) ---
        # Instead of clamping eigenvalues explicitly, we add jitter to diagonal.
        # This is numerically safer and avoids the degenerate eigenvalue problem.
        
        # 1. Enforce floor via addition (Soft Clamp)
        # This guarantees evals >= min_cov_eig without decomposition
        # (Assuming G_inv is already PSD from the metric definition)
        floor_jitter = max(self.min_cov_eig, 1e-6)
        Ginv_safe = Ginv + floor_jitter * eye

        # 2. Optional: Cap huge values (Hard Clamp on diagonal only)
        # This is approximate but gradient-safe
        if math.isfinite(self.metric_eig_ceiling):
            Ginv_safe = torch.clamp(Ginv_safe, max=self.metric_eig_ceiling)

        # 3. Normalization (Volume preservation) using Cholesky (Safe)
        mode = str(getattr(self, 'sigma_normalization_mode', 'none')).lower()
        
        if mode == 'geomean':
            # Use logdet (via Cholesky) instead of product of eigenvalues
            try:
                chol_norm, _ = _safe_cholesky(Ginv_safe, 1e-6)
                diag_chol = torch.diagonal(chol_norm, dim1=-2, dim2=-1)
                logdet = 2.0 * torch.log(diag_chol.clamp(min=1e-12)).sum(-1, keepdim=True)
                gm = torch.exp(logdet / d).unsqueeze(-1)
                Ginv_norm = Ginv_safe / (gm + 1e-12)
            except Exception:
                Ginv_norm = Ginv_safe
                
        elif mode == 'trace':
            tr = torch.einsum('bii->b', Ginv_safe).unsqueeze(-1).unsqueeze(-1)
            Ginv_norm = d * Ginv_safe / tr.clamp(min=1e-12)
        else:
            Ginv_norm = Ginv_safe

        # 4. Scale by alpha (Logic: Variance ~ Alpha * Precision)
        # Note: We use Precision (Ginv) directly as Variance basis per your configuration
        Sigma = alpha * Ginv_norm + self.eps_reg * eye

        # 5. Clamp covariance eigenvalues to control spread in flat regions
        try:
            eigvals, eigvecs = torch.linalg.eigh(_to_float32(Sigma))
            eig_floor = float(max(self.min_cov_eig, 0.0))
            eig_cap = float(self.metric_eig_ceiling) if math.isfinite(self.metric_eig_ceiling) else None
            eigvals_clamped = torch.clamp(eigvals, min=eig_floor)
            if eig_cap is not None:
                eigvals_clamped = torch.clamp(eigvals_clamped, max=eig_cap)
            Sigma = (eigvecs @ torch.diag_embed(eigvals_clamped) @ eigvecs.transpose(-1, -2)).to(Sigma.dtype)
        except Exception:
            # Fallback: leave Sigma as-is if eigendecomposition fails
            pass

        return Sigma

    def _resolve_alpha(self) -> float:
        """
        Resolve the current α scaling (supports epoch-dependent schedule).
        """
        model = self._ctx['model']
        direct_alpha = getattr(model, 'rhmc_alpha', None)
        if direct_alpha is not None:
            try:
                alpha = float(direct_alpha)
                if math.isfinite(alpha) and alpha > 0:
                    # TRACE: log source of alpha
                    try:
                        if os.environ.get('RLVAE_TRACE', '0') == '1' and not self._trace_printed:
                            print(f"TRACE OVERRIDE α source=model.rhmc_alpha -> {alpha}")
                    except Exception:
                        pass
                    return alpha
                # Treat zero or negative alpha as a signal to fall back downstream
            except Exception:
                pass
        current_epoch = getattr(model, '_current_epoch', None)
        if hasattr(model, 'get_current_posterior_alpha'):
            try:
                alpha = float(model.get_current_posterior_alpha(current_epoch))
                if math.isfinite(alpha) and alpha > 0:
                    try:
                        if os.environ.get('RLVAE_TRACE', '0') == '1' and not self._trace_printed:
                            print(f"TRACE OVERRIDE α source=get_current_posterior_alpha -> {alpha}")
                    except Exception:
                        pass
                    return alpha
                # Guard against ramps returning 0 when disabled
            except Exception:
                pass

        cfg_alpha = None
        try:
            cfg_alpha = getattr(model.config, 'rhmc_alpha', None)
        except Exception:
            cfg_alpha = None
        if cfg_alpha is None and hasattr(model, 'config') and hasattr(model.config, 'model'):
            cfg_alpha = getattr(model.config.model, 'rhmc_alpha', None)
        if cfg_alpha is not None:
            try:
                alpha = float(cfg_alpha)
                if math.isfinite(alpha) and alpha > 0:
                    try:
                        if os.environ.get('RLVAE_TRACE', '0') == '1' and not self._trace_printed:
                            print(f"TRACE OVERRIDE α source=config.rhmc_alpha -> {alpha}")
                    except Exception:
                        pass
                    return alpha
            except Exception:
                pass
        fallbacks = [
            getattr(model, 'posterior_local_alpha', None),
            getattr(self, 'rhmc_alpha', None),
            getattr(model, 'rhmc_alpha_default', None),
        ]
        for cand in fallbacks:
            if cand is None:
                continue
            try:
                alpha = float(cand)
                if math.isfinite(alpha) and alpha > 0:
                    return alpha
            except Exception:
                continue
        # Final safety: small positive epsilon
        return 1e-3

    def _initial_accept_volume(
        self,
        z0: torch.Tensor,
        mu: torch.Tensor,
        Sigma: Optional[torch.Tensor],
    ) -> torch.Tensor:
        try:
            tol = float(getattr(self, 'initial_volume_tolerance', 0.0))
            max_refine = int(getattr(self, 'initial_max_retries', 0))
            if (tol <= 0 and max_refine <= 0) or Sigma is None:
                return z0

            with torch.no_grad():
                G_mu = self._ctx['model'].G(mu)
                target = half_logdet_volume(G_mu, 'g', jitter=self.eps_reg) - tol
                try:
                    chol, _, _ = _safe_cholesky(Sigma, self.min_cov_eig)
                except Exception:
                    chol = None

                z = z0.clone()
                total_attempts = max(0, max_refine) + 1
                for attempt in range(total_attempts):
                    Gz = self._ctx['model'].G(z)
                    hz = half_logdet_volume(Gz, 'g', jitter=self.eps_reg)
                    accept_mask = hz >= target
                    if accept_mask.all():
                        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

                    if attempt == total_attempts - 1 or chol is None:
                        # Out of retries or no covariance available; return current samples
                        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

                    deficit = (~accept_mask).nonzero(as_tuple=True)[0]
                    if deficit.numel() == 0:
                        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

                    eps = torch.randn(
                        deficit.numel(),
                        z.shape[-1],
                        device=z.device,
                        dtype=z.dtype,
                    )
                    chol_subset = chol[deficit]
                    step = torch.einsum('bij,bj->bi', chol_subset, eps)
                    z = z.clone()
                    z[deficit] = mu[deficit] + step

                return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            return z0

    def _get_inverse_metric(self, pts: torch.Tensor) -> torch.Tensor:
        """Fetch Ĝ^{-1}(pts) with symmetry and fallback safeguards."""
        model = self._ctx['model']
        if hasattr(model, 'G_inv'):
            G_inv = model.G_inv(pts)
        elif hasattr(model, 'G'):
            G = model.G(pts)
            G_inv = torch.linalg.inv(_symmetrize(G))
        else:
            raise AttributeError("Model must expose G_inv or G to compute RHMC posterior.")
        G_inv = _symmetrize(G_inv)
        G_inv = torch.nan_to_num(G_inv, nan=float('inf'), posinf=float('inf'), neginf=float('inf'))
        if not torch.isfinite(G_inv).all():
            if not hasattr(self, "_warned_metric_nan"):
                print("[RHMC WARN] Inverse metric returned non-finite values; falling back to identity.")
                self._warned_metric_nan = True
            d = pts.shape[-1]
            eye = torch.eye(d, device=pts.device, dtype=pts.dtype).unsqueeze(0)
            return eye.expand(pts.shape[0], -1, -1)
        try:
            m32 = G_inv.float() if G_inv.dtype in (torch.float16, torch.bfloat16) else G_inv
            evals, evecs = torch.linalg.eigh(m32)
            floor = max(self.eps_reg, 1e-6)
            evals = torch.clamp(evals, min=floor)
            ceil = float(getattr(self, "metric_eig_ceiling", float('inf')))
            if math.isfinite(ceil) and ceil > 0:
                evals = torch.clamp(evals, max=ceil)
            G_inv = (evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)).to(G_inv.dtype)
        except Exception:
            pass
        return _symmetrize(torch.nan_to_num(G_inv, nan=0.0, posinf=0.0, neginf=0.0))

    def _stabilize_spd(self, matrix: torch.Tensor, min_eig: float) -> torch.Tensor:
        """Clamp eigenvalues from below to guarantee SPD-ness."""
        try:
            m32 = matrix.float() if matrix.dtype in (torch.float16, torch.bfloat16) else matrix
            eigenvalues, eigenvectors = torch.linalg.eigh(m32)
            eigenvalues = torch.clamp(eigenvalues, min=min_eig)
            Sigma = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)
            Sigma = Sigma.to(matrix.dtype)
            return _symmetrize(Sigma)
        except RuntimeError:
            # Fallback: add jitter and hope for the best
            d = matrix.shape[-1]
            eye = torch.eye(d, device=matrix.device, dtype=matrix.dtype).unsqueeze(0)
            return _symmetrize(matrix + min_eig * eye)

    def _cap_covariance_eigs(self, Sigma: torch.Tensor) -> torch.Tensor:
        """Enforce an upper bound on covariance eigenvalues to avoid runaway spreads."""
        if not math.isfinite(self.max_cov_eig):
            return Sigma
        try:
            Sigma32 = Sigma.float() if Sigma.dtype in (torch.float16, torch.bfloat16) else Sigma
            evals, evecs = torch.linalg.eigh(_symmetrize(Sigma32))
            capped = evals.clamp(min=self.min_cov_eig, max=self.max_cov_eig)
            if os.environ.get("RLVAE_DEBUG", "0") == "1" and (capped != evals).any():
                diff = torch.abs(capped - evals).max().item()
                print(f"[COV CLAMP] Applied max_cov_eig={self.max_cov_eig:.4f} (max adjustment={diff:.4e})")
            Sigma_capped = evecs @ torch.diag_embed(capped) @ evecs.transpose(-1, -2)
            return _symmetrize(Sigma_capped.to(Sigma.dtype))
        except Exception as exc:
            if os.environ.get("RLVAE_DEBUG", "0") == "1":
                print(f"[COV CLAMP] Failed to cap eigenvalues: {exc}")
            return Sigma

    def _diagnose_initial_sample(
        self, 
        z_selected: torch.Tensor, 
        z0: torch.Tensor, 
        mu: torch.Tensor, 
        Sigma: torch.Tensor,
        alpha: float
    ) -> None:
        """
        Comprehensive diagnostics for initial sampling with stage-by-stage comparison.
        Shows Σ_μ properties, selection effects, volume acceptance, and Mahalanobis decomposition.
        
        Args:
            z_selected: Sample after multi-try selection, before volume acceptance
            z0: Final sample after volume acceptance (actual RHMC starting point)
            mu: Encoder mean
            Sigma: Covariance matrix Σ_μ = α·G⁻¹(μ) + ε·I
            alpha: Scaling parameter
        """
        import numpy as np
        
        print(f"\n{'='*80}")
        print(f"[INITIAL SAMPLING DIAGNOSTICS]")
        print(f"{'='*80}")
        
        Sigma_dbg = _to_float32(Sigma)
        # 1. Σ_μ Properties
        try:
            eigvals_sigma = torch.linalg.eigvalsh(Sigma_dbg)
            logdet_sigma = torch.linalg.slogdet(Sigma_dbg)[1]
            trace_sigma = torch.diagonal(Sigma, dim1=-2, dim2=-1).sum(dim=-1)
            cond_sigma = eigvals_sigma.max(dim=-1)[0] / eigvals_sigma.min(dim=-1)[0].clamp(min=1e-12)
            
            print(f"\n[Σ_μ PROPERTIES]")
            print(f"  alpha:                 {alpha:.6f}")
            print(f"  Σ eigenvalues:         min={eigvals_sigma.min().item():.6e}, max={eigvals_sigma.max().item():.6f}")
            print(f"  Σ trace:               {trace_sigma.mean().item():.6f}")
            print(f"  log|Σ|:                {logdet_sigma.mean().item():.6f}")
            print(f"  Condition number:      {cond_sigma.mean().item():.2f}")
            print(f"  Anisotropy ratio:      {(eigvals_sigma.max(dim=-1)[0] / eigvals_sigma.min(dim=-1)[0]).mean().item():.2f}")
        except Exception as e:
            print(f"  [Error computing Σ properties: {e}]")
        
        # 2. Distance Analysis: z_selected → z0 (stage-by-stage)
        if z_selected is not None:
            dist_selected_mu = torch.norm(z_selected - mu, dim=-1)
            dist_z0_mu = torch.norm(z0 - mu, dim=-1)
            dist_selected_z0 = torch.norm(z_selected - z0, dim=-1)
            
            print(f"\n[STAGE-BY-STAGE DISTANCE ANALYSIS]")
            print(f"  ||z_selected - μ||:    mean={dist_selected_mu.mean().item():.4f}, std={dist_selected_mu.std().item():.4f}")
            print(f"  ||z0 - μ||:            mean={dist_z0_mu.mean().item():.4f}, std={dist_z0_mu.std().item():.4f}")
            print(f"  ||z_selected - z0||:   mean={dist_selected_z0.mean().item():.4f}, std={dist_selected_z0.std().item():.4f}")
            print(f"  Δ||·-μ|| (volume acc): {(dist_z0_mu.mean() - dist_selected_mu.mean()).item():+.4f}")
            
            # Mahalanobis distances for both stages
            try:
                Sigma_inv = torch.linalg.inv(Sigma_dbg)
                diff_selected = z_selected - mu
                diff_z0 = z0 - mu
                mahal_sq_selected = torch.einsum('bi,bij,bj->b', diff_selected.float(), Sigma_inv, diff_selected.float())
                mahal_sq_z0 = torch.einsum('bi,bij,bj->b', diff_z0.float(), Sigma_inv, diff_z0.float())
                
                print(f"\n[SELECTION STAGE COMPARISON]")
                print(f"  z_selected Mahal²:     mean={mahal_sq_selected.mean().item():.4f}")
                print(f"  z0 Mahal²:             mean={mahal_sq_z0.mean().item():.4f}")
                print(f"  Δ Mahal² (vol acc):    {(mahal_sq_z0.mean() - mahal_sq_selected.mean()).item():+.4f}")
                
                if dist_z0_mu.mean() < dist_selected_mu.mean():
                    print(f"  → Volume acceptance MOVED CLOSER to μ")
                elif dist_z0_mu.mean() > dist_selected_mu.mean():
                    print(f"  → Volume acceptance MOVED AWAY from μ")
                else:
                    print(f"  → Volume acceptance had NO EFFECT on distance")
            except Exception as e:
                print(f"  [Error computing stage Mahalanobis: {e}]")
        else:
            dist_z0_mu = torch.norm(z0 - mu, dim=-1)
            print(f"\n[DISTANCE ANALYSIS]")
            print(f"  ||z0 - μ||:            mean={dist_z0_mu.mean().item():.4f}, std={dist_z0_mu.std().item():.4f}")
        
        # 3. Expected vs Actual Distance
        try:
            # For N(μ, Σ), E[||z-μ||] ≈ √(tr(Σ))
            expected_dist = torch.sqrt(trace_sigma)
            print(f"\n[EXPECTED VS ACTUAL]")
            print(f"  Expected ||z-μ||:      √(tr(Σ)) = {expected_dist.mean().item():.4f}")
            print(f"  Actual ||z0-μ||:       {dist_z0_mu.mean().item():.4f}")
            print(f"  Ratio (actual/expected): {(dist_z0_mu.mean() / expected_dist.mean()).item():.4f}")
            
            if dist_z0_mu.mean() > 1.5 * expected_dist.mean():
                print(f"  ⚠️  WARNING: z0 is FAR from μ (>1.5× expected)")
            elif dist_z0_mu.mean() < 0.5 * expected_dist.mean():
                print(f"  ⚠️  WARNING: z0 is TOO CLOSE to μ (<0.5× expected)")
        except Exception as e:
            print(f"  [Error computing expected distance: {e}]")
        
        # 4. Mahalanobis Distance & Decomposition
        try:
            diff = z0 - mu
            Sigma_inv = torch.linalg.inv(Sigma_dbg)
            mahal_sq = torch.einsum('bi,bij,bj->b', diff.float(), Sigma_inv, diff.float())
            mahal_dist = torch.sqrt(mahal_sq)
            
            # Compare Euclidean vs Mahalanobis
            euclidean_sq = torch.norm(diff, dim=-1) ** 2
            
            print(f"\n[MAHALANOBIS ANALYSIS]")
            print(f"  Euclidean dist²:       mean={euclidean_sq.mean().item():.4f}")
            print(f"  Mahalanobis dist²:     mean={mahal_sq.mean().item():.4f}")
            print(f"  Mahalanobis dist:      mean={mahal_dist.mean().item():.4f}")
            print(f"  Ratio (Mahal²/Euclid²): {(mahal_sq.mean() / euclidean_sq.mean().clamp(min=1e-12)).item():.4f}")
            
            # Decompose in eigenbasis of Σ
            eigvals, eigvecs = torch.linalg.eigh(Sigma_dbg)
            # Transform diff to eigenspace: y = V^T (z-μ)
            y = torch.einsum('bij,bj->bi', eigvecs.transpose(-1, -2), diff)
            # Contribution per eigenvalue: y_i² / λ_i
            contrib_per_eig = (y ** 2) / eigvals.clamp(min=1e-12)
            
            print(f"\n[MAHALANOBIS DECOMPOSITION in Eigenbasis]")
            D = eigvals.shape[-1]
            for i in range(D):
                print(f"  Eigenvalue {i}: λ={eigvals[0, i].item():.4f}, "
                      f"y²={y[0, i].item()**2:.4f}, "
                      f"contrib=(y²/λ)={contrib_per_eig[0, i].item():.4f}")
            
            # Check if Mahalanobis² ~ χ²(D)
            D = diff.shape[-1]
            print(f"\n[CHI-SQUARED TEST]")
            print(f"  Dimension D:           {D}")
            print(f"  Expected Mahal²:       χ²({D}) mean = {D:.1f}")
            print(f"  Actual Mahal²:         {mahal_sq.mean().item():.4f}")
            print(f"  Deviation:             {(mahal_sq.mean().item() - D):.4f} ({100*(mahal_sq.mean().item() - D)/D:.1f}%)")
            
            if abs(mahal_sq.mean().item() - D) > 2 * np.sqrt(2 * D):
                print(f"  ⚠️  WARNING: Mahalanobis² significantly deviates from χ²({D})")
                print(f"               This suggests mismatch between sampling and Σ_μ")
            
            # Enhanced: Histogram comparison and KS test
            try:
                from scipy import stats
                mahal_sq_np = mahal_sq.cpu().numpy()
                
                # KS test against χ²(D) distribution
                ks_stat, ks_pval = stats.kstest(mahal_sq_np, lambda x: stats.chi2.cdf(x, D))
                print(f"\n[KOLMOGOROV-SMIRNOV TEST]")
                print(f"  KS statistic:          {ks_stat:.4f}")
                print(f"  KS p-value:            {ks_pval:.4e}")
                if ks_pval < 0.01:
                    print(f"  ⚠️  REJECT null hypothesis: distribution does NOT match χ²({D})")
                else:
                    print(f"  ✓ Cannot reject: distribution consistent with χ²({D})")
                
                # Histogram bins: compare empirical density to theoretical
                if len(mahal_sq_np) >= 10:
                    hist_bins = np.linspace(0, max(mahal_sq_np.max(), 3*D), 10)
                    hist_counts, _ = np.histogram(mahal_sq_np, bins=hist_bins, density=True)
                    hist_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
                    theoretical_density = stats.chi2.pdf(hist_centers, D)
                    
                    print(f"\n[HISTOGRAM COMPARISON] (first 5 bins)")
                    print(f"  {'Bin Center':>12} {'Empirical':>12} {'Theoretical':>12} {'Ratio':>10}")
                    for i in range(min(5, len(hist_centers))):
                        ratio = hist_counts[i] / (theoretical_density[i] + 1e-12)
                        print(f"  {hist_centers[i]:12.2f} {hist_counts[i]:12.4f} {theoretical_density[i]:12.4f} {ratio:10.2f}")
            except ImportError:
                print(f"  [scipy not available for KS test]")
            except Exception as e:
                print(f"  [Error in enhanced χ² test: {e}]")
        except Exception as e:
            print(f"  [Error computing Mahalanobis: {e}]")
        
        # 5. Statistical Validation: Empirical vs Theoretical
        try:
            # For a batch, compute empirical covariance of (z-μ)
            if z0.shape[0] > 1:
                diff_centered = z0 - mu
                emp_cov = torch.einsum('bi,bj->ij', diff_centered, diff_centered) / z0.shape[0]
                
                # Compare with Σ
                sigma_mean = Sigma.mean(dim=0)
                frobenius_diff = torch.norm(emp_cov - sigma_mean, p='fro')
                frobenius_sigma = torch.norm(sigma_mean, p='fro')
                
                print(f"\n[EMPIRICAL VS THEORETICAL COVARIANCE]")
                print(f"  Batch size:            {z0.shape[0]}")
                print(f"  ||Empirical - Σ||_F:   {frobenius_diff.item():.4f}")
                print(f"  ||Σ||_F:               {frobenius_sigma.item():.4f}")
                print(f"  Relative error:        {(frobenius_diff / frobenius_sigma.clamp(min=1e-12)).item():.4f}")
                
                if z0.shape[0] < 10:
                    print(f"  Note: Batch too small ({z0.shape[0]}) for reliable empirical covariance")
        except Exception as e:
            print(f"  [Error computing empirical covariance: {e}]")
        
        print(f"\n{'='*80}\n")

    def _diagnose_candidates(
        self,
        z_cand: torch.Tensor,  # [B, K, D]
        mu: torch.Tensor,      # [B, D]
        Sigma: torch.Tensor,   # [B, D, D] or [1, D, D]
        h_scores: torch.Tensor,  # [B, K]
        selected_idx: torch.Tensor,  # [B]
        batch_counter: int = 0,
    ) -> None:
        """
        Diagnose raw candidates vs selection prior to volume acceptance.

        Logs summary stats across candidates and highlights selection bias by
        comparing Mahalanobis distances and correlation with h_scores.
        """
        import os
        if os.environ.get("RLVAE_DEBUG", "0") != "1":
            return

        try:
            B, K, D = z_cand.shape
            # Ensure Sigma broadcast to [B, D, D]
            if Sigma.dim() == 3 and Sigma.shape[0] == 1 and B > 1:
                Sigma_b = Sigma.expand(B, D, D)
            else:
                Sigma_b = Sigma

            diff = z_cand - mu.unsqueeze(1)  # [B, K, D]

            # Euclidean norms
            euclid = torch.norm(diff, dim=-1)  # [B, K]

            # Mahalanobis using explicit inverse (D is small)
            Sigma_inv = torch.linalg.inv(Sigma_b.float())  # [B, D, D]
            mahal_sq = torch.einsum('bkd,bij,bkj->bk', diff.float(), Sigma_inv, diff.float())  # [B, K]

            # Selected candidate stats
            gather_idx = selected_idx.view(B, 1)
            sel_euclid = torch.gather(euclid, 1, gather_idx).squeeze(1)
            sel_mahal_sq = torch.gather(mahal_sq, 1, gather_idx).squeeze(1)
            sel_h = torch.gather(h_scores, 1, gather_idx).squeeze(1)

            # Correlation h_scores vs mahal_sq across all candidates
            h_flat = h_scores.reshape(-1).float()
            m_flat = mahal_sq.reshape(-1).float()
            h_c = h_flat - h_flat.mean()
            m_c = m_flat - m_flat.mean()
            corr = (h_c @ m_c) / (torch.norm(h_c) * torch.norm(m_c) + 1e-12)

            print(f"\n{'='*80}")
            print(f"[CANDIDATE DIAGNOSTICS] B={B}, K={K}, D={D}")
            print(f"{'='*80}")

            # Summary over all B×K
            print(f"\n[SUMMARY OVER ALL CANDIDATES]")
            print(f"  ||z-μ||:              mean={euclid.mean().item():.4f}, std={euclid.std().item():.4f}, min={euclid.min().item():.4f}, max={euclid.max().item():.4f}")
            print(f"  Mahal²:               mean={mahal_sq.mean().item():.4f}, std={mahal_sq.std().item():.4f}, min={mahal_sq.min().item():.4f}, max={mahal_sq.max().item():.4f}")
            print(f"  h=0.5·log|G⁻¹|:      mean={h_scores.mean().item():.4f}, std={h_scores.std().item():.4f}, min={h_scores.min().item():.4f}, max={h_scores.max().item():.4f}")
            print(f"  Corr(h, Mahal²):      {corr.item():+.4f}")

            # Chi-squared expectation
            print(f"\n[CHI-SQUARED EXPECTATION]")
            print(f"  D:                    {D}")
            print(f"  Expected Mahal² mean: {float(D):.4f}")
            print(f"  Observed Mahal² mean: {mahal_sq.mean().item():.4f}  (Δ={(mahal_sq.mean()-D).item():+.4f})")

            # Selected vs pool comparison
            print(f"\n[SELECTION VS POOL]")
            print(f"  Selected h:           mean={sel_h.mean().item():.4f}")
            print(f"  Selected ||z-μ||:     mean={sel_euclid.mean().item():.4f}")
            print(f"  Selected Mahal²:      mean={sel_mahal_sq.mean().item():.4f}")
            print(f"  Δ Mahal²(sel - pool): {(sel_mahal_sq.mean() - mahal_sq.mean()).item():+.4f}")

            # Detailed dump for first few batches
            if batch_counter < 3:
                max_print_b = min(B, 2)
                for b in range(max_print_b):
                    print(f"\n  [BATCH {batch_counter} SAMPLE {b}] selected k={int(selected_idx[b].item())}")
                    for k in range(K):
                        mark = '*' if k == int(selected_idx[b].item()) else ' '
                        print(
                            f"   {mark} k={k:02d}  h={h_scores[b,k].item():+.4f}  ||·||={euclid[b,k].item():.4f}  Mahal²={mahal_sq[b,k].item():.4f}"
                        )

            print(f"\n{'='*80}\n")
        except Exception as e:
            print(f"[CANDIDATE DIAGNOSTICS ERROR] {e}")

    def _diagnose_metric_at_samples(
        self,
        z0: torch.Tensor,
        mu: torch.Tensor,
        z_base: torch.Tensor = None
    ) -> None:
        """
        Diagnose metric (det G^{-1}) values at sample points.
        
        Check if samples are biased toward high or low volume regions,
        and if distance from μ correlates with metric values.
        
        Args:
            z0: Final samples (after multi-try/volume acceptance)
            mu: Encoder mean
            z_base: Initial samples before acceptance (optional)
        """
        import os
        if os.environ.get("RLVAE_DEBUG", "0") != "1":
            return
            
        print(f"\n{'='*80}")
        print(f"[METRIC AT SAMPLES DIAGNOSTICS]")
        print(f"{'='*80}")
        
        try:
            # Compute G^{-1} at μ and at z0
            model = self._ctx['model']
            if hasattr(model, 'metric_tensor'):
                G_inv_mu = model.metric_tensor.compute_metric_inv(mu)
                G_inv_z0 = model.metric_tensor.compute_metric_inv(z0)
            elif hasattr(model, 'G_inv'):
                G_inv_mu = model.G_inv(mu)
                G_inv_z0 = model.G_inv(z0)
            else:
                print(f"  [Cannot compute G⁻¹: model has no metric_tensor or G_inv attribute]")
                return
            
            # Compute log determinants
            _, logdet_mu = torch.linalg.slogdet(_to_float32(G_inv_mu))
            _, logdet_z0 = torch.linalg.slogdet(_to_float32(G_inv_z0))
            
            print(f"\n[LOG DET G⁻¹ DISTRIBUTION]")
            print(f"  At μ:                  mean={logdet_mu.mean().item():.4f}")
            print(f"  At z0:                 mean={logdet_z0.mean().item():.4f}, std={logdet_z0.std().item():.4f}")
            print(f"  min(z0):               {logdet_z0.min().item():.4f}")
            print(f"  max(z0):               {logdet_z0.max().item():.4f}")
            print(f"  Δ(z0 - μ):             {(logdet_z0.mean() - logdet_mu.mean()).item():.4f}")
            
            if logdet_z0.mean() < logdet_mu.mean():
                print(f"  ⚠️  WARNING: Samples have LOWER log|G⁻¹| than μ (moving to low-volume regions!)")
            else:
                print(f"  ✓ Samples have higher log|G⁻¹| than μ (good for prior)")
            
            # Check correlation: ||z-μ|| vs log|G⁻¹(z)|
            dist_z0_mu = torch.norm(z0 - mu, dim=-1)
            if z0.shape[0] > 1:
                # Pearson correlation
                dist_mean = dist_z0_mu.mean()
                logdet_mean = logdet_z0.mean()
                dist_centered = dist_z0_mu - dist_mean
                logdet_centered = logdet_z0 - logdet_mean
                corr_num = (dist_centered * logdet_centered).sum()
                corr_denom = torch.sqrt((dist_centered**2).sum() * (logdet_centered**2).sum())
                correlation = (corr_num / corr_denom.clamp(min=1e-12)).item()
                
                print(f"\n[CORRELATION: ||z-μ|| vs log|G⁻¹(z)|]")
                print(f"  Pearson r:             {correlation:.4f}")
                if abs(correlation) > 0.5:
                    direction = "positive" if correlation > 0 else "negative"
                    print(f"  Strong {direction} correlation: Farther samples {'have higher' if correlation > 0 else 'have lower'} log|G⁻¹|")
                
                # CRITICAL: Check for inversion behavior
                if correlation < -0.3:
                    print(f"  ⚠️  INVERSION DETECTED: Strong negative correlation!")
                    print(f"      This suggests samples move AWAY from high-volume regions as expected.")
                    print(f"      This contradicts the uniform prior objective p(z) ∝ √det(G⁻¹(z)).")
                    print(f"      Possible causes:")
                    print(f"        1. Sign error in RHMC gradient calculation")
                    print(f"        2. Sign error in KL prior term calculation") 
                    print(f"        3. Numerical instability with large α")
                    print(f"        4. Interaction with reconstruction loss")
            
            # If z_base is provided, compare before/after acceptance
            if z_base is not None:
                if hasattr(model, 'metric_tensor'):
                    G_inv_base = model.metric_tensor.compute_metric_inv(z_base)
                elif hasattr(model, 'G_inv'):
                    G_inv_base = model.G_inv(z_base)
                else:
                    G_inv_base = None
                
                if G_inv_base is not None:
                    _, logdet_base = torch.linalg.slogdet(_to_float32(G_inv_base))
                
                print(f"\n[VOLUME ACCEPTANCE EFFECT]")
                print(f"  log|G⁻¹(z_base)|:      mean={logdet_base.mean().item():.4f}")
                print(f"  log|G⁻¹(z0)|:          mean={logdet_z0.mean().item():.4f}")
                print(f"  Δ(z0 - z_base):        {(logdet_z0.mean() - logdet_base.mean()).item():.4f}")
                
                if logdet_z0.mean() > logdet_base.mean():
                    print(f"  ✓ Volume acceptance increased log|G⁻¹| (moved to higher-volume regions)")
                else:
                    print(f"  Volume acceptance had no effect or decreased log|G⁻¹|")
                    
        except Exception as e:
            print(f"  [Error computing metric diagnostics: {e}]")
            import traceback
            traceback.print_exc()
        
        print(f"\n{'='*80}\n")

    def _diagnose_posterior_mismatch(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        Sigma: torch.Tensor
    ) -> dict:
        """
        Statistical tests for posterior mismatch.
        Returns dictionary with test statistics and Q-Q plot data.
        
        Args:
            z: Samples [B, D]
            mu: Mean [B, D] or [1, D]
            Sigma: Covariance [B, D, D] or [1, D, D]
        
        Returns:
            dict with keys: 'chi_sq_stat', 'qq_theoretical', 'qq_empirical', 'cov_mismatch'
        """
        results = {}
        
        try:
            Sigma_dbg = _to_float32(Sigma)
            diff = z - mu
            Sigma_inv = torch.linalg.inv(Sigma_dbg)
            
            # Chi-squared test: (z-μ)ᵀ Σ⁻¹ (z-μ) ~ χ²(D)
            mahal_sq = torch.einsum('bi,bij,bj->b', diff, Sigma_inv, diff)
            D = z.shape[-1]
            
            results['chi_sq_stat'] = {
                'observed': mahal_sq.mean().item(),
                'expected': float(D),
                'deviation_sigmas': abs(mahal_sq.mean().item() - D) / np.sqrt(2 * D)
            }
            
            # Q-Q plot data
            if z.shape[0] >= 5:
                sorted_mahal, _ = torch.sort(mahal_sq)
                n = sorted_mahal.shape[0]
                # Theoretical quantiles from χ²(D)
                from scipy import stats
                theoretical_quantiles = [stats.chi2.ppf((i + 0.5) / n, D) for i in range(n)]
                
                results['qq_data'] = {
                    'theoretical': theoretical_quantiles,
                    'empirical': sorted_mahal.cpu().numpy().tolist()
                }
            
            # Empirical covariance mismatch
            if z.shape[0] > 1:
                emp_cov = torch.cov(diff.T)
                sigma_mean = Sigma_dbg.mean(dim=0) if Sigma_dbg.shape[0] > 1 else Sigma_dbg[0]
                frobenius_error = torch.norm(emp_cov - sigma_mean, p='fro') / torch.norm(sigma_mean, p='fro').clamp(min=1e-12)
                
                results['cov_mismatch'] = {
                    'relative_frobenius_error': frobenius_error.item(),
                    'batch_size': z.shape[0]
                }
        
        except Exception as e:
            results['error'] = str(e)
        
        return results

    def _rhmc_exploration(
        self,
        z0: torch.Tensor,
        record_traj: bool = False,
        mu: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]], torch.Tensor]:
        """
        Step 2: Simple RHMC exploration without acceptance/rejection.
        
        Args:
            z0: Starting position [B, D]
            record_traj: Whether to record trajectory
            mu: Encoder mean for diagnostics [B, D]
        """
        z = z0.clone()
        
        # Sample initial momentum
        rho = self._sample_momentum(z)
        
        # DIAGNOSTIC: RHMC Trajectory Analysis
        import os
        if os.environ.get("RLVAE_DEBUG", "0") == "1" and mu is not None:
            print(f"\n{'='*80}")
            print(f"[RHMC TRAJECTORY DIAGNOSTICS]")
            print(f"{'='*80}")
            
            # Initial state
            dist_z0_mu = torch.norm(z0 - mu, dim=-1)
            rho_norm = torch.norm(rho, dim=-1)
            
            print(f"\n[INITIAL STATE (k=0)]")
            print(f"  ||z0 - μ||:            mean={dist_z0_mu.mean().item():.4f}, max={dist_z0_mu.max().item():.4f}")
            print(f"  ||ρ0||:                mean={rho_norm.mean().item():.4f}, max={rho_norm.max().item():.4f}")
            
            # Track trajectory including metric values
            try:
                model = self._ctx['model']
                if hasattr(model, 'metric_tensor'):
                    G_inv_z0 = model.metric_tensor.compute_metric_inv(z0)
                elif hasattr(model, 'G_inv'):
                    G_inv_z0 = model.G_inv(z0)
                else:
                    G_inv_z0 = None
                
                if G_inv_z0 is not None:
                    _, logdet_z0 = torch.linalg.slogdet(_to_float32(G_inv_z0))
                    init_logdet = logdet_z0.mean().item()
                else:
                    init_logdet = 0.0
            except Exception:
                init_logdet = 0.0
            
            trajectory_data = {
                'distances_from_mu': [dist_z0_mu.mean().item()],
                'distances_from_z0': [0.0],
                'momentum_norms': [rho_norm.mean().item()],
                'logdet_Ginv': [init_logdet],
            }
        
        delta_logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        # AMP-safe: compute quadratic energy in float32 to avoid Half/Float mismatch
        G_inv_ref = self._get_inverse_metric(z)
        rho32 = rho.float() if rho.dtype in (torch.float16, torch.bfloat16) else rho
        G_inv_ref32 = G_inv_ref.float() if G_inv_ref.dtype in (torch.float16, torch.bfloat16) else G_inv_ref
        quadratic_ref32 = torch.einsum('bi,bij,bj->b', rho32, G_inv_ref32, rho32)
        quadratic_ref = quadratic_ref32.to(z.dtype)
        quadratic_start = quadratic_ref.clone()
        resample_counter = 0
        # Simple leapfrog integration
        if record_traj:
            traj = [{'step': 0, 'z': z.clone().detach(), 'rho': rho.clone().detach(), 'logdet': torch.zeros_like(delta_logdet)}]
        else:
            traj = None

        # DEBUG: Log step size right before loop
        import os
        if os.environ.get("RLVAE_DEBUG", "0") == "1" and not hasattr(self, '_rhmc_loop_logged'):
            print(f"\n[RHMC LOOP DEBUG] self.rhmc_step_size = {self.rhmc_step_size}")
            print(f"[RHMC LOOP DEBUG] About to call _leapfrog_step with step_size={self.rhmc_step_size}")
            self._rhmc_loop_logged = True

        for step in range(self.rhmc_steps):
            z, rho, logdet_step = self._leapfrog_step(z, rho, self.rhmc_step_size)
            delta_logdet = delta_logdet + logdet_step
            try:
                G_inv_cur = self._get_inverse_metric(z)
                rho32 = rho.float() if rho.dtype in (torch.float16, torch.bfloat16) else rho
                G_inv_cur32 = G_inv_cur.float() if G_inv_cur.dtype in (torch.float16, torch.bfloat16) else G_inv_cur
                q_cur32 = torch.einsum('bi,bij,bj->b', rho32, G_inv_cur32, rho32)
                q_cur = q_cur32.to(z.dtype)
                drift_from_start = q_cur - quadratic_start
                drift_active = q_cur - quadratic_ref
                if self.max_quadratic_growth > 0:
                    exceed_mask = drift_active > self.max_quadratic_growth
                    if exceed_mask.any():
                        with torch.no_grad():
                            rho_new = self._sample_momentum(z[exceed_mask].detach())
                        rho = rho.clone()
                        rho[exceed_mask] = rho_new
                        rho_new32 = rho_new.float() if rho_new.dtype in (torch.float16, torch.bfloat16) else rho_new
                        G_inv_exc32 = G_inv_cur[exceed_mask].float() if G_inv_cur.dtype in (torch.float16, torch.bfloat16) else G_inv_cur[exceed_mask]
                        q_new32 = torch.einsum('bi,bij,bj->b', rho_new32, G_inv_exc32, rho_new32)
                        q_new = q_new32.to(z.dtype)
                        q_cur = q_cur.clone()
                        q_cur[exceed_mask] = q_new
                        quadratic_ref = quadratic_ref.clone()
                        quadratic_ref[exceed_mask] = q_new
                        quadratic_start = quadratic_start.clone()
                        quadratic_start[exceed_mask] = q_new
                        if record_traj and traj:
                            traj[0]['rho'][exceed_mask] = rho_new.detach()
                        resample_counter += int(exceed_mask.sum().item())
                        if os.environ.get("RLVAE_DEBUG", "0") == "1":
                            print(f"[RHMC QUADRATIC ENERGY] resampled momentum for {int(exceed_mask.sum().item())} sample(s) at step {step+1}")
                        drift_from_start = q_cur - quadratic_start
                        drift_active = q_cur - quadratic_ref
                quadratic_ref = q_cur.detach()
                if os.environ.get("RLVAE_DEBUG", "0") == "1":
                    print(
                        f"[RHMC QUADRATIC ENERGY] step={step+1} "
                        f"start mean={quadratic_start.mean().item():.4f} "
                        f"current mean={q_cur.mean().item():.4f} "
                        f"Δ initial mean={drift_from_start.mean().item():+.4f} "
                        f"Δ active mean={drift_active.mean().item():+.4f}"
                    )
            except Exception as quad_exc:
                if os.environ.get("RLVAE_DEBUG", "0") == "1":
                    print(f"[RHMC QUADRATIC ENERGY] diagnostic failed: {quad_exc}")
            if record_traj:
                traj.append({'step': step + 1, 'z': z.clone().detach(), 'rho': rho.clone().detach(), 'logdet': logdet_step.clone().detach()})
            
            # DIAGNOSTIC: Track each step
            if os.environ.get("RLVAE_DEBUG", "0") == "1" and mu is not None:
                dist_z_mu = torch.norm(z - mu, dim=-1)
                dist_z_z0 = torch.norm(z - z0, dim=-1)
                rho_norm = torch.norm(rho, dim=-1)
                
                # Compute log|G⁻¹(z)| at current position
                try:
                    model = self._ctx['model']
                    if hasattr(model, 'metric_tensor'):
                        G_inv_z = model.metric_tensor.compute_metric_inv(z)
                    elif hasattr(model, 'G_inv'):
                        G_inv_z = model.G_inv(z)
                    else:
                        G_inv_z = None
                    
                    if G_inv_z is not None:
                        _, logdet_z = torch.linalg.slogdet(_to_float32(G_inv_z))
                        logdet_val = logdet_z.mean().item()
                    else:
                        logdet_val = 0.0
                except Exception:
                    logdet_val = 0.0
                
                trajectory_data['distances_from_mu'].append(dist_z_mu.mean().item())
                trajectory_data['distances_from_z0'].append(dist_z_z0.mean().item())
                trajectory_data['momentum_norms'].append(rho_norm.mean().item())
                trajectory_data['logdet_Ginv'].append(logdet_val)
                
                print(f"\n[STEP k={step+1}]")
                print(f"  ||z{step+1} - μ||:          mean={dist_z_mu.mean().item():.4f}")
                print(f"  ||z{step+1} - z0||:         mean={dist_z_z0.mean().item():.4f}")
                print(f"  ||ρ{step+1}||:             mean={rho_norm.mean().item():.4f}")
                print(f"  log|G⁻¹(z{step+1})|:       {logdet_val:.4f}")
                
                if len(trajectory_data['logdet_Ginv']) >= 2:
                    prev_logdet = trajectory_data['logdet_Ginv'][-2]
                    delta_logdet = logdet_val - prev_logdet
                    print(f"  Δ log|G⁻¹| (step):        {delta_logdet:+.4f}")
                    if delta_logdet < 0:
                        print("  ⚠️  Volume drop detected this step (log|G⁻¹| decreased).")
                    
                    if len(trajectory_data['logdet_Ginv']) >= 3:
                        import numpy as np
                        steps_arr = np.arange(len(trajectory_data['logdet_Ginv']))
                        logdets_arr = np.array(trajectory_data['logdet_Ginv'])
                        if np.std(logdets_arr) > 1e-9:
                            corr_running = float(np.corrcoef(steps_arr, logdets_arr)[0, 1])
                            print(f"  corr(step, log|G⁻¹|):    {corr_running:+.4f} (running)")
                            if corr_running < -0.3:
                                print("  ⚠️  Running correlation negative — check force orientation!")
                        else:
                            print("  corr(step, log|G⁻¹|):    +0.0000 (insufficient variance)")
        
        # DIAGNOSTIC: Trajectory Summary
        if os.environ.get("RLVAE_DEBUG", "0") == "1" and mu is not None:
            dist_zK_mu = torch.norm(z - mu, dim=-1)
            dist_zK_z0 = torch.norm(z - z0, dim=-1)
            
            print(f"\n[TRAJECTORY SUMMARY]")
            print(f"  Initial ||z0 - μ||:    {trajectory_data['distances_from_mu'][0]:.4f}")
            print(f"  Final ||zK - μ||:      {dist_zK_mu.mean().item():.4f}")
            print(f"  Total drift from z0:   {dist_zK_z0.mean().item():.4f}")
            print(f"  Net change in ||·-μ||: {(dist_zK_mu.mean() - trajectory_data['distances_from_mu'][0]):.4f}")
            
            if dist_zK_mu.mean() > trajectory_data['distances_from_mu'][0]:
                print(f"  → RHMC MOVED AWAY from μ")
            elif dist_zK_mu.mean() < trajectory_data['distances_from_mu'][0]:
                print(f"  → RHMC MOVED TOWARD μ")
            else:
                print(f"  → RHMC STAYED AT SAME DISTANCE from μ")
            
            # Monotonicity check
            distances = trajectory_data['distances_from_mu']
            if all(distances[i] <= distances[i+1] for i in range(len(distances)-1)):
                print(f"  → Trajectory MONOTONICALLY MOVES AWAY from μ")
            elif all(distances[i] >= distances[i+1] for i in range(len(distances)-1)):
                print(f"  → Trajectory MONOTONICALLY MOVES TOWARD μ")
            else:
                print(f"  → Trajectory is NON-MONOTONIC (oscillating)")
            
            # Metric evolution analysis
            logdets = trajectory_data['logdet_Ginv']
            if len(logdets) > 1:
                print(f"\n[METRIC EVOLUTION]")
                print(f"  Initial log|G⁻¹(z0)|:  {logdets[0]:.4f}")
                print(f"  Final log|G⁻¹(zK)|:    {logdets[-1]:.4f}")
                print(f"  Δ log|G⁻¹|:            {(logdets[-1] - logdets[0]):.4f}")
                
                if logdets[-1] > logdets[0]:
                    print(f"  ✓ RHMC MOVED TO HIGHER log|G⁻¹| (toward prior!)")
                elif logdets[-1] < logdets[0]:
                    print(f"  ⚠️  RHMC MOVED TO LOWER log|G⁻¹| (away from prior!)")
                else:
                    print(f"  → No change in log|G⁻¹|")
                
                # Correlation: step number vs log|G⁻¹|
                import numpy as np
                steps = np.arange(len(logdets))
                logdets_arr = np.array(logdets)
                if len(steps) > 2:
                    corr = np.corrcoef(steps, logdets_arr)[0, 1]
                    print(f"  Correlation(step, log|G⁻¹|): {corr:.4f}")
                    if abs(corr) > 0.7:
                        direction = "increasing" if corr > 0 else "decreasing"
                        print(f"    → Strong {direction} trend!")
                    
                    # CRITICAL: Check RHMC trajectory inversion
                    if corr < -0.5:
                        print(f"  ⚠️  RHMC INVERSION: Trajectory moves AWAY from high-volume regions!")
                        print(f"      This contradicts RHMC objective to maximize log|G⁻¹|.")
                        print(f"      Check _compute_potential_gradient sign and volume_force_sign.")
                    
                    # Also check correlation between distance from μ and log|G⁻¹|
                    distances_arr = np.array(trajectory_data['distances_from_mu'])
                    if len(distances_arr) > 2:
                        corr_dist_vol = np.corrcoef(distances_arr, logdets_arr)[0, 1]
                        print(f"  Correlation(||z-μ||, log|G⁻¹|): {corr_dist_vol:.4f}")
                        if corr_dist_vol < -0.3:
                            print(f"  ⚠️  DISTANCE-VOLUME INVERSION: Farther from μ → Lower volume!")
                            print(f"      This suggests RHMC pushes samples to low-volume regions.")
            
            print(f"\n{'='*80}\n")
            try:
                logdet_stats = delta_logdet.detach()
                print(f"[RHMC VOLUME JACOBIAN]")
                print(f"  log|det DΦ| mean: {logdet_stats.mean().item():.4f}")
                print(f"  log|det DΦ| min:  {logdet_stats.min().item():.4f}")
                print(f"  log|det DΦ| max:  {logdet_stats.max().item():.4f}")
                if resample_counter > 0:
                    print(
                        f"[RHMC QUADRATIC ENERGY] momentum resampled {resample_counter} "
                        f"time(s) (threshold {self.max_quadratic_growth})"
                    )
            except Exception:
                pass

        return z, traj, delta_logdet
    
    def _sample_momentum(self, z: torch.Tensor) -> torch.Tensor:
        """
        Simple momentum sampling: ρ ~ N(0, G(z))
        """
        try:
            G = self._ctx['model'].G(z)
            G = _symmetrize(G)
            L, _, _ = _safe_cholesky(G + self.eps_reg * torch.eye(z.shape[-1], device=z.device, dtype=G.dtype).unsqueeze(0), self.eps_reg)
            eps = torch.randn_like(z, dtype=L.dtype)
            rho32 = torch.einsum('bij,bj->bi', L, eps)
            rho = rho32.to(z.dtype)
            rho = torch.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
            # Momentum clipping for safety
            rho_norm = torch.norm(rho, dim=-1, keepdim=True)
            rho = torch.where(rho_norm > self.max_momentum_norm, rho * (self.max_momentum_norm / (rho_norm + 1e-12)), rho)
            return rho
        except:
            # Fallback to isotropic sampling
            rho = torch.randn_like(z)
            rho = torch.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
            rho_norm = torch.norm(rho, dim=-1, keepdim=True)
            rho = torch.where(rho_norm > self.max_momentum_norm, rho * (self.max_momentum_norm / (rho_norm + 1e-12)), rho)
            return rho
    
    def _leapfrog_step(self, z: torch.Tensor, rho: torch.Tensor, step_size: float) -> tuple:
        """
        Simple leapfrog integration step (approximate RMHMC).

        Notes:
        - Omits the kinetic position-dependence term -0.5 ∇_z [ρᵀ G^{-1}(z) ρ] for
          simplicity and stability; use small steps and clipping.
        - Jacobian of the integrator is ignored (KL jacobian mode is a placeholder).
        """
        try:
            batch_size = z.shape[0]
            latent_dim = z.shape[-1]
            logdet_step = torch.zeros(batch_size, device=z.device, dtype=z.dtype)

            # Ensure z requires grad to build autograd graph inside integrator
            if not z.requires_grad:
                z = z.clone().requires_grad_(True)
            # Half step for momentum
            grad_U = self._compute_potential_gradient(z)
            grad_U = torch.nan_to_num(grad_U, nan=0.0, posinf=0.0, neginf=0.0)
            # Optional kinetic position-dependent gradient term
            if self.kinetic_grad_enabled:
                try:
                    z_req = z
                    G_inv_k = self._get_inverse_metric(z_req)
                    # AMP-safe: compute kinetic energy in float32 when needed
                    rho_work = rho.float() if rho.dtype in (torch.float16, torch.bfloat16) else rho
                    G_inv_k_work = G_inv_k.float() if G_inv_k.dtype in (torch.float16, torch.bfloat16) else G_inv_k
                    s_work = torch.einsum('bi,bij,bj->b', rho_work, G_inv_k_work, rho_work)
                    kin_grad = torch.autograd.grad(s_work.sum(), z_req, retain_graph=True, create_graph=False, allow_unused=True)[0]
                    if kin_grad is None:
                        kin_grad = torch.zeros_like(z_req)
                    grad_U = grad_U + self.kinetic_grad_weight * kin_grad.to(z.dtype)
                except Exception:
                    pass
            rho = rho - 0.5 * step_size * grad_U
            rho = torch.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Full step for position
            G_inv = self._get_inverse_metric(z)
            # Regularize inverse metric for numerical safety
            d = z.shape[-1]
            I = torch.eye(d, device=z.device, dtype=G_inv.dtype).unsqueeze(0)
            G_inv_reg = G_inv + self.eps_reg * I
            # AMP-safe velocity computation
            rho_work = rho.float() if rho.dtype in (torch.float16, torch.bfloat16) else rho
            G_inv_reg_work = G_inv_reg.float() if G_inv_reg.dtype in (torch.float16, torch.bfloat16) else G_inv_reg
            velocity_work = torch.einsum('bij,bj->bi', G_inv_reg_work, rho_work)
            velocity = velocity_work.to(z.dtype)
            # Guard against non-finite velocity components
            velocity = torch.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
            # Clip velocity
            v_norm = torch.norm(velocity, dim=-1, keepdim=True)
            if self.max_velocity_norm > 0:
                vel_scale = torch.where(
                    v_norm > self.max_velocity_norm,
                    self.max_velocity_norm / (v_norm + 1e-12),
                    torch.ones_like(v_norm)
                )
                velocity = velocity * vel_scale
                rho = rho * vel_scale
                logdet_step = logdet_step + latent_dim * torch.log(vel_scale.squeeze(-1).clamp(min=1e-6))
            # Adaptive effective step to bound position change
            eff_step = step_size#torch.clamp(self.max_position_step / (v_norm + 1e-12), max=1.0) * step_size
            
            # DEBUG: Log step size scaling
            import os
            if os.environ.get("RLVAE_DEBUG", "0") == "1" and hasattr(self, '_step_scaling_logged') and not self._step_scaling_logged:
                print(f"\n[STEP SIZE SCALING DEBUG]")
                print(f"  base step_size: {step_size:.6f}")
                print(f"  velocity norm mean: {v_norm.mean().item():.6e}")
                print(f"  max_position_step: {self.max_position_step}")
                print(f"  scaling factor (max_pos_step/v_norm): {(self.max_position_step / (v_norm.mean() + 1e-12)).item():.6e}")
                #print(f"  eff_step before stiff: {eff_step.mean().item():.6e}")
            
            # FIX: Geometry-aware scaling was TOO aggressive, crushing steps to 0.0002
            # Old: stiff_scale = sqrt(min_eig) → could be ~0.002 → 600× reduction!
            # New: Bounded scaling to prevent excessive reduction
            try:
                evals_Ginv = torch.linalg.eigvalsh(G_inv_reg.float())
                min_eig_Ginv = evals_Ginv.min(dim=-1, keepdim=True).values.to(z.dtype)
                # FIX: Clamp min_eig to reasonable range before sqrt
                # This prevents the step from becoming microscopic in high-curvature regions
                min_eig_clamped = torch.clamp(min_eig_Ginv, min=0.01, max=100.0)  # Bounded: 0.1 to 10.0 after sqrt
                stiff_scale = torch.sqrt(min_eig_clamped)
                eff_step = eff_step #* stiff_scale
                
                # DEBUG: Log stiffness scaling
                if os.environ.get("RLVAE_DEBUG", "0") == "1" and hasattr(self, '_step_scaling_logged') and not self._step_scaling_logged:
                    print(f"  min_eig_Ginv mean: {min_eig_Ginv.mean().item():.6e} (raw)")
                    print(f"  min_eig_clamped mean: {min_eig_clamped.mean().item():.6e}")
                    print(f"  stiff_scale mean: {stiff_scale.mean().item():.6e}")
                    #print(f"  eff_step after stiff: {eff_step.mean().item():.6e}")
                    print(f"  → Effective step is {(eff_step.mean() / step_size).item():.2%} of configured step_size!")
                    self._step_scaling_logged = True
            except Exception:
                pass
            z = z + eff_step * velocity
            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            # Soft braking on momentum when approaching norm boundary
            soft_limit = float(getattr(self, "soft_position_norm", 5.5))
            if soft_limit > 0:
                z_norm_soft = torch.norm(z, dim=-1, keepdim=True)
                soft_mask = z_norm_soft > soft_limit
                if soft_mask.any():
                    scale_full = torch.ones_like(z_norm_soft)
                    scale_full[soft_mask] = torch.tanh(z_norm_soft[soft_mask] / soft_limit).clamp(min=1e-6)
                    rho = rho * scale_full
                    logdet_step = logdet_step + torch.where(
                        soft_mask.squeeze(-1),
                        latent_dim * torch.log(scale_full.squeeze(-1).clamp(min=1e-6)),
                        torch.zeros_like(logdet_step),
                    )
            # Optional single-step projection toward high log|G^{-1}|
            try:
                tau = float(self.projection_step_scale) * float(step_size)
                if tau > 0:
                    z_req = z.clone().detach().requires_grad_(True)
                    # grad 0.5·log|G^{-1}| via metric route
                    G_proj = self._ctx['model'].G(z_req)
                    vol_g = half_logdet_volume(G_proj, 'g', jitter=self.eps_reg)
                    grad_vol_g = torch.autograd.grad(vol_g.sum(), z_req, retain_graph=False, create_graph=False, allow_unused=True)[0]
                    if grad_vol_g is not None:
                        grad_logdet_ginv = 2.0 * grad_vol_g
                        z = (z_req + tau * grad_logdet_ginv).detach()
            except Exception:
                pass
            # Clamp absolute position norm only if enabled (> 0).
            # Hard clamping here was causing samples to accumulate on a
            # spherical shell (ring) in PCA plots. Allow disabling by
            # setting `max_position_norm <= 0` in config.
            try:
                import math
                norm_limit = float(getattr(self, 'max_position_norm', float('inf')))
            except Exception:
                norm_limit = float('inf')
            if math.isfinite(norm_limit) and norm_limit > 0:
                z_norm = torch.norm(z, dim=-1, keepdim=True)
                if (z_norm > norm_limit).any():
                    n_clipped = (z_norm > norm_limit).sum().item()
                    print(
                        f"🔴 [RHMC] Position norm clipping: {n_clipped}/{z.shape[0]} exceeded {norm_limit} "
                        f"(max norm: {z_norm.max().item():.2f})"
                    )
                clip_factor = torch.where(
                    z_norm > norm_limit,
                    norm_limit / (z_norm + 1e-12),
                    torch.ones_like(z_norm)
                )
                logdet_step = logdet_step + (z_norm > norm_limit).squeeze(-1).to(logdet_step.dtype) * latent_dim * torch.log(clip_factor.squeeze(-1).clamp(min=1e-6))
                z = torch.where(z_norm > norm_limit, z * clip_factor, z)
            
            # Half step for momentum
            grad_U = self._compute_potential_gradient(z)
            grad_U = torch.nan_to_num(grad_U, nan=0.0, posinf=0.0, neginf=0.0)
            if self.kinetic_grad_enabled:
                try:
                    z_req = z
                    G_inv_k = self._get_inverse_metric(z_req)
                    s = torch.einsum('bi,bij,bj->b', rho, G_inv_k, rho)
                    kin_grad = torch.autograd.grad(s.sum(), z_req, retain_graph=True, create_graph=False, allow_unused=True)[0]
                    if kin_grad is None:
                        kin_grad = torch.zeros_like(z_req)
                    grad_U = grad_U + self.kinetic_grad_weight * kin_grad
                except Exception:
                    pass
            rho = rho - 0.5 * step_size * grad_U
            rho = torch.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
            
            return z, rho, logdet_step
            
        except Exception as e:
            print(f"⚠️ Leapfrog step failed: {e}")
            zeros = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
            return z, rho, zeros
    
    def _compute_log_posterior(self, z: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Approximate log q(z|x) using the Riemannian Gaussian centred at μ.
        """
        return self._compute_log_riemannian_gaussian(z, mu, log_var)

    def _compute_log_riemannian_gaussian(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        *,
        covariance: Optional[torch.Tensor] = None,
        alpha_override: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute log-density of the Riemannian Gaussian q_Riem(z|x).
        """
        try:
            if covariance is not None:
                Sigma = covariance
                alpha = None  # Not computed when covariance is provided
            else:
                alpha = alpha_override if alpha_override is not None else self._resolve_alpha()
                G_inv_mu = self._get_inverse_metric(mu)
                Sigma = self._make_covariance(G_inv_mu, alpha)
            
            # DIAGNOSTIC: Log Sigma details before passing to log_q_riem
            if os.environ.get("RLVAE_DEBUG", "0") == "1":
                try:
                    sigma_dbg = _to_float32(Sigma)
                    eigvals_sigma = torch.linalg.eigvalsh(sigma_dbg)
                    logdet_sigma = torch.linalg.slogdet(sigma_dbg)[1]
                    trace_sigma = torch.diagonal(Sigma, dim1=-2, dim2=-1).sum(dim=-1)
                    
                    print(f"\n[_compute_log_riemannian_gaussian BEFORE log_q_riem]")
                    print(f"  min_cov_eig:           {self.min_cov_eig:.6f}")
                    print(f"  covariance_provided:   {covariance is not None}")
                    print(f"  alpha:                 {alpha if alpha is not None else 'N/A (covariance provided)'}")
                    print(f"  Sigma eigenvalues:     min={eigvals_sigma.min().item():.6f}, max={eigvals_sigma.max().item():.6f}")
                    print(f"  Sigma trace:           {trace_sigma.mean().item():.6f}")
                    print(f"  log|Sigma|:            {logdet_sigma.mean().item():.6f}")
                    print(f"  ||z - μ||:             {torch.norm(z - mu, dim=-1).mean().item():.6f}")
                    
                    # Always compare with raw G_inv_mu
                    print(f"  [Comparison with raw G⁻¹(μ)]")
                    G_inv_raw = self._ctx['model'].G_inv(mu)  # FIX: Use G_inv() not Ginv()
                    G_inv_dbg = _to_float32(G_inv_raw)
                    eigvals_ginv_raw = torch.linalg.eigvalsh(G_inv_dbg)
                    logdet_ginv_raw = torch.linalg.slogdet(G_inv_dbg)[1]
                    print(f"    G⁻¹(μ) eigenvalues:  min={eigvals_ginv_raw.min().item():.6f}, max={eigvals_ginv_raw.max().item():.6f}")
                    print(f"    log|G⁻¹(μ)|:         {logdet_ginv_raw.mean().item():.6f}")
                    print(f"    Expected log|Σ|:     {logdet_ginv_raw.mean().item():.6f} (if α=1, ε≈0)")
                    print(f"    Actual log|Σ|:       {logdet_sigma.mean().item():.6f}")
                    print(f"    Δ log|Σ|:            {(logdet_sigma - logdet_ginv_raw).mean().item():+.6f}")
                except Exception as e:
                    print(f"[_compute_log_riemannian_gaussian] Diagnostic failed: {e}")
            
            return log_q_riem(z, mu, Sigma, min_eig=self.min_cov_eig)
        except Exception as exc:
            if getattr(self, '_ctx', None) and getattr(self._ctx.get('model', None), 'riemannian_strict', False) or os.environ.get('RLVAE_STRICT', '0') == '1':
                raise RuntimeError(f"RiemannianRHMCPosterior: log q_Riem failed under strict mode: {exc}")
            print(f"⚠️ Log q_Riem computation failed: {exc}, using isotropic fallback")
            diff = z - mu
            quad_form = torch.sum(diff ** 2, dim=-1)
            d = z.shape[-1]
            return -0.5 * quad_form - 0.5 * d * math.log(2 * math.pi)
    
    def _compute_log_prior(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(z) for Riemannian volume prior.
        
        Corrected formulation: p(z) ∝ √det(G⁻¹(z))
        
        This favors regions where the precision matrix G⁻¹ has large determinant,
        corresponding to high-confidence/low-variance regions in the latent space.
        
        log p(z) = 0.5 * log det(G⁻¹(z)) + constant
        
        Args:
            z: Latent samples [B, D]
            
        Returns:
            log_p: Log prior probability [B]
        """
        try:
            model = self._ctx['model']
            # Retrieve G_inv (precision matrix) directly - this is what Stage B provides
            if hasattr(model, 'G_inv'):
                G_inv_z = model.G_inv(z)
            elif hasattr(model, 'metric_tensor'):
                mt = model.metric_tensor
                if hasattr(mt, 'compute_inverse_metric'):
                    G_inv_z = mt.compute_inverse_metric(z)
                elif hasattr(mt, 'compute_metric'):
                    # If only G is available, invert it to get G_inv
                    G_z = mt.compute_metric(z)
                    G_inv_z = torch.linalg.inv(G_z)
                else:
                    raise RuntimeError("Metric tensor does not provide G(z) or G^{-1}(z).")
            elif hasattr(model, 'G'):
                # Fallback: invert G to get G_inv
                G_z = model.G(z)
                G_inv_z = torch.linalg.inv(G_z)
            else:
                raise RuntimeError("Model does not expose G(z) or G^{-1}(z).")
            
            # Use G_inv directly for volume term (represents precision)
            G_inv_z32 = G_inv_z.float() if G_inv_z.dtype in (torch.float16, torch.bfloat16) else G_inv_z
            half_logdet = half_logdet_volume(G_inv_z32, 'ginv', jitter=self.eps_reg)

            d = z.shape[-1]
            log_det_term = (half_logdet * self.volume_bias_weight).to(z.dtype)
            
            mode = getattr(self, 'kl_prior_mode', 'uniform')
            if mode == 'uniform':
                # Uniform prior on the manifold: p(z) ∝ √det(G⁻¹(z))
                log_p = log_det_term + float(self.uniform_prior_log_norm)
            elif mode in ('volume_gaussian', 'gaussian'):
                # Volume-weighted Gaussian prior (currently not used)
                # Would need to define appropriate quadratic term with G or G_inv
                # For now, fallback to uniform
                log_p = log_det_term + float(self.uniform_prior_log_norm)
            else:
                # Fallback: uniform mode
                log_p = log_det_term + float(self.uniform_prior_log_norm)
            
            return log_p
            
        except Exception as e:
            # Fallback: standard Gaussian prior (no volume term)
            d = z.shape[-1]
            mode = getattr(self, 'kl_prior_mode', 'volume_gaussian')
            if mode == 'uniform':
                return torch.full(
                    (z.shape[0],),
                    float(self.uniform_prior_log_norm),
                    device=z.device,
                    dtype=z.dtype,
                )
            z_norm_sq = torch.sum(z ** 2, dim=-1)
            return -0.5 * z_norm_sq - 0.5 * d * math.log(2 * math.pi)
    
    def _compute_potential_gradient(self, z: torch.Tensor) -> torch.Tensor:
        """
        ∇U(z) for uniform volume prior p(z) ∝ √det(G^{-1}(z)).
        We minimize: U(z) = -w · 0.5 log det G^{-1}(z) (no Gaussian term).
        
        This aligns the RHMC dynamics with the uniform volume prior objective.

        Robust to cases where G does not depend on z (allow_unused=True).
        Falls back to zero volume-gradient in that case.
        """
        # If autograd is disabled (e.g., validation), return zero gradient for uniform prior
        if not torch.is_grad_enabled():
            return torch.zeros_like(z)
        # Ensure we can take grads w.r.t. z
        if not z.requires_grad:
            z = z.clone().requires_grad_(True)

        # CORRECTION: Remove Gaussian term to align with uniform volume prior
        # For uniform volume prior p(z) ∝ √det(G^{-1}(z)), we want U(z) = -½ log det G^{-1}(z)
        # This means we only need the volume gradient, not the Gaussian term
        base = torch.zeros_like(z)

        # Volume term: attract to high log|G^{-1}| by following ∇U(z) = -∇log p(z),
        # where log p(z) = ½ log|G^{-1}(z)| for the uniform volume prior.
        #print('?!!!!computing potential gradient 1!!!')
        try:
            #computing potential gradient 2!!!')
            rep = getattr(self, 'volume_force_representation', 'g')
            rep = rep if rep in ('g', 'ginv') else 'g'
            grad_jitter = float(getattr(self, 'volume_grad_jitter', self.eps_reg))
            eig_floor = float(getattr(self, 'volume_grad_eig_floor', 0.0))
            if rep == 'g':
                #print('using g??????')
                G = self._ctx['model'].G(z)
                G32 = G.float() if G.dtype in (torch.float16, torch.bfloat16) else G
                # Clamp spectrum before logdet gradient to avoid NaNs
                try:
                    evals, evecs = torch.linalg.eigh(0.5 * (G32 + G32.transpose(-1, -2)))
                    if eig_floor > 0:
                        evals = torch.clamp(evals, min=eig_floor)
                    G32 = evecs @ (evals.unsqueeze(-1) * evecs.transpose(-1, -2))
                except Exception:
                    pass
                d = G32.shape[-1]
                eye = torch.eye(d, device=G32.device, dtype=G32.dtype).unsqueeze(0)
                log_vol = half_logdet_volume(G32 + grad_jitter * eye, 'g', jitter=grad_jitter)  # = +½ log|G^{-1}|
            else:
                Ginv = self._get_inverse_metric(z)
                Ginv32 = Ginv.float() if Ginv.dtype in (torch.float16, torch.bfloat16) else Ginv
                try:
                    evals, evecs = torch.linalg.eigh(0.5 * (Ginv32 + Ginv32.transpose(-1, -2)))
                    if eig_floor > 0:
                        evals = torch.clamp(evals, min=eig_floor)
                    Ginv32 = evecs @ (evals.unsqueeze(-1) * evecs.transpose(-1, -2))
                except Exception:
                    pass
                d = Ginv32.shape[-1]
                eye = torch.eye(d, device=Ginv32.device, dtype=Ginv32.dtype).unsqueeze(0)
                log_vol = half_logdet_volume(Ginv32 + grad_jitter * eye, 'ginv', jitter=grad_jitter)  # = +½ log|G^{-1}|

            grad_log_vol, = torch.autograd.grad(
                log_vol.sum(),
                z,
                retain_graph=True,
                create_graph=True,
                allow_unused=True
            )
            if grad_log_vol is None:
                grad_log_vol = torch.zeros_like(z)
            # Resolve force sign/scale from config so we can flip during debugging
            sign = float(getattr(self, 'volume_force_sign', 1.0))
            scale = sign * float(getattr(self, 'volume_grad_scale', 1.0)) * float(getattr(self, 'volume_bias_weight', 1.0))
            grad = base - scale * grad_log_vol  # ∇U = -scale * ∇(½ log|G^{-1}|)
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Optional finite-difference sanity checks on the force direction
            debug_force = os.environ.get("RLVAE_FORCE_SANITY", "0") == "1"
            if debug_force:
                try:
                    eps_fd = float(os.environ.get("RLVAE_FORCE_EPS", "1e-4"))
                    # Normalise gradient direction safely
                    dir_unit = F.normalize(grad_log_vol.detach(), dim=-1, eps=1e-12)
                    z_detached = z.detach()
                    plus = self._evaluate_half_logdet(z_detached + eps_fd * dir_unit, rep)
                    minus = self._evaluate_half_logdet(z_detached - eps_fd * dir_unit, rep)
                    fd_proj = (plus - minus) / (2.0 * eps_fd)
                    dot = (grad * grad_log_vol).sum(dim=-1)
                    cos = dot / (
                        torch.norm(grad, dim=-1) * torch.norm(grad_log_vol, dim=-1).clamp_min(1e-12)
                    )
                    if not hasattr(self, "_force_sanity_printed") or not self._force_sanity_printed:
                        print(
                            "[FORCE SANITY] "
                            f"rep={rep} sign={sign:.1f} "
                            f"dot_mean={dot.mean().item():+.4e} "
                            f"cos_mean={cos.mean().item():+.4f} "
                            f"fd_proj_mean={fd_proj.mean().item():+.4e}"
                        )
                        self._force_sanity_printed = True
                except Exception as dbg_exc:
                    if os.environ.get("RLVAE_DEBUG", "0") == "1":
                        print(f"[FORCE SANITY] diagnostic failed: {dbg_exc}")
            
            # DEBUG: Log volume force direction
            if os.environ.get("RLVAE_DEBUG", "0") == "1":
                print(f"\n[VOLUME FORCE & GRADIENT DIRECTION DEBUG]")
                print(f"  representation: {rep}")
                print(f"  sign: {sign}")
                print(f"  scale: {scale}")
                print(f"  grad_log_vol mean: {grad_log_vol.mean().item():.6e}, norm: {torch.norm(grad_log_vol, dim=-1).mean().item():.6e}")
                print(f"  grad_U mean: {grad.mean().item():.6e}, norm: {torch.norm(grad, dim=-1).mean().item():.6e}")
                
                # Compute both G and G⁻¹ to verify direction
                try:
                    G_z = self._ctx['model'].G(z)
                    Ginv_z = self._get_inverse_metric(z)
                    G32 = G_z.float()
                    Ginv32 = Ginv_z.float()
                    
                    # Use Cholesky-based half-logdet for numerical stability under SPD assumptions.
                    half_logdet_g = half_logdet_volume(G32, 'g', jitter=self.eps_reg)  # = -½ log|G|
                    half_logdet_ginv = half_logdet_volume(Ginv32, 'ginv', jitter=self.eps_reg)  # = +½ log|G⁻¹|
                    logdet_G = -half_logdet_g
                    logdet_Ginv = half_logdet_ginv
                    
                    print(f"\n  Current metric values:")
                    print(f"    +0.5 log|G(z)|:     {logdet_G.mean().item():.6f}")
                    print(f"    +0.5 log|G⁻¹(z)|:   {logdet_Ginv.mean().item():.6f}")
                    print(f"    Sum (should ≈ 0):   {(logdet_G.mean() + logdet_Ginv.mean()).item():.6f}")
                    if not torch.isfinite(logdet_G).all() or not torch.isfinite(logdet_Ginv).all():
                        print("  ⚠️  Non-finite logdet encountered; inspect metric conditioning.")
                        try:
                            eigvals_G = torch.linalg.eigvalsh(G32)
                            eigvals_Ginv = torch.linalg.eigvalsh(Ginv32)
                            print(f"    eig(G)  min={eigvals_G.min().item():.4e}, max={eigvals_G.max().item():.4e}")
                            print(f"    eig(G⁻¹) min={eigvals_Ginv.min().item():.4e}, max={eigvals_Ginv.max().item():.4e}")
                        except Exception as eig_exc:
                            print(f"    (Eigenvalue check failed: {eig_exc})")
                    
                    # Compute gradient in BOTH representations using detached copies to avoid polluting the training graph.
                    grad_G = grad_Ginv = None
                    try:
                        with torch.enable_grad():
                            z_dbg = z.detach().clone().requires_grad_(True)
                            G_dbg = self._ctx['model'].G(z_dbg)
                            G_dbg = _symmetrize(G_dbg)
                            half_logdet_g_dbg = half_logdet_volume(G_dbg.float(), 'g', jitter=self.eps_reg)
                            logdet_G_dbg = -half_logdet_g_dbg
                            grad_G = -torch.autograd.grad(
                                logdet_G_dbg.sum(),
                                z_dbg,
                                retain_graph=True,
                                create_graph=False,
                                allow_unused=True,
                            )[0]

                            # Invert without additional clamps for diagnostic purposes only.
                            Ginv_dbg = torch.linalg.inv(G_dbg.float()).to(G_dbg.dtype)
                            half_logdet_ginv_dbg = half_logdet_volume(Ginv_dbg.float(), 'ginv', jitter=self.eps_reg)
                            logdet_Ginv_dbg = half_logdet_ginv_dbg
                            grad_Ginv = -torch.autograd.grad(
                                logdet_Ginv_dbg.sum(),
                                z_dbg,
                                retain_graph=False,
                                create_graph=False,
                                allow_unused=True,
                            )[0]
                            if grad_G is not None:
                                grad_G = grad_G.detach()
                            if grad_Ginv is not None:
                                grad_Ginv = grad_Ginv.detach()
                    except Exception as grad_exc:
                        print(f"  [Gradient diagnostic failed: {grad_exc}]")
                    
                    def _grad_stats(name: str, tensor: Optional[torch.Tensor]) -> Tuple[float, bool]:
                        if tensor is None:
                            return float("nan"), False
                        finite_mask = torch.isfinite(tensor)
                        if not finite_mask.all():
                            print(f"  ⚠️  {name} contains non-finite entries "
                                  f"(nan={torch.isnan(tensor).any().item()}, "
                                  f"inf={torch.isinf(tensor).any().item()}).")
                        norm = torch.linalg.norm(tensor, dim=-1)
                        finite_norm = norm[torch.isfinite(norm)]
                        norm_mean = finite_norm.mean().item() if finite_norm.numel() > 0 else float("nan")
                        return norm_mean, finite_mask.all().item()
                    
                    norm_grad_G, grad_G_ok = _grad_stats("∇(0.5 log|G|)", grad_G)
                    norm_grad_Ginv, grad_Ginv_ok = _grad_stats("∇(0.5 log|G⁻¹|)", grad_Ginv)
                    
                    if grad_G is not None and grad_Ginv is not None:
                        dot = (grad_G * grad_Ginv).sum(dim=-1)
                        if torch.isfinite(dot).any():
                            dot_val = dot[torch.isfinite(dot)].mean().item()
                        else:
                            dot_val = float("nan")
                        print(f"\n  Gradient directions:")
                        print(f"    ∇(0.5 log|G|) norm:     {norm_grad_G:.6e}")
                        print(f"    ∇(0.5 log|G⁻¹|) norm:   {norm_grad_Ginv:.6e}")
                        print(f"    ∇G · ∇G⁻¹ (should <0):  {dot_val:.6e}")
                        if (not grad_G_ok) or (not grad_Ginv_ok) or not torch.isfinite(dot).all():
                            print("    ⚠️  Gradient diagnostics detected non-finite values; check conditioning.")
                            try:
                                eigvals_G = torch.linalg.eigvalsh(G32.detach())
                                eigvals_Ginv = torch.linalg.eigvalsh(Ginv32.detach())
                                print(f"       eig(G)  min={eigvals_G.min().item():.4e}, max={eigvals_G.max().item():.4e}")
                                print(f"       eig(G⁻¹) min={eigvals_Ginv.min().item():.4e}, max={eigvals_Ginv.max().item():.4e}")
                            except Exception as eig_exc:
                                print(f"       (Eigenvalue check failed: {eig_exc})")
                        
                        # What grad_U actually is
                        if rep == 'g':
                            print(f"\n  Using rep='g': grad_log_vol = ∇(0.5 log|G⁻¹|)")
                        else:
                            print(f"\n  Using rep='ginv': grad_log_vol = ∇(0.5 log|G⁻¹|) via G⁻¹")
                        
                        print(f"  grad_U = -{sign} * grad_log_vol")
                        print("  → For uniform prior p ∝ √det(G⁻¹), we want grad_U = -∇(0.5 log|G⁻¹|) = ∇(0.5 log|G|)")
                    
                        # Expected behavior in leapfrog: ρ -= 0.5 * step * grad_U
                        # Then z += step * velocity
                        # So grad_U with positive sign nudges momentum to increase log|G⁻¹|
                        if sign > 0:
                            print(f"  ✓ CORRECT: sign=+1 → grad_U ascends log|G| → pushes TOWARD high G⁻¹ regions")
                        else:
                            print(f"  ⚠️  WARNING: sign<=0 flips the force and may push AWAY from high G⁻¹!")
                    
                except Exception as e:
                    print(f"  [Error computing gradient verification: {e}]")
                
                self._volume_force_logged = True

            # EXTENDED TRACE: gradient sanity with finite differences and route comparisons
            # Only run heavy RHMC gradient trace if explicitly requested
            if os.environ.get("RLVAE_TRACE_RHMC", "0") == "1":
                try:
                    volume_grad_sanity(
                        self._ctx['model'],
                        z.detach(),
                        rep=rep,
                        sign=float(sign),
                        jitter=float(self.eps_reg),
                        eig_floor=float(getattr(self, 'volume_grad_eig_floor', 0.0)),
                        label="_compute_potential_gradient",
                    )
                except Exception as _e:
                    if os.environ.get("RLVAE_DEBUG", "0") == "1":
                        print(f"[GRAD TRACE] volume_grad_sanity failed: {_e}")
            
            return grad.to(z.dtype)
        except Exception:
            # Safe fallback: zero gradient for uniform prior
            return torch.zeros_like(z)
    
    def _evaluate_half_logdet(self, z: torch.Tensor, rep: str) -> torch.Tensor:
        """Evaluate ±½ log|det| for diagnostics without building autograd graphs."""
        with torch.no_grad():
            if rep == 'g':
                G = self._ctx['model'].G(z)
                return half_logdet_volume(G, 'g', jitter=self.eps_reg)
            else:
                Ginv = self._get_inverse_metric(z)
                return half_logdet_volume(Ginv, 'ginv', jitter=self.eps_reg)
    
    def snapshot_state(self) -> Dict[str, Any]:
        """Capture mutable sampler state for safe restoration."""
        state = {
            'rhmc_steps': int(getattr(self, 'rhmc_steps', 4)),
            'rhmc_step_size': float(getattr(self, 'rhmc_step_size', 0.1)),
            'rhmc_alpha': float(getattr(self, 'rhmc_alpha', 0.5)),
            'eps_reg': float(getattr(self, 'eps_reg', 0.0001)),
            'min_cov_eig': float(getattr(self, 'min_cov_eig', 0.001)),
        }
        last_sigma = getattr(self, '_last_sigma_mu', None)
        if isinstance(last_sigma, torch.Tensor):
            state['_last_sigma_mu'] = last_sigma.detach().clone()
        else:
            state['_last_sigma_mu'] = None
        return state

    def restore_state(self, snapshot: Optional[Dict[str, Any]]) -> None:
        """Restore sampler state captured by ``snapshot_state``."""
        if not isinstance(snapshot, dict):
            return
        for key in ('rhmc_steps', 'rhmc_step_size', 'rhmc_alpha', 'eps_reg', 'min_cov_eig'):
            if key in snapshot:
                try:
                    setattr(self, key, snapshot[key])
                except Exception:
                    pass
        sigma = snapshot.get('_last_sigma_mu', None)
        if isinstance(sigma, torch.Tensor):
            self._last_sigma_mu = sigma.detach().clone()
        else:
            self._last_sigma_mu = None

    def get_config(self) -> Dict[str, Any]:
        """Return current configuration."""
        return {
            'rhmc_steps': self.rhmc_steps,
            'rhmc_step_size': self.rhmc_step_size,
            'rhmc_alpha': self.rhmc_alpha,
            'eps_regularization': self.eps_reg
        }
