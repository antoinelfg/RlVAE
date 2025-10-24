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

from .metric_utils import half_logdet_volume


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    """Return the symmetric part of a batch of matrices."""
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _safe_cholesky(matrix: torch.Tensor, jitter: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute a stable Cholesky factor with AMP safety (upcast to float32).

    Prefer torch.linalg.cholesky_ex when available to detect failures and
    retry with a small diagonal jitter for numerical stability.
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
        return chol, matrix
    except RuntimeError:
        d = matrix.shape[-1]
        eye = torch.eye(d, device=matrix.device, dtype=matrix.dtype).unsqueeze(0)
        stabilized = matrix + jitter * eye
        chol, _ = _chol(stabilized)
        return chol, stabilized


def _log_kinetic_density(
    model: nn.Module,
    z: torch.Tensor,
    rho: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """log π_kin(ρ|z) with kinetic metric G(z)."""
    if not hasattr(model, "G"):
        raise RuntimeError("Model missing metric tensor G for kinetic density computation.")
    G = model.G(z)
    G = _symmetrize(G)
    d = G.shape[-1]
    eye = torch.eye(d, device=G.device, dtype=G.dtype).unsqueeze(0)
    jitter = float(max(eps, 1e-8))
    G = G + jitter * eye
    G_inv = torch.linalg.inv(G.float())
    quad = torch.einsum('bi,bij,bj->b', rho.float(), G_inv, rho.float())
    half_logdet = half_logdet_volume(G, 'g', jitter=jitter)
    if torch.isnan(half_logdet).any():
        jitter = max(jitter * 10, 1e-5)
        G = G + jitter * eye
        G_inv = torch.linalg.inv(G.float())
        quad = torch.einsum('bi,bij,bj->b', rho.float(), G_inv, rho.float())
        half_logdet = half_logdet_volume(G, 'g', jitter=jitter)
    const = z.shape[-1] * math.log(2 * math.pi)
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
    chol, stabilized_sigma = _safe_cholesky(Sigma_flat, min_eig)
    diff = (z_flat - mu_flat).unsqueeze(-1)
    diff32 = diff.float() if diff.dtype != chol.dtype else diff
    sol32 = torch.cholesky_solve(diff32, chol)
    quad_form = torch.einsum('bij,bij->b', diff32, sol32)
    log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-18).sum(dim=-1)
    const = 0.5 * D * math.log(2 * math.pi)
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
        self.rhmc_steps = int(_cfg_get('rhmc_steps', 1))
        self.rhmc_step_size = float(_cfg_get('rhmc_step_size', 5e-3))
        raw_alpha = float(_cfg_get('rhmc_alpha', 1e-2))
        if not math.isfinite(raw_alpha) or raw_alpha <= 0.0:
            fallback_alpha = getattr(model, 'posterior_local_alpha', 0.5)
            self.rhmc_alpha = float(fallback_alpha)
            print(f"[RHMC INIT] α<=0 detected ({raw_alpha}); using fallback {self.rhmc_alpha}")
        else:
            self.rhmc_alpha = raw_alpha
        self.eps_reg = float(_cfg_get('rhmc_eps_reg', _cfg_get('eps_regularization', 1e-3)))

        # Numerical guards
        self.min_cov_eig = float(_cfg_get('min_cov_eig', 1e-3))  # ensure >= eps_reg
        if self.min_cov_eig < self.eps_reg:
            self.min_cov_eig = self.eps_reg
        self.max_momentum_norm = float(_cfg_get('max_momentum_norm', 3.0))
        self.max_velocity_norm = float(_cfg_get('max_velocity_norm', 0.5))
        self.max_position_step = float(_cfg_get('max_position_step', 0.1))
        self.max_position_norm = float(_cfg_get('max_position_norm', 4.0))
        # Keep factorized path disabled by default
        self.use_factorized_G_mu = False
        # Prior mode (affects log p computation under Monte-Carlo KL)
        prior_mode_cfg = _cfg_get('kl_prior_mode', _cfg_get('riemannian_prior_mode', 'volume_gaussian'))
        try:
            self.kl_prior_mode = str(prior_mode_cfg).lower() if prior_mode_cfg is not None else 'volume_gaussian'
        except Exception:
            self.kl_prior_mode = 'volume_gaussian'
        if self.kl_prior_mode not in {'volume_gaussian', 'uniform', 'gaussian'}:
            print(f"[RHMC INIT] ⚠️ Unknown kl_prior_mode='{self.kl_prior_mode}', falling back to 'volume_gaussian'")
            self.kl_prior_mode = 'volume_gaussian'
        self.uniform_prior_log_norm = float(_cfg_get('uniform_prior_log_norm', 0.0))
        self.volume_bias_weight = float(_cfg_get('volume_bias_weight', 1.0))
        self.volume_grad_scale = float(_cfg_get('volume_grad_scale', 1.0))
        # New stability/geometry options
        self.soft_position_norm = float(_cfg_get('soft_position_norm', 5.5))
        self.kinetic_grad_enabled = bool(_cfg_get('kinetic_grad_enabled', False))
        self.kinetic_grad_weight = float(_cfg_get('kinetic_grad_weight', 1.0))
        self.projection_step_scale = float(_cfg_get('projection_step_scale', 0.05))
        self.initial_max_norm = float(_cfg_get('initial_max_norm', 1.5))
        # Control how the volume force is computed (diagnostics/toggles)
        self.volume_force_representation = str(_cfg_get('volume_force_representation', 'g')).lower()  # 'g' or 'ginv'
        self.volume_force_sign = float(_cfg_get('volume_force_sign', 1.0))  # +1 or -1

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
        try:
            print(f"[RHMC INIT] Params: rhmc_steps={int(self.rhmc_steps)}, rhmc_step_size={float(self.rhmc_step_size):.6g}, rhmc_alpha={float(self.rhmc_alpha):.6g}, rhmc_eps_reg={float(self.eps_reg):.6g}")
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
            z0, Sigma_mu = self._sample_initial_riemannian(mu, log_var, alpha_eff)
            self._last_sigma_mu = Sigma_mu.detach().clone()

            # Step 2: RHMC exploration (differentiable pushforward)
            z_final, traj_states = self._rhmc_exploration(z0, record_traj=traj_flag)

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
                    'delta_vol': torch.zeros(
                        z_final.shape[0],
                        device=z_final.device,
                        dtype=z_final.dtype,
                    ),
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
    
    def _sample_initial_riemannian(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        alpha: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step 1: Sample z₀ ~ N_Riem(μ, Σ(μ)) with Σ = α·Ĝ^{-1}(μ) + εI.
        Returns both the samples and the covariance used (for log-density).
        """
        batch_size, latent_dim = mu.shape
        try:
            if self.use_factorized_G_mu:
                # Efficient sampling using factorization of G(μ):
                # z0 = μ + C^{-T}·sqrt(α)·ξ1 + sqrt(ε)·ξ2 where G(μ) = C Cᵀ
                # Build covariance explicitly for return: Σ = α·G^{-1}(μ) + εI
                # 1) Factorize G(μ)
                d = mu.shape[-1]
                I = torch.eye(d, device=mu.device, dtype=mu.dtype).unsqueeze(0)
                G_mu = self._ctx['model'].G(mu)
                G_mu = _symmetrize(G_mu)
                C, _ = _safe_cholesky(G_mu + self.eps_reg * I, self.min_cov_eig)
                # 2) Multi-try sampling
                if C.ndim == 2:
                    C = C.unsqueeze(0)
                B, D = mu.shape
                K = int(getattr(self, 'initial_n_candidates', 8))
                K = max(1, K)
                xi1 = torch.randn(B, K, D, device=mu.device, dtype=C.dtype)
                xi2 = torch.randn(B, K, D, device=mu.device, dtype=mu.dtype)
                Ct = C.transpose(-1, -2).unsqueeze(1)  # [B,1,D,D]
                y = torch.linalg.solve_triangular(Ct, xi1.unsqueeze(-1), upper=True).squeeze(-1)
                z_cand = mu.float().unsqueeze(1) + (alpha ** 0.5) * y + (self.eps_reg ** 0.5) * xi2.float()
                with torch.no_grad():
                    z_eval = z_cand.reshape(B * K, D)
                    from .metric_utils import half_logdet_volume
                    Gz = self._ctx['model'].G(z_eval)
                    h = half_logdet_volume(Gz, 'g', jitter=self.eps_reg).reshape(B, K)
                best = torch.argmax(h, dim=1)
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
                # 5) Prepare Σ for return (for log q computations)
                # Compute G^{-1}(μ) in float32 for stability
                G_inv_mu = torch.linalg.inv(G_mu.float())
                Sigma = self._make_covariance(G_inv_mu.to(mu.dtype), alpha)
                # Optional manifold-aware acceptance: require non-decrease in log|G^{-1}|
                z0 = self._initial_accept_volume(z0, mu)
                return z0, Sigma
            else:
                # Standard covariance-based path with multi-try sampling
                G_inv_mu = self._get_inverse_metric(mu)
                Sigma = self._make_covariance(G_inv_mu, alpha)
                chol, Sigma = _safe_cholesky(Sigma, self.min_cov_eig)
                
                B, D = mu.shape
                K = int(getattr(self, 'initial_n_candidates', 8))
                K = max(1, K)
                
                # Ensure chol has the correct shape [B, D, D]
                if chol.ndim == 2:
                    chol = chol.unsqueeze(0)  # [1, D, D] -> [B, D, D]
                elif chol.ndim > 3:
                    # Handle unexpected extra dimensions by reshaping
                    # First, try to infer the correct dimensions
                    total_elements = chol.numel()
                    expected_elements = B * D * D
                    if total_elements == expected_elements:
                        chol = chol.reshape(B, D, D)
                    else:
                        # If dimensions don't match, try to infer from the tensor
                        # Find the square root of the last two dimensions
                        last_two_dims = chol.shape[-2:]
                        if last_two_dims[0] == last_two_dims[1]:  # Square matrix
                            inferred_D = last_two_dims[0]
                            inferred_B = total_elements // (inferred_D * inferred_D)
                            if inferred_B == B:
                                chol = chol.reshape(B, inferred_D, inferred_D)
                                D = inferred_D  # Update D to match the actual dimension
                            else:
                                # Fallback: use the last two dimensions as D x D
                                chol = chol.reshape(-1, inferred_D, inferred_D)
                                if chol.shape[0] != B:
                                    chol = chol[:B] if chol.shape[0] > B else chol.expand(B, -1, -1)
                                D = inferred_D
                        else:
                            raise RuntimeError(f"Cannot reshape chol tensor of shape {chol.shape} to [B={B}, D={D}, D={D}]. Total elements: {total_elements}, expected: {B * D * D}")
                elif chol.shape[0] != B:
                    # Handle batch size mismatch
                    if chol.shape[0] == 1:
                        chol = chol.expand(B, -1, -1)
                    else:
                        chol = chol[:B]  # Truncate or pad as needed
                
                eps = torch.randn(B, K, D, device=mu.device, dtype=chol.dtype)
                
                # Use batch matrix multiplication for efficiency
                # chol: [B, D, D], eps: [B, K, D]
                # We want: chol @ eps.transpose(-1, -2) -> [B, D, K]
                # Then transpose to get [B, K, D]
                transformed = torch.bmm(chol, eps.transpose(-1, -2)).transpose(-1, -2)  # [B, K, D]
                z_cand = mu.float().unsqueeze(1) + transformed
                # Evaluate 0.5 log|G^{-1}| at candidates and pick best per batch
                with torch.enable_grad():
                    z_eval = z_cand.reshape(B * K, D).detach().requires_grad_(False)
                    Gz = self._ctx['model'].G(z_eval)
                    from .metric_utils import half_logdet_volume
                    h = half_logdet_volume(Gz, 'g', jitter=self.eps_reg).reshape(B, K)  # 0.5 log|G^{-1}|
                best_idx = torch.argmax(h, dim=1)
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
                z0 = self._initial_accept_volume(z0, mu)
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
    
    def _make_covariance(self, G_inv: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Build SPD covariance Σ = α·Ĝ^{-1} + εI with eigenvalue clipping and
        optional normalization so that initial draws have a predictable radius.
        """
        Ginv = _symmetrize(G_inv)
        if Ginv.ndim == 2:
            Ginv = Ginv.unsqueeze(0)
        d = Ginv.shape[-1]
        # Clamp spectrum for robustness
        try:
            evals, evecs = torch.linalg.eigh(Ginv.float())
            evals = torch.clamp(evals, min=self.min_cov_eig)
            # Optional normalization by geometric mean so det(Ginv_norm)=1
            mode = str(getattr(self, 'sigma_normalization_mode', 'geomean')).lower()
            if mode == 'geomean':
                gm = torch.exp(torch.log(torch.clamp(evals, min=1e-12)).mean(dim=-1, keepdim=True))
                evals = evals / gm
            elif mode == 'trace':
                tr = evals.sum(dim=-1, keepdim=True)
                evals = d * evals / torch.clamp(tr, min=1e-12)
            # Recompose normalized precision
            Ginv_norm = (evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)).to(Ginv.dtype)
        except Exception:
            Ginv_norm = Ginv

        eye = torch.eye(d, device=Ginv.device, dtype=Ginv.dtype).unsqueeze(0)

        # Adapt alpha to hit a target Euclidean radius for initial draws
        target_r = float(getattr(self, 'initial_target_radius', 1.0))
        if target_r > 0:
            try:
                tr_ginv = torch.einsum('bii->b', Ginv_norm.float()).unsqueeze(-1)
                # E||δ||^2 ≈ trace(Sigma) = alpha*trace(Ginv_norm)+ d*eps
                alpha_eff = ((target_r ** 2) - d * self.eps_reg) / torch.clamp(tr_ginv, min=1e-12)
                # Keep batch-shapes aligned
                alpha_eff = alpha_eff.clamp(min=1e-6).to(Ginv_norm.dtype)
                Sigma = alpha_eff.unsqueeze(-1).unsqueeze(-1) * Ginv_norm + self.eps_reg * eye
            except Exception:
                Sigma = alpha * Ginv_norm + self.eps_reg * eye
        else:
            Sigma = alpha * Ginv_norm + self.eps_reg * eye

        return self._stabilize_spd(_symmetrize(Sigma), self.min_cov_eig)

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

    def _initial_accept_volume(self, z0: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        try:
            tol = float(getattr(self, 'initial_volume_tolerance', 0.0))
            kmax = int(getattr(self, 'initial_max_retries', 3))
            if tol <= 0 and kmax <= 0:
                return z0
            from .metric_utils import half_logdet_volume
            G_mu = self._ctx['model'].G(mu)
            h_mu = half_logdet_volume(G_mu, 'g', jitter=self.eps_reg)  # 0.5 log|G^{-1}(mu)|
            z = z0
            for _ in range(max(0, kmax)):
                z_req = z.clone().detach().requires_grad_(True)
                Gz = self._ctx['model'].G(z_req)
                hz = half_logdet_volume(Gz, 'g', jitter=self.eps_reg)
                if (hz >= h_mu - tol).all():
                    return z_req.detach()
                grad_h, = torch.autograd.grad(hz.sum(), z_req, retain_graph=False, create_graph=False, allow_unused=True)
                if grad_h is None:
                    break
                step = float(getattr(self, 'projection_step_scale', 0.05))
                z = (z_req + step * grad_h).detach()
            return z
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
            ceil = float(getattr(self, "metric_eig_ceiling", 1e6))
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

    def _rhmc_exploration(
        self,
        z0: torch.Tensor,
        record_traj: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
        """
        Step 2: Simple RHMC exploration without acceptance/rejection.
        """
        z = z0.clone()
        
        # Sample initial momentum
        rho = self._sample_momentum(z)
        
        # Simple leapfrog integration
        if record_traj:
            traj = [{'step': 0, 'z': z.clone(), 'rho': rho.clone()}]
        else:
            traj = None

        for step in range(self.rhmc_steps):
            z, rho = self._leapfrog_step(z, rho, self.rhmc_step_size)
            if record_traj:
                traj.append({'step': step + 1, 'z': z.clone(), 'rho': rho.clone()})

        return z, traj
    
    def _sample_momentum(self, z: torch.Tensor) -> torch.Tensor:
        """
        Simple momentum sampling: ρ ~ N(0, G(z))
        """
        try:
            G = self._ctx['model'].G(z)
            G = _symmetrize(G)
            L, _ = _safe_cholesky(G + self.eps_reg * torch.eye(z.shape[-1], device=z.device, dtype=G.dtype).unsqueeze(0), self.eps_reg)
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
                    s = torch.einsum('bi,bij,bj->b', rho, G_inv_k, rho)
                    kin_grad = torch.autograd.grad(s.sum(), z_req, retain_graph=True, create_graph=False, allow_unused=True)[0]
                    if kin_grad is None:
                        kin_grad = torch.zeros_like(z_req)
                    grad_U = grad_U + self.kinetic_grad_weight * kin_grad
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
            velocity = torch.einsum('bij,bj->bi', G_inv_reg, rho)
            # Guard against non-finite velocity components
            velocity = torch.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
            # Clip velocity
            v_norm = torch.norm(velocity, dim=-1, keepdim=True)
            velocity = torch.where(v_norm > self.max_velocity_norm, velocity * (self.max_velocity_norm / (v_norm + 1e-12)), velocity)
            # Adaptive effective step to bound position change
            eff_step = torch.clamp(self.max_position_step / (v_norm + 1e-12), max=1.0) * step_size
            # Geometry-aware scaling by stiffness: scale by sqrt(min_eig(G^{-1})) = 1/sqrt(lambda_max(G))
            try:
                evals_Ginv = torch.linalg.eigvalsh(G_inv_reg.float())
                min_eig_Ginv = evals_Ginv.min(dim=-1, keepdim=True).values.to(z.dtype)
                stiff_scale = torch.sqrt(torch.clamp(min_eig_Ginv, min=1e-6))
                eff_step = eff_step * stiff_scale
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
                    scale = torch.tanh(z_norm_soft / soft_limit)
                    rho = rho * scale
            # Optional single-step projection toward high log|G^{-1}|
            try:
                tau = float(self.projection_step_scale) * float(step_size)
                if tau > 0:
                    z_req = z.clone().detach().requires_grad_(True)
                    # grad log|G^{-1}| = -2 * grad(half_logdet_volume(G, 'g'))
                    G_proj = self._ctx['model'].G(z_req)
                    from .metric_utils import half_logdet_volume
                    vol_g = half_logdet_volume(G_proj, 'g', jitter=self.eps_reg)
                    grad_vol_g = torch.autograd.grad(vol_g.sum(), z_req, retain_graph=False, create_graph=False, allow_unused=True)[0]
                    if grad_vol_g is not None:
                        grad_logdet_ginv = -2.0 * grad_vol_g
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
                z = torch.where(z_norm > norm_limit, z * (norm_limit / (z_norm + 1e-12)), z)
            
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
            
            return z, rho
            
        except Exception as e:
            print(f"⚠️ Leapfrog step failed: {e}")
            return z, rho
    
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
            else:
                alpha = alpha_override if alpha_override is not None else self._resolve_alpha()
                G_inv_mu = self._get_inverse_metric(mu)
                Sigma = self._make_covariance(G_inv_mu, alpha)
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
        Compute log p(z) for Riemannian volume prior expressed via G.
        
        p(z) ∝ √det(G(z)) · exp(-0.5 * zᵀ G(z) z)
        
        With proper normalization:
        log p(z) = 0.5 * log det(G(z)) - 0.5 * zᵀ G(z) z - 0.5*d*log(2π)
        
        Args:
            z: Latent samples [B, D]
            
        Returns:
            log_p: Log prior probability [B]
        """
        try:
            model = self._ctx['model']
            if hasattr(model, 'G'):
                G_z = model.G(z)
            elif hasattr(model, 'metric_tensor'):
                mt = model.metric_tensor
                if hasattr(mt, 'compute_metric'):
                    G_z = mt.compute_metric(z)
                elif hasattr(mt, 'compute_inverse_metric'):
                    G_inv_z = mt.compute_inverse_metric(z)
                    G_z = torch.linalg.inv(G_inv_z)
                else:
                    raise RuntimeError("Metric tensor does not provide G(z).")
            elif hasattr(model, 'G_inv'):
                G_inv_z = model.G_inv(z)
                G_z = torch.linalg.inv(G_inv_z)
            else:
                raise RuntimeError("Model does not expose G(z) or G^{-1}(z).")            # AMP-safe logdet
            G_z32 = G_z.float() if G_z.dtype in (torch.float16, torch.bfloat16) else G_z
            half_logdet = half_logdet_volume(G_z32, 'g', jitter=self.eps_reg)

            d = z.shape[-1]
            G_z_cast = G_z32.to(z.dtype)
            quad = torch.einsum('bi,bij,bj->b', z, G_z_cast, z)
            log_det_term = (-half_logdet * self.volume_bias_weight).to(z.dtype)
            gaussian_term = -0.5 * quad.to(z.dtype) - 0.5 * float(d) * math.log(2 * math.pi)

            mode = getattr(self, 'kl_prior_mode', 'volume_gaussian')
            if mode in ('volume_gaussian', 'gaussian'):
                log_p = log_det_term + gaussian_term
            elif mode == 'uniform':
                log_p = log_det_term + float(self.uniform_prior_log_norm)
            else:
                # Fallback: behave like volume_gaussian for unknown modes
                log_p = log_det_term + gaussian_term
            
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
        ∇U(z) with volume correction designed to attract to HIGH log|G^{-1}| regions.
        We minimize: U(z) = 0.5||z||^2 - w · 0.5 log det G^{-1}(z) = 0.5||z||^2 + w · 0.5 log det G(z).

        Robust to cases where G does not depend on z (allow_unused=True).
        Falls back to zero volume-gradient in that case.
        """
        # If autograd is disabled (e.g., validation), use base gradient only
        if not torch.is_grad_enabled():
            return z
        # Ensure we can take grads w.r.t. z
        if not z.requires_grad:
            z = z.clone().requires_grad_(True)

        # Base grad of 0.5 ||z||^2 is simply z
        base = z

        # Volume term: attract to high log|G^{-1}|. Two equivalent ways:
        #  - using G:   +∇(0.5 log|G|) with positive sign in grad_U and descent will reduce log|G|
        #  - using G⁻¹: -∇(0.5 log|G⁻¹|).
        #  Controlled by self.volume_force_representation + self.volume_force_sign for quick A/B.
        try:
            rep = getattr(self, 'volume_force_representation', 'g')
            rep = rep if rep in ('g', 'ginv') else 'g'
            if rep == 'g':
                G = self._ctx['model'].G(z)
                G32 = G.float() if G.dtype in (torch.float16, torch.bfloat16) else G
                target = -half_logdet_volume(G32, 'g', jitter=self.eps_reg)  # +0.5 log|G|
                grad_vol, = torch.autograd.grad(target.sum(), z, retain_graph=True, create_graph=True, allow_unused=True)
            else:
                Ginv = self._get_inverse_metric(z)
                Ginv32 = Ginv.float() if Ginv.dtype in (torch.float16, torch.bfloat16) else Ginv
                target = half_logdet_volume(Ginv32, 'ginv', jitter=self.eps_reg)  # +0.5 log|G^{-1}|
                # We want to minimize -0.5 log|G^{-1}|, so grad contribution is -∇target
                grad_vol_raw, = torch.autograd.grad(target.sum(), z, retain_graph=True, create_graph=True, allow_unused=True)
                grad_vol = -grad_vol_raw
            if grad_vol is None:
                grad_vol = torch.zeros_like(z)
            sign = float(getattr(self, 'volume_force_sign', 1.0))
            scale = sign * float(getattr(self, 'volume_grad_scale', 1.0)) * float(getattr(self, 'volume_bias_weight', 1.0))
            grad = base + scale * grad_vol
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            return grad.to(z.dtype)
        except Exception:
            # Safe fallback: standard Gaussian gradient only
            return torch.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    
    def snapshot_state(self) -> Dict[str, Any]:
        """Capture mutable sampler state for safe restoration."""
        state = {
            'rhmc_steps': int(getattr(self, 'rhmc_steps', 0)),
            'rhmc_step_size': float(getattr(self, 'rhmc_step_size', 0.0)),
            'rhmc_alpha': float(getattr(self, 'rhmc_alpha', 0.0)),
            'eps_reg': float(getattr(self, 'eps_reg', 0.0)),
            'min_cov_eig': float(getattr(self, 'min_cov_eig', 0.0)),
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
