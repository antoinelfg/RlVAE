"""
Baseline Riemannian RHMC Posterior - Simplified Version
======================================================

Minimal implementation without complex constraints or stability checks.
"""
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


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
    Sigma = _symmetrize(Sigma)
    chol, stabilized_sigma = _safe_cholesky(Sigma, min_eig)
    diff = (z - mu).unsqueeze(-1)
    diff32 = diff.float() if diff.dtype != chol.dtype else diff
    sol32 = torch.cholesky_solve(diff32, chol)
    quad_form = torch.einsum('bij,bij->b', diff32, sol32)
    log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-18).sum(dim=-1)
    const = 0.5 * mu.shape[-1] * math.log(2 * math.pi)
    out = -0.5 * quad_form - 0.5 * log_det - const
    return out.to(mu.dtype)


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
        self.rhmc_alpha = float(_cfg_get('rhmc_alpha', 1e-2))
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
        # If grad is globally disabled (e.g., during validation), fall back to base potential only
        if not torch.is_grad_enabled() and not self._warned_no_grad:
            print("[RHMC WARN] Grad disabled; using base potential (no volume correction).")
            self._warned_no_grad = True
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
        # Allow per-call override of epsilon regularization
        eps_backup = self.eps_reg
        if eps_reg is not None:
            try:
                self.eps_reg = float(eps_reg)
            except Exception:
                pass
        z0, Sigma_mu = self._sample_initial_riemannian(mu, log_var, alpha_eff)

        # Step 2: RHMC exploration (differentiable pushforward)
        z_final, traj_states = self._rhmc_exploration(z0, record_traj=traj_flag)

        # TRACE: invariants and dtype diagnostics (first batch only)
        try:
            import os
            if os.environ.get('RLVAE_TRACE', '0') == '1' and not self._trace_printed:
                z0_norm = torch.norm(z0, dim=1)
                zK_norm = torch.norm(z_final, dim=1)
                max_diff = (z_final - z0).abs().max().item()
                # eig diagnostics on Sigma
                evals = torch.linalg.eigvalsh(Sigma_mu.float())
                eigmin = evals.min().item()
                eigmed = evals.median().item()
                cond = (evals.max() / evals.min().clamp_min(1e-12)).item()
                logdet_S = torch.logdet(Sigma_mu.float()).median().item()
                # dtype trace
                mu_dt = str(mu.dtype)
                # Try computing G_inv(mu) for dtype
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
                'jac_logdet': None if jac_flag else None,
                'trajectory': traj_states,
            }

        # Assemble return tuple following (zK, log_q?, z0?, traj_info?)
        outputs = [z_final]
        if log_prob_flag:
            outputs.append(log_q)
        if initial_flag:
            outputs.append(z0)
        if traj_flag:
            outputs.append(traj_info)

        # Restore epsilon regularization
        self.eps_reg = eps_backup

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
    
    def _sample_initial_riemannian(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        alpha: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step 1: Sample z₀ ~ N_Riem(μ, Σ(μ)) with Σ = α·Ĝ⁻¹(μ) + εI.
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
                # 2) Sample two independent noises
                xi1 = torch.randn_like(mu, dtype=C.dtype)
                xi2 = torch.randn_like(mu, dtype=mu.dtype)
                # 3) Solve C^T y = xi1  => y = C^{-T} xi1
                try:
                    y = torch.linalg.solve_triangular(C.transpose(-1, -2), xi1, upper=True)
                except Exception:
                    # Fallback to generic solver
                    y = torch.linalg.solve(C.transpose(-1, -2), xi1)
                # 4) Compose z0
                z032 = mu.float() + (alpha ** 0.5) * y + (self.eps_reg ** 0.5) * xi2.float()
                z0 = z032.to(mu.dtype)
                # 5) Prepare Σ for return (for log q computations)
                # Compute G^{-1}(μ) in float32 for stability
                G_inv_mu = torch.linalg.inv(G_mu.float())
                Sigma = self._make_covariance(G_inv_mu.to(mu.dtype), alpha)
                return z0, Sigma
            else:
                # Standard covariance-based path
                G_inv_mu = self._get_inverse_metric(mu)
                Sigma = self._make_covariance(G_inv_mu, alpha)
                chol, Sigma = _safe_cholesky(Sigma, self.min_cov_eig)
                eps = torch.randn_like(mu, dtype=chol.dtype)
                z032 = mu.float() + torch.bmm(chol, eps.unsqueeze(-1)).squeeze(-1)
                z0 = z032.to(mu.dtype)
                return z0, Sigma
        except Exception as exc:
            import os
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
        Build SPD covariance Σ = α·Ĝ⁻¹ + εI with eigenvalue clipping for stability.
        """
        Sigma = alpha * _symmetrize(G_inv)
        d = Sigma.shape[-1]
        eye = torch.eye(d, device=Sigma.device, dtype=Sigma.dtype).unsqueeze(0)
        Sigma = Sigma + self.eps_reg * eye
        return self._stabilize_spd(Sigma, self.min_cov_eig)

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
                        import os
                        if os.environ.get('RLVAE_TRACE', '0') == '1' and not self._trace_printed:
                            print(f"TRACE OVERRIDE α source=model.rhmc_alpha -> {alpha}")
                    except Exception:
                        pass
                    return alpha
            except Exception:
                pass
        current_epoch = getattr(model, '_current_epoch', None)
        if hasattr(model, 'get_current_posterior_alpha'):
            try:
                alpha = float(model.get_current_posterior_alpha(current_epoch))
                if math.isfinite(alpha) and alpha > 0:
                    try:
                        import os
                        if os.environ.get('RLVAE_TRACE', '0') == '1' and not self._trace_printed:
                            print(f"TRACE OVERRIDE α source=get_current_posterior_alpha -> {alpha}")
                    except Exception:
                        pass
                    return alpha
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
                        import os
                        if os.environ.get('RLVAE_TRACE', '0') == '1' and not self._trace_printed:
                            print(f"TRACE OVERRIDE α source=config.rhmc_alpha -> {alpha}")
                    except Exception:
                        pass
                    return alpha
            except Exception:
                pass
        return max(self.rhmc_alpha, 1e-6)

    def _get_inverse_metric(self, pts: torch.Tensor) -> torch.Tensor:
        """Fetch Ĝ⁻¹(pts) with symmetry and fallback safeguards."""
        model = self._ctx['model']
        if hasattr(model, 'G_inv'):
            G_inv = model.G_inv(pts)
        elif hasattr(model, 'G'):
            G = model.G(pts)
            G_inv = torch.linalg.inv(_symmetrize(G))
        else:
            raise AttributeError("Model must expose G_inv or G to compute RHMC posterior.")
        return _symmetrize(G_inv)

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
            # Momentum clipping for safety
            rho_norm = torch.norm(rho, dim=-1, keepdim=True)
            rho = torch.where(rho_norm > self.max_momentum_norm, rho * (self.max_momentum_norm / (rho_norm + 1e-12)), rho)
            return rho
        except:
            # Fallback to isotropic sampling
            rho = torch.randn_like(z)
            rho_norm = torch.norm(rho, dim=-1, keepdim=True)
            rho = torch.where(rho_norm > self.max_momentum_norm, rho * (self.max_momentum_norm / (rho_norm + 1e-12)), rho)
            return rho
    
    def _leapfrog_step(self, z: torch.Tensor, rho: torch.Tensor, step_size: float) -> tuple:
        """
        Simple leapfrog integration step (approximate RMHMC).

        Notes:
        - Omits the kinetic position‑dependence term −0.5 ∇_z [ρᵀ G⁻¹(z) ρ] for
          simplicity and stability; use small steps and clipping.
        - Jacobian of the integrator is ignored (KL jacobian mode is a placeholder).
        """
        try:
            # Ensure z requires grad to build autograd graph inside integrator
            if not z.requires_grad:
                z = z.clone().requires_grad_(True)
            # Half step for momentum
            grad_U = self._compute_potential_gradient(z)
            rho = rho - 0.5 * step_size * grad_U
            
            # Full step for position
            G_inv = self._get_inverse_metric(z)
            # Regularize inverse metric for numerical safety
            d = z.shape[-1]
            I = torch.eye(d, device=z.device, dtype=G_inv.dtype).unsqueeze(0)
            G_inv_reg = G_inv + self.eps_reg * I
            velocity = torch.einsum('bij,bj->bi', G_inv_reg, rho)
            # Clip velocity
            v_norm = torch.norm(velocity, dim=-1, keepdim=True)
            velocity = torch.where(v_norm > self.max_velocity_norm, velocity * (self.max_velocity_norm / (v_norm + 1e-12)), velocity)
            # Adaptive effective step to bound position change
            eff_step = torch.clamp(self.max_position_step / (v_norm + 1e-12), max=1.0) * step_size
            z = z + eff_step * velocity
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
            rho = rho - 0.5 * step_size * grad_U
            
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
            import os
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
        
        p(z) ∝ √det(G(z)) · exp(-0.5 * zᵀ z)
        
        With proper normalization:
        log p(z) = 0.5 * log det(G(z)) - 0.5 * ||z||² - 0.5*d*log(2π)
        
        Args:
            z: Latent samples [B, D]
            
        Returns:
            log_p: Log prior probability [B]
        """
        try:
            # model.G returns the metric tensor G(z)
            G_z = self._ctx['model'].G(z)
            # AMP‑safe logdet
            G_z32 = G_z.float() if G_z.dtype in (torch.float16, torch.bfloat16) else G_z
            sign, log_det_G = torch.slogdet(G_z32)
            
            # Volume term + Gaussian prior + normalization
            d = z.shape[-1]
            z_norm_sq = torch.sum(z ** 2, dim=-1)
            log_p = 0.5 * log_det_G - 0.5 * z_norm_sq - 0.5 * d * math.log(2 * math.pi)
            
            return log_p
            
        except Exception as e:
            # Fallback: standard Gaussian prior (no volume term)
            d = z.shape[-1]
            z_norm_sq = torch.sum(z ** 2, dim=-1)
            return -0.5 * z_norm_sq - 0.5 * d * math.log(2 * math.pi)
    
    def _compute_potential_gradient(self, z: torch.Tensor) -> torch.Tensor:
        """
        ∇U(z) with volume correction: U(z) = 0.5 ||z||^2 - 0.5 log det G(z)

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

        # Volume term: ∇_z [ -0.5 log det G(z) ]
        try:
            G = self._ctx['model'].G(z)
            G32 = G.float() if G.dtype in (torch.float16, torch.bfloat16) else G
            sign, logdet = torch.slogdet(G32)
            vol_term = -0.5 * logdet
            grad_vol, = torch.autograd.grad(
                outputs=vol_term.sum(),
                inputs=z,
                retain_graph=True,        # used twice per step
                create_graph=True,        # higher-order safe
                allow_unused=True         # if G is constant wrt z
            )
            if grad_vol is None:
                grad_vol = torch.zeros_like(z)
            grad = base + grad_vol
            return grad.to(z.dtype)
        except Exception:
            # Safe fallback: standard Gaussian gradient only
            return base
    
    def get_config(self) -> Dict[str, Any]:
        """Return current configuration."""
        return {
            'rhmc_steps': self.rhmc_steps,
            'rhmc_step_size': self.rhmc_step_size,
            'rhmc_alpha': self.rhmc_alpha,
            'eps_regularization': self.eps_reg
        }
