"""
Posterior Sampler Component
===========================

Implements the local metric–aligned Gaussian posterior used by the original
RiemannianFlowVAE, factorized for modular reuse.

Key behavior:
- Uses covariance Σ = α G(μ) with α from the model's ramp schedule
- Cholesky sampling with numerical regularization
- Preserves gradients in training; detaches in eval/no_grad
"""

from typing import Optional
import torch
import torch.nn as nn


class PosteriorSampler(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        import weakref
        self._ctx = {'model': weakref.proxy(model)}
        self.device = getattr(model, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # TRACE-once guard
        self._trace_printed = False

    def sample_metric_aware_posterior(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Local metric–aligned Gaussian (reparameterized) posterior sampling.

        Posterior: q(z|x) ∝ √det(G^{-1}(z)) exp(-1/2 (z-μ)ᵀ G^{-1}(z) (z-μ))

        Sampling uses Σ = α G(μ): z = μ + L ε, with L Lᵀ = Σ.
        """
        # Fallback if metric is unavailable
        if not hasattr(self._ctx['model'], 'G'):
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(0.5 * log_var)

        # Get current alpha with optional ramping
        current_epoch = getattr(self._ctx['model'], '_current_epoch', None)
        alpha = self._get_current_alpha(current_epoch)

        mu_f32 = mu.float()
        batch_size, latent_dim = mu_f32.shape

        # Compute covariance at μ: Σ = α · G_inv(μ) + εI (Riemannian Gaussian has precision G)
        if hasattr(self._ctx['model'], 'G_inv'):
            G_inv_mu = self._ctx['model'].G_inv(mu_f32)
        else:
            G_mu = self._ctx['model'].G(mu_f32)
            G_inv_mu = torch.linalg.inv(G_mu.float()).to(G_mu.dtype)

        # Optional normalization of covariance metric for sampling to prevent
        # excessively large steps when G has very large eigenvalues.
        cov_norm_mode = getattr(self._ctx['model'], 'posterior_cov_norm_mode', 'none')
        if cov_norm_mode and cov_norm_mode != 'none':
            d = G_inv_mu.shape[-1]
            Ginv_f32 = G_inv_mu.float()
            if cov_norm_mode == 'geomean':
                sign, logabsdet = torch.slogdet(Ginv_f32)
                s = torch.exp(logabsdet / d).unsqueeze(-1).unsqueeze(-1)
                G_inv_mu = G_inv_mu / (s.to(G_inv_mu.dtype) + 1e-12)
            elif cov_norm_mode == 'trace':
                s = (torch.einsum('bii->b', Ginv_f32) / d).unsqueeze(-1).unsqueeze(-1)
                G_inv_mu = G_inv_mu / (s.to(G_inv_mu.dtype) + 1e-12)

        # Sample ε ~ N(0, I)
        eps = torch.randn_like(mu_f32)
        # Clamp ε magnitude to suppress rare 4–5σ excursions in high‑D
        try:
            clip_scale = float(getattr(self._ctx['model'], 'posterior_step_clip_scale', 3.0))
            max_norm = clip_scale * (float(mu_f32.shape[1]) ** 0.5)
            eps_norm = torch.norm(eps, dim=1, keepdim=True) + 1e-12
            scale = torch.clamp(max_norm / eps_norm, max=1.0)
            eps = eps * scale
        except Exception:
            pass

        # Numerical regularization
        I = torch.eye(latent_dim, device=G_inv_mu.device, dtype=G_inv_mu.dtype).unsqueeze(0)
        eps_chol = getattr(self._ctx['model'], 'eps_chol', 1e-6)
        Sigma = alpha * G_inv_mu + eps_chol * I

        # TRACE: First-batch diagnostics (dtype, norms, spectrum)
        try:
            import os
            if os.environ.get('RLVAE_TRACE', '0') == '1' and not self._trace_printed:
                # dtypes
                mu_dt = str(mu.dtype)
                G_dt = str(G_inv_mu.dtype)
                S_dt = str(Sigma.dtype)
                # compute a cholesky in float32 to probe dtype safely
                chol_probe = torch.linalg.cholesky(Sigma.float())
                chol_dt = str(chol_probe.dtype)
                # eigen stats (float32 for stability)
                evals = torch.linalg.eigvalsh(Sigma.float())
                eigmin = evals.min().item()
                eigmed = evals.median().item()
                cond = (evals.max() / (evals.min().clamp_min(1e-12))).item()
                logdet_S = torch.logdet(Sigma.float()).median().item()
                # sample stats (we’ll compute z right after L exists)
                print(f"TRACE RHMC dtype: mu={mu_dt}, Ginv={G_dt}, Sigma={S_dt}, chol={chol_dt}")
                print(f"TRACE RHMC Σ(μ): eigmin={eigmin:.3e}, median={eigmed:.3e}, cond={cond:.3e}, logdet_med={logdet_S:.3e}")
                print(f"TRACE RHMC alpha, eps: alpha={float(alpha):.6g}, eps_chol={float(eps_chol):.3g}, autocast={torch.is_autocast_enabled()}")
        except Exception:
            pass

        try:
            L = torch.linalg.cholesky(Sigma)  # [B, D, D]
            # Preserve gradient path in training mode
            if self._ctx['model'].training and torch.is_grad_enabled():
                if not L.requires_grad:
                    L = L.detach().requires_grad_(True)
                if not eps.requires_grad:
                    eps = eps.detach().requires_grad_(True)
            else:
                L = L.detach()
                eps = eps.detach()

            z = mu_f32 + torch.einsum('bij,bj->bi', L, eps)
            try:
                import os
                if os.environ.get('RLVAE_TRACE', '0') == '1' and not self._trace_printed:
                    # Norm stats of z0
                    z0 = z
                    z0_norm = torch.norm(z0, dim=1)
                    print(f"TRACE RHMC z0,zK norms: ||z0|| mean={z0_norm.mean().item():.4g} std={z0_norm.std().item():.4g}; (no RHMC here)")
                    self._trace_printed = True
            except Exception:
                pass
            # Optional Mahalanobis clamp using local precision at μ
            try:
                r2 = float(getattr(self._ctx['model'], 'posterior_maha_clip', 0.0))
                if r2 is not None and r2 > 0.0:
                    G_inv_mu = self._ctx['model'].G_inv(mu_f32)
                    diff = z - mu_f32
                    d2 = torch.einsum('bi,bij,bj->b', diff, G_inv_mu.float(), diff)
                    scale = torch.ones_like(d2)
                    over = d2 > r2
                    # scale diff so that new d2 equals r2
                    scale[over] = torch.sqrt(r2 / (d2[over] + 1e-12))
                    z = mu_f32 + diff * scale.unsqueeze(-1)
            except Exception:
                pass
            return z.to(mu.dtype)
        except Exception as e:
            # STRICT: optionally disallow fallback to standard
            try:
                import os
                strict = bool(getattr(self._ctx['model'], 'riemannian_strict', False) or os.environ.get('RLVAE_STRICT', '0') == '1')
            except Exception:
                strict = False
            if strict:
                raise RuntimeError(f"PosteriorSampler: Riemannian sampling failed under strict mode: {e}")
            # Fallback to standard Gaussian sampling (non-strict only)
            eps = torch.randn_like(mu)
            try:
                import os
                if os.environ.get('RLVAE_TRACE', '0') == '1':
                    print(f"⚠️ Riemannian sampling failed in PosteriorSampler: {e}; using STANDARD reparam")
            except Exception:
                pass
            return mu + eps * torch.exp(0.5 * log_var)

    def _get_current_alpha(self, current_epoch: Optional[int]) -> float:
        # Prefer model's schedule when available
        mdl = self._ctx['model'] if hasattr(self, '_ctx') else getattr(self, 'model', None)
        if mdl is not None and hasattr(mdl, 'get_current_posterior_alpha'):
            try:
                return float(mdl.get_current_posterior_alpha(current_epoch))
            except Exception:
                pass
        # Fallback to model.posterior_local_alpha
        alpha = getattr(mdl, 'posterior_local_alpha', 0.5) if mdl is not None else 0.5
        try:
            return float(alpha)
        except Exception:
            return 0.5
