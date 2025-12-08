import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ..models.components.metric_utils import half_logdet_volume


def _subset(x: torch.Tensor, n: int) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        return x
    if x.ndim == 0:
        return x
    if x.shape[0] <= n:
        return x
    return x[:n]


def tensor_stats(name: str, t: torch.Tensor) -> str:
    if not isinstance(t, torch.Tensor):
        return f"{name}: <non-tensor>"
    t = t.detach()
    return (
        f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"mean={t.mean().item():+.4e} std={t.std(unbiased=False).item():.4e} "
        f"min={t.min().item():+.4e} max={t.max().item():+.4e}"
    )


def volume_grad_sanity(
    model,
    z: torch.Tensor,
    *,
    rep: str = "g",
    sign: float = 1.0,
    jitter: float = 1e-6,
    eig_floor: float = 0.0,
    label: str = "",
    max_items: int = 8,
) -> None:
    """Deep diagnostic for ∇(½ log|G^{-1}|) at given points.

    Prints:
    - Half–logdet from 'g' and 'ginv' routes
    - Autograd gradient norms and cosine between routes
    - Finite-difference projection along the chosen gradient direction
    - Expected descent direction for U(z) = -sign * ½ log|G^{-1}(z)|
    """
    if os.environ.get("RLVAE_GRAD_TRACE", "0") != "1":
        return

    try:
        z_req = _subset(z.detach(), int(os.environ.get("RLVAE_DEBUG_N", "8"))).clone().requires_grad_(True)
        rep = rep if rep in ("g", "ginv") else "g"

        # Build SPD matrices in float32
        if rep == "g":
            G = model.G(z_req)
            G32 = 0.5 * (G + G.transpose(-1, -2))
            try:
                e, V = torch.linalg.eigh(G32.float())
                if eig_floor > 0:
                    e = torch.clamp(e, min=float(eig_floor))
                G32 = V @ (e.unsqueeze(-1) * V.transpose(-1, -2))
            except Exception:
                pass
            half = half_logdet_volume(G32, 'g', jitter=float(jitter))
        else:
            Ginv = model.G_inv(z_req)
            Ginv32 = 0.5 * (Ginv + Ginv.transpose(-1, -2))
            try:
                e, V = torch.linalg.eigh(Ginv32.float())
                if eig_floor > 0:
                    e = torch.clamp(e, min=float(eig_floor))
                Ginv32 = V @ (e.unsqueeze(-1) * V.transpose(-1, -2))
            except Exception:
                pass
            half = half_logdet_volume(Ginv32, 'ginv', jitter=float(jitter))

        grad, = torch.autograd.grad(half.sum(), z_req, retain_graph=False, create_graph=False, allow_unused=True)
        if grad is None:
            grad = torch.zeros_like(z_req)

        # Cross-check via other route
        try:
            if rep == "g":
                Ginv = model.G_inv(z_req)
                half2 = half_logdet_volume(Ginv.float(), 'ginv', jitter=float(jitter))
            else:
                G = model.G(z_req)
                half2 = half_logdet_volume(G.float(), 'g', jitter=float(jitter))
            # finite-difference sanity on grad direction
            dir_unit = F.normalize(grad.detach(), dim=-1, eps=1e-12)
            eps_fd = float(os.environ.get("RLVAE_FORCE_EPS", "1e-4"))
            with torch.no_grad():
                plus = half_logdet_volume(
                    (model.G(z_req + eps_fd * dir_unit).float() if rep == 'g' else model.G_inv(z_req + eps_fd * dir_unit).float()),
                    rep,
                    jitter=float(jitter),
                )
                minus = half_logdet_volume(
                    (model.G(z_req - eps_fd * dir_unit).float() if rep == 'g' else model.G_inv(z_req - eps_fd * dir_unit).float()),
                    rep,
                    jitter=float(jitter),
                )
                fd_proj = (plus - minus) / (2.0 * eps_fd)
        except Exception:
            half2 = None
            fd_proj = None

        print("\n[GRAD TRACE] volume_grad_sanity", (f"({label})" if label else ""))
        print(tensor_stats("z", z_req))
        print(tensor_stats("0.5 log|G^{-1}| (route rep)", half))
        if half2 is not None:
            print(tensor_stats("0.5 log|G^{-1}| (alt route)", half2))
            print(f"  mean|Δ| (routes): {(half2 - half).abs().mean().item():.3e}")
        print(tensor_stats("grad 0.5 log|G^{-1}|", grad))
        if fd_proj is not None:
            print(tensor_stats("finite-diff proj along grad", fd_proj))
        # Expected effect on U = -sign * half
        if grad.numel() > 0:
            grad_norm = grad.norm(dim=-1).mean().item()
        else:
            grad_norm = 0.0
        print(f"  sign={sign:+.1f}, expected dU ~ {-sign:.1f} * d(half)")
        print(f"  ||grad|| mean={grad_norm:.3e}")
    except Exception as e:
        print(f"[GRAD TRACE] volume_grad_sanity failed: {e}")


def kl_grad_sanity(
    *,
    mu: torch.Tensor,
    zS: Optional[torch.Tensor],
    Sigma_mu: Optional[torch.Tensor],
    log_q: Optional[torch.Tensor],
    log_p_half_ginv: Optional[torch.Tensor],
    loss_manager,
    metric_tensor,
    rhmc_posterior,
    max_items: int = 4,
):
    if os.environ.get("RLVAE_GRAD_TRACE", "0") != "1":
        return
    try:
        n = int(os.environ.get("RLVAE_DEBUG_N", str(max_items)))
        mu_req = _subset(mu.detach(), n).clone().requires_grad_(True)
        if zS is not None:
            z_req = _subset(zS.detach(), n).clone().requires_grad_(True)
        else:
            z_req = _subset(mu.detach(), n).clone().requires_grad_(True)

        # Rebuild Σ_μ if needed
        Sigma = None
        if Sigma_mu is not None:
            Sigma = _subset(Sigma_mu.detach(), n).clone().requires_grad_(False)
        else:
            Sigma = loss_manager._resolve_sigma_mu(mu_req, None, metric_tensor, rhmc_posterior, None)

        # log_q gradients w.r.t. z and μ
        if Sigma is not None:
            try:
                chol = torch.linalg.cholesky(Sigma.float())
                diff = (z_req - mu_req).unsqueeze(-1)
                sol = torch.cholesky_solve(diff.float(), chol)
                quad = torch.einsum('bij,bij->b', diff.float(), sol)
                logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-18).sum(dim=-1)
                const = z_req.shape[-1] * torch.log(torch.tensor(2.0 * 3.141592653589793, device=z_req.device))
                log_q_local = -0.5 * quad - 0.5 * logdet - 0.5 * const
                grads = torch.autograd.grad(
                    log_q_local.sum(),
                    (z_req, mu_req),
                    retain_graph=False,
                    allow_unused=True,
                )
                g_z, g_mu = grads
                if g_z is None:
                    g_z = torch.zeros_like(z_req)
                if g_mu is None:
                    g_mu = torch.zeros_like(mu_req)
            except Exception as exc:
                print(f"[GRAD TRACE] kl_grad_sanity inner grad failed: {exc}")
                g_z = g_mu = None
        else:
            g_z = g_mu = None

        # Volume prior gradient at zS (if available)
        g_vol = None
        if zS is not None:
            # Use loss manager's preferred rep to compute +0.5 log|G^{-1}|
            G_eval, rep = loss_manager._evaluate_metric(z_req, metric_tensor, rhmc_posterior, with_rep=True)
            if G_eval is not None and rep is not None:
                half = loss_manager._half_logdet_volume(G_eval.float(), rep.lower(), jitter=1e-6)
                g_vol, = torch.autograd.grad(half.sum(), z_req, retain_graph=False, allow_unused=True)
        print("\n[GRAD TRACE] KL sanities")
        if g_z is not None:
            print(tensor_stats("∂ log_q / ∂ z", g_z))
        if g_mu is not None:
            print(tensor_stats("∂ log_q / ∂ μ", g_mu))
        if g_vol is not None:
            print(tensor_stats("∂ (0.5 log|G^{-1}|) / ∂ z", g_vol))
    except Exception as e:
        print(f"[GRAD TRACE] kl_grad_sanity failed: {e}")
