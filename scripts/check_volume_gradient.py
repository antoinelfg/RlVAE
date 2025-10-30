#!/usr/bin/env python
"""
Finite-difference sanity check for the RHMC volume force.

Usage:
    python scripts/check_volume_gradient.py

This script instantiates ``RiemannianRHMCPosterior`` with a simple toy metric
and compares:
  * autograd-based ``_compute_potential_gradient``
  * finite-difference estimates of ½·log|G⁻¹|

It prints cosine similarities and projected finite differences for both
``volume_force_representation`` choices (``'g'`` and ``'ginv'``).
"""

import math
from typing import Dict, Optional

import torch

from src.rlvae.models.components.metric_utils import half_logdet_volume
from src.rlvae.models.components.riemannian_rhmc_posterior import (
    RiemannianRHMCPosterior,
)


class ToyMetricModel(torch.nn.Module):
    """Minimal model exposing ``G``/``G_inv`` used by the RHMC posterior."""

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cpu")

    def G(self, z: torch.Tensor) -> torch.Tensor:
        # Simple SPD metric: diag(exp(2 z_0), 1.0 + 0.5 sin(z_1))
        z = z.to(self.device)
        diag0 = torch.exp(2.0 * z[..., 0])
        diag1 = 1.0 + 0.5 * torch.sin(z[..., 1])
        diag1 = torch.clamp(diag1, min=0.2)  # keep SPD
        diag = torch.stack((diag0, diag1), dim=-1)
        return torch.diag_embed(diag)

    def G_inv(self, z: torch.Tensor) -> torch.Tensor:
        G = self.G(z)
        return torch.linalg.inv(G)


def run_check(rep: str, config: Dict[str, float], device: torch.device) -> None:
    model = ToyMetricModel(device=device)
    posterior = RiemannianRHMCPosterior(model, config)
    posterior.volume_force_representation = rep
    posterior.volume_force_sign = 1.0

    torch.manual_seed(0)
    z = torch.randn(8, 2, device=device, dtype=torch.double, requires_grad=True)

    posterior.volume_force_sign = 1.0
    posterior.volume_grad_scale = 1.0
    posterior.volume_bias_weight = 1.0

    # Compute raw ∇ log volume and corresponding ∇U manually
    z_req = z.clone().detach().requires_grad_(True)
    if rep == "g":
        G = posterior._ctx['model'].G(z_req)
        log_vol = half_logdet_volume(G, "g", jitter=posterior.eps_reg)
    else:
        Ginv = posterior._get_inverse_metric(z_req)
        log_vol = half_logdet_volume(Ginv, "ginv", jitter=posterior.eps_reg)
    grad_raw, = torch.autograd.grad(log_vol.sum(), z_req)
    grad_u = -grad_raw  # ∇U = -∇(½ log|G⁻¹|)

    eps = 1e-4
    dir_unit = torch.nn.functional.normalize(grad_raw.detach(), dim=-1, eps=1e-12)
    z_plus = z.detach() + eps * dir_unit
    z_minus = z.detach() - eps * dir_unit
    fd = (
        posterior._evaluate_half_logdet(z_plus, rep)
        - posterior._evaluate_half_logdet(z_minus, rep)
    ) / (2.0 * eps)

    dot = (grad_u * grad_raw).sum(dim=-1)
    cos = dot / (
        torch.norm(grad_u, dim=-1) * torch.norm(grad_raw, dim=-1).clamp_min(1e-12)
    )

    print("[FORCE CHECK] rep={} ".format(rep)
          + f"grad_u_norm={torch.norm(grad_u, dim=-1).mean().item():.4e} "
          + f"grad_raw_norm={torch.norm(grad_raw, dim=-1).mean().item():.4e} "
          + f"dot_mean={dot.mean().item():+.4e} "
          + f"cos_mean={cos.mean().item():+.4f} "
          + f"fd_mean={fd.mean().item():+.4e} "
          + f"fd_std={fd.std().item():.4e}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_config = {
        "rhmc_steps": 0,
        "rhmc_alpha": 1.0,
        "rhmc_eps_reg": 1e-4,
        "volume_force_sign": 1.0,
        "sigma_normalization_mode": "none",
        "initial_target_radius": 0.0,
        "initial_max_retries": 0,
    }
    for rep in ("g", "ginv"):
        run_check(rep, base_config, device)


if __name__ == "__main__":
    main()
