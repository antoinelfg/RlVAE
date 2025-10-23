#!/usr/bin/env python3
"""
Quick diagnostic comparing LossManager behaviour under metric_representation='g' vs 'ginv'.

The script loads a metric checkpoint, samples synthetic latent batches, and evaluates the
core helper methods used inside the ELBO (volume term, quadratic forms, covariance resolver,
pushforward metric, and total loss assembly).  It prints the absolute differences between
the two configurations to show that almost all quantities coincide up to numerical noise.

Usage:
    python scripts/metric_representation_probe.py \
        --metric data/pretrained/metric_diverse_mlp_ld16_20250828_123543.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

import sys
sys.path.append("src")

from rlvae.models.components.metric_tensor import MetricTensor
from rlvae.models.components.loss_manager import LossManager
from rlvae.models.components.riemannian_rhmc_posterior import _log_kinetic_density as rhmc_log_kin


def load_metric(metric_path: Path, latent_dim: int | None = None) -> MetricTensor:
    """Load MetricTensor from checkpoint on CPU."""
    state = torch.load(metric_path, map_location="cpu", weights_only=False)
    if latent_dim is None:
        latent_dim = int(state.get("latent_dim", 0)) or state["centroids"].shape[-1]
    centroids = state["centroids"]
    matrices = state.get("metric_matrices", state.get("M_matrices"))
    temperature = state.get("temperature", 0.1)
    regularization = state.get("regularization", 0.0)

    metric = MetricTensor(latent_dim=latent_dim, device=torch.device("cpu"))
    metric.load_pretrained(centroids, matrices, temperature, regularization)
    metric.eval()
    return metric


def collect_loss_outputs(lm: LossManager, metric: MetricTensor, latent_dim: int) -> Dict[str, torch.Tensor]:
    """Evaluate a representative mini-batch of loss components for a given LossManager."""
    torch.manual_seed(13)
    batch = 5
    z = torch.randn(batch, latent_dim)
    rho = torch.randn(batch, latent_dim)
    x = torch.randn(batch, 3, 16, 16)
    x_recon = x + 0.05 * torch.randn_like(x)
    mu = torch.randn(batch, latent_dim)
    log_var = 0.05 * torch.randn(batch, latent_dim)
    z_samples = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

    outputs: Dict[str, torch.Tensor] = {}

    G_tensor, G_rep = lm._evaluate_metric(z, metric, None, with_rep=True)
    if G_tensor is None or G_rep is None:
        raise RuntimeError("Metric evaluation failed in probe.")
    outputs["half_logdet"] = lm._half_logdet_volume(G_tensor, G_rep)
    outputs["quad"] = lm._quad_with_G(rho, G_tensor, G_rep)
    outputs["sigma_mu"] = lm._resolve_sigma_mu(mu, None, metric, None, None)
    push_tensor, min_sv, half_logdet_push_g, half_logdet_push_ginv = lm._pushforward_metric_via_flows(z, None, metric, None)
    outputs["push_metric_tensor"], outputs["push_metric_rep"] = push_tensor
    outputs["push_half_logdet_g"] = half_logdet_push_g
    outputs["push_half_logdet_ginv"] = half_logdet_push_ginv
    outputs["push_min_sv"] = min_sv
    outputs["log_kinetic_lossmgr"] = lm._log_kinetic_density(rho, z, metric, None)

    loss_dict = lm.compute_total_loss(
        x=x,
        x_recon=x_recon,
        mu=mu,
        log_var=log_var,
        z_samples=z_samples,
        log_det_jacobians=None,
        z_seq=[z_samples],
        flow_manager=None,
        loop_mode="open",
        metric_tensor=metric,
        use_riemannian_kl=True,
        rhmc_posterior=None,
    )
    for key, value in loss_dict.items():
        if torch.is_tensor(value):
            outputs[f"loss_{key}"] = value.detach()

    class _MetricWrapper(nn.Module):
        def __init__(self, tensor: MetricTensor):
            super().__init__()
            self.tensor = tensor

        def G(self, pts: torch.Tensor) -> torch.Tensor:
            return self.tensor.compute_metric(pts)

    # For reference: kinetic density computed directly from the sampler helper (always uses G).
    outputs["log_kinetic_rhmc"] = rhmc_log_kin(_MetricWrapper(metric), z, rho, eps=1e-3)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LossManager metric representations.")
    parser.add_argument(
        "--metric",
        type=Path,
        default=Path("data/pretrained/metric_diverse_mlp_ld16_20250828_123543.pt"),
        help="Path to metric checkpoint (default: %(default)s)",
    )
    parser.add_argument("--latent-dim", type=int, default=None, help="Override latent dimension.")
    args = parser.parse_args()

    metric = load_metric(args.metric, args.latent_dim)
    latent_dim = metric.latent_dim

    summaries = {}
    for rep in ("g", "ginv"):
        lm = LossManager(metric_representation=rep)
        lm.eval()
        summaries[rep] = collect_loss_outputs(lm, metric, latent_dim)

    print(f"Loaded metric '{args.metric}' with latent_dim={latent_dim}, "
          f"logdet(G) mean={metric.compute_log_det_metric(torch.randn(32, latent_dim)).mean():.3f}")

    def _to_metric(tensor: torch.Tensor, rep: str) -> torch.Tensor:
        rep = (rep or "").lower()
        if rep == "g":
            return tensor
        if rep == "ginv":
            eye = torch.eye(tensor.shape[-1], device=tensor.device, dtype=tensor.dtype).unsqueeze(0).expand_as(tensor)
            chol = torch.linalg.cholesky(tensor.float())
            metric = torch.cholesky_solve(eye.float(), chol).to(tensor.dtype)
            return metric
        raise ValueError(f"Unknown representation '{rep}'")

    compare_keys = [
        ("half_logdet", None),
        ("quad", None),
        ("sigma_mu", None),
        ("log_kinetic_lossmgr", None),
        ("push_half_logdet_g", None),
        ("push_half_logdet_ginv", None),
    ]

    print("\n=== Representation comparison (|g - ginv|) ===")
    for key, _ in compare_keys:
        g_val = summaries["g"][key]
        ginv_val = summaries["ginv"][key]
        diff = (g_val - ginv_val).abs().max().item()
        print(f"{key:26s} : {diff:.4e}")

    # Compare transported metric tensors in canonical (metric) form.
    g_tensor = summaries["g"]["push_metric_tensor"]
    g_rep = summaries["g"]["push_metric_rep"]
    ginv_tensor = summaries["ginv"]["push_metric_tensor"]
    ginv_rep = summaries["ginv"]["push_metric_rep"]
    g_metric = _to_metric(g_tensor, g_rep)
    ginv_metric = _to_metric(ginv_tensor, ginv_rep)
    push_diff = (g_metric - ginv_metric).abs().max().item()
    print(f"{'push_metric_tensor':26s} : {push_diff:.4e}")

    print("\nNote: 'log_kinetic_lossmgr' differs when LossManager recomputes the kinetic term, "
          "but the training pipeline usually reuses RHMC-provided delta_kin (see log_kinetic_rhmc).")


if __name__ == "__main__":
    main()
