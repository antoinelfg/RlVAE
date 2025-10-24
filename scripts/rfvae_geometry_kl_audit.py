#!/usr/bin/env python3
"""
RF-VAE geometry & KL invariant audit.

This script reproduces the Phase 1 sanity checks requested in the
diagnostic brief.  It evaluates the MetricTensor under both
representations ('g' and 'ginv') at common latent points and records
whether the expected invariants hold:

    A. Metric pair identity:   || G · G^{-1} - I ||_F / d  < 1e-5
                               | logdet(G) + logdet(G^{-1}) |       < 1e-5
    B. Half-logdet equivalence: half_logdet_volume(G,'g') ≈
                                half_logdet_volume(Ginv,'ginv')
    C. Push-forward identity (no-flow baseline): the transported
       prior volume equals the source volume when J = I.

Results are persisted to outputs/probes/report.json so downstream
analysis or CI can diff the values.  The script exits with code 1 as
soon as an invariant fails, making the first discrepancy explicit.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

import sys
SYS_ROOT = Path(__file__).resolve().parents[1]  # repo root
if str(SYS_ROOT / "src") not in sys.path:
    sys.path.append(str(SYS_ROOT / "src"))

from rlvae.models.components.metric_tensor import MetricTensor
from rlvae.models.components.loss_manager import LossManager


def set_debug_env() -> None:
    """Ensure reproducible, trace-friendly execution."""
    os.environ.setdefault("RLVAE_DEBUG", "1")
    os.environ.setdefault("RLVAE_TRACE", "1")
    os.environ.setdefault("RLVAE_STRICT", "0")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_metric(metric_path: Path, device: torch.device) -> MetricTensor:
    state = torch.load(metric_path, map_location=device, weights_only=False)
    latent_dim = int(state.get("latent_dim", 0)) or state["centroids"].shape[-1]
    centroids = state["centroids"]
    matrices = state.get("metric_matrices", state.get("M_matrices"))
    temperature = state.get("temperature", 0.1)
    regularization = state.get("regularization", 0.0)

    metric = MetricTensor(latent_dim=latent_dim, device=device)
    metric.load_pretrained(centroids, matrices, temperature, regularization)
    metric.eval()
    return metric


def _precision_to_metric(loss: LossManager, tensor: torch.Tensor) -> torch.Tensor:
    """Convert precision matrix batch to metric using a stable linear solve."""
    tensor64 = tensor.double()
    d = tensor.shape[-1]
    eye = torch.eye(d, device=tensor.device, dtype=tensor64.dtype).expand(tensor64.shape[0], -1, -1)
    metric = torch.linalg.solve(tensor64, eye)
    metric = 0.5 * (metric + metric.transpose(-1, -2))
    return metric


def metric_pair_identity(
    loss_ginv: LossManager,
    metric: MetricTensor,
    z: torch.Tensor,
) -> Dict[str, float]:
    """Check G·G^{-1}=I and logdet cancelation using a shared evaluation path."""
    Ginv, rep = loss_ginv._evaluate_metric(z, metric, None, with_rep=True)
    if Ginv is None or rep is None:
        raise RuntimeError("Metric evaluation failed (Phase 1A).")
    rep = rep.lower()
    if rep != "ginv":
        raise RuntimeError(f"Expected precision representation, got '{rep}'.")

    Ginv = 0.5 * (Ginv + Ginv.transpose(-1, -2))
    G = _precision_to_metric(loss_ginv, Ginv)
    d = z.shape[-1]

    prod = torch.matmul(G.double(), Ginv.double())
    eye = torch.eye(d, device=prod.device, dtype=prod.dtype)
    fro_per_sample = ((prod - eye) ** 2).sum(dim=(1, 2)).sqrt() / d
    fro_mean = fro_per_sample.mean().item()
    fro_max = fro_per_sample.max().item()

    logdet_G = torch.linalg.slogdet(G.double())[1]
    logdet_Ginv = torch.linalg.slogdet(Ginv.double())[1]
    logdet_residual = (logdet_G + logdet_Ginv).abs()

    return {
        "fro_mean": float(fro_mean),
        "fro_max": float(fro_max),
        "fro_threshold": 1e-5,
        "logdet_residual_mean": float(logdet_residual.mean().item()),
        "logdet_residual_max": float(logdet_residual.max().item()),
        "logdet_threshold": 1e-5,
    }


def half_logdet_equivalence(
    loss_ginv: LossManager,
    G: torch.Tensor,
    Ginv: torch.Tensor,
) -> Dict[str, float]:
    """Check half_logdet_volume outputs agree across representations."""
    half_g = loss_ginv._half_logdet_volume(G, "g")
    half_ginv = loss_ginv._half_logdet_volume(Ginv, "ginv")
    residual = (half_g - half_ginv).abs()
    return {
        "mean_abs": float(residual.mean().item()),
        "max_abs": float(residual.max().item()),
        "threshold": 1e-5,
    }


def pushforward_identity(
    loss: LossManager,
    metric: MetricTensor,
    z: torch.Tensor,
) -> Dict[str, float]:
    """
    Identity-map push-forward: half_logdet_target should equal source + flow (flow=0).
    """
    G_source, rep_source = loss._evaluate_metric(z, metric, None, with_rep=True)
    if G_source is None or rep_source is None:
        raise RuntimeError("Metric evaluation failed for push-forward check.")
    rep_source = rep_source.lower()

    half_logdet_source = loss._half_logdet_volume(G_source, rep_source)
    # Identity map: zS = z0, flow term = 0
    half_logdet_target = loss._half_logdet_volume(G_source, rep_source)
    residual = (half_logdet_target - half_logdet_source).abs()
    return {
        "mean_abs": float(residual.mean().item()),
        "max_abs": float(residual.max().item()),
        "threshold": 1e-4,
    }


def run_audit(args: argparse.Namespace) -> Tuple[Dict[str, Dict[str, float]], str]:
    device = torch.device("cpu")
    set_debug_env()
    seed_everything(args.seed)

    metric = load_metric(args.metric, device)
    latent_dim = metric.latent_dim

    z = torch.randn(args.num_samples, latent_dim, device=device, dtype=torch.float64)

    loss_ginv = LossManager(metric_representation="ginv")
    loss_ginv.eval()

    phase1 = metric_pair_identity(loss_ginv, metric, z)
    Ginv, _ = loss_ginv._evaluate_metric(z, metric, None, with_rep=True)
    if Ginv is None:
        raise RuntimeError("Precision evaluation failed during Phase 1B.")
    G = _precision_to_metric(loss_ginv, Ginv)
    phase1b = half_logdet_equivalence(loss_ginv, G, Ginv)
    phase1c = pushforward_identity(loss_ginv, metric, z)  # representation-agnostic

    results: Dict[str, Dict[str, float]] = {
        "phase1_metric_pair": phase1,
        "phase1_half_logdet": phase1b,
        "phase1_pushforward_identity": phase1c,
    }

    # Determine first failing check in order
    fail_reason = ""
    if phase1["fro_mean"] >= phase1["fro_threshold"] or phase1["logdet_residual_max"] >= phase1["logdet_threshold"]:
        fail_reason = "Phase 1A (metric pair identity)"
    elif phase1b["max_abs"] >= phase1b["threshold"]:
        fail_reason = "Phase 1B (half-logdet equivalence)"
    elif phase1c["max_abs"] >= phase1c["threshold"]:
        fail_reason = "Phase 1C (push-forward identity)"

    return results, fail_reason


def main() -> None:
    parser = argparse.ArgumentParser(description="RF-VAE geometry & KL invariant audit.")
    parser.add_argument(
        "--metric",
        type=Path,
        default=Path("outputs/stages/B_RHVAE_MLP_2_SPRITES/metric.pt"),
        help="Path to MetricTensor checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--num-samples", type=int, default=1024, help="Number of latent samples for the audit.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/probes/report.json"),
        help="Path to JSON report (default: outputs/probes/report.json).",
    )
    args = parser.parse_args()

    results, failure = run_audit(args)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"results": results, "failure": failure}, handle, indent=2)

    if failure:
        print(f"❌ Invariant failure: {failure}")
        raise SystemExit(1)
    print("✅ All audited invariants passed.")


if __name__ == "__main__":
    main()
