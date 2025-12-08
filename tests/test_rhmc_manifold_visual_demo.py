#!/usr/bin/env python3
"""
RHMC Sampler on a Synthetic Manifold - Visual Demo (Test)
=========================================================

This test builds a simple synthetic 2D manifold via an RBF-interpolated
inverse metric G^{-1}(z), runs the existing Riemannian HMC sampler, and
produces visualizations:
- Scatter of RHMC samples
- Metric determinant contour (det(G^{-1}))
- Combined overlay (samples + centroids + determinant)

Plots are saved under outputs/rhmc_manifold_visual_demo/ using a headless
matplotlib backend, so the test can run in CI environments.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Tuple

import matplotlib

# Ensure headless backend for plotting in tests
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


# Import the sampler from the codebase
from src.models.samplers.hmc_sampler import RiemannianHMCSampler


class MetricModelStub:
    """Minimal metric model providing G and G_inv expected by RiemannianHMCSampler.

    We construct G^{-1}(z) by RBF-weighted interpolation of a set of SPD anchors
    located at "centroids" laid out on a ring (synthetic manifold). The metric
    is then regularized with λI to ensure numerical stability.
    """

    def __init__(
        self,
        centroids: torch.Tensor,
        inv_metric_anchors: torch.Tensor,
        temperature: float,
        regularization_lambda: float,
        device: torch.device,
    ) -> None:
        self.device = device
        self.latent_dim = int(centroids.shape[1])
        self.centroids_tens = centroids.to(device)
        self.M_tens = inv_metric_anchors.to(device)  # anchors for G^{-1}
        self.temperature = torch.tensor(float(temperature), device=device)
        self.lbd = torch.tensor(float(regularization_lambda), device=device)

        # Provide parameters() to mirror a Module-like API used in some places
        self._dummy = torch.nn.Parameter(torch.empty(0, device=device), requires_grad=False)

    def parameters(self):  # pragma: no cover - tiny helper
        yield self._dummy

    def G_inv(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, D]; centroids: [K, D]
        # Compute RBF weights w_k(z) = exp(-||z-c_k||^2 / T^2)
        diff = self.centroids_tens.unsqueeze(0) - z.unsqueeze(1)  # (B, K, D)
        d2 = torch.sum(diff * diff, dim=-1)  # (B, K)
        T = torch.clamp(self.temperature, min=1e-8)
        w = torch.exp(-d2 / (T * T))  # (B, K)

        # Weighted sum of inverse metric anchors
        weighted = self.M_tens.unsqueeze(0) * w.unsqueeze(-1).unsqueeze(-1)  # (B, K, D, D)
        base = weighted.sum(dim=1)  # (B, D, D)

        # Regularization for SPD and stability
        eye = torch.eye(self.latent_dim, device=z.device, dtype=z.dtype)
        return base + self.lbd * eye

    def G(self, z: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(self.G_inv(z))


def _make_ring_centroids(num: int = 36, radius: float = 2.0, device: torch.device | None = None) -> torch.Tensor:
    device = device or torch.device("cpu")
    # Avoid endpoint argument for broad PyTorch compatibility
    angles = torch.linspace(0.0, 2.0 * math.pi, steps=num + 1, device=device, dtype=torch.float32)[:-1]
    xs = radius * torch.cos(angles)
    ys = radius * torch.sin(angles)
    return torch.stack([xs, ys], dim=1)


def _make_anisotropic_inv_metrics_for_ring(centroids: torch.Tensor) -> torch.Tensor:
    """Build orientation-aware SPD anchors for G^{-1} at each centroid.

    For each centroid on the ring, orient axes radial/tangential and assign
    eigenvalues that yield anistropy in G^{-1}. This produces a smoothly
    varying field when interpolated.
    """
    device = centroids.device
    num = centroids.shape[0]
    inv_metrics: list[torch.Tensor] = []

    for i in range(num):
        cx, cy = centroids[i]
        # Radial unit vector
        r = torch.tensor([cx, cy], device=device)
        r_norm = torch.norm(r).clamp(min=1e-6)
        e_rad = r / r_norm
        # Tangential unit vector (rotate radial by +90°)
        e_tan = torch.tensor([-e_rad[1], e_rad[0]], device=device)

        # Form rotation matrix R = [e_rad, e_tan]
        R = torch.stack([e_rad, e_tan], dim=1)  # (2,2)

        # Choose eigenvalues for G^{-1}: stronger along tangential, weaker radial
        # This biases movement around the ring
        lambda_rad = 0.6
        lambda_tan = 2.0
        D = torch.diag(torch.tensor([lambda_rad, lambda_tan], device=device))

        M = R @ D @ R.T  # SPD
        inv_metrics.append(M)

    return torch.stack(inv_metrics, dim=0)  # (K, 2, 2)


def _ensure_out_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    env_out = os.environ.get("RHMC_OUT_DIR")
    if env_out:
        # Treat absolute paths as-is; otherwise relative to repo root
        env_path = Path(env_out)
        out_dir = env_path if env_path.is_absolute() else (repo_root / env_path)
    else:
        out_dir = repo_root / "outputs" / "rhmc_manifold_visual_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _plot_metric_determinant_contour(model: MetricModelStub, out_path: Path, bounds: Tuple[float, float] = (-4.0, 4.0), n: int = 150) -> None:
    xs = np.linspace(bounds[0], bounds[1], n)
    ys = np.linspace(bounds[0], bounds[1], n)
    X, Y = np.meshgrid(xs, ys)
    grid = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
    grid_t = torch.tensor(grid, device=model.device)

    with torch.no_grad():
        G_inv = model.G_inv(grid_t)  # (N, 2, 2)
        det_vals = torch.linalg.det(G_inv).detach().cpu().numpy().reshape(X.shape)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    cntr = ax.contourf(X, Y, det_vals, levels=25, cmap="viridis", alpha=0.9)
    ctr = model.centroids_tens.detach().cpu().numpy()
    ax.scatter(ctr[:, 0], ctr[:, 1], c="red", s=30, marker="*", edgecolors="white", linewidths=0.6, label="centroids")
    ax.set_title("det(G^{-1}(z)) contour")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.legend(loc="upper right")
    fig.colorbar(cntr, ax=ax, label="det(G^{-1})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_samples_scatter(samples: torch.Tensor, out_path: Path, title: str) -> None:
    z = samples.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(z[:, 0], z[:, 1], s=6, alpha=0.5, c="black")
    ax.set_title(title)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_combined(samples: torch.Tensor, model: MetricModelStub, out_path: Path, bounds: Tuple[float, float] = (-4.0, 4.0), n: int = 160) -> None:
    xs = np.linspace(bounds[0], bounds[1], n)
    ys = np.linspace(bounds[0], bounds[1], n)
    X, Y = np.meshgrid(xs, ys)
    grid = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
    grid_t = torch.tensor(grid, device=model.device)
    with torch.no_grad():
        det_vals = torch.linalg.det(model.G_inv(grid_t)).detach().cpu().numpy().reshape(X.shape)

    z = samples.detach().cpu().numpy()
    ctr = model.centroids_tens.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    cntr = ax.contourf(X, Y, det_vals, levels=25, cmap="viridis", alpha=0.7)
    ax.scatter(z[:, 0], z[:, 1], s=5, alpha=0.45, c="black", label="RHMC samples")
    ax.scatter(ctr[:, 0], ctr[:, 1], c="red", s=36, marker="*", edgecolors="white", linewidths=0.6, label="centroids")
    ax.set_title("RHMC samples over det(G^{-1})")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.legend(loc="upper right")
    fig.colorbar(cntr, ax=ax, label="det(G^{-1})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def test_rhmc_sampler_visual_demo() -> None:
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cpu")

    # Build synthetic manifold metric
    centroids = _make_ring_centroids(num=36, radius=2.2, device=device)
    inv_metric_anchors = _make_anisotropic_inv_metrics_for_ring(centroids)
    temperature = 0.6  # closer to earlier good setup
    lbd = 0.03  # lighter regularization

    metric_model = MetricModelStub(
        centroids=centroids,
        inv_metric_anchors=inv_metric_anchors,
        temperature=temperature,
        regularization_lambda=lbd,
        device=device,
    )

    # Configure RHMC sampler and set a ring-shaped target density
    sampler = RiemannianHMCSampler(
        model=metric_model,
        mcmc_steps_nbr=160,
        n_lf=28,
        eps_lf=0.013,
        beta_zero=1.0,
        include_volume_grad=True,
    )

    # Target: annulus centered at radius r0 with width sigma
    r0 = 2.2
    sigma = 0.25

    def log_pi_ring(z: torch.Tensor) -> torch.Tensor:
        r = torch.norm(z, dim=1)
        return -0.5 * ((r - r0) ** 2) / (sigma ** 2)

    def grad_log_pi_ring(z: torch.Tensor) -> torch.Tensor:
        r = torch.norm(z, dim=1, keepdim=True).clamp(min=1e-8)
        return -((r - r0) / (sigma ** 2)) * (z / r)

    sampler.log_pi = log_pi_ring
    sampler.grad_func = grad_log_pi_ring

    # Draw more samples but drop a short unbiased burn-in (do not initialize on ring)
    total_draws = 3000
    with torch.no_grad():
        # Add mild jitter to avoid resonance and help chains escape the center
        full_chain = sampler.sample(n_samples=total_draws, init_std=1.4, eps_jitter=0.1, n_lf_jitter=3)
    burn_in = 1000
    samples = full_chain[burn_in:]

    # Prepare output directory and plots
    out_dir = _ensure_out_dir()
    scatter_path = out_dir / "rhmc_samples_scatter.png"
    det_contour_path = out_dir / "metric_determinant_contour.png"
    combined_path = out_dir / "combined_overlay.png"

    _plot_samples_scatter(
        samples, scatter_path, title=f"RHMC samples (acc={getattr(sampler, 'last_acceptance_rate', float('nan')):.3f})"
    )
    _plot_metric_determinant_contour(metric_model, det_contour_path)
    _plot_combined(samples, metric_model, combined_path)

    # Basic assertions: files exist and acceptance rate is valid
    assert scatter_path.exists() and scatter_path.stat().st_size > 0
    assert det_contour_path.exists() and det_contour_path.stat().st_size > 0
    assert combined_path.exists() and combined_path.stat().st_size > 0

    acc = getattr(sampler, "last_acceptance_rate", None)
    assert acc is None or (0.0 <= acc <= 1.0)


