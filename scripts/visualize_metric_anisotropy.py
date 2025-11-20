#!/usr/bin/env python3
"""
Quick probe to visualise a 2D metric's volume and anisotropy.

Loads the metric from config (Stage C) or a provided path and renders:
  - log|det(G^{-1})| heatmap over a PCA-projected grid
  - Anisotropy ratio heatmap (λ_max / λ_min of G^{-1})

Usage:
  python scripts/visualize_metric_anisotropy.py --metric outputs/stages/B_RHVAE_MLP_2_SPRITES/metric.pt
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure repo src/ is importable when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlvae.models.components.metric_tensor import MetricTensor


def load_metric(metric_path: Path):
    state = torch.load(metric_path, map_location="cpu", weights_only=False)
    centroids = state["centroids"]
    M = state.get("metric_matrices", state.get("M_matrices"))
    T = float(state.get("temperature", 0.2))
    lam = float(state.get("regularization", 0.01))

    mt = MetricTensor(
        latent_dim=centroids.shape[1],
        temperature=T,
        regularization=lam,
        trainable=False,
        normalize_weight_sum=True,
        weight_kernel="mahalanobis_normed",
        weight_metric_normalization="none",
        use_background_identity=False,
    )
    mt.load_pretrained(centroids, M, temperature=T, regularization=lam)
    return mt


def project_to_pca(points: torch.Tensor, centroids: torch.Tensor):
    # Simple 2D PCA for plotting when latent_dim==2
    return points.cpu().numpy(), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default=None, help="Path to metric.pt")
    args = parser.parse_args()

    metric_path = Path(args.metric) if args.metric else Path("outputs/stages/B_RHVAE_MLP_2_SPRITES/metric_diverse_mlp_ld2_20251120_135948.pt")
    if not metric_path.exists():
        raise SystemExit(f"Metric file not found: {metric_path}")

    mt = load_metric(metric_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mt = mt.to(device)

    cent = mt.centroids.to(device)
    latent_dim = cent.shape[1]
    if latent_dim != 2:
        raise SystemExit("Only latent_dim=2 supported for this quick viz.")

    # Build a grid around centroids
    all_pts = cent
    lo = all_pts.min(dim=0).values - 1.0
    hi = all_pts.max(dim=0).values + 1.0
    xs = torch.linspace(lo[0], hi[0], 200, device=device)
    ys = torch.linspace(lo[1], hi[1], 200, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)

    with torch.no_grad():
        Ginv, _ = mt._compute_precision_components(grid, return_metric=False)
    # log|det G^{-1}|
    logdet = torch.logdet(Ginv.float()).reshape(200, 200).cpu().numpy()
    # anisotropy λmax/λmin
    evals = torch.linalg.eigvalsh(Ginv.float()).reshape(200, 200, 2)
    ratio = (evals[..., 1] / evals[..., 0].clamp_min(1e-12)).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    extent = [lo[0].cpu().item(), hi[0].cpu().item(), lo[1].cpu().item(), hi[1].cpu().item()]
    im0 = axes[0].imshow(logdet.T, origin="lower", extent=extent, cmap="viridis")
    axes[0].set_title("log|det G⁻¹(z)|")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(np.log10(ratio.T + 1e-12), origin="lower", extent=extent, cmap="magma")
    axes[1].set_title("log10 anisotropy ratio (λ_max/λ_min)")
    fig.colorbar(im1, ax=axes[1])

    axes[0].scatter(cent[:, 0].cpu(), cent[:, 1].cpu(), c="cyan", s=10, label="centroids")
    axes[1].scatter(cent[:, 0].cpu(), cent[:, 1].cpu(), c="cyan", s=10, label="centroids")
    for ax in axes:
        ax.legend(loc="upper right")
    plt.tight_layout()
    out = metric_path.parent / f"metric_anisotropy_{metric_path.stem}.png"
    plt.savefig(out, dpi=200)
    print(f"Saved anisotropy visual to {out}")


if __name__ == "__main__":
    main()
