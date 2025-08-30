#!/usr/bin/env python3
"""
Metric & RHMC Verification on Sprites
====================================

Stage 1 verification script that:
- trains a quick vanilla VAE on cyclic Sprites,
- extracts an RHVAE-style metric (centroids + M matrices),
- computes/plots latent diagnostics (PCA scatter, det(G^{-1}) scatter, centroid overlay, geodesic-like connections),
- runs RHMC sampling in the learned latent manifold and reconstructs an interpolation grid.

Usage (example):
  python scripts/metric_and_rhmc_verification.py \
      --data-path /home/alaforgu/scratch/longitudinal_experiments/RlVAE/data/processed/Sprites_train_cyclic.pt \
      --architecture cnn --latent-dim 16 --epochs 5 --output-dir outputs/metric_verification
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dataclasses import dataclass
import argparse
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.utils as vutils

# Ensure local src is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_diverse_metric_vae import (
    SpritesDataset as _SpritesDatasetLoader,
    create_model as _create_model,
    extract_diverse_metric as _extract_diverse_metric,
)
from models.samplers.hmc_sampler import RiemannianHMCSampler


@dataclass
class Args:
    data_path: str
    architecture: str = "cnn"  # cnn|resnet|mlp|pythae
    latent_dim: int = 16
    epochs: int = 5
    batch_size: int = 32
    temperature: float = 0.5
    regularization: float = 0.01
    num_centroids: int = 50
    standardize_latents: bool = False
    centroid_method: str = "kmedoids"  # kmedoids|kmeans|fps|balanced
    neighbor_mode: str = "global"      # global|knn
    knn_k: int = 300
    coarse_k: int = 8
    normalize_M: str = "trace"  # none|trace|det
    target_mean_eig: float = 1.0
    hmc_steps: int = 30
    hmc_n_lf: int = 15
    hmc_eps: float = 0.03
    output_dir: str = "outputs/metric_verification"
    seed: int = 42
    device: Optional[str] = None


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_quick_vae(args: Args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load and flatten sequences across time for vanilla VAE training
    train_dataset = _SpritesDatasetLoader(args.data_path, normalize=False)
    if hasattr(train_dataset, "data") and train_dataset.data.dim() == 4:
        # Already [N, C, H, W]
        flat = train_dataset.data
    else:
        # [N_seq, T, C, H, W] -> [N_seq*T, C, H, W]
        data = train_dataset.data
        flat = data.reshape(-1, *data.shape[2:])

    train_loader = DataLoader(flat, batch_size=args.batch_size, shuffle=True)

    input_dim = (3, 64, 64)
    model = _create_model(args.architecture, input_dim=input_dim, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    history = {"loss": [], "recon": [], "kld": []}
    model.train()
    first_batch_for_recon = None
    for epoch in range(args.epochs):
        total, recon_total, kld_total = 0.0, 0.0, 0.0
        for bidx, batch in enumerate(train_loader):
            batch = batch.to(device)
            if first_batch_for_recon is None:
                first_batch_for_recon = batch.detach().cpu()
            if args.architecture.lower() in ["mlp", "pythae"]:
                output = model({"data": batch})
                loss = output.loss
                recon = output.recon_loss
                kld = output.reg_loss
            else:
                output = model(batch)
                loss = output.loss
                recon = output.reconstruction_loss
                kld = output.reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            recon_total += float(recon.item())
            kld_total += float(kld.item())
        epoch_loss = total/len(train_loader)
        epoch_recon = recon_total/len(train_loader)
        epoch_kld = kld_total/len(train_loader)
        history["loss"].append(epoch_loss)
        history["recon"].append(epoch_recon)
        history["kld"].append(epoch_kld)
        print(f"[Train] Epoch {epoch+1}/{args.epochs} loss={epoch_loss:.4f}"
              f" recon={epoch_recon:.4f} kld={epoch_kld:.4f}")

    model.eval()
    return model, device, history, first_batch_for_recon


def extract_metric(args: Args, model, device) -> Path:
    metric_path_str = _extract_diverse_metric(
        model=model,
        architecture=args.architecture,
        latent_dim=args.latent_dim,
        temperature=args.temperature,
        regularization=args.regularization,
        num_centroids=args.num_centroids,
        input_dim=(3, 64, 64),
        data_path=args.data_path,
        standardize_latents=args.standardize_latents,
        centroid_method=args.centroid_method,
        neighbor_mode=args.neighbor_mode,
        knn_k=args.knn_k,
        coarse_k=args.coarse_k,
        normalize_M=args.normalize_M,
        target_mean_eig=args.target_mean_eig,
    )
    return Path(metric_path_str)


class MetricModelStub:
    """Minimal wrapper providing G and G_inv for RHMC using centroids + M tensors."""
    def __init__(self, centroids: torch.Tensor, M_tens: torch.Tensor, temperature: float, lbd: float, device):
        self.device = device
        self.latent_dim = centroids.shape[1]
        self.centroids_tens = centroids.to(device)
        self.M_tens = M_tens.to(device)
        self.temperature = torch.tensor(temperature, device=device)
        self.lbd = torch.tensor(lbd, device=device)
        # Provide a minimal parameters() interface expected by BaseRiemannianSampler
        self._dummy = torch.nn.Parameter(torch.empty(0, device=device), requires_grad=False)

    def parameters(self):
        # Iterator with at least one tensor resident on the device
        yield self._dummy

    def G_inv(self, z: torch.Tensor) -> torch.Tensor:
        # Squared distances to centroids
        diff = self.centroids_tens.unsqueeze(0) - z.unsqueeze(1)  # (B, K, D)
        d2 = torch.sum(diff * diff, dim=-1)
        # Gaussian weights WITHOUT normalization (RHVAE-style)
        temp2 = (self.temperature ** 2).clamp(min=1e-8)
        w = torch.exp(-d2 / temp2)
        # Weighted sum of local precision matrices
        weighted_M = self.M_tens.unsqueeze(0) * w.unsqueeze(-1).unsqueeze(-1)
        base = weighted_M.sum(dim=1)
        # Add λI
        G_inv = base + self.lbd * torch.eye(self.latent_dim, device=z.device, dtype=z.dtype)
        return G_inv

    def G(self, z: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(self.G_inv(z))


def compute_latents(model, dataloader, device, max_points=5000) -> torch.Tensor:
    latents = []
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            if batch.dim() == 5:
                # [B, T, C, H, W] -> take t=0 for embedding cloud
                batch = batch[:, 0]
            if hasattr(model, "encode"):
                # Modular VAE
                mu, _ = model.encode(batch)
            else:
                # Pythae VAE
                mu = model.encoder(batch).embedding
            latents.append(mu.detach().cpu())
            count += mu.shape[0]
            if count >= max_points:
                break
    return torch.cat(latents, dim=0)


def plot_latent_and_metric(args: Args, latents: torch.Tensor, centroids: torch.Tensor,
                           metric_model: MetricModelStub, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set(style="whitegrid")

    # PCA to 2D for visualization
    from sklearn.decomposition import PCA
    z = latents.numpy()
    pca = PCA(n_components=2)
    z2 = pca.fit_transform(z)
    c2 = pca.transform(centroids.cpu().numpy())

    # Sample subset and compute det(G_inv)
    idx = np.random.choice(z2.shape[0], size=min(2000, z2.shape[0]), replace=False)
    z2_sub = z2[idx]
    z_sub = torch.tensor(pca.inverse_transform(z2_sub), device=metric_model.device, dtype=torch.float32)
    with torch.no_grad():
        det_vals = torch.linalg.det(metric_model.G_inv(z_sub)).cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(z2[:, 0], z2[:, 1], s=6, c="#1f77b4", alpha=0.4)
    ax[0].scatter(c2[:, 0], c2[:, 1], s=20, c="#d62728", label="centroids")
    ax[0].set_title("Latent PCA with centroids")
    ax[0].legend()

    sc = ax[1].scatter(z2_sub[:, 0], z2_sub[:, 1], s=8, c=np.log10(np.clip(det_vals, 1e-12, None)), cmap="viridis")
    plt.colorbar(sc, ax=ax[1], label="log10 det(G^{-1})")
    ax[1].set_title("det(G^{-1}) over latent cloud (PCA view)")
    fig.tight_layout()
    path = out_dir / "latent_and_metric_scatter.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    # Determinant heatmap across PCA plane
    # Define grid bounds with margin
    x_min, x_max = z2[:, 0].min(), z2[:, 0].max()
    y_min, y_max = z2[:, 1].min(), z2[:, 1].max()
    mx = 0.05 * (x_max - x_min + 1e-6)
    my = 0.05 * (y_max - y_min + 1e-6)
    x = np.linspace(x_min - mx, x_max + mx, 60)
    y = np.linspace(y_min - my, y_max + my, 60)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)  # [N, 2]
    # Map back to latent space via inverse PCA transform
    z_grid = torch.tensor(pca.inverse_transform(XY), device=metric_model.device, dtype=torch.float32)
    with torch.no_grad():
        det_grid = torch.linalg.det(metric_model.G_inv(z_grid)).cpu().numpy().reshape(X.shape)
    fig, ax = plt.subplots(figsize=(6, 6))
    hm = ax.imshow(np.log10(np.clip(det_grid, 1e-12, None)), origin="lower",
                   extent=[x.min(), x.max(), y.min(), y.max()], cmap="plasma", aspect="auto")
    plt.colorbar(hm, ax=ax, label="log10 det(G^{-1})")
    ax.scatter(c2[:, 0], c2[:, 1], s=10, c="white", alpha=0.8, label="centroids")
    ax.set_title("det(G^{-1}) heatmap (PCA plane)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "determinant_heatmap_pca.png", dpi=200)
    plt.close(fig)

    # Simple geodesic-like connections between nearest centroid pairs in PCA plane
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(c2)
    distances, indices = nn.kneighbors(c2)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(c2[:, 0], c2[:, 1], s=20, c="#d62728")
    for i in range(c2.shape[0]):
        j = indices[i, 1]
        ax.plot([c2[i, 0], c2[j, 0]], [c2[i, 1], c2[j, 1]], c="#ff7f0e", alpha=0.5)
    ax.set_title("Centroid graph (nearest-neighbor connections)")
    fig.tight_layout()
    path = out_dir / "centroid_geodesic_graph.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)


def rhmc_and_reconstruct(args: Args, model, metric_model: MetricModelStub, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    sampler = RiemannianHMCSampler(
        metric_model,
        mcmc_steps_nbr=args.hmc_steps,
        n_lf=args.hmc_n_lf,
        eps_lf=args.hmc_eps,
        beta_zero=1.0,
    )
    with torch.no_grad():
        z_samples = sampler.sample(128)

    # Decode a small grid of samples
    model_device = next(model.parameters()).device
    z_samples = z_samples.to(model_device)
    with torch.no_grad():
        if hasattr(model, "decode"):
            imgs = model.decode(z_samples)
        else:
            imgs = model.decoder(z_samples)["reconstruction"]

    imgs = imgs.clamp(0, 1).cpu()
    # Create a grid
    n = min(8, imgs.shape[0])
    grid_rows = []
    for r in range(n):
        grid_rows.append(imgs[r])
    grid = torch.stack(grid_rows)

    # Save grid
    fig, ax = plt.subplots(figsize=(8, 4))
    # Make a tidy square grid as mosaic
    from torchvision.utils import make_grid
    mosaic = make_grid(grid, nrow=8)
    ax.imshow(mosaic.permute(1, 2, 0).numpy())
    ax.axis("off")
    ax.set_title("RHMC samples decoded")
    fig.tight_layout()
    path = out_dir / "rhmc_samples_decoded.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    # Interpolation between two encoded images at t=0 with reconstructions
    # Use timestep_only=0 to ensure [B, C, H, W]
    seq_dataset = _SpritesDatasetLoader(args.data_path, normalize=False, timestep_only=0)
    seq_loader = DataLoader(seq_dataset, batch_size=16, shuffle=True)
    x_batch = next(iter(seq_loader)).to(model_device)  # [B, C, H, W]
    with torch.no_grad():
        if hasattr(model, "encode"):
            mu, _ = model.encode(x_batch)
        else:
            mu = model.encoder(x_batch).embedding
    if mu.shape[0] >= 2:
        a, b = mu[0:1], mu[1:2]
    else:
        a = b = mu[0:1]
    ts = torch.linspace(0, 1, steps=12, device=mu.device).view(-1, 1)
    z_line = (1 - ts) * a + ts * b
    with torch.no_grad():
        if hasattr(model, "decode"):
            rec = model.decode(z_line)
        else:
            rec = model.decoder(z_line)["reconstruction"]
    rec = rec.clamp(0, 1).cpu()
    from torchvision.utils import make_grid as _make_grid
    fig, ax = plt.subplots(figsize=(12, 2))
    grid = _make_grid(rec, nrow=12)
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    ax.set_title("Linear latent interpolation (decoded)")
    fig.tight_layout()
    fig.savefig(out_dir / "interpolation_decoded.png", dpi=200)
    plt.close(fig)


def plot_training_summaries(history, batch_for_recon: torch.Tensor, model, device, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Loss curves
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["loss"], label="loss")
    ax.plot(history["recon"], label="recon")
    ax.plot(history["kld"], label="kld")
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.set_title("Training curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=200)
    plt.close(fig)
    # Recon grid on first batch
    with torch.no_grad():
        batch = batch_for_recon.to(device)
        if hasattr(model, "encode"):
            if hasattr(model, "reparameterize"):
                mu, logvar = model.encode(batch)
                z = model.reparameterize(mu, logvar)
                recon = model.decode(z)
            else:
                # pythae path
                out = model({"data": batch})
                recon = out.recon_x
        else:
            out = model({"data": batch})
            recon = out.recon_x
    recon = recon.clamp(0, 1).cpu()
    grid = vutils.make_grid(recon[:16], nrow=8)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    ax.set_title("Training reconstructions")
    fig.tight_layout()
    fig.savefig(out_dir / "training_recon_grid.png", dpi=200)
    plt.close(fig)


def plot_metric_histograms(metric_blob: dict, out_dir: Path):
    try:
        M = metric_blob.get("M_matrices", metric_blob.get("metric_matrices"))
        evals = torch.linalg.eigvals(M).real
        dets = torch.linalg.det(M)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].hist(evals.cpu().numpy().ravel(), bins=50, color="#1f77b4")
        ax[0].set_title("Metric eigenvalues")
        ax[1].hist(np.log10(np.clip(dets.cpu().numpy(), 1e-20, None)), bins=50, color="#ff7f0e")
        ax[1].set_title("log10 det(M)")
        fig.tight_layout()
        fig.savefig(out_dir / "metric_eigs_and_dets.png", dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] metric histogram plotting failed: {e}")


def plot_weight_vs_distance_diagnostics(latents: torch.Tensor, centroids: torch.Tensor, metric_model: MetricModelStub, out_dir: Path):
    """Sanity check: weights to nearest centroid should decay with distance; det(G^{-1}) should anti-correlate with distance if M scale is normalized.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    device = metric_model.device
    with torch.no_grad():
        z = latents.to(device)
        diff = centroids.to(device).unsqueeze(0) - z.unsqueeze(1)
        d2 = torch.sum(diff * diff, dim=-1)
        temp2 = (metric_model.temperature ** 2).clamp(min=1e-8)
        w = torch.exp(-d2 / temp2)
        d_min, idx = torch.min(d2, dim=1)
        w_near = w.gather(1, idx.view(-1, 1)).squeeze(1)
        det_vals = torch.linalg.det(metric_model.G_inv(z)).detach().cpu()
    import numpy as np
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(torch.sqrt(d_min).cpu().numpy(), w_near.cpu().numpy(), s=6, alpha=0.4)
    ax[0].set_xlabel("euclidean distance to nearest centroid")
    ax[0].set_ylabel("normalized weight of nearest centroid")
    ax[0].set_title("w_nearest vs distance")
    ax[1].scatter(torch.sqrt(d_min).cpu().numpy(), np.log10(np.clip(det_vals.numpy(), 1e-12, None)), s=6, alpha=0.4)
    ax[1].set_xlabel("euclidean distance to nearest centroid")
    ax[1].set_ylabel("log10 det(G^{-1})")
    ax[1].set_title("det(G^{-1}) vs distance")
    fig.tight_layout()
    fig.savefig(out_dir / "weight_and_det_vs_distance.png", dpi=200)
    plt.close(fig)


def run(args: Args):
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Training quick vanilla VAE...")
    model, device, history, first_batch_for_recon = train_quick_vae(args)
    plot_training_summaries(history, first_batch_for_recon, model, device, out_dir)

    print("[2/5] Extracting RHVAE-style metric (centroids + M matrices)...")
    metric_path = extract_metric(args, model, device)
    metric_blob = torch.load(metric_path, map_location=device, weights_only=False)
    plot_metric_histograms(metric_blob, out_dir)
    centroids = metric_blob["centroids"]
    M_matrices = metric_blob.get("M_matrices", metric_blob.get("metric_matrices"))
    temperature = float(metric_blob.get("temperature", args.temperature))
    regularization = float(metric_blob.get("regularization", args.regularization))
    metric_model = MetricModelStub(centroids, M_matrices, temperature=temperature, lbd=regularization, device=device)

    print("[3/5] Computing latent cloud and plotting metric diagnostics...")
    # Use sequences without flattening to get latent cloud from t=0
    seq_dataset = _SpritesDatasetLoader(args.data_path, normalize=False)
    seq_loader = DataLoader(seq_dataset, batch_size=128, shuffle=False)
    latents = compute_latents(model, seq_loader, device, max_points=4000)
    plot_latent_and_metric(args, latents, centroids, metric_model, out_dir)
    # New diagnostic: weight and det vs distance
    try:
        plot_weight_vs_distance_diagnostics(latents[:3000], centroids, metric_model, out_dir)
    except Exception as e:
        print(f"[WARN] weight-vs-distance diagnostic failed: {e}")

    print("[4/5] Running RHMC sampling and decoding...")
    rhmc_and_reconstruct(args, model, metric_model, out_dir)

    print("[5/5] Done. See training_curves, training_recon_grid, metric histograms, and manifold plots.")

    print(f"\n✅ Metric & RHMC verification complete. Outputs in: {out_dir}")


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Metric and RHMC verification on Sprites")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Absolute path to Sprites_train_cyclic.pt")
    parser.add_argument("--architecture", type=str, default="cnn", choices=["cnn", "resnet", "mlp", "pythae"]) 
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--num-centroids", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="outputs/metric_verification")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda; default auto")
    parser.add_argument("--standardize-latents", action="store_true")
    parser.add_argument("--hmc-steps", type=int, default=30)
    parser.add_argument("--hmc-n-lf", type=int, default=15)
    parser.add_argument("--hmc-eps", type=float, default=0.03)
    parser.add_argument("--centroid-method", type=str, default="kmedoids", choices=["kmedoids","kmeans","fps","balanced"])
    parser.add_argument("--neighbor-mode", type=str, default="global", choices=["global","knn"])
    parser.add_argument("--knn-k", type=int, default=300)
    parser.add_argument("--coarse-k", type=int, default=8)
    parser.add_argument("--normalize-M", type=str, default="trace", choices=["none","trace","det"])
    parser.add_argument("--target-mean-eig", type=float, default=1.0)
    ns = parser.parse_args()
    return Args(
        data_path=ns.data_path,
        architecture=ns.architecture,
        latent_dim=ns.latent_dim,
        epochs=ns.epochs,
        batch_size=ns.batch_size,
        temperature=ns.temperature,
        regularization=ns.regularization,
        num_centroids=ns.num_centroids,
        output_dir=ns.output_dir,
        seed=ns.seed,
        device=ns.device,
        standardize_latents=ns.standardize_latents,
        hmc_steps=ns.hmc_steps,
        hmc_n_lf=ns.hmc_n_lf,
        hmc_eps=ns.hmc_eps,
        centroid_method=ns.centroid_method,
        neighbor_mode=ns.neighbor_mode,
        knn_k=ns.knn_k,
        coarse_k=ns.coarse_k,
        normalize_M=ns.normalize_M,
        target_mean_eig=ns.target_mean_eig,
    )


if __name__ == "__main__":
    run(parse_args())

