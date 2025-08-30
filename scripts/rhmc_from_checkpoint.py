"""
RHMC-from-checkpoint utility
----------------------------

Load centroids/M metric tensors from a checkpoint file and run RHMC sampling
using our sampler. Saves diagnostics and samples to an output directory.

Hydra-configurable for quick iterations.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import math

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from hydra import main as hydra_main

# Ensure repository root is on sys.path when running this script directly
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.components.metric_loader import MetricLoader
from src.models.samplers.hmc_sampler import (
    RiemannianHMCSampler,
    DualRiemannianHMCSampler,
    RHVAEVolumeElementHMCSampler,
)


class MetricModelStub:
    """Minimal wrapper that provides G and G_inv from centroids + M.
    Matches the interface expected by RiemannianHMCSampler.
    """

    def __init__(self, centroids: torch.Tensor, M_tens: torch.Tensor, temperature: float, lbd: float, device: torch.device):
        self.device = device
        self.latent_dim = centroids.shape[1]
        self.centroids_tens = centroids.to(device)
        self.M_tens = M_tens.to(device)
        self.temperature = torch.tensor(temperature, device=device)
        self.lbd = torch.tensor(lbd, device=device)
        # Provide parameters() to anchor device placement
        self._dummy = torch.nn.Parameter(torch.empty(0, device=device), requires_grad=False)

    def parameters(self):
        yield self._dummy

    def G_inv(self, z: torch.Tensor) -> torch.Tensor:
        diff = self.centroids_tens.unsqueeze(0) - z.unsqueeze(1)  # (B,K,D)
        d2 = torch.sum(diff * diff, dim=-1)
        w = torch.exp(-d2 / (self.temperature.clamp(min=1e-8) ** 2))
        weighted_M = self.M_tens.unsqueeze(0) * w.unsqueeze(-1).unsqueeze(-1)
        base = weighted_M.sum(dim=1)
        eye = torch.eye(self.latent_dim, device=z.device, dtype=z.dtype)
        return base + self.lbd * eye

    def G(self, z: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(self.G_inv(z))


@dataclass
class Config:
    # Inputs
    checkpoint: str = "outputs/phaseB_sprites_ld10_pca2_rawpanel_rhmc_fix/final_model.pt"
    metric_key: Optional[str] = None  # if using non-standard file, ignored here
    device: str = "auto"  # auto|cpu|cuda
    synthetic: Optional[str] = None  # None | 'ring'

    # Sampler selection: 'rhmc' | 'dual' | 'volume' (original RHVAE behavior)
    sampler: str = "rhmc"

    # RHMC params
    n_samples: int = 2048
    mcmc_steps: int = 80
    n_lf: int = 20
    eps_lf: float = 0.02
    beta_zero: float = 1.0

    # Metric overrides
    temperature: Optional[float] = None
    regularization: Optional[float] = 1e-2

    # Synthetic ring params (used when synthetic == 'ring')
    ring_radius: float = 2.0
    ring_num_centroids: int = 180
    ring_temperature: float = 0.25
    ring_regularization: float = 1e-4
    ring_metric_scale: float = 1.0

    # Output
    out_dir: str = "outputs/rhmc_from_ckpt"


cs = ConfigStore.instance()
cs.store(name="rhmc_from_ckpt_config", node=Config)


def _choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _plot_scatter(z: torch.Tensor, path: Path, title: str = "RHMC samples") -> None:
    z = z.detach().cpu().numpy()
    if z.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(z[:, 0], z[:, 1], s=6, alpha=0.5, color="black")
        ax.set_title(title)
        ax.set_xlabel("z1"); ax.set_ylabel("z2")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)


def _pca2_basis(pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean, Up) where Up is Dx2 PCA basis computed with SVD."""
    with torch.no_grad():
        mean = pts.mean(dim=0)
        Xc = pts - mean
        # SVD on CPU for stability if large
        U, S, Vh = torch.linalg.svd(Xc.cpu(), full_matrices=False)
        Up = Vh[:2].T.contiguous()  # D x 2
    return mean.to(pts.device), Up.to(pts.device)


def _plot_metric_contour_and_samples(
    metric_model: MetricModelStub,
    centroids: torch.Tensor,
    samples: torch.Tensor,
    out_dir: Path,
    grid_size: int = 64,
    title: str = "det(G^{-1}) on PCA(2)"
) -> None:
    device = metric_model.device
    C = centroids.to(device)
    Zs = samples.to(device)
    mean, Up = _pca2_basis(C)

    # Project to 2D for plotting ranges
    C2 = ((C - mean) @ Up).detach().cpu().numpy()
    S2 = ((Zs - mean) @ Up).detach().cpu().numpy()

    x_min, x_max = float(C2[:, 0].min()), float(C2[:, 0].max())
    y_min, y_max = float(C2[:, 1].min()), float(C2[:, 1].max())
    mx = 0.10 * (x_max - x_min + 1e-6)
    my = 0.10 * (y_max - y_min + 1e-6)
    x = np.linspace(x_min - mx, x_max + mx, grid_size)
    y = np.linspace(y_min - my, y_max + my, grid_size)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)  # [N,2]

    # Map back to latent space: z = mean + Up @ [x;y]
    XY_t = torch.tensor(XY, device=device, dtype=torch.float32)
    Z_full = mean.unsqueeze(0) + XY_t @ Up.T  # [N,D]

    # Evaluate det(G^{-1}) in chunks
    det_list = []
    with torch.no_grad():
        B = 4096
        for i in range(0, Z_full.shape[0], B):
            Zi = Z_full[i:i+B]
            Ginv = metric_model.G_inv(Zi)
            det = torch.linalg.det(Ginv).clamp(min=1e-12)
            det_list.append(det.detach().cpu())
    det_grid = torch.cat(det_list, dim=0).numpy().reshape(X.shape)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(6, 6))
    hm = ax.imshow(
        np.log10(det_grid), origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()], cmap="plasma", aspect="auto"
    )
    plt.colorbar(hm, ax=ax, label="log10 det(G^{-1})")
    ax.scatter(C2[:, 0], C2[:, 1], s=10, c="white", alpha=0.8, label="centroids")
    ax.scatter(S2[:, 0], S2[:, 1], s=4, c="black", alpha=0.4, label="samples")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "metric_determinant_contour.png", dpi=200)
    plt.close(fig)

    # Combined overlay (scatter-only on top of heatmap already saved)
    fig, ax = plt.subplots(figsize=(6, 6))
    hm = ax.imshow(
        np.log10(det_grid), origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()], cmap="plasma", aspect="auto"
    )
    ax.scatter(S2[:, 0], S2[:, 1], s=5, c="#222222", alpha=0.5)
    ax.scatter(C2[:, 0], C2[:, 1], s=12, c="#ff4444", alpha=0.9)
    ax.set_title("Samples over det(G^{-1})")
    fig.tight_layout()
    fig.savefig(out_dir / "combined_overlay.png", dpi=200)
    plt.close(fig)


@hydra_main(config_path=None, config_name="rhmc_from_ckpt_config", version_base=None)
def run(cfg: Config):
    device = _choose_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Obtain metric tensors
    if cfg.synthetic == "ring":
        # Create a 2D ring of centroids with isotropic M matrices
        K = int(cfg.ring_num_centroids)
        theta = torch.linspace(0.0, 2.0 * math.pi, steps=K + 1)[:-1]
        x = cfg.ring_radius * torch.cos(theta)
        y = cfg.ring_radius * torch.sin(theta)
        centroids = torch.stack([x, y], dim=1).to(device)
        D = 2
        eye = torch.eye(D, device=device).unsqueeze(0).repeat(K, 1, 1)
        M = (float(cfg.ring_metric_scale) * eye).contiguous()
        T = float(cfg.ring_temperature)
        lbd = float(cfg.ring_regularization)
    else:
        # Load metric tensors (centroids, metric_matrices, temperature, regularization)
        loader = MetricLoader(device=device)
        metric_blob = loader.load_from_file(cfg.checkpoint, cfg.temperature, cfg.regularization)
        centroids = metric_blob['centroids']
        M = metric_blob['metric_matrices']
        T = float(metric_blob['temperature'])
        lbd = float(metric_blob['regularization'])

    # Build minimal metric model and sampler
    metric_model = MetricModelStub(centroids, M, temperature=T, lbd=lbd, device=device)

    if cfg.sampler == "dual":
        sampler = DualRiemannianHMCSampler(metric_model, mcmc_steps_nbr=int(cfg.mcmc_steps), n_lf=int(cfg.n_lf), eps_lf=float(cfg.eps_lf))
    elif cfg.sampler == "volume":
        sampler = RHVAEVolumeElementHMCSampler(metric_model, mcmc_steps_nbr=int(cfg.mcmc_steps), n_lf=int(cfg.n_lf), eps_lf=float(cfg.eps_lf), beta_zero=float(cfg.beta_zero))
    else:
        sampler = RiemannianHMCSampler(metric_model, mcmc_steps_nbr=int(cfg.mcmc_steps), n_lf=int(cfg.n_lf), eps_lf=float(cfg.eps_lf), beta_zero=float(cfg.beta_zero))

    z = sampler.sample(int(cfg.n_samples))

    # Save outputs
    torch.save({
        'z_samples': z.cpu(),
        'acceptance_rate': getattr(sampler, 'last_acceptance_rate', None),
        'centroids': centroids.cpu(),
        'metric_matrices': M.cpu(),
        'temperature': T,
        'regularization': lbd,
    }, out_dir / 'rhmc_samples.pt')

    _plot_scatter(z, out_dir / 'rhmc_samples_scatter.png', title=f"RHMC samples (acc={getattr(sampler, 'last_acceptance_rate', np.nan):.3f})")
    # Metric visualization in PCA(2) with overlay
    try:
        _plot_metric_contour_and_samples(metric_model, centroids, z, out_dir, grid_size=64)
    except Exception as e:
        print(f"[WARN] metric contour plot failed: {e}")
    print(f"✅ Saved samples and plot to: {out_dir}")


if __name__ == "__main__":
    run()  # type: ignore


