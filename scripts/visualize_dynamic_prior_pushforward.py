#!/usr/bin/env python3
"""
Visualize the dynamic Stage‑C prior (½ log|G_T^{-1}|) after pushing a grid through the flows.

Procedure (matches the planned Graph Enhanced KL at T):
1) Build a grid in S-space around provided RHMC samples z_S.
2) Push the grid through the flow stack to obtain z_T coordinates.
3) Use LossManager._pushforward_metric_via_flows to get ½ log|G_T^{-1}| on each grid point.
4) Scatter-plot z_T with colour = ½ log|G_T^{-1}| and overlay the actual posterior z_T samples.

Usage:
    python scripts/visualize_dynamic_prior_pushforward.py \\
        --checkpoint outputs/checkpoints/epoch=04-val_loss=-0.207.ckpt \\
        --stage-zS-path path/to/stage_zS.pt \\
        --output dynamic_prior.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib

# Use a non-interactive backend for headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from training.lightning_trainer import LightningRlVAETrainer  # noqa: E402


def _select_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _load_trainer(checkpoint: Path, device: torch.device) -> LightningRlVAETrainer:
    """Restore the Lightning wrapper and its underlying model from checkpoint."""
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config = state.get("hyper_parameters", None)
    if config is None:
        raise RuntimeError("Checkpoint is missing 'hyper_parameters'; cannot rebuild model.")
    # Detach from the checkpoint object to avoid side effects in Lightning init.
    config_detached = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    # Hard-disable heavy eval/viz blocks for quick offline visualization.
    try:
        if "visualization" in config_detached:
            config_detached.visualization.enabled = False
        if "evaluation" in config_detached:
            config_detached.evaluation.enabled = False
        if "settings" in config_detached:
            if "visualization" in config_detached.settings:
                config_detached.settings.visualization.enabled = False
                config_detached.settings.visualization.level = "none"
            if "evaluation" in config_detached.settings:
                config_detached.settings.evaluation.enabled = False
    except Exception:
        pass
    trainer = LightningRlVAETrainer(config=config_detached)
    trainer.load_state_dict(state["state_dict"], strict=False)
    trainer.to(device)
    trainer.eval()
    if hasattr(trainer, "model"):
        trainer.model.to(device)
        trainer.model.eval()
    return trainer


def _pick_tensor(obj: object, *, device: torch.device) -> Optional[torch.Tensor]:
    """Extract a latent tensor from a variety of common container formats."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, dtype=torch.float32)

    if isinstance(obj, dict):
        for key in ("stage_zS", "zS", "zK", "z_samples", "latents", "z"):
            val = obj.get(key)
            if isinstance(val, torch.Tensor):
                return val.to(device=device, dtype=torch.float32)
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, torch.Tensor):
                return item.to(device=device, dtype=torch.float32)
    return None


def _load_stage_zS(path: Optional[Path], device: torch.device, latent_dim: int, max_samples: int) -> torch.Tensor:
    """Load Stage‑B/Stage‑S latents if provided; otherwise fall back to standard normal samples."""
    if path is None:
        print("⚠️ No --stage-zS-path provided; sampling synthetic z_S ~ N(0, I) for extent.")
        return torch.randn(min(max_samples, 2048), latent_dim, device=device)

    obj = torch.load(path, map_location=device)
    tensor = _pick_tensor(obj, device=device)
    if tensor is None:
        raise RuntimeError(f"Could not extract a tensor from {path}")

    if tensor.ndim > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    if tensor.shape[-1] != latent_dim:
        raise ValueError(f"Loaded z_S has latent_dim={tensor.shape[-1]}, expected {latent_dim}")
    if tensor.shape[0] > max_samples:
        tensor = tensor[:max_samples]
    return tensor


def _build_grid(
    z_s: torch.Tensor,
    dims: Sequence[int],
    grid_size: int,
    padding: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a grid in the selected latent plane, filling other dims with mean(z_s)."""
    d0, d1 = int(dims[0]), int(dims[1])
    mins = z_s[:, [d0, d1]].min(dim=0).values
    maxs = z_s[:, [d0, d1]].max(dim=0).values
    span = maxs - mins
    lo = mins - padding * span
    hi = maxs + padding * span

    xs = torch.linspace(lo[0], hi[0], grid_size, device=z_s.device)
    ys = torch.linspace(lo[1], hi[1], grid_size, device=z_s.device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    grid_plane = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)  # [N, 2]

    base = z_s.mean(dim=0, keepdim=True).repeat(grid_plane.shape[0], 1)  # [N, D]
    base[:, d0] = grid_plane[:, 0]
    base[:, d1] = grid_plane[:, 1]
    return base, grid_plane, lo, hi


def _push_grid(
    grid_s: torch.Tensor,
    flow_manager,
    metric_tensor,
    loss_manager,
    rhmc_posterior=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Push grid to T (for coordinates) and compute ½ log|G_T^{-1}|."""
    if flow_manager is None or getattr(flow_manager, "n_flows", 0) == 0:
        z_t = grid_s
    else:
        with torch.no_grad():
            z_seq, _ = flow_manager.apply_flows([grid_s], n_obs=flow_manager.n_flows + 1)
            z_t = z_seq[-1]

    ((_, _), _, _, half_logdet_push_ginv) = loss_manager._pushforward_metric_via_flows(
        grid_s, flow_manager, metric_tensor, rhmc_posterior
    )
    if half_logdet_push_ginv is None:
        print("⚠️ Pushforward failed (SVD clamp or NaNs). Falling back to base metric without flows.")
        ((_, _), _, _, half_logdet_push_ginv) = loss_manager._pushforward_metric_via_flows(
            grid_s, None, metric_tensor, rhmc_posterior
        )
        if half_logdet_push_ginv is None:
            raise RuntimeError("Pushforward metric computation failed even without flows.")
    return z_t, half_logdet_push_ginv


def _push_samples(flow_manager, z_s: torch.Tensor) -> torch.Tensor:
    """Push actual z_S samples through the flow stack to T."""
    if flow_manager is None or getattr(flow_manager, "n_flows", 0) == 0:
        return z_s
    with torch.no_grad():
        z_seq, _ = flow_manager.apply_flows([z_s], n_obs=flow_manager.n_flows + 1)
        return z_seq[-1]


def _plot(
    z_t_grid: torch.Tensor,
    logdet: torch.Tensor,
    z_t_samples: torch.Tensor,
    dims: Sequence[int],
    output: Path,
) -> None:
    """Scatter plot of transported grid coloured by ½ log|G_T^{-1}| with sample overlay."""
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        z_t_grid[:, 0],
        z_t_grid[:, 1],
        c=logdet,
        cmap="viridis",
        s=12,
        alpha=0.9,
        linewidths=0.0,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("½ log|G_T^{-1}| (pushforward precision)")

    if z_t_samples.numel() > 0:
        ax.scatter(
            z_t_samples[:, 0],
            z_t_samples[:, 1],
            c="red",
            s=20,
            alpha=0.65,
            marker="x",
            label="posterior z_T samples",
        )
        ax.legend(loc="upper right")

    ax.set_xlabel(f"z_T[{dims[0]}]")
    ax.set_ylabel(f"z_T[{dims[1]}]")
    ax.set_title("Dynamic prior at T (pushed grid coloured by ½ log|G_T^{-1}|)")
    with torch.no_grad():
        xmin = min(z_t_grid[:, 0].min(), z_t_samples[:, 0].min() if z_t_samples.numel() > 0 else z_t_grid[:, 0].min())
        xmax = max(z_t_grid[:, 0].max(), z_t_samples[:, 0].max() if z_t_samples.numel() > 0 else z_t_grid[:, 0].max())
        ymin = min(z_t_grid[:, 1].min(), z_t_samples[:, 1].min() if z_t_samples.numel() > 0 else z_t_grid[:, 1].min())
        ymax = max(z_t_grid[:, 1].max(), z_t_samples[:, 1].max() if z_t_samples.numel() > 0 else z_t_grid[:, 1].max())
        pad_x = float(0.05 * max(1e-6, xmax - xmin))
        pad_y = float(0.05 * max(1e-6, ymax - ymin))
    ax.set_xlim(float(xmin - pad_x), float(xmax + pad_x))
    ax.set_ylim(float(ymin - pad_y), float(ymax + pad_y))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"✅ Saved dynamic prior visualization to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize dynamic Stage‑C prior via pushforward metric.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to Stage‑C Lightning checkpoint (.ckpt).")
    parser.add_argument("--stage-zS-path", type=Path, default=None, help="Optional path to saved z_S (torch .pt).")
    parser.add_argument("--grid-size", type=int, default=80, help="Grid resolution per axis (default: 80).")
    parser.add_argument("--padding", type=float, default=0.15, help="Fractional padding beyond z_S min/max (default: 0.15).")
    parser.add_argument(
        "--dims",
        type=int,
        nargs=2,
        default=None,
        help="Latent dims to visualize (default: first two).",
    )
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto (default).")
    parser.add_argument("--max-samples", type=int, default=4000, help="Cap on z_S points used for extent/overlay.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path (defaults next to checkpoint).")
    parser.add_argument("--save-data", type=Path, default=None, help="Optional path to save raw grid/logdet tensors.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _select_device(args.device)
    print(f"▶️ Using device: {device}")

    trainer = _load_trainer(args.checkpoint, device)
    if not hasattr(trainer, "model"):
        raise RuntimeError("Loaded checkpoint does not expose a `.model` attribute.")
    model = trainer.model

    flow_manager = getattr(model, "flow_manager", None)
    loss_manager = getattr(model, "loss_manager", None)
    metric_tensor = getattr(model, "modular_metric", None)
    rhmc_post = getattr(model, "posterior_sampler_rhmc", None)
    if metric_tensor is None or loss_manager is None:
        raise RuntimeError("Model is missing metric_tensor or loss_manager; cannot compute pushforward metric.")

    metric_tensor = metric_tensor.to(device)
    loss_manager = loss_manager.to(device)
    if flow_manager is not None:
        flow_manager = flow_manager.to(device)
        flow_manager.eval()

    latent_dim = int(getattr(model, "latent_dim", getattr(model, "ld", 0)))
    if latent_dim <= 0:
        raise RuntimeError("Could not infer latent_dim from model.")

    dims = args.dims if args.dims is not None else (0, 1)
    if len(dims) != 2:
        raise ValueError("--dims must specify exactly two dimensions.")
    if max(dims) >= latent_dim or min(dims) < 0:
        raise ValueError(f"--dims {dims} out of range for latent_dim={latent_dim}")

    z_s = _load_stage_zS(args.stage_zS_path, device, latent_dim, args.max_samples)
    if z_s.shape[-1] != latent_dim:
        raise RuntimeError(f"z_S latent_dim mismatch: got {z_s.shape[-1]}, expected {latent_dim}")

    grid_s, grid_plane, lo, hi = _build_grid(z_s, dims, args.grid_size, args.padding)

    z_t_grid, half_logdet_push_ginv = _push_grid(grid_s, flow_manager, metric_tensor, loss_manager, rhmc_post)

    z_t_samples = _push_samples(flow_manager, z_s)

    # Project onto selected dims for plotting
    z_t_grid_plot = z_t_grid[:, dims].detach().cpu()
    z_t_samples_plot = z_t_samples[:, dims].detach().cpu()
    logdet_plot = half_logdet_push_ginv.detach().cpu()

    output_path = args.output
    if output_path is None:
        output_path = args.checkpoint.with_name(f"dynamic_prior_pushforward_{args.checkpoint.stem}.png")
    _plot(z_t_grid_plot, logdet_plot, z_t_samples_plot, dims, output_path)

    if args.save_data is not None:
        payload = {
            "grid_zS": grid_s.detach().cpu(),
            "grid_zT": z_t_grid.detach().cpu(),
            "half_logdet_push_ginv": logdet_plot,
            "zT_samples": z_t_samples.detach().cpu(),
            "dims": tuple(dims),
            "lo": lo.cpu(),
            "hi": hi.cpu(),
        }
        args.save_data.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.save_data)
        print(f"💾 Saved raw tensors to {args.save_data}")


if __name__ == "__main__":
    main()
