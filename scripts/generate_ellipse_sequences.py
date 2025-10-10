#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Tuple

import torch
from omegaconf import OmegaConf

from src.data.ellipse_sequences import EllipseSequenceDataset


def _dataset_from_conf(cfg) -> EllipseSequenceDataset:
    return EllipseSequenceDataset(
        num_sequences=int(cfg.get("num_sequences", 1000)),
        seq_len=int(cfg.get("seq_len", 8)),
        image_size=tuple(cfg.get("image_size", (64, 64))),
        min_eccentricity=float(cfg.get("min_eccentricity", 0.0)),
        max_eccentricity=float(cfg.get("max_eccentricity", 0.9)),
        min_radius=int(cfg.get("min_radius", 8)),
        max_radius=int(cfg.get("max_radius", 20)),
        center_jitter=int(cfg.get("center_jitter", 4)),
        antialias=bool(cfg.get("antialias", False)),
        seed=int(cfg.get("seed", 42)),
        fix_center=bool(cfg.get("fix_center", False)),
        fix_theta=bool(cfg.get("fix_theta", False)),
        fix_intensity=bool(cfg.get("fix_intensity", False)),
        keep_major_axis_constant=bool(cfg.get("keep_major_axis_constant", True)),
        keep_area_constant=bool(cfg.get("keep_area_constant", False)),
        outline_only=bool(cfg.get("outline_only", False)),
        outline_width=int(cfg.get("outline_width", 2)),
        schedule_type=str(cfg.get("schedule_type", "sinusoidal")),
        sinusoidal_amplitude_range=tuple(cfg.get("sinusoidal_amplitude_range", (0.25, 0.60))),
        sinusoidal_phase_range=tuple(cfg.get("sinusoidal_phase_range", (0.0, 2 * 3.141592653589793))),
        sinusoidal_center=cfg.get("sinusoidal_center", None),
        sinusoidal_cycle=bool(cfg.get("sinusoidal_cycle", False)),
        sinusoidal_frequency=float(cfg.get("sinusoidal_frequency", 1.0)),
    )


def _save_split(tensor: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    print(f"Saved {tuple(tensor.shape)} to {path}")


def _materialize_subset(ds: EllipseSequenceDataset, indices: Tuple[int, ...]) -> torch.Tensor:
    # Stack sequences into a single tensor [N, T, 1, H, W]
    seqs = [ds[i][0] for i in indices]
    return torch.stack(seqs, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Generate ellipse sequence tensors from a data config YAML")
    parser.add_argument("--config", required=True, help="Path to data YAML (e.g., conf/data/ellipse_sequences_sinusoidal.yaml)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    ds = _dataset_from_conf(cfg)
    N = len(ds)
    print(f"Dataset: N={N}, T={ds.seq_len}, HxW={ds.H}x{ds.W}")

    # Split ratios from config or defaults 0.8/0.1/0.1
    tr = float(cfg.get("train_ratio", 0.8)); vr = float(cfg.get("val_ratio", 0.1)); te = 1.0 - tr - vr
    n_train = max(1, int(N * tr)); n_val = max(1, int(N * vr)); n_test = max(1, N - n_train - n_val)
    if n_train + n_val + n_test != N:
        n_test = N - n_train - n_val

    # Deterministic split
    all_idx = torch.arange(N).tolist()
    train_idx = tuple(all_idx[:n_train])
    val_idx = tuple(all_idx[n_train:n_train+n_val])
    test_idx = tuple(all_idx[n_train+n_val:])

    train_t = _materialize_subset(ds, train_idx)
    val_t = _materialize_subset(ds, val_idx)
    test_t = _materialize_subset(ds, test_idx)

    # Quick uniqueness diagnostic at t=0
    def _uniq_frames(x: torch.Tensor) -> int:
        flat = torch.round(x[:, 0].reshape(x.shape[0], -1) * 255).to(torch.int16)
        return int(torch.unique(flat, dim=0).shape[0])
    print(f"Unique t0 frames — train:{_uniq_frames(train_t)} val:{_uniq_frames(val_t)} test:{_uniq_frames(test_t)}")

    train_path = Path(cfg.get("train_path", "outputs/data/ellipse_sinusoidal_train.pt"))
    val_path = Path(cfg.get("val_path", "outputs/data/ellipse_sinusoidal_val.pt"))
    test_path = Path(cfg.get("test_path", "outputs/data/ellipse_sinusoidal_test.pt"))
    _save_split(train_t, train_path)
    _save_split(val_t, val_path)
    _save_split(test_t, test_path)


if __name__ == "__main__":
    main()

