#!/usr/bin/env python3
"""
Utility for materialising ellipse sequence tensors using the unified settings tree.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.data.ellipse_sequences import EllipseSequenceDataset


def _to_dict(cfg: Any) -> Dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)  # type: ignore[return-value]
    if isinstance(cfg, dict):
        return dict(cfg)
    raise TypeError(f"Unsupported cfg type: {type(cfg)}")


def _dataset_from_params(params: Dict[str, Any]) -> EllipseSequenceDataset:
    return EllipseSequenceDataset(
        num_sequences=int(params.get("num_sequences", 1000)),
        seq_len=int(params.get("seq_len", 8)),
        image_size=tuple(params.get("image_size", (64, 64))),
        min_eccentricity=float(params.get("min_eccentricity", 0.0)),
        max_eccentricity=float(params.get("max_eccentricity", 0.9)),
        min_radius=int(params.get("min_radius", 8)),
        max_radius=int(params.get("max_radius", 20)),
        center_jitter=int(params.get("center_jitter", 4)),
        antialias=bool(params.get("antialias", False)),
        seed=int(params.get("seed", 42)),
        fix_center=bool(params.get("fix_center", False)),
        fix_theta=bool(params.get("fix_theta", False)),
        fix_intensity=bool(params.get("fix_intensity", False)),
        keep_major_axis_constant=bool(params.get("keep_major_axis_constant", True)),
        keep_area_constant=bool(params.get("keep_area_constant", False)),
        outline_only=bool(params.get("outline_only", False)),
        outline_width=int(params.get("outline_width", 2)),
        schedule_type=str(params.get("schedule_type", "sinusoidal")),
        sinusoidal_amplitude_range=tuple(params.get("sinusoidal_amplitude_range", (0.25, 0.60))),
        sinusoidal_phase_range=tuple(params.get("sinusoidal_phase_range", (0.0, 2 * 3.141592653589793))),
        sinusoidal_center=params.get("sinusoidal_center", None),
        sinusoidal_cycle=bool(params.get("sinusoidal_cycle", False)),
        sinusoidal_frequency=float(params.get("sinusoidal_frequency", 1.0)),
    )


def _save_split(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    print(f"💾 Saved {tuple(tensor.shape)} to {path}")


def _materialize_subset(ds: EllipseSequenceDataset, indices: Tuple[int, ...]) -> torch.Tensor:
    seqs = [ds[i][0] for i in indices]
    return torch.stack(seqs, dim=0)


def _build_params_from_settings(cfg: DictConfig, dataset: str, overrides: argparse.Namespace) -> Dict[str, Any]:
    data_cfg = cfg.settings.data
    dataset_name = dataset or str(data_cfg.dataset)
    base = _to_dict(data_cfg.common)
    variant = {}
    if hasattr(data_cfg, dataset_name):
        variant = _to_dict(getattr(data_cfg, dataset_name))
    params: Dict[str, Any] = {**base, **variant}

    seq_len = params.pop("sequence_length", overrides.sequence_length or params.get("seq_len", 8))
    params["seq_len"] = int(overrides.sequence_length or seq_len)
    params["image_size"] = tuple(params.get("image_size", (64, 64)))
    params["num_sequences"] = int(overrides.num_sequences or params.get("num_sequences", 1000))
    params["seed"] = int(overrides.seed if overrides.seed is not None else params.get("seed", 42))
    params.setdefault("schedule_type", "sinusoidal")
    return params


def _load_params(config_path: str, dataset: str, overrides: argparse.Namespace) -> Dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    if "settings" in cfg:
        return _build_params_from_settings(cfg, dataset, overrides)
    params = _to_dict(cfg)
    if dataset:
        params["dataset"] = dataset
    params["seq_len"] = int(overrides.sequence_length or params.get("seq_len", params.pop("sequence_length", 8)))
    params["num_sequences"] = int(overrides.num_sequences or params.get("num_sequences", 1000))
    params["seed"] = int(overrides.seed if overrides.seed is not None else params.get("seed", 42))
    params["image_size"] = tuple(params.get("image_size", params.get("resolution", (64, 64))))
    return params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ellipse sequence tensors using unified settings.")
    parser.add_argument("--config", default="conf/config.yaml", help="Path to the mega-config or legacy data YAML.")
    parser.add_argument("--dataset", default=None, help="Dataset key under settings.data.* (defaults to settings.data.dataset).")
    parser.add_argument("--num-sequences", type=int, default=None, help="Number of sequences to generate (override).")
    parser.add_argument("--sequence-length", type=int, default=None, help="Sequence length override.")
    parser.add_argument("--seed", type=int, default=None, help="Dataset RNG seed override.")
    parser.add_argument("--train-ratio", type=float, default=None, help="Training split ratio (default from config or 0.8).")
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation split ratio (default from config or 0.1).")
    parser.add_argument("--train-path", type=Path, default=None, help="Output path for train tensor.")
    parser.add_argument("--val-path", type=Path, default=None, help="Output path for val tensor.")
    parser.add_argument("--test-path", type=Path, default=None, help="Output path for test tensor.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = _load_params(args.config, args.dataset, args)
    ds = _dataset_from_params(params)
    N = len(ds)
    print(f"Dataset: N={N}, T={ds.seq_len}, HxW={ds.H}x{ds.W}")

    cfg_ratios = {
        "train_ratio": params.get("train_ratio", 0.8),
        "val_ratio": params.get("val_ratio", 0.1),
    }
    train_ratio = float(args.train_ratio if args.train_ratio is not None else cfg_ratios["train_ratio"])
    val_ratio = float(args.val_ratio if args.val_ratio is not None else cfg_ratios["val_ratio"])
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)

    n_train = max(1, int(round(N * train_ratio)))
    n_val = max(1, int(round(N * val_ratio)))
    n_test = max(1, N - n_train - n_val)
    if n_train + n_val + n_test != N:
        n_test = N - n_train - n_val

    indices = torch.arange(N).tolist()
    train_idx = tuple(indices[:n_train])
    val_idx = tuple(indices[n_train:n_train + n_val])
    test_idx = tuple(indices[n_train + n_val:])

    train_t = _materialize_subset(ds, train_idx)
    val_t = _materialize_subset(ds, val_idx)
    test_t = _materialize_subset(ds, test_idx)

    def _uniq_frames(x: torch.Tensor) -> int:
        flat = torch.round(x[:, 0].reshape(x.shape[0], -1) * 255).to(torch.int16)
        return int(torch.unique(flat, dim=0).shape[0])

    print(f"Unique t0 frames — train:{_uniq_frames(train_t)} val:{_uniq_frames(val_t)} test:{_uniq_frames(test_t)}")

    default_dir = Path("outputs/data")
    train_path = args.train_path or Path(params.get("train_path", default_dir / "ellipse_train.pt"))
    val_path = args.val_path or Path(params.get("val_path", default_dir / "ellipse_val.pt"))
    test_path = args.test_path or Path(params.get("test_path", default_dir / "ellipse_test.pt"))

    _save_split(train_t, train_path)
    _save_split(val_t, val_path)
    _save_split(test_t, test_path)


if __name__ == "__main__":
    main()
