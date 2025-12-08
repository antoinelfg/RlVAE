import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Subset

try:
    import torchvision.utils as vutils
except Exception:
    vutils = None

from .ellipse_sequences import EllipseSequenceDataset


def _basic_stats(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def _unwrap_base_dataset(data_module) -> Tuple[Optional[EllipseSequenceDataset], Dict[str, np.ndarray]]:
    """
    Recover the underlying EllipseSequenceDataset and split indices from the data module.
    """
    base_dataset = None
    split_indices: Dict[str, np.ndarray] = {}

    def _extract(ds):
        if ds is None:
            return None, None
        candidate = getattr(ds, "_base", ds)
        if isinstance(candidate, Subset):
            return candidate.dataset, np.asarray(candidate.indices, dtype=int)
        if isinstance(ds, Subset):
            return ds.dataset, np.asarray(ds.indices, dtype=int)
        return candidate, None

    for name in ("train_dataset", "val_dataset", "test_dataset"):
        ds = getattr(data_module, name, None)
        base, idx = _extract(ds)
        if base_dataset is None and isinstance(base, EllipseSequenceDataset):
            base_dataset = base
        if idx is not None:
            split_indices[name.replace("_dataset", "")] = idx

    if base_dataset is None and hasattr(data_module, "_build_dataset"):
        try:
            base_dataset = data_module._build_dataset()
        except Exception:
            base_dataset = None

    return base_dataset, split_indices


def generate_ellipse_data_report(
    cfg: DictConfig,
    data_module: Optional[Any] = None,
    output_dir: Optional[Path] = None,
    max_sequences: int = 256,
    pairwise_pairs: int = 120,
    preview_sequences: int = 9,
    log_to_wandb: bool = False,
) -> Dict[str, Any]:
    """
    Build a quick structural report for the ellipse dataset used by run_experiment.

    Returns a dict with numeric summaries and saves a multi-panel figure plus JSON.
    """
    cfg_data = getattr(cfg, "data", cfg)
    dataset_name = str(getattr(cfg_data, "name", getattr(cfg_data, "dataset", "")))
    if "ellipse" not in dataset_name.lower():
        return {"skipped": f"Dataset '{dataset_name}' is not ellipse-based."}

    if data_module is None:
        from .datamodule_factory import build_data_module

        data_module = build_data_module(cfg_data)
        data_module.setup("fit", getattr(cfg, "training", None))

    dataset, split_indices = _unwrap_base_dataset(data_module)
    if not isinstance(dataset, EllipseSequenceDataset):
        return {"skipped": f"Unsupported dataset type: {type(dataset)}"}
    if not dataset._params:
        return {"error": "No ellipse parameters sampled in dataset."}

    a0 = np.array([p["a0"] for p in dataset._params], dtype=float)
    b0 = np.array([p["b0"] for p in dataset._params], dtype=float)
    theta = np.array([p["theta"] for p in dataset._params], dtype=float)
    cy = np.array([p["cy"] for p in dataset._params], dtype=float)
    cx = np.array([p["cx"] for p in dataset._params], dtype=float)
    intensity = np.array([p["intensity"] for p in dataset._params], dtype=float)

    ecc_start = []
    ecc_end = []
    ecc_min = []
    ecc_max = []
    schedule_types = []
    for p in dataset._params:
        schedule = p["schedule"]
        schedule_types.append(schedule["type"])
        if schedule["type"] == "linear":
            e_s = float(schedule["start"])
            e_e = float(schedule["end"])
            ecc_start.append(e_s)
            ecc_end.append(e_e)
            ecc_min.append(min(e_s, e_e))
            ecc_max.append(max(e_s, e_e))
        else:
            base = float(schedule["base"])
            amp = float(schedule["amplitude"])
            ecc_start.append(base - amp)
            ecc_end.append(base + amp)
            ecc_min.append(max(dataset.min_e, base - amp))
            ecc_max.append(min(dataset.max_e, base + amp))
    ecc_start = np.asarray(ecc_start, dtype=float)
    ecc_end = np.asarray(ecc_end, dtype=float)
    ecc_min = np.asarray(ecc_min, dtype=float)
    ecc_max = np.asarray(ecc_max, dtype=float)

    major_min = []
    minor_min = []
    major_max = []
    minor_max = []
    for p, e_lo, e_hi in zip(dataset._params, ecc_min, ecc_max):
        a_lo, b_lo = dataset._eccentricity_to_axes(p["a0"], p["b0"], e_lo)
        a_hi, b_hi = dataset._eccentricity_to_axes(p["a0"], p["b0"], e_hi)
        major_min.append(max(a_lo, b_lo))
        minor_min.append(min(a_lo, b_lo))
        major_max.append(max(a_hi, b_hi))
        minor_max.append(min(a_hi, b_hi))
    major_min = np.asarray(major_min, dtype=float)
    minor_min = np.asarray(minor_min, dtype=float)
    major_max = np.asarray(major_max, dtype=float)
    minor_max = np.asarray(minor_max, dtype=float)

    rng = np.random.default_rng(int(getattr(cfg, "seed", 42)))
    sample_count = min(int(max_sequences), len(dataset))
    sample_indices = rng.choice(len(dataset), size=sample_count, replace=False)
    sample_sequences = torch.stack([dataset[int(i)][0].float() for i in sample_indices], dim=0)
    flat = sample_sequences.view(sample_sequences.shape[0], -1)

    pairwise_stats: Dict[str, float] = {}
    pair_subset = min(int(pairwise_pairs), sample_sequences.shape[0])
    if pair_subset >= 2:
        subset_idx = rng.choice(sample_sequences.shape[0], size=pair_subset, replace=False)
        rms = torch.pdist(flat[subset_idx], p=2) / math.sqrt(flat.shape[1])
        rms_np = rms.cpu().numpy()
        pairwise_stats = {
            "min": float(rms_np.min()),
            "max": float(rms_np.max()),
            "mean": float(rms_np.mean()),
            "median": float(np.median(rms_np)),
            "p10": float(np.percentile(rms_np, 10)),
            "p90": float(np.percentile(rms_np, 90)),
            "n_pairs": int(rms_np.shape[0]),
        }
    else:
        pairwise_stats = {"note": "Not enough samples for pairwise RMS."}

    mean_frame = sample_sequences.mean(dim=(0, 1)).squeeze().cpu().numpy()
    std_frame = sample_sequences.std(dim=(0, 1)).squeeze().cpu().numpy()

    preview_count = min(int(preview_sequences), sample_sequences.shape[0])
    preview = sample_sequences[:preview_count, 0]  # first frame per sequence
    nrow = max(1, int(math.ceil(math.sqrt(preview_count))))
    if vutils is not None:
        grid = vutils.make_grid(preview, nrow=nrow, padding=2, normalize=True, pad_value=0.2)
        grid_img = grid.permute(1, 2, 0).cpu().numpy()
    else:
        # Manual tiling fallback
        c, h, w = preview.shape[1:]
        grid_rows = int(math.ceil(preview_count / nrow))
        canvas = torch.zeros((1, grid_rows * h, nrow * w))
        for idx, frame in enumerate(preview):
            r = idx // nrow
            c_idx = idx % nrow
            canvas[:, r * h : (r + 1) * h, c_idx * w : (c_idx + 1) * w] = frame
        grid_img = canvas.squeeze(0).cpu().numpy()

    out_dir = Path(output_dir) if output_dir is not None else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / "ellipse_data_panel.png"

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    flat_axes = axes.ravel()

    flat_axes[0].hist(ecc_min, bins=30, alpha=0.6, label="e_min")
    flat_axes[0].hist(ecc_max, bins=30, alpha=0.6, label="e_max")
    flat_axes[0].set_title("Eccentricity range per sequence")
    flat_axes[0].legend()

    flat_axes[1].scatter(cx, cy, s=10, alpha=0.6)
    flat_axes[1].set_title("Centers (cx, cy)")
    flat_axes[1].set_xlabel("cx")
    flat_axes[1].set_ylabel("cy")

    flat_axes[2].hist(np.rad2deg(theta), bins=30, color="tab:purple", alpha=0.8)
    flat_axes[2].set_title("Orientation θ (degrees)")

    flat_axes[3].hist(major_min, bins=30, alpha=0.6, label="major @ e_min")
    flat_axes[3].hist(major_max, bins=30, alpha=0.6, label="major @ e_max")
    flat_axes[3].set_title("Major axis lengths")
    flat_axes[3].legend()

    flat_axes[4].hist(intensity, bins=30, color="tab:green", alpha=0.8)
    flat_axes[4].set_title("Intensity")

    if pair_subset >= 2:
        flat_axes[5].hist(rms_np, bins=30, color="tab:red", alpha=0.8)
        flat_axes[5].set_title("Pairwise RMS distance")
    else:
        flat_axes[5].text(0.5, 0.5, "Not enough samples", ha="center", va="center")
        flat_axes[5].set_xticks([])
        flat_axes[5].set_yticks([])

    im0 = flat_axes[6].imshow(mean_frame, cmap="magma")
    flat_axes[6].set_title("Mean frame")
    plt.colorbar(im0, ax=flat_axes[6], fraction=0.046, pad=0.04)

    im1 = flat_axes[7].imshow(std_frame, cmap="magma")
    flat_axes[7].set_title("Std frame")
    plt.colorbar(im1, ax=flat_axes[7], fraction=0.046, pad=0.04)

    flat_axes[8].imshow(grid_img.squeeze(), cmap="gray")
    flat_axes[8].set_title(f"Preview t=0 (n={preview_count})")
    flat_axes[8].axis("off")

    fig.tight_layout()
    fig.savefig(panel_path, dpi=200)
    plt.close(fig)

    schedule_counts = {k: schedule_types.count(k) for k in set(schedule_types)}

    param_stats = {
        "eccentricity_min": _basic_stats(ecc_min),
        "eccentricity_max": _basic_stats(ecc_max),
        "eccentricity_span": _basic_stats(ecc_max - ecc_min),
        "theta_deg": _basic_stats(np.rad2deg(theta)),
        "major_axis_at_min_e": _basic_stats(major_min),
        "major_axis_at_max_e": _basic_stats(major_max),
        "minor_axis_at_min_e": _basic_stats(minor_min),
        "minor_axis_at_max_e": _basic_stats(minor_max),
        "center_x": _basic_stats(cx),
        "center_y": _basic_stats(cy),
        "intensity": _basic_stats(intensity),
    }

    split_stats: Dict[str, Any] = {}
    for split_name, idxs in split_indices.items():
        split_stats[split_name] = {
            "n_sequences": int(len(idxs)),
            "eccentricity_min": _basic_stats(ecc_min[idxs]),
            "eccentricity_max": _basic_stats(ecc_max[idxs]),
            "major_axis_at_min_e": _basic_stats(major_min[idxs]),
            "major_axis_at_max_e": _basic_stats(major_max[idxs]),
        }

    n_centroids_cfg = None
    centroid_method_cfg = ""
    use_stage_b = False
    if hasattr(cfg, "experiment"):
        try:
            n_centroids_cfg = getattr(cfg.experiment.stage_b, "n_centroids", None)
            centroid_method_cfg = getattr(cfg.experiment.stage_b, "centroid_method", "")
            use_stage_b = bool(getattr(cfg.experiment, "run_stage_b", False))
        except Exception:
            n_centroids_cfg = None
    centroid_info = {
        "configured_n_centroids": int(n_centroids_cfg) if n_centroids_cfg is not None else None,
        "centroid_method": str(centroid_method_cfg),
        "use_stage_b": use_stage_b,
    }

    report: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "num_sequences": int(len(dataset)),
        "sequence_length": int(dataset.seq_len),
        "image_size": [int(dataset.H), int(dataset.W)],
        "splits": {k: int(len(v)) for k, v in split_indices.items()},
        "schedule_types": schedule_counts,
        "pairwise_rms": pairwise_stats,
        "param_stats": param_stats,
        "split_param_stats": split_stats,
        "centroids": centroid_info,
        "sampled_sequences_for_panel": int(sample_sequences.shape[0]),
        "panel_path": str(panel_path),
    }

    report_path = out_dir / "ellipse_data_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    report["report_path"] = str(report_path)

    if log_to_wandb:
        try:
            import wandb

            if wandb.run is not None:
                wandb.log({"data/ellipse_panel": wandb.Image(str(panel_path))})
        except Exception as e:
            print(f"[DATA ANALYSIS] WandB logging skipped: {e}")

    print(f"[DATA ANALYSIS] Saved ellipse data panel to {panel_path}")
    return report
