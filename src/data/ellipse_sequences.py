import math
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset


class EllipseSequenceDataset(Dataset):
    """
    Procedural dataset of sequences of grayscale images containing a single ellipse
    whose eccentricity evolves over time.

    Each sample is a sequence of length `seq_len`, with frames of shape (1, H, W).
    Eccentricity e(t) varies linearly from e_start to e_end per sequence, with
    random start/end ranges to diversify dynamics.
    """

    def __init__(
        self,
        num_sequences: int = 1000,
        seq_len: int = 8,
        image_size: Tuple[int, int] = (64, 64),
        min_eccentricity: float = 0.0,
        max_eccentricity: float = 0.9,
        min_radius: int = 8,
        max_radius: int = 20,
        center_jitter: int = 4,
        antialias: bool = False,
        seed: Optional[int] = 42,
        # DoF control
        fix_center: bool = True,
        fix_theta: bool = True,
        fix_intensity: bool = True,
        keep_major_axis_constant: bool = True,
        keep_area_constant: bool = False,
        # Border-only rendering
        outline_only: bool = True,
        outline_width: int = 4,
        # Eccentricity schedule options
        schedule_type: str = 'linear',
        sinusoidal_amplitude_range: Tuple[float, float] = (0.35, 0.45),
        sinusoidal_phase_range: Tuple[float, float] = (0.0, 2 * math.pi),
        sinusoidal_center: Optional[float] = None,
        sinusoidal_cycle: bool = False,
        sinusoidal_frequency: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_sequences = int(num_sequences)
        self.seq_len = int(seq_len)
        self.H, self.W = int(image_size[0]), int(image_size[1])
        self.min_e = float(min_eccentricity)
        self.max_e = float(max_eccentricity)
        self.min_r = int(min_radius)
        self.max_r = int(max_radius)
        self.center_jitter = int(center_jitter)
        self.antialias = bool(antialias)
        self.fix_center = bool(fix_center)
        self.fix_theta = bool(fix_theta)
        self.fix_intensity = bool(fix_intensity)
        self.keep_major_axis_constant = bool(keep_major_axis_constant)
        self.keep_area_constant = bool(keep_area_constant)
        self.outline_only = bool(outline_only)
        self.outline_width = int(outline_width)

        self.schedule_type = str(schedule_type).lower()
        self.sinusoidal_amplitude_range = tuple(float(x) for x in sinusoidal_amplitude_range)
        self.sinusoidal_phase_range = tuple(float(x) for x in sinusoidal_phase_range)
        self.sinusoidal_center = float(sinusoidal_center) if sinusoidal_center is not None else None
        self.sinusoidal_cycle = bool(sinusoidal_cycle)
        self.sinusoidal_frequency = float(sinusoidal_frequency)

        if self.schedule_type not in {"linear", "sinusoidal"}:
            raise ValueError(f"Unsupported schedule_type '{schedule_type}'. Use 'linear' or 'sinusoidal'.")

        if seed is not None:
            rng = np.random.default_rng(int(seed))
        else:
            rng = np.random.default_rng()
        self._rng = rng

        # Pre-sample per-sequence parameters for reproducibility
        self._params: list[Dict[str, Any]] = []
        # Optionally fix global parameters for all sequences to reduce DoF
        if self.fix_center:
            global_cy = int(self.H // 2)
            global_cx = int(self.W // 2)
        if self.fix_theta:
            global_theta = 0.0
        if self.fix_intensity:
            global_intensity = 1.0

        for _ in range(self.num_sequences):
            # Base radii and orientation
            a0 = int(rng.integers(self.min_r, self.max_r + 1))
            b0 = int(rng.integers(self.min_r, self.max_r + 1))
            theta = float(global_theta if self.fix_theta else rng.uniform(0, math.pi))

            # Eccentricity schedule
            e_start = float(rng.uniform(self.min_e, self.max_e * 0.5))
            e_end = float(rng.uniform(max(e_start, self.max_e * 0.5), self.max_e))

            # Center position (roughly centered + jitter unless fixed)
            if self.fix_center:
                cy = int(global_cy)
                cx = int(global_cx)
            else:
                cy = int(rng.integers(self.H // 2 - self.center_jitter, self.H // 2 + self.center_jitter + 1))
                cx = int(rng.integers(self.W // 2 - self.center_jitter, self.W // 2 + self.center_jitter + 1))

            # Intensity
            intensity = float(global_intensity if self.fix_intensity else rng.uniform(0.7, 1.0))

            params: Dict[str, Any] = {
                "a0": a0,
                "b0": b0,
                "theta": theta,
                "cy": cy,
                "cx": cx,
                "intensity": intensity,
            }

            if self.schedule_type == 'sinusoidal':
                amp_lo, amp_hi = self.sinusoidal_amplitude_range
                amp_lim = max(1e-6, min(self.max_e - self.min_e, self.max_e - self.min_e) / 2.0)
                amplitude = float(rng.uniform(min(amp_lo, amp_lim), min(amp_hi, amp_lim)))
                phase = float(rng.uniform(self.sinusoidal_phase_range[0], self.sinusoidal_phase_range[1]))
                base = self.sinusoidal_center
                if base is None:
                    base = 0.5 * (self.max_e + self.min_e)
                base = float(np.clip(base, self.min_e, self.max_e))
                params["schedule"] = {
                    "type": "sinusoidal",
                    "base": base,
                    "amplitude": amplitude,
                    "phase": phase,
                    "period": 2 * math.pi,
                }
                # Store whether the sequence should be cyclic (first == last)
                params["cycle"] = self.sinusoidal_cycle
            else:
                params["schedule"] = {
                    "type": "linear",
                    "start": e_start,
                    "end": e_end,
                }
                params["cycle"] = False

            self._params.append(params)

        # Prepare coordinate grid
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        self._yy = yy.astype(np.float32)
        self._xx = xx.astype(np.float32)
        # No supersampling for simplified renderer
        self._yy_hr = None
        self._xx_hr = None

    def __len__(self) -> int:
        return self.num_sequences

    def _draw_ellipse(self, cy: float, cx: float, a: float, b: float, theta: float, intensity: float) -> np.ndarray:
        # Base grid (simple, no supersampling)
        y_grid = self._yy - cy
        x_grid = self._xx - cx
        a_eff = a
        b_eff = b

        # Rotate by theta
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        xr = x_grid * cos_t + y_grid * sin_t
        yr = -x_grid * sin_t + y_grid * cos_t

        # Implicit ellipse equation: (xr/a)^2 + (yr/b)^2 <= 1
        val = (xr / (a_eff + 1e-6)) ** 2 + (yr / (b_eff + 1e-6)) ** 2
        img = np.zeros((self.H, self.W), dtype=np.float32)

        if self.outline_only:
            # Approximate border band in implicit space around val == 1
            # Width normalization: convert pixel width to normalized ellipse space via major axis
            # Width defined in pixels; convert to implicit space via major axis
            denom = max(a_eff, b_eff) + 1e-6
            width_norm = float(self.outline_width) / float(denom)
            band_mask = (np.abs(val - 1.0) <= width_norm)
            img[band_mask] = intensity
        else:
            mask = val <= 1.0
            img[mask] = intensity

        return img

    def _eccentricity_to_axes(self, a_base: float, b_base: float, e: float) -> Tuple[float, float]:
        # Ensure a0 >= b0
        a0 = max(a_base, b_base)
        b0 = min(a_base, b_base)

        if self.keep_major_axis_constant:
            # Keep a fixed, vary b with eccentricity: b = a * sqrt(1 - e^2)
            a_scaled = float(a0)
            b_scaled = float(a0) * float(np.sqrt(max(1e-6, 1.0 - e * e)))
        else:
            # Start from a0,b0 and modify toward target eccentricity while preserving area if requested
            a_scaled = float(a0)
            b_scaled = float(a0) * float(np.sqrt(max(1e-6, 1.0 - e * e)))
            if self.keep_area_constant:
                target_area = math.pi * a0 * b0
                cur_area = math.pi * a_scaled * b_scaled
                scale = math.sqrt(max(1e-6, target_area / (cur_area + 1e-6)))
                a_scaled *= scale
                b_scaled *= scale
        return a_scaled, b_scaled

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        p = self._params[idx]
        a0 = p["a0"]
        b0 = p["b0"]
        theta = p["theta"]
        cy = p["cy"]
        cx = p["cx"]
        intensity = p["intensity"]
        schedule = p["schedule"]
        cycle = bool(p.get("cycle", False))

        frames = []
        for t in range(self.seq_len):
            if schedule["type"] == "sinusoidal":
                if self.seq_len <= 1:
                    alpha = 0.0
                else:
                    alpha = t / self.seq_len
                freq = float(self.sinusoidal_frequency)
                # If we require exact cycle closure, snap to nearest integer frequency
                if cycle:
                    freq = max(1.0, round(freq))
                angle = schedule["phase"] + alpha * (2 * math.pi * freq)
                e_t = schedule["base"] + schedule["amplitude"] * math.sin(angle)
                e_t = float(np.clip(e_t, self.min_e, self.max_e))
            else:
                alpha = 0.0 if self.seq_len == 1 else t / (self.seq_len - 1)
                e_start = schedule["start"]
                e_end = schedule["end"]
                e_t = (1.0 - alpha) * e_start + alpha * e_end
            a_t, b_t = self._eccentricity_to_axes(a0, b0, e_t)
            img = self._draw_ellipse(cy, cx, a_t, b_t, theta, intensity)
            frames.append(torch.from_numpy(img)[None, ...])  # (1, H, W)

        if cycle and len(frames) > 1:
            frames[-1] = frames[0].clone()

        seq = torch.stack(frames, dim=0)  # (T, 1, H, W)
        return seq, 0
