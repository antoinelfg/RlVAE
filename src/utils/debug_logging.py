"""
Utility helpers to convert verbose numerical diagnostics into WandB-friendly
figures. Logging can be enabled by setting the environment variable
``RLVAE_WANDB_DEBUG=1``.
"""
from __future__ import annotations

import os
import math
from collections import defaultdict
from typing import Dict, Iterable, Optional, Sequence, List

import numpy as np

try:  # Optional dependency
    import wandb  # type: ignore
except Exception:  # pragma: no cover - graceful degradation when wandb missing
    wandb = None  # type: ignore


class DebugPlotLogger:
    """
    Centralised WandB logger for heavy debugging plots.

    Usage:
        DebugPlotLogger.log_initial_sampling(...)

    Logging only triggers when:
        * ``RLVAE_WANDB_DEBUG=1`` and
        * ``wandb.run`` is active.

    Controls:
        * ``RLVAE_WANDB_DEBUG_MAX``: maximum plots per tag (``-1``/``inf`` for unlimited).
        * ``RLVAE_WANDB_DEBUG_EVERY``: log every Nth invocation of a tag (default=1).
    """

    _tag_counts: Dict[str, int] = defaultdict(int)
    _tag_invocations: Dict[str, int] = defaultdict(int)

    @classmethod
    def _is_enabled(cls) -> bool:
        if os.environ.get("RLVAE_WANDB_DEBUG", "0") != "1":
            return False
        if wandb is None:
            return False
        return getattr(wandb, "run", None) is not None

    @classmethod
    def _log_frequency(cls) -> int:
        raw = os.environ.get("RLVAE_WANDB_DEBUG_EVERY", "1")
        try:
            value = int(raw)
        except ValueError:
            value = 1
        return max(1, value)

    @classmethod
    def _max_per_tag(cls) -> Optional[int]:
        raw = os.environ.get("RLVAE_WANDB_DEBUG_MAX", "6").strip().lower()
        if raw in {"-1", "none", "inf", "infinite", "unlimited", "all"}:
            return None
        try:
            value = int(raw)
        except ValueError:
            return 6
        if value <= 0:
            return None
        return value

    @classmethod
    def _should_log(cls, tag: str) -> bool:
        if not cls._is_enabled():
            return False
        cls._tag_invocations[tag] += 1
        freq = cls._log_frequency()
        if freq > 1 and (cls._tag_invocations[tag] - 1) % freq != 0:
            return False
        max_per_tag = cls._max_per_tag()
        if max_per_tag is not None and cls._tag_counts[tag] >= max_per_tag:
            return False
        cls._tag_counts[tag] += 1
        return True

    @staticmethod
    def _to_numpy(data) -> Optional[np.ndarray]:
        if data is None:
            return None
        try:
            import torch

            if isinstance(data, torch.Tensor):
                if data.numel() == 0:
                    return None
                return data.detach().float().cpu().numpy()
        except Exception:
            pass
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, (list, tuple)):
            if len(data) == 0:
                return None
            return np.asarray(data)
        try:
            return np.asarray(data)
        except Exception:
            return None

    @staticmethod
    def _log_wandb(tag: str, fig, extra: Optional[Dict[str, float]] = None) -> None:
        if wandb is None or getattr(wandb, "run", None) is None:
            import matplotlib.pyplot as plt

            plt.close(fig)
            return
        payload = {f"Debug/{tag}": wandb.Image(fig)}
        if extra:
            for key, value in extra.items():
                if value is None or (isinstance(value, float) and not math.isfinite(value)):
                    continue
                payload[f"Debug/{tag}/{key}"] = value
        wandb.log(payload, commit=False)
        import matplotlib.pyplot as plt

        plt.close(fig)

    @classmethod
    def log_initial_sampling(
        cls,
        *,
        logdet_mu,
        logdet_z0,
        distances,
        mahal_sq=None,
        sigma_eigs=None,
        tag: str = "initial_sampling",
    ) -> None:
        if not cls._should_log(tag):
            return
        mu_vals = cls._to_numpy(logdet_mu)
        z_vals = cls._to_numpy(logdet_z0)
        dist_vals = cls._to_numpy(distances)
        mahal_vals = cls._to_numpy(mahal_sq)
        if mu_vals is None or z_vals is None or dist_vals is None:
            return
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=120)
        bins = int(os.environ.get("RLVAE_WANDB_DEBUG_BINS", 30))
        axes[0].hist(mu_vals, bins=bins, alpha=0.6, label="μ")
        axes[0].hist(z_vals, bins=bins, alpha=0.6, label="z₀")
        axes[0].set_title("log|G⁻¹| distribution")
        axes[0].set_xlabel("log|G⁻¹|")
        axes[0].set_ylabel("count")
        axes[0].legend()

        if mahal_vals is not None:
            scatter = axes[1].scatter(dist_vals, z_vals, c=mahal_vals, s=16, cmap="viridis")
            fig.colorbar(scatter, ax=axes[1], label="Mahalanobis²")
        else:
            axes[1].scatter(dist_vals, z_vals, s=16, alpha=0.7)
        axes[1].set_xlabel("||z₀ - μ||")
        axes[1].set_ylabel("log|G⁻¹(z₀)|")
        axes[1].set_title("Distance vs log|G⁻¹|")
        fig.suptitle("Initial Sampling Diagnostics")

        extra = {
            "mu_logdet_mean": float(np.mean(mu_vals)),
            "z0_logdet_mean": float(np.mean(z_vals)),
            "distance_mean": float(np.mean(dist_vals)),
        }
        if sigma_eigs is not None:
            eigs = cls._to_numpy(sigma_eigs)
            if eigs is not None:
                extra["sigma_eig_min"] = float(np.min(eigs))
                extra["sigma_eig_max"] = float(np.max(eigs))
        cls._log_wandb(tag, fig, extra)

    @classmethod
    def log_metric_distribution(
        cls,
        *,
        logdet_mu,
        logdet_z0,
        distances,
        logdet_base=None,
        tag: str = "metric_distribution",
    ) -> None:
        if not cls._should_log(tag):
            return
        mu_vals = cls._to_numpy(logdet_mu)
        z_vals = cls._to_numpy(logdet_z0)
        dist_vals = cls._to_numpy(distances)
        base_vals = cls._to_numpy(logdet_base)
        if mu_vals is None or z_vals is None or dist_vals is None:
            return
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=120)
        bins = int(os.environ.get("RLVAE_WANDB_DEBUG_BINS", 30))
        axes[0].hist(z_vals, bins=bins, alpha=0.7, label="z₀")
        axes[0].hist(mu_vals, bins=bins, alpha=0.5, label="μ")
        if base_vals is not None:
            axes[0].hist(base_vals, bins=bins, alpha=0.4, label="z_base")
        axes[0].legend()
        axes[0].set_title("log|G⁻¹| distribution (μ vs z₀)")
        axes[0].set_xlabel("log|G⁻¹|")

        axes[1].scatter(dist_vals, z_vals, s=16, alpha=0.7)
        axes[1].set_xlabel("||z - μ||")
        axes[1].set_ylabel("log|G⁻¹(z)||")
        axes[1].set_title("Distance / Volume correlation")

        extra = {
            "mu_logdet_mean": float(np.mean(mu_vals)),
            "z0_logdet_mean": float(np.mean(z_vals)),
            "dist_logdet_corr": float(
                np.corrcoef(dist_vals, z_vals)[0, 1]
            )
            if len(dist_vals) > 1
            else float("nan"),
        }
        cls._log_wandb(tag, fig, extra)

    @classmethod
    def log_sigma_consistency(
        cls,
        *,
        diff_loss_rhmc: Optional[float] = None,
        diff_loss_cache: Optional[float] = None,
        logdet_loss: Optional[float] = None,
        logdet_ref: Optional[float] = None,
        tag: str = "sigma_consistency",
    ) -> None:
        if not cls._should_log(tag):
            return
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), dpi=120)
        labels = []
        values = []
        if diff_loss_rhmc is not None:
            labels.append("||Σ_loss-Σ_rhmc||_F")
            values.append(diff_loss_rhmc)
        if diff_loss_cache is not None:
            labels.append("||Σ_loss-Σ_cache||_F")
            values.append(diff_loss_cache)
        if labels:
            axes[0].bar(labels, values, color="tab:blue")
            axes[0].set_ylabel("Frobenius norm")
            axes[0].set_title("Σ consistency")
        else:
            axes[0].axis("off")

        if logdet_loss is not None or logdet_ref is not None:
            loss_val = float(logdet_loss) if logdet_loss is not None else float("nan")
            ref_val = float(logdet_ref) if logdet_ref is not None else float("nan")
            axes[1].bar(["Σ_loss", "Σ_ref"], [loss_val, ref_val], color=["tab:green", "tab:orange"])
            axes[1].set_ylabel("log|Σ|")
            axes[1].set_title("logdet comparison")
        else:
            axes[1].axis("off")

        extra = {}
        if diff_loss_rhmc is not None:
            extra["diff_loss_rhmc"] = float(diff_loss_rhmc)
        if diff_loss_cache is not None:
            extra["diff_loss_cache"] = float(diff_loss_cache)
        if logdet_loss is not None:
            extra["logdet_loss"] = float(logdet_loss)
        if logdet_ref is not None:
            extra["logdet_ref"] = float(logdet_ref)
        cls._log_wandb(tag, fig, extra)

    @classmethod
    def log_rhmc_trajectory(
        cls,
        *,
        logdets: Sequence[float],
        distances: Sequence[float],
        momenta: Optional[Sequence[float]] = None,
        tag: str = "rhmc_trajectory",
    ) -> None:
        if not cls._should_log(tag):
            return
        logdets_np = cls._to_numpy(logdets)
        dists_np = cls._to_numpy(distances)
        momenta_np = cls._to_numpy(momenta)
        if logdets_np is None or dists_np is None or logdets_np.size < 2:
            return
        steps = np.arange(len(logdets_np))
        import matplotlib.pyplot as plt

        cols = 3 if momenta_np is not None else 2
        fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 3.2), dpi=120)
        axes = np.atleast_1d(axes)
        axes[0].plot(steps, logdets_np, marker="o")
        axes[0].set_title("log|G⁻¹| over leapfrog steps")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("log|G⁻¹|")

        axes[1].plot(steps, dists_np, marker="o", color="tab:orange")
        axes[1].set_title("||z - μ|| over steps")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("distance")

        if momenta_np is not None and len(axes) > 2:
            axes[2].plot(steps[: len(momenta_np)], momenta_np, marker="o", color="tab:green")
            axes[2].set_title("||ρ|| over steps")
            axes[2].set_xlabel("step")
            axes[2].set_ylabel("momentum norm")

        extra = {
            "logdet_delta": float(logdets_np[-1] - logdets_np[0]),
            "distance_delta": float(dists_np[-1] - dists_np[0]),
        }
        cls._log_wandb(tag, fig, extra)

    @classmethod
    def log_kl_components(
        cls,
        *,
        log_q: float,
        log_p: float,
        flow: float,
        delta_kin: float,
        delta_vol: float,
        kl_mean: float,
        tag: str = "kl_components",
    ) -> None:
        if not cls._should_log(tag):
            return
        import matplotlib.pyplot as plt

        labels = ["log_q", "log_p", "flow", "Δkin", "Δvol", "KL"]
        values = [log_q, log_p, flow, delta_kin, delta_vol, kl_mean]
        fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
        ax.bar(labels, values, color=["tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:gray", "tab:red"])
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Contribution")
        ax.set_title("KL decomposition")
        cls._log_wandb(tag, fig, {"kl_mean": kl_mean})

    @classmethod
    def log_alignment(
        cls,
        *,
        delta_norm,
        cosines,
        grad_norm,
        tag: str = "logq_alignment",
    ) -> None:
        if not cls._should_log(tag):
            return
        delta_np = cls._to_numpy(delta_norm)
        cos_np = cls._to_numpy(cosines)
        grad_np = cls._to_numpy(grad_norm)
        if delta_np is None or cos_np is None:
            return
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=120)
        axes[0].hist(delta_np, bins=30, color="tab:blue", alpha=0.8)
        axes[0].set_title("||z₀ - μ|| histogram")
        axes[1].hist(cos_np, bins=30, color="tab:orange", alpha=0.8)
        axes[1].set_title("cos alignment histogram")
        axes[2].scatter(delta_np, cos_np, s=12, alpha=0.7)
        axes[2].set_xlabel("||z₀ - μ||")
        axes[2].set_ylabel("cos(delta, ∇μ ½log|G⁻¹|)")
        axes[2].set_title("Alignment scatter")
        extra = {
            "cos_mean": float(np.mean(cos_np)),
            "cos_std": float(np.std(cos_np)),
        }
        if grad_np is not None:
            extra["grad_norm_mean"] = float(np.mean(grad_np))
        cls._log_wandb(tag, fig, extra)

    @classmethod
    def log_sigma_spectrum(
        cls,
        eigvals,
        *,
        tag: str = "sigma_spectrum",
        cond_threshold: float = 50.0,
        eig_min_threshold: float = 1e-3,
    ) -> None:
        if not cls._should_log(tag):
            return
        vals = cls._to_numpy(eigvals)
        if vals is None:
            return
        flat = vals.reshape(-1)
        if flat.size == 0:
            return
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        ax.boxplot(flat, vert=True, widths=0.4)
        ax.set_ylabel("Eigenvalue")
        ax.set_title("Σ eigenvalue spectrum")
        ax.axhline(eig_min_threshold, color="red", linestyle="--", label="min threshold")
        ax.legend(loc="upper right")

        cond = float(np.max(flat) / max(np.min(flat), 1e-12))
        extra = {
            "eig_min": float(np.min(flat)),
            "eig_max": float(np.max(flat)),
            "cond": cond,
        }
        if cond > cond_threshold:
            ax.text(
                1.05,
                np.max(flat),
                f"cond={cond:.1f} > {cond_threshold}",
                color="red",
            )
        cls._log_wandb(tag, fig, extra)

    @classmethod
    def log_metric_raw(
        cls,
        *,
        logdet_mu,
        logdet_z0=None,
        logdet_base=None,
        fallback=None,
        tag: str = "metric_raw",
    ) -> None:
        if not cls._should_log(tag):
            return
        mu_vals = cls._to_numpy(logdet_mu)
        z_vals = cls._to_numpy(logdet_z0)
        base_vals = cls._to_numpy(logdet_base)
        if mu_vals is None:
            return
        import matplotlib.pyplot as plt

        cols = 1 + int(z_vals is not None) + int(base_vals is not None)
        fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 3.2), dpi=120)
        axes = np.atleast_1d(axes)
        bins = int(os.environ.get("RLVAE_WANDB_DEBUG_BINS", 30))

        axes[0].hist(mu_vals, bins=bins, color="tab:green", alpha=0.8)
        axes[0].set_title("log|G⁻¹(μ)|")
        if z_vals is not None:
            axes[1].hist(z_vals, bins=bins, color="tab:blue", alpha=0.8)
            axes[1].set_title("log|G⁻¹(z₀)|")
        if base_vals is not None:
            idx = 2 if z_vals is not None else 1
            axes[idx].hist(base_vals, bins=bins, color="tab:orange", alpha=0.8)
            axes[idx].set_title("log|G⁻¹(z_base)|")
        if fallback is not None:
            for ax in axes:
                ax.axvline(fallback, color="red", linestyle="--", linewidth=1.0, label="fallback")
            axes[0].legend()

        def _fallback_ratio(vals):
            if fallback is None or vals is None:
                return None
            diff = np.abs(vals - fallback)
            return float(np.mean(diff < 1e-3))

        extra = {
            "mu_logdet_mean": float(np.mean(mu_vals)),
            "mu_fallback_ratio": _fallback_ratio(mu_vals),
        }
        if z_vals is not None:
            extra["z0_logdet_mean"] = float(np.mean(z_vals))
            extra["z0_fallback_ratio"] = _fallback_ratio(z_vals)
        if base_vals is not None:
            extra["base_logdet_mean"] = float(np.mean(base_vals))
            extra["base_fallback_ratio"] = _fallback_ratio(base_vals)
        cls._log_wandb(tag, fig, extra)

    @classmethod
    def log_metric_support(
        cls,
        *,
        centroid_norms,
        nearest_dist=None,
        span=None,
        projection_margin=None,
        temperature=None,
        regularization=None,
        tag: str = "metric_support",
    ) -> None:
        if not cls._should_log(tag):
            return
        norm_vals = cls._to_numpy(centroid_norms)
        if norm_vals is None or norm_vals.size == 0:
            return
        cols = 1 + int(nearest_dist is not None) + int(span is not None)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 3.5), dpi=120)
        axes = np.atleast_1d(axes)
        idx = 0
        bins = int(os.environ.get("RLVAE_WANDB_DEBUG_BINS", 30))

        axes[idx].hist(norm_vals, bins=bins, color="tab:blue", alpha=0.8)
        axes[idx].set_title("||centroid|| histogram")
        axes[idx].set_xlabel("norm")
        idx += 1

        nearest_vals = None
        if nearest_dist is not None:
            nearest_vals = cls._to_numpy(nearest_dist)
            ax = axes[idx]
            if nearest_vals is None or nearest_vals.size == 0:
                ax.axis("off")
            else:
                ax.hist(nearest_vals, bins=bins, color="tab:green", alpha=0.8)
                ax.set_title("Nearest-centroid distance")
                ax.set_xlabel("distance")
            idx += 1

        span_vals = None
        if span is not None:
            span_vals = cls._to_numpy(span)
            ax = axes[idx]
            if span_vals is None or span_vals.size == 0:
                ax.axis("off")
            else:
                sorted_span = np.sort(span_vals)
                k = min(len(sorted_span), 64)
                sample_idx = np.linspace(0, len(sorted_span) - 1, k, dtype=int)
                ax.plot(range(k), sorted_span[sample_idx], marker="o", linewidth=1.0)
                ax.set_title("Per-dim span (sorted)")
                ax.set_xlabel("dimension sample")
                ax.set_ylabel("extent")

        extra = {
            "centroid_norm_mean": float(np.mean(norm_vals)),
            "centroid_norm_min": float(np.min(norm_vals)),
            "centroid_norm_max": float(np.max(norm_vals)),
        }
        if nearest_vals is not None and nearest_vals.size:
            extra["nearest_dist_mean"] = float(np.mean(nearest_vals))
            extra["nearest_dist_min"] = float(np.min(nearest_vals))
            extra["nearest_dist_max"] = float(np.max(nearest_vals))
        if span_vals is not None and span_vals.size:
            extra["span_min"] = float(np.min(span_vals))
            extra["span_max"] = float(np.max(span_vals))
        if projection_margin is not None:
            extra["projection_margin"] = float(projection_margin)
        if temperature is not None:
            extra["temperature"] = float(temperature)
        if regularization is not None:
            extra["regularization"] = float(regularization)
        cls._log_wandb(tag, fig, extra)

    @classmethod
    def log_metric_projection(
        cls,
        *,
        shift_norms,
        margin=None,
        tag: str = "metric_projection",
    ) -> None:
        if not cls._should_log(tag):
            return
        shift_vals = cls._to_numpy(shift_norms)
        if shift_vals is None or shift_vals.size == 0:
            return
        shift_vals = shift_vals[shift_vals > 0]
        if shift_vals.size == 0:
            return
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(5, 3.2), dpi=120)
        bins = min(len(shift_vals), int(os.environ.get("RLVAE_WANDB_DEBUG_BINS", 30)))
        ax.hist(shift_vals, bins=max(bins, 10), color="tab:red", alpha=0.8)
        ax.set_title("Projection shift norms")
        ax.set_xlabel("||Δ||")

        extra = {
            "shift_mean": float(np.mean(shift_vals)),
            "shift_max": float(np.max(shift_vals)),
            "shift_count": float(len(shift_vals)),
        }
        if margin is not None:
            extra["projection_margin"] = float(margin)
        cls._log_wandb(tag, fig, extra)

    @classmethod
    def log_mahalanobis_fit(
        cls,
        mahal_sq,
        *,
        dim: int,
        tag: str = "mahalanobis_fit",
    ) -> None:
        if not cls._should_log(tag):
            return
        vals = cls._to_numpy(mahal_sq)
        if vals is None or vals.size == 0:
            return
        import matplotlib.pyplot as plt
        from torch.distributions import Chi2
        import torch

        chi = Chi2(dim)
        xs = torch.linspace(0, max(float(vals.max()), dim * 4), 200)
        pdf = chi.log_prob(xs).exp().numpy()
        xs = xs.numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120)
        axes[0].hist(vals, bins=30, density=True, alpha=0.7, label="empirical")
        axes[0].plot(xs, pdf, color="black", label=f"χ²({dim})")
        axes[0].set_title("Mahalanobis² density")
        axes[0].legend()

        sorted_vals = np.sort(vals)
        probs = (np.arange(len(sorted_vals)) + 0.5) / len(sorted_vals)
        theo = chi.icdf(torch.from_numpy(probs)).numpy()
        axes[1].plot(theo, sorted_vals, marker="o", linestyle="none", alpha=0.6)
        min_len = min(len(theo), len(sorted_vals))
        lo = min(sorted_vals[0], theo[0])
        hi = max(sorted_vals[min_len - 1], theo[min_len - 1])
        axes[1].plot([lo, hi], [lo, hi], color="gray", linestyle="--")
        axes[1].set_title("Q-Q vs χ²")
        axes[1].set_xlabel("χ² theoretical quantile")
        axes[1].set_ylabel("Empirical quantile")

        deviation = abs(float(vals.mean()) - dim) / max(np.sqrt(2 * dim), 1e-6)
        cls._log_wandb(
            tag,
            fig,
            extra={
                "mahal_mean": float(vals.mean()),
                "dim": float(dim),
                "sigma_dev": deviation,
            },
        )

    @classmethod
    def log_mu_metrics(
        cls,
        *,
        logdet_mu,
        centroid_dist=None,
        cos_align=None,
        tag: str = "mu_metrics",
    ) -> None:
        if not cls._should_log(tag):
            return
        logdet_vals = cls._to_numpy(logdet_mu)
        if logdet_vals is None:
            return
        import matplotlib.pyplot as plt

        cols = 2 if centroid_dist is not None else 1
        fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 3.5), dpi=120)
        axes = np.atleast_1d(axes)
        axes[0].hist(logdet_vals, bins=30, color="tab:blue", alpha=0.8)
        axes[0].set_title("log|G⁻¹(μ)| distribution")
        axes[0].set_xlabel("log|G⁻¹|")

        if centroid_dist is not None:
            cent_vals = cls._to_numpy(centroid_dist)
            if cent_vals is not None:
                axes[1].hist(cent_vals, bins=30, color="tab:green", alpha=0.8)
                axes[1].set_title("Nearest-centroid distance")
                axes[1].set_xlabel("distance")
        extra = {
            "mu_logdet_mean": float(np.mean(logdet_vals)),
        }
        if cos_align is not None:
            cos_vals = cls._to_numpy(cos_align)
            if cos_vals is not None:
                extra["mu_cos_align"] = float(np.mean(cos_vals))
        cls._log_wandb(tag, fig, extra)

    @classmethod
    def log_mu_gradient(
        cls,
        *,
        grad_norms,
        logdet_mu,
        centroid_dist=None,
        tag: str = "mu_gradient",
    ) -> None:
        if not cls._should_log(tag):
            return
        grad_vals = cls._to_numpy(grad_norms)
        logdet_vals = cls._to_numpy(logdet_mu)
        centroid_vals = cls._to_numpy(centroid_dist) if centroid_dist is not None else None
        if grad_vals is None or logdet_vals is None:
            return
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), dpi=120)
        axes[0].hist(grad_vals, bins=30, color="tab:blue", alpha=0.8)
        axes[0].set_title("||∇μ log|G⁻¹|| histogram")
        axes[0].set_xlabel("norm")

        axes[1].hist(logdet_vals, bins=30, color="tab:green", alpha=0.8)
        axes[1].set_title("log|G⁻¹(μ)|")
        axes[1].set_xlabel("log|G⁻¹|")

        axes[2].scatter(logdet_vals[: len(grad_vals)], grad_vals, s=12, alpha=0.7)
        axes[2].set_xlabel("log|G⁻¹(μ)|")
        axes[2].set_ylabel("||∇μ log|G⁻¹||")
        axes[2].set_title("Gradient vs volume")

        extra = {
            "grad_norm_mean": float(np.mean(grad_vals)),
            "logdet_mu_mean": float(np.mean(logdet_vals)),
        }
        if centroid_vals is not None:
            extra["centroid_dist_mean"] = float(np.mean(centroid_vals))
        cls._log_wandb(tag, fig, extra)

    @classmethod
    def log_volume_force(
        cls,
        *,
        log_stats,
        grad_log_stats,
        grad_stats,
        cos_stats,
        tag: str = "volume_force",
        target_cosine: float = 0.9,
    ) -> None:
        if not cls._should_log(tag):
            return
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), dpi=120)
        labels = ["mean", "std", "min", "max"]
        axes[0].bar(labels, log_stats, color="tab:blue")
        axes[0].set_title("½log|G⁻¹| stats")
        axes[1].bar(labels, grad_log_stats, color="tab:orange")
        axes[1].set_title("||∇(½log|G⁻¹|)|| stats")
        axes[2].bar(labels, grad_stats, color="tab:green")
        axes[2].set_title("||grad_U|| stats")
        axes[2].axhline(target_cosine, color="red", linestyle="--", label="target cosine")
        axes[2].legend()

        extra = {
            "cos_mean": float(cos_stats[0]),
        }
        cls._log_wandb(tag, fig, extra)

    @classmethod
    def log_volume_gradient_check(
        cls,
        *,
        grad_log_norm,
        grad_u_norm,
        cosines,
        logdet_manual=None,
        logdet_half=None,
        tag: str = "volume_gradient_check",
    ) -> None:
        if not cls._should_log(tag):
            return
        grad_log_vals = cls._to_numpy(grad_log_norm)
        grad_u_vals = cls._to_numpy(grad_u_norm)
        cos_vals = cls._to_numpy(cosines)
        logdet_manual_vals = cls._to_numpy(logdet_manual)
        logdet_half_vals = cls._to_numpy(logdet_half)
        if grad_log_vals is None or grad_u_vals is None or cos_vals is None:
            return
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=120)
        axes[0].scatter(grad_log_vals, grad_u_vals, s=12, alpha=0.7)
        axes[0].set_xlabel("||∇½log|G⁻¹||")
        axes[0].set_ylabel("||grad_U||")
        axes[0].set_title("Gradient norms")

        axes[1].hist(cos_vals, bins=30, color="tab:orange", alpha=0.8)
        axes[1].set_title("cos(∇log, grad_U)")
        axes[1].set_xlabel("cosine")

        diff_mean = None
        if logdet_manual_vals is not None and logdet_half_vals is not None:
            diff = logdet_manual_vals - logdet_half_vals
            diff_mean = float(np.mean(diff))
            axes[2].scatter(logdet_manual_vals, logdet_half_vals, s=12, alpha=0.7)
            axes[2].set_xlabel("torch slogdet(G⁻¹)")
            axes[2].set_ylabel("2·half_logdet")
            axes[2].set_title("log|G⁻¹| consistency")
            lims = [
                np.min([axes[2].get_xlim()[0], axes[2].get_ylim()[0]]),
                np.max([axes[2].get_xlim()[1], axes[2].get_ylim()[1]]),
            ]
            axes[2].plot(lims, lims, color="gray", linestyle="--")
        else:
            axes[2].axis("off")

        extra = {
            "grad_log_norm_mean": float(np.mean(grad_log_vals)),
            "grad_u_norm_mean": float(np.mean(grad_u_vals)),
            "cos_mean": float(np.mean(cos_vals)),
        }
        if diff_mean is not None:
            extra["logdet_diff_mean"] = diff_mean
        cls._log_wandb(tag, fig, extra)

    @classmethod
    def log_candidate_sampling(
        cls,
        *,
        h_scores,
        mahal_sq,
        euclid,
        selected_h,
        selected_mahal,
        selected_euclid,
        tag: str = "candidate_sampling",
    ) -> None:
        if not cls._should_log(tag):
            return
        h_vals = cls._to_numpy(h_scores)
        mahal_vals = cls._to_numpy(mahal_sq)
        euclid_vals = cls._to_numpy(euclid)
        sel_h = cls._to_numpy(selected_h)
        sel_mahal = cls._to_numpy(selected_mahal)
        sel_euclid = cls._to_numpy(selected_euclid)
        if h_vals is None or mahal_vals is None or euclid_vals is None:
            return
        import matplotlib.pyplot as plt

        pool_h = h_vals.reshape(-1)
        pool_mahal = mahal_vals.reshape(-1)
        pool_euclid = euclid_vals.reshape(-1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=120)
        bins = int(os.environ.get("RLVAE_WANDB_DEBUG_BINS", 30))
        axes[0].hist(pool_h, bins=bins, color="tab:blue", alpha=0.7, label="All candidates")
        if sel_h is not None:
            axes[0].hist(sel_h.reshape(-1), bins=bins, color="tab:orange", alpha=0.6, label="Selected")
        axes[0].set_title("h = ½log|G⁻¹| scores")
        axes[0].legend()

        axes[1].hist(pool_mahal, bins=bins, color="tab:green", alpha=0.7, label="All")
        if sel_mahal is not None:
            axes[1].hist(sel_mahal.reshape(-1), bins=bins, color="tab:red", alpha=0.6, label="Selected")
        axes[1].set_title("Mahalanobis² distribution")
        axes[1].legend()

        axes[2].scatter(pool_euclid, pool_mahal, s=12, alpha=0.6, label="All")
        if sel_euclid is not None and sel_mahal is not None:
            axes[2].scatter(sel_euclid.reshape(-1), sel_mahal.reshape(-1), s=30, color="tab:red", label="Selected")
        axes[2].set_xlabel("||z-μ||")
        axes[2].set_ylabel("Mahalanobis²")
        axes[2].set_title("Euclid vs Mahal²")
        axes[2].legend()

        extra = {
            "pool_mahal_mean": float(np.mean(pool_mahal)),
            "sel_mahal_mean": float(np.mean(sel_mahal)) if sel_mahal is not None else float("nan"),
            "pool_h_mean": float(np.mean(pool_h)),
        }
        cls._log_wandb(tag, fig, extra)

    @classmethod
    def log_sampling_distance(
        cls,
        *,
        expected,
        actual,
        selected=None,
        tag: str = "sampling_distance",
    ) -> None:
        if not cls._should_log(tag):
            return
        actual_vals = cls._to_numpy(actual)
        if actual_vals is None:
            return
        expected_val = float(expected)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
        axes[0].bar(["expected", "z0"], [expected_val, float(np.mean(actual_vals))], color=["tab:gray", "tab:blue"])
        axes[0].set_ylabel("||z-μ||")
        axes[0].set_title("Expected vs actual distance")

        axes[1].hist(actual_vals, bins=int(os.environ.get("RLVAE_WANDB_DEBUG_BINS", 30)), color="tab:blue", alpha=0.7)
        axes[1].axvline(expected_val, color="green", linestyle="--", label="Expected")
        if selected is not None:
            sel_vals = cls._to_numpy(selected)
            if sel_vals is not None:
                axes[1].axvline(float(np.mean(sel_vals)), color="orange", linestyle="--", label="Selected mean")
        axes[1].set_xlabel("||z-μ||")
        axes[1].set_title("Distribution of ||z0-μ||")
        axes[1].legend()

        extra = {
            "expected": expected_val,
            "actual_mean": float(np.mean(actual_vals)),
            "actual_std": float(np.std(actual_vals)),
        }
        cls._log_wandb(tag, fig, extra)

    @classmethod
    def log_step_scaling(
        cls,
        *,
        scale,
        min_eig,
        base_step,
        tag: str = "step_scaling",
        min_eig_threshold: float = 0.05,
    ) -> None:
        if not cls._should_log(tag):
            return
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(6, 4), dpi=120)
        ax1.bar(["scale"], [scale], color="tab:blue")
        ax1.axhline(1.0, color="gray", linestyle="--", label="max")
        ax1.set_ylim(0, max(1.2, scale + 0.1))
        ax1_2 = ax1.twinx()
        ax1_2.bar(["min eig"], [min_eig], color="tab:orange", alpha=0.6)
        ax1_2.axhline(min_eig_threshold, color="red", linestyle="--")
        ax1.set_title("Step scaling & min eig")
        extra = {"scale": float(scale), "min_eig": float(min_eig), "base_step": float(base_step)}
        cls._log_wandb(tag, fig, extra)

    @classmethod
    def log_warmup(
        cls,
        *,
        logdet_history: Sequence[float],
        target: Optional[float] = None,
        tag: str = "warmup_logdet",
    ) -> None:
        if not cls._should_log(tag):
            return
        history = cls._to_numpy(logdet_history)
        if history is None or history.size == 0:
            return
        import matplotlib.pyplot as plt

        steps = np.arange(len(history))
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
        ax.plot(steps, history, marker="o")
        if target is not None:
            ax.axhline(target, color="green", linestyle="--", label="μ log|G⁻¹| mean")
            ax.legend()
        ax.set_xlabel("Warm-up step")
        ax.set_ylabel("log|G⁻¹|")
        ax.set_title("Volume warm-up log|G⁻¹| trace")
        cls._log_wandb(
            tag,
            fig,
            extra={"delta": float(history[-1] - history[0])},
        )

    @classmethod
    def log_volume_acceptance(
        cls,
        *,
        logdet_base,
        logdet_z0,
        tag: str = "volume_acceptance",
    ) -> None:
        if not cls._should_log(tag):
            return
        base_vals = cls._to_numpy(logdet_base)
        z_vals = cls._to_numpy(logdet_z0)
        if base_vals is None or z_vals is None:
            return
        delta = float(np.mean(z_vals) - np.mean(base_vals))
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=120)
        ax.bar(["z_base", "z0"], [np.mean(base_vals), np.mean(z_vals)], color=["tab:gray", "tab:green"])
        ax.set_ylabel("mean log|G⁻¹|")
        ax.set_title("Volume acceptance effect")
        ax.text(0.5, max(np.mean(base_vals), np.mean(z_vals)), f"Δ={delta:+.3f}", ha="center")
        cls._log_wandb(tag, fig, {"delta": delta})

    @classmethod
    def log_latent_norms(
        cls,
        *,
        mu_norms,
        z0_norms,
        zS_norms,
        tag: str = "latent_norms",
    ) -> None:
        if not cls._should_log(tag):
            return
        mu_vals = cls._to_numpy(mu_norms)
        z0_vals = cls._to_numpy(z0_norms)
        zS_vals = cls._to_numpy(zS_norms)
        if mu_vals is None or z0_vals is None or zS_vals is None:
            return
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
        bins = 30
        ax.hist(mu_vals, bins=bins, alpha=0.5, label="||μ||")
        ax.hist(z0_vals, bins=bins, alpha=0.5, label="||z₀-μ||")
        ax.hist(zS_vals, bins=bins, alpha=0.5, label="||zS-μ||")
        ax.legend()
        ax.set_title("Latent norm distributions")
        cls._log_wandb(
            tag,
            fig,
            {
                "mu_norm_mean": float(np.mean(mu_vals)),
                "z0_norm_mean": float(np.mean(z0_vals)),
                "zS_norm_mean": float(np.mean(zS_vals)),
            },
        )


__all__ = ["DebugPlotLogger"]
