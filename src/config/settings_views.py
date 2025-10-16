"""
Settings Views
==============

Helper utilities to materialize read-only views of the unified `settings.*`
tree into the legacy structures expected by older subsystems (model factory,
Lightning trainer, etc.) without mutating the underlying configuration.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


def _select(cfg: DictConfig, path: str, default: Any = None) -> Any:
    """Safe OmegaConf.select with default fallback."""
    try:
        value = OmegaConf.select(cfg, path)
    except Exception:
        value = None
    if value is None:
        return default
    return value


def _to_dict(cfg: Optional[DictConfig]) -> Dict[str, Any]:
    """Convert a DictConfig to a deep-copied python dict (empty if None)."""
    if cfg is None:
        return {}
    return deepcopy(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))


def _set_if_missing(target: Dict[str, Any], key: str, value: Any) -> None:
    """Populate ``target[key]`` if not already set and value is not None."""
    if key not in target and value is not None:
        target[key] = value


def build_model_config_from_settings(settings: DictConfig) -> DictConfig:
    """
    Materialize a legacy-style model configuration from the unified settings tree.

    The resulting DictConfig mirrors the historical ``config.model`` layout while
    keeping the richer nested structure from ``settings.model``. No mutations are
    applied to the input configuration.
    """
    if not isinstance(settings, DictConfig):
        raise TypeError("settings must be a DictConfig")

    model_cfg = _to_dict(_select(settings, "model"))
    data_common = _to_dict(_select(settings, "data.common"))

    flows_cfg = model_cfg.get("flows", {})
    losses_cfg = model_cfg.get("losses", {})
    loop_cfg = model_cfg.get("loop", {})
    posterior_cfg = model_cfg.get("posterior", {})
    sampling_cfg = model_cfg.get("sampling", {})

    # Flatten frequently accessed fields expected by legacy code paths.
    _set_if_missing(model_cfg, "_target_", _select(settings, "model.target"))
    _set_if_missing(model_cfg, "flow_hidden_size", flows_cfg.get("hidden_size"))
    _set_if_missing(model_cfg, "flow_n_blocks", flows_cfg.get("n_blocks"))
    _set_if_missing(model_cfg, "flow_n_hidden", flows_cfg.get("n_hidden"))
    _set_if_missing(model_cfg, "beta", losses_cfg.get("beta"))
    _set_if_missing(
        model_cfg,
        "riemannian_beta",
        losses_cfg.get("riemannian_beta", losses_cfg.get("beta")),
    )
    _set_if_missing(model_cfg, "mu_l2_weight", losses_cfg.get("mu_l2_weight"))
    _set_if_missing(model_cfg, "loop_penalty_weight", losses_cfg.get("loop_penalty_weight"))
    _set_if_missing(model_cfg, "kl_prior_mode", losses_cfg.get("kl_prior_mode"))
    _set_if_missing(model_cfg, "kl_metric_eval_point", losses_cfg.get("kl_metric_eval_point"))
    _set_if_missing(
        model_cfg,
        "kl_use_metric_normalization",
        losses_cfg.get("kl_use_metric_normalization"),
    )
    _set_if_missing(model_cfg, "kl_metric_norm_mode", losses_cfg.get("kl_metric_norm_mode"))
    _set_if_missing(model_cfg, "flow_weight", losses_cfg.get("flow_weight"))
    _set_if_missing(model_cfg, "posterior_type", posterior_cfg.get("type"))

    for key in (
        "rhmc_steps",
        "rhmc_step_size",
        "rhmc_alpha",
        "rhmc_eps_reg",
        "rhmc_kl_mode",
        "rhmc_kl_source",
        "rhmc_kl_jacobian",
        "max_momentum_norm",
        "max_velocity_norm",
        "max_position_step",
        "max_position_norm",
        "posterior_local_alpha",
        "posterior_alpha_ramp_enabled",
        "maha_clip",
        "cov_norm_mode",
    ):
        _set_if_missing(model_cfg, key, posterior_cfg.get(key))

    # Loop overrides: preserve nested loop config but expose penalty on the loop view.
    loop_penalty = losses_cfg.get("loop_penalty_weight")
    if isinstance(loop_cfg, dict):
        loop_cfg = deepcopy(loop_cfg)
    else:
        loop_cfg = {}
    if loop_penalty is not None:
        loop_cfg.setdefault("penalty", loop_penalty)
    model_cfg["loop"] = loop_cfg

    # Sampling defaults
    if isinstance(sampling_cfg, dict):
        sampling_cfg = deepcopy(sampling_cfg)
    model_cfg["sampling"] = sampling_cfg

    # Derive input dimensionality and sequence metadata from the data settings
    channels = data_common.get("channels", 1)
    image_size = data_common.get("image_size", [1, 1])
    seq_len = data_common.get("sequence_length")
    if isinstance(image_size, tuple):
        image_size = list(image_size)
    if not isinstance(image_size, list):
        image_size = [image_size, image_size]

    input_dim = [channels]
    input_dim.extend(int(dim) for dim in image_size[:2])
    model_cfg["input_dim"] = tuple(input_dim)
    if seq_len is not None:
        model_cfg["sequence_length"] = int(seq_len)
        model_cfg.setdefault("n_flows", max(0, int(seq_len) - 1))

    # Convenience mirrors for training subsystems
    _set_if_missing(model_cfg, "update_metric_during_training", model_cfg.get("update_metric_during_training"))
    _set_if_missing(model_cfg, "metric_update_frequency", model_cfg.get("metric_update_frequency"))
    _set_if_missing(model_cfg, "metric_update_alpha", model_cfg.get("metric_update_alpha"))
    _set_if_missing(model_cfg, "metric_update_temperature", model_cfg.get("metric_update_temperature"))
    _set_if_missing(model_cfg, "metric_update_regularization", model_cfg.get("metric_update_regularization"))

    # Re-wrap into DictConfig for downstream consumers.
    materialized_cfg = OmegaConf.create(model_cfg)
    OmegaConf.set_struct(materialized_cfg, False)
    return materialized_cfg


__all__ = ["build_model_config_from_settings"]
