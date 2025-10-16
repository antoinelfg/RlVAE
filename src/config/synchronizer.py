"""
Legacy Configuration Synchronizer
=================================

The unified `settings.*` tree makes the historical stage-by-stage synchronizer
obsolete. The helper now simply returns a detached copy of the configuration and
emits a deprecation warning the first time it is used.
"""

from __future__ import annotations

from typing import Union
import warnings

from omegaconf import DictConfig, OmegaConf


class ConfigSynchronizer:
    """Deprecated no-op synchronizer kept for backward compatibility."""

    _warned: bool = False

    def sync_pipeline_config(self, config: Union[DictConfig, dict]) -> DictConfig:
        """
        Return a detached copy of the provided configuration.
        """
        if not ConfigSynchronizer._warned:
            warnings.warn(
                "ConfigSynchronizer is deprecated: the monolithic `settings.*` tree "
                "is authoritative and no pipeline synchronization is performed.",
                DeprecationWarning,
                stacklevel=2,
            )
            ConfigSynchronizer._warned = True

        if isinstance(config, DictConfig):
            return OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        return OmegaConf.create(config)


def sync_pipeline_config(config: Union[DictConfig, dict]) -> DictConfig:
    """Convenience shim returning a detached copy of `config`."""
    synchronizer = ConfigSynchronizer()
    return synchronizer.sync_pipeline_config(config)
