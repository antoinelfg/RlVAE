from omegaconf import DictConfig

from .cyclic_dataset import CyclicSpritesDataModule
from .ellipse_datamodule import EllipseSequenceDataModule


_REGISTRY = {
    "cyclic_sprites": CyclicSpritesDataModule,
    "sprites": CyclicSpritesDataModule,
    "ellipse_sequences": EllipseSequenceDataModule,
    "ellipses": EllipseSequenceDataModule,
}


def build_data_module(config: DictConfig):
    """Instantiate the appropriate Lightning DataModule based on config."""
    name = getattr(config, "name", None)
    if name is None:
        # Heuristic fallback: if a data path exists, assume sprites; otherwise ellipses
        if hasattr(config, "train_path"):
            name = "cyclic_sprites"
        else:
            name = "ellipse_sequences"
    key = str(name).lower()
    if key not in _REGISTRY:
        # Allow custom ellipse variants without needing explicit registry entries
        if key.startswith("ellipse_sequences") or key.startswith("ellipses"):
            return EllipseSequenceDataModule(config)
        raise ValueError(
            f"Unsupported dataset '{name}'. Available datasets: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key](config)
