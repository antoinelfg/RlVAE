import torch
from omegaconf import OmegaConf

from src.data.datamodule_factory import build_data_module
from src.data.ellipse_datamodule import EllipseSequenceDataModule


def test_build_data_module_returns_ellipse_sequences_module():
    cfg = OmegaConf.create(
        {
            "name": "ellipse_sequences",
            "num_sequences": 12,
            "seq_len": 4,
            "sequence_length": 4,
            "image_size": [16, 16],
            "batch_size": 3,
            "shuffle": False,
            "num_workers": 0,
            "channels": 1,
        }
    )

    dm = build_data_module(cfg)
    assert isinstance(dm, EllipseSequenceDataModule)

    dm.setup("fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert batch.shape == (3, 4, 1, 16, 16)
    assert torch.isfinite(batch).all()
