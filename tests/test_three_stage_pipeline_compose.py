import os
from pathlib import Path
import pytest
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir


def test_compose_three_stage_pipeline():
    repo_root = Path(__file__).resolve().parents[1]
    conf_dir = str(repo_root / 'conf')
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name='config', overrides=[
            'experiment=rlvae_three_stage_pipeline',
            'metric=rhvae',
            'sampling=rhmc_default',
            'checkpoint=default',
        ])
    # Basic assertions
    assert cfg.experiment.type == 'three_stage'
    assert cfg.metric.implementation in ['rhvae', 'precision']
    assert isinstance(cfg.sampling.n_steps, int)
    assert 'dir' in cfg.checkpoint and cfg.checkpoint.dir is not None

