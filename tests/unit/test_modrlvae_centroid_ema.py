import torch
from omegaconf import OmegaConf

from rlvae.models.modrlvae import ModRLVAE


def build_cfg():
    base = {
        'input_dim': [1, 8, 8],
        'latent_dim': 2,
        'sequence_length': 2,
        'posterior_type': 'riemannian_metric',
        'encoder': {'architecture': 'mlp'},
        'decoder': {'architecture': 'mlp'},
        'beta': 1.0,
        'riemannian_beta': 1.0,
        'loop': {'mode': 'open', 'penalty': 0.0},
        'metric': {
            'trainable': False,
            'temperature_override': 0.5,
            'regularization_override': 0.01,
        },
        # Enable Phase 2 + EMA
        'phase2_training': True,
        'centroid_ema_enabled': True,
        'centroid_ema_rate': 0.5,
        'centroid_ema_update_frequency': 1,
    }
    return OmegaConf.create(base)


def prepare_metric(model):
    D = model.latent_dim
    centroids = torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float32, device=model.device)
    mats = torch.stack([
        torch.eye(D, device=model.device),
        2.0 * torch.eye(D, device=model.device),
    ], dim=0)
    model.mod_metric.load_pretrained(centroids, mats, temperature=0.5, regularization=0.01)
    model.centroids_tens = model.mod_metric.centroids
    model.M_tens = model.mod_metric.metric_matrices


@torch.no_grad()
def test_centroid_ema_updates_centroids():
    cfg = build_cfg()
    model = ModRLVAE(cfg)
    prepare_metric(model)
    before = model.centroids_tens.clone()
    x = torch.rand(16, 1, 1, 8, 8, device=model.device)
    _ = model(x)
    after = model.centroids_tens
    assert not torch.allclose(before, after), "Centroids should be updated by EMA"

