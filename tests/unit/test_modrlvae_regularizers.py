import torch
from omegaconf import OmegaConf

from rlvae.models.modrlvae import ModRLVAE


def build_cfg(overrides=None):
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
        'phase2_training': True,
        'spectral_penalty_enabled': True,
        'smoothness_penalty_enabled': True,
        'anisotropy_alignment_enabled': True,
        # Force spectral violations
        'eigenval_min_bound': 10.0,
        'eigenval_max_bound': 0.1,
    }
    if overrides:
        base.update(overrides)
    return OmegaConf.create(base)


def prepare_metric(model):
    D = model.latent_dim
    centroids = torch.tensor([[0.0, 0.0], [2.0, -2.0]], dtype=torch.float32, device=model.device)
    mats = torch.stack([
        torch.eye(D, device=model.device),
        3.0 * torch.eye(D, device=model.device),
    ], dim=0)
    model.mod_metric.load_pretrained(centroids, mats, temperature=0.5, regularization=0.01)
    model.centroids_tens = model.mod_metric.centroids
    model.M_tens = model.mod_metric.metric_matrices


@torch.no_grad()
def test_regularizers_present_and_finite():
    cfg = build_cfg()
    model = ModRLVAE(cfg)
    prepare_metric(model)
    x = torch.rand(8, 1, 1, 8, 8, device=model.device)
    out = model(x)
    # Check presence and finiteness
    keys = ['spectral_penalty', 'smoothness_penalty', 'anisotropy_penalty']
    for k in keys:
        assert k in out, f"Missing {k} in output"
        val = out[k]
        assert isinstance(val, torch.Tensor) and torch.isfinite(val), f"{k} is not finite"
    # Non-negativity
    assert (out['spectral_penalty'] >= 0) and (out['smoothness_penalty'] >= 0) and (out['anisotropy_penalty'] >= 0)

