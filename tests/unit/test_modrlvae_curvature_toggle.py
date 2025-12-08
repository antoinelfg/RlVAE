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
    }
    if overrides:
        base.update(overrides)
    return OmegaConf.create(base)


def prepare_metric(model):
    # Two centroids with different inverse metrics to ensure variation
    D = model.latent_dim
    centroids = torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float32, device=model.device)
    mats = torch.stack([
        torch.eye(D, device=model.device),
        2.0 * torch.eye(D, device=model.device),
    ], dim=0)
    model.mod_metric.load_pretrained(centroids, mats, temperature=0.5, regularization=0.01)
    # Refresh convenience handles
    model.centroids_tens = model.mod_metric.centroids
    model.M_tens = model.mod_metric.metric_matrices


@torch.no_grad()
def test_curvature_correction_toggle_changes_kl():
    # Model A: evaluate KL metric at z (curvature-corrected)
    cfg_a = build_cfg({'kl_metric_eval_point': 'z'})
    model_a = ModRLVAE(cfg_a)
    prepare_metric(model_a)
    # Model B: evaluate at mu
    cfg_b = build_cfg({'kl_metric_eval_point': 'mu'})
    model_b = ModRLVAE(cfg_b)
    prepare_metric(model_b)

    x = torch.rand(4, 1, 1, 8, 8, device=model_a.device)  # [B,T,C,H,W]
    out_a = model_a(x)
    out_b = model_b(x)

    kl_a = out_a['riemannian_kl'] if 'riemannian_kl' in out_a else out_a['kld_loss']
    kl_b = out_b['riemannian_kl'] if 'riemannian_kl' in out_b else out_b['kld_loss']

    assert torch.isfinite(kl_a) and torch.isfinite(kl_b)
    # Expect a difference for nontrivial metric
    assert not torch.allclose(kl_a, kl_b), "KL should differ between eval at z vs mu"

