import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rlvae.models.riemannian_flow_vae import RiemannianFlowVAE  # noqa: E402
from models.samplers.hmc_sampler import RiemannianHMCSampler  # noqa: E402


def _load_dummy_metric(model: RiemannianFlowVAE, latent_dim: int, k: int = 8):
    centroids = torch.randn(k, latent_dim) * 0.25
    M = torch.stack([torch.eye(latent_dim) for _ in range(k)], dim=0)
    model.load_pretrained_metrics_from_tensor(
        centroids=centroids,
        metric_matrices=M,
        temperature=0.2,
        regularization=1e-2,
    )


def test_mu_anchored_posterior_hmc_sampling_path():
    torch.manual_seed(1)

    input_dim = (1, 16, 16)
    latent_dim = 5
    model = RiemannianFlowVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        n_flows=0,
        beta=1.0,
        posterior_type="gaussian",  # We'll bypass forward and call sampler directly
        riemannian_kl_mode="quadratic",
    )

    _load_dummy_metric(model, latent_dim)

    # Create a small batch and encode once to get μ, log_var through the forward encoder path
    batch_size = 3
    seq_len = 2
    x = torch.rand(batch_size, seq_len, *input_dim)
    enc = model.encoder(x[:, 0])
    mu = enc.embedding
    log_var = enc.log_covariance

    sampler = RiemannianHMCSampler(model, mcmc_steps_nbr=10, n_lf=5, eps_lf=0.01)
    z_post = sampler.sample_posterior(mu, log_var)

    # Must be finite and not too far from μ on average
    assert torch.isfinite(z_post).all()
    mean_dist = torch.norm(z_post - mu, dim=1).mean().item()
    assert mean_dist < 2.0


