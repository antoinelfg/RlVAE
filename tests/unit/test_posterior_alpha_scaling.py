"""
Quick check: posterior alpha controls mu-z distance and KL scale.
"""

import torch

def test_posterior_alpha_controls_distance():
    from rlvae.models.riemannian_flow_vae import RiemannianFlowVAE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = 8
    batch = 16

    model = RiemannianFlowVAE(
        input_dim=(3, 64, 64),
        latent_dim=d,
        n_flows=0,
        beta=1.0,
        riemannian_beta=1.0,
        identity_metric_mode=True,
        kl_use_metric_normalization=True,
        kl_metric_norm_mode='geomean',
    ).to(device).eval()
    model._initialize_identity_metric()

    mu = torch.zeros(batch, d, device=device)
    log_var = torch.zeros(batch, d, device=device)

    with torch.no_grad():
        model.posterior_local_alpha = 0.02
        z_small = model.sample_metric_aware_posterior(mu, log_var)
        dist_small = torch.norm(z_small - mu, dim=1).mean().item()

        model.posterior_local_alpha = 0.20
        z_large = model.sample_metric_aware_posterior(mu, log_var)
        dist_large = torch.norm(z_large - mu, dim=1).mean().item()

    assert dist_large > dist_small, "Posterior alpha should increase mu-z distance"

