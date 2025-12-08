"""
Invariance test: With geomean normalization, scaling metric by c should not change KL.
"""

import torch

def test_kl_geomean_invariance_under_scaling():
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
    # Identity metric, geomean normalization active
    model._initialize_identity_metric()

    mu = torch.zeros(batch, d, device=device)
    log_var = torch.zeros(batch, d, device=device)

    # Baseline KL at alpha
    model.posterior_local_alpha = 0.05
    with torch.no_grad():
        z = model.sample_metric_aware_posterior(mu, log_var)
        kl_base = model.compute_riemannian_metric_kl_loss(mu, log_var, z).item()

    # Wrap G_inv to emulate scaling by c: G_inv' = (1/c) * G_inv => G' = c * G
    c = 10.0
    G_inv_orig = model.G_inv
    def G_inv_scaled(z):
        return (1.0 / c) * G_inv_orig(z)
    model.G_inv = G_inv_scaled

    with torch.no_grad():
        z2 = model.sample_metric_aware_posterior(mu, log_var)  # Sampling uses G(mu); identity -> scales covariance but normalization applies in KL
        kl_scaled = model.compute_riemannian_metric_kl_loss(mu, log_var, z2).item()

    # With geomean normalization, KL values should be close
    assert abs(kl_scaled - kl_base) / (abs(kl_base) + 1e-12) < 0.25, (
        f"Geomean normalization failed to stabilize KL under scaling: base={kl_base:.4f}, scaled={kl_scaled:.4f}"
    )

