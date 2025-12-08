"""
Quick invariant test: Identity metric KL ≈ 0.5 * alpha * d
"""

import torch

def test_kl_identity_metric_alpha_scaling():
    from rlvae.models.riemannian_flow_vae import RiemannianFlowVAE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = 8
    batch = 16

    # Base config with identity metric mode and geomean normalization
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
    # Ensure identity metric
    model._initialize_identity_metric()

    # Dummy encoder outputs
    mu = torch.zeros(batch, d, device=device)
    log_var = torch.zeros(batch, d, device=device)

    def estimate_kl(alpha: float) -> float:
        # set posterior covariance alpha
        model.posterior_local_alpha = alpha
        with torch.no_grad():
            z = model.sample_metric_aware_posterior(mu, log_var)
            kl = model.compute_riemannian_metric_kl_loss(mu, log_var, z)
        return float(kl.item())

    alpha1, alpha2 = 0.05, 0.10
    kl1 = estimate_kl(alpha1)
    kl2 = estimate_kl(alpha2)

    # Expected scaling: KL ~ 0.5 * alpha * d
    expected_ratio = alpha2 / alpha1
    actual_ratio = kl2 / (kl1 + 1e-12)

    assert abs(actual_ratio - expected_ratio) < 0.25, (
        f"KL scaling mismatch: got ratio {actual_ratio:.3f}, expected {expected_ratio:.3f}"
    )

