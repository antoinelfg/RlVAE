import torch
from types import SimpleNamespace

from src.rlvae.models.components.loss_manager import LossManager
from src.rlvae.models.components.riemannian_rhmc_posterior import (
    RiemannianRHMCPosterior,
)


class IdentityMetricModel(torch.nn.Module):
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(rhmc_alpha=1.0)
        self.rhmc_alpha = 1.0
        self._current_epoch = 0
        self.posterior_config = {}
        self.register_parameter("_dummy", torch.nn.Parameter(torch.zeros(1)))

    def G(self, z: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(self.latent_dim, device=z.device, dtype=z.dtype)
        return eye.unsqueeze(0).expand(z.shape[0], -1, -1)

    def G_inv(self, z: torch.Tensor) -> torch.Tensor:
        return self.G(z)


def test_sample_returns_expected_shapes_and_log_prob():
    torch.manual_seed(42)
    model = IdentityMetricModel(latent_dim=3)
    posterior = RiemannianRHMCPosterior(
        model, config={"rhmc_steps": 0, "rhmc_alpha": 1.0, "rhmc_eps_reg": 1e-4}
    )
    mu = torch.zeros(4, 3)
    log_var = torch.zeros_like(mu)

    zK, log_q, z0, traj = posterior.sample_riemannian_rhmc_posterior(
        mu, log_var, return_traj=True
    )

    assert zK.shape == mu.shape
    assert log_q.shape == (4,)
    assert z0.shape == mu.shape
    assert isinstance(traj, dict)
    assert traj["rhmc_steps"] == 0

    expected_log_q = posterior._compute_log_riemannian_gaussian(z0, mu, log_var)
    assert torch.allclose(log_q, expected_log_q, atol=1e-6)


def test_sample_minimal_interface_when_flags_disabled():
    torch.manual_seed(0)
    model = IdentityMetricModel(latent_dim=2)
    posterior = RiemannianRHMCPosterior(model, config={"rhmc_steps": 1})
    mu = torch.zeros(2, 2)
    log_var = torch.zeros_like(mu)

    sample_only = posterior.sample_riemannian_rhmc_posterior(
        mu,
        log_var,
        return_log_prob=False,
        return_initial=False,
        return_traj=False,
    )

    assert isinstance(sample_only, torch.Tensor)
    assert sample_only.shape == mu.shape


def test_loss_manager_prefers_z0_when_available():
    torch.manual_seed(123)
    model = IdentityMetricModel(latent_dim=2)
    posterior = RiemannianRHMCPosterior(
        model, config={"rhmc_steps": 0, "rhmc_alpha": 0.5, "rhmc_eps_reg": 1e-4}
    )
    mu = torch.zeros(5, 2)
    log_var = torch.zeros_like(mu)
    zK, log_q, z0, traj = posterior.sample_riemannian_rhmc_posterior(
        mu, log_var, return_traj=True
    )

    loss_mgr = LossManager(
        beta=1.0,
        riemannian_beta=1.0,
        device=torch.device("cpu"),
        rhmc_kl_mode="mc",
        rhmc_kl_source="z0",
        rhmc_kl_jacobian=False,
    )

    losses = loss_mgr.compute_total_loss(
        x=torch.zeros(1, 1, 2, 2),
        x_recon=torch.zeros(1, 1, 2, 2),
        mu=mu,
        log_var=log_var,
        z_samples=zK,
        log_det_jacobians=None,
        z_seq=None,
        loop_mode="open",
        metric_tensor=None,
        use_riemannian_kl=True,
        rhmc_z0=z0,
        rhmc_zK=zK,
        rhmc_log_q=log_q,
        rhmc_traj_info=traj,
        rhmc_posterior=posterior,
    )

    expected_log_p = posterior._compute_log_prior(zK)
    expected_kl = (log_q - expected_log_p).mean()

    assert torch.allclose(losses["kl_divergence_loss"], expected_kl, atol=1e-6)
