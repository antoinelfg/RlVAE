import math
import sys
from pathlib import Path

import torch

# Ensure repository root (containing `src`) is on the Python path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rlvae.models.components.loss_manager import LossManager


class DummyMetric:
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def compute_metric(self, z: torch.Tensor) -> torch.Tensor:
        b, d = z.shape
        eye = torch.eye(d, device=z.device, dtype=z.dtype).unsqueeze(0)
        return self.scale * eye.expand(b, -1, -1).clone()


class DummyPosterior:
    def __init__(self, metric: DummyMetric):
        self.metric = metric
        self.eps_reg = 1e-6
        self.rhmc_steps = 0
        self.rhmc_alpha = 1.0
        self._last_sigma_mu = None

        class _Model(torch.nn.Module):
            def __init__(self, metric: DummyMetric):
                super().__init__()
                self.metric = metric

            def G(self, z: torch.Tensor) -> torch.Tensor:
                return self.metric.compute_metric(z)

        self._ctx = {'model': _Model(metric)}


def _make_inputs(batch: int = 2, dim: int = 2):
    x = torch.zeros(batch, 1, dim)
    x_recon = torch.zeros_like(x)
    mu = torch.zeros(batch, dim, requires_grad=True)
    log_var = torch.zeros(batch, dim)
    return x, x_recon, mu, log_var


def test_uniform_kl_identity_metric_matches_log_q0_mean():
    metric = DummyMetric(scale=1.0)
    posterior = DummyPosterior(metric)
    loss_manager = LossManager(beta=1.0, kl_prior_mode='uniform')
    x, x_recon, mu, log_var = _make_inputs()

    z0 = torch.tensor([[0.1, -0.2], [-0.05, 0.3]])
    zS = z0.clone()
    Sigma_mu = torch.eye(z0.shape[1]).unsqueeze(0).repeat(z0.shape[0], 1, 1)
    posterior._last_sigma_mu = Sigma_mu
    rhmc_traj = {
        'trajectory': [
            {'z': z0.clone(), 'rho': torch.zeros_like(z0)},
            {'z': zS.clone(), 'rho': torch.zeros_like(zS)},
        ],
        'delta_kin': torch.zeros(2),
        'delta_vol': torch.zeros(2),
        'Sigma_mu': Sigma_mu,
    }

    log_q0 = loss_manager._log_gaussian_density(z0, mu, Sigma_mu)

    losses = loss_manager.compute_total_loss(
        x=x,
        x_recon=x_recon,
        mu=mu,
        log_var=log_var,
        z_samples=zS,
        log_det_jacobians=[],
        z_seq=[zS],
        flow_manager=None,
        loop_mode="open",
        metric_tensor=metric,
        use_riemannian_kl=True,
        z0=z0,
        zS=zS,
        zF=zS,
        Sigma_mu=Sigma_mu,
        rhmc_z0=z0,
        rhmc_zK=zS,
        rhmc_log_q=log_q0,
        rhmc_traj_info=rhmc_traj,
        rhmc_posterior=posterior,
        rhmc_kl_mode="mc",
        rhmc_kl_source="z0",
        rhmc_kl_jacobian=False,
    )

    kl_loss = losses['kl_divergence_loss']
    expected = log_q0.mean()
    assert torch.allclose(kl_loss, expected, atol=1e-6)
    assert torch.allclose(losses['flow_loss'], torch.tensor(0.0))
    details = losses['loss_details']
    assert torch.allclose(details['loss/log_q_source_mean'], log_q0.mean(), atol=1e-6)
    zero_ref = details['loss/log_q_source_mean'].new_tensor(0.0)
    assert torch.isclose(details['loss/half_logdet_ginv_source_mean'], zero_ref, atol=1e-6)
    assert 'preview/log_q0' in details
    assert 'preview/half_logdet_ginv_source' in details


def test_uniform_kl_scaled_metric_includes_half_logdet():
    scale = 4.0
    metric = DummyMetric(scale=scale)
    posterior = DummyPosterior(metric)
    loss_manager = LossManager(beta=1.0, kl_prior_mode='uniform')
    x, x_recon, mu, log_var = _make_inputs()

    z0 = torch.tensor([[0.0, 0.0], [0.1, -0.2]])
    zS = z0.clone()
    dim = z0.shape[1]
    Sigma_mu = (1.0 / scale) * torch.eye(dim).unsqueeze(0).repeat(z0.shape[0], 1, 1)
    posterior._last_sigma_mu = Sigma_mu
    half_logdet = -0.5 * z0.shape[1] * math.log(scale)
    log_q0 = loss_manager._log_gaussian_density(z0, mu, Sigma_mu)
    traj = {
        'trajectory': [
            {'z': z0.clone(), 'rho': torch.zeros_like(z0)},
            {'z': zS.clone(), 'rho': torch.zeros_like(zS)},
        ],
        'delta_kin': torch.zeros(2),
        'delta_vol': torch.zeros(2),
        'Sigma_mu': Sigma_mu,
    }

    losses = loss_manager.compute_total_loss(
        x=x,
        x_recon=x_recon,
        mu=mu,
        log_var=log_var,
        z_samples=zS,
        log_det_jacobians=[],
        z_seq=[zS],
        flow_manager=None,
        loop_mode="open",
        metric_tensor=metric,
        use_riemannian_kl=True,
        z0=z0,
        zS=zS,
        zF=zS,
        Sigma_mu=Sigma_mu,
        rhmc_z0=z0,
        rhmc_zK=zS,
        rhmc_log_q=log_q0,
        rhmc_traj_info=traj,
        rhmc_posterior=posterior,
        rhmc_kl_mode="mc",
        rhmc_kl_source="z0",
        rhmc_kl_jacobian=False,
    )

    expected = (log_q0 - half_logdet).mean()
    assert torch.allclose(losses['kl_divergence_loss'], expected, atol=1e-6)
    details = losses['loss_details']
    assert torch.allclose(details['loss/log_q_source_mean'], log_q0.mean(), atol=1e-6)
    expected_half_logdet = details['loss/log_q_source_mean'].new_tensor(half_logdet)
    assert torch.allclose(details['loss/half_logdet_ginv_source_mean'], expected_half_logdet, atol=1e-6)
    assert 'preview/log_q0' in details
    assert 'preview/half_logdet_ginv_source' in details


def test_uniform_kl_accounts_for_flow_logdet_and_corrections():
    metric = DummyMetric(scale=1.0)
    posterior = DummyPosterior(metric)
    loss_manager = LossManager(beta=1.0, kl_prior_mode='uniform')
    x, x_recon, mu, log_var = _make_inputs()

    z0 = torch.tensor([[0.2, -0.1], [0.4, 0.3]])
    zS = z0 + 0.1
    zF = zS + 0.05
    Sigma_mu = torch.eye(z0.shape[1]).unsqueeze(0).repeat(z0.shape[0], 1, 1)
    posterior._last_sigma_mu = Sigma_mu
    flow_terms = torch.tensor([0.3, -0.1])
    delta_kin = torch.tensor([0.05, -0.02])
    delta_vol = torch.tensor([0.01, 0.0])
    log_q0 = loss_manager._log_gaussian_density(z0, mu, Sigma_mu)

    traj = {
        'trajectory': [
            {'z': z0.clone(), 'rho': torch.zeros_like(z0)},
            {'z': zS.clone(), 'rho': torch.zeros_like(zS)},
        ],
        'delta_kin': delta_kin,
        'delta_vol': delta_vol,
        'Sigma_mu': Sigma_mu,
    }

    losses = loss_manager.compute_total_loss(
        x=x,
        x_recon=x_recon,
        mu=mu,
        log_var=log_var,
        z_samples=zS,
        log_det_jacobians=[flow_terms],
        z_seq=[zS, zF],
        flow_manager=None,
        loop_mode="open",
        metric_tensor=metric,
        use_riemannian_kl=True,
        z0=z0,
        zS=zS,
        zF=zF,
        Sigma_mu=Sigma_mu,
        rhmc_z0=z0,
        rhmc_zK=zS,
        rhmc_log_q=log_q0,
        rhmc_traj_info=traj,
        rhmc_posterior=posterior,
        rhmc_kl_mode="mc",
        rhmc_kl_source="z0",
        rhmc_kl_jacobian=False,
    )

    half_logdet = torch.zeros_like(flow_terms)
    expected_terms = log_q0 - half_logdet - flow_terms + (delta_kin - delta_vol)
    assert torch.allclose(
        losses['kl_divergence_loss'],
        expected_terms.mean(),
        atol=1e-6,
    )
    loss_details = losses['loss_details']
    assert torch.allclose(loss_details['loss/log_q_source_mean'], log_q0.mean(), atol=1e-6)
    assert torch.allclose(loss_details['loss/sum_logdet_flow_mean'], flow_terms.mean(), atol=1e-6)
    assert torch.allclose(loss_details['rhmc/delta_kin_mean'], delta_kin.mean(), atol=1e-6)
    assert 'preview/half_logdet_ginv_source' in loss_details


def test_gradients_flow_through_uniform_kl_terms():
    metric = DummyMetric(scale=2.0)
    posterior = DummyPosterior(metric)
    loss_manager = LossManager(beta=1.0, kl_prior_mode='uniform')
    x, x_recon, mu, log_var = _make_inputs()

    z0 = torch.tensor([[0.3, -0.2], [-0.4, 0.1]], requires_grad=True)
    zS = z0 + 0.05
    Sigma_mu = 0.5 * torch.eye(z0.shape[1]).unsqueeze(0).repeat(z0.shape[0], 1, 1)
    posterior._last_sigma_mu = Sigma_mu
    traj = {
        'trajectory': [
            {'z': z0.clone(), 'rho': torch.zeros_like(z0)},
            {'z': zS.clone(), 'rho': torch.zeros_like(zS)},
        ],
        'delta_kin': torch.zeros(2),
        'delta_vol': torch.zeros(2),
        'Sigma_mu': Sigma_mu,
    }

    log_q0 = loss_manager._log_gaussian_density(z0, mu, Sigma_mu)

    losses = loss_manager.compute_total_loss(
        x=x,
        x_recon=x_recon,
        mu=mu,
        log_var=log_var,
        z_samples=zS,
        log_det_jacobians=[],
        z_seq=[zS],
        flow_manager=None,
        loop_mode="open",
        metric_tensor=metric,
        use_riemannian_kl=True,
        z0=z0,
        zS=zS,
        zF=zS,
        Sigma_mu=Sigma_mu,
        rhmc_z0=z0,
        rhmc_zK=zS,
        rhmc_log_q=log_q0,
        rhmc_traj_info=traj,
        rhmc_posterior=posterior,
        rhmc_kl_mode="mc",
        rhmc_kl_source="z0",
        rhmc_kl_jacobian=False,
    )

    kl_loss = losses['kl_divergence_loss']
    kl_loss.backward()
    assert mu.grad is not None


def test_log_kinetic_density_matches_manual_formula():
    scale = 3.0
    metric = DummyMetric(scale=scale)
    posterior = DummyPosterior(metric)
    loss_manager = LossManager()

    rho = torch.tensor([[0.3, -0.7], [1.2, 0.5]])
    z = torch.zeros_like(rho)
    actual = loss_manager._log_kinetic_density(
        rho,
        z,
        metric_tensor=metric,
        rhmc_posterior=posterior,
    )
    expected = -0.5 * (
        (rho.pow(2).sum(dim=-1) / scale)
        + rho.shape[1] * math.log(scale)
        + rho.shape[1] * math.log(2 * math.pi)
    )
    assert torch.allclose(actual, expected.to(actual.dtype), atol=1e-6)


def test_uniform_prior_half_logdet_constant_under_constant_metric():
    scale = 5.0
    metric = DummyMetric(scale=scale)
    posterior = DummyPosterior(metric)
    loss_manager = LossManager(beta=1.0, kl_prior_mode='uniform')

    batch, dim = 2, 2
    x, x_recon, mu, log_var = _make_inputs(batch=batch, dim=dim)
    z0 = torch.randn(batch, dim)
    Sigma_mu = (1.0 / scale) * torch.eye(dim).unsqueeze(0).repeat(batch, 1, 1)
    posterior._last_sigma_mu = Sigma_mu
    traj = {
        'trajectory': [
            {'z': z0.clone(), 'rho': torch.zeros_like(z0)},
            {'z': z0.clone(), 'rho': torch.zeros_like(z0)},
        ],
        'delta_kin': torch.zeros(batch),
        'delta_vol': torch.zeros(batch),
        'Sigma_mu': Sigma_mu,
    }

    zS = z0.clone()
    outputs_same = loss_manager.compute_total_loss(
        x=x,
        x_recon=x_recon,
        mu=mu,
        log_var=log_var,
        z_samples=zS,
        log_det_jacobians=[],
        z_seq=[zS],
        flow_manager=None,
        loop_mode="open",
        metric_tensor=metric,
        use_riemannian_kl=True,
        z0=z0,
        zS=zS,
        zF=zS,
        Sigma_mu=Sigma_mu,
        rhmc_z0=z0,
        rhmc_zK=zS,
        rhmc_traj_info=traj,
        rhmc_posterior=posterior,
        rhmc_kl_mode="mc",
        rhmc_kl_source="z0",
        rhmc_kl_jacobian=False,
    )

    zF_shifted = zS + 0.123
    outputs_shifted = loss_manager.compute_total_loss(
        x=x,
        x_recon=x_recon,
        mu=mu,
        log_var=log_var,
        z_samples=zS,
        log_det_jacobians=[],
        z_seq=[zS, zF_shifted],
        flow_manager=None,
        loop_mode="open",
        metric_tensor=metric,
        use_riemannian_kl=True,
        z0=z0,
        zS=zS,
        zF=zF_shifted,
        Sigma_mu=Sigma_mu,
        rhmc_z0=z0,
        rhmc_zK=zS,
        rhmc_traj_info=traj,
        rhmc_posterior=posterior,
        rhmc_kl_mode="mc",
        rhmc_kl_source="z0",
        rhmc_kl_jacobian=False,
    )

    half_logdet_expected = -0.5 * dim * math.log(scale)
    half_logdet_key = 'loss/half_logdet_ginv_source_mean'
    expected_tensor = outputs_same['loss_details'][half_logdet_key].new_tensor(half_logdet_expected)
    assert torch.isclose(
        outputs_same['loss_details'][half_logdet_key],
        expected_tensor,
        atol=1e-6,
    )
    assert 'preview/half_logdet_ginv_source' in outputs_same['loss_details']
    assert torch.allclose(
        outputs_same['loss_details'][half_logdet_key],
        outputs_shifted['loss_details'][half_logdet_key],
        atol=1e-6,
    )
    assert torch.allclose(
        outputs_same['loss_details']['loss/log_q_source_mean'],
        outputs_shifted['loss_details']['loss/log_q_source_mean'],
        atol=1e-6,
    )


def test_half_logdet_volume_consistency_across_representations():
    G = torch.tensor([[[2.0, 0.0], [0.0, 5.0]]])
    G_inv = torch.linalg.inv(G)

    loss_g = LossManager(metric_representation="G")
    loss_ginv = LossManager(metric_representation="Ginv")

    hl_g = loss_g._half_logdet_volume(G, rep='g')
    hl_ginv = loss_ginv._half_logdet_volume(G_inv)

    assert torch.allclose(hl_g, hl_ginv, atol=1e-8)


def test_quad_with_g_consistency_across_representations():
    G = torch.tensor([[[3.0, 0.0], [0.0, 2.0]]])
    G_inv = torch.linalg.inv(G)
    v = torch.tensor([[1.5, -0.5]])

    loss_g = LossManager(metric_representation="G")
    loss_ginv = LossManager(metric_representation="Ginv")

    quad_g = loss_g._quad_with_G(v, G)
    quad_ginv = loss_ginv._quad_with_G(v, G_inv)

    assert torch.allclose(quad_g, quad_ginv, atol=1e-8)
