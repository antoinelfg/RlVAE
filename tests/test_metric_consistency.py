"""
Tests for metric representation consistency and log-determinant helpers.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rlvae.models.components.loss_manager import LossManager
from src.rlvae.models.components.metric_tensor import MetricTensor
from src.rlvae.models.components.metric_utils import half_logdet_volume
from src.rlvae.models.components.flow_manager import FlowManager


class _ConstantMetric:
    def __init__(self, G: torch.Tensor):
        self.G = G
        self.G_inv = torch.linalg.inv(G)

    def compute_metric(self, z: torch.Tensor) -> torch.Tensor:
        return self.G.expand(z.shape[0], -1, -1)

    def compute_inverse_metric(self, z: torch.Tensor) -> torch.Tensor:
        return self.G_inv.expand(z.shape[0], -1, -1)


def _make_spd_matrix(dim: int) -> torch.Tensor:
    A = torch.randn(dim, dim)
    return (A @ A.T) + dim * torch.eye(dim)


def test_loss_manager_evaluate_metric_roundtrip():
    torch.manual_seed(0)
    G = _make_spd_matrix(3)
    stub = _ConstantMetric(G)
    z = torch.randn(5, 3)

    lm_g = LossManager(metric_representation="g")
    evaluated_g = lm_g._evaluate_metric(z, stub, None)
    assert torch.allclose(evaluated_g, stub.compute_metric(z))

    lm_ginv = LossManager(metric_representation="ginv")
    evaluated_ginv = lm_ginv._evaluate_metric(z, stub, None)
    assert torch.allclose(evaluated_ginv, stub.compute_inverse_metric(z))


def test_riemannian_kl_differs_between_representations():
    torch.manual_seed(1)
    G = _make_spd_matrix(2)
    metric = _ConstantMetric(G)
    mu = torch.randn(4, 2)
    log_var = torch.zeros_like(mu)
    z_samples = mu + 0.1 * torch.randn_like(mu)

    lm_g = LossManager(metric_representation="g")
    loss_g = lm_g.compute_riemannian_kl_loss(mu, log_var, z_samples, metric)

    lm_ginv = LossManager(metric_representation="ginv")
    loss_ginv = lm_ginv.compute_riemannian_kl_loss(mu, log_var, z_samples, metric)

    assert torch.abs(loss_g - loss_ginv) > 1e-5


def test_half_logdet_volume_mirror_symmetry():
    torch.manual_seed(2)
    G = _make_spd_matrix(4)
    G_inv = torch.linalg.inv(G)

    half_logdet_g = half_logdet_volume(G.unsqueeze(0), "g")
    half_logdet_ginv = half_logdet_volume(G_inv.unsqueeze(0), "ginv")

    full_logdet_g = -2.0 * half_logdet_g
    full_logdet_ginv = 2.0 * half_logdet_ginv

    assert torch.allclose(full_logdet_ginv, -full_logdet_g, atol=1e-6)


def test_pushforward_identity_with_stageb_metric():
    metric_path = Path("outputs/stages/B_RHVAE_MLP_2_SPRITES/metric.pt")
    if not metric_path.exists():
        pytest.skip(f"Stage-B metric file not found at {metric_path}")

    state = torch.load(metric_path, map_location="cpu", weights_only=False)
    centroids = state["centroids"]
    matrices = state.get("metric_matrices", state.get("M_matrices"))
    temperature = state.get("temperature", 0.1)
    regularization = state.get("regularization", 0.01)

    metric = MetricTensor(latent_dim=centroids.shape[1], device=torch.device("cpu"))
    metric.load_pretrained(centroids, matrices, temperature, regularization)
    metric.eval()

    loss_manager = LossManager(metric_representation="ginv")

    batch = 8
    latent_dim = centroids.shape[1]
    z0 = torch.randn(batch, latent_dim)

    flow_manager = FlowManager(
        latent_dim=latent_dim,
        n_flows=0,
        device=torch.device("cpu"),
    )

    (G_push, rep_push), _, _, half_logdet_push_ginv = loss_manager._pushforward_metric_via_flows(
        z0, flow_manager, metric, None
    )
    G_source, rep_source = loss_manager._evaluate_metric(z0, metric, None, with_rep=True)
    half_logdet_source = loss_manager._half_logdet_volume(G_source, rep_source)

    assert rep_source == "ginv"
    assert rep_push == "ginv"
    assert torch.allclose(G_push, G_source, atol=1e-6)
    assert torch.allclose(half_logdet_push_ginv, half_logdet_source, atol=1e-6)
