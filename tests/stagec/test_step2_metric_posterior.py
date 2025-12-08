import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rlvae.models.riemannian_flow_vae import RiemannianFlowVAE  # noqa: E402


def _load_dummy_metric(model: RiemannianFlowVAE, latent_dim: int, k: int = 8):
    # Create a simple diagonal metric around origin
    centroids = torch.randn(k, latent_dim) * 0.5
    M = torch.stack([torch.eye(latent_dim) for _ in range(k)], dim=0)
    model.load_pretrained_metrics_from_tensor(
        centroids=centroids,
        metric_matrices=M,
        temperature=0.1,
        regularization=1e-2,
    )


def test_metric_anchored_posterior_with_samplewise_kl():
    torch.manual_seed(0)

    input_dim = (1, 16, 16)
    latent_dim = 6
    model = RiemannianFlowVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        n_flows=2,
        beta=1.0,
        posterior_type="riemannian_metric",
        riemannian_kl_mode="sample_logq_logp",
        loop_mode="open",
    )

    _load_dummy_metric(model, latent_dim)

    batch_size = 4
    seq_len = 4
    x = torch.rand(batch_size, seq_len, *input_dim)

    out = model(x)

    # The KL should be positive and finite
    assert torch.isfinite(out.kl_loss).all()
    assert out.kl_loss.item() >= 0.0

    # Total loss should include recon + kl (flows add positive term)
    assert torch.isfinite(out.total_loss).all()


