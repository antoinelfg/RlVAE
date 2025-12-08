import sys
from pathlib import Path

import torch


# Ensure src/ is on path when running tests directly
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rlvae.models.riemannian_flow_vae import RiemannianFlowVAE  # noqa: E402


def test_stagec_recon_only_forward_cpu():
    torch.manual_seed(42)

    # Minimal model: Gaussian posterior, no flows
    input_dim = (1, 16, 16)
    latent_dim = 4
    model = RiemannianFlowVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        n_flows=0,
        beta=1.0,
        posterior_type="gaussian",
        riemannian_kl_mode="quadratic",
        loop_mode="open",
    )

    # Synthetic batch: [B, T, C, H, W]
    batch_size = 4
    seq_len = 3
    x = torch.rand(batch_size, seq_len, *input_dim)

    out = model(x)

    # Basic assertions: keys present and finite scalars
    assert hasattr(out, "recon_x")
    assert hasattr(out, "recon_loss")
    assert hasattr(out, "kl_loss")
    assert hasattr(out, "total_loss")

    for key in ["recon_loss", "kl_loss", "total_loss"]:
        val = getattr(out, key)
        assert torch.isfinite(val).all(), f"{key} is not finite"

    # Flow loss must be zero if no flows
    assert torch.allclose(out.flow_loss, torch.tensor(0.0, device=out.total_loss.device))


