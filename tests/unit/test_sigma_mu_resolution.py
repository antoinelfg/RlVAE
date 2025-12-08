import sys
from pathlib import Path
import torch

# Ensure repository root (containing `src`) is on the Python path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rlvae.models.components.loss_manager import LossManager


def test_resolve_sigma_mu_prefers_rhmc_traj_and_extracts_diag():
    # Setup
    B, D = 3, 4
    device = torch.device('cpu')
    mu = torch.randn(B, D, device=device)

    # Build per-batch covariance blocks [B, D, D]
    sig_blocks = torch.zeros(B, D, D, device=device)
    for b in range(B):
        A = torch.randn(D, D, device=device)
        sig_blocks[b] = (A @ A.T) + 1e-3 * torch.eye(D, device=device)

    # Assemble a [B,B,D,D] tensor with diagonal blocks = sig_blocks
    Sigma_full = torch.zeros(B, B, D, D, device=device)
    idx = torch.arange(B, device=device)
    Sigma_full[idx, idx] = sig_blocks

    rhmc_traj_info = {'Sigma_mu': Sigma_full}

    # Exercise
    lm = LossManager(device=device)
    Sigma_resolved = lm._resolve_sigma_mu(mu, None, metric_tensor=None, rhmc_posterior=None, rhmc_traj_info=rhmc_traj_info)

    # Verify shape and equality with diagonal extraction
    assert Sigma_resolved.shape == (B, D, D)
    # Frobenius norm of difference should be zero
    diff = (Sigma_resolved - sig_blocks).abs().max().item()
    assert diff == 0.0

    # Logdet equality
    ld_resolved = torch.linalg.slogdet(Sigma_resolved.float())[1]
    ld_expected = torch.linalg.slogdet(sig_blocks.float())[1]
    assert torch.allclose(ld_resolved, ld_expected, atol=0.0, rtol=0.0)
