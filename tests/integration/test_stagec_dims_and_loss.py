import torch
import torch.nn.functional as F


def test_stagec_shapes_and_loss():
    from src.models.riemannian_flow_vae import RiemannianFlowVAE

    torch.manual_seed(0)

    # Synthetic batch: [B, T, C, H, W]
    batch_size, seq_len, channels, height, width = 4, 5, 3, 64, 64
    x = torch.rand(batch_size, seq_len, channels, height, width)

    # Lightweight model, no Riemannian deps needed for shape/loss check
    model = RiemannianFlowVAE(
        input_dim=(channels, height, width),
        latent_dim=10,
        n_flows=2,
        posterior_type="gaussian",
        riemannian_kl_mode="quadratic",
        reconstruction_mode="all",
    )

    model.eval()
    out = model(x)

    # Shape checks
    assert out.recon_x.shape == x.shape, f"recon_x shape {tuple(out.recon_x.shape)} != input {tuple(x.shape)}"
    assert out.z.dim() == 3 and out.z.shape[0] == batch_size and out.z.shape[1] == seq_len, \
        f"z shape {tuple(out.z.shape)} unexpected; expected [B,T,D]"

    # Reconstruction loss scaling check (default MSE * 255.0)
    mse = F.mse_loss(out.recon_x, x, reduction='mean') * 255.0
    assert torch.isfinite(out.reconstruction_loss), "Non-finite reconstruction_loss"
    assert abs(out.reconstruction_loss.item() - mse.item()) / (mse.item() + 1e-8) < 1e-4, \
        f"reconstruction_loss mismatch: got {out.reconstruction_loss.item():.6f}, expected {mse.item():.6f}"

    # Basic no-mixing sanity: variance across batch vs time should both be > 0
    var_batch = out.recon_x[:, 0].var().item()
    var_time = out.recon_x[0].var().item()
    assert var_batch > 0 and var_time > 0, "Unexpected zero variance indicating potential mixing"


if __name__ == "__main__":
    test_stagec_shapes_and_loss()
    print("OK: Stage C dims and loss checks passed")


