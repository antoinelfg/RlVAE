import torch

from src.rlvae.models.components.encoder_manager import EncoderManager
from src.rlvae.models.components.decoder_manager import DecoderManager


def test_cnn_gray_encoder_forward_shapes():
    manager = EncoderManager(
        input_dim=(1, 32, 32),
        latent_dim=8,
        architecture="cnn_gray",
        device=torch.device("cpu"),
    )

    x = torch.randn(4, 1, 32, 32)
    outputs = manager(x)

    assert hasattr(outputs, "embedding")
    assert hasattr(outputs, "log_covariance")
    assert outputs.embedding.shape == (4, 8)
    assert outputs.log_covariance.shape == (4, 8)


def test_cnn_gray_decoder_forward_shapes():
    manager = DecoderManager(
        input_dim=(1, 32, 32),
        latent_dim=8,
        architecture="cnn_gray",
        device=torch.device("cpu"),
    )

    z = torch.randn(4, 8)
    outputs = manager(z)

    assert hasattr(outputs, "reconstruction")
    assert outputs.reconstruction.shape == (4, 1, 32, 32)


def test_mlp_gray_aliases():
    enc = EncoderManager(
        input_dim=(1, 16, 16),
        latent_dim=4,
        architecture="mlp_gray",
        device=torch.device("cpu"),
    )
    dec = DecoderManager(
        input_dim=(1, 16, 16),
        latent_dim=4,
        architecture="mlp_gray",
        device=torch.device("cpu"),
    )
    x = torch.randn(2, 1, 16, 16)
    z_out = enc(x)
    assert z_out.embedding.shape == (2, 4)
    recon = dec(torch.randn(2, 4)).reconstruction
    assert recon.shape == (2, 1, 16, 16)
