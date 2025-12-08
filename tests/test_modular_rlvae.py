"""
Test for modular RLVAE architecture.
"""

import torch
from omegaconf import OmegaConf

import sys
sys.path.append('.')
from src.models.composite.rlvae import RLVAE


def test_modular_rlvae_creation():
    """Test that modular RLVAE can be created."""
    # Create config
    config = OmegaConf.create({
        "input_dim": [3, 64, 64],
        "latent_dim": 16,
        "encoder": {
            "_target_": "src.models.components.encoders.mlp_encoder.MLPEncoder",
            "hidden_dims": [256, 128],
            "dropout": 0.1,
            "activation": "relu"
        },
        "decoder": {
            "_target_": "src.models.components.decoders.mlp_decoder.MLPDecoder",
            "hidden_dims": [128, 256],
            "dropout": 0.1,
            "activation": "relu",
            "output_activation": "sigmoid"
        },
        "metric": {
            "_target_": "src.models.components.metric.identity_metric.IdentityMetric"
        },
        "posterior": {
            "_target_": "src.models.components.posteriors.local_riemannian.LocalRiemannianPosterior",
            "alpha": 0.5
        },
        "reconstruction_loss": {
            "_target_": "src.models.components.losses.reconstruction.GaussianReconstructionLoss",
            "sigma": 0.1
        },
        "kl_loss": {
            "_target_": "src.models.components.losses.kl.KLEuclideanLoss",
            "beta": 1.0
        },
        "elbo_loss": {
            "_target_": "src.models.components.losses.elbo.ELBOLoss",
            "flow_loss_weight": 1.0,
            "loop_penalty_weight": 0.0
        }
    })
    
    # Create model
    model = RLVAE(**config)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    
    with torch.no_grad():
        outputs = model(x)
    
    # Check outputs
    assert "reconstruction" in outputs
    assert "latent_samples" in outputs
    assert "mu" in outputs
    assert "log_var" in outputs
    assert "loss" in outputs
    
    # Check shapes
    assert outputs["reconstruction"].shape == x.shape
    assert outputs["latent_samples"].shape == (batch_size, 16)
    assert outputs["mu"].shape == (batch_size, 16)
    assert outputs["log_var"].shape == (batch_size, 16)
    
    print("✅ Modular RLVAE test passed!")


def test_modular_rlvae_with_learned_metric():
    """Test modular RLVAE with learned metric."""
    # Create config with learned metric
    config = OmegaConf.create({
        "input_dim": [3, 64, 64],
        "latent_dim": 8,  # Smaller for faster test
        "encoder": {
            "_target_": "src.models.components.encoders.mlp_encoder.MLPEncoder",
            "hidden_dims": [128, 64],
            "dropout": 0.1,
            "activation": "relu"
        },
        "decoder": {
            "_target_": "src.models.components.decoders.mlp_decoder.MLPDecoder",
            "hidden_dims": [64, 128],
            "dropout": 0.1,
            "activation": "relu",
            "output_activation": "sigmoid"
        },
        "metric": {
            "_target_": "src.models.components.metric.learned_metric.LearnedMetric",
            "hidden_dims": [64],
            "temperature": 0.1,
            "regularization": 0.01,
            "normalize_for_kl": "geomean"
        },
        "posterior": {
            "_target_": "src.models.components.posteriors.local_riemannian.LocalRiemannianPosterior",
            "alpha": 0.5
        },
        "reconstruction_loss": {
            "_target_": "src.models.components.losses.reconstruction.GaussianReconstructionLoss",
            "sigma": 0.1
        },
        "kl_loss": {
            "_target_": "src.models.components.losses.kl.KLVolumePriorLoss",
            "beta": 1.0
        },
        "elbo_loss": {
            "_target_": "src.models.components.losses.elbo.ELBOLoss",
            "flow_loss_weight": 1.0,
            "loop_penalty_weight": 0.0
        }
    })
    
    # Create model
    model = RLVAE(**config)
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 64, 64)
    
    with torch.no_grad():
        outputs = model(x)
    
    # Check outputs
    assert "reconstruction" in outputs
    assert "latent_samples" in outputs
    assert "mu" in outputs
    assert "loss" in outputs
    
    # Check shapes
    assert outputs["reconstruction"].shape == x.shape
    assert outputs["latent_samples"].shape == (batch_size, 8)
    assert outputs["mu"].shape == (batch_size, 8)
    
    print("✅ Modular RLVAE with learned metric test passed!")


if __name__ == "__main__":
    test_modular_rlvae_creation()
    test_modular_rlvae_with_learned_metric()
    print("🎉 All modular RLVAE tests passed!")
