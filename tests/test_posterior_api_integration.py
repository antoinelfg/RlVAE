#!/usr/bin/env python3
"""
Test for Posterior API Integration
==================================

This test validates the integration of different posterior types including the new RHMC posterior.
Tests include:
1. riem_hmc posterior type configuration
2. RHMC sampler integration in forward pass
3. Comparison with other posterior types
4. Loss computation for different posteriors
"""

import torch
import numpy as np
from pathlib import Path
import sys
import time
sys.path.append(str(Path(__file__).parent.parent))

from src.models.riemannian_flow_vae import RiemannianFlowVAE


def test_riem_hmc_posterior_configuration():
    """Test that riem_hmc posterior type is properly configured."""
    print("🧪 Testing RHMC Posterior Configuration...")
    
    # Load pretrained model
    model = RiemannianFlowVAE(
        input_dim=(3, 64, 64),
        latent_dim=2,
        n_flows=0
    )
    
    # Move model to device first
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.load_pretrained_components(
        encoder_path="data/pretrained/encoder_diverse_mlp_ld2_20250730_105419.pt",
        decoder_path="data/pretrained/decoder_diverse_mlp_ld2_20250730_105419.pt",
        metric_path="data/pretrained/metric_diverse_mlp_ld2_20250730_105420.pt"
    )
    
    # Test setting riem_hmc posterior type
    model.set_posterior_type("riem_hmc")
    
    print(f"✅ Posterior type: {model.posterior_type}")
    assert model.posterior_type == "riem_hmc", f"Posterior type not set correctly: {model.posterior_type}"
    
    # Test that RHMC sampler is initialized during forward pass
    batch_size = 4
    n_obs = 5
    x_test = torch.randn(batch_size, n_obs, 3, 64, 64, device=device)
    
    # Forward pass should initialize RHMC sampler
    with torch.no_grad():
        output = model(x_test)
    
    print(f"✅ RHMC sampler initialized: {hasattr(model, '_rhmc_sampler')}")
    assert hasattr(model, '_rhmc_sampler'), "RHMC sampler not initialized"
    
    print("✅ RHMC posterior configuration test PASSED")
    return True


def test_posterior_type_comparison():
    """Compare different posterior types."""
    print("\n🧪 Testing Posterior Type Comparison...")
    
    # Load pretrained model
    model = RiemannianFlowVAE(
        input_dim=(3, 64, 64),
        latent_dim=2,
        n_flows=0
    )
    
    # Move model to device first
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.load_pretrained_components(
        encoder_path="data/pretrained/encoder_diverse_mlp_ld2_20250730_105419.pt",
        decoder_path="data/pretrained/decoder_diverse_mlp_ld2_20250730_105419.pt",
        metric_path="data/pretrained/metric_diverse_mlp_ld2_20250730_105420.pt"
    )
    
    batch_size = 3
    n_obs = 4
    x_test = torch.randn(batch_size, n_obs, 3, 64, 64, device=device)
    
    posterior_types = ["gaussian", "riemannian_metric", "riem_hmc"]
    results = {}
    
    for posterior_type in posterior_types:
        print(f"\n--- Testing {posterior_type} posterior ---")
        
        # Set posterior type
        model.set_posterior_type(posterior_type)
        
        # Set model to training mode to trigger debugging
        model.train()
        
        # Forward pass
        output = model(x_test)
        
        # Store results
        results[posterior_type] = {
            'total_loss': output.total_loss.item(),
            'recon_loss': output.recon_loss.item(),
            'kl_loss': output.kl_loss.item(),
            'z_shape': [z.shape for z in output.z],
            'mu_shape': output.mu.shape,
            'log_var_shape': output.log_var.shape
        }
        
        print(f"✅ {posterior_type} - Total loss: {results[posterior_type]['total_loss']:.3f}")
        print(f"✅ {posterior_type} - Recon loss: {results[posterior_type]['recon_loss']:.3f}")
        print(f"✅ {posterior_type} - KL loss: {results[posterior_type]['kl_loss']:.3f}")
    
    # Verify all posteriors produce reasonable results
    for posterior_type, result in results.items():
        assert result['total_loss'] > 0, f"{posterior_type}: Total loss should be positive"
        assert result['recon_loss'] > 0, f"{posterior_type}: Reconstruction loss should be positive"
        assert result['kl_loss'] > 0, f"{posterior_type}: KL loss should be positive"
        assert len(result['z_shape']) == n_obs, f"{posterior_type}: Wrong number of latent timesteps"
    
    print("✅ Posterior type comparison test PASSED")
    return True


def test_rhcm_sampler_integration():
    """Test that RHMC sampler is properly integrated in the forward pass."""
    print("\n🧪 Testing RHMC Sampler Integration...")
    
    # Load pretrained model
    model = RiemannianFlowVAE(
        input_dim=(3, 64, 64),
        latent_dim=2,
        n_flows=0
    )
    
    # Move model to device first
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.load_pretrained_components(
        encoder_path="data/pretrained/encoder_diverse_mlp_ld2_20250730_105419.pt",
        decoder_path="data/pretrained/decoder_diverse_mlp_ld2_20250730_105419.pt",
        metric_path="data/pretrained/metric_diverse_mlp_ld2_20250730_105420.pt"
    )
    
    # Set to RHMC posterior
    model.set_posterior_type("riem_hmc")
    
    batch_size = 2
    n_obs = 3
    x_test = torch.randn(batch_size, n_obs, 3, 64, 64, device=device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x_test)
    
    print(f"✅ RHMC sampler initialized: {hasattr(model, '_rhmc_sampler')}")
    print(f"✅ RHMC sampler type: {type(model._rhmc_sampler)}")
    print(f"✅ Output z shapes: {[z.shape for z in output.z]}")
    print(f"✅ Output mu shape: {output.mu.shape}")
    print(f"✅ Output log_var shape: {output.log_var.shape}")
    print(f"✅ Total loss: {output.total_loss.item():.3f}")
    
    # Verify RHMC sampler properties
    assert hasattr(model, '_rhmc_sampler'), "RHMC sampler not initialized"
    assert hasattr(model._rhmc_sampler, 'sample_posterior'), "RHMC sampler missing sample_posterior method"
    assert hasattr(model._rhmc_sampler, '_compute_hamiltonian'), "RHMC sampler missing Hamiltonian computation"
    assert hasattr(model._rhmc_sampler, '_initialize_momentum'), "RHMC sampler missing momentum initialization"
    
    # Verify output properties
    assert len(output.z) == n_obs, f"Wrong number of latent timesteps: {len(output.z)}"
    assert output.mu.shape == (batch_size, 2), f"Wrong mu shape: {output.mu.shape}"
    assert output.log_var.shape == (batch_size, 2), f"Wrong log_var shape: {output.log_var.shape}"
    assert torch.all(torch.isfinite(output.total_loss)), "Total loss is not finite"
    
    print("✅ RHMC sampler integration test PASSED")
    return True


def test_loss_computation_for_different_posteriors():
    """Test that loss computation works correctly for different posterior types."""
    print("\n🧪 Testing Loss Computation for Different Posteriors...")
    
    # Load pretrained model
    model = RiemannianFlowVAE(
        input_dim=(3, 64, 64),
        latent_dim=2,
        n_flows=0
    )
    
    # Move model to device first
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.load_pretrained_components(
        encoder_path="data/pretrained/encoder_diverse_mlp_ld2_20250730_105419.pt",
        decoder_path="data/pretrained/decoder_diverse_mlp_ld2_20250730_105419.pt",
        metric_path="data/pretrained/metric_diverse_mlp_ld2_20250730_105420.pt"
    )
    
    batch_size = 2
    n_obs = 3
    x_test = torch.randn(batch_size, n_obs, 3, 64, 64, device=device)
    
    posterior_types = ["gaussian", "riemannian_metric", "riem_hmc"]
    loss_results = {}
    
    for posterior_type in posterior_types:
        print(f"\n--- Testing loss computation for {posterior_type} ---")
        
        # Set posterior type
        model.set_posterior_type(posterior_type)
        
        # Forward pass
        with torch.no_grad():
            output = model(x_test)
        
        # Store loss components
        loss_results[posterior_type] = {
            'total_loss': output.total_loss.item(),
            'recon_loss': output.recon_loss.item(),
            'kl_loss': output.kl_loss.item(),
            'flow_loss': output.flow_loss.item(),
            'loop_penalty': output.loop_penalty.item(),
            'riemannian_loss': output.riemannian_loss.item()
        }
        
        print(f"✅ {posterior_type} losses:")
        print(f"   Total: {loss_results[posterior_type]['total_loss']:.3f}")
        print(f"   Recon: {loss_results[posterior_type]['recon_loss']:.3f}")
        print(f"   KL: {loss_results[posterior_type]['kl_loss']:.3f}")
        print(f"   Flow: {loss_results[posterior_type]['flow_loss']:.3f}")
        print(f"   Loop: {loss_results[posterior_type]['loop_penalty']:.3f}")
        print(f"   Riemannian: {loss_results[posterior_type]['riemannian_loss']:.3f}")
    
    # Verify loss properties
    for posterior_type, losses in loss_results.items():
        # All losses should be finite
        for loss_name, loss_value in losses.items():
            assert np.isfinite(loss_value), f"{posterior_type} {loss_name} is not finite: {loss_value}"
        
        # Total loss should be positive
        assert losses['total_loss'] > 0, f"{posterior_type}: Total loss should be positive"
        
        # Reconstruction loss should be positive
        assert losses['recon_loss'] > 0, f"{posterior_type}: Reconstruction loss should be positive"
        
        # KL loss should be positive
        assert losses['kl_loss'] > 0, f"{posterior_type}: KL loss should be positive"
    
    # Riemannian posteriors should have similar KL losses (both use metric-aware KL)
    riemannian_kl_losses = [loss_results['riemannian_metric']['kl_loss'], loss_results['riem_hmc']['kl_loss']]
    kl_difference = abs(riemannian_kl_losses[0] - riemannian_kl_losses[1])
    print(f"✅ KL loss difference between Riemannian posteriors: {kl_difference:.3f}")
    assert kl_difference < 1.0, "Riemannian posteriors should have similar KL losses"
    
    print("✅ Loss computation test PASSED")
    return True


def test_configuration_file_integration():
    """Test that the riem_hmc.yaml configuration file works correctly."""
    print("\n🧪 Testing Configuration File Integration...")
    
    # Test that the configuration file exists and can be loaded
    config_path = Path("conf/model/riem_hmc.yaml")
    assert config_path.exists(), f"Configuration file not found: {config_path}"
    
    # Test that the configuration has the correct structure
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Configuration file loaded: {config_path}")
    print(f"✅ Posterior type in config: {config.get('posterior_type', 'NOT_FOUND')}")
    print(f"✅ HMC config present: {'hmc_config' in config}")
    
    # Verify configuration structure
    assert config.get('posterior_type') == 'riem_hmc', "Configuration should specify riem_hmc posterior"
    assert 'hmc_config' in config, "Configuration should include HMC parameters"
    assert 'mcmc_steps_nbr' in config['hmc_config'], "HMC config should include mcmc_steps_nbr"
    assert 'n_lf' in config['hmc_config'], "HMC config should include n_lf"
    assert 'eps_lf' in config['hmc_config'], "HMC config should include eps_lf"
    
    print("✅ Configuration file integration test PASSED")
    return True


def main():
    """Run all posterior API integration tests."""
    print("🚀 Testing Posterior API Integration")
    print("=" * 60)
    
    try:
        test_riem_hmc_posterior_configuration()
        test_posterior_type_comparison()
        test_rhcm_sampler_integration()
        test_loss_computation_for_different_posteriors()
        test_configuration_file_integration()
        
        print("\n✅ ALL POSTERIOR API INTEGRATION TESTS PASSED!")
        print("=" * 60)
        
        print("\n📊 Posterior API Integration Summary:")
        print("✅ riem_hmc posterior type properly configured")
        print("✅ RHMC sampler integrated in forward pass")
        print("✅ Comparison with other posterior types working")
        print("✅ Loss computation correct for all posteriors")
        print("✅ Configuration file integration working")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 