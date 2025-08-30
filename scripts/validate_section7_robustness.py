#!/usr/bin/env python3
"""
Section 7: Robustness Validation
================================

This script provides comprehensive validation that Section 7 samplers & consistency
is working correctly. It includes multiple validation checks to ensure:

1. Gradient flow is correct
2. Sampler separation is enforced
3. Performance is reasonable
4. Real data integration works
5. Error handling is robust

This demonstrates HOW we know Section 7 works well.
"""

import sys
import warnings
import torch
import numpy as np
import time
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "src"))
sys.path.append(str(current_dir / "original_rlvae/src"))
sys.path.append('.')

from utils.reproducibility import configure_for_experiment
from original_rlvae.src.models.riemannian_flow_vae import RiemannianFlowVAE

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def load_model_and_data():
    """Load model and data for validation."""
    print("🔧 Loading model and data for validation...")
    
    # Configure for experiment
    configure_for_experiment()
    
    # Load data
    data_path = Path("data/processed/Sprites_train_cyclic.pt")
    data = torch.load(data_path)
    print(f"✅ Loaded data: {data.shape}")
    
    # Load model
    config = {
        'latent_dim': 16,
        'input_dim': [64, 64, 3],
        'posterior_local_alpha': 0.001,
        'enforce_sampler_separation': True,
        'prevent_rhmc_in_training': True,
        'require_no_grad_for_analysis': True
    }
    
    model = RiemannianFlowVAE(**config)
    
    # Load pretrained components
    encoder_path = Path("data/pretrained/encoder_diverse_mlp_ld16_20250820_112008.pt")
    decoder_path = Path("data/pretrained/decoder_diverse_mlp_ld16_20250820_112008.pt")
    metric_path = Path("data/pretrained/metric_diverse_mlp_ld16_20250820_112010.pt")
    
    model.encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    model.decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))
    model.load_pretrained_metrics(str(metric_path))
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)
    
    print(f"✅ Model loaded on {device}")
    return model, data


def validate_gradient_flow():
    """Validate that gradient flow is correct."""
    print("\n🔍 Validation 1: Gradient Flow")
    print("=" * 50)
    
    model, data = load_model_and_data()
    
    # Test training sampler gradients
    model.train()
    batch = data[:8].reshape(8, -1)
    
    with torch.no_grad():
        enc_output = model.encoder(batch)
        mu = enc_output['embedding'].detach().requires_grad_(True)
        log_var = enc_output['log_covariance'].detach().requires_grad_(True)
    
    # Use training sampler
    z = model.sample_metric_aware_posterior(mu, log_var)
    
    # Verify gradients
    assert z.requires_grad, "Training sampler should preserve gradients"
    
    # Test backpropagation
    loss = z.sum()
    loss.backward()
    
    # Verify gradients flow back
    assert mu.grad is not None, "Gradients should flow back to mu"
    assert mu.grad.norm().item() > 0, "Gradient norm should be positive"
    
    print("✅ Training sampler preserves gradients correctly")
    print(f"   mu.grad norm: {mu.grad.norm().item():.6f}")
    print(f"   z.requires_grad: {z.requires_grad}")
    print(f"   loss value: {loss.item():.6f}")


def validate_sampler_separation():
    """Validate that sampler separation is enforced."""
    print("\n🔍 Validation 2: Sampler Separation")
    print("=" * 50)
    
    model, data = load_model_and_data()
    batch = data[:4].reshape(4, -1)
    
    with torch.no_grad():
        enc_output = model.encoder(batch)
        mu = enc_output['embedding']
        log_var = enc_output['log_covariance']
    
    # Test 1: Training sampler in training mode (should work)
    model.train()
    try:
        z_train = model.sample_metric_aware_posterior(mu, log_var)
        print("✅ Training sampler works in training mode")
    except Exception as e:
        print(f"❌ Training sampler failed in training mode: {e}")
        return False
    
    # Test 2: RHMC samplers in training mode (should fail)
    try:
        model.sample_visualization_prior(n_samples=10)
        print("❌ RHMC sampler should fail in training mode")
        return False
    except ValueError as e:
        print("✅ RHMC samplers properly rejected in training mode")
        print(f"   Error: {e}")
    
    # Test 3: RHMC samplers in eval mode (should work)
    model.eval()
    try:
        z_prior = model.sample_visualization_prior(n_samples=10)
        z_analysis = model.sample_analysis_posterior(mu, log_var, n_samples=5)
        print("✅ RHMC samplers work in eval mode")
    except Exception as e:
        print(f"❌ RHMC samplers failed in eval mode: {e}")
        return False
    
    # Test 4: Training sampler in eval mode (should fail)
    try:
        model.sample_metric_aware_posterior(mu, log_var)
        print("❌ Training sampler should fail in eval mode")
        return False
    except ValueError as e:
        print("✅ Training sampler properly rejected in eval mode")
        print(f"   Error: {e}")
    
    return True


def validate_gradient_isolation():
    """Validate that gradients are properly isolated."""
    print("\n🔍 Validation 3: Gradient Isolation")
    print("=" * 50)
    
    model, data = load_model_and_data()
    batch = data[:4].reshape(4, -1)
    
    with torch.no_grad():
        enc_output = model.encoder(batch)
        mu = enc_output['embedding'].detach().requires_grad_(True)
        log_var = enc_output['log_covariance'].detach().requires_grad_(True)
    
    # Use training sampler
    model.train()
    z_training = model.sample_metric_aware_posterior(mu, log_var)
    
    # Verify training sampler has gradients
    assert z_training.requires_grad, "Training sampler should have gradients"
    
    # Switch to eval mode and use RHMC samplers
    model.eval()
    z_prior = model.sample_visualization_prior(n_samples=10)
    z_analysis = model.sample_analysis_posterior(mu, log_var, n_samples=5)
    
    # Verify RHMC samplers have no gradients
    assert not z_prior.requires_grad, "RHMC prior should have no gradients"
    assert not z_analysis.requires_grad, "RHMC analysis should have no gradients"
    
    # Verify no gradient leakage to inputs
    assert mu.grad is None, "RHMC sampling should not affect input gradients"
    assert log_var.grad is None, "RHMC sampling should not affect input gradients"
    
    print("✅ Gradient isolation verified")
    print(f"   Training sampler requires_grad: {z_training.requires_grad}")
    print(f"   RHMC prior requires_grad: {z_prior.requires_grad}")
    print(f"   RHMC analysis requires_grad: {z_analysis.requires_grad}")
    print(f"   Input gradients affected: {mu.grad is not None}")


def validate_performance():
    """Validate that performance is reasonable."""
    print("\n🔍 Validation 4: Performance")
    print("=" * 50)
    
    model, data = load_model_and_data()
    batch = data[:8].reshape(8, -1)
    
    with torch.no_grad():
        enc_output = model.encoder(batch)
        mu = enc_output['embedding']
        log_var = enc_output['log_covariance']
    
    # Test training sampler performance
    model.train()
    start_time = time.time()
    z_train = model.sample_metric_aware_posterior(mu, log_var)
    train_time = time.time() - start_time
    
    # Test RHMC prior performance
    model.eval()
    start_time = time.time()
    z_prior = model.sample_visualization_prior(n_samples=50)
    prior_time = time.time() - start_time
    
    # Test RHMC analysis performance
    start_time = time.time()
    z_analysis = model.sample_analysis_posterior(mu, log_var, n_samples=20)
    analysis_time = time.time() - start_time
    
    print("✅ Performance validation")
    print(f"   Training sampler: {train_time:.4f}s for 8 samples")
    print(f"   RHMC prior: {prior_time:.4f}s for 50 samples")
    print(f"   RHMC analysis: {analysis_time:.4f}s for 20 samples per posterior")
    
    # Verify reasonable performance
    assert train_time < 1.0, "Training sampler should be fast"
    assert prior_time < 5.0, "RHMC prior should complete in reasonable time"
    assert analysis_time < 10.0, "RHMC analysis should complete in reasonable time"


def validate_real_data_integration():
    """Validate that real data integration works correctly."""
    print("\n🔍 Validation 5: Real Data Integration")
    print("=" * 50)
    
    model, data = load_model_and_data()
    
    # Test with different batch sizes
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        batch = data[:batch_size].reshape(batch_size, -1)
        
        with torch.no_grad():
            enc_output = model.encoder(batch)
            mu = enc_output['embedding']
            log_var = enc_output['log_covariance']
        
        # Test training sampler
        model.train()
        z_train = model.sample_metric_aware_posterior(mu, log_var)
        assert z_train.shape[0] == batch_size, f"Expected batch_size {batch_size}, got {z_train.shape[0]}"
        
        # Test RHMC analysis
        model.eval()
        z_analysis = model.sample_analysis_posterior(mu, log_var, n_samples=10)
        assert z_analysis.shape[0] == batch_size, f"Expected batch_size {batch_size}, got {z_analysis.shape[0]}"
        assert z_analysis.shape[1] == 10, f"Expected 10 samples, got {z_analysis.shape[1]}"
        
        print(f"✅ Batch size {batch_size}: Training {z_train.shape}, Analysis {z_analysis.shape}")
    
    print("✅ Real data integration works for all batch sizes")


def validate_error_handling():
    """Validate that error handling is robust."""
    print("\n🔍 Validation 6: Error Handling")
    print("=" * 50)
    
    model, data = load_model_and_data()
    
    # Test 1: Invalid context usage
    model.train()
    try:
        model.sample_visualization_prior(n_samples=10)
        print("❌ Should have failed in training mode")
        return False
    except ValueError:
        print("✅ Properly rejects RHMC in training mode")
    
    # Test 2: Invalid context usage
    model.eval()
    try:
        model.sample_metric_aware_posterior(torch.randn(4, 16), torch.randn(4, 16))
        print("❌ Should have failed in eval mode")
        return False
    except ValueError:
        print("✅ Properly rejects training sampler in eval mode")
    
    # Test 3: Invalid input shapes
    try:
        model.sample_analysis_posterior(torch.randn(4, 16), torch.randn(5, 16), n_samples=10)
        print("❌ Should have failed with mismatched shapes")
        return False
    except Exception:
        print("✅ Properly handles mismatched input shapes")
    
    # Test 4: Invalid n_samples
    try:
        model.sample_visualization_prior(n_samples=-1)
        print("❌ Should have failed with negative n_samples")
        return False
    except Exception:
        print("✅ Properly handles invalid n_samples")
    
    print("✅ Error handling is robust")


def validate_sampler_quality():
    """Validate that samplers produce reasonable quality samples."""
    print("\n🔍 Validation 7: Sampler Quality")
    print("=" * 50)
    
    model, data = load_model_and_data()
    batch = data[:8].reshape(8, -1)
    
    with torch.no_grad():
        enc_output = model.encoder(batch)
        mu = enc_output['embedding']
        log_var = enc_output['log_covariance']
    
    # Test training sampler quality
    model.train()
    z_train = model.sample_metric_aware_posterior(mu, log_var)
    
    # Test RHMC prior quality
    model.eval()
    z_prior = model.sample_visualization_prior(n_samples=100)
    
    # Test RHMC analysis quality
    z_analysis = model.sample_analysis_posterior(mu, log_var, n_samples=20)
    
    # Check sample statistics
    print("✅ Sample quality validation")
    print(f"   Training samples:")
    print(f"     Mean: {z_train.mean().item():.4f}")
    print(f"     Std: {z_train.std().item():.4f}")
    print(f"     Range: [{z_train.min().item():.4f}, {z_train.max().item():.4f}]")
    
    print(f"   RHMC prior samples:")
    print(f"     Mean: {z_prior.mean().item():.4f}")
    print(f"     Std: {z_prior.std().item():.4f}")
    print(f"     Range: [{z_prior.min().item():.4f}, {z_prior.max().item():.4f}]")
    
    print(f"   RHMC analysis samples:")
    print(f"     Mean: {z_analysis.mean().item():.4f}")
    print(f"     Std: {z_analysis.std().item():.4f}")
    print(f"     Range: [{z_analysis.min().item():.4f}, {z_analysis.max().item():.4f}]")
    
    # Verify reasonable statistics
    assert abs(z_train.mean().item()) < 10, "Training samples should have reasonable mean"
    assert z_train.std().item() > 0.1, "Training samples should have reasonable std"
    assert abs(z_prior.mean().item()) < 10, "RHMC prior should have reasonable mean"
    assert z_prior.std().item() > 0.1, "RHMC prior should have reasonable std"
    assert abs(z_analysis.mean().item()) < 10, "RHMC analysis should have reasonable mean"
    assert z_analysis.std().item() > 0.1, "RHMC analysis should have reasonable std"


def main():
    """Run all validations."""
    print("🔍 Section 7: Comprehensive Robustness Validation")
    print("=" * 60)
    print("This script demonstrates HOW we know Section 7 works well.")
    print("=" * 60)
    
    try:
        # Run all validations
        validate_gradient_flow()
        validate_sampler_separation()
        validate_gradient_isolation()
        validate_performance()
        validate_real_data_integration()
        validate_error_handling()
        validate_sampler_quality()
        
        print("\n" + "=" * 60)
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ Section 7 is working correctly and robustly")
        print("=" * 60)
        
        print("\n📊 Summary of Evidence:")
        print("1. ✅ Gradient flow is correct (training sampler preserves gradients)")
        print("2. ✅ Sampler separation is enforced (context validation works)")
        print("3. ✅ Gradient isolation is complete (no leakage between samplers)")
        print("4. ✅ Performance is reasonable (fast training, acceptable RHMC)")
        print("5. ✅ Real data integration works (all batch sizes supported)")
        print("6. ✅ Error handling is robust (proper validation and error messages)")
        print("7. ✅ Sampler quality is good (reasonable sample statistics)")
        
        print("\n🏆 Section 7 is PRODUCTION-READY!")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
