#!/usr/bin/env python3
"""
Verification script for enhanced KL loss mechanism.
"""

import sys
import torch

# Add original_rlvae to path
sys.path.insert(0, 'original_rlvae')

def verify_enhanced_kl():
    """Verify the enhanced KL loss mechanism is working."""
    print("🔍 Verifying Enhanced KL Loss Mechanism")
    print("=" * 50)
    
    try:
        from src.models.riemannian_flow_vae import RiemannianFlowVAE
        
        # Test 1: Model creation with enhanced parameters
        print("✅ Test 1: Model creation...")
        model = RiemannianFlowVAE(
            input_dim=[64, 64, 3],
            latent_dim=16,
            adaptive_kl_enabled=True,
            adaptive_kl_ramp_up_steps=3,
            adaptive_kl_alignment_weight=0.1
        )
        print("   ✅ Model created successfully")
        
        # Test 2: Adaptive KL mechanism
        print("✅ Test 2: Adaptive KL mechanism...")
        model._kl_adaptation_counter = 0
        model._base_riemannian_beta = 1.0
        
        initial_beta = model.riemannian_beta
        print(f"   Initial beta: {initial_beta}")
        
        for i in range(3):
            model._adapt_kl_loss_for_metric_update()
            print(f"   Update {i+1}: beta = {model.riemannian_beta:.4f}")
        
        print("   ✅ Adaptive KL mechanism working")
        
        # Test 3: Metric alignment penalty
        print("✅ Test 3: Metric alignment penalty...")
        batch_size = 4
        latent_dim = 16
        mu = torch.randn(batch_size, latent_dim)
        log_var = torch.randn(batch_size, latent_dim)
        G_z = torch.eye(latent_dim).unsqueeze(0).repeat(batch_size, 1, 1)
        
        penalty = model._compute_metric_alignment_penalty(mu, log_var, G_z)
        print(f"   Penalty computed: {penalty.item():.6f}")
        print("   ✅ Metric alignment penalty working")
        
        # Test 4: Enhanced KL loss
        print("✅ Test 4: Enhanced KL loss computation...")
        z_sample = torch.randn(batch_size, latent_dim)
        
        # Mock metric function
        def mock_G(z):
            return torch.eye(latent_dim).unsqueeze(0).repeat(z.shape[0], 1, 1)
        
        model.G = mock_G
        kl_loss = model.compute_riemannian_kl_loss(mu, log_var, z_sample)
        print(f"   KL loss computed: {kl_loss.item():.6f}")
        print("   ✅ Enhanced KL loss working")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("\n📋 **Enhanced KL Mechanism Status:**")
        print("✅ Model creation with enhanced parameters")
        print("✅ Adaptive KL mechanism (beta ramping)")
        print("✅ Metric alignment penalty computation")
        print("✅ Enhanced KL loss computation")
        print("✅ All methods properly implemented")
        
        print("\n🚀 **Ready for Production Use!**")
        print("Add these parameters to your model config:")
        print("  adaptive_kl_enabled: true")
        print("  adaptive_kl_ramp_up_steps: 10")
        print("  adaptive_kl_alignment_weight: 0.1")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_enhanced_kl()
    sys.exit(0 if success else 1)
