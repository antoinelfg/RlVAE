#!/usr/bin/env python3
"""
Test script to verify the clean Riemannian Flow VAE setup.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir / "src"
lib_src_dir = src_dir / "lib" / "src"

sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(lib_src_dir))

def test_imports():
    """Test that all necessary imports work."""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import lightning
        print(f"✅ Lightning: {lightning.__version__}")
    except ImportError as e:
        print(f"❌ Lightning import failed: {e}")
        return False
    
    try:
        from src.lib.src.pythae.models.rhvae.rhvae_model import RHVAE
        print("✅ RHVAE components available")
    except ImportError as e:
        print(f"⚠️ RHVAE components not available: {e}")
    
    try:
        from src.models.riemannian_flow_vae import RiemannianFlowVAE
        print("✅ RiemannianFlowVAE import successful")
    except ImportError as e:
        print(f"❌ RiemannianFlowVAE import failed: {e}")
        return False
    
    return True

def test_data_files():
    """Test that all data files exist."""
    print("\n📊 Testing data files...")
    
    from config import validate_paths
    return validate_paths()

def test_model_creation():
    """Test creating a model instance."""
    print("\n🧠 Testing model creation...")
    
    try:
        from src.models.riemannian_flow_vae import RiemannianFlowVAE
        import torch
        
        model = RiemannianFlowVAE(
            input_dim=(3, 64, 64),
            latent_dim=16,
            n_flows=5,
        )
        print("✅ Model created successfully")
        
        # Test forward pass with dummy data
        batch_size = 4
        n_obs = 6  # n_flows + 1
        dummy_x = torch.randn(batch_size, n_obs, 3, 64, 64)
        
        with torch.no_grad():
            output = model(dummy_x)
        
        print(f"✅ Forward pass successful, output shape: {output.recon_x.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Model creation/testing failed: {e}")
        return False

def test_pretrained_loading():
    """Test loading pretrained components."""
    print("\n🔧 Testing pretrained component loading...")
    
    try:
        from src.models.riemannian_flow_vae import RiemannianFlowVAE
        from config import ENCODER_PATH, DECODER_PATH, METRIC_SCALED_PATH
        
        model = RiemannianFlowVAE(
            input_dim=(3, 64, 64),
            latent_dim=16,
            n_flows=5,
        )
        
        model.load_pretrained_components(
            encoder_path=str(ENCODER_PATH),
            decoder_path=str(DECODER_PATH),
            metric_path=str(METRIC_SCALED_PATH),
            temperature_override=0.7
        )
        
        print("✅ Pretrained components loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Pretrained loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running Clean Riemannian Flow VAE Setup Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_files,
        test_model_creation,
        test_pretrained_loading,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("🏁 Test Results:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {i+1}. {test.__name__}: {status}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Setup is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the setup.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 