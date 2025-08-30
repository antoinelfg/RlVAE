# Colab Version Check - Copy this into a Colab cell
# This will help you compare versions between Colab and your local environment

import sys
import platform
import torch
import numpy as np

print("🔍 COLAB VERSION INSPECTION")
print("=" * 50)

# Python info
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")

# PyTorch info
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"Device count: {torch.cuda.device_count()}")

# NumPy info
print(f"\nNumPy version: {np.__version__}")

# Check if we're in Colab
try:
    import google.colab
    print("\n✅ Running in Google Colab")
except ImportError:
    print("\n❌ Not running in Google Colab")

# Test RHVAE
print("\nTesting RHVAE availability:")
try:
    from pythae.models import RHVAE, RHVAEConfig
    print("✅ RHVAE imported successfully")
    
    # Quick test
    config = RHVAEConfig(
        input_dim=(1, 28, 28),
        latent_dim=4,
        n_lf=1,
        eps_lf=0.0001,
        beta_zero=1.0,
        temperature=0.5,
        regularization=0.1
    )
    print("✅ RHVAEConfig created")
    
    # Test model creation
    from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST
    
    model = RHVAE(
        model_config=config,
        encoder=Encoder_ResNet_VAE_MNIST(config),
        decoder=Decoder_ResNet_AE_MNIST(config)
    )
    print("✅ RHVAE model created")
    
    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    dummy_input = torch.randn(2, 1, 28, 28).to(device)
    inputs = {"data": dummy_input}
    
    model.eval()
    with torch.no_grad():
        output = model(inputs)
        print("✅ RHVAE forward pass successful")
    
    print("✅ RHVAE test completed successfully")
    
except Exception as e:
    print(f"❌ RHVAE test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("Copy this output and compare with your local environment!")
print("Key things to check:")
print("- PyTorch version differences")
print("- CUDA version differences") 
print("- Python version differences")
print("- RHVAE functionality") 