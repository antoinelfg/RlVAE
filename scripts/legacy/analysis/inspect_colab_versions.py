#!/usr/bin/env python3
"""
Comprehensive version inspection script for Colab.
Run this in Colab to compare versions with your local environment.
"""

import sys
import platform
import subprocess
import os

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def get_python_info():
    """Get Python version and platform information."""
    print_section("PYTHON & PLATFORM INFO")
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")

def get_pytorch_info():
    """Get PyTorch and CUDA information."""
    print_section("PYTORCH & CUDA INFO")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch location: {torch.__file__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name()}")
            print(f"Device capability: {torch.cuda.get_device_capability()}")
            
            # Memory info
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            print(f"GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("CUDA not available")
            
    except ImportError as e:
        print(f"PyTorch not installed: {e}")

def get_pythae_info():
    """Get Pythae library information."""
    print_section("PYTHAE INFO")
    
    try:
        import pythae
        print(f"Pythae location: {pythae.__file__}")
        
        # Try to get version
        try:
            version = pythae.__version__
            print(f"Pythae version: {version}")
        except AttributeError:
            print("Pythae version: Not available (no __version__ attribute)")
        
        # Check available models
        try:
            from pythae.models import RHVAE, RHVAEConfig
            print("✅ RHVAE available")
            
            # Test basic functionality
            config = RHVAEConfig(
                input_dim=(1, 28, 28),
                latent_dim=4,
                n_lf=1,
                eps_lf=0.0001,
                beta_zero=1.0,
                temperature=0.5,
                regularization=0.1
            )
            print("✅ RHVAEConfig works")
            
        except Exception as e:
            print(f"❌ RHVAE test failed: {e}")
            
    except ImportError as e:
        print(f"Pythae not installed: {e}")

def get_numpy_info():
    """Get NumPy information."""
    print_section("NUMPY INFO")
    
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        print(f"NumPy location: {np.__file__}")
    except ImportError as e:
        print(f"NumPy not installed: {e}")

def get_system_info():
    """Get system information."""
    print_section("SYSTEM INFO")
    
    # OS info
    print(f"OS: {platform.system()}")
    print(f"OS version: {platform.release()}")
    print(f"OS version details: {platform.version()}")
    
    # CPU info
    print(f"CPU count: {os.cpu_count()}")
    
    # Memory info (if available)
    try:
        import psutil
        print(f"Total RAM: {psutil.virtual_memory().total / 1024**3:.2f} GB")
        print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.2f} GB")
    except ImportError:
        print("psutil not available for memory info")

def get_environment_info():
    """Get environment and package information."""
    print_section("ENVIRONMENT INFO")
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
    except ImportError:
        print("❌ Not running in Google Colab")
    
    # Check for common ML libraries
    libraries = [
        'torch', 'torchvision', 'torchaudio', 'numpy', 'scipy', 
        'matplotlib', 'seaborn', 'pandas', 'sklearn', 'tqdm',
        'pythae', 'transformers', 'datasets'
    ]
    
    print("\nInstalled libraries:")
    for lib in libraries:
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'Unknown')
            print(f"  {lib}: {version}")
        except ImportError:
            print(f"  {lib}: Not installed")

def test_rhvae_basic():
    """Test basic RHVAE functionality."""
    print_section("RHVAE BASIC TEST")
    
    try:
        import torch
        from pythae.models import RHVAE, RHVAEConfig
        from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        # Create minimal config
        config = RHVAEConfig(
            input_dim=(1, 28, 28),
            latent_dim=4,
            n_lf=1,
            eps_lf=0.0001,
            beta_zero=1.0,
            temperature=0.5,
            regularization=0.1
        )
        
        print("✅ Config created")
        
        # Create model
        model = RHVAE(
            model_config=config,
            encoder=Encoder_ResNet_VAE_MNIST(config),
            decoder=Decoder_ResNet_AE_MNIST(config)
        )
        
        print("✅ Model created")
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        print(f"✅ Model moved to {device}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 1, 28, 28).to(device)
        inputs = {"data": dummy_input}
        
        model.eval()
        with torch.no_grad():
            output = model(inputs)
            print("✅ Forward pass successful")
            print(f"Output keys: {list(output.keys())}")
        
        print("✅ Basic RHVAE test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic RHVAE test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_local():
    """Provide comparison with local environment."""
    print_section("COMPARISON WITH LOCAL ENVIRONMENT")
    
    print("To compare with your local environment, run this script locally:")
    print("python inspect_colab_versions.py")
    print("\nKey things to check:")
    print("1. PyTorch version differences")
    print("2. CUDA version differences")
    print("3. Pythae version differences")
    print("4. Python version differences")
    print("5. System architecture differences")

def main():
    """Run all inspections."""
    print("COLAB VERSION INSPECTION TOOL")
    print("Run this in Colab to compare with your local environment")
    
    get_python_info()
    get_pytorch_info()
    get_pythae_info()
    get_numpy_info()
    get_system_info()
    get_environment_info()
    test_rhvae_basic()
    compare_with_local()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Copy the output above and compare it with your local environment.")
    print("Look for differences in:")
    print("- PyTorch version")
    print("- CUDA version")
    print("- Python version")
    print("- System architecture")
    print("- Pythae installation")

if __name__ == "__main__":
    main() 