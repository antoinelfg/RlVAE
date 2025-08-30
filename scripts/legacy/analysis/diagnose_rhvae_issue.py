#!/usr/bin/env python3
"""
Diagnostic script to understand RHVAE compatibility issues between environments.
This helps identify why RHVAE works on Colab but fails locally.
"""

import torch
import numpy as np
import sys
import os

def print_environment_info():
    """Print detailed environment information."""
    print("=" * 60)
    print("ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)
    
    # PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Python info
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    
    # System info
    import platform
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    
    # Memory info
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def test_rhvae_components():
    """Test RHVAE components individually."""
    print("\n" + "=" * 60)
    print("RHVAE COMPONENT TESTING")
    print("=" * 60)
    
    try:
        from pythae.models import RHVAE, RHVAEConfig
        from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST
        print("✅ Successfully imported RHVAE components")
        
        # Test configuration
        config = RHVAEConfig(
            input_dim=(1, 28, 28),
            latent_dim=8,
            n_lf=1,
            eps_lf=0.0001,
            beta_zero=1.0,
            temperature=0.5,
            regularization=0.1
        )
        print("✅ RHVAEConfig created successfully")
        
        # Test model creation
        model = RHVAE(
            model_config=config,
            encoder=Encoder_ResNet_VAE_MNIST(config),
            decoder=Decoder_ResNet_AE_MNIST(config)
        )
        print("✅ RHVAE model created successfully")
        
        # Test forward pass with dummy data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        dummy_input = torch.randn(2, 1, 28, 28).to(device)
        print(f"Testing with dummy input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            try:
                output = model(dummy_input)
                print("✅ Forward pass successful")
                print(f"Output keys: {output.keys()}")
            except Exception as e:
                print(f"❌ Forward pass failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ RHVAE component test failed: {e}")
        return False

def test_numerical_stability():
    """Test numerical stability with different precision settings."""
    print("\n" + "=" * 60)
    print("NUMERICAL STABILITY TESTING")
    print("=" * 60)
    
    # Test different precision settings
    precisions = [torch.float32, torch.float64]
    
    for precision in precisions:
        print(f"\nTesting with {precision}...")
        torch.set_default_dtype(precision)
        
        try:
            from pythae.models import RHVAE, RHVAEConfig
            from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST
            
            config = RHVAEConfig(
                input_dim=(1, 28, 28),
                latent_dim=4,  # Very small for testing
                n_lf=1,
                eps_lf=0.00001,
                beta_zero=2.0,
                temperature=0.1,
                regularization=1.0
            )
            
            model = RHVAE(
                model_config=config,
                encoder=Encoder_ResNet_VAE_MNIST(config),
                decoder=Decoder_ResNet_AE_MNIST(config)
            )
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.train()
            
            dummy_input = torch.randn(4, 1, 28, 28, dtype=precision).to(device)
            
            # Test a few training steps
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            for step in range(3):
                optimizer.zero_grad()
                output = model(dummy_input)
                loss = output.loss
                
                # Check for NaN
                if torch.isnan(loss):
                    print(f"❌ NaN detected in loss at step {step}")
                    break
                
                loss.backward()
                optimizer.step()
                print(f"Step {step}: Loss = {loss.item():.6f}")
            
            print(f"✅ {precision} test completed successfully")
            
        except Exception as e:
            print(f"❌ {precision} test failed: {e}")
    
    # Reset to float32
    torch.set_default_dtype(torch.float32)

def compare_with_colab_parameters():
    """Show the differences between your parameters and typical Colab parameters."""
    print("\n" + "=" * 60)
    print("PARAMETER COMPARISON")
    print("=" * 60)
    
    print("Your current parameters (from notebook):")
    print("  - latent_dim: 16")
    print("  - n_lf: 1")
    print("  - eps_lf: 0.001")
    print("  - beta_zero: 0.3")
    print("  - temperature: 1.5")
    print("  - regularization: 0.001")
    print("  - learning_rate: 1e-4")
    print("  - batch_size: 64")
    
    print("\nRecommended stable parameters:")
    print("  - latent_dim: 8 (smaller)")
    print("  - n_lf: 1")
    print("  - eps_lf: 0.0001 (much smaller)")
    print("  - beta_zero: 1.0 (higher)")
    print("  - temperature: 0.5 (lower)")
    print("  - regularization: 0.1 (higher)")
    print("  - learning_rate: 5e-5 (lower)")
    print("  - batch_size: 32 (smaller)")
    
    print("\nKey differences that might cause issues:")
    print("1. eps_lf too large (0.001 vs 0.0001)")
    print("2. beta_zero too low (0.3 vs 1.0)")
    print("3. temperature too high (1.5 vs 0.5)")
    print("4. regularization too low (0.001 vs 0.1)")
    print("5. learning rate too high (1e-4 vs 5e-5)")

def main():
    """Run all diagnostics."""
    print("RHVAE Environment Diagnostic Tool")
    print("This will help identify why RHVAE works on Colab but fails locally.\n")
    
    print_environment_info()
    test_rhvae_components()
    test_numerical_stability()
    compare_with_colab_parameters()
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("1. Use the stable parameters provided in testrhvae_stable.ipynb")
    print("2. If still failing, try even more conservative parameters")
    print("3. Consider using a vanilla VAE if RHVAE continues to fail")
    print("4. The issue is likely numerical instability, not a fundamental bug")
    print("5. Colab might have different PyTorch/CUDA versions that are more stable")

if __name__ == "__main__":
    main() 