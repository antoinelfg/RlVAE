#!/usr/bin/env python3
"""
Minimal test to isolate RHVAE issues - FIXED VERSION.
"""

import torch
import numpy as np

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def test_minimal_rhvae():
    """Test RHVAE with minimal setup."""
    print("Testing minimal RHVAE setup...")
    
    try:
        from pythae.models import RHVAE, RHVAEConfig
        from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST
        
        print("✅ Imports successful")
        
        # Create minimal config
        config = RHVAEConfig(
            input_dim=(1, 28, 28),
            latent_dim=4,  # Very small
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
        
        # Create minimal input - FIXED: Pass as dictionary
        dummy_input = torch.randn(2, 1, 28, 28).to(device)
        inputs = {"data": dummy_input}  # This is the correct format
        print(f"✅ Input created: {dummy_input.shape}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(inputs)
            print("✅ Forward pass successful")
            print(f"Output keys: {list(output.keys())}")
            
        # Test training mode
        model.train()
        output = model(inputs)
        print("✅ Training forward pass successful")
        print(f"Loss: {output.loss.item()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_original_parameters():
    """Test with the original parameters from the notebook."""
    print("\nTesting with original notebook parameters...")
    
    try:
        from pythae.models import RHVAE, RHVAEConfig
        from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST
        
        # Original parameters from notebook
        config = RHVAEConfig(
            input_dim=(1, 28, 28),
            latent_dim=16,
            n_lf=1,
            eps_lf=0.001,
            beta_zero=0.3,
            temperature=1.5,
            regularization=0.001
        )
        
        print("✅ Original config created")
        
        model = RHVAE(
            model_config=config,
            encoder=Encoder_ResNet_VAE_MNIST(config),
            decoder=Decoder_ResNet_AE_MNIST(config)
        )
        
        print("✅ Original model created")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Test with small batch - FIXED: Pass as dictionary
        dummy_input = torch.randn(4, 1, 28, 28).to(device)
        inputs = {"data": dummy_input}
        
        model.train()
        output = model(inputs)
        print("✅ Original parameters forward pass successful")
        print(f"Loss: {output.loss.item()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Original parameters failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_pipeline():
    """Test with the training pipeline format."""
    print("\nTesting with training pipeline format...")
    
    try:
        from pythae.models import RHVAE, RHVAEConfig
        from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST
        from pythae.trainers import BaseTrainerConfig
        from pythae.pipelines.training import TrainingPipeline
        
        # Stable parameters
        config = BaseTrainerConfig(
            output_dir='test_rhvae',
            learning_rate=5e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_epochs=1,
        )

        model_config = RHVAEConfig(
            input_dim=(1, 28, 28),
            latent_dim=4,
            n_lf=1,
            eps_lf=0.0001,
            beta_zero=1.0,
            temperature=0.5,
            regularization=0.1
        )

        model = RHVAE(
            model_config=model_config,
            encoder=Encoder_ResNet_VAE_MNIST(model_config),
            decoder=Decoder_ResNet_AE_MNIST(model_config)
        )

        pipeline = TrainingPipeline(
            training_config=config,
            model=model
        )
        
        # Create small datasets
        train_data = torch.randn(20, 1, 28, 28)
        eval_data = torch.randn(10, 1, 28, 28)
        
        print("✅ Pipeline created")
        print("✅ Test data created")
        
        # This should work with the pipeline
        pipeline(
            train_data=train_data,
            eval_data=eval_data
        )
        
        print("✅ Pipeline training successful")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Minimal RHVAE Test - FIXED VERSION")
    print("=" * 50)
    
    test_minimal_rhvae()
    test_original_parameters()
    test_training_pipeline()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print("- The issue was input format: RHVAE expects {'data': tensor}")
    print("- This explains why it works in Colab but not locally")
    print("- The training pipeline handles this format conversion automatically") 