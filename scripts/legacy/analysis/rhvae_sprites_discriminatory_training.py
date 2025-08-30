#!/usr/bin/env python3
"""
RHVAE Sprites Discriminatory Training
=====================================

Train RHVAE with hyperparameters that make the metric much more discriminatory
towards points not being at the center.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pythae.models import RHVAE, RHVAEConfig
from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines import TrainingPipeline
from pythae.models.nn.default_architectures import Metric_MLP
from pythae.models.base.base_utils import ModelOutput
import numpy as np
import os

def load_sprites_data():
    """Load and process sprites data."""
    print("📂 Loading sprites data...")
    sprites_data = torch.load('/home/alaforgu/scratch/longitudinal_experiments/RlVAE/data/processed/Sprites_train_cyclic.pt')
    
    # Process data: take first frame from each sequence and convert to grayscale
    first_frame = sprites_data[:, 0, :, :, :]  # [batch, 3, 64, 64]
    
    # Convert RGB to grayscale
    grayscale = first_frame[:, 0, :, :] * 0.299 + first_frame[:, 1, :, :] * 0.587 + first_frame[:, 2, :, :] * 0.114
    grayscale = grayscale.unsqueeze(1)  # [batch, 1, 64, 64]
    
    # Resize to 28x28 to match MNIST encoder
    processed_data = F.interpolate(grayscale, size=(28, 28), mode='bilinear', align_corners=False)
    
    print(f"📊 Processed data shape: {processed_data.shape}")
    return processed_data

def create_discriminatory_config():
    """Create RHVAE config with discriminatory hyperparameters."""
    
    # More discriminatory hyperparameters
    config = RHVAEConfig(
        input_dim=(1, 28, 28),
        latent_dim=8,
        n_lf=1,  # Fewer leapfrog steps for more direct metric learning
        eps_lf=0.0001,  # Smaller step size for precision
        beta_zero=0.1,  # Lower beta for more aggressive metric learning
        temperature=0.01,  # MUCH LOWER temperature for sharp discrimination
        regularization=0.001,  # Lower regularization to allow more variation
        # per_device_train_batch_size and num_epochs set in trainer config
        uses_default_metric=True,
    )
    
    print("🔧 Discriminatory RHVAE Configuration:")
    print(f"   - Temperature: {config.temperature} (MUCH LOWER for sharp discrimination)")
    print(f"   - Regularization: {config.regularization} (lower for more variation)")
    print(f"   - Beta zero: {config.beta_zero} (lower for aggressive learning)")
    print(f"   - Eps LF: {config.eps_lf} (smaller for precision)")
    print(f"   - N LF: {config.n_lf} (fewer steps)")
    print(f"   - Learning rate: 1e-4 (lower for stability)")
    
    return config

def create_discriminatory_metric():
    """Create a more discriminatory metric network."""
    
    class DiscriminatoryMetric_MLP(Metric_MLP):
        def __init__(self, args):
            super().__init__(args)
            
            # Increase network capacity for more discriminatory learning
            self.layers = nn.Sequential(
                nn.Linear(np.prod(args.input_dim), 800),  # Larger hidden layer
                nn.ReLU(),
                nn.Dropout(0.1),  # Add dropout for regularization
                nn.Linear(800, 400),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            # More expressive diagonal and lower triangular outputs
            self.diag = nn.Sequential(
                nn.Linear(400, 200),
                nn.ReLU(),
                nn.Linear(200, self.latent_dim)
            )
            
            k = int(self.latent_dim * (self.latent_dim - 1) / 2)
            self.lower = nn.Sequential(
                nn.Linear(400, 200),
                nn.ReLU(),
                nn.Linear(200, k)
            )
        
        def forward(self, x):
            h1 = self.layers(x.reshape(-1, np.prod(self.input_dim)))
            h21, h22 = self.diag(h1), self.lower(h1)
            
            L = torch.zeros((x.shape[0], self.latent_dim, self.latent_dim)).to(x.device)
            indices = torch.tril_indices(
                row=self.latent_dim, col=self.latent_dim, offset=-1
            )
            
            # Get non-diagonal coefficients with more aggressive scaling
            L[:, indices[0], indices[1]] = h22 * 2.0  # Scale up off-diagonal elements
            
            # Add diagonal coefficients with more aggressive positivity
            L = L + torch.diag_embed(torch.exp(h21) + 0.1)  # Ensure minimum positive values
            
            return ModelOutput(L=L)
    
    return DiscriminatoryMetric_MLP

def train_discriminatory_rhvae():
    """Train RHVAE with discriminatory hyperparameters."""
    
    print("🎯 RHVAE Sprites Discriminatory Training")
    print("=" * 50)
    
    # Load data
    data = load_sprites_data()
    
    # Create discriminatory config
    config = create_discriminatory_config()
    
    # Create encoder and decoder
    encoder = Encoder_ResNet_VAE_MNIST(config)
    decoder = Decoder_ResNet_AE_MNIST(config)
    
    # Create discriminatory metric
    DiscriminatoryMetric = create_discriminatory_metric()
    metric = DiscriminatoryMetric(config)
    
    # Create RHVAE model
    model = RHVAE(
        model_config=config,
        encoder=encoder,
        decoder=decoder,
        metric=metric
    )
    
    # Create trainer config
    trainer_config = BaseTrainerConfig(
        output_dir="rhvae_sprites_discriminatory",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_epochs=50,
        learning_rate=1e-4,
        steps_saving=10,
        save_best_after_epoch=True,
        no_cuda=False,
        show_progress_bar=True,
        checkpoint_dir="rhvae_sprites_discriminatory/checkpoint_dir",
        save_every_n_epochs=10,
    )
    
    # Create training pipeline
    pipeline = TrainingPipeline(
        training_config=trainer_config,
        model=model
    )
    
    # Train the model
    print("🚀 Starting discriminatory training...")
    pipeline(
        train_data=data,
        eval_data=data
    )
    
    print("✅ Discriminatory training completed!")
    print("📁 Model saved to: rhvae_sprites_discriminatory/")

if __name__ == "__main__":
    train_discriminatory_rhvae() 