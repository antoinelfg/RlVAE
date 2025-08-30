#!/usr/bin/env python3
"""
RHVAE Sprites Balanced Discriminatory Training
==============================================

Train RHVAE with balanced hyperparameters that maintain smooth geometry
while still being discriminatory towards points not being at the center.
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

def create_balanced_discriminatory_config():
    """Create RHVAE config with balanced discriminatory hyperparameters."""
    
    # Balanced discriminatory hyperparameters - smooth but still discriminatory
    config = RHVAEConfig(
        input_dim=(1, 28, 28),
        latent_dim=8,
        n_lf=2,  # More leapfrog steps for smoother geometry
        eps_lf=0.001,  # Balanced step size
        beta_zero=0.3,  # Balanced beta for smooth learning
        temperature=0.1,  # BALANCED temperature - discriminatory but smooth
        regularization=0.005,  # BALANCED regularization - some smoothing
        # per_device_train_batch_size and num_epochs set in trainer config
        uses_default_metric=True,
    )
    
    print("🔧 Balanced Discriminatory RHVAE Configuration:")
    print(f"   - Temperature: {config.temperature} (BALANCED - discriminatory but smooth)")
    print(f"   - Regularization: {config.regularization} (BALANCED - some smoothing)")
    print(f"   - Beta zero: {config.beta_zero} (BALANCED for smooth learning)")
    print(f"   - Eps LF: {config.eps_lf} (BALANCED for smooth geometry)")
    print(f"   - N LF: {config.n_lf} (MORE steps for smoother geometry)")
    print(f"   - Learning rate: 5e-4 (BALANCED for stability)")
    
    return config

def create_balanced_discriminatory_metric():
    """Create a balanced discriminatory metric network."""
    
    class BalancedDiscriminatoryMetric_MLP(Metric_MLP):
        def __init__(self, args):
            super().__init__(args)
            
            # Balanced network capacity - expressive but not too aggressive
            self.layers = nn.Sequential(
                nn.Linear(np.prod(args.input_dim), 600),  # Balanced hidden layer
                nn.ReLU(),
                nn.Dropout(0.05),  # Light dropout for regularization
                nn.Linear(600, 300),
                nn.ReLU(),
                nn.Dropout(0.05)
            )
            
            # Balanced diagonal and lower triangular outputs
            self.diag = nn.Sequential(
                nn.Linear(300, 150),
                nn.ReLU(),
                nn.Linear(150, self.latent_dim)
            )
            
            k = int(self.latent_dim * (self.latent_dim - 1) / 2)
            self.lower = nn.Sequential(
                nn.Linear(300, 150),
                nn.ReLU(),
                nn.Linear(150, k)
            )
        
        def forward(self, x):
            h1 = self.layers(x.reshape(-1, np.prod(self.input_dim)))
            h21, h22 = self.diag(h1), self.lower(h1)
            
            L = torch.zeros((x.shape[0], self.latent_dim, self.latent_dim)).to(x.device)
            indices = torch.tril_indices(
                row=self.latent_dim, col=self.latent_dim, offset=-1
            )
            
            # Get non-diagonal coefficients with BALANCED scaling
            L[:, indices[0], indices[1]] = h22 * 1.0  # BALANCED scaling
            
            # Add diagonal coefficients with BALANCED positivity
            L = L + torch.diag_embed(torch.exp(h21) + 0.01)  # BALANCED minimum values
            
            return ModelOutput(L=L)
    
    return BalancedDiscriminatoryMetric_MLP

def train_balanced_discriminatory_rhvae():
    """Train RHVAE with balanced discriminatory hyperparameters."""
    
    print("🎯 RHVAE Sprites Balanced Discriminatory Training")
    print("=" * 55)
    
    # Load data
    data = load_sprites_data()
    
    # Create balanced discriminatory config
    config = create_balanced_discriminatory_config()
    
    # Create encoder and decoder
    encoder = Encoder_ResNet_VAE_MNIST(config)
    decoder = Decoder_ResNet_AE_MNIST(config)
    
    # Create balanced discriminatory metric
    BalancedDiscriminatoryMetric = create_balanced_discriminatory_metric()
    metric = BalancedDiscriminatoryMetric(config)
    
    # Create RHVAE model
    model = RHVAE(
        model_config=config,
        encoder=encoder,
        decoder=decoder,
        metric=metric
    )
    
    # Create trainer config
    trainer_config = BaseTrainerConfig(
        output_dir="rhvae_sprites_balanced_discriminatory",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_epochs=40,
        learning_rate=5e-4,
        steps_saving=10,
        save_best_after_epoch=True,
        no_cuda=False,
        show_progress_bar=True,
        checkpoint_dir="rhvae_sprites_balanced_discriminatory/checkpoint_dir",
        save_every_n_epochs=10,
    )
    
    # Create training pipeline
    pipeline = TrainingPipeline(
        training_config=trainer_config,
        model=model
    )
    
    # Train the model
    print("🚀 Starting balanced discriminatory training...")
    pipeline(
        train_data=data,
        eval_data=data
    )
    
    print("✅ Balanced discriminatory training completed!")
    print("📁 Model saved to: rhvae_sprites_balanced_discriminatory/")

if __name__ == "__main__":
    train_balanced_discriminatory_rhvae() 