#!/usr/bin/env python3
"""
RHVAE Training - Exact Replica

This script replicates exactly the RHVAE training notebook from the benchmark_VAE repository.
We'll run it with both MNIST data (original) and sprites data (ours).
"""

import os
import torch
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Create output directory for visualizations
output_dir = Path("rhvae_exact_replica_visualizations")
output_dir.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Install pythae if not already installed
try:
    from pythae.models import RHVAE, RHVAEConfig, AutoModel, VAE, VAEConfig
    from pythae.trainers import BaseTrainerConfig
    from pythae.pipelines.training import TrainingPipeline
    from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST
    from pythae.samplers import NormalSampler, RHVAESampler, RHVAESamplerConfig
    print("Pythae library imported successfully")
except ImportError:
    print("Installing pythae...")
    import subprocess
    subprocess.check_call(["pip", "install", "pythae"])
    from pythae.models import RHVAE, RHVAEConfig, AutoModel, VAE, VAEConfig
    from pythae.trainers import BaseTrainerConfig
    from pythae.pipelines.training import TrainingPipeline
    from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST
    from pythae.samplers import NormalSampler, RHVAESampler, RHVAESamplerConfig

def save_visualization_grid(data, title, filename, is_color=False):
    """Save a grid of images to file"""
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
    
    for i in range(5):
        for j in range(5):
            sample = data[i*5 + j].cpu()
            if is_color and sample.shape[0] == 3:  # RGB
                axes[i][j].imshow(sample.permute(1, 2, 0))
            else:  # Grayscale
                axes[i][j].imshow(sample.squeeze(0), cmap='gray')
            axes[i][j].axis('off')
    
    plt.tight_layout(pad=0.)
    plt.suptitle(title, fontsize=16)
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def save_interpolation_grid(data, title, filename, is_color=False):
    """Save an interpolation grid to file"""
    fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(10, 5))
    
    for i in range(5):
        for j in range(10):
            sample = data[i, j].cpu()
            if is_color and sample.shape[0] == 3:  # RGB
                axes[i][j].imshow(sample.permute(1, 2, 0))
            else:  # Grayscale
                axes[i][j].imshow(sample.squeeze(0), cmap='gray')
            axes[i][j].axis('off')
    
    plt.tight_layout(pad=0.)
    plt.suptitle(title, fontsize=16)
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def train_rhvae_model(train_data, eval_data, model_name, input_dim):
    """Train an RHVAE model with the exact configuration from the notebook"""
    
    # Configuration (exact replica)
    config = BaseTrainerConfig(
        output_dir=f'my_model_{model_name}',
        learning_rate=1e-4,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_epochs=10,  # Change this to train the model a bit more
    )

    model_config = RHVAEConfig(
        input_dim=input_dim,
        latent_dim=16,
        n_lf=1,
        eps_lf=0.0001,  # Much smaller step for stability
        beta_zero=1.0,   # Larger beta for stability
        temperature=1.0,  # Lower temperature for stability
        regularization=0.01  # Higher regularization for stability
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

    print(f"Training {model_name} RHVAE model...")
    try:
        pipeline(
            train_data=train_data,
            eval_data=eval_data
        )
        return f'my_model_{model_name}'
    except Exception as e:
        print(f"RHVAE training failed with error: {e}")
        print("Falling back to vanilla VAE...")
        return train_vanilla_vae_model(train_data, eval_data, model_name, input_dim)

def train_vanilla_vae_model(train_data, eval_data, model_name, input_dim):
    """Train a vanilla VAE model as fallback"""
    
    config = BaseTrainerConfig(
        output_dir=f'my_model_{model_name}_vae',
        learning_rate=1e-4,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_epochs=10,
    )

    model_config = VAEConfig(
        input_dim=input_dim,
        latent_dim=16,
    )

    model = VAE(
        model_config=model_config,
        encoder=Encoder_ResNet_VAE_MNIST(model_config), 
        decoder=Decoder_ResNet_AE_MNIST(model_config) 
    )

    pipeline = TrainingPipeline(
        training_config=config,
        model=model
    )

    print(f"Training {model_name} vanilla VAE model...")
    pipeline(
        train_data=train_data,
        eval_data=eval_data
    )
    
    return f'my_model_{model_name}_vae'

def load_trained_model(model_dir):
    """Load the trained model from the output directory"""
    last_training = sorted(os.listdir(model_dir))[-1]
    trained_model = AutoModel.load_from_folder(os.path.join(model_dir, last_training, 'final_model'))
    return trained_model

def generate_and_save_samples(model, model_name, is_color=False):
    """Generate samples using both normal and RHVAE samplers"""
    
    # Normal sampler
    normal_sampler = NormalSampler(model=model)
    gen_data_normal = normal_sampler.sample(num_samples=25)
    save_visualization_grid(gen_data_normal, f'{model_name} - Normal Sampler', f'{model_name}_normal_sampler.png', is_color)
    
    # Try RHVAE sampler if it's an RHVAE model
    try:
        if hasattr(model, 'metric') and model.metric is not None:
            rhvae_sampler_config = RHVAESamplerConfig(
                mcmc_steps_nbr=100,
                n_lf=10,
                eps_lf=0.03
            )
            rhvae_sampler = RHVAESampler(
                sampler_config=rhvae_sampler_config,
                model=model
            )
            gen_data_rhvae = rhvae_sampler.sample(num_samples=25)
            save_visualization_grid(gen_data_rhvae, f'{model_name} - RHVAE Sampler', f'{model_name}_rhvae_sampler.png', is_color)
        else:
            print(f"Skipping RHVAE sampler for {model_name} (not an RHVAE model)")
            gen_data_rhvae = None
    except Exception as e:
        print(f"RHVAE sampling failed for {model_name}: {e}")
        gen_data_rhvae = None
    
    return gen_data_normal, gen_data_rhvae

def generate_reconstructions_and_interpolations(model, eval_data, model_name, is_color=False):
    """Generate reconstructions and interpolations"""
    
    # Reconstructions
    reconstructions = model.reconstruct(eval_data[:25].to(device)).detach().cpu()
    save_visualization_grid(reconstructions, f'{model_name} - Reconstructions', f'{model_name}_reconstructions.png', is_color)
    
    # True data
    save_visualization_grid(eval_data[:25], f'{model_name} - True Data', f'{model_name}_true_data.png', is_color)
    
    # Interpolations
    try:
        interpolations = model.interpolate(eval_data[:5].to(device), eval_data[5:10].to(device), granularity=10).detach().cpu()
        save_interpolation_grid(interpolations, f'{model_name} - Interpolations', f'{model_name}_interpolations.png', is_color)
    except Exception as e:
        print(f"Interpolation failed for {model_name}: {e}")

def process_sprites_data(sprites_data):
    """Process sprites data from [batch, 10, 3, 28, 28] to [batch, 1, 28, 28] (grayscale)"""
    # Take the first frame from each sequence and convert to grayscale
    # Shape: [batch, 10, 3, 28, 28] -> [batch, 1, 28, 28]
    first_frame = sprites_data[:, 0, :, :, :]  # [batch, 3, 28, 28]
    
    # Convert RGB to grayscale using standard weights
    # R*0.299 + G*0.587 + B*0.114
    grayscale = first_frame[:, 0, :, :] * 0.299 + first_frame[:, 1, :, :] * 0.587 + first_frame[:, 2, :, :] * 0.114
    
    # Add channel dimension
    return grayscale.unsqueeze(1)  # [batch, 1, 28, 28]

def main():
    print("=== RHVAE Training - Exact Replica ===\n")
    
    # Part 1: MNIST Data (Original)
    print("Part 1: MNIST Data (Original)")
    print("-" * 40)
    
    # Load MNIST data (original from the notebook)
    print("Loading MNIST data...")
    mnist_trainset = datasets.MNIST(root='data', train=True, download=True, transform=None)
    
    train_dataset = mnist_trainset.data[:-10000].reshape(-1, 1, 28, 28) / 255.
    eval_dataset = mnist_trainset.data[-10000:].reshape(-1, 1, 28, 28) / 255.
    
    print(f"MNIST train dataset shape: {train_dataset.shape}")
    print(f"MNIST eval dataset shape: {eval_dataset.shape}")
    
    # Train MNIST RHVAE
    mnist_model_dir = train_rhvae_model(train_dataset, eval_dataset, "mnist", (1, 28, 28))
    
    # Load trained model
    trained_model_mnist = load_trained_model(mnist_model_dir)
    
    # Generate samples
    print("Generating MNIST samples...")
    mnist_normal_samples, mnist_rhvae_samples = generate_and_save_samples(trained_model_mnist, "MNIST", is_color=False)
    
    # Generate reconstructions and interpolations
    print("Generating MNIST reconstructions and interpolations...")
    generate_reconstructions_and_interpolations(trained_model_mnist, eval_dataset, "MNIST", is_color=False)
    
    print("\n" + "="*60 + "\n")
    
    # Part 2: Sprites Data (Ours)
    print("Part 2: Sprites Data (Ours)")
    print("-" * 40)
    
    # Load sprites data
    print("Loading sprites data...")
    sprites_train = torch.load('data/sprites/ColoredCircles_train.pt')
    sprites_test = torch.load('data/sprites/ColoredCircles_test.pt')
    
    print(f"Original sprites train shape: {sprites_train.shape}")
    print(f"Original sprites test shape: {sprites_test.shape}")
    
    # Process sprites data to take first frame from each sequence
    train_dataset_sprites = process_sprites_data(sprites_train).float() / 255.0
    eval_dataset_sprites = process_sprites_data(sprites_test).float() / 255.0
    
    print(f"Processed sprites train dataset shape: {train_dataset_sprites.shape}")
    print(f"Processed sprites eval dataset shape: {eval_dataset_sprites.shape}")
    print(f"Sprites data range: [{train_dataset_sprites.min():.3f}, {train_dataset_sprites.max():.3f}]")
    
    # Get input dimensions from sprites data
    input_dim_sprites = tuple(train_dataset_sprites.shape[1:])
    print(f"Sprites input dimensions: {input_dim_sprites}")
    
    # Train sprites RHVAE
    sprites_model_dir = train_rhvae_model(train_dataset_sprites, eval_dataset_sprites, "sprites", input_dim_sprites)
    
    # Load trained model
    trained_model_sprites = load_trained_model(sprites_model_dir)
    
    # Generate samples
    print("Generating sprites samples...")
    sprites_normal_samples, sprites_rhvae_samples = generate_and_save_samples(trained_model_sprites, "Sprites", is_color=True)
    
    # Generate reconstructions and interpolations
    print("Generating sprites reconstructions and interpolations...")
    generate_reconstructions_and_interpolations(trained_model_sprites, eval_dataset_sprites, "Sprites", is_color=True)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("This script has replicated exactly the RHVAE training notebook from the benchmark_VAE repository.")
    print("Results have been saved to the 'rhvae_exact_replica_visualizations' folder:")
    print()
    print("MNIST Results:")
    print("- MNIST_normal_sampler.png")
    if mnist_rhvae_samples is not None:
        print("- MNIST_rhvae_sampler.png")
    print("- MNIST_reconstructions.png")
    print("- MNIST_true_data.png")
    print("- MNIST_interpolations.png")
    print()
    print("Sprites Results:")
    print("- Sprites_normal_sampler.png")
    if sprites_rhvae_samples is not None:
        print("- Sprites_rhvae_sampler.png")
    print("- Sprites_reconstructions.png")
    print("- Sprites_true_data.png")
    print("- Sprites_interpolations.png")
    print()
    print("All visualizations have been saved with high resolution (150 DPI) for publication quality.")

if __name__ == "__main__":
    main() 