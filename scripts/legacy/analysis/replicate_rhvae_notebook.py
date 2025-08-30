#!/usr/bin/env python3
"""
Replicate RHVAE Training Notebook
================================

Exact replication of the official RHVAE training notebook with all graphs, 
sampling, and visualizations.
"""

import torch
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pythae.models import RHVAE, RHVAEConfig, AutoModel
from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
from pythae.samplers import NormalSampler, RHVAESampler, RHVAESamplerConfig

def replicate_rhvae_notebook():
    """Replicate the exact RHVAE training notebook."""
    print("🎯 REPLICATING RHVAE TRAINING NOTEBOOK")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cell 1: Load MNIST data (exact same as notebook)
    print("📂 Loading MNIST data...")
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    
    train_dataset = mnist_trainset.data[:-10000].reshape(-1, 1, 28, 28) / 255.
    eval_dataset = mnist_trainset.data[-10000:].reshape(-1, 1, 28, 28) / 255.
    
    print(f"   Train dataset shape: {train_dataset.shape}")
    print(f"   Eval dataset shape: {eval_dataset.shape}")
    
    # Cell 2: Create configuration (exact same as notebook)
    print("\n🔧 Creating RHVAE configuration...")
    config = BaseTrainerConfig(
        output_dir='my_model',
        learning_rate=1e-4,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_epochs=10,  # Change this to train the model a bit more
    )
    
    model_config = RHVAEConfig(
        input_dim=(1, 28, 28),
        latent_dim=16,
        n_lf=1,
        eps_lf=0.001,
        beta_zero=0.3,
        temperature=1.5,
        regularization=0.001
    )
    
    # Cell 3: Create model (exact same as notebook)
    model = RHVAE(
        model_config=model_config,
        encoder=Encoder_ResNet_VAE_MNIST(model_config), 
        decoder=Decoder_ResNet_AE_MNIST(model_config) 
    )
    
    print(f"✅ Model created")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Cell 4: Create pipeline (exact same as notebook)
    print("\n🚀 Creating training pipeline...")
    pipeline = TrainingPipeline(
        training_config=config,
        model=model
    )
    
    # Cell 5: Train the model (exact same as notebook)
    print("\n🎓 Training model...")
    pipeline(
        train_data=train_dataset,
        eval_data=eval_dataset
    )
    
    # Cell 6: Load trained model (exact same as notebook)
    print("\n📂 Loading trained model...")
    last_training = sorted(os.listdir('my_model'))[-1]
    trained_model = AutoModel.load_from_folder(os.path.join('my_model', last_training, 'final_model'))
    trained_model.to(device)
    
    print(f"✅ Trained model loaded")
    
    # Create output directory for visualizations
    output_dir = "rhvae_notebook_replication"
    os.makedirs(output_dir, exist_ok=True)
    
    # Cell 7-8: Normal sampling (exact same as notebook)
    print("\n🎲 Testing Normal Sampler...")
    normal_sampler = NormalSampler(model=trained_model)
    
    # Sample
    gen_data_normal = normal_sampler.sample(num_samples=25)
    
    # Show results with normal sampler (Cell 9)
    print("   Creating normal sampler visualization...")
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
    
    for i in range(5):
        for j in range(5):
            axes[i][j].imshow(gen_data_normal[i*5 + j].cpu().squeeze(0), cmap='gray')
            axes[i][j].axis('off')
    plt.tight_layout(pad=0.)
    plt.savefig(f'{output_dir}/01_normal_sampler_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Cell 10-11: RHVAE sampling (exact same as notebook)
    print("\n🎲 Testing RHVAE Sampler...")
    rhvae_sampler_config = RHVAESamplerConfig(
        mcmc_steps_nbr=100,
        n_lf=10,
        eps_lf=0.03
    )
    
    # Create RHVAE sampler
    rhvae_sampler = RHVAESampler(
        sampler_config=rhvae_sampler_config,
        model=trained_model
    )
    
    # Sample
    gen_data_rhvae = rhvae_sampler.sample(num_samples=25)
    
    # Show results with RHVAE sampler (Cell 12)
    print("   Creating RHVAE sampler visualization...")
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
    
    for i in range(5):
        for j in range(5):
            axes[i][j].imshow(gen_data_rhvae[i*5 + j].cpu().squeeze(0), cmap='gray')
            axes[i][j].axis('off')
    plt.tight_layout(pad=0.)
    plt.savefig(f'{output_dir}/02_rhvae_sampler_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Cell 13-14: Visualizing reconstructions (exact same as notebook)
    print("\n🔄 Creating reconstructions...")
    reconstructions = trained_model.reconstruct(eval_dataset[:25].to(device)).detach().cpu()
    
    # Show reconstructions (Cell 14)
    print("   Creating reconstruction visualization...")
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
    
    for i in range(5):
        for j in range(5):
            axes[i][j].imshow(reconstructions[i*5 + j].cpu().squeeze(0), cmap='gray')
            axes[i][j].axis('off')
    plt.tight_layout(pad=0.)
    plt.savefig(f'{output_dir}/03_reconstructions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Cell 15: Show the true data (exact same as notebook)
    print("\n📊 Creating true data visualization...")
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
    
    for i in range(5):
        for j in range(5):
            axes[i][j].imshow(eval_dataset[i*5 + j].cpu().squeeze(0), cmap='gray')
            axes[i][j].axis('off')
    plt.tight_layout(pad=0.)
    plt.savefig(f'{output_dir}/04_true_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Cell 16-17: Visualizing interpolations (exact same as notebook)
    print("\n🔄 Creating interpolations...")
    interpolations = trained_model.interpolate(
        eval_dataset[:5].to(device), 
        eval_dataset[5:10].to(device), 
        granularity=10
    ).detach().cpu()
    
    # Show interpolations (Cell 17)
    print("   Creating interpolation visualization...")
    fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(10, 5))
    
    for i in range(5):
        for j in range(10):
            axes[i][j].imshow(interpolations[i, j].cpu().squeeze(0), cmap='gray')
            axes[i][j].axis('off')
    plt.tight_layout(pad=0.)
    plt.savefig(f'{output_dir}/05_interpolations.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Additional analysis: Compare samplers
    print("\n📊 Creating sampler comparison...")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # Row 1: Normal sampler samples
    for j in range(5):
        axes[0, j].imshow(gen_data_normal[j].cpu().squeeze(0), cmap='gray')
        axes[0, j].set_title(f'Normal {j+1}')
        axes[0, j].axis('off')
    
    # Row 2: RHVAE sampler samples
    for j in range(5):
        axes[1, j].imshow(gen_data_rhvae[j].cpu().squeeze(0), cmap='gray')
        axes[1, j].set_title(f'RHVAE {j+1}')
        axes[1, j].axis('off')
    
    plt.suptitle('Normal vs RHVAE Sampler Comparison', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_sampler_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Additional analysis: Reconstruction quality
    print("\n📊 Creating reconstruction quality analysis...")
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    
    # Row 1: Original data
    for j in range(5):
        axes[0, j].imshow(eval_dataset[j].cpu().squeeze(0), cmap='gray')
        axes[0, j].set_title(f'Original {j+1}')
        axes[0, j].axis('off')
    
    # Row 2: Reconstructions
    for j in range(5):
        axes[1, j].imshow(reconstructions[j].cpu().squeeze(0), cmap='gray')
        axes[1, j].set_title(f'Reconstruction {j+1}')
        axes[1, j].axis('off')
    
    # Row 3: Difference
    for j in range(5):
        diff = torch.abs(eval_dataset[j] - reconstructions[j])
        axes[2, j].imshow(diff.cpu().squeeze(0), cmap='hot')
        axes[2, j].set_title(f'Difference {j+1}')
        axes[2, j].axis('off')
    
    plt.suptitle('Reconstruction Quality Analysis', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_reconstruction_quality.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Additional analysis: Metric analysis
    print("\n🔧 Creating metric analysis...")
    
    # Test the trained model's metric
    trained_model.eval()
    test_batch = eval_dataset[:32].to(device)
    
    with torch.no_grad():
        # Get metric information
        metric_output = trained_model.metric(test_batch)
        L = metric_output["L"]
        M = L @ torch.transpose(L, 1, 2)
        
        # Get G_inv for analysis
        model_output = trained_model({"data": test_batch})
        G_inv = model_output.G_inv
        det_G_inv = torch.exp(-model_output.G_log_det)
    
    # Create metric analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: L matrix determinants
    ax1 = axes[0, 0]
    det_L = torch.det(L).cpu().numpy()
    ax1.hist(det_L, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('det(L)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('1. L Matrix Determinants\nTrained Model', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: M matrix determinants
    ax2 = axes[0, 1]
    det_M = torch.det(M).cpu().numpy()
    ax2.hist(det_M, bins=20, alpha=0.7, color='red', edgecolor='black')
    ax2.set_xlabel('det(M)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('2. M Matrix Determinants\nM = L L^T', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: G⁻¹ determinants
    ax3 = axes[0, 2]
    ax3.hist(det_G_inv.cpu().numpy(), bins=20, alpha=0.7, color='green', edgecolor='black')
    ax3.set_xlabel('det(G⁻¹)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('3. G⁻¹ Determinants\nTrained Model', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Sample L matrix
    ax4 = axes[1, 0]
    sample_L = L[0].cpu().numpy()
    im = ax4.imshow(sample_L, cmap='viridis', alpha=0.8)
    ax4.set_title(f'4. Sample L Matrix\ndet = {det_L[0]:.2f}', fontweight='bold')
    plt.colorbar(im, ax=ax4)
    
    # Plot 5: Sample M matrix
    ax5 = axes[1, 1]
    sample_M = M[0].cpu().numpy()
    im = ax5.imshow(sample_M, cmap='viridis', alpha=0.8)
    ax5.set_title(f'5. Sample M Matrix\ndet = {det_M[0]:.2f}', fontweight='bold')
    plt.colorbar(im, ax=ax5)
    
    # Plot 6: Sample G⁻¹ matrix
    ax6 = axes[1, 2]
    sample_G_inv = G_inv[0].cpu().numpy()
    im = ax6.imshow(sample_G_inv, cmap='viridis', alpha=0.8)
    ax6.set_title(f'6. Sample G⁻¹ Matrix\ndet = {det_G_inv[0]:.1e}', fontweight='bold')
    plt.colorbar(im, ax=ax6)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_metric_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ All visualizations saved to {output_dir}/")
    print(f"📊 Generated {len(os.listdir(output_dir))} comprehensive analysis graphs")
    
    print(f"\n📝 SUMMARY:")
    print("=" * 20)
    print(f"✅ RHVAE training completed successfully")
    print(f"✅ Normal sampler: {len(gen_data_normal)} samples")
    print(f"✅ RHVAE sampler: {len(gen_data_rhvae)} samples")
    print(f"✅ Reconstructions: {reconstructions.shape}")
    print(f"✅ Interpolations: {interpolations.shape}")
    print(f"✅ Metric analysis: L det range [{det_L.min():.1e}, {det_L.max():.1e}]")
    print(f"✅ Metric analysis: G⁻¹ det range [{det_G_inv.min():.1e}, {det_G_inv.max():.1e}]")
    
    return output_dir

if __name__ == "__main__":
    output_folder = replicate_rhvae_notebook() 