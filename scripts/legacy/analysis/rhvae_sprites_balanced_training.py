#!/usr/bin/env python3
"""
RHVAE Balanced Training on Sprites Data
=======================================

This script trains RHVAE with balanced hyperparameters that should be stable
while still allowing the metric to learn the data geometry.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pythae.models import RHVAE, RHVAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST
import warnings
warnings.filterwarnings("ignore")

# Create output directory
output_dir = Path("rhvae_sprites_balanced")
output_dir.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def load_sprites_data():
    """Load sprites data for training."""
    print("📂 Loading sprites data...")
    sprites_data = torch.load('/home/alaforgu/scratch/longitudinal_experiments/RlVAE/data/processed/Sprites_train_cyclic.pt')
    
    print(f"📊 Original sprites data shape: {sprites_data.shape}")
    
    # Process data: take first frame from each sequence and convert to grayscale
    # Shape: [batch, 8, 3, 64, 64] -> [batch, 1, 28, 28]
    first_frame = sprites_data[:, 0, :, :, :]  # [batch, 3, 64, 64]
    
    # Convert RGB to grayscale
    grayscale = first_frame[:, 0, :, :] * 0.299 + first_frame[:, 1, :, :] * 0.587 + first_frame[:, 2, :, :] * 0.114
    grayscale = grayscale.unsqueeze(1)  # [batch, 1, 64, 64]
    
    # Resize to 28x28 to match MNIST encoder
    import torch.nn.functional as F
    processed_data = F.interpolate(grayscale, size=(28, 28), mode='bilinear', align_corners=False)
    
    # Split into train/eval
    train_size = int(0.8 * len(processed_data))
    train_data = processed_data[:train_size]
    eval_data = processed_data[train_size:]
    
    print(f"📊 Processed data: train={train_data.shape}, eval={eval_data.shape}")
    print(f"📊 Data range: [{train_data.min():.3f}, {train_data.max():.3f}]")
    
    return train_data, eval_data

def train_balanced_rhvae():
    """Train RHVAE with balanced hyperparameters."""
    print("🚀 Training RHVAE with balanced hyperparameters...")
    
    # Load data
    train_data, eval_data = load_sprites_data()
    
    # BALANCED Configuration for stable metric learning
    config = BaseTrainerConfig(
        output_dir='rhvae_sprites_balanced_model',
        learning_rate=5e-4,  # Moderate learning rate
        per_device_train_batch_size=64,  # Larger batch for stability
        per_device_eval_batch_size=64,
        num_epochs=30,  # Reasonable number of epochs
    )

    # BALANCED RHVAE hyperparameters
    model_config = RHVAEConfig(
        input_dim=(1, 28, 28),  # Resized sprites data to 28x28
        latent_dim=12,  # Moderate latent dim
        n_lf=2,  # Moderate leapfrog steps
        eps_lf=0.001,  # Moderate step size
        beta_zero=0.3,  # Balanced beta
        temperature=0.8,  # Moderate temperature
        regularization=0.001  # Moderate regularization
    )

    print(f"🔧 Model config: {model_config}")
    print(f"🔧 Training config: {config}")

    model = RHVAE(
        model_config=model_config,
        encoder=Encoder_ResNet_VAE_MNIST(model_config), 
        decoder=Decoder_ResNet_AE_MNIST(model_config) 
    )

    pipeline = TrainingPipeline(
        training_config=config,
        model=model
    )

    print("🚀 Starting balanced RHVAE training...")
    pipeline(
        train_data=train_data,
        eval_data=eval_data
    )
    
    return 'rhvae_sprites_balanced_model'

def create_metric_analysis_script():
    """Create a script to analyze the trained model's metric."""
    analysis_script = '''#!/usr/bin/env python3
"""
RHVAE Sprites Metric Analysis
=============================

Analyze the trained RHVAE model's metric tensor and geometry.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pythae.models import AutoModel
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_trained_model():
    """Load the trained RHVAE model."""
    model_dir = "rhvae_sprites_balanced_model/RHVAE_training_*/final_model"
    import glob
    model_paths = glob.glob(model_dir)
    if not model_paths:
        raise FileNotFoundError("No trained model found!")
    
    latest_model = sorted(model_paths)[-1]
    trained_model = AutoModel.load_from_folder(latest_model)
    trained_model = trained_model.to(device)
    trained_model.eval()
    print(f"✅ Loaded trained model from {latest_model}")
    return trained_model

def analyze_metric_tensor(model, n_samples=200):
    """Analyze the metric tensor across the latent space."""
    print("🔍 Analyzing metric tensor...")
    
    # Sample random latent points
    z_samples = torch.randn(n_samples, model.latent_dim).to(device)
    
    # Compute metric tensors
    metric_determinants = []
    metric_eigenvalues = []
    metric_traces = []
    
    with torch.no_grad():
        for i, z in enumerate(z_samples):
            try:
                G_inv = model.G_inv(z.unsqueeze(0))
                G = torch.inverse(G_inv)
                
                det_G = torch.det(G)
                trace_G = torch.trace(G)
                
                metric_determinants.append(det_G.item())
                metric_traces.append(trace_G.item())
                
                eigenvals = torch.linalg.eigvals(G)
                metric_eigenvalues.append(eigenvals.cpu())
                
                if i % 50 == 0:
                    print(f"  Processed {i}/{n_samples} points...")
                
            except Exception as e:
                print(f"⚠️ Failed to compute metric for point {i}: {e}")
                continue
    
    # Create comprehensive analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Metric determinant distribution
    det_values = np.array(metric_determinants)
    axes[0, 0].hist(det_values, bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('det(G)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Metric Determinant Distribution')
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Log metric determinant
    log_det_values = np.log(det_values + 1e-8)
    axes[0, 1].hist(log_det_values, bins=30, alpha=0.7, color='green')
    axes[0, 1].set_xlabel('log(det(G))')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Log Metric Determinant Distribution')
    
    # Plot 3: Metric trace distribution
    trace_values = np.array(metric_traces)
    axes[0, 2].hist(trace_values, bins=30, alpha=0.7, color='red')
    axes[0, 2].set_xlabel('trace(G)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Metric Trace Distribution')
    
    # Plot 4: Eigenvalue distribution
    all_eigenvals = torch.cat(metric_eigenvalues, dim=0).numpy()
    axes[1, 0].hist(all_eigenvals.real, bins=40, alpha=0.7, color='purple')
    axes[1, 0].set_xlabel('Eigenvalue (Real Part)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Metric Eigenvalue Distribution')
    
    # Plot 5: Metric determinant vs latent dimension 1
    latent_dim1 = z_samples[:len(det_values), 0].cpu().numpy()
    scatter = axes[1, 1].scatter(latent_dim1, det_values, alpha=0.6, c=det_values, cmap='viridis')
    axes[1, 1].set_xlabel('Latent Dimension 1')
    axes[1, 1].set_ylabel('det(G)')
    axes[1, 1].set_title('Metric Determinant vs Latent Dim 1')
    plt.colorbar(scatter, ax=axes[1, 1])
    
    # Plot 6: Metric determinant vs latent dimension 2
    latent_dim2 = z_samples[:len(det_values), 1].cpu().numpy()
    scatter = axes[1, 2].scatter(latent_dim2, det_values, alpha=0.6, c=det_values, cmap='viridis')
    axes[1, 2].set_xlabel('Latent Dimension 2')
    axes[1, 2].set_ylabel('det(G)')
    axes[1, 2].set_title('Metric Determinant vs Latent Dim 2')
    plt.colorbar(scatter, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('rhvae_sprites_metric_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Metric analysis saved to rhvae_sprites_metric_analysis.png")
    print(f"📊 Metric determinant stats:")
    print(f"   - min: {det_values.min():.6f}")
    print(f"   - max: {det_values.max():.6f}")
    print(f"   - mean: {det_values.mean():.6f}")
    print(f"   - std: {det_values.std():.6f}")
    print(f"   - coefficient of variation: {det_values.std()/det_values.mean():.3f}")
    
    # Check if metric is varying
    if det_values.std() / det_values.mean() > 0.1:
        print("✅ Good! Metric determinant is varying significantly across latent space.")
    else:
        print("⚠️ Metric determinant is too uniform. Consider more aggressive training.")

def main():
    """Main analysis function."""
    print("🔍 RHVAE Sprites Metric Analysis")
    print("=" * 40)
    
    model = load_trained_model()
    analyze_metric_tensor(model)

if __name__ == "__main__":
    main()
'''
    
    with open('analyze_rhvae_sprites_metric.py', 'w') as f:
        f.write(analysis_script)
    
    print("✅ Created analysis script: analyze_rhvae_sprites_metric.py")

def main():
    """Main training function."""
    print("🚀 RHVAE Balanced Training on Sprites")
    print("=" * 50)
    
    # Train the model
    model_dir = train_balanced_rhvae()
    
    # Create analysis script
    create_metric_analysis_script()
    
    print(f"\n✅ Training completed!")
    print(f"📁 Model saved to: {model_dir}")
    print(f"📊 Analysis script: analyze_rhvae_sprites_metric.py")
    print(f"\n🔍 Next steps:")
    print(f"   1. Run: python analyze_rhvae_sprites_metric.py")
    print(f"   2. Check the metric determinant distribution")
    print(f"   3. If metric is still flat, we can try more aggressive parameters")

if __name__ == "__main__":
    main() 