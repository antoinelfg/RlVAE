#!/usr/bin/env python3
"""
RHMC Samples Visualization Script
=================================

Visualizes RHMC sampling results with scatter plots, distributions, and analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import argparse
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_rhmc_samples(samples_path):
    """Load RHMC samples from file."""
    print(f"Loading RHMC samples from: {samples_path}")
    data = torch.load(samples_path, map_location='cpu')
    
    if 'samples' not in data:
        raise ValueError(f"No 'samples' key found in {samples_path}")
    
    samples = data['samples']
    print(f"RHMC samples shape: {samples.shape}")
    print(f"Sample range: [{samples.min():.4f}, {samples.max():.4f}]")
    print(f"Sample mean: {samples.mean():.4f}")
    print(f"Sample std: {samples.std():.4f}")
    
    return samples

def create_pca_visualization(samples, title_prefix="", log_to_wandb=True):
    """Create PCA visualization of RHMC samples."""
    print("Creating PCA visualization...")
    
    # Apply PCA
    pca = PCA(n_components=2)
    samples_2d = pca.fit_transform(samples.numpy())
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'{title_prefix}RHMC Samples PCA Analysis', fontsize=16, fontweight='bold')
    
    # Scatter plot
    scatter = axes[0].scatter(samples_2d[:, 0], samples_2d[:, 1], 
                             alpha=0.6, s=20, c=np.arange(len(samples_2d)), 
                             cmap='viridis')
    axes[0].set_title('RHMC Samples (PCA)')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='Sample Index')
    
    # Explained variance
    n_components = min(10, samples.shape[1])
    pca_full = PCA(n_components=n_components)
    pca_full.fit(samples.numpy())
    
    axes[1].bar(range(1, n_components + 1), pca_full.explained_variance_ratio_)
    axes[1].set_title('Explained Variance Ratio')
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Explained Variance Ratio')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if log_to_wandb:
        wandb.log({"rhmc_analysis/pca_visualization": wandb.Image(fig)})
    
    return fig

def create_distribution_analysis(samples, title_prefix="", log_to_wandb=True):
    """Create distribution analysis of RHMC samples."""
    print("Creating distribution analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{title_prefix}RHMC Samples Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overall distribution
    axes[0, 0].hist(samples.flatten().numpy(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Overall Sample Distribution')
    axes[0, 0].set_xlabel('Sample Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Per-dimension distributions
    for i in range(min(5, samples.shape[1])):
        axes[0, 1].hist(samples[:, i].numpy(), bins=30, alpha=0.5, label=f'Dim {i+1}')
    axes[0, 1].set_title('Per-Dimension Distributions (First 5)')
    axes[0, 1].set_xlabel('Sample Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Correlation matrix
    corr_matrix = np.corrcoef(samples.T)
    im = axes[1, 0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 0].set_title('Sample Correlation Matrix')
    axes[1, 0].set_xlabel('Dimension')
    axes[1, 0].set_ylabel('Dimension')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. Variance per dimension
    variances = samples.var(dim=0)
    axes[1, 1].bar(range(1, len(variances) + 1), variances.numpy())
    axes[1, 1].set_title('Variance per Dimension')
    axes[1, 1].set_xlabel('Dimension')
    axes[1, 1].set_ylabel('Variance')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if log_to_wandb:
        wandb.log({"rhmc_analysis/distribution_analysis": wandb.Image(fig)})
    
    return fig

def create_tsne_visualization(samples, title_prefix="", log_to_wandb=True):
    """Create t-SNE visualization of RHMC samples."""
    print("Creating t-SNE visualization...")
    
    # Use subset for t-SNE (it's computationally expensive)
    n_samples = min(1000, len(samples))
    subset = samples[:n_samples]
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples//4))
    samples_2d = tsne.fit_transform(subset.numpy())
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(f'{title_prefix}RHMC Samples t-SNE Visualization', fontsize=16, fontweight='bold')
    
    scatter = ax.scatter(samples_2d[:, 0], samples_2d[:, 1], 
                        alpha=0.6, s=30, c=np.arange(len(samples_2d)), 
                        cmap='viridis')
    ax.set_title(f't-SNE (n={n_samples})')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Sample Index')
    
    plt.tight_layout()
    
    if log_to_wandb:
        wandb.log({"rhmc_analysis/tsne_visualization": wandb.Image(fig)})
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Visualize RHMC samples")
    parser.add_argument("samples_path", help="Path to RHMC samples file")
    parser.add_argument("--wandb-project", default="rhmc_analysis", help="Wandb project name")
    parser.add_argument("--wandb-name", help="Wandb run name (default: auto from filename)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Initialize wandb
    if not args.no_wandb:
        run_name = args.wandb_name or Path(args.samples_path).stem
        wandb.init(
            project=args.wandb_project,
            name=f"rhmc_analysis_{run_name}",
            config={
                "samples_path": args.samples_path,
                "analysis_type": "rhmc_samples_visualization"
            }
        )
    
    # Load samples
    samples = load_rhmc_samples(args.samples_path)
    
    print(f"\n{'='*80}")
    print(f"RHMC SAMPLES VISUALIZATION")
    print(f"{'='*80}")
    
    # Create visualizations
    title_prefix = f"Stage B RHVAE MLP 16D - "
    
    # 1. PCA visualization
    print(f"\n📊 Creating PCA visualization...")
    create_pca_visualization(samples, title_prefix, log_to_wandb=not args.no_wandb)
    
    # 2. Distribution analysis
    print(f"\n📈 Creating distribution analysis...")
    create_distribution_analysis(samples, title_prefix, log_to_wandb=not args.no_wandb)
    
    # 3. t-SNE visualization
    print(f"\n🎨 Creating t-SNE visualization...")
    create_tsne_visualization(samples, title_prefix, log_to_wandb=not args.no_wandb)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"RHMC SAMPLES SUMMARY")
    print(f"{'='*80}")
    
    stats = {
        "n_samples": len(samples),
        "n_dimensions": samples.shape[1],
        "mean": samples.mean().item(),
        "std": samples.std().item(),
        "min": samples.min().item(),
        "max": samples.max().item(),
        "variance": samples.var().item(),
        "skewness": float(torch.mean(((samples - samples.mean()) / samples.std())**3)),
        "kurtosis": float(torch.mean(((samples - samples.mean()) / samples.std())**4) - 3)
    }
    
    print(f"📊 Sample Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    if not args.no_wandb:
        wandb.log({"rhmc_analysis/sample_statistics": stats})
    
    print(f"\n✅ RHMC samples visualization complete!")
    print(f"📊 Check WandB for detailed visualizations")

if __name__ == "__main__":
    main()
