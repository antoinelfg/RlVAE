#!/usr/bin/env python3
"""
Metric Analysis and Visualization Script
========================================

Analyzes extracted RHVAE-style metrics and creates comprehensive visualizations
to evaluate metric quality including eigenvalue distributions, condition numbers,
heatmaps, and metric function properties.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_metric(metric_path):
    """Load metric data from file."""
    print(f"Loading metric from: {metric_path}")
    metric_data = torch.load(metric_path, map_location='cpu', weights_only=False)
    
    print(f"Metric data keys: {list(metric_data.keys())}")
    print(f"Centroids shape: {metric_data['centroids'].shape}")
    print(f"M_matrices shape: {metric_data['M_matrices'].shape}")
    print(f"Temperature: {metric_data['temperature'].item()}")
    print(f"Regularization: {metric_data['regularization'].item()}")
    
    return metric_data

def analyze_eigenvalues(M_matrices, title_prefix=""):
    """Analyze eigenvalue properties of metric matrices."""
    print(f"\n=== {title_prefix}Eigenvalue Analysis ===")
    
    # Compute eigenvalues for all matrices
    eigenvals = torch.linalg.eigvals(M_matrices).real
    min_eigenvals = eigenvals.min(dim=-1)[0]
    max_eigenvals = eigenvals.max(dim=-1)[0]
    mean_eigenvals = eigenvals.mean(dim=-1)
    
    # Condition numbers and determinants
    cond_nums = max_eigenvals / (min_eigenvals + 1e-12)
    determinants = torch.linalg.det(M_matrices)
    
    # Print statistics
    print(f"Number of matrices: {len(M_matrices)}")
    print(f"Matrix size: {M_matrices.shape[-1]}x{M_matrices.shape[-1]}")
    print(f"Min eigenvalue range: [{min_eigenvals.min():.6f}, {min_eigenvals.max():.6f}]")
    print(f"Max eigenvalue range: [{max_eigenvals.min():.6f}, {max_eigenvals.max():.6f}]")
    print(f"Mean eigenvalue range: [{mean_eigenvals.min():.6f}, {mean_eigenvals.max():.6f}]")
    print(f"Condition number range: [{cond_nums.min():.2f}, {cond_nums.max():.2f}]")
    print(f"Determinant range: [{determinants.min():.6e}, {determinants.max():.6e}]")
    
    # Check positive definiteness
    negative_eigenvals = (min_eigenvals < 0).sum().item()
    print(f"Matrices with negative eigenvalues: {negative_eigenvals}/{len(M_matrices)}")
    
    return {
        'eigenvals': eigenvals,
        'min_eigenvals': min_eigenvals,
        'max_eigenvals': max_eigenvals, 
        'mean_eigenvals': mean_eigenvals,
        'cond_nums': cond_nums,
        'determinants': determinants,
        'negative_count': negative_eigenvals
    }

def create_eigenvalue_plots(stats, title_prefix="", log_to_wandb=True):
    """Create comprehensive eigenvalue distribution plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{title_prefix}Eigenvalue Analysis', fontsize=16, fontweight='bold')
    
    # 1. Min eigenvalue distribution
    axes[0,0].hist(stats['min_eigenvals'].numpy(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0,0].set_title('Min Eigenvalue Distribution')
    axes[0,0].set_xlabel('Min eigenvalue')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].axvline(stats['min_eigenvals'].mean(), color='darkred', linestyle='--', linewidth=2,
                     label=f'Mean: {stats["min_eigenvals"].mean():.4f}')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # 2. Max eigenvalue distribution  
    axes[0,1].hist(stats['max_eigenvals'].numpy(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0,1].set_title('Max Eigenvalue Distribution')
    axes[0,1].set_xlabel('Max eigenvalue')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].axvline(stats['max_eigenvals'].mean(), color='darkblue', linestyle='--', linewidth=2,
                     label=f'Mean: {stats["max_eigenvals"].mean():.4f}')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # 3. Mean eigenvalue distribution
    axes[0,2].hist(stats['mean_eigenvals'].numpy(), bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0,2].set_title('Mean Eigenvalue Distribution')
    axes[0,2].set_xlabel('Mean eigenvalue')
    axes[0,2].set_ylabel('Frequency')
    axes[0,2].axvline(stats['mean_eigenvals'].mean(), color='darkgreen', linestyle='--', linewidth=2,
                     label=f'Mean: {stats["mean_eigenvals"].mean():.4f}')
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].legend()
    
    # 4. Condition number distribution
    axes[1,0].hist(stats['cond_nums'].numpy(), bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1,0].set_title('Condition Number Distribution')
    axes[1,0].set_xlabel('Condition number')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].axvline(stats['cond_nums'].mean(), color='darkorange', linestyle='--', linewidth=2,
                     label=f'Mean: {stats["cond_nums"].mean():.2f}')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    
    # 5. Determinant distribution (log scale)
    log_dets = torch.log10(torch.abs(stats['determinants']) + 1e-50)
    axes[1,1].hist(log_dets.numpy(), bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1,1].set_title('Log₁₀(|Determinant|) Distribution')
    axes[1,1].set_xlabel('Log₁₀(|determinant|)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].axvline(log_dets.mean(), color='indigo', linestyle='--', linewidth=2,
                     label=f'Mean: {log_dets.mean():.2f}')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend()
    
    # 6. Eigenvalue spread (max - min)
    eigenval_spread = stats['max_eigenvals'] - stats['min_eigenvals']
    axes[1,2].hist(eigenval_spread.numpy(), bins=50, alpha=0.7, color='cyan', edgecolor='black')
    axes[1,2].set_title('Eigenvalue Spread Distribution')
    axes[1,2].set_xlabel('Max - Min eigenvalue')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].axvline(eigenval_spread.mean(), color='darkcyan', linestyle='--', linewidth=2,
                     label=f'Mean: {eigenval_spread.mean():.4f}')
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].legend()
    
    plt.tight_layout()
    
    if log_to_wandb:
        wandb.log({f"metric_analysis/{title_prefix.lower().replace(' ', '_')}eigenvalue_distributions": wandb.Image(fig)})
    
    return fig

def create_metric_heatmaps(M_matrices, centroids, max_matrices=10, title_prefix="", log_to_wandb=True):
    """Create heatmaps of selected metric matrices."""
    
    n_matrices = min(max_matrices, len(M_matrices))
    indices = torch.linspace(0, len(M_matrices)-1, n_matrices).long()
    
    # Create subplot grid
    cols = min(5, n_matrices)
    rows = (n_matrices + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    fig.suptitle(f'{title_prefix}Metric Matrix Heatmaps (Sample)', fontsize=16, fontweight='bold')
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
            
        matrix = M_matrices[idx].numpy()
        centroid = centroids[idx].numpy()
        
        # Create heatmap
        im = axes[i].imshow(matrix, cmap='RdYlBu_r', aspect='auto')
        axes[i].set_title(f'Matrix {idx.item()}\nCentroid norm: {torch.norm(centroids[idx]).item():.3f}')
        axes[i].set_xlabel('Latent dimension')
        axes[i].set_ylabel('Latent dimension')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(len(indices), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if log_to_wandb:
        wandb.log({f"metric_analysis/{title_prefix.lower().replace(' ', '_')}matrix_heatmaps": wandb.Image(fig)})
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Analyze extracted RHVAE metric")
    parser.add_argument("metric_path", help="Path to metric file")
    parser.add_argument("--wandb-project", default="metric_analysis", help="Wandb project name")
    parser.add_argument("--wandb-name", help="Wandb run name (default: auto from filename)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Initialize wandb
    if not args.no_wandb:
        run_name = args.wandb_name or Path(args.metric_path).stem
        wandb.init(
            project=args.wandb_project,
            name=f"metric_analysis_{run_name}",
            config={
                "metric_path": args.metric_path,
                "analysis_type": "comprehensive_metric_analysis"
            }
        )
    
    # Load metric
    metric_data = load_metric(args.metric_path)
    
    centroids = metric_data['centroids']
    M_matrices = metric_data['M_matrices'] 
    temperature = metric_data['temperature'].item()
    regularization = metric_data['regularization'].item()
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE METRIC ANALYSIS")
    print(f"{'='*80}")
    
    # 1. Analyze base metric matrices (Mᵢ)
    print(f"\n🔍 Analyzing base metric matrices...")
    M_stats = analyze_eigenvalues(M_matrices, "Base M_i ")
    create_eigenvalue_plots(M_stats, "Base M_i ", log_to_wandb=not args.no_wandb)
    
    # 2. Create metric heatmaps
    print(f"\n🎨 Creating metric matrix heatmaps...")
    create_metric_heatmaps(M_matrices, centroids, max_matrices=12, 
                          title_prefix="Base M_i ", log_to_wandb=not args.no_wandb)
    
    # Summary assessment
    print(f"\n{'='*80}")
    print(f"METRIC QUALITY ASSESSMENT")
    print(f"{'='*80}")
    
    quality_score = 0
    max_score = 5
    
    # Check 1: Positive definiteness
    if M_stats['negative_count'] == 0:
        print("✅ All matrices are positive definite")
        quality_score += 1
    else:
        print(f"❌ {M_stats['negative_count']} matrices have negative eigenvalues")
    
    # Check 2: Reasonable condition numbers  
    mean_cond = M_stats['cond_nums'].mean().item()
    if mean_cond < 10:
        print(f"✅ Good condition numbers (mean: {mean_cond:.2f})")
        quality_score += 1
    elif mean_cond < 100:
        print(f"⚠️  Acceptable condition numbers (mean: {mean_cond:.2f})")
        quality_score += 0.5
    else:
        print(f"❌ Poor condition numbers (mean: {mean_cond:.2f})")
    
    # Check 3: Eigenvalue range
    eigenval_range = M_stats['max_eigenvals'].max() / M_stats['min_eigenvals'].min()
    if eigenval_range < 100:
        print(f"✅ Good eigenvalue range (ratio: {eigenval_range:.2f})")
        quality_score += 1
    elif eigenval_range < 1000:
        print(f"⚠️  Acceptable eigenvalue range (ratio: {eigenval_range:.2f})")
        quality_score += 0.5
    else:
        print(f"❌ Poor eigenvalue range (ratio: {eigenval_range:.2f})")
    
    # Check 4: Centroid coverage
    centroid_coverage = torch.norm(centroids.std(dim=0)).item()
    if centroid_coverage > 1.0:
        print(f"✅ Good centroid coverage (std norm: {centroid_coverage:.3f})")
        quality_score += 1
    elif centroid_coverage > 0.5:
        print(f"⚠️  Acceptable centroid coverage (std norm: {centroid_coverage:.3f})")
        quality_score += 0.5
    else:
        print(f"❌ Poor centroid coverage (std norm: {centroid_coverage:.3f})")
    
    # Check 5: Parameter reasonableness
    if 0.05 <= temperature <= 0.5 and 0.001 <= regularization <= 0.1:
        print(f"✅ Reasonable parameters (T={temperature}, λ={regularization})")
        quality_score += 1
    else:
        print(f"⚠️  Check parameters (T={temperature}, λ={regularization})")
        quality_score += 0.5
    
    final_score = quality_score / max_score * 100
    print(f"\n🏆 Overall Quality Score: {final_score:.1f}% ({quality_score:.1f}/{max_score})")
    
    if not args.no_wandb:
        wandb.log({
            "metric_quality/overall_score": final_score,
            "metric_quality/positive_definite": M_stats['negative_count'] == 0,
            "metric_quality/mean_condition_number": mean_cond,
            "metric_quality/eigenvalue_range_ratio": eigenval_range,
            "metric_quality/centroid_coverage": centroid_coverage,
            "metric_quality/temperature": temperature,
            "metric_quality/regularization": regularization,
        })
        
        wandb.finish()
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
