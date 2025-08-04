#!/usr/bin/env python3
"""
Enhanced Inference and Latent Space Visualization for RlVAE
===========================================================

This script provides comprehensive visualization of inference capabilities including:
- Latent space trajectory analysis for cyclic sequences
- Geodesic path visualization in Riemannian manifold
- Reconstruction quality analysis with uncertainty quantification
- Manifold structure exploration and interpolation

Usage:
    python enhanced_inference_visualization.py --model-path path/to/model.ckpt
    python enhanced_inference_visualization.py --model-path path/to/model.ckpt --sequence-analysis
"""

import argparse
import sys
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.modular_rlvae import ModularRiemannianFlowVAE
from src.generation.generator import RlVAEGenerator, GenerationConfig
from src.inference.inference_pipeline import RlVAEInferencePipeline, InferenceConfig

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def setup_plotting():
    """Setup matplotlib for beautiful plots."""
    plt.rcParams.update({
        'figure.figsize': (15, 10),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> ModularRiemannianFlowVAE:
    """Load model from checkpoint."""
    print(f"📂 Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif 'hyper_parameters' in checkpoint:
        config = checkpoint['hyper_parameters'].get('model')
        if config is None:
            raise ValueError("No model config found in hyper_parameters")
    else:
        raise ValueError("No config found in checkpoint")
    
    # Create model
    model = ModularRiemannianFlowVAE(config)
    
    # Handle PyTorch Lightning checkpoint format (remove "model." prefix)
    state_dict = checkpoint['state_dict']
    if any(key.startswith('model.') for key in state_dict.keys()):
        state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"   ✅ Model loaded successfully")
    return model


def load_cyclic_sequences(dataset_path: Path, num_sequences: int = 20):
    """Load cyclic sequences for latent space analysis."""
    print(f"🔄 Loading cyclic sequences...")
    
    # Try to load dSprites cyclic data
    test_data_path = dataset_path / "processed" / "Sprites_test_cyclic.pt"
    
    if test_data_path.exists():
        sequences = torch.load(test_data_path)
        print(f"   📊 Loaded cyclic sequences: {sequences.shape}")
        
        # Select subset for analysis
        indices = torch.randperm(len(sequences))[:num_sequences]
        selected_sequences = sequences[indices]
        
        return selected_sequences
    else:
        print(f"   ❌ Cyclic sequences not found at {test_data_path}")
        return None


def encode_sequences(sequences: torch.Tensor, inference_pipeline: RlVAEInferencePipeline) -> Dict:
    """Encode sequences to latent space."""
    print(f"🧠 Encoding {len(sequences)} sequences to latent space...")
    
    config = InferenceConfig(
        batch_size=16,
        use_mean=False,  # Sample for diversity
        sampling_method="geodesic",
        sequence_mode="sequence",
        return_uncertainties=True
    )
    
    # Encode sequences
    encoding_result = inference_pipeline.encode_images(sequences, config)
    
    print(f"   ✅ Encoded to latent space: {encoding_result['latents'].shape}")
    
    return {
        'latents': encoding_result['latents'],
        'uncertainties': encoding_result.get('uncertainties'),
        'posteriors': encoding_result.get('posteriors'),
        'config': config
    }


def analyze_latent_trajectories(latent_sequences: torch.Tensor, save_path: Path):
    """Analyze and visualize latent space trajectories."""
    print(f"🎯 Analyzing latent space trajectories...")
    
    N, T, D = latent_sequences.shape
    print(f"   📊 Sequences: {N}, Time steps: {T}, Latent dim: {D}")
    
    # Compute trajectory statistics
    trajectory_stats = compute_trajectory_statistics(latent_sequences)
    
    # Create comprehensive trajectory visualization
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 4, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)
    
    # 1. PCA projection of trajectories
    ax1 = fig.add_subplot(gs[0, 0])
    plot_pca_trajectories(latent_sequences, ax1)
    
    # 2. t-SNE projection of all latent points
    ax2 = fig.add_subplot(gs[0, 1])
    plot_tsne_latent_space(latent_sequences, ax2)
    
    # 3. 3D trajectory visualization
    ax3 = fig.add_subplot(gs[0, 2:], projection='3d')
    plot_3d_trajectories(latent_sequences, ax3)
    
    # 4. Trajectory distances and curvature
    ax4 = fig.add_subplot(gs[1, 0])
    plot_trajectory_distances(latent_sequences, ax4)
    
    # 5. Cyclic consistency analysis
    ax5 = fig.add_subplot(gs[1, 1])
    plot_cyclic_consistency(latent_sequences, ax5)
    
    # 6. Latent space norms over time
    ax6 = fig.add_subplot(gs[1, 2])
    plot_latent_norms(latent_sequences, ax6)
    
    # 7. Inter-sequence similarity
    ax7 = fig.add_subplot(gs[1, 3])
    plot_sequence_similarity(latent_sequences, ax7)
    
    # 8. Summary statistics
    ax8 = fig.add_subplot(gs[2, :])
    plot_trajectory_summary(trajectory_stats, ax8)
    
    plt.suptitle('Latent Space Trajectory Analysis with Riemannian Geometry', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'latent_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Trajectory analysis saved to {save_path / 'latent_trajectories.png'}")
    
    return trajectory_stats


def compute_trajectory_statistics(latent_sequences: torch.Tensor) -> Dict:
    """Compute comprehensive trajectory statistics."""
    N, T, D = latent_sequences.shape
    
    stats = {}
    
    # Cyclic consistency (distance from first to last frame)
    cycle_distances = torch.norm(latent_sequences[:, 0] - latent_sequences[:, -1], dim=1)
    stats['cycle_consistency'] = {
        'mean_distance': cycle_distances.mean().item(),
        'std_distance': cycle_distances.std().item(),
        'max_distance': cycle_distances.max().item(),
        'perfect_cycles': (cycle_distances < 0.1).sum().item(),
        'total_sequences': N
    }
    
    # Trajectory smoothness (consecutive frame distances)
    consecutive_distances = torch.norm(
        latent_sequences[:, 1:] - latent_sequences[:, :-1], dim=2
    )
    stats['smoothness'] = {
        'mean_step_distance': consecutive_distances.mean().item(),
        'std_step_distance': consecutive_distances.std().item(),
        'max_step_distance': consecutive_distances.max().item()
    }
    
    # Total trajectory length
    trajectory_lengths = consecutive_distances.sum(dim=1)
    stats['trajectory_length'] = {
        'mean_length': trajectory_lengths.mean().item(),
        'std_length': trajectory_lengths.std().item(),
        'min_length': trajectory_lengths.min().item(),
        'max_length': trajectory_lengths.max().item()
    }
    
    # Latent space coverage
    all_latents = latent_sequences.view(-1, D)
    latent_norms = torch.norm(all_latents, dim=1)
    stats['latent_coverage'] = {
        'mean_norm': latent_norms.mean().item(),
        'std_norm': latent_norms.std().item(),
        'min_norm': latent_norms.min().item(),
        'max_norm': latent_norms.max().item()
    }
    
    return stats


def plot_pca_trajectories(latent_sequences: torch.Tensor, ax):
    """Plot trajectories in PCA space."""
    N, T, D = latent_sequences.shape
    
    # Fit PCA on all latent points
    all_latents = latent_sequences.view(-1, D).cpu().numpy()
    pca = PCA(n_components=2)
    pca_latents = pca.fit_transform(all_latents)
    pca_sequences = pca_latents.reshape(N, T, 2)
    
    # Plot trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, min(N, 10)))
    
    for i in range(min(N, 10)):  # Plot first 10 sequences
        ax.plot(pca_sequences[i, :, 0], pca_sequences[i, :, 1], 
               color=colors[i], alpha=0.7, linewidth=1.5)
        ax.scatter(pca_sequences[i, 0, 0], pca_sequences[i, 0, 1], 
                  color=colors[i], s=50, marker='o', label=f'Seq {i+1} start' if i < 5 else "")
        ax.scatter(pca_sequences[i, -1, -1], pca_sequences[i, -1, 1], 
                  color=colors[i], s=50, marker='x')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title('PCA Trajectory Projection')
    ax.grid(True, alpha=0.3)
    if min(N, 10) <= 5:
        ax.legend(fontsize=8)


def plot_tsne_latent_space(latent_sequences: torch.Tensor, ax):
    """Plot t-SNE projection of latent space."""
    N, T, D = latent_sequences.shape
    all_latents = latent_sequences.view(-1, D).cpu().numpy()
    
    # Use subset for t-SNE (it's slow)
    subset_size = min(1000, len(all_latents))
    indices = np.random.choice(len(all_latents), subset_size, replace=False)
    subset_latents = all_latents[indices]
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_latents = tsne.fit_transform(subset_latents)
    
    # Color by sequence position
    sequence_indices = indices // T
    time_indices = indices % T
    
    scatter = ax.scatter(tsne_latents[:, 0], tsne_latents[:, 1], 
                        c=time_indices, cmap='viridis', alpha=0.6, s=20)
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE Latent Space\n(colored by time)')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time Step')


def plot_3d_trajectories(latent_sequences: torch.Tensor, ax):
    """Plot 3D trajectories using first 3 PCA components."""
    N, T, D = latent_sequences.shape
    
    # Fit PCA on all latent points
    all_latents = latent_sequences.view(-1, D).cpu().numpy()
    pca = PCA(n_components=3)
    pca_latents = pca.fit_transform(all_latents)
    pca_sequences = pca_latents.reshape(N, T, 3)
    
    # Plot trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, min(N, 8)))
    
    for i in range(min(N, 8)):  # Plot first 8 sequences
        ax.plot(pca_sequences[i, :, 0], pca_sequences[i, :, 1], pca_sequences[i, :, 2],
               color=colors[i], alpha=0.7, linewidth=2)
        ax.scatter(pca_sequences[i, 0, 0], pca_sequences[i, 0, 1], pca_sequences[i, 0, 2],
                  color=colors[i], s=100, marker='o')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax.set_title('3D Trajectory Visualization')


def plot_trajectory_distances(latent_sequences: torch.Tensor, ax):
    """Plot trajectory distances over time."""
    N, T, D = latent_sequences.shape
    
    # Compute distances from start point
    start_points = latent_sequences[:, 0:1]  # [N, 1, D]
    distances = torch.norm(latent_sequences - start_points, dim=2)  # [N, T]
    
    # Plot mean and std
    mean_distances = distances.mean(dim=0).cpu()
    std_distances = distances.std(dim=0).cpu()
    time_steps = np.arange(T)
    
    ax.plot(time_steps, mean_distances, 'b-', linewidth=2, label='Mean distance')
    ax.fill_between(time_steps, 
                   mean_distances - std_distances, 
                   mean_distances + std_distances,
                   alpha=0.3, color='blue', label='±1 std')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Distance from Start')
    ax.set_title('Trajectory Distance Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_cyclic_consistency(latent_sequences: torch.Tensor, ax):
    """Plot cyclic consistency analysis."""
    N, T, D = latent_sequences.shape
    
    # Compute cycle distances
    cycle_distances = torch.norm(latent_sequences[:, 0] - latent_sequences[:, -1], dim=1)
    
    # Create histogram
    ax.hist(cycle_distances.cpu().numpy(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(cycle_distances.mean().item(), color='red', linestyle='--', 
              label=f'Mean: {cycle_distances.mean():.3f}')
    ax.axvline(0.1, color='green', linestyle='--', 
              label='Perfect cycle threshold (0.1)')
    
    ax.set_xlabel('Cycle Distance (||z_0 - z_T||)')
    ax.set_ylabel('Frequency')
    ax.set_title('Cyclic Consistency Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_latent_norms(latent_sequences: torch.Tensor, ax):
    """Plot latent space norms over time."""
    N, T, D = latent_sequences.shape
    
    # Compute norms over time
    norms = torch.norm(latent_sequences, dim=2)  # [N, T]
    
    # Plot statistics
    mean_norms = norms.mean(dim=0).cpu()
    std_norms = norms.std(dim=0).cpu()
    time_steps = np.arange(T)
    
    ax.plot(time_steps, mean_norms, 'g-', linewidth=2, label='Mean norm')
    ax.fill_between(time_steps,
                   mean_norms - std_norms,
                   mean_norms + std_norms,
                   alpha=0.3, color='green', label='±1 std')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Latent Norm')
    ax.set_title('Latent Space Norms Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_sequence_similarity(latent_sequences: torch.Tensor, ax):
    """Plot inter-sequence similarity matrix."""
    N, T, D = latent_sequences.shape
    
    # Compute pairwise similarities using first frame
    first_frames = latent_sequences[:, 0]  # [N, D]
    similarity_matrix = torch.mm(first_frames, first_frames.t())
    similarity_matrix = similarity_matrix.cpu().numpy()
    
    # Normalize to [0, 1]
    similarity_matrix = (similarity_matrix - similarity_matrix.min()) / \
                       (similarity_matrix.max() - similarity_matrix.min())
    
    im = ax.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Sequence Index')
    ax.set_ylabel('Sequence Index')
    ax.set_title('Inter-Sequence Similarity\n(First Frame)')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)


def plot_trajectory_summary(trajectory_stats: Dict, ax):
    """Plot summary statistics."""
    # Create summary text
    summary_text = []
    summary_text.append("Trajectory Analysis Summary")
    summary_text.append("=" * 30)
    
    # Cycle consistency
    cc = trajectory_stats['cycle_consistency']
    summary_text.append(f"Cycle Consistency:")
    summary_text.append(f"  Mean distance: {cc['mean_distance']:.4f}")
    total_sequences = cc.get('total_sequences', 'N/A')
    summary_text.append(f"  Perfect cycles: {cc['perfect_cycles']}/{total_sequences}")
    
    # Smoothness
    sm = trajectory_stats['smoothness']
    summary_text.append(f"\nTrajectory Smoothness:")
    summary_text.append(f"  Mean step: {sm['mean_step_distance']:.4f}")
    summary_text.append(f"  Max step: {sm['max_step_distance']:.4f}")
    
    # Length
    tl = trajectory_stats['trajectory_length']
    summary_text.append(f"\nTrajectory Length:")
    summary_text.append(f"  Mean: {tl['mean_length']:.4f}")
    summary_text.append(f"  Range: [{tl['min_length']:.4f}, {tl['max_length']:.4f}]")
    
    # Coverage
    lc = trajectory_stats['latent_coverage']
    summary_text.append(f"\nLatent Space Coverage:")
    summary_text.append(f"  Mean norm: {lc['mean_norm']:.4f}")
    summary_text.append(f"  Norm range: [{lc['min_norm']:.4f}, {lc['max_norm']:.4f}]")
    
    ax.text(0.05, 0.95, '\n'.join(summary_text), transform=ax.transAxes,
           fontfamily='monospace', fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def analyze_reconstruction_quality(sequences: torch.Tensor, inference_pipeline: RlVAEInferencePipeline,
                                 save_path: Path):
    """Analyze reconstruction quality and uncertainty."""
    print(f"🔍 Analyzing reconstruction quality...")
    
    # Select subset for analysis
    analysis_sequences = sequences[:5]  # Analyze first 5 sequences
    
    # Encode and reconstruct
    config = InferenceConfig(
        batch_size=5,
        use_mean=False,
        sampling_method="geodesic",
        sequence_mode="sequence",
        return_uncertainties=True
    )
    
    reconstruction_result = inference_pipeline.encode_and_reconstruct(analysis_sequences, config)
    
    # Create reconstruction comparison
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    
    for seq_idx in range(min(3, len(analysis_sequences))):
        for frame_idx in range(min(8, analysis_sequences.shape[1])):
            # Original
            orig_img = analysis_sequences[seq_idx, frame_idx].permute(1, 2, 0).cpu().numpy()
            orig_img = np.clip(orig_img, 0, 1)
            
            # Reconstruction
            recon_img = reconstruction_result['reconstructions'][seq_idx, frame_idx].permute(1, 2, 0).cpu().numpy()
            recon_img = np.clip(recon_img, 0, 1)
            
            # Show images
            if seq_idx == 0:
                axes[0, frame_idx].imshow(orig_img)
                axes[0, frame_idx].set_title(f'Original F{frame_idx+1}', fontsize=8)
                axes[1, frame_idx].imshow(recon_img)
                axes[1, frame_idx].set_title(f'Recon F{frame_idx+1}', fontsize=8)
                # Difference
                diff_img = np.abs(orig_img - recon_img)
                axes[2, frame_idx].imshow(diff_img, cmap='hot')
                axes[2, frame_idx].set_title(f'Diff F{frame_idx+1}', fontsize=8)
            
            for row in range(3):
                axes[row, frame_idx].axis('off')
    
    plt.suptitle('Reconstruction Quality Analysis (Sequence 1)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'reconstruction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Reconstruction analysis saved to {save_path / 'reconstruction_analysis.png'}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Enhanced Inference Visualization for RlVAE")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-path", type=str, default="data", help="Path to data directory")
    parser.add_argument("--num-sequences", type=int, default=20, help="Number of sequences to analyze")
    parser.add_argument("--output-dir", type=str, default="inference_analysis", help="Output directory")
    parser.add_argument("--sequence-analysis", action="store_true", help="Perform detailed sequence analysis")
    
    args = parser.parse_args()
    
    # Setup
    setup_plotting()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    model = load_model_from_checkpoint(args.model_path, device)
    
    # Create inference pipeline
    inference_pipeline = RlVAEInferencePipeline(model, device)
    
    # Load cyclic sequences
    sequences = load_cyclic_sequences(Path(args.data_path), args.num_sequences)
    
    if sequences is None:
        print("❌ No cyclic sequences available for analysis")
        return
    
    sequences = sequences.to(device)
    
    # Encode sequences to latent space
    encoding_result = encode_sequences(sequences, inference_pipeline)
    latent_sequences = encoding_result['latents']
    
    # Analyze latent trajectories
    trajectory_stats = analyze_latent_trajectories(latent_sequences, output_dir)
    
    # Analyze reconstruction quality if requested
    if args.sequence_analysis:
        analyze_reconstruction_quality(sequences, inference_pipeline, output_dir)
    
    # Save analysis results
    analysis_summary = {
        'timestamp': datetime.now().isoformat(),
        'model_path': args.model_path,
        'num_sequences_analyzed': len(sequences),
        'latent_dimension': latent_sequences.shape[-1],
        'sequence_length': latent_sequences.shape[1],
        'trajectory_statistics': trajectory_stats
    }
    
    with open(output_dir / 'inference_analysis_summary.json', 'w') as f:
        json.dump(analysis_summary, f, indent=2)
    
    # Print results
    print(f"\n🎉 Enhanced Inference Analysis Complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"🧠 Analyzed {len(sequences)} sequences with {latent_sequences.shape[-1]}D latents")
    
    if trajectory_stats:
        cc = trajectory_stats['cycle_consistency']
        print(f"🔄 Cycle consistency: {cc['perfect_cycles']}/{len(sequences)} perfect cycles")
        print(f"📏 Mean cycle distance: {cc['mean_distance']:.4f}")


if __name__ == "__main__":
    main() 