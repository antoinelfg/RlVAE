#!/usr/bin/env python3
"""
Enhanced Generation and Visualization Script for RlVAE
======================================================

This script provides comprehensive visualization of generation capabilities including:
- Generated sequence visualization with FID scores
- Geodesic and Riemannian sampling comparisons
- Latent space analysis and trajectory plotting
- Quality metric analysis with beautiful graphs

Usage:
    python enhanced_generation_visualization.py --model-path path/to/model.ckpt
    python enhanced_generation_visualization.py --config conf/model/stage1_vanilla_vae_mlp_ld32.yaml
"""

import argparse
import sys
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
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
from src.evaluation.fid_scorer import create_fid_scorer

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def setup_plotting():
    """Setup matplotlib for beautiful plots."""
    plt.rcParams.update({
        'figure.figsize': (12, 8),
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


def get_test_dataset(dataset_name: str = "dsprites", subset_size: int = 500):
    """Get test dataset for evaluation."""
    print(f"🔍 Loading {dataset_name} dataset...")
    
    if dataset_name.lower() == "dsprites":
        # Load the dSprites cyclic dataset
        data_path = Path("data/processed")
        if (data_path / "Sprites_test_cyclic.pt").exists():
            test_data = torch.load(data_path / "Sprites_test_cyclic.pt")
            print(f"   📊 Loaded dSprites test data: {test_data.shape}")
            
            # Convert to standard format [N*T, C, H, W]
            if test_data.dim() == 5:  # [N, T, C, H, W]
                N, T, C, H, W = test_data.shape
                test_data = test_data.view(-1, C, H, W)
            
            # Create subset
            indices = torch.randperm(len(test_data))[:subset_size]
            subset_data = test_data[indices]
            
            # Create simple dataset
            dataset = torch.utils.data.TensorDataset(subset_data)
            return DataLoader(dataset, batch_size=32, shuffle=False)
        else:
            print("   ⚠️ dSprites data not found, falling back to CIFAR-10")
            dataset_name = "cifar10"
    
    # Standard transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    if dataset_name.lower() == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    # Create subset
    indices = torch.randperm(len(dataset))[:subset_size]
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=32, shuffle=False)


def generate_with_different_methods(generator: RlVAEGenerator, num_samples: int = 64) -> Dict[str, Dict]:
    """Generate samples using different methods and collect results."""
    print(f"🎨 Generating samples with different methods...")
    
    methods = ["geodesic", "enhanced", "basic", "standard"]
    results = {}
    
    for method in methods:
        print(f"   🎯 Generating with {method} method...")
        
        try:
            config = GenerationConfig(
                num_samples=num_samples,
                batch_size=16,
                sampling_method=method,
                sampler_type="working",
                sequence_length=8,  # Generate sequences
                use_flows=True
            )
            
            gen_result = generator.generate_from_prior(config)
            
            results[method] = {
                'images': gen_result['images'],
                'latents': gen_result['latents'],
                'config': config,
                'success': True
            }
            
            print(f"      ✅ Generated {len(gen_result['images'])} samples")
            
        except Exception as e:
            print(f"      ❌ Failed: {e}")
            results[method] = {'success': False, 'error': str(e)}
    
    return results


def compute_fid_scores(model: ModularRiemannianFlowVAE, real_images: torch.Tensor, 
                      generation_results: Dict[str, Dict]) -> Dict[str, float]:
    """Compute FID scores for different generation methods."""
    print(f"📊 Computing FID scores...")
    
    fid_scores = {}
    
    # Use subset of real images for FID computation
    real_subset = real_images[:200]
    
    for method, result in generation_results.items():
        if not result.get('success', False):
            continue
            
        print(f"   🔍 Computing FID for {method}...")
        
        try:
            # Get generated images and convert to proper format
            gen_images = result['images']
            if gen_images.dim() == 5:  # [N, T, C, H, W]
                gen_images = gen_images[:, 0]  # Take first frame
            
            # Compute FID
            fid_result = model.compute_fid_score(
                real_images=real_subset,
                num_generated=min(100, len(gen_images)),
                cache_key=f"enhanced_viz_{method}",
                sampling_method=method,
                sampler_type="working"
            )
            
            if fid_result and 'fid_score' in fid_result:
                fid_scores[method] = fid_result['fid_score']
                print(f"      ✅ FID Score: {fid_result['fid_score']:.2f}")
            else:
                print(f"      ❌ FID computation failed")
                
        except Exception as e:
            print(f"      ❌ Error computing FID: {e}")
    
    return fid_scores


def plot_generation_comparison(generation_results: Dict[str, Dict], fid_scores: Dict[str, float], 
                             save_path: Path):
    """Create comprehensive visualization of generation results."""
    print(f"📈 Creating generation comparison plots...")
    
    # Filter successful results
    successful_methods = [m for m in generation_results.keys() if generation_results[m].get('success', False)]
    
    if not successful_methods:
        print("   ❌ No successful generation results to plot")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, len(successful_methods), height_ratios=[3, 2, 1], hspace=0.3, wspace=0.2)
    
    # Plot generated images for each method
    for i, method in enumerate(successful_methods):
        images = generation_results[method]['images']
        
        # Take first frame if sequence
        if images.dim() == 5:
            images = images[:, 0]
        
        # Create grid of images (4x4)
        ax = fig.add_subplot(gs[0, i])
        grid_images = images[:16]  # Take first 16 images
        
        # Create image grid
        grid_size = 4
        combined_image = torch.zeros(3, grid_size * 64, grid_size * 64)
        
        for idx in range(min(16, len(grid_images))):
            row = idx // grid_size
            col = idx % grid_size
            img = grid_images[idx]
            combined_image[:, row*64:(row+1)*64, col*64:(col+1)*64] = img
        
        # Convert to numpy and plot
        combined_np = combined_image.permute(1, 2, 0).cpu().numpy()
        combined_np = np.clip(combined_np, 0, 1)
        
        ax.imshow(combined_np)
        ax.set_title(f'{method.capitalize()} Generation\n(16 samples)', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Plot FID scores comparison
    if fid_scores:
        ax_fid = fig.add_subplot(gs[1, :])
        methods = list(fid_scores.keys())
        scores = list(fid_scores.values())
        
        bars = ax_fid.bar(methods, scores, alpha=0.7, color=sns.color_palette("husl", len(methods)))
        ax_fid.set_ylabel('FID Score (lower is better)', fontweight='bold')
        ax_fid.set_title('FID Score Comparison Across Generation Methods', fontsize=14, fontweight='bold')
        ax_fid.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax_fid.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot latent space statistics
    ax_latent = fig.add_subplot(gs[2, :])
    latent_stats = []
    
    for method in successful_methods:
        latents = generation_results[method]['latents']
        if latents.dim() == 3:  # [N, T, D]
            latents = latents[:, 0]  # Take first frame
        
        mean_norm = torch.norm(latents, dim=1).mean().item()
        std_norm = torch.norm(latents, dim=1).std().item()
        latent_stats.append({'method': method, 'mean_norm': mean_norm, 'std_norm': std_norm})
    
    if latent_stats:
        methods = [s['method'] for s in latent_stats]
        mean_norms = [s['mean_norm'] for s in latent_stats]
        std_norms = [s['std_norm'] for s in latent_stats]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax_latent.bar(x - width/2, mean_norms, width, label='Mean Norm', alpha=0.7)
        bars2 = ax_latent.bar(x + width/2, std_norms, width, label='Std Norm', alpha=0.7)
        
        ax_latent.set_ylabel('Latent Norm')
        ax_latent.set_title('Latent Space Statistics by Generation Method')
        ax_latent.set_xticks(x)
        ax_latent.set_xticklabels(methods)
        ax_latent.legend()
        ax_latent.grid(True, alpha=0.3)
    
    plt.suptitle('Enhanced Generation Analysis with Geodesic Sampling', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'generation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Generation comparison saved to {save_path / 'generation_comparison.png'}")


def plot_sequence_generation(generation_results: Dict[str, Dict], save_path: Path):
    """Plot generated sequences showing temporal evolution."""
    print(f"🎬 Creating sequence generation plots...")
    
    successful_methods = [m for m in generation_results.keys() if generation_results[m].get('success', False)]
    
    if not successful_methods:
        return
    
    # Focus on geodesic method for sequence visualization
    method = "geodesic" if "geodesic" in successful_methods else successful_methods[0]
    images = generation_results[method]['images']
    
    if images.dim() != 5:  # Not sequences
        print("   ⚠️ No sequence data available for temporal visualization")
        return
    
    # Plot sequences
    n_sequences = min(4, images.shape[0])
    seq_len = images.shape[1]
    
    fig, axes = plt.subplots(n_sequences, seq_len, figsize=(seq_len * 2, n_sequences * 2))
    
    for seq_idx in range(n_sequences):
        for frame_idx in range(seq_len):
            ax = axes[seq_idx, frame_idx] if n_sequences > 1 else axes[frame_idx]
            
            img = images[seq_idx, frame_idx].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            ax.axis('off')
            
            if seq_idx == 0:
                ax.set_title(f'Frame {frame_idx + 1}', fontsize=10)
            if frame_idx == 0:
                ax.set_ylabel(f'Sequence {seq_idx + 1}', fontsize=10)
    
    plt.suptitle(f'Generated Sequences using {method.capitalize()} Sampling', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'sequence_generation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Sequence generation saved to {save_path / 'sequence_generation.png'}")


def create_results_summary(generation_results: Dict[str, Dict], fid_scores: Dict[str, float], 
                          save_path: Path):
    """Create a comprehensive results summary."""
    print(f"📋 Creating results summary...")
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'generation_methods_tested': list(generation_results.keys()),
        'successful_methods': [m for m in generation_results.keys() if generation_results[m].get('success', False)],
        'fid_scores': fid_scores,
        'best_fid_method': min(fid_scores.items(), key=lambda x: x[1])[0] if fid_scores else None,
        'generation_stats': {}
    }
    
    # Add generation statistics
    for method, result in generation_results.items():
        if result.get('success', False):
            images = result['images']
            if images.dim() == 5:
                images = images[:, 0]  # First frame
            
            summary['generation_stats'][method] = {
                'num_samples': len(images),
                'image_shape': list(images.shape[1:]),
                'value_range': [images.min().item(), images.max().item()],
                'mean_pixel_value': images.mean().item()
            }
    
    # Save summary
    with open(save_path / 'generation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   ✅ Summary saved to {save_path / 'generation_summary.json'}")
    
    return summary


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Enhanced Generation Visualization for RlVAE")
    parser.add_argument("--model-path", type=str, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to model config")
    parser.add_argument("--dataset", type=str, default="dsprites", help="Dataset to use")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of samples to generate")
    parser.add_argument("--output-dir", type=str, default="generation_analysis", help="Output directory")
    
    args = parser.parse_args()
    
    if not args.model_path and not args.config:
        print("❌ Please provide either --model-path or --config")
        return
    
    # Setup
    setup_plotting()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    if args.model_path:
        model = load_model_from_checkpoint(args.model_path, device)
    else:
        print("⚠️ Config-only mode not implemented in this example")
        return
    
    # Get test dataset
    test_loader = get_test_dataset(args.dataset, subset_size=500)
    
    # Get real images for FID computation
    real_images = []
    for batch in test_loader:
        if isinstance(batch, (list, tuple)):
            batch_imgs = batch[0]
        else:
            batch_imgs = batch
        real_images.append(batch_imgs)
        if len(torch.cat(real_images)) >= 500:
            break
    
    real_images = torch.cat(real_images)[:500]
    print(f"📊 Using {len(real_images)} real images for evaluation")
    
    # Create generator
    generator = RlVAEGenerator(model, device)
    
    # Generate samples with different methods
    generation_results = generate_with_different_methods(generator, args.num_samples)
    
    # Compute FID scores
    fid_scores = compute_fid_scores(model, real_images, generation_results)
    
    # Create visualizations
    plot_generation_comparison(generation_results, fid_scores, output_dir)
    plot_sequence_generation(generation_results, output_dir)
    
    # Create summary
    summary = create_results_summary(generation_results, fid_scores, output_dir)
    
    # Print results
    print(f"\n🎉 Enhanced Generation Analysis Complete!")
    print(f"📁 Results saved to: {output_dir}")
    
    if fid_scores:
        print(f"🏆 Best FID Score: {summary['best_fid_method']} ({fid_scores[summary['best_fid_method']]:.2f})")
        print(f"📊 All FID Scores:")
        for method, score in sorted(fid_scores.items(), key=lambda x: x[1]):
            print(f"   {method}: {score:.2f}")


if __name__ == "__main__":
    main() 