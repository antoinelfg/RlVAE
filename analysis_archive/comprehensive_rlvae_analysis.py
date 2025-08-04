#!/usr/bin/env python3
"""
Comprehensive RlVAE Analysis and Visualization Suite
====================================================

This script provides a complete analysis suite for RlVAE models including:
- Enhanced generation with FID scores and multiple sampling methods
- Latent space trajectory analysis with Riemannian geometry
- Geodesic interpolation and manifold exploration
- Reconstruction quality and uncertainty analysis
- Interactive visualizations and comprehensive reports

Usage:
    python comprehensive_rlvae_analysis.py --model-path path/to/model.ckpt --full-analysis
    python comprehensive_rlvae_analysis.py --model-path path/to/model.ckpt --generation-only
    python comprehensive_rlvae_analysis.py --model-path path/to/model.ckpt --inference-only
"""

import argparse
import sys
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
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
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })


class RlVAEAnalyzer:
    """Comprehensive analyzer for RlVAE models."""
    
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        """Initialize analyzer with model."""
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model_from_checkpoint(model_path)
        
        # Initialize components
        self.generator = RlVAEGenerator(self.model, self.device)
        self.inference_pipeline = RlVAEInferencePipeline(self.model, self.device)
        
        # Analysis results
        self.generation_results = {}
        self.inference_results = {}
        self.geodesic_results = {}
        
        print(f"🚀 RlVAE Analyzer initialized on {self.device}")
    
    def load_model_from_checkpoint(self, checkpoint_path: str) -> ModularRiemannianFlowVAE:
        """Load model from checkpoint."""
        print(f"📂 Loading model from {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
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
        model.to(self.device)
        model.eval()
        
        print(f"   ✅ Model loaded successfully")
        print(f"   📊 Latent dimension: {getattr(config, 'latent_dim', 'Unknown')}")
        
        return model
    
    def load_test_data(self, dataset_name: str = "dsprites", subset_size: int = 500) -> torch.Tensor:
        """Load test dataset for analysis."""
        print(f"🔍 Loading {dataset_name} dataset...")
        
        if dataset_name.lower() == "dsprites":
            # Load dSprites cyclic data
            data_path = Path("data/processed")
            test_data_path = data_path / "Sprites_test_cyclic.pt"
            
            if test_data_path.exists():
                sequences = torch.load(test_data_path)
                print(f"   📊 Loaded dSprites sequences: {sequences.shape}")
                
                # Convert to individual images for general analysis
                if sequences.dim() == 5:  # [N, T, C, H, W]
                    N, T, C, H, W = sequences.shape
                    images = sequences.view(-1, C, H, W)
                    sequences_subset = sequences[:min(subset_size//T, N)]
                else:
                    images = sequences
                    sequences_subset = None
                
                # Select subset
                indices = torch.randperm(len(images))[:subset_size]
                image_subset = images[indices]
                
                return {
                    'images': image_subset,
                    'sequences': sequences_subset,
                    'is_cyclic': True
                }
            else:
                print("   ⚠️ dSprites data not found, falling back to CIFAR-10")
                dataset_name = "cifar10"
        
        # Standard datasets
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
        
        # Convert to tensor
        loader = DataLoader(subset, batch_size=subset_size, shuffle=False)
        images = next(iter(loader))[0]
        
        return {
            'images': images,
            'sequences': None,
            'is_cyclic': False
        }
    
    def run_generation_analysis(self, num_samples: int = 64) -> Dict:
        """Run comprehensive generation analysis."""
        print(f"\n🎨 Running Generation Analysis")
        print("=" * 50)
        
        # Test different generation methods
        methods = ["geodesic", "enhanced", "basic", "standard"]
        generation_results = {}
        
        for method in methods:
            print(f"   🎯 Testing {method} generation...")
            
            try:
                config = GenerationConfig(
                    num_samples=num_samples,
                    batch_size=16,
                    sampling_method=method,
                    sampler_type="working",
                    sequence_length=8,
                    use_flows=True
                )
                
                gen_result = self.generator.generate_from_prior(config)
                
                generation_results[method] = {
                    'images': gen_result['images'],
                    'latents': gen_result['latents'],
                    'config': config,
                    'success': True
                }
                
                print(f"      ✅ Generated {len(gen_result['images'])} samples")
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                generation_results[method] = {'success': False, 'error': str(e)}
        
        self.generation_results = generation_results
        return generation_results
    
    def run_inference_analysis(self, test_data: Dict, num_sequences: int = 20) -> Dict:
        """Run comprehensive inference analysis."""
        print(f"\n🧠 Running Inference Analysis")
        print("=" * 50)
        
        inference_results = {}
        
        # Analyze individual images
        if test_data['images'] is not None:
            print("   🔍 Analyzing image encoding...")
            
            # Select subset for encoding
            subset_images = test_data['images'][:num_sequences*4].to(self.device)
            
            config = InferenceConfig(
                batch_size=16,
                use_mean=False,
                sampling_method="geodesic",
                sequence_mode="single",
                return_uncertainties=True
            )
            
            encoding_result = self.inference_pipeline.encode_images(subset_images, config)
            
            inference_results['image_encoding'] = {
                'latents': encoding_result['latents'],
                'uncertainties': encoding_result.get('uncertainties'),
                'original_images': subset_images,
                'success': True
            }
            
            print(f"      ✅ Encoded {len(subset_images)} images")
        
        # Analyze sequences if available
        if test_data['sequences'] is not None and test_data['is_cyclic']:
            print("   🔄 Analyzing cyclic sequences...")
            
            sequences = test_data['sequences'][:num_sequences].to(self.device)
            
            config = InferenceConfig(
                batch_size=8,
                use_mean=False,
                sampling_method="geodesic",
                sequence_mode="sequence",
                return_uncertainties=True
            )
            
            sequence_encoding = self.inference_pipeline.encode_images(sequences, config)
            
            # Analyze trajectories
            trajectory_stats = self.analyze_trajectory_statistics(sequence_encoding['latents'])
            
            inference_results['sequence_analysis'] = {
                'latent_sequences': sequence_encoding['latents'],
                'original_sequences': sequences,
                'trajectory_stats': trajectory_stats,
                'success': True
            }
            
            print(f"      ✅ Analyzed {len(sequences)} cyclic sequences")
        
        self.inference_results = inference_results
        return inference_results
    
    def run_geodesic_analysis(self) -> Dict:
        """Run geodesic interpolation and manifold analysis."""
        print(f"\n🌐 Running Geodesic Analysis")
        print("=" * 50)
        
        geodesic_results = {}
        
        # Generate random latent points for interpolation
        latent_dim = getattr(self.model.config, 'latent_dim', 16)
        
        # Sample random points
        z1 = torch.randn(1, latent_dim).to(self.device)
        z2 = torch.randn(1, latent_dim).to(self.device)
        
        try:
            # Generate interpolation using generator
            interpolation_result = self.generator.interpolate(
                z1, z2, num_steps=10, method="geodesic"
            )
            
            geodesic_results['interpolation'] = {
                'start_latent': z1,
                'end_latent': z2,
                'interpolated_latents': interpolation_result['latents'],
                'interpolated_images': interpolation_result['images'],
                'success': True
            }
            
            print(f"      ✅ Generated geodesic interpolation with {len(interpolation_result['images'])} steps")
            
        except Exception as e:
            print(f"      ❌ Geodesic interpolation failed: {e}")
            geodesic_results['interpolation'] = {'success': False, 'error': str(e)}
        
        # Manifold sampling analysis
        try:
            # Sample points around a central location
            center = torch.zeros(1, latent_dim).to(self.device)
            
            # Generate samples at different radii
            radii = [0.5, 1.0, 1.5, 2.0]
            manifold_samples = {}
            
            for radius in radii:
                # Sample on sphere
                directions = torch.randn(8, latent_dim)
                directions = directions / torch.norm(directions, dim=1, keepdim=True)
                points = center + radius * directions.to(self.device)
                
                # Generate images
                config = GenerationConfig(
                    num_samples=len(points),
                    batch_size=len(points),
                    sampling_method="geodesic"
                )
                
                gen_result = self.generator.generate_from_latents(points, config)
                
                manifold_samples[f'radius_{radius}'] = {
                    'latents': points,
                    'images': gen_result['images']
                }
            
            geodesic_results['manifold_sampling'] = {
                'center': center,
                'samples': manifold_samples,
                'success': True
            }
            
            print(f"      ✅ Analyzed manifold structure at {len(radii)} radii")
            
        except Exception as e:
            print(f"      ❌ Manifold analysis failed: {e}")
            geodesic_results['manifold_sampling'] = {'success': False, 'error': str(e)}
        
        self.geodesic_results = geodesic_results
        return geodesic_results
    
    def analyze_trajectory_statistics(self, latent_sequences: torch.Tensor) -> Dict:
        """Compute trajectory statistics for cyclic sequences."""
        N, T, D = latent_sequences.shape
        
        stats = {}
        
        # Cyclic consistency
        cycle_distances = torch.norm(latent_sequences[:, 0] - latent_sequences[:, -1], dim=1)
        stats['cycle_consistency'] = {
            'mean_distance': cycle_distances.mean().item(),
            'std_distance': cycle_distances.std().item(),
            'perfect_cycles': (cycle_distances < 0.1).sum().item(),
            'total_sequences': N
        }
        
        # Trajectory smoothness
        consecutive_distances = torch.norm(
            latent_sequences[:, 1:] - latent_sequences[:, :-1], dim=2
        )
        stats['smoothness'] = {
            'mean_step_distance': consecutive_distances.mean().item(),
            'std_step_distance': consecutive_distances.std().item(),
            'max_step_distance': consecutive_distances.max().item()
        }
        
        # Trajectory length
        trajectory_lengths = consecutive_distances.sum(dim=1)
        stats['trajectory_length'] = {
            'mean_length': trajectory_lengths.mean().item(),
            'std_length': trajectory_lengths.std().item(),
            'min_length': trajectory_lengths.min().item(),
            'max_length': trajectory_lengths.max().item()
        }
        
        return stats
    
    def compute_fid_scores(self, real_images: torch.Tensor) -> Dict[str, float]:
        """Compute FID scores for different generation methods."""
        print(f"📊 Computing FID scores...")
        
        fid_scores = {}
        real_subset = real_images[:200].to(self.device)
        
        for method, result in self.generation_results.items():
            if not result.get('success', False):
                continue
                
            print(f"   🔍 Computing FID for {method}...")
            
            try:
                # Get generated images
                gen_images = result['images']
                if gen_images.dim() == 5:  # [N, T, C, H, W]
                    gen_images = gen_images[:, 0]  # Take first frame
                
                # Compute FID using model's built-in method
                fid_result = self.model.compute_fid_score(
                    real_images=real_subset,
                    num_generated=min(100, len(gen_images)),
                    cache_key=f"comprehensive_{method}",
                    sampling_method=method,
                    sampler_type="working"
                )
                
                if fid_result and 'fid_score' in fid_result:
                    fid_scores[method] = fid_result['fid_score']
                    print(f"      ✅ FID Score: {fid_result['fid_score']:.2f}")
                
            except Exception as e:
                print(f"      ❌ Error computing FID: {e}")
        
        return fid_scores
    
    def create_comprehensive_visualization(self, save_path: Path, fid_scores: Dict = None):
        """Create comprehensive visualization of all analysis results."""
        print(f"📈 Creating comprehensive visualization...")
        
        # Create master figure
        fig = plt.figure(figsize=(24, 16))
        gs = gridspec.GridSpec(4, 6, height_ratios=[2, 2, 2, 1], hspace=0.3, wspace=0.3)
        
        # 1. Generation Comparison (top row)
        if self.generation_results:
            self.plot_generation_grid(fig, gs[0, :3])
            if fid_scores:
                self.plot_fid_comparison(fig, gs[0, 3:], fid_scores)
        
        # 2. Latent Space Analysis (second row)
        if 'sequence_analysis' in self.inference_results:
            latent_seqs = self.inference_results['sequence_analysis']['latent_sequences']
            self.plot_latent_trajectories(fig, gs[1, :3], latent_seqs)
            self.plot_trajectory_statistics(fig, gs[1, 3:], 
                                          self.inference_results['sequence_analysis']['trajectory_stats'])
        
        # 3. Geodesic Analysis (third row)
        if self.geodesic_results:
            if 'interpolation' in self.geodesic_results and self.geodesic_results['interpolation']['success']:
                self.plot_geodesic_interpolation(fig, gs[2, :3], 
                                                self.geodesic_results['interpolation'])
            if 'manifold_sampling' in self.geodesic_results and self.geodesic_results['manifold_sampling']['success']:
                self.plot_manifold_analysis(fig, gs[2, 3:], 
                                           self.geodesic_results['manifold_sampling'])
        
        # 4. Summary Statistics (bottom row)
        self.plot_analysis_summary(fig, gs[3, :])
        
        plt.suptitle('Comprehensive RlVAE Analysis with Riemannian Geometry', 
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Comprehensive visualization saved to {save_path / 'comprehensive_analysis.png'}")
    
    def plot_generation_grid(self, fig, grid_spec):
        """Plot generation results grid."""
        ax = fig.add_subplot(grid_spec)
        
        successful_methods = [m for m in self.generation_results.keys() 
                            if self.generation_results[m].get('success', False)]
        
        if not successful_methods:
            ax.text(0.5, 0.5, 'No successful generation results', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Generation Results')
            return
        
        # Create grid showing samples from each method
        n_methods = len(successful_methods)
        n_samples_per_method = 4
        
        grid_image = torch.zeros(3, n_methods * 64, n_samples_per_method * 64)
        
        for i, method in enumerate(successful_methods):
            images = self.generation_results[method]['images']
            if images.dim() == 5:
                images = images[:, 0]  # First frame
            
            for j in range(min(n_samples_per_method, len(images))):
                img = images[j]
                grid_image[:, i*64:(i+1)*64, j*64:(j+1)*64] = img
        
        grid_np = grid_image.permute(1, 2, 0).numpy()
        grid_np = np.clip(grid_np, 0, 1)
        
        ax.imshow(grid_np)
        ax.set_title('Generation Results by Method')
        ax.axis('off')
        
        # Add method labels
        for i, method in enumerate(successful_methods):
            ax.text(32, i*64 + 32, method.capitalize(), 
                   rotation=90, ha='center', va='center',
                   fontweight='bold', color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    def plot_fid_comparison(self, fig, grid_spec, fid_scores):
        """Plot FID score comparison."""
        ax = fig.add_subplot(grid_spec)
        
        if not fid_scores:
            ax.text(0.5, 0.5, 'No FID scores available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('FID Scores')
            return
        
        methods = list(fid_scores.keys())
        scores = list(fid_scores.values())
        
        bars = ax.bar(methods, scores, alpha=0.7, color=sns.color_palette("husl", len(methods)))
        ax.set_ylabel('FID Score (lower is better)', fontweight='bold')
        ax.set_title('FID Score Comparison', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    def plot_latent_trajectories(self, fig, grid_spec, latent_sequences):
        """Plot latent space trajectories."""
        ax = fig.add_subplot(grid_spec)
        
        N, T, D = latent_sequences.shape
        
        # Use PCA for visualization
        all_latents = latent_sequences.view(-1, D).cpu().numpy()
        pca = PCA(n_components=2)
        pca_latents = pca.fit_transform(all_latents)
        pca_sequences = pca_latents.reshape(N, T, 2)
        
        # Plot trajectories
        colors = plt.cm.tab10(np.linspace(0, 1, min(N, 10)))
        
        for i in range(min(N, 10)):
            ax.plot(pca_sequences[i, :, 0], pca_sequences[i, :, 1], 
                   color=colors[i], alpha=0.7, linewidth=1.5)
            ax.scatter(pca_sequences[i, 0, 0], pca_sequences[i, 0, 1], 
                      color=colors[i], s=50, marker='o')
            ax.scatter(pca_sequences[i, -1, 0], pca_sequences[i, -1, 1], 
                      color=colors[i], s=50, marker='x')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        ax.set_title('Latent Space Trajectories (PCA)')
        ax.grid(True, alpha=0.3)
    
    def plot_trajectory_statistics(self, fig, grid_spec, trajectory_stats):
        """Plot trajectory statistics."""
        ax = fig.add_subplot(grid_spec)
        
        if not trajectory_stats:
            ax.text(0.5, 0.5, 'No trajectory statistics', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Trajectory Statistics')
            return
        
        # Create summary text
        summary_text = []
        summary_text.append("Trajectory Analysis")
        summary_text.append("=" * 20)
        
        cc = trajectory_stats['cycle_consistency']
        summary_text.append(f"Cycle Consistency:")
        summary_text.append(f"  Perfect cycles: {cc['perfect_cycles']}/{cc['total_sequences']}")
        summary_text.append(f"  Mean distance: {cc['mean_distance']:.4f}")
        
        sm = trajectory_stats['smoothness']
        summary_text.append(f"\nSmoothness:")
        summary_text.append(f"  Mean step: {sm['mean_step_distance']:.4f}")
        summary_text.append(f"  Max step: {sm['max_step_distance']:.4f}")
        
        tl = trajectory_stats['trajectory_length']
        summary_text.append(f"\nTrajectory Length:")
        summary_text.append(f"  Mean: {tl['mean_length']:.4f}")
        summary_text.append(f"  Range: [{tl['min_length']:.3f}, {tl['max_length']:.3f}]")
        
        ax.text(0.05, 0.95, '\n'.join(summary_text), transform=ax.transAxes,
               fontfamily='monospace', fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def plot_geodesic_interpolation(self, fig, grid_spec, interpolation_result):
        """Plot geodesic interpolation results."""
        ax = fig.add_subplot(grid_spec)
        
        images = interpolation_result['interpolated_images']
        n_steps = len(images)
        
        # Create interpolation strip
        if images.dim() == 4:  # [N, C, H, W]
            strip_height = 64
            strip_width = n_steps * 64
            interpolation_strip = torch.zeros(3, strip_height, strip_width)
            
            for i, img in enumerate(images):
                interpolation_strip[:, :, i*64:(i+1)*64] = img
            
            strip_np = interpolation_strip.permute(1, 2, 0).numpy()
            strip_np = np.clip(strip_np, 0, 1)
            
            ax.imshow(strip_np)
            ax.set_title('Geodesic Interpolation')
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'Invalid interpolation format', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def plot_manifold_analysis(self, fig, grid_spec, manifold_result):
        """Plot manifold sampling analysis."""
        ax = fig.add_subplot(grid_spec)
        
        samples = manifold_result['samples']
        radii = sorted([float(k.split('_')[1]) for k in samples.keys()])
        
        # Create grid showing samples at different radii
        n_radii = len(radii)
        n_samples_per_radius = 4
        
        grid_image = torch.zeros(3, n_radii * 64, n_samples_per_radius * 64)
        
        for i, radius in enumerate(radii):
            key = f'radius_{radius}'
            images = samples[key]['images']
            
            for j in range(min(n_samples_per_radius, len(images))):
                img = images[j]
                if img.dim() == 4:
                    img = img[0]  # Remove batch dimension
                grid_image[:, i*64:(i+1)*64, j*64:(j+1)*64] = img
        
        grid_np = grid_image.permute(1, 2, 0).numpy()
        grid_np = np.clip(grid_np, 0, 1)
        
        ax.imshow(grid_np)
        ax.set_title('Manifold Structure Analysis')
        ax.axis('off')
        
        # Add radius labels
        for i, radius in enumerate(radii):
            ax.text(32, i*64 + 32, f'r={radius}', 
                   rotation=90, ha='center', va='center',
                   fontweight='bold', color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    def plot_analysis_summary(self, fig, grid_spec):
        """Plot overall analysis summary."""
        ax = fig.add_subplot(grid_spec)
        
        # Collect summary statistics
        summary_data = {
            'Generation Methods Tested': len(self.generation_results),
            'Successful Generations': sum(1 for r in self.generation_results.values() if r.get('success', False)),
            'Inference Analyses': len(self.inference_results),
            'Geodesic Analyses': len(self.geodesic_results)
        }
        
        # Add more specific stats
        if 'sequence_analysis' in self.inference_results:
            seq_analysis = self.inference_results['sequence_analysis']
            if 'trajectory_stats' in seq_analysis:
                cc = seq_analysis['trajectory_stats']['cycle_consistency']
                summary_data['Perfect Cycles'] = f"{cc['perfect_cycles']}/{cc['total_sequences']}"
        
        # Create summary table
        summary_text = "Analysis Summary\n" + "="*30 + "\n"
        for key, value in summary_data.items():
            summary_text += f"{key}: {value}\n"
        
        summary_text += f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontfamily='monospace', fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Analysis Summary', fontweight='bold', fontsize=14)
    
    def save_results(self, save_path: Path):
        """Save all analysis results to files."""
        print(f"💾 Saving analysis results...")
        
        # Prepare serializable results
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'latent_dim': getattr(self.model.config, 'latent_dim', 'Unknown'),
                'n_flows': getattr(self.model, 'n_flows', 0)
            },
            'generation_summary': {},
            'inference_summary': {},
            'geodesic_summary': {}
        }
        
        # Generation results summary
        for method, result in self.generation_results.items():
            if result.get('success', False):
                images = result['images']
                if images.dim() == 5:
                    images = images[:, 0]
                
                results['generation_summary'][method] = {
                    'num_samples': len(images),
                    'image_shape': list(images.shape[1:]),
                    'mean_pixel_value': images.mean().item(),
                    'value_range': [images.min().item(), images.max().item()]
                }
        
        # Inference results summary
        if 'sequence_analysis' in self.inference_results:
            seq_analysis = self.inference_results['sequence_analysis']
            if 'trajectory_stats' in seq_analysis:
                results['inference_summary'] = seq_analysis['trajectory_stats']
        
        # Geodesic results summary
        for analysis_type, result in self.geodesic_results.items():
            if result.get('success', False):
                results['geodesic_summary'][analysis_type] = 'completed'
        
        # Save to JSON
        with open(save_path / 'comprehensive_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   ✅ Results saved to {save_path / 'comprehensive_analysis_results.json'}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Comprehensive RlVAE Analysis Suite")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="dsprites", help="Dataset to use")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of samples for generation")
    parser.add_argument("--num-sequences", type=int, default=20, help="Number of sequences for inference")
    parser.add_argument("--output-dir", type=str, default="comprehensive_analysis", help="Output directory")
    
    # Analysis options
    parser.add_argument("--full-analysis", action="store_true", help="Run complete analysis suite")
    parser.add_argument("--generation-only", action="store_true", help="Run only generation analysis")
    parser.add_argument("--inference-only", action="store_true", help="Run only inference analysis")
    parser.add_argument("--geodesic-only", action="store_true", help="Run only geodesic analysis")
    
    args = parser.parse_args()
    
    # Setup
    setup_plotting()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize analyzer
    analyzer = RlVAEAnalyzer(args.model_path)
    
    # Load test data
    test_data = analyzer.load_test_data(args.dataset, subset_size=500)
    
    # Determine which analyses to run
    if args.full_analysis or not any([args.generation_only, args.inference_only, args.geodesic_only]):
        run_generation = run_inference = run_geodesic = True
    else:
        run_generation = args.generation_only or args.full_analysis
        run_inference = args.inference_only or args.full_analysis
        run_geodesic = args.geodesic_only or args.full_analysis
    
    # Run analyses
    fid_scores = None
    
    if run_generation:
        analyzer.run_generation_analysis(args.num_samples)
        fid_scores = analyzer.compute_fid_scores(test_data['images'])
    
    if run_inference:
        analyzer.run_inference_analysis(test_data, args.num_sequences)
    
    if run_geodesic:
        analyzer.run_geodesic_analysis()
    
    # Create comprehensive visualization
    analyzer.create_comprehensive_visualization(output_dir, fid_scores)
    
    # Save results
    analyzer.save_results(output_dir)
    
    # Print summary
    print(f"\n🎉 Comprehensive RlVAE Analysis Complete!")
    print(f"📁 Results saved to: {output_dir}")
    
    if fid_scores:
        best_method = min(fid_scores.items(), key=lambda x: x[1])
        print(f"🏆 Best FID Score: {best_method[0]} ({best_method[1]:.2f})")


if __name__ == "__main__":
    main() 