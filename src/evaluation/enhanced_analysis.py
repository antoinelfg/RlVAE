"""
Enhanced Analysis Module for RlVAE

This module provides comprehensive analysis capabilities including:
- Enhanced generation visualization with multiple sampling methods
- Advanced inference analysis with latent space trajectories
- Geodesic and Riemannian sampling for manifold exploration
- FID score evaluation
- Comprehensive reporting and visualization
"""

import os
import json
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import wandb
from tqdm import tqdm
import logging

try:
    from ..models.modular_rlvae import ModularRiemannianFlowVAE as RlVAE
    from ..data.datasets import get_dataloader
    from .fid_scorer import FIDScorer
    from ..visualizations.manager import VisualizationManager
    from ..visualizations.manifold import ManifoldVisualizations as ManifoldVisualizer
except ImportError:
    # Fallback for direct imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.models.modular_rlvae import ModularRiemannianFlowVAE as RlVAE
    from src.data.datasets import get_dataloader
    from src.evaluation.fid_scorer import FIDScorer
    from src.visualizations.manager import VisualizationManager
    from src.visualizations.manifold import ManifoldVisualizations as ManifoldVisualizer

logger = logging.getLogger(__name__)

class EnhancedAnalyzer:
    """Enhanced analysis suite for RlVAE models with advanced visualizations."""
    
    def __init__(self, model: RlVAE, device: str = "cuda", 
                 output_dir: str = "enhanced_analysis_outputs"):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.fid_scorer = FIDScorer(device=device)
        # Pass config to VisualizationManager (required argument)
        config = getattr(model, 'config', {})
        self.viz_manager = VisualizationManager(model, device, config)
        self.manifold_viz = ManifoldVisualizer(model, device, config)
        
        # Analysis results storage
        self.results = {}
        
    def run_comprehensive_analysis(self, dataloader, 
                                 num_samples: int = 1000,
                                 num_cycles: int = 50,
                                 geodesic_steps: int = 20,
                                 log_to_wandb: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive analysis including generation, inference, and geodesic exploration.
        
        Args:
            dataloader: DataLoader for evaluation
            num_samples: Number of samples for generation analysis
            num_cycles: Number of cycles for inference analysis
            geodesic_steps: Number of steps for geodesic interpolation
            log_to_wandb: Whether to log results to wandb
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting comprehensive RlVAE analysis...")
        
        # 1. Generation Analysis
        logger.info("Running generation analysis...")
        gen_results = self.analyze_generation(dataloader, num_samples)
        self.results['generation'] = gen_results
        
        # 2. Inference Analysis
        logger.info("Running inference analysis...")
        inf_results = self.analyze_inference(dataloader, num_cycles)
        self.results['inference'] = inf_results
        
        # 3. Geodesic Analysis
        logger.info("Running geodesic analysis...")
        geo_results = self.analyze_geodesic_sampling(geodesic_steps)
        self.results['geodesic'] = geo_results
        
        # 4. Create master visualizations
        logger.info("Creating master visualizations...")
        master_viz = self.create_master_visualizations()
        self.results['master_visualizations'] = master_viz
        
        # 5. Save comprehensive report
        self.save_comprehensive_report()
        
        # 6. Log to wandb if requested
        if log_to_wandb:
            self.log_to_wandb()
            
        logger.info("Comprehensive analysis completed!")
        return self.results
    
    def analyze_generation(self, dataloader, num_samples: int = 1000) -> Dict[str, Any]:
        """Enhanced generation analysis with multiple sampling methods."""
        self.model.eval()
        
        # Generate samples using different methods
        sampling_methods = ['random', 'grid', 'gaussian', 'uniform']
        generated_samples = {}
        fid_scores = {}
        
        with torch.no_grad():
            for method in sampling_methods:
                logger.info(f"Generating samples using {method} sampling...")
                
                if method == 'random':
                    z = torch.randn(num_samples, self.model.latent_dim, device=self.device)
                elif method == 'grid':
                    z = self._create_grid_samples(num_samples)
                elif method == 'gaussian':
                    z = torch.randn(num_samples, self.model.latent_dim, device=self.device) * 0.5
                elif method == 'uniform':
                    z = torch.rand(num_samples, self.model.latent_dim, device=self.device) * 2 - 1
                
                # Generate images
                generated = self.model.decode(z)
                generated_samples[method] = generated.cpu()
                
                # Calculate FID score
                fid_score = self.fid_scorer.calculate_fid(generated, dataloader)
                fid_scores[method] = fid_score
                logger.info(f"FID score ({method}): {fid_score:.2f}")
        
        # Create generation visualizations
        gen_viz = self._create_generation_visualizations(generated_samples, fid_scores)
        
        return {
            'generated_samples': generated_samples,
            'fid_scores': fid_scores,
            'visualizations': gen_viz
        }
    
    def analyze_inference(self, dataloader, num_cycles: int = 50) -> Dict[str, Any]:
        """Enhanced inference analysis with latent space trajectories."""
        self.model.eval()
        
        # Get real data samples
        real_data = []
        latent_trajectories = []
        cycle_consistencies = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc="Inference Analysis")):
                if batch_idx >= num_cycles:
                    break
                    
                data = data.to(self.device)
                
                # Encode to get latent representation
                mu, logvar = self.model.encode(data)
                z = self.model.reparameterize(mu, logvar)
                
                # Create cycle through latent space
                cycle_z = self._create_latent_cycle(z)
                latent_trajectories.append(cycle_z.cpu())
                
                # Decode cycle to check consistency
                cycle_reconstructed = self.model.decode(cycle_z)
                cycle_consistency = torch.mean((data - cycle_reconstructed) ** 2).item()
                cycle_consistencies.append(cycle_consistency)
                
                real_data.append(data.cpu())
        
        # Create inference visualizations
        inf_viz = self._create_inference_visualizations(
            real_data, latent_trajectories, cycle_consistencies
        )
        
        return {
            'real_data': real_data,
            'latent_trajectories': latent_trajectories,
            'cycle_consistencies': cycle_consistencies,
            'mean_cycle_consistency': np.mean(cycle_consistencies),
            'visualizations': inf_viz
        }
    
    def analyze_geodesic_sampling(self, geodesic_steps: int = 20) -> Dict[str, Any]:
        """Geodesic sampling analysis for manifold exploration."""
        self.model.eval()
        
        # Create geodesic paths
        geodesic_paths = []
        manifold_samples = []
        
        with torch.no_grad():
            # Create multiple geodesic paths
            for i in range(5):  # 5 different paths
                # Random start and end points
                z_start = torch.randn(1, self.model.latent_dim, device=self.device)
                z_end = torch.randn(1, self.model.latent_dim, device=self.device)
                
                # Interpolate along geodesic
                path = self._interpolate_geodesic(z_start, z_end, geodesic_steps)
                geodesic_paths.append(path.cpu())
                
                # Generate samples along path
                samples = self.model.decode(path)
                manifold_samples.append(samples.cpu())
        
        # Create geodesic visualizations
        geo_viz = self._create_geodesic_visualizations(geodesic_paths, manifold_samples)
        
        return {
            'geodesic_paths': geodesic_paths,
            'manifold_samples': manifold_samples,
            'visualizations': geo_viz
        }
    
    def create_master_visualizations(self) -> Dict[str, Any]:
        """Create master visualizations combining all analyses."""
        # Create comprehensive figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RlVAE Comprehensive Analysis', fontsize=16, fontweight='bold')
        
        # 1. FID Scores Comparison
        if 'generation' in self.results:
            fid_scores = self.results['generation']['fid_scores']
            methods = list(fid_scores.keys())
            scores = list(fid_scores.values())
            
            axes[0, 0].bar(methods, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            axes[0, 0].set_title('FID Scores by Sampling Method')
            axes[0, 0].set_ylabel('FID Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(scores):
                axes[0, 0].text(i, v + max(scores) * 0.01, f'{v:.1f}', 
                              ha='center', va='bottom', fontweight='bold')
        
        # 2. Cycle Consistency Distribution
        if 'inference' in self.results:
            consistencies = self.results['inference']['cycle_consistencies']
            axes[0, 1].hist(consistencies, bins=20, alpha=0.7, color='#FF6B6B')
            axes[0, 1].axvline(np.mean(consistencies), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(consistencies):.4f}')
            axes[0, 1].set_title('Cycle Consistency Distribution')
            axes[0, 1].set_xlabel('MSE')
            axes[0, 1].legend()
        
        # 3. Latent Space Visualization
        if 'inference' in self.results:
            # Take first trajectory and plot in 2D
            trajectory = self.results['inference']['latent_trajectories'][0]
            if trajectory.shape[1] > 2:
                # Use PCA for dimensionality reduction
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                trajectory_2d = pca.fit_transform(trajectory.numpy())
            else:
                trajectory_2d = trajectory.numpy()
            
            axes[0, 2].plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 'b-', alpha=0.7)
            axes[0, 2].scatter(trajectory_2d[0, 0], trajectory_2d[0, 1], 
                             c='green', s=100, label='Start', zorder=5)
            axes[0, 2].scatter(trajectory_2d[-1, 0], trajectory_2d[-1, 1], 
                             c='red', s=100, label='End', zorder=5)
            axes[0, 2].set_title('Latent Space Trajectory')
            axes[0, 2].legend()
        
        # 4. Generated Samples Grid
        if 'generation' in self.results:
            samples = self.results['generation']['generated_samples']['random'][:16]
            grid = self._create_image_grid(samples, 4, 4)
            axes[1, 0].imshow(grid, cmap='gray')
            axes[1, 0].set_title('Generated Samples (Random)')
            axes[1, 0].axis('off')
        
        # 5. Real vs Reconstructed
        if 'inference' in self.results:
            real = self.results['inference']['real_data'][0][:8]
            # Reconstruct
            with torch.no_grad():
                mu, logvar = self.model.encode(real.to(self.device))
                z = self.model.reparameterize(mu, logvar)
                reconstructed = self.model.decode(z).cpu()
            
            comparison = torch.cat([real, reconstructed], dim=0)
            grid = self._create_image_grid(comparison, 4, 4)
            axes[1, 1].imshow(grid, cmap='gray')
            axes[1, 1].set_title('Real vs Reconstructed')
            axes[1, 1].axis('off')
        
        # 6. Geodesic Interpolation
        if 'geodesic' in self.results:
            geodesic_samples = self.results['geodesic']['manifold_samples'][0]
            grid = self._create_image_grid(geodesic_samples, 1, geodesic_samples.shape[0])
            axes[1, 2].imshow(grid, cmap='gray')
            axes[1, 2].set_title('Geodesic Interpolation')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save master visualization
        master_path = self.output_dir / 'master_analysis.png'
        plt.savefig(master_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {'master_visualization_path': str(master_path)}
    
    def save_comprehensive_report(self):
        """Save comprehensive analysis report as JSON."""
        report = {
            'analysis_timestamp': str(datetime.datetime.now()),
            'model_config': {
                'latent_dim': self.model.latent_dim,
                'input_dim': self.model.input_dim,
                'hidden_dims': self.model.hidden_dims
            },
            'results': {}
        }
        
        # Extract key metrics
        if 'generation' in self.results:
            report['results']['generation'] = {
                'fid_scores': self.results['generation']['fid_scores'],
                'best_fid_method': min(self.results['generation']['fid_scores'].items(), 
                                     key=lambda x: x[1])[0]
            }
        
        if 'inference' in self.results:
            report['results']['inference'] = {
                'mean_cycle_consistency': self.results['inference']['mean_cycle_consistency'],
                'cycle_consistency_std': np.std(self.results['inference']['cycle_consistencies'])
            }
        
        # Save report
        report_path = self.output_dir / 'comprehensive_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive report saved to {report_path}")
    
    def log_to_wandb(self):
        """Log analysis results to wandb."""
        if not wandb.run:
            logger.warning("No active wandb run found. Skipping wandb logging.")
            return
        
        # Log metrics
        if 'generation' in self.results:
            for method, fid in self.results['generation']['fid_scores'].items():
                wandb.log({f'fid_{method}': fid})
        
        if 'inference' in self.results:
            wandb.log({
                'mean_cycle_consistency': self.results['inference']['mean_cycle_consistency'],
                'cycle_consistency_std': np.std(self.results['inference']['cycle_consistencies'])
            })
        
        # Log visualizations
        if 'master_visualizations' in self.results:
            master_path = self.results['master_visualizations']['master_visualization_path']
            if os.path.exists(master_path):
                wandb.log({"master_analysis": wandb.Image(master_path)})
        
        logger.info("Results logged to wandb successfully")
    
    def _create_grid_samples(self, num_samples: int) -> torch.Tensor:
        """Create grid samples in latent space."""
        grid_size = int(np.sqrt(num_samples))
        x = torch.linspace(-2, 2, grid_size, device=self.device)
        y = torch.linspace(-2, 2, grid_size, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # For higher dimensional latent spaces, use first two dimensions
        z = torch.zeros(grid_size * grid_size, self.model.latent_dim, device=self.device)
        z[:, 0] = xx.flatten()
        z[:, 1] = yy.flatten()
        
        return z
    
    def _create_latent_cycle(self, z: torch.Tensor) -> torch.Tensor:
        """Create a cycle in latent space."""
        # Simple circular trajectory
        t = torch.linspace(0, 2 * np.pi, z.shape[0], device=self.device)
        radius = torch.norm(z, dim=1, keepdim=True)
        
        # Create circular path
        cycle_z = z.clone()
        cycle_z[:, 0] = radius.squeeze() * torch.cos(t)
        cycle_z[:, 1] = radius.squeeze() * torch.sin(t)
        
        return cycle_z
    
    def _interpolate_geodesic(self, z_start: torch.Tensor, z_end: torch.Tensor, 
                            steps: int) -> torch.Tensor:
        """Interpolate along geodesic between two points."""
        t = torch.linspace(0, 1, steps, device=self.device).unsqueeze(1)
        return z_start + t * (z_end - z_start)
    
    def _create_image_grid(self, images: torch.Tensor, rows: int, cols: int) -> np.ndarray:
        """Create a grid of images for visualization."""
        images = images.squeeze()  # Remove channel dimension if present
        if images.dim() == 3:
            images = images.unsqueeze(1)  # Add channel dimension
        
        # Reshape to grid
        grid = images[:rows * cols].view(rows, cols, *images.shape[1:])
        
        # Convert to numpy and normalize
        grid = grid.cpu().numpy()
        grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
        
        # Combine into single image
        if grid.shape[-1] == 1:  # Grayscale
            combined = np.concatenate([np.concatenate(row, axis=1) for row in grid], axis=0)
        else:  # RGB
            combined = np.concatenate([np.concatenate(row, axis=1) for row in grid], axis=0)
        
        return combined
    
    def _create_generation_visualizations(self, generated_samples: Dict[str, torch.Tensor], 
                                        fid_scores: Dict[str, float]) -> Dict[str, str]:
        """Create generation visualizations."""
        viz_paths = {}
        
        # Create sample grids for each method
        for method, samples in generated_samples.items():
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            fig.suptitle(f'Generated Samples - {method.capitalize()} (FID: {fid_scores[method]:.2f})')
            
            for i in range(16):
                row, col = i // 4, i % 4
                axes[row, col].imshow(samples[i].squeeze(), cmap='gray')
                axes[row, col].axis('off')
            
            plt.tight_layout()
            path = self.output_dir / f'generation_{method}.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            viz_paths[method] = str(path)
        
        return viz_paths
    
    def _create_inference_visualizations(self, real_data: List[torch.Tensor], 
                                       latent_trajectories: List[torch.Tensor],
                                       cycle_consistencies: List[float]) -> Dict[str, str]:
        """Create inference visualizations."""
        viz_paths = {}
        
        # Latent trajectory visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot first trajectory
        trajectory = latent_trajectories[0]
        if trajectory.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            trajectory_2d = pca.fit_transform(trajectory.numpy())
        else:
            trajectory_2d = trajectory.numpy()
        
        axes[0, 0].plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 'b-', alpha=0.7)
        axes[0, 0].scatter(trajectory_2d[0, 0], trajectory_2d[0, 1], 
                          c='green', s=100, label='Start')
        axes[0, 0].scatter(trajectory_2d[-1, 0], trajectory_2d[-1, 1], 
                          c='red', s=100, label='End')
        axes[0, 0].set_title('Latent Space Trajectory')
        axes[0, 0].legend()
        
        # Cycle consistency histogram
        axes[0, 1].hist(cycle_consistencies, bins=20, alpha=0.7, color='#FF6B6B')
        axes[0, 1].axvline(np.mean(cycle_consistencies), color='red', linestyle='--',
                          label=f'Mean: {np.mean(cycle_consistencies):.4f}')
        axes[0, 1].set_title('Cycle Consistency Distribution')
        axes[0, 1].set_xlabel('MSE')
        axes[0, 1].legend()
        
        # Real vs reconstructed comparison
        real = real_data[0][:8]
        with torch.no_grad():
            mu, logvar = self.model.encode(real.to(self.device))
            z = self.model.reparameterize(mu, logvar)
            reconstructed = self.model.decode(z).cpu()
        
        comparison = torch.cat([real, reconstructed], dim=0)
        grid = self._create_image_grid(comparison, 4, 4)
        axes[1, 0].imshow(grid, cmap='gray')
        axes[1, 0].set_title('Real vs Reconstructed')
        axes[1, 0].axis('off')
        
        # Consistency over time
        axes[1, 1].plot(cycle_consistencies[:20], 'b-', alpha=0.7)
        axes[1, 1].set_title('Cycle Consistency Over Time')
        axes[1, 1].set_xlabel('Cycle Index')
        axes[1, 1].set_ylabel('MSE')
        
        plt.tight_layout()
        path = self.output_dir / 'inference_analysis.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        viz_paths['main'] = str(path)
        
        return viz_paths
    
    def _create_geodesic_visualizations(self, geodesic_paths: List[torch.Tensor],
                                      manifold_samples: List[torch.Tensor]) -> Dict[str, str]:
        """Create geodesic visualizations."""
        viz_paths = {}
        
        # Create geodesic interpolation visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Geodesic Interpolation Analysis')
        
        # Plot geodesic paths in latent space
        for i, path in enumerate(geodesic_paths[:3]):
            if path.shape[1] > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                path_2d = pca.fit_transform(path.numpy())
            else:
                path_2d = path.numpy()
            
            axes[0, i].plot(path_2d[:, 0], path_2d[:, 1], 'b-', alpha=0.7)
            axes[0, i].scatter(path_2d[0, 0], path_2d[0, 1], c='green', s=100, label='Start')
            axes[0, i].scatter(path_2d[-1, 0], path_2d[-1, 1], c='red', s=100, label='End')
            axes[0, i].set_title(f'Geodesic Path {i+1}')
            axes[0, i].legend()
        
        # Show interpolated samples
        for i, samples in enumerate(manifold_samples[:3]):
            grid = self._create_image_grid(samples, 1, samples.shape[0])
            axes[1, i].imshow(grid, cmap='gray')
            axes[1, i].set_title(f'Interpolation {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        path = self.output_dir / 'geodesic_analysis.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        viz_paths['main'] = str(path)
        
        return viz_paths 