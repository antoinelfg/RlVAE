#!/usr/bin/env python3
"""
Metric-Aware Landscape Sampler
==============================

Uses the trained metric tensor to:
1. Sample points according to the Riemannian manifold density
2. Compute det(G⁻¹) for each sampled point  
3. Create a true density-based landscape profile
4. Compare with grid-based landscape

This gives the REAL metric-aware density profile of your latent space!
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from omegaconf import DictConfig
import logging
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import wandb

# Setup paths
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir.parent / "src"
lib_src_dir = src_dir / "lib" / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(lib_src_dir) not in sys.path:
    sys.path.insert(0, str(lib_src_dir))

# Project imports
from models.modular_rlvae import ModularRiemannianFlowVAE
from data.cyclic_dataset import CyclicSpritesDataModule

# RHVAE sampler imports
from pythae.samplers.manifold_sampler.rhvae_sampler import RHVAESampler
from pythae.samplers.manifold_sampler.rhvae_sampler_config import RHVAESamplerConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetricAwareLandscapeSampler:
    """Sample points according to metric-aware density and create true landscape profile."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """Initialize the sampler."""
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.model = None
        self.rhvae_sampler = None
        
        self.output_dir = Path("outputs/metric_aware_landscape") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎯 Metric-aware landscape sampler initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model_and_setup_sampler(self) -> None:
        """Load model and setup the RHVAE-style sampler."""
        logger.info(f"🔄 Loading model and setting up metric-aware sampler")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model_hparams = checkpoint['hyper_parameters']['model']
        
        # Create config
        config = DictConfig(model_hparams)
        config.pretrained = {'encoder_path': None, 'decoder_path': None, 'metric_path': None}
        
        # Create and load model
        self.model = ModularRiemannianFlowVAE(config)
        
        # Load state dict
        state_dict = checkpoint['state_dict']
        clean_state_dict = {k.replace('model.', '') if k.startswith('model.') else k: v.to(self.device) 
                           for k, v in state_dict.items()}
        
        # Resize metric tensor
        for name, param in clean_state_dict.items():
            if 'modular_metric.centroids' in name:
                self.model.modular_metric.centroids = torch.nn.Parameter(torch.zeros_like(param))
            elif 'modular_metric.metric_matrices' in name:
                self.model.modular_metric.metric_matrices = torch.nn.Parameter(torch.zeros_like(param))
        
        self.model.load_state_dict(clean_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("✅ Model loaded successfully")
        
        # Setup RHVAE-style sampler
        self._setup_rhvae_compatible_sampler()
    
    def _setup_rhvae_compatible_sampler(self) -> None:
        """Setup RHVAE-compatible sampler using our metric tensor."""
        logger.info("🎯 Setting up RHVAE-compatible sampler")
        
        # Create a wrapper that makes our model compatible with RHVAE sampler
        class RHVAECompatibleWrapper:
            def __init__(self, rlvae_model, device):
                self.rlvae_model = rlvae_model
                self.device = device
                self.latent_dim = rlvae_model.latent_dim
                
                # Extract metric tensor components with consistent dtype
                self.centroids_tens = rlvae_model.modular_metric.centroids.clone().detach().to(dtype=torch.float32)
                self.M_tens = rlvae_model.modular_metric.metric_matrices.clone().detach().to(dtype=torch.float32)
                
                # Set temperature (you can adjust this)
                self.temperature = 3.0
                
                logger.info(f"📊 RHVAE wrapper created:")
                logger.info(f"   - Centroids: {self.centroids_tens.shape}")
                logger.info(f"   - Metric matrices: {self.M_tens.shape}")
                logger.info(f"   - Temperature: {self.temperature}")
            
            def eval(self):
                """Required by BaseSampler."""
                self.rlvae_model.eval()
                return self
            
            def to(self, device):
                """Required by BaseSampler."""
                self.device = device
                self.rlvae_model.to(device)
                self.centroids_tens = self.centroids_tens.to(device)
                self.M_tens = self.M_tens.to(device)
                return self
            
            def decoder(self, z):
                """Decoder wrapper."""
                with torch.no_grad():
                    output = self.rlvae_model.decoder(z)
                    if isinstance(output, dict):
                        return output
                    else:
                        return {"reconstruction": output}
            
            def G_inv(self, z):
                """Compute G^(-1) at latent points z."""
                batch_size = z.shape[0]
                
                # Ensure consistent dtypes
                z = z.to(dtype=torch.float32)
                centroids = self.centroids_tens.to(dtype=torch.float32)
                matrices = self.M_tens.to(dtype=torch.float32)
                
                # Find closest centroids
                distances = torch.cdist(z, centroids)  # [batch_size, n_centroids]
                closest_indices = torch.argmin(distances, dim=1)  # [batch_size]
                
                # Get corresponding metric matrices
                G_inv_batch = matrices[closest_indices]  # [batch_size, latent_dim, latent_dim]
                
                return G_inv_batch
            
            def G(self, z):
                """Compute G at latent points z."""
                G_inv = self.G_inv(z)
                # Compute inverse using batch inverse with consistent dtype
                eye = torch.eye(self.latent_dim, device=self.device, dtype=torch.float32)
                G = torch.inverse(G_inv + 1e-6 * eye)
                return G
        
        # Create wrapper
        self.rhvae_wrapper = RHVAECompatibleWrapper(self.model, self.device)
        
        # Create sampler config
        sampler_config = RHVAESamplerConfig(
            mcmc_steps_nbr=200,  # More steps for better sampling
            n_lf=20,             # More leapfrog steps
            eps_lf=0.01,         # Smaller step size for stability
            beta_zero=1.0
        )
        
        # Create RHVAE sampler
        self.rhvae_sampler = RHVAESampler(
            model=self.rhvae_wrapper,
            sampler_config=sampler_config
        )
        
        logger.info("✅ RHVAE-compatible sampler created")
    
    def sample_metric_aware_points(self, n_samples: int = 2000) -> tuple:
        """Sample points according to metric-aware density."""
        logger.info(f"🎲 Sampling {n_samples} points according to metric-aware density")
        
        # Use RHVAE sampler to get metric-aware samples
        with torch.no_grad():
            sampled_z = self.rhvae_sampler.hmc_sampling(n_samples)
        
        logger.info(f"✅ Generated {len(sampled_z)} metric-aware samples")
        logger.info(f"   Sample range: Z1=[{sampled_z[:, 0].min():.3f}, {sampled_z[:, 0].max():.3f}], Z2=[{sampled_z[:, 1].min():.3f}, {sampled_z[:, 1].max():.3f}]")
        
        # Compute det(G^-1) for each sampled point
        determinants = []
        for i in range(len(sampled_z)):
            z_point = sampled_z[i:i+1]  # Keep batch dimension
            det_g_inv = self._compute_metric_determinant_precise(z_point)
            determinants.append(det_g_inv)
            
            if (i + 1) % 500 == 0:
                logger.info(f"   Computed determinants for {i+1}/{len(sampled_z)} points")
        
        determinants = np.array(determinants)
        
        logger.info(f"✅ Computed determinants:")
        logger.info(f"   Range: [{determinants.min():.3f}, {determinants.max():.3f}]")
        logger.info(f"   Mean: {determinants.mean():.3f} ± {determinants.std():.3f}")
        
        return sampled_z.cpu().numpy(), determinants
    
    def _compute_metric_determinant_precise(self, z_point: torch.Tensor) -> float:
        """Compute metric determinant with high precision."""
        try:
            # Ensure z_point is float32 and on correct device
            z_point = z_point.to(device=self.device, dtype=torch.float32)
            
            # Use the RHVAE wrapper's G_inv method
            G_inv = self.rhvae_wrapper.G_inv(z_point)  # [1, 2, 2]
            
            # Ensure G_inv is float32
            G_inv = G_inv.to(dtype=torch.float32)
            
            det_g_inv = torch.det(G_inv[0])  # Determinant of the 2x2 matrix
            
            # Return log10(det(G^-1))
            return torch.log10(det_g_inv + 1e-10).cpu().item()
            
        except Exception as e:
            logger.warning(f"⚠️ Error computing determinant: {e}")
            return 0.0
    
    def create_grid_based_landscape(self, sampled_points: np.ndarray, resolution: int = 50) -> tuple:
        """Create grid-based landscape for comparison."""
        logger.info("🗺️ Creating grid-based landscape for comparison")
        
        # Create grid based on sampled points range
        x_min, x_max = sampled_points[:, 0].min() - 0.5, sampled_points[:, 0].max() + 0.5
        y_min, y_max = sampled_points[:, 1].min() - 0.5, sampled_points[:, 1].max() + 0.5
        
        x_grid = np.linspace(x_min, x_max, resolution)
        y_grid = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Compute determinants on grid
        Z = np.zeros_like(X)
        total_points = resolution * resolution
        
        for i in range(resolution):
            for j in range(resolution):
                z_point = torch.tensor([[X[i, j], Y[i, j]]], device=self.device, dtype=torch.float32)
                Z[i, j] = self._compute_metric_determinant_precise(z_point)
            
            if (i + 1) % 10 == 0:
                logger.info(f"   Grid computation: {(i+1)*resolution}/{total_points} points")
        
        logger.info("✅ Grid-based landscape created")
        return X, Y, Z
    
    def create_comprehensive_comparison_visualization(self, sampled_points: np.ndarray, sampled_dets: np.ndarray, 
                                                     grid_X: np.ndarray, grid_Y: np.ndarray, grid_Z: np.ndarray) -> None:
        """Create comprehensive comparison visualization."""
        logger.info("🎨 Creating comprehensive comparison visualization")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "🎲 Metric-Aware Sampled Points (Density-Based)",
                "🗺️ Grid-Based Landscape (Regular Grid)", 
                "📊 Determinant Distribution Comparison",
                "🎯 Combined View: Samples + Grid Landscape"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Metric-aware sampled points
        fig.add_trace(
            go.Scatter(
                x=sampled_points[:, 0],
                y=sampled_points[:, 1],
                mode='markers',
                marker=dict(
                    size=4,
                    color=sampled_dets,
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title="det(G⁻¹)", x=0.48)
                ),
                name='Metric-Aware Samples',
                hovertemplate="Z1: %{x:.3f}<br>Z2: %{y:.3f}<br>det(G⁻¹): %{marker.color:.3f}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # 2. Grid-based contour
        fig.add_trace(
            go.Contour(
                x=grid_X[0, :],
                y=grid_Y[:, 0], 
                z=grid_Z,
                colorscale='plasma',
                showscale=True,
                colorbar=dict(title="det(G⁻¹)", x=1.02),
                name='Grid Landscape'
            ),
            row=1, col=2
        )
        
        # 3. Distribution comparison
        fig.add_trace(
            go.Histogram(
                x=sampled_dets,
                nbinsx=50,
                name='Sampled Points',
                opacity=0.7,
                marker_color='blue'
            ),
            row=2, col=1
        )
        
        grid_dets_flat = grid_Z.flatten()
        fig.add_trace(
            go.Histogram(
                x=grid_dets_flat,
                nbinsx=50,
                name='Grid Points',
                opacity=0.7,
                marker_color='red'
            ),
            row=2, col=1
        )
        
        # 4. Combined view
        # Add grid as background
        fig.add_trace(
            go.Contour(
                x=grid_X[0, :],
                y=grid_Y[:, 0],
                z=grid_Z,
                colorscale='plasma',
                opacity=0.4,
                showscale=False,
                name='Grid Background'
            ),
            row=2, col=2
        )
        
        # Add sampled points on top
        fig.add_trace(
            go.Scatter(
                x=sampled_points[:, 0],
                y=sampled_points[:, 1],
                mode='markers',
                marker=dict(
                    size=6,
                    color=sampled_dets,
                    colorscale='viridis',
                    showscale=False,
                    line=dict(width=1, color='white')
                ),
                name='Metric Samples',
                hovertemplate="Sampled<br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<br>det(G⁻¹): %{marker.color:.3f}<extra></extra>"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title=dict(
                text="🎯 Metric-Aware Landscape vs Grid-Based Landscape",
                x=0.5,
                font=dict(size=16)
            ),
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title="Latent Dimension 1", row=1, col=1)
        fig.update_yaxes(title="Latent Dimension 2", row=1, col=1)
        fig.update_xaxes(title="Latent Dimension 1", row=1, col=2)
        fig.update_yaxes(title="Latent Dimension 2", row=1, col=2)
        fig.update_xaxes(title="det(G⁻¹)", row=2, col=1)
        fig.update_yaxes(title="Count", row=2, col=1)
        fig.update_xaxes(title="Latent Dimension 1", row=2, col=2)
        fig.update_yaxes(title="Latent Dimension 2", row=2, col=2)
        
        # Save visualization
        comparison_path = self.output_dir / "metric_aware_vs_grid_comparison.html"
        fig.write_html(str(comparison_path))
        
        logger.info(f"✅ Comparison visualization saved to {comparison_path}")
        return fig
    
    def create_density_analysis(self, sampled_points: np.ndarray, sampled_dets: np.ndarray) -> None:
        """Create detailed density analysis."""
        logger.info("📊 Creating detailed density analysis")
        
        # Compute statistics
        stats = {
            'n_samples': len(sampled_points),
            'det_mean': sampled_dets.mean(),
            'det_std': sampled_dets.std(), 
            'det_min': sampled_dets.min(),
            'det_max': sampled_dets.max(),
            'det_range': sampled_dets.max() - sampled_dets.min()
        }
        
        # Find high and low density regions
        high_det_threshold = np.percentile(sampled_dets, 90)
        low_det_threshold = np.percentile(sampled_dets, 10)
        
        high_det_mask = sampled_dets >= high_det_threshold
        low_det_mask = sampled_dets <= low_det_threshold
        
        high_det_points = sampled_points[high_det_mask]
        low_det_points = sampled_points[low_det_mask]
        
        # Create detailed analysis plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "🔥 High det(G⁻¹) Regions (Top 10%)",
                "❄️ Low det(G⁻¹) Regions (Bottom 10%)",
                "📈 det(G⁻¹) vs Z1 Coordinate", 
                "📈 det(G⁻¹) vs Z2 Coordinate"
            ]
        )
        
        # High density regions
        fig.add_trace(
            go.Scatter(
                x=high_det_points[:, 0],
                y=high_det_points[:, 1],
                mode='markers',
                marker=dict(size=8, color='red', opacity=0.7),
                name=f'High det(G⁻¹) (n={len(high_det_points)})',
                hovertemplate="High density<br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Low density regions  
        fig.add_trace(
            go.Scatter(
                x=low_det_points[:, 0],
                y=low_det_points[:, 1], 
                mode='markers',
                marker=dict(size=8, color='blue', opacity=0.7),
                name=f'Low det(G⁻¹) (n={len(low_det_points)})',
                hovertemplate="Low density<br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<extra></extra>"
            ),
            row=1, col=2
        )
        
        # Correlations
        fig.add_trace(
            go.Scatter(
                x=sampled_points[:, 0],
                y=sampled_dets,
                mode='markers',
                marker=dict(size=4, color=sampled_dets, colorscale='viridis', opacity=0.6),
                name='det(G⁻¹) vs Z1',
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=sampled_points[:, 1],
                y=sampled_dets,
                mode='markers', 
                marker=dict(size=4, color=sampled_dets, colorscale='viridis', opacity=0.6),
                name='det(G⁻¹) vs Z2',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title=dict(
                text=f"📊 Detailed Density Analysis (n={stats['n_samples']} samples)",
                x=0.5
            ),
            annotations=[
                dict(
                    text=f"<b>Statistics:</b><br>Mean: {stats['det_mean']:.3f}<br>Std: {stats['det_std']:.3f}<br>Range: [{stats['det_min']:.3f}, {stats['det_max']:.3f}]",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="black", borderwidth=1
                )
            ]
        )
        
        # Update axes
        fig.update_xaxes(title="Z1", row=1, col=1)
        fig.update_yaxes(title="Z2", row=1, col=1) 
        fig.update_xaxes(title="Z1", row=1, col=2)
        fig.update_yaxes(title="Z2", row=1, col=2)
        fig.update_xaxes(title="Z1", row=2, col=1)
        fig.update_yaxes(title="det(G⁻¹)", row=2, col=1)
        fig.update_xaxes(title="Z2", row=2, col=2)
        fig.update_yaxes(title="det(G⁻¹)", row=2, col=2)
        
        # Save analysis
        analysis_path = self.output_dir / "detailed_density_analysis.html"
        fig.write_html(str(analysis_path))
        
        logger.info(f"✅ Detailed analysis saved to {analysis_path}")
        
        return stats
    
    def create_comprehensive_index(self, stats: dict) -> None:
        """Create comprehensive index page."""
        index_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>🎯 Metric-Aware Landscape Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }}
        h1 {{ color: #667eea; text-align: center; font-size: 2.5em; margin-bottom: 30px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
        .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; border-left: 4px solid #667eea; }}
        .stat-number {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .stat-label {{ color: #666; margin-top: 10px; }}
        .btn {{ display: inline-block; padding: 15px 30px; margin: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 25px; font-weight: bold; }}
        .section {{ margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 10px; }}
        .highlight {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 20px; border-radius: 10px; margin: 20px 0; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Metric-Aware Landscape Analysis</h1>
        
        <div class="highlight">
            <h2>🎲 Riemannian Sampling Results</h2>
            <p>Generated {stats['n_samples']} points according to the learned metric density!</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{stats['n_samples']}</div>
                <div class="stat-label">Metric-Aware Samples</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['det_mean']:.3f}</div>
                <div class="stat-label">Mean det(G⁻¹)</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['det_range']:.3f}</div>
                <div class="stat-label">det(G⁻¹) Range</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['det_std']:.3f}</div>
                <div class="stat-label">det(G⁻¹) Std</div>
            </div>
        </div>
        
        <div class="section">
            <h2>🎨 Visualizations</h2>
            <a href="metric_aware_vs_grid_comparison.html" class="btn">🎯 Metric vs Grid Comparison</a>
            <a href="detailed_density_analysis.html" class="btn">📊 Detailed Density Analysis</a>
        </div>
        
        <div class="section">
            <h2>🔬 What This Shows</h2>
            <ul>
                <li><strong>🎲 Metric-Aware Sampling:</strong> Points sampled according to the actual Riemannian density learned by your model</li>
                <li><strong>🗺️ Grid vs Density:</strong> Comparison between uniform grid and density-based sampling</li>
                <li><strong>🎯 True Landscape Profile:</strong> Real metric determinant distribution from trained manifold</li>
                <li><strong>📊 High/Low Density Regions:</strong> Identification of metric tensor "hot spots" and "cold spots"</li>
            </ul>
        </div>
    </div>
</body>
</html>
        """
        
        index_path = self.output_dir / "index.html"
        with open(index_path, 'w') as f:
            f.write(index_html)
        
        logger.info(f"✅ Index page created at {index_path}")
    
    def run_metric_aware_analysis(self) -> None:
        """Run the complete metric-aware landscape analysis."""
        logger.info("🚀 Starting metric-aware landscape analysis")
        
        wandb.init(
            project="rlvae-metric-aware-landscape",
            name=f"metric_aware_landscape_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "checkpoint": self.checkpoint_path,
                "device": self.device,
                "analysis_type": "metric_aware_density_sampling"
            }
        )
        
        try:
            # Load model and setup sampler
            self.load_model_and_setup_sampler()
            
            # Sample points according to metric-aware density
            sampled_points, sampled_dets = self.sample_metric_aware_points(n_samples=2000)
            
            # Create grid-based landscape for comparison
            grid_X, grid_Y, grid_Z = self.create_grid_based_landscape(sampled_points, resolution=40)
            
            # Create comprehensive comparison
            comparison_fig = self.create_comprehensive_comparison_visualization(
                sampled_points, sampled_dets, grid_X, grid_Y, grid_Z
            )
            
            # Create detailed density analysis
            stats = self.create_density_analysis(sampled_points, sampled_dets)
            
            # Create index
            self.create_comprehensive_index(stats)
            
            # Log to WandB
            wandb.log({
                "comparison": wandb.Html(str(self.output_dir / "metric_aware_vs_grid_comparison.html")),
                "density_analysis": wandb.Html(str(self.output_dir / "detailed_density_analysis.html")),
                "n_samples": len(sampled_points),
                "det_mean": stats['det_mean'],
                "det_std": stats['det_std'],
                "det_range": stats['det_range']
            })
            
            logger.info("🎉 Metric-aware landscape analysis completed!")
            logger.info(f"📁 Results in: {self.output_dir}")
            logger.info(f"🌐 Open: {self.output_dir}/index.html")
            logger.info("🎯 This shows the TRUE density profile of your metric manifold!")
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            wandb.finish()


def main():
    """Main execution."""
    checkpoint_path = "outputs/checkpoints/epoch=197-val_loss=6.402.ckpt"
    
    sampler = MetricAwareLandscapeSampler(checkpoint_path)
    sampler.run_metric_aware_analysis()


if __name__ == "__main__":
    main() 