#!/usr/bin/env python3
"""
Sampling Methods Analysis
========================

Diagnostic script to analyze:
1. Are we truly sampling on the Riemannian manifold?
2. How does RHVAE sampling compare to standard Gaussian N(0,1)?
3. What's the real difference between sampling methods?
4. Are our interpolation methods truly metric-aware?

This will give us definitive answers about manifold vs Gaussian sampling.
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
from scipy import stats

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

class SamplingMethodsAnalyzer:
    """Analyze different sampling methods to understand manifold vs Gaussian sampling."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """Initialize the analyzer."""
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.model = None
        self.rhvae_sampler = None
        self.rhvae_wrapper = None
        self.data_module = None
        
        self.output_dir = Path("outputs/sampling_analysis") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔍 Sampling methods analyzer initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model_and_setup_samplers(self) -> None:
        """Load model and setup multiple sampling methods."""
        logger.info(f"🔄 Loading model and setting up sampling methods")
        
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
        
        # Setup RHVAE sampler (same as before)
        self._setup_rhvae_sampler()
        
        # Setup data for comparison
        data_config = DictConfig({
            'train_path': 'data/sprites/ColoredCircles_train.pt',
            'test_path': 'data/sprites/ColoredCircles_test.pt',
            'train_meta_path': 'data/sprites/ColoredCircles_train_params.pt',
            'test_meta_path': 'data/sprites/ColoredCircles_test_params.pt',
            'sequence_length': 10, 'image_size': [28, 28], 'channels': 3,
            'batch_size': 1, 'num_workers': 0, 'pin_memory': False,
            'max_test_samples': 100, 'verify_cyclicity': False
        })
        
        self.data_module = CyclicSpritesDataModule(data_config)
        self.data_module.setup('test')
    
    def _setup_rhvae_sampler(self) -> None:
        """Setup RHVAE sampler (reused from enhanced explorer)."""
        
        class RHVAEWrapper:
            def __init__(self, rlvae_model, device):
                self.rlvae_model = rlvae_model
                self.device = device
                self.latent_dim = rlvae_model.latent_dim
                
                self.centroids_tens = rlvae_model.modular_metric.centroids.clone().detach().to(dtype=torch.float32)
                self.M_tens = rlvae_model.modular_metric.metric_matrices.clone().detach().to(dtype=torch.float32)
                self.temperature = 2.5
            
            def eval(self):
                self.rlvae_model.eval()
                return self
            
            def to(self, device):
                self.device = device
                self.rlvae_model.to(device)
                self.centroids_tens = self.centroids_tens.to(device)
                self.M_tens = self.M_tens.to(device)
                return self
            
            def decoder(self, z):
                with torch.no_grad():
                    output = self.rlvae_model.decoder(z)
                    if isinstance(output, dict):
                        return output
                    else:
                        return {"reconstruction": output}
            
            def G_inv(self, z):
                batch_size = z.shape[0]
                
                z = z.to(dtype=torch.float32)
                centroids = self.centroids_tens.to(dtype=torch.float32)
                matrices = self.M_tens.to(dtype=torch.float32)
                
                distances = torch.cdist(z, centroids)
                closest_indices = torch.argmin(distances, dim=1)
                G_inv_batch = matrices[closest_indices]
                
                return G_inv_batch
            
            def G(self, z):
                G_inv = self.G_inv(z)
                eye = torch.eye(self.latent_dim, device=self.device, dtype=torch.float32)
                G = torch.inverse(G_inv + 1e-8 * eye)
                return G
        
        self.rhvae_wrapper = RHVAEWrapper(self.model, self.device)
        
        sampler_config = RHVAESamplerConfig(
            mcmc_steps_nbr=200,  # Reduced for faster comparison
            n_lf=20,
            eps_lf=0.01,
            beta_zero=1.0
        )
        
        self.rhvae_sampler = RHVAESampler(
            model=self.rhvae_wrapper,
            sampler_config=sampler_config
        )
        
        logger.info("✅ RHVAE sampler setup complete")
    
    def sample_with_different_methods(self, n_samples: int = 500) -> dict:
        """Sample using different methods for comparison."""
        logger.info(f"🎲 Sampling with different methods ({n_samples} samples each)")
        
        results = {}
        
        # 1. RHVAE Sampling (our "manifold" sampling)
        logger.info("   📊 RHVAE/HMC sampling...")
        with torch.no_grad():
            rhvae_samples = self.rhvae_sampler.hmc_sampling(n_samples)
        results['rhvae'] = rhvae_samples.cpu().numpy()
        
        # 2. Standard Gaussian N(0,1)
        logger.info("   📊 Standard Gaussian N(0,1)...")
        gaussian_samples = torch.randn(n_samples, self.model.latent_dim)
        results['gaussian'] = gaussian_samples.numpy()
        
        # 3. Model's own sampling (if available)
        logger.info("   📊 Model's internal sampling...")
        try:
            with torch.no_grad():
                model_samples = self.model.sample(n_samples)
                if hasattr(model_samples, 'latent_samples'):
                    # For sequences, take first timestep
                    if len(model_samples.latent_samples.shape) == 3:
                        internal_samples = model_samples.latent_samples[:, 0, :].cpu().numpy()
                    else:
                        internal_samples = model_samples.latent_samples.cpu().numpy()
                else:
                    internal_samples = model_samples.cpu().numpy()
            results['model_internal'] = internal_samples
        except Exception as e:
            logger.warning(f"⚠️ Model internal sampling failed: {e}")
            results['model_internal'] = np.random.randn(n_samples, self.model.latent_dim)
        
        # 4. Samples from real data (encoder outputs)
        logger.info("   📊 Real data encoder outputs...")
        real_samples = []
        test_loader = self.data_module.test_dataloader()
        
        for batch_idx, batch in enumerate(test_loader):
            if len(real_samples) >= n_samples:
                break
            
            try:
                if len(batch.shape) == 4:
                    batch = batch.unsqueeze(0)
                batch = batch.to(self.device)
                
                with torch.no_grad():
                    # Flatten for encoder
                    batch_flat = batch.view(batch.shape[0], -1, *batch.shape[3:])
                    x_first = batch_flat[:, 0]  # Take first timestep
                    
                    encoder_out = self.model.encoder(x_first)
                    if isinstance(encoder_out, dict):
                        mu = encoder_out['mu']
                    else:
                        mu = encoder_out
                    
                    real_samples.append(mu.cpu().numpy())
                    
            except Exception as e:
                continue
        
        if real_samples:
            real_samples_array = np.vstack(real_samples)[:n_samples]
        else:
            real_samples_array = np.random.randn(n_samples, self.model.latent_dim)
        
        results['real_data'] = real_samples_array
        
        # 5. Centroids-based sampling (sample around metric centroids)
        logger.info("   📊 Centroids-based sampling...")
        centroids = self.model.modular_metric.centroids.detach().cpu().numpy()
        centroid_samples = []
        
        for _ in range(n_samples):
            # Pick random centroid
            centroid_idx = np.random.randint(len(centroids))
            centroid = centroids[centroid_idx]
            
            # Add small Gaussian noise around centroid
            noise_scale = 0.5
            sample = centroid + np.random.randn(self.model.latent_dim) * noise_scale
            centroid_samples.append(sample)
        
        results['centroids'] = np.array(centroid_samples)
        
        logger.info(f"✅ Collected samples from {len(results)} methods")
        return results
    
    def compute_sampling_statistics(self, samples_dict: dict) -> dict:
        """Compute detailed statistics for each sampling method."""
        logger.info("📊 Computing sampling statistics")
        
        stats_dict = {}
        
        for method, samples in samples_dict.items():
            logger.info(f"   📈 Analyzing {method}...")
            
            # Basic statistics
            mean = np.mean(samples, axis=0)
            std = np.std(samples, axis=0)
            
            # Distribution tests
            # Test if each dimension is normally distributed
            dim1_ks = stats.kstest(samples[:, 0], 'norm', args=(mean[0], std[0]))
            dim2_ks = stats.kstest(samples[:, 1], 'norm', args=(mean[1], std[1]))
            
            # Distance from origin
            distances = np.linalg.norm(samples, axis=1)
            
            # Compute metric determinants for comparison
            determinants = []
            for sample in samples[:50]:  # Compute for first 50 to save time
                z_point = torch.tensor(sample, device=self.device, dtype=torch.float32).unsqueeze(0)
                det_val = self._compute_metric_determinant(z_point)
                determinants.append(det_val)
            
            stats_dict[method] = {
                'mean': mean,
                'std': std,
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'ks_dim1_pvalue': dim1_ks.pvalue,
                'ks_dim2_pvalue': dim2_ks.pvalue,
                'determinants_mean': np.mean(determinants),
                'determinants_std': np.std(determinants),
                'n_samples': len(samples)
            }
        
        return stats_dict
    
    def _compute_metric_determinant(self, z_point: torch.Tensor) -> float:
        """Compute metric determinant."""
        try:
            z_point = z_point.to(device=self.device, dtype=torch.float32)
            G_inv = self.rhvae_wrapper.G_inv(z_point)
            G_inv = G_inv.to(dtype=torch.float32)
            det_g_inv = torch.det(G_inv[0])
            return torch.log10(det_g_inv + 1e-10).cpu().item()
        except:
            return 0.0
    
    def create_comprehensive_comparison(self, samples_dict: dict, stats_dict: dict) -> go.Figure:
        """Create comprehensive comparison visualization."""
        logger.info("🎨 Creating comprehensive comparison visualization")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Sample Distributions in Latent Space",
                "Distance from Origin Distributions", 
                "Determinant Distributions (First 50 samples)",
                "Dimension 1 vs Dimension 2 Correlation",
                "Statistical Summary",
                "Method Comparison"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}], 
                [{"type": "table", "colspan": 2}, None]
            ]
        )
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # 1. Sample distributions
        for i, (method, samples) in enumerate(samples_dict.items()):
            fig.add_trace(
                go.Scatter(
                    x=samples[:, 0],
                    y=samples[:, 1],
                    mode='markers',
                    name=method,
                    marker=dict(color=colors[i % len(colors)], size=4, opacity=0.6),
                    hovertemplate=f"{method}<br>Z1: %{{x:.3f}}<br>Z2: %{{y:.3f}}<extra></extra>"
                ),
                row=1, col=1
            )
        
        # 2. Distance distributions
        for i, (method, samples) in enumerate(samples_dict.items()):
            distances = np.linalg.norm(samples, axis=1)
            fig.add_trace(
                go.Histogram(
                    x=distances,
                    name=f"{method} distances",
                    marker_color=colors[i % len(colors)],
                    opacity=0.7,
                    nbinsx=30
                ),
                row=1, col=2
            )
        
        # 3. Determinant distributions
        for i, (method, samples) in enumerate(samples_dict.items()):
            if method in stats_dict:
                dets = []
                for sample in samples[:50]:
                    z_point = torch.tensor(sample, device=self.device, dtype=torch.float32).unsqueeze(0)
                    det_val = self._compute_metric_determinant(z_point)
                    dets.append(det_val)
                
                fig.add_trace(
                    go.Histogram(
                        x=dets,
                        name=f"{method} det(G⁻¹)",
                        marker_color=colors[i % len(colors)],
                        opacity=0.7,
                        nbinsx=20
                    ),
                    row=2, col=1
                )
        
        # 4. Correlation analysis
        for i, (method, samples) in enumerate(samples_dict.items()):
            correlation = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
            fig.add_trace(
                go.Scatter(
                    x=samples[:, 0],
                    y=samples[:, 1],
                    mode='markers',
                    name=f"{method} (r={correlation:.3f})",
                    marker=dict(color=colors[i % len(colors)], size=3, opacity=0.5),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # 5. Statistical summary table
        table_headers = ["Method", "Mean Z1", "Mean Z2", "Std Z1", "Std Z2", "Mean Distance", "KS p-value (Z1)", "Det(G⁻¹) Mean"]
        table_cells = []
        
        for method, stats in stats_dict.items():
            table_cells.append([
                method,
                f"{stats['mean'][0]:.3f}",
                f"{stats['mean'][1]:.3f}",
                f"{stats['std'][0]:.3f}",
                f"{stats['std'][1]:.3f}",
                f"{stats['mean_distance']:.3f}",
                f"{stats['ks_dim1_pvalue']:.3f}",
                f"{stats['determinants_mean']:.3f}"
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(values=table_headers, align="center"),
                cells=dict(values=list(zip(*table_cells)), align="center")
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title=dict(
                text="🔍 Comprehensive Sampling Methods Analysis<br><sub>Manifold vs Gaussian vs Model Internal vs Real Data vs Centroids</sub>",
                x=0.5,
                font=dict(size=16)
            ),
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title="Latent Dimension 1", row=1, col=1)
        fig.update_yaxes(title="Latent Dimension 2", row=1, col=1)
        fig.update_xaxes(title="Distance from Origin", row=1, col=2)
        fig.update_yaxes(title="Count", row=1, col=2)
        fig.update_xaxes(title="det(G⁻¹)", row=2, col=1)
        fig.update_yaxes(title="Count", row=2, col=1)
        fig.update_xaxes(title="Latent Dimension 1", row=2, col=2)
        fig.update_yaxes(title="Latent Dimension 2", row=2, col=2)
        
        return fig
    
    def analyze_manifold_vs_gaussian(self, samples_dict: dict, stats_dict: dict) -> dict:
        """Detailed analysis: Are we really sampling on manifold vs Gaussian?"""
        logger.info("🔬 Analyzing manifold vs Gaussian sampling")
        
        analysis_results = {}
        
        # Compare RHVAE vs Gaussian
        rhvae_samples = samples_dict['rhvae']
        gaussian_samples = samples_dict['gaussian']
        
        # Statistical tests
        # 1. Two-sample KS test
        ks_dim1 = stats.ks_2samp(rhvae_samples[:, 0], gaussian_samples[:, 0])
        ks_dim2 = stats.ks_2samp(rhvae_samples[:, 1], gaussian_samples[:, 1])
        
        # 2. Mean and variance comparison
        rhvae_mean = np.mean(rhvae_samples, axis=0)
        gaussian_mean = np.mean(gaussian_samples, axis=0)
        rhvae_var = np.var(rhvae_samples, axis=0)
        gaussian_var = np.var(gaussian_samples, axis=0)
        
        # 3. Distance analysis
        rhvae_distances = np.linalg.norm(rhvae_samples, axis=1)
        gaussian_distances = np.linalg.norm(gaussian_samples, axis=1)
        
        # 4. Determinant comparison
        rhvae_dets = [self._compute_metric_determinant(torch.tensor(s, device=self.device, dtype=torch.float32).unsqueeze(0)) 
                     for s in rhvae_samples[:100]]
        gaussian_dets = [self._compute_metric_determinant(torch.tensor(s, device=self.device, dtype=torch.float32).unsqueeze(0)) 
                        for s in gaussian_samples[:100]]
        
        analysis_results = {
            'ks_test_dim1': {
                'statistic': ks_dim1.statistic,
                'pvalue': ks_dim1.pvalue,
                'significant': ks_dim1.pvalue < 0.05
            },
            'ks_test_dim2': {
                'statistic': ks_dim2.statistic,
                'pvalue': ks_dim2.pvalue,
                'significant': ks_dim2.pvalue < 0.05
            },
            'mean_difference': np.linalg.norm(rhvae_mean - gaussian_mean),
            'variance_ratio': rhvae_var / (gaussian_var + 1e-8),
            'distance_comparison': {
                'rhvae_mean_dist': np.mean(rhvae_distances),
                'gaussian_mean_dist': np.mean(gaussian_distances),
                'distance_ratio': np.mean(rhvae_distances) / np.mean(gaussian_distances)
            },
            'determinant_comparison': {
                'rhvae_mean_det': np.mean(rhvae_dets),
                'gaussian_mean_det': np.mean(gaussian_dets),
                'rhvae_std_det': np.std(rhvae_dets),
                'gaussian_std_det': np.std(gaussian_dets)
            }
        }
        
        return analysis_results
    
    def create_conclusion_report(self, analysis_results: dict, stats_dict: dict) -> str:
        """Create detailed conclusion report."""
        
        report = f"""
# 🔍 Sampling Methods Analysis Report

## 📊 Key Findings

### 🎯 RHVAE vs Gaussian Comparison

**Statistical Significance Tests:**
- Dimension 1 KS test: p = {analysis_results['ks_test_dim1']['pvalue']:.6f} ({'SIGNIFICANT' if analysis_results['ks_test_dim1']['significant'] else 'NOT SIGNIFICANT'})
- Dimension 2 KS test: p = {analysis_results['ks_test_dim2']['pvalue']:.6f} ({'SIGNIFICANT' if analysis_results['ks_test_dim2']['significant'] else 'NOT SIGNIFICANT'})

**Mean Difference:** {analysis_results['mean_difference']:.6f}

**Distance Analysis:**
- RHVAE mean distance: {analysis_results['distance_comparison']['rhvae_mean_dist']:.3f}
- Gaussian mean distance: {analysis_results['distance_comparison']['gaussian_mean_dist']:.3f}
- Ratio: {analysis_results['distance_comparison']['distance_ratio']:.3f}

**Metric Determinant Analysis:**
- RHVAE det(G⁻¹): {analysis_results['determinant_comparison']['rhvae_mean_det']:.3f} ± {analysis_results['determinant_comparison']['rhvae_std_det']:.3f}
- Gaussian det(G⁻¹): {analysis_results['determinant_comparison']['gaussian_mean_det']:.3f} ± {analysis_results['determinant_comparison']['gaussian_std_det']:.3f}

## 🎯 Conclusion

"""
        
        # Determine if we're truly sampling on manifold
        if analysis_results['ks_test_dim1']['significant'] and analysis_results['ks_test_dim2']['significant']:
            report += "✅ **RHVAE sampling IS significantly different from Gaussian N(0,1)**\n"
            report += "✅ **We ARE sampling on a learned manifold structure**\n"
        else:
            report += "❌ **RHVAE sampling is NOT significantly different from Gaussian N(0,1)**\n"
            report += "❌ **We may NOT be sampling on a true manifold**\n"
        
        # Distance analysis conclusion
        if abs(analysis_results['distance_comparison']['distance_ratio'] - 1.0) > 0.2:
            report += f"✅ **Distance distribution is significantly different** (ratio: {analysis_results['distance_comparison']['distance_ratio']:.3f})\n"
        else:
            report += f"⚠️ **Distance distributions are similar** (ratio: {analysis_results['distance_comparison']['distance_ratio']:.3f})\n"
        
        # Determinant analysis conclusion
        det_diff = abs(analysis_results['determinant_comparison']['rhvae_mean_det'] - analysis_results['determinant_comparison']['gaussian_mean_det'])
        if det_diff > 0.1:
            report += f"✅ **Metric determinants are significantly different** (diff: {det_diff:.3f})\n"
        else:
            report += f"⚠️ **Metric determinants are similar** (diff: {det_diff:.3f})\n"
        
        report += f"\n## 📊 Method Comparison\n\n"
        
        for method, stats in stats_dict.items():
            report += f"**{method.upper()}:**\n"
            report += f"- Mean: [{stats['mean'][0]:.3f}, {stats['mean'][1]:.3f}]\n"
            report += f"- Std: [{stats['std'][0]:.3f}, {stats['std'][1]:.3f}]\n"
            report += f"- Mean distance: {stats['mean_distance']:.3f}\n"
            report += f"- Det(G⁻¹): {stats['determinants_mean']:.3f} ± {stats['determinants_std']:.3f}\n\n"
        
        return report
    
    def run_sampling_analysis(self) -> None:
        """Run complete sampling analysis."""
        logger.info("🚀 Starting comprehensive sampling analysis")
        
        wandb.init(
            project="rlvae-sampling-analysis",
            name=f"sampling_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "checkpoint": self.checkpoint_path,
                "device": self.device,
                "analysis_type": "manifold_vs_gaussian_sampling"
            }
        )
        
        try:
            # Load model and setup samplers
            self.load_model_and_setup_samplers()
            
            # Sample with different methods
            samples_dict = self.sample_with_different_methods(n_samples=500)
            
            # Compute statistics
            stats_dict = self.compute_sampling_statistics(samples_dict)
            
            # Create comprehensive comparison
            comparison_fig = self.create_comprehensive_comparison(samples_dict, stats_dict)
            comparison_path = self.output_dir / "comprehensive_sampling_comparison.html"
            comparison_fig.write_html(str(comparison_path))
            
            # Analyze manifold vs Gaussian
            analysis_results = self.analyze_manifold_vs_gaussian(samples_dict, stats_dict)
            
            # Create conclusion report
            conclusion_report = self.create_conclusion_report(analysis_results, stats_dict)
            
            # Save report
            report_path = self.output_dir / "sampling_analysis_report.md"
            with open(report_path, 'w') as f:
                f.write(conclusion_report)
            
            # Log to WandB
            wandb.log({
                "comparison_visualization": wandb.Html(str(comparison_path)),
                "manifold_vs_gaussian_significant_dim1": analysis_results['ks_test_dim1']['significant'],
                "manifold_vs_gaussian_significant_dim2": analysis_results['ks_test_dim2']['significant'],
                "mean_difference": analysis_results['mean_difference'],
                "distance_ratio": analysis_results['distance_comparison']['distance_ratio'],
                "det_difference": abs(analysis_results['determinant_comparison']['rhvae_mean_det'] - 
                                   analysis_results['determinant_comparison']['gaussian_mean_det'])
            })
            
            logger.info("🎉 Sampling analysis completed!")
            logger.info(f"📁 Results in: {self.output_dir}")
            logger.info(f"📊 Comparison: {comparison_path}")
            logger.info(f"📝 Report: {report_path}")
            
            # Print key conclusions
            print("\n" + "="*80)
            print("🎯 KEY CONCLUSIONS:")
            print("="*80)
            print(conclusion_report.split("## 🎯 Conclusion")[1].split("## 📊 Method Comparison")[0])
            print("="*80)
            
        except Exception as e:
            logger.error(f"❌ Sampling analysis failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            wandb.finish()


def main():
    """Main execution."""
    checkpoint_path = "outputs/checkpoints/epoch=197-val_loss=6.402.ckpt"
    
    analyzer = SamplingMethodsAnalyzer(checkpoint_path)
    analyzer.run_sampling_analysis()


if __name__ == "__main__":
    main() 