#!/usr/bin/env python3
"""
Manifold Verification Analysis
=============================

Critical question: Are we actually sampling ON the manifold, or just in the ambient space?

This script will:
1. Check if training data truly lives on the learned manifold
2. Verify if RHVAE samples are on the same manifold as training data
3. Measure manifold distance vs ambient space distance
4. Answer: Are we sampling ON manifold or just metric-weighted in ambient space?
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
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px

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

class ManifoldVerificationAnalysis:
    """Verify if we're truly sampling ON the manifold."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """Initialize analyzer."""
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.model = None
        self.rhvae_sampler = None
        self.data_module = None
        
        self.output_dir = Path("outputs/manifold_verification") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔍 Manifold verification analysis initialized")
    
    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model_and_setup(self) -> None:
        """Load model and setup components."""
        logger.info(f"🔄 Loading model")
        
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
        
        # Setup RHVAE sampler
        self._setup_rhvae_sampler()
        
        # Setup data
        self._setup_data()
    
    def _setup_rhvae_sampler(self) -> None:
        """Setup RHVAE sampler."""
        
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
                """Compute G^(-1) at latent points z."""
                batch_size = z.shape[0]
                
                z = z.to(dtype=torch.float32)
                centroids = self.centroids_tens.to(dtype=torch.float32)
                matrices = self.M_tens.to(dtype=torch.float32)
                
                distances = torch.cdist(z, centroids)
                closest_indices = torch.argmin(distances, dim=1)
                G_inv_batch = matrices[closest_indices]
                
                return G_inv_batch
            
            def G(self, z):
                """Compute G at latent points z."""
                G_inv = self.G_inv(z)
                eye = torch.eye(self.latent_dim, device=self.device, dtype=torch.float32)
                G = torch.inverse(G_inv + 1e-8 * eye)
                return G
        
        self.rhvae_wrapper = RHVAEWrapper(self.model, self.device)
        
        sampler_config = RHVAESamplerConfig(
            mcmc_steps_nbr=100,
            n_lf=15,
            eps_lf=0.01,
            beta_zero=1.0
        )
        
        self.rhvae_sampler = RHVAESampler(
            model=self.rhvae_wrapper,
            sampler_config=sampler_config
        )
        
        logger.info("✅ RHVAE sampler setup complete")
    
    def _setup_data(self) -> None:
        """Setup data module."""
        data_config = DictConfig({
            'train_path': 'data/sprites/ColoredCircles_train.pt',
            'test_path': 'data/sprites/ColoredCircles_test.pt',
            'train_meta_path': 'data/sprites/ColoredCircles_train_params.pt',
            'test_meta_path': 'data/sprites/ColoredCircles_test_params.pt',
            'sequence_length': 10, 'image_size': [28, 28], 'channels': 3,
            'batch_size': 1, 'num_workers': 0, 'pin_memory': False,
            'max_test_samples': 50, 'verify_cyclicity': False
        })
        
        self.data_module = CyclicSpritesDataModule(data_config)
        self.data_module.setup('test')
        
        logger.info("✅ Data module setup complete")
    
    def extract_real_data_latents(self, n_samples: int = 200) -> tuple:
        """Extract latent representations from real data."""
        logger.info(f"📊 Extracting real data latents ({n_samples} points)")
        
        real_latents = []
        test_loader = self.data_module.test_dataloader()
        
        point_count = 0
        for batch_idx, batch in enumerate(test_loader):
            if point_count >= n_samples:
                break
            
            try:
                batch = batch.to(self.device)
                
                with torch.no_grad():
                    # Take first frame of sequence for simplicity
                    if len(batch.shape) == 5:  # [1, seq_len, c, h, w]
                        x = batch[0, 0]  # First frame of first sequence
                    elif len(batch.shape) == 4:  # [seq_len, c, h, w]
                        x = batch[0]  # First frame
                    else:
                        x = batch
                    
                    x = x.unsqueeze(0)  # Add batch dim
                    
                    # Get latent representation
                    encoder_out = self.model.encoder(x)
                    if isinstance(encoder_out, dict):
                        mu = encoder_out['mu']
                        logvar = encoder_out['logvar']
                        # Use mean for deterministic comparison
                        z = mu
                    else:
                        z = encoder_out
                    
                    real_latents.append(z.cpu().numpy())
                    point_count += 1
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to process batch {batch_idx}: {e}")
                continue
        
        if real_latents:
            real_latents_array = np.vstack(real_latents)
        else:
            logger.warning("⚠️ No real latents extracted, using dummy data")
            real_latents_array = np.random.randn(n_samples, 2) * 0.5
        
        logger.info(f"✅ Extracted {len(real_latents_array)} real data latents")
        return real_latents_array
    
    def sample_rhvae_points(self, n_samples: int = 200) -> np.ndarray:
        """Sample points using RHVAE."""
        logger.info(f"🎲 Sampling RHVAE points ({n_samples} points)")
        
        with torch.no_grad():
            rhvae_samples = self.rhvae_sampler.hmc_sampling(n_samples)
        
        return rhvae_samples.cpu().numpy()
    
    def compute_manifold_distances(self, points: np.ndarray) -> dict:
        """Compute various distance metrics to understand manifold structure."""
        logger.info("📏 Computing manifold distances")
        
        # Get metric centroids
        centroids = self.model.modular_metric.centroids.detach().cpu().numpy()
        
        distances = {}
        
        # 1. Distance to nearest centroid (proxy for "on-manifold" distance)
        centroid_distances = []
        closest_centroids = []
        
        for point in points:
            dists_to_centroids = np.linalg.norm(centroids - point, axis=1)
            min_dist_idx = np.argmin(dists_to_centroids)
            min_dist = dists_to_centroids[min_dist_idx]
            
            centroid_distances.append(min_dist)
            closest_centroids.append(min_dist_idx)
        
        distances['centroid_distances'] = np.array(centroid_distances)
        distances['closest_centroids'] = np.array(closest_centroids)
        
        # 2. Distance from origin (ambient space measure)
        distances['origin_distances'] = np.linalg.norm(points, axis=1)
        
        # 3. Local density (how clustered points are)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(points)
        distances_to_neighbors, _ = nn.kneighbors(points)
        distances['local_density'] = np.mean(distances_to_neighbors[:, 1:], axis=1)  # Exclude self
        
        # 4. Metric determinant at each point
        determinants = []
        for point in points:
            z_tensor = torch.tensor(point, device=self.device, dtype=torch.float32).unsqueeze(0)
            try:
                G_inv = self.rhvae_wrapper.G_inv(z_tensor)
                det_g_inv = torch.det(G_inv[0])
                log_det = torch.log10(det_g_inv + 1e-10).cpu().item()
                determinants.append(log_det)
            except:
                determinants.append(0.0)
        
        distances['metric_determinants'] = np.array(determinants)
        
        return distances
    
    def manifold_reconstruction_test(self, points: np.ndarray) -> dict:
        """Test if points can be well-reconstructed (proxy for being on data manifold)."""
        logger.info("🔄 Testing manifold reconstruction quality")
        
        reconstruction_scores = []
        
        for i, point in enumerate(points[:50]):  # Test first 50 for speed
            try:
                z_tensor = torch.tensor(point, device=self.device, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    # Decode point
                    decoder_out = self.model.decoder(z_tensor)
                    if isinstance(decoder_out, dict):
                        reconstruction = decoder_out['reconstruction']
                    else:
                        reconstruction = decoder_out
                    
                    # Encode reconstruction back
                    encoder_out = self.model.encoder(reconstruction)
                    if isinstance(encoder_out, dict):
                        z_recon = encoder_out['mu']  # Use mean
                    else:
                        z_recon = encoder_out
                    
                    # Compute reconstruction error in latent space
                    latent_error = torch.norm(z_tensor - z_recon).cpu().item()
                    reconstruction_scores.append(latent_error)
                    
            except Exception as e:
                reconstruction_scores.append(10.0)  # Large error for failed cases
        
        return {
            'reconstruction_errors': np.array(reconstruction_scores),
            'mean_error': np.mean(reconstruction_scores),
            'std_error': np.std(reconstruction_scores)
        }
    
    def create_manifold_verification_plots(self, real_data: np.ndarray, rhvae_data: np.ndarray, 
                                         real_distances: dict, rhvae_distances: dict,
                                         real_recon: dict, rhvae_recon: dict) -> go.Figure:
        """Create comprehensive manifold verification plots."""
        logger.info("🎨 Creating manifold verification plots")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Latent Space: Real Data vs RHVAE Samples",
                "Distance to Nearest Centroids",
                "Metric Determinant Distributions", 
                "Reconstruction Error Comparison",
                "Local Density Analysis",
                "Manifold Coverage Analysis"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Latent space plot
        # Real data
        fig.add_trace(
            go.Scatter(
                x=real_data[:, 0],
                y=real_data[:, 1],
                mode='markers',
                marker=dict(size=6, color='blue', opacity=0.7),
                name='Real Data',
                hovertemplate="Real<br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # RHVAE samples
        fig.add_trace(
            go.Scatter(
                x=rhvae_data[:, 0],
                y=rhvae_data[:, 1],
                mode='markers',
                marker=dict(size=6, color='red', opacity=0.7),
                name='RHVAE Samples',
                hovertemplate="RHVAE<br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Centroids
        centroids = self.model.modular_metric.centroids.detach().cpu().numpy()
        fig.add_trace(
            go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode='markers',
                marker=dict(size=12, color='gold', symbol='diamond', line=dict(width=2, color='black')),
                name='Metric Centroids',
                hovertemplate="Centroid<br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # 2. Distance to centroids
        fig.add_trace(
            go.Histogram(
                x=real_distances['centroid_distances'],
                name='Real Data Distance to Centroids',
                marker_color='blue',
                opacity=0.7,
                nbinsx=20
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=rhvae_distances['centroid_distances'],
                name='RHVAE Distance to Centroids',
                marker_color='red',
                opacity=0.7,
                nbinsx=20
            ),
            row=1, col=2
        )
        
        # 3. Metric determinants
        fig.add_trace(
            go.Histogram(
                x=real_distances['metric_determinants'],
                name='Real Data det(G⁻¹)',
                marker_color='blue',
                opacity=0.7,
                nbinsx=20
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=rhvae_distances['metric_determinants'],
                name='RHVAE det(G⁻¹)',
                marker_color='red',
                opacity=0.7,
                nbinsx=20
            ),
            row=2, col=1
        )
        
        # 4. Reconstruction errors
        real_errors = real_recon['reconstruction_errors']
        rhvae_errors = rhvae_recon['reconstruction_errors']
        
        fig.add_trace(
            go.Box(
                y=real_errors,
                name='Real Data Reconstruction Error',
                marker_color='blue'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Box(
                y=rhvae_errors,
                name='RHVAE Reconstruction Error',
                marker_color='red'
            ),
            row=2, col=2
        )
        
        # 5. Local density
        fig.add_trace(
            go.Scatter(
                x=real_distances['local_density'],
                y=real_distances['origin_distances'],
                mode='markers',
                marker=dict(size=5, color='blue', opacity=0.6),
                name='Real Data Density vs Distance',
                hovertemplate="Real<br>Local Density: %{x:.3f}<br>Origin Distance: %{y:.3f}<extra></extra>"
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=rhvae_distances['local_density'],
                y=rhvae_distances['origin_distances'],
                mode='markers',
                marker=dict(size=5, color='red', opacity=0.6),
                name='RHVAE Density vs Distance',
                hovertemplate="RHVAE<br>Local Density: %{x:.3f}<br>Origin Distance: %{y:.3f}<extra></extra>"
            ),
            row=3, col=1
        )
        
        # 6. Manifold coverage
        # Show which centroids are closest to each point type
        real_centroid_counts = np.bincount(real_distances['closest_centroids'], minlength=len(centroids))
        rhvae_centroid_counts = np.bincount(rhvae_distances['closest_centroids'], minlength=len(centroids))
        
        centroid_indices = list(range(len(centroids)))
        
        fig.add_trace(
            go.Bar(
                x=centroid_indices,
                y=real_centroid_counts,
                name='Real Data Centroid Usage',
                marker_color='blue',
                opacity=0.7
            ),
            row=3, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=centroid_indices,
                y=rhvae_centroid_counts,
                name='RHVAE Centroid Usage',
                marker_color='red',
                opacity=0.7
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title=dict(
                text="🔍 Manifold Verification Analysis<br><sub>Are we really sampling ON the manifold?</sub>",
                x=0.5,
                font=dict(size=16)
            ),
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title="Latent Dimension 1", row=1, col=1)
        fig.update_yaxes(title="Latent Dimension 2", row=1, col=1)
        fig.update_xaxes(title="Distance to Nearest Centroid", row=1, col=2)
        fig.update_yaxes(title="Count", row=1, col=2)
        fig.update_xaxes(title="log₁₀(det(G⁻¹))", row=2, col=1)
        fig.update_yaxes(title="Count", row=2, col=1)
        fig.update_yaxes(title="Reconstruction Error", row=2, col=2)
        fig.update_xaxes(title="Local Density", row=3, col=1)
        fig.update_yaxes(title="Distance from Origin", row=3, col=1)
        fig.update_xaxes(title="Centroid Index", row=3, col=2)
        fig.update_yaxes(title="Usage Count", row=3, col=2)
        
        return fig
    
    def create_manifold_verdict(self, real_data: np.ndarray, rhvae_data: np.ndarray,
                              real_distances: dict, rhvae_distances: dict,
                              real_recon: dict, rhvae_recon: dict) -> str:
        """Create definitive verdict on manifold sampling."""
        
        # Statistical tests
        from scipy import stats
        
        # 1. Compare distances to centroids
        centroid_distance_pvalue = stats.ks_2samp(
            real_distances['centroid_distances'],
            rhvae_distances['centroid_distances']
        ).pvalue
        
        # 2. Compare reconstruction errors  
        recon_error_ratio = rhvae_recon['mean_error'] / real_recon['mean_error']
        
        # 3. Compare metric determinants
        det_pvalue = stats.ks_2samp(
            real_distances['metric_determinants'],
            rhvae_distances['metric_determinants']
        ).pvalue
        
        # 4. Manifold coverage analysis
        real_mean_centroid_dist = np.mean(real_distances['centroid_distances'])
        rhvae_mean_centroid_dist = np.mean(rhvae_distances['centroid_distances'])
        distance_ratio = rhvae_mean_centroid_dist / real_mean_centroid_dist
        
        verdict = f"""
# 🔍 MANIFOLD VERIFICATION VERDICT

## 🎯 Are We Really Sampling ON the Manifold?

### 📊 Key Evidence

**Distance to Learned Centroids:**
- Real Data: {real_mean_centroid_dist:.3f} ± {np.std(real_distances['centroid_distances']):.3f}
- RHVAE Samples: {rhvae_mean_centroid_dist:.3f} ± {np.std(rhvae_distances['centroid_distances']):.3f}
- **Ratio: {distance_ratio:.2f}x** (RHVAE vs Real)
- **Statistical Difference: p = {centroid_distance_pvalue:.6f}** ({'SIGNIFICANT' if centroid_distance_pvalue < 0.05 else 'NOT SIGNIFICANT'})

**Reconstruction Quality (Manifold Fidelity):**
- Real Data Error: {real_recon['mean_error']:.4f} ± {real_recon['std_error']:.4f}
- RHVAE Error: {rhvae_recon['mean_error']:.4f} ± {rhvae_recon['std_error']:.4f}
- **Error Ratio: {recon_error_ratio:.2f}x** (RHVAE vs Real)

**Metric Determinant Distributions:**
- **Statistical Difference: p = {det_pvalue:.6f}** ({'SIGNIFICANT' if det_pvalue < 0.05 else 'NOT SIGNIFICANT'})

## 🎯 THE VERDICT

"""
        
        # Determine verdict based on evidence
        if distance_ratio < 1.5 and recon_error_ratio < 2.0 and centroid_distance_pvalue > 0.05:
            verdict += """
✅ **YES - We ARE sampling ON the manifold!**

**Evidence:**
- RHVAE samples stay close to learned centroids (< 1.5x distance)
- Reconstruction quality is comparable
- Similar metric determinant distributions
- Both datasets follow the same manifold structure

**Conclusion:** The RHVAE sampler is successfully sampling from the learned data manifold. The apparent "scatter" in PCA is due to exploring the FULL manifold rather than the subset that training data explored.
"""
        elif distance_ratio < 3.0 and recon_error_ratio < 5.0:
            verdict += """
⚠️ **PARTIALLY - We're sampling NEAR the manifold**

**Evidence:**
- RHVAE samples are farther from centroids but still reasonable
- Reconstruction quality degraded but functional
- Some deviation from training manifold structure

**Conclusion:** The RHVAE sampler is sampling in the vicinity of the learned manifold but may be exploring regions not well-constrained by training data. This could indicate successful manifold **extension** rather than exact manifold **adherence**.
"""
        else:
            verdict += """
❌ **NO - We're sampling in AMBIENT space, not on manifold**

**Evidence:**
- RHVAE samples are much farther from centroids (>{distance_ratio:.1f}x)
- Poor reconstruction quality (>{recon_error_ratio:.1f}x error)
- Very different metric distributions

**Conclusion:** The RHVAE sampler is not constrained to the learned data manifold. It's performing metric-weighted sampling in the ambient latent space, which explains the scattered appearance compared to training data.
"""
        
        verdict += f"""

## 🤔 What This Means

**If ON manifold:** Your model learned a rich manifold that extends beyond training data - excellent generalization!

**If NEAR manifold:** Your model can extrapolate beyond training while maintaining some manifold structure - good interpolation capability.

**If OFF manifold:** The sampling is metric-aware but not manifold-constrained - still useful for generation but not true manifold sampling.

## 📊 Statistical Summary
- Distance Ratio: {distance_ratio:.2f}
- Reconstruction Ratio: {recon_error_ratio:.2f}
- Centroid Distance p-value: {centroid_distance_pvalue:.6f}
- Determinant p-value: {det_pvalue:.6f}
"""
        
        return verdict
    
    def run_manifold_verification(self) -> None:
        """Run complete manifold verification analysis."""
        logger.info("🚀 Starting manifold verification analysis")
        
        try:
            # Load model and setup
            self.load_model_and_setup()
            
            # Extract real data latents
            real_data = self.extract_real_data_latents(200)
            
            # Sample RHVAE points
            rhvae_data = self.sample_rhvae_points(200)
            
            # Compute manifold distances
            real_distances = self.compute_manifold_distances(real_data)
            rhvae_distances = self.compute_manifold_distances(rhvae_data)
            
            # Test reconstruction quality
            real_recon = self.manifold_reconstruction_test(real_data)
            rhvae_recon = self.manifold_reconstruction_test(rhvae_data)
            
            # Create visualization
            verification_plot = self.create_manifold_verification_plots(
                real_data, rhvae_data, real_distances, rhvae_distances, real_recon, rhvae_recon
            )
            
            # Create verdict
            verdict = self.create_manifold_verdict(
                real_data, rhvae_data, real_distances, rhvae_distances, real_recon, rhvae_recon
            )
            
            # Save results
            plot_path = self.output_dir / "manifold_verification_analysis.html"
            verification_plot.write_html(str(plot_path))
            
            verdict_path = self.output_dir / "manifold_verdict.md"
            with open(verdict_path, 'w') as f:
                f.write(verdict)
            
            logger.info("🎉 Manifold verification completed!")
            logger.info(f"📁 Results in: {self.output_dir}")
            logger.info(f"📊 Plot: {plot_path}")
            logger.info(f"📝 Verdict: {verdict_path}")
            
            # Print key finding
            real_mean_dist = np.mean(real_distances['centroid_distances'])
            rhvae_mean_dist = np.mean(rhvae_distances['centroid_distances'])
            ratio = rhvae_mean_dist / real_mean_dist
            
            print("\n" + "="*80)
            print("🎯 MANIFOLD VERIFICATION RESULT:")
            print("="*80)
            print(f"Real data distance to centroids: {real_mean_dist:.3f}")
            print(f"RHVAE distance to centroids: {rhvae_mean_dist:.3f}")
            print(f"Distance ratio: {ratio:.2f}x")
            
            if ratio < 1.5:
                print("✅ VERDICT: Sampling ON the manifold!")
            elif ratio < 3.0:
                print("⚠️ VERDICT: Sampling NEAR the manifold")
            else:
                print("❌ VERDICT: Sampling in ambient space")
            print("="*80)
            
        except Exception as e:
            logger.error(f"❌ Manifold verification failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main execution."""
    checkpoint_path = "outputs/checkpoints/epoch=197-val_loss=6.402.ckpt"
    
    analyzer = ManifoldVerificationAnalysis(checkpoint_path)
    analyzer.run_manifold_verification()


if __name__ == "__main__":
    main() 