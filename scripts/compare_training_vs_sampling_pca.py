#!/usr/bin/env python3
"""
Training vs Sampling PCA Comparison
===================================

Investigate the discrepancy between:
1. Training trajectories PCA (clean, structured)
2. RHVAE sampling PCA (noisy, widespread)

This will help us understand if there's a mismatch in the learned vs sampled manifold.
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

class TrainingVsSamplingAnalyzer:
    """Compare training trajectories vs RHVAE sampling patterns."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """Initialize the analyzer."""
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.model = None
        self.rhvae_sampler = None
        self.rhvae_wrapper = None
        self.data_module = None
        
        self.output_dir = Path("outputs/training_vs_sampling") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔍 Training vs Sampling analyzer initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model_and_setup(self) -> None:
        """Load model and setup samplers (reusing working code)."""
        logger.info(f"🔄 Loading model and setting up components")
        
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
        data_config = DictConfig({
            'train_path': 'data/sprites/ColoredCircles_train.pt',
            'test_path': 'data/sprites/ColoredCircles_test.pt',
            'train_meta_path': 'data/sprites/ColoredCircles_train_params.pt',
            'test_meta_path': 'data/sprites/ColoredCircles_test_params.pt',
            'sequence_length': 10, 'image_size': [28, 28], 'channels': 3,
            'batch_size': 1, 'num_workers': 0, 'pin_memory': False,
            'max_test_samples': 200, 'verify_cyclicity': False
        })
        
        self.data_module = CyclicSpritesDataModule(data_config)
        self.data_module.setup('test')
    
    def _setup_rhvae_sampler(self) -> None:
        """Setup RHVAE sampler (reused from previous working code)."""
        
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
            mcmc_steps_nbr=300,  # More steps for better quality
            n_lf=25,
            eps_lf=0.008,
            beta_zero=1.0
        )
        
        self.rhvae_sampler = RHVAESampler(
            model=self.rhvae_wrapper,
            sampler_config=sampler_config
        )
        
        logger.info("✅ RHVAE sampler setup complete")
    
    def extract_training_trajectories(self, n_sequences: int = 100) -> tuple:
        """Extract real training trajectories by passing data through encoder."""
        logger.info(f"📊 Extracting training trajectories ({n_sequences} sequences)")
        
        trajectories = []
        determinants = []
        test_loader = self.data_module.test_dataloader()
        
        sequence_count = 0
        for batch_idx, batch in enumerate(test_loader):
            if sequence_count >= n_sequences:
                break
            
            try:
                if len(batch.shape) == 4:
                    batch = batch.unsqueeze(0)
                batch = batch.to(self.device)
                
                with torch.no_grad():
                    # Process the full sequence through encoder
                    batch_flat = batch.view(batch.shape[0], -1, *batch.shape[3:])
                    
                    sequence_trajectory = []
                    sequence_determinants = []
                    
                    for t in range(batch_flat.shape[1]):  # For each timestep
                        x_t = batch_flat[:, t]
                        
                        # Get latent representation
                        encoder_out = self.model.encoder(x_t)
                        if isinstance(encoder_out, dict):
                            mu = encoder_out['mu']
                            logvar = encoder_out['logvar']
                            # Sample from encoder distribution
                            std = torch.exp(0.5 * logvar)
                            eps = torch.randn_like(std)
                            z_t = mu + eps * std
                        else:
                            z_t = encoder_out
                        
                        sequence_trajectory.append(z_t.cpu().numpy())
                        
                        # Compute determinant for this point
                        det_val = self._compute_metric_determinant(z_t)
                        sequence_determinants.append(det_val)
                    
                    trajectories.append(np.array(sequence_trajectory).squeeze())
                    determinants.append(sequence_determinants)
                    sequence_count += 1
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to process sequence {batch_idx}: {e}")
                continue
        
        logger.info(f"✅ Extracted {len(trajectories)} training trajectories")
        return np.array(trajectories), determinants
    
    def sample_rhvae_trajectories(self, n_sequences: int = 100) -> tuple:
        """Sample pseudo-trajectories using RHVAE sampler."""
        logger.info(f"🎲 Sampling RHVAE pseudo-trajectories ({n_sequences} sequences)")
        
        # Sample individual points
        n_total_points = n_sequences * 10  # 10 points per "trajectory"
        with torch.no_grad():
            rhvae_samples = self.rhvae_sampler.hmc_sampling(n_total_points)
        
        # Reshape into pseudo-trajectories
        rhvae_trajectories = rhvae_samples.cpu().numpy().reshape(n_sequences, 10, -1)
        
        # Compute determinants
        determinants = []
        for seq_idx in range(n_sequences):
            seq_dets = []
            for t in range(10):
                z_point = torch.tensor(rhvae_trajectories[seq_idx, t], device=self.device, dtype=torch.float32).unsqueeze(0)
                det_val = self._compute_metric_determinant(z_point)
                seq_dets.append(det_val)
            determinants.append(seq_dets)
        
        logger.info(f"✅ Generated {len(rhvae_trajectories)} RHVAE pseudo-trajectories")
        return rhvae_trajectories, determinants
    
    def sample_rhvae_individual_points(self, n_points: int = 1000) -> tuple:
        """Sample individual points using RHVAE (not as trajectories)."""
        logger.info(f"🎲 Sampling individual RHVAE points ({n_points} points)")
        
        with torch.no_grad():
            rhvae_samples = self.rhvae_sampler.hmc_sampling(n_points)
        
        # Compute determinants
        determinants = []
        for i in range(n_points):
            z_point = rhvae_samples[i:i+1]
            det_val = self._compute_metric_determinant(z_point)
            determinants.append(det_val)
        
        return rhvae_samples.cpu().numpy(), determinants
    
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
    
    def create_comprehensive_pca_comparison(self, training_trajs, training_dets, 
                                          rhvae_trajs, rhvae_dets, 
                                          rhvae_points, rhvae_points_dets) -> go.Figure:
        """Create comprehensive PCA comparison."""
        logger.info("🎨 Creating comprehensive PCA comparison")
        
        # Flatten all data for PCA
        # Training data: shape (n_seq, 10, 2)
        training_flat = training_trajs.reshape(-1, 2)
        training_det_flat = [det for seq_dets in training_dets for det in seq_dets]
        
        # RHVAE trajectory data
        rhvae_flat = rhvae_trajs.reshape(-1, 2)
        rhvae_det_flat = [det for seq_dets in rhvae_dets for det in seq_dets]
        
        # Fit PCA on training data
        pca = PCA(n_components=2)
        training_pca = pca.fit_transform(training_flat)
        
        # Transform RHVAE data using the same PCA
        rhvae_pca = pca.transform(rhvae_flat)
        rhvae_points_pca = pca.transform(rhvae_points)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "Training Trajectories (Original PCA like your image)",
                "RHVAE Pseudo-Trajectories (Same PCA)",
                "RHVAE Individual Points (Same PCA)",
                "Training vs RHVAE Overlay",
                "Determinant Distributions",
                "PCA Component Analysis"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Training trajectories (like your image)
        for seq_idx in range(min(50, len(training_trajs))):  # Show first 50 sequences
            seq_pca = pca.transform(training_trajs[seq_idx])
            seq_dets = training_dets[seq_idx]
            
            fig.add_trace(
                go.Scatter(
                    x=seq_pca[:, 0],
                    y=seq_pca[:, 1],
                    mode='lines+markers',
                    line=dict(width=1, color=f'rgba(100, 150, 200, 0.6)'),
                    marker=dict(size=4, color=seq_dets, colorscale='Viridis', showscale=False),
                    name=f'Training Seq {seq_idx}',
                    showlegend=False,
                    hovertemplate="Training<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>det(G⁻¹): %{marker.color:.2f}<extra></extra>"
                ),
                row=1, col=1
            )
        
        # 2. RHVAE pseudo-trajectories
        for seq_idx in range(min(50, len(rhvae_trajs))):  # Show first 50 sequences
            seq_pca = pca.transform(rhvae_trajs[seq_idx])
            seq_dets = rhvae_dets[seq_idx]
            
            fig.add_trace(
                go.Scatter(
                    x=seq_pca[:, 0],
                    y=seq_pca[:, 1],
                    mode='lines+markers',
                    line=dict(width=1, color=f'rgba(200, 100, 100, 0.6)'),
                    marker=dict(size=4, color=seq_dets, colorscale='Plasma', showscale=False),
                    name=f'RHVAE Seq {seq_idx}',
                    showlegend=False,
                    hovertemplate="RHVAE Traj<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>det(G⁻¹): %{marker.color:.2f}<extra></extra>"
                ),
                row=1, col=2
            )
        
        # 3. RHVAE individual points
        fig.add_trace(
            go.Scatter(
                x=rhvae_points_pca[:, 0],
                y=rhvae_points_pca[:, 1],
                mode='markers',
                marker=dict(size=3, color=rhvae_points_dets, colorscale='Plasma', 
                           opacity=0.6, showscale=True, colorbar=dict(title="det(G⁻¹)", x=0.99)),
                name='RHVAE Points',
                showlegend=False,
                hovertemplate="RHVAE Point<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>det(G⁻¹): %{marker.color:.2f}<extra></extra>"
            ),
            row=1, col=3
        )
        
        # 4. Overlay comparison
        # Training points (sample)
        training_sample_idx = np.random.choice(len(training_pca), size=min(500, len(training_pca)), replace=False)
        fig.add_trace(
            go.Scatter(
                x=training_pca[training_sample_idx, 0],
                y=training_pca[training_sample_idx, 1],
                mode='markers',
                marker=dict(size=3, color='blue', opacity=0.5),
                name='Training Data',
                hovertemplate="Training<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
            ),
            row=2, col=1
        )
        
        # RHVAE points
        rhvae_sample_idx = np.random.choice(len(rhvae_points_pca), size=min(500, len(rhvae_points_pca)), replace=False)
        fig.add_trace(
            go.Scatter(
                x=rhvae_points_pca[rhvae_sample_idx, 0],
                y=rhvae_points_pca[rhvae_sample_idx, 1],
                mode='markers',
                marker=dict(size=3, color='red', opacity=0.5),
                name='RHVAE Sampled',
                hovertemplate="RHVAE<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
            ),
            row=2, col=1
        )
        
        # 5. Determinant distributions
        fig.add_trace(
            go.Histogram(
                x=training_det_flat,
                name='Training det(G⁻¹)',
                marker_color='blue',
                opacity=0.7,
                nbinsx=30
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=rhvae_points_dets,
                name='RHVAE det(G⁻¹)',
                marker_color='red',
                opacity=0.7,
                nbinsx=30
            ),
            row=2, col=2
        )
        
        # 6. PCA components analysis
        component_importance = pca.explained_variance_ratio_
        fig.add_trace(
            go.Bar(
                x=['PC1', 'PC2'],
                y=component_importance,
                name='PCA Components',
                marker_color=['#1f77b4', '#ff7f0e']
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title=dict(
                text="🔍 Training vs RHVAE Sampling PCA Comparison<br><sub>Why do they look different?</sub>",
                x=0.5,
                font=dict(size=16)
            ),
            showlegend=True
        )
        
        # Update axes
        for row in range(1, 3):
            for col in range(1, 4):
                if col <= 2 or (row == 2 and col == 1):  # PCA plots
                    fig.update_xaxes(title="PC1 (Principal Component 1)", row=row, col=col)
                    fig.update_yaxes(title="PC2 (Principal Component 2)", row=row, col=col)
        
        fig.update_xaxes(title="det(G⁻¹)", row=2, col=2)
        fig.update_yaxes(title="Count", row=2, col=2)
        fig.update_xaxes(title="Component", row=2, col=3)
        fig.update_yaxes(title="Explained Variance Ratio", row=2, col=3)
        
        return fig
    
    def analyze_discrepancy(self, training_trajs, rhvae_points) -> dict:
        """Analyze why training and sampling PCAs look different."""
        logger.info("🔬 Analyzing PCA discrepancy")
        
        # Flatten for analysis
        training_flat = training_trajs.reshape(-1, 2)
        
        # Compare statistics
        training_mean = np.mean(training_flat, axis=0)
        training_std = np.std(training_flat, axis=0)
        training_range = np.ptp(training_flat, axis=0)
        
        rhvae_mean = np.mean(rhvae_points, axis=0)
        rhvae_std = np.std(rhvae_points, axis=0)
        rhvae_range = np.ptp(rhvae_points, axis=0)
        
        # Distance from origin
        training_distances = np.linalg.norm(training_flat, axis=1)
        rhvae_distances = np.linalg.norm(rhvae_points, axis=1)
        
        analysis = {
            'training_stats': {
                'mean': training_mean,
                'std': training_std,
                'range': training_range,
                'mean_distance': np.mean(training_distances),
                'std_distance': np.std(training_distances)
            },
            'rhvae_stats': {
                'mean': rhvae_mean,
                'std': rhvae_std,
                'range': rhvae_range,
                'mean_distance': np.mean(rhvae_distances),
                'std_distance': np.std(rhvae_distances)
            },
            'differences': {
                'mean_diff': np.linalg.norm(training_mean - rhvae_mean),
                'std_ratio': rhvae_std / (training_std + 1e-8),
                'range_ratio': rhvae_range / (training_range + 1e-8),
                'distance_ratio': np.mean(rhvae_distances) / np.mean(training_distances)
            }
        }
        
        return analysis
    
    def create_conclusion_report(self, analysis: dict) -> str:
        """Create detailed conclusion about the discrepancy."""
        
        report = f"""
# 🔍 Training vs Sampling PCA Analysis Report

## 📊 Key Findings

### 🎯 Statistical Comparison

**Training Data:**
- Mean: [{analysis['training_stats']['mean'][0]:.3f}, {analysis['training_stats']['mean'][1]:.3f}]
- Std: [{analysis['training_stats']['std'][0]:.3f}, {analysis['training_stats']['std'][1]:.3f}]
- Range: [{analysis['training_stats']['range'][0]:.3f}, {analysis['training_stats']['range'][1]:.3f}]
- Mean distance: {analysis['training_stats']['mean_distance']:.3f}

**RHVAE Sampling:**
- Mean: [{analysis['rhvae_stats']['mean'][0]:.3f}, {analysis['rhvae_stats']['mean'][1]:.3f}]
- Std: [{analysis['rhvae_stats']['std'][0]:.3f}, {analysis['rhvae_stats']['std'][1]:.3f}]
- Range: [{analysis['rhvae_stats']['range'][0]:.3f}, {analysis['rhvae_stats']['range'][1]:.3f}]
- Mean distance: {analysis['rhvae_stats']['mean_distance']:.3f}

### 🔍 Key Differences

- **Mean difference:** {analysis['differences']['mean_diff']:.3f}
- **Std ratio (RHVAE/Training):** [{analysis['differences']['std_ratio'][0]:.3f}, {analysis['differences']['std_ratio'][1]:.3f}]
- **Range ratio (RHVAE/Training):** [{analysis['differences']['range_ratio'][0]:.3f}, {analysis['differences']['range_ratio'][1]:.3f}]
- **Distance ratio:** {analysis['differences']['distance_ratio']:.3f}

## 🎯 Possible Explanations for the Discrepancy

"""
        
        # Analyze the differences
        std_ratio_mean = np.mean(analysis['differences']['std_ratio'])
        range_ratio_mean = np.mean(analysis['differences']['range_ratio'])
        distance_ratio = analysis['differences']['distance_ratio']
        
        if std_ratio_mean > 1.5:
            report += "❗ **RHVAE samples have MUCH higher variance than training data**\n"
            report += "   → RHVAE is sampling from a BROADER distribution than the training manifold\n\n"
        
        if range_ratio_mean > 1.5:
            report += "❗ **RHVAE samples cover a MUCH wider range than training data**\n"
            report += "   → RHVAE is exploring regions NOT seen during training\n\n"
        
        if distance_ratio > 1.5:
            report += "❗ **RHVAE samples are FARTHER from origin than training data**\n"
            report += "   → RHVAE is sampling from the manifold 'exterior' rather than 'interior'\n\n"
        
        report += """
## 🤔 Why This Might Be Happening

1. **🎯 RHVAE Samples the Prior, Not the Posterior**
   - Training data follows the learned posterior q(z|x)
   - RHVAE samples from the learned prior p(z) via HMC
   - These are DIFFERENT distributions!

2. **🏔️ Metric Tensor Learned from Different Data**
   - The metric was learned from encoder outputs during training
   - But RHVAE uses this metric to sample the PRIOR space
   - The prior space might be broader than the posterior space

3. **⚙️ HMC Sampling Hyperparameters**
   - MCMC steps, leapfrog steps, step size might be too aggressive
   - Could be allowing exploration beyond the training manifold

4. **📊 PCA Fitting Difference**
   - Your training image: PCA fitted on training trajectories
   - Our comparison: PCA fitted on the same training data
   - Different PCA bases could make things look different

## 🚀 Recommendations

1. **Sample from Posterior Distribution**: Use encoder outputs + small noise instead of pure HMC
2. **Reduce HMC Aggressiveness**: Lower step size, fewer steps
3. **Use Training Data PCA**: Fit PCA on actual training trajectories
4. **Check Metric Learning**: Verify the metric was learned correctly during training
"""
        
        return report
    
    def run_comparison_analysis(self) -> None:
        """Run complete comparison analysis."""
        logger.info("🚀 Starting training vs sampling comparison")
        
        wandb.init(
            project="rlvae-training-vs-sampling",
            name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "checkpoint": self.checkpoint_path,
                "device": self.device,
                "analysis_type": "training_vs_sampling_pca"
            }
        )
        
        try:
            # Load model and setup
            self.load_model_and_setup()
            
            # Extract training trajectories
            training_trajs, training_dets = self.extract_training_trajectories(n_sequences=100)
            
            # Sample RHVAE trajectories
            rhvae_trajs, rhvae_trajs_dets = self.sample_rhvae_trajectories(n_sequences=100)
            
            # Sample individual RHVAE points
            rhvae_points, rhvae_points_dets = self.sample_rhvae_individual_points(n_points=1000)
            
            # Create comprehensive comparison
            comparison_fig = self.create_comprehensive_pca_comparison(
                training_trajs, training_dets, 
                rhvae_trajs, rhvae_trajs_dets,
                rhvae_points, rhvae_points_dets
            )
            
            comparison_path = self.output_dir / "training_vs_sampling_pca_comparison.html"
            comparison_fig.write_html(str(comparison_path))
            
            # Analyze discrepancy
            analysis = self.analyze_discrepancy(training_trajs, rhvae_points)
            
            # Create conclusion report
            conclusion_report = self.create_conclusion_report(analysis)
            
            # Save report
            report_path = self.output_dir / "discrepancy_analysis_report.md"
            with open(report_path, 'w') as f:
                f.write(conclusion_report)
            
            # Log to WandB
            wandb.log({
                "comparison_visualization": wandb.Html(str(comparison_path)),
                "std_ratio_mean": np.mean(analysis['differences']['std_ratio']),
                "range_ratio_mean": np.mean(analysis['differences']['range_ratio']),
                "distance_ratio": analysis['differences']['distance_ratio'],
                "mean_difference": analysis['differences']['mean_diff']
            })
            
            logger.info("🎉 Comparison analysis completed!")
            logger.info(f"📁 Results in: {self.output_dir}")
            logger.info(f"📊 Comparison: {comparison_path}")
            logger.info(f"📝 Report: {report_path}")
            
            # Print key findings
            print("\n" + "="*80)
            print("🎯 KEY FINDINGS:")
            print("="*80)
            print(f"Distance ratio (RHVAE/Training): {analysis['differences']['distance_ratio']:.3f}")
            print(f"Std ratio (RHVAE/Training): {np.mean(analysis['differences']['std_ratio']):.3f}")
            print(f"Range ratio (RHVAE/Training): {np.mean(analysis['differences']['range_ratio']):.3f}")
            print("="*80)
            
        except Exception as e:
            logger.error(f"❌ Comparison analysis failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            wandb.finish()


def main():
    """Main execution."""
    checkpoint_path = "outputs/checkpoints/epoch=197-val_loss=6.402.ckpt"
    
    analyzer = TrainingVsSamplingAnalyzer(checkpoint_path)
    analyzer.run_comparison_analysis()


if __name__ == "__main__":
    main() 