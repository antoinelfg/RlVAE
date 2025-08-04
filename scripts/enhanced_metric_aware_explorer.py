#!/usr/bin/env python3
"""
Enhanced Metric-Aware Explorer
==============================

Advanced version with:
1. 3D metric landscape visualization
2. True metric-based image generation via RHVAE sampler
3. Sophisticated interpolation and extrapolation
4. Color-coded image borders for different purposes
5. Sequence generation and prediction
6. Interactive 3D exploration

The ultimate metric-aware analysis suite!
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
import base64
import io
from PIL import Image, ImageDraw

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

class EnhancedMetricAwareExplorer:
    """Enhanced explorer with 3D viz, generation, interpolation, and extrapolation."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """Initialize the enhanced explorer."""
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.model = None
        self.rhvae_sampler = None
        self.rhvae_wrapper = None
        self.data_module = None
        
        self.output_dir = Path("outputs/enhanced_metric_explorer") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Enhanced metric-aware explorer initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model_and_setup_enhanced_sampler(self) -> None:
        """Load model and setup the enhanced RHVAE sampler."""
        logger.info(f"🔄 Loading model and setting up enhanced metric-aware sampler")
        
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
        
        # Setup data
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
        
        logger.info("✅ Model loaded successfully")
        
        # Setup enhanced RHVAE sampler
        self._setup_enhanced_rhvae_sampler()
    
    def _setup_enhanced_rhvae_sampler(self) -> None:
        """Setup enhanced RHVAE-compatible sampler."""
        logger.info("🎯 Setting up enhanced RHVAE-compatible sampler")
        
        # Enhanced wrapper with additional features
        class EnhancedRHVAEWrapper:
            def __init__(self, rlvae_model, device):
                self.rlvae_model = rlvae_model
                self.device = device
                self.latent_dim = rlvae_model.latent_dim
                
                # Extract metric tensor components with consistent dtype
                self.centroids_tens = rlvae_model.modular_metric.centroids.clone().detach().to(dtype=torch.float32)
                self.M_tens = rlvae_model.modular_metric.metric_matrices.clone().detach().to(dtype=torch.float32)
                
                # Enhanced temperature control
                self.temperature = 2.5  # Slightly lower for better sampling
                
                logger.info(f"📊 Enhanced RHVAE wrapper created:")
                logger.info(f"   - Centroids: {self.centroids_tens.shape}")
                logger.info(f"   - Metric matrices: {self.M_tens.shape}")
                logger.info(f"   - Temperature: {self.temperature}")
            
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
                """Enhanced decoder wrapper with proper output formatting."""
                with torch.no_grad():
                    # Handle sequences vs single latents
                    if len(z.shape) == 3:  # [batch, seq_len, latent_dim]
                        batch_size, seq_len = z.shape[:2]
                        z_flat = z.view(-1, z.shape[-1])  # [batch*seq_len, latent_dim]
                        
                        output = self.rlvae_model.decoder(z_flat)
                        if isinstance(output, dict):
                            recon = output['reconstruction']
                        else:
                            recon = output
                        
                        # Reshape back to sequences
                        recon = recon.view(batch_size, seq_len, *recon.shape[1:])
                        return {"reconstruction": recon}
                    else:
                        output = self.rlvae_model.decoder(z)
                        if isinstance(output, dict):
                            return output
                        else:
                            return {"reconstruction": output}
            
            def G_inv(self, z):
                """Compute G^(-1) with enhanced precision."""
                batch_size = z.shape[0]
                
                # Ensure consistent dtypes
                z = z.to(dtype=torch.float32)
                centroids = self.centroids_tens.to(dtype=torch.float32)
                matrices = self.M_tens.to(dtype=torch.float32)
                
                # Find closest centroids
                distances = torch.cdist(z, centroids)
                closest_indices = torch.argmin(distances, dim=1)
                
                # Get corresponding metric matrices
                G_inv_batch = matrices[closest_indices]
                
                return G_inv_batch
            
            def G(self, z):
                """Compute G with enhanced numerical stability."""
                G_inv = self.G_inv(z)
                eye = torch.eye(self.latent_dim, device=self.device, dtype=torch.float32)
                G = torch.inverse(G_inv + 1e-8 * eye)  # Better numerical stability
                return G
        
        # Create enhanced wrapper
        self.rhvae_wrapper = EnhancedRHVAEWrapper(self.model, self.device)
        
        # Create enhanced sampler config
        sampler_config = RHVAESamplerConfig(
            mcmc_steps_nbr=300,  # More steps for higher quality
            n_lf=25,             # More leapfrog steps
            eps_lf=0.008,        # Smaller step for precision
            beta_zero=1.0
        )
        
        # Create RHVAE sampler
        self.rhvae_sampler = RHVAESampler(
            model=self.rhvae_wrapper,
            sampler_config=sampler_config
        )
        
        logger.info("✅ Enhanced RHVAE-compatible sampler created")
    
    def generate_metric_aware_images(self, n_samples: int = 50) -> tuple:
        """Generate images using true metric-aware sampling."""
        logger.info(f"🎨 Generating {n_samples} metric-aware images")
        
        # Sample latent points
        with torch.no_grad():
            sampled_z = self.rhvae_sampler.hmc_sampling(n_samples)
        
        # Generate images
        with torch.no_grad():
            generated_images = self.rhvae_wrapper.decoder(sampled_z)["reconstruction"]
        
        # Compute determinants for each sample
        determinants = []
        for i in range(len(sampled_z)):
            z_point = sampled_z[i:i+1]
            det_val = self._compute_metric_determinant(z_point)
            determinants.append(det_val)
        
        logger.info(f"✅ Generated {len(generated_images)} metric-aware images")
        return sampled_z.cpu().numpy(), generated_images.cpu(), np.array(determinants)
    
    def _compute_metric_determinant(self, z_point: torch.Tensor) -> float:
        """Compute metric determinant with high precision."""
        try:
            z_point = z_point.to(device=self.device, dtype=torch.float32)
            G_inv = self.rhvae_wrapper.G_inv(z_point)
            G_inv = G_inv.to(dtype=torch.float32)
            det_g_inv = torch.det(G_inv[0])
            return torch.log10(det_g_inv + 1e-10).cpu().item()
        except Exception as e:
            return 0.0
    
    def create_interpolation_sequences(self, n_interpolations: int = 5) -> tuple:
        """Create sophisticated interpolation sequences with different types."""
        logger.info(f"🌈 Creating {n_interpolations} interpolation sequences")
        
        # Sample endpoint pairs for interpolation
        with torch.no_grad():
            endpoints = self.rhvae_sampler.hmc_sampling(n_interpolations * 2)
        
        interpolation_results = []
        interpolation_types = []
        
        for i in range(n_interpolations):
            start_z = endpoints[i*2:i*2+1]
            end_z = endpoints[i*2+1:i*2+2]
            
            # Create different types of interpolation
            if i % 3 == 0:
                # Linear interpolation in latent space
                interp_z, interp_type = self._linear_interpolation(start_z, end_z)
            elif i % 3 == 1:
                # Geodesic interpolation (if possible)
                interp_z, interp_type = self._geodesic_interpolation(start_z, end_z)
            else:
                # Spherical interpolation
                interp_z, interp_type = self._spherical_interpolation(start_z, end_z)
            
            # Generate images for interpolation
            with torch.no_grad():
                interp_images = self.rhvae_wrapper.decoder(interp_z)["reconstruction"]
            
            interpolation_results.append({
                'latent_path': interp_z.cpu().numpy(),
                'images': interp_images.cpu(),
                'start_z': start_z.cpu().numpy(),
                'end_z': end_z.cpu().numpy(),
                'type': interp_type
            })
            interpolation_types.append(interp_type)
        
        logger.info(f"✅ Created {len(interpolation_results)} interpolation sequences")
        return interpolation_results, interpolation_types
    
    def _linear_interpolation(self, start_z: torch.Tensor, end_z: torch.Tensor, n_steps: int = 10) -> tuple:
        """Linear interpolation in latent space."""
        steps = torch.linspace(0, 1, n_steps, device=self.device).view(-1, 1, 1)
        interp_z = start_z.unsqueeze(0) * (1 - steps) + end_z.unsqueeze(0) * steps
        return interp_z.view(-1, interp_z.shape[-1]), "Linear"
    
    def _geodesic_interpolation(self, start_z: torch.Tensor, end_z: torch.Tensor, n_steps: int = 10) -> tuple:
        """Attempt geodesic interpolation using metric tensor."""
        try:
            # Simple geodesic approximation using metric tensor
            steps = torch.linspace(0, 1, n_steps, device=self.device)
            interp_z = []
            
            for step in steps:
                # Weighted interpolation with metric consideration
                mid_z = start_z * (1 - step) + end_z * step
                
                # Adjust based on local metric
                G_inv = self.rhvae_wrapper.G_inv(mid_z)
                # Simple metric-aware adjustment (this is a simplification)
                adjustment = torch.trace(G_inv[0]) * 0.01 * (end_z - start_z)
                adjusted_z = mid_z + adjustment * step * (1 - step)
                
                interp_z.append(adjusted_z)
            
            return torch.cat(interp_z, dim=0), "Geodesic"
        except:
            # Fallback to linear if geodesic fails
            return self._linear_interpolation(start_z, end_z, n_steps)
    
    def _spherical_interpolation(self, start_z: torch.Tensor, end_z: torch.Tensor, n_steps: int = 10) -> tuple:
        """Spherical linear interpolation (SLERP)."""
        try:
            # Normalize vectors
            start_norm = start_z / (torch.norm(start_z, dim=-1, keepdim=True) + 1e-8)
            end_norm = end_z / (torch.norm(end_z, dim=-1, keepdim=True) + 1e-8)
            
            # Compute angle
            dot_product = torch.sum(start_norm * end_norm, dim=-1, keepdim=True)
            dot_product = torch.clamp(dot_product, -1 + 1e-7, 1 - 1e-7)
            angle = torch.acos(dot_product)
            
            sin_angle = torch.sin(angle)
            
            steps = torch.linspace(0, 1, n_steps, device=self.device).view(-1, 1, 1)
            
            if sin_angle.abs() < 1e-6:
                # Vectors are parallel, use linear interpolation
                return self._linear_interpolation(start_z, end_z, n_steps)
            
            # SLERP formula
            weight_start = torch.sin((1 - steps) * angle) / sin_angle
            weight_end = torch.sin(steps * angle) / sin_angle
            
            interp_norm = weight_start * start_norm + weight_end * end_norm
            
            # Scale back to original magnitudes
            start_mag = torch.norm(start_z, dim=-1, keepdim=True)
            end_mag = torch.norm(end_z, dim=-1, keepdim=True)
            interp_mag = start_mag * (1 - steps.squeeze(-1)) + end_mag * steps.squeeze(-1)
            
            interp_z = interp_norm * interp_mag.unsqueeze(-1)
            
            return interp_z.view(-1, interp_z.shape[-1]), "Spherical"
        except:
            # Fallback to linear if SLERP fails
            return self._linear_interpolation(start_z, end_z, n_steps)
    
    def create_extrapolation_sequences(self, n_extrapolations: int = 3) -> tuple:
        """Create extrapolation/prediction sequences."""
        logger.info(f"🔮 Creating {n_extrapolations} extrapolation sequences")
        
        extrapolation_results = []
        
        for i in range(n_extrapolations):
            # Sample starting points
            with torch.no_grad():
                start_points = self.rhvae_sampler.hmc_sampling(2)
            
            # Create direction vector
            direction = start_points[1] - start_points[0]
            
            # Extrapolate beyond
            extrap_factors = torch.linspace(0, 2.0, 12, device=self.device)  # Go beyond 1.0
            extrap_z = start_points[0].unsqueeze(0) + direction.unsqueeze(0) * extrap_factors.view(-1, 1)
            
            # Generate images
            with torch.no_grad():
                extrap_images = self.rhvae_wrapper.decoder(extrap_z)["reconstruction"]
            
            extrapolation_results.append({
                'latent_path': extrap_z.cpu().numpy(),
                'images': extrap_images.cpu(),
                'start_point': start_points[0].cpu().numpy(),
                'direction': direction.cpu().numpy()
            })
        
        logger.info(f"✅ Created {len(extrapolation_results)} extrapolation sequences")
        return extrapolation_results
    
    def add_colored_border(self, img_tensor: torch.Tensor, border_color: str, border_width: int = 3) -> torch.Tensor:
        """Add colored border to image tensor."""
        # Convert to PIL
        if len(img_tensor.shape) == 3:
            img_np = img_tensor.permute(1, 2, 0).numpy()
        else:
            img_np = img_tensor.numpy()
        
        img_np = (img_np * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        # Create border
        draw = ImageDraw.Draw(img_pil)
        width, height = img_pil.size
        
        # Define color map
        color_map = {
            'red': (255, 0, 0),
            'blue': (0, 0, 255), 
            'green': (0, 255, 0),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255)
        }
        
        color = color_map.get(border_color, (255, 255, 255))
        
        # Draw border
        for i in range(border_width):
            draw.rectangle([i, i, width-1-i, height-1-i], outline=color)
        
        # Convert back to tensor
        img_array = np.array(img_pil) / 255.0
        if len(img_array.shape) == 3:
            return torch.tensor(img_array).permute(2, 0, 1)
        else:
            return torch.tensor(img_array)
    
    def tensor_to_base64(self, tensor_img: torch.Tensor) -> str:
        """Convert tensor image to base64."""
        if len(tensor_img.shape) == 3:
            img_np = tensor_img.permute(1, 2, 0).numpy()
        else:
            img_np = tensor_img.numpy()
        
        img_np = (img_np * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, 'RGB')
        
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_base64}"
    
    def create_3d_metric_landscape(self, sampled_points: np.ndarray, sampled_dets: np.ndarray) -> go.Figure:
        """Create stunning 3D metric landscape visualization."""
        logger.info("🏔️ Creating 3D metric landscape visualization")
        
        # Create grid for surface
        x_min, x_max = sampled_points[:, 0].min() - 1, sampled_points[:, 0].max() + 1
        y_min, y_max = sampled_points[:, 1].min() - 1, sampled_points[:, 1].max() + 1
        
        grid_res = 40
        x_grid = np.linspace(x_min, x_max, grid_res)
        y_grid = np.linspace(y_min, y_max, grid_res)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Compute metric determinants on grid
        Z = np.zeros_like(X)
        for i in range(grid_res):
            for j in range(grid_res):
                z_point = torch.tensor([[X[i, j], Y[i, j]]], device=self.device, dtype=torch.float32)
                Z[i, j] = self._compute_metric_determinant(z_point)
        
        # Create 3D figure
        fig = go.Figure()
        
        # Add surface
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z,
                colorscale='earth',
                showscale=True,
                colorbar=dict(title="det(G⁻¹)", x=1.15, thickness=20, len=0.8),
                opacity=0.7,
                name="Metric Landscape",
                hovertemplate="Z1: %{x:.2f}<br>Z2: %{y:.2f}<br>det(G⁻¹): %{z:.3f}<extra></extra>"
            )
        )
        
        # Add sampled points as scatter
        fig.add_trace(
            go.Scatter3d(
                x=sampled_points[:, 0],
                y=sampled_points[:, 1],
                z=sampled_dets + 0.05,  # Slightly above surface
                mode='markers',
                marker=dict(
                    size=4,
                    color=sampled_dets,
                    colorscale='viridis',
                    showscale=False,
                    line=dict(width=1, color='white')
                ),
                name='Metric Samples',
                hovertemplate="Sample<br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<br>det(G⁻¹): %{z:.3f}<extra></extra>"
            )
        )
        
        # Add high-density region markers
        high_det_mask = sampled_dets >= np.percentile(sampled_dets, 85)
        if np.any(high_det_mask):
            fig.add_trace(
                go.Scatter3d(
                    x=sampled_points[high_det_mask, 0],
                    y=sampled_points[high_det_mask, 1],
                    z=sampled_dets[high_det_mask] + 0.1,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='red',
                        symbol='diamond',
                        line=dict(width=2, color='darkred')
                    ),
                    name='High Density Peaks',
                    hovertemplate="Peak<br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<br>det(G⁻¹): %{z:.3f}<extra></extra>"
                )
            )
        
        # Update layout for 3D
        fig.update_layout(
            title=dict(
                text="🏔️ 3D Metric Tensor Landscape - Interactive Exploration",
                x=0.5,
                font=dict(size=18)
            ),
            scene=dict(
                xaxis_title="Latent Dimension 1",
                yaxis_title="Latent Dimension 2", 
                zaxis_title="det(G⁻¹) - Metric Strength",
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.5),
                    center=dict(x=0, y=0, z=0)
                ),
                bgcolor="rgba(240,240,240,0.1)",
                xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.3)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.3)"),
                zaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.3)")
            ),
            height=800,
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )
        
        logger.info("✅ 3D metric landscape created")
        return fig
    
    def create_generation_gallery(self, generated_z: np.ndarray, generated_images: torch.Tensor, 
                                generated_dets: np.ndarray) -> str:
        """Create interactive gallery of generated images."""
        logger.info("🎨 Creating generation gallery")
        
        # Sort by determinant for organized display
        sort_indices = np.argsort(generated_dets)
        
        fig = make_subplots(
            rows=5, cols=10,
            subplot_titles=[f"det={generated_dets[idx]:.2f}" for idx in sort_indices],
            vertical_spacing=0.02,
            horizontal_spacing=0.02
        )
        
        for i, idx in enumerate(sort_indices):
            row = i // 10 + 1
            col = i % 10 + 1
            
            # Add colored border based on determinant value
            if generated_dets[idx] > np.percentile(generated_dets, 75):
                bordered_img = self.add_colored_border(generated_images[idx], 'red', 2)  # High det
            elif generated_dets[idx] < np.percentile(generated_dets, 25):
                bordered_img = self.add_colored_border(generated_images[idx], 'blue', 2)  # Low det
            else:
                bordered_img = self.add_colored_border(generated_images[idx], 'green', 2)  # Medium det
            
            img_base64 = self.tensor_to_base64(bordered_img)
            
            fig.add_layout_image(
                dict(
                    source=img_base64,
                    xref=f"x{i+1}", yref=f"y{i+1}",
                    x=0, y=1, sizex=1, sizey=1,
                    sizing="stretch", layer="below"
                )
            )
            
            # Add invisible scatter for hover
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1], mode='markers',
                    marker=dict(opacity=0), showlegend=False,
                    hovertemplate=f"Generated Image {idx}<br>Z1: {generated_z[idx, 0]:.3f}<br>Z2: {generated_z[idx, 1]:.3f}<br>det(G⁻¹): {generated_dets[idx]:.3f}<extra></extra>"
                ), row=row, col=col
            )
        
        fig.update_layout(
            title="🎨 Metric-Aware Generated Images Gallery<br><sub>🔴 High det(G⁻¹) | 🔵 Low det(G⁻¹) | 🟢 Medium det(G⁻¹)</sub>",
            height=600, showlegend=False
        )
        
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        
        gallery_path = self.output_dir / "generation_gallery.html"
        fig.write_html(str(gallery_path))
        
        logger.info(f"✅ Generation gallery saved to {gallery_path}")
        return str(gallery_path)
    
    def create_interpolation_showcase(self, interpolation_results: list, interpolation_types: list) -> str:
        """Create showcase of interpolation sequences."""
        logger.info("🌈 Creating interpolation showcase")
        
        n_interpolations = len(interpolation_results)
        fig = make_subplots(
            rows=n_interpolations, cols=12,  # 10 interpolation steps + 2 endpoints
            subplot_titles=[f"{interpolation_types[i//12]} {i%12}" if i%12==0 else "" 
                          for i in range(n_interpolations * 12)],
            vertical_spacing=0.03,
            horizontal_spacing=0.02
        )
        
        for interp_idx, result in enumerate(interpolation_results):
            row = interp_idx + 1
            images = result['images']
            interp_type = result['type']
            
            # Define border colors for different interpolation types
            border_colors = {
                'Linear': 'blue',
                'Geodesic': 'red', 
                'Spherical': 'green'
            }
            border_color = border_colors.get(interp_type, 'orange')
            
            for step_idx in range(len(images)):
                col = step_idx + 1
                
                # Add special borders for endpoints
                if step_idx == 0:
                    bordered_img = self.add_colored_border(images[step_idx], 'magenta', 4)  # Start
                elif step_idx == len(images) - 1:
                    bordered_img = self.add_colored_border(images[step_idx], 'cyan', 4)    # End
                else:
                    bordered_img = self.add_colored_border(images[step_idx], border_color, 2)  # Interpolation
                
                img_base64 = self.tensor_to_base64(bordered_img)
                
                fig.add_layout_image(
                    dict(
                        source=img_base64,
                        xref=f"x{interp_idx*12 + step_idx + 1}",
                        yref=f"y{interp_idx*12 + step_idx + 1}",
                        x=0, y=1, sizex=1, sizey=1,
                        sizing="stretch", layer="below"
                    )
                )
                
                # Add invisible scatter for hover
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1], y=[0, 1], mode='markers',
                        marker=dict(opacity=0), showlegend=False,
                        hovertemplate=f"{interp_type} Interpolation<br>Step: {step_idx}<br>Position: {step_idx/(len(images)-1):.2f}<extra></extra>"
                    ), row=row, col=col
                )
        
        fig.update_layout(
            title="🌈 Sophisticated Interpolation Showcase<br><sub>💜 Start | 🩵 End | 🔵 Linear | 🔴 Geodesic | 🟢 Spherical</sub>",
            height=n_interpolations * 100 + 100,
            showlegend=False
        )
        
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        
        interpolation_path = self.output_dir / "interpolation_showcase.html"
        fig.write_html(str(interpolation_path))
        
        logger.info(f"✅ Interpolation showcase saved to {interpolation_path}")
        return str(interpolation_path)
    
    def create_extrapolation_showcase(self, extrapolation_results: list) -> str:
        """Create showcase of extrapolation/prediction sequences."""
        logger.info("🔮 Creating extrapolation showcase")
        
        n_extrapolations = len(extrapolation_results)
        fig = make_subplots(
            rows=n_extrapolations, cols=12,
            subplot_titles=[f"Extrapolation {i//12 + 1} - Step {i%12}" if i%12==0 else ""
                          for i in range(n_extrapolations * 12)],
            vertical_spacing=0.03,
            horizontal_spacing=0.02
        )
        
        for extrap_idx, result in enumerate(extrapolation_results):
            row = extrap_idx + 1
            images = result['images']
            
            for step_idx in range(len(images)):
                col = step_idx + 1
                
                # Color code based on extrapolation factor
                if step_idx < 6:
                    bordered_img = self.add_colored_border(images[step_idx], 'yellow', 2)  # Past/interpolation
                elif step_idx < 9:
                    bordered_img = self.add_colored_border(images[step_idx], 'orange', 3)  # Near future
                else:
                    bordered_img = self.add_colored_border(images[step_idx], 'red', 4)     # Far future
                
                img_base64 = self.tensor_to_base64(bordered_img)
                
                fig.add_layout_image(
                    dict(
                        source=img_base64,
                        xref=f"x{extrap_idx*12 + step_idx + 1}",
                        yref=f"y{extrap_idx*12 + step_idx + 1}",
                        x=0, y=1, sizex=1, sizey=1,
                        sizing="stretch", layer="below"
                    )
                )
                
                extrap_factor = step_idx / 6.0  # Factor relative to midpoint
                
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1], y=[0, 1], mode='markers',
                        marker=dict(opacity=0), showlegend=False,
                        hovertemplate=f"Extrapolation {extrap_idx+1}<br>Step: {step_idx}<br>Factor: {extrap_factor:.2f}<extra></extra>"
                    ), row=row, col=col
                )
        
        fig.update_layout(
            title="🔮 Extrapolation/Prediction Showcase<br><sub>🟡 Past/Interpolation | 🟠 Near Future | 🔴 Far Future</sub>",
            height=n_extrapolations * 100 + 100,
            showlegend=False
        )
        
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        
        extrapolation_path = self.output_dir / "extrapolation_showcase.html"
        fig.write_html(str(extrapolation_path))
        
        logger.info(f"✅ Extrapolation showcase saved to {extrapolation_path}")
        return str(extrapolation_path)
    
    def create_enhanced_index(self, paths_dict: dict) -> None:
        """Create enhanced index page."""
        index_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>🚀 Enhanced Metric-Aware Explorer</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: #333; 
        }}
        .container {{ 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            min-height: 100vh; 
            box-shadow: 0 0 30px rgba(0,0,0,0.2); 
            padding: 40px;
        }}
        .header {{ 
            text-align: center; 
            margin-bottom: 40px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 40px; 
            border-radius: 15px; 
            margin: -40px -40px 40px -40px;
        }}
        .header h1 {{ 
            margin: 0; 
            font-size: 3em; 
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .section {{ 
            margin: 30px 0; 
            padding: 30px; 
            border-radius: 15px; 
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
            border-left: 6px solid #667eea; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
        }}
        .section h2 {{ 
            margin-top: 0; 
            color: #667eea; 
            font-size: 1.8em; 
        }}
        .btn {{ 
            display: inline-block; 
            padding: 18px 35px; 
            margin: 12px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            text-decoration: none; 
            border-radius: 30px; 
            font-weight: bold; 
            transition: all 0.3s ease; 
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3); 
            text-transform: uppercase; 
            letter-spacing: 1px; 
        }}
        .btn:hover {{ 
            transform: translateY(-3px); 
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5); 
        }}
        .btn.new {{ 
            background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%); 
            box-shadow: 0 5px 20px rgba(255, 107, 107, 0.3); 
        }}
        .btn.new:hover {{ 
            box-shadow: 0 8px 25px rgba(255, 107, 107, 0.5); 
        }}
        .features {{ 
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
            color: white; 
            padding: 30px; 
            border-radius: 15px; 
            margin: 30px 0; 
        }}
        .features h3 {{ 
            margin-top: 0; 
            font-size: 1.5em; 
        }}
        .features ul {{ 
            margin: 20px 0; 
            padding-left: 25px; 
            line-height: 1.8; 
        }}
        .color-legend {{ 
            background: #f8f9fa; 
            padding: 20px; 
            border-radius: 10px; 
            margin: 20px 0; 
            border: 2px solid #dee2e6; 
        }}
        .color-box {{ 
            display: inline-block; 
            width: 20px; 
            height: 20px; 
            margin-right: 10px; 
            border: 2px solid #333; 
            vertical-align: middle; 
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Enhanced Metric-Aware Explorer</h1>
            <p>Complete suite with 3D visualization, generation, and interpolation</p>
        </div>
        
        <div class="section">
            <h2>🏔️ 3D Metric Landscape</h2>
            <p>Interactive 3D visualization of the metric tensor landscape with surface topology and sample points.</p>
            <a href="{Path(paths_dict['3d_landscape']).name}" class="btn new">🎮 Explore 3D Landscape</a>
        </div>
        
        <div class="section">
            <h2>🎨 True Metric-Based Generation</h2>
            <p>Images generated using authentic RHVAE sampling with color-coded borders based on metric determinant values.</p>
            <a href="{Path(paths_dict['generation']).name}" class="btn new">🖼️ View Generated Images</a>
        </div>
        
        <div class="section">
            <h2>🌈 Sophisticated Interpolation</h2>
            <p>Multiple interpolation methods with color-coded sequences showing different geometric approaches.</p>
            <a href="{Path(paths_dict['interpolation']).name}" class="btn new">🔄 Explore Interpolations</a>
        </div>
        
        <div class="section">
            <h2>🔮 Extrapolation & Prediction</h2>
            <p>Venture beyond the training manifold with extrapolation sequences showing future predictions.</p>
            <a href="{Path(paths_dict['extrapolation']).name}" class="btn new">⭐ View Extrapolations</a>
        </div>
        
        <div class="color-legend">
            <h3>🎨 Color Coding Legend</h3>
            <p><span class="color-box" style="background: red;"></span><strong>Red borders:</strong> High det(G⁻¹) / Geodesic interpolation / Far future extrapolation</p>
            <p><span class="color-box" style="background: blue;"></span><strong>Blue borders:</strong> Low det(G⁻¹) / Linear interpolation</p>
            <p><span class="color-box" style="background: green;"></span><strong>Green borders:</strong> Medium det(G⁻¹) / Spherical interpolation</p>
            <p><span class="color-box" style="background: magenta;"></span><strong>Magenta borders:</strong> Interpolation start points</p>
            <p><span class="color-box" style="background: cyan;"></span><strong>Cyan borders:</strong> Interpolation end points</p>
            <p><span class="color-box" style="background: yellow;"></span><strong>Yellow borders:</strong> Past/interpolation region</p>
            <p><span class="color-box" style="background: orange;"></span><strong>Orange borders:</strong> Near future extrapolation</p>
        </div>
        
        <div class="features">
            <h3>🌟 Enhanced Features</h3>
            <ul>
                <li><strong>🏔️ Interactive 3D Landscape:</strong> Full rotation, zoom, and exploration of metric tensor surface</li>
                <li><strong>🎲 True RHVAE Sampling:</strong> Authentic metric-aware generation using HMC sampling</li>
                <li><strong>🌈 Multiple Interpolation Types:</strong> Linear, Geodesic, and Spherical interpolation methods</li>
                <li><strong>🔮 Extrapolation/Prediction:</strong> Venture beyond training data with directional extrapolation</li>
                <li><strong>🎨 Color-Coded Borders:</strong> Visual distinction of different image types and regions</li>
                <li><strong>📊 Enhanced Numerical Stability:</strong> Improved dtype consistency and error handling</li>
                <li><strong>🎮 Interactive Elements:</strong> Hover details, responsive design, and smooth navigation</li>
                <li><strong>🔬 Advanced Sampling:</strong> 300 MCMC steps, 25 leapfrog steps for highest quality</li>
            </ul>
        </div>
    </div>
</body>
</html>
        """
        
        index_path = self.output_dir / "index.html"
        with open(index_path, 'w') as f:
            f.write(index_html)
        
        logger.info(f"✅ Enhanced index created at {index_path}")
    
    def run_enhanced_analysis(self) -> None:
        """Run the complete enhanced metric-aware analysis."""
        logger.info("🚀 Starting ENHANCED metric-aware analysis")
        
        wandb.init(
            project="rlvae-enhanced-metric-explorer",
            name=f"enhanced_explorer_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "checkpoint": self.checkpoint_path,
                "device": self.device,
                "analysis_type": "enhanced_3d_generation_interpolation"
            }
        )
        
        try:
            # Load model and setup enhanced sampler
            self.load_model_and_setup_enhanced_sampler()
            
            # Generate metric-aware images
            generated_z, generated_images, generated_dets = self.generate_metric_aware_images(n_samples=50)
            
            # Create 3D landscape
            landscape_3d_fig = self.create_3d_metric_landscape(generated_z, generated_dets)
            landscape_3d_path = self.output_dir / "3d_metric_landscape.html"
            landscape_3d_fig.write_html(str(landscape_3d_path))
            
            # Create generation gallery
            generation_path = self.create_generation_gallery(generated_z, generated_images, generated_dets)
            
            # Create interpolation sequences
            interpolation_results, interpolation_types = self.create_interpolation_sequences(n_interpolations=5)
            interpolation_path = self.create_interpolation_showcase(interpolation_results, interpolation_types)
            
            # Create extrapolation sequences
            extrapolation_results = self.create_extrapolation_sequences(n_extrapolations=3)
            extrapolation_path = self.create_extrapolation_showcase(extrapolation_results)
            
            # Create enhanced index
            paths_dict = {
                '3d_landscape': str(landscape_3d_path),
                'generation': generation_path,
                'interpolation': interpolation_path,
                'extrapolation': extrapolation_path
            }
            self.create_enhanced_index(paths_dict)
            
            # Log to WandB
            wandb.log({
                "3d_landscape": wandb.Html(str(landscape_3d_path)),
                "generation_gallery": wandb.Html(generation_path),
                "interpolation_showcase": wandb.Html(interpolation_path),
                "extrapolation_showcase": wandb.Html(extrapolation_path),
                "n_generated": len(generated_images),
                "n_interpolations": len(interpolation_results),
                "n_extrapolations": len(extrapolation_results)
            })
            
            logger.info("🎉 ENHANCED metric-aware analysis completed!")
            logger.info(f"📁 Results in: {self.output_dir}")
            logger.info(f"🌐 Open: {self.output_dir}/index.html")
            logger.info("🚀 Full suite: 3D landscape + generation + interpolation + extrapolation!")
            
        except Exception as e:
            logger.error(f"❌ Enhanced analysis failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            wandb.finish()


def main():
    """Main execution."""
    checkpoint_path = "outputs/checkpoints/epoch=197-val_loss=6.402.ckpt"
    
    explorer = EnhancedMetricAwareExplorer(checkpoint_path)
    explorer.run_enhanced_analysis()


if __name__ == "__main__":
    main() 