"""
Comprehensive Inference Pipeline for Modular RlVAE
=================================================

This module provides inference capabilities for encoding new images to latent space,
performing reconstruction, and analyzing the learned representations using the
modular RlVAE architecture.

Key Features:
- Image encoding to latent space with multiple sampling methods
- High-quality reconstruction from latent codes
- Latent space analysis and visualization
- Batch processing for large datasets
- Memory-efficient computation
- Integration with existing model components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
from pathlib import Path
import warnings
from dataclasses import dataclass, fields
from PIL import Image
import torchvision.transforms as transforms
from omegaconf import DictConfig

from src.models.modular_rlvae import ModularRiemannianFlowVAE


@dataclass
class InferenceConfig:
    """Configuration for inference operations."""
    
    # Processing parameters
    batch_size: int = 32
    max_batch_size: int = 64
    
    # Encoding parameters
    use_mean: bool = False  # Use posterior mean vs. sampling
    sampling_method: str = "enhanced"  # For reparameterization sampling
    
    # Sequence handling
    sequence_mode: str = "single"  # single, sequence, first_frame
    max_sequence_length: int = 32
    
    # Output format
    return_uncertainties: bool = True
    return_intermediate: bool = False
    normalize_latents: bool = False
    
    # Memory management
    clear_cache: bool = True
    use_half_precision: bool = False


def dictconfig_to_inference_config(cfg: Union[DictConfig, dict, None]) -> InferenceConfig:
    if cfg is None:
        return InferenceConfig()
    if isinstance(cfg, InferenceConfig):
        return cfg
    # Only keep keys that are valid for InferenceConfig
    valid_keys = {f.name for f in fields(InferenceConfig)}
    filtered = {k: v for k, v in dict(cfg).items() if k in valid_keys}
    return InferenceConfig(**filtered)

class RlVAEInferencePipeline:
    """
    Comprehensive inference pipeline for RlVAE models.
    
    Provides methods for encoding images to latent space, reconstruction,
    and latent space analysis.
    """
    
    def __init__(self, model: ModularRiemannianFlowVAE, device: Optional[torch.device] = None):
        """
        Initialize inference pipeline with a trained model.
        
        Args:
            model: Trained modular RlVAE model
            device: Device for computation
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Setup preprocessing transforms
        self._setup_transforms()
        
        # Cache for repeated operations
        self._inference_cache = {}
        
    def _setup_transforms(self):
        """Setup image preprocessing transforms."""
        # Get expected input size from model
        if hasattr(self.model, 'input_dim') and isinstance(self.model.input_dim, (list, tuple)):
            if len(self.model.input_dim) == 3:  # [C, H, W]
                expected_size = (self.model.input_dim[1], self.model.input_dim[2])
            else:
                expected_size = (64, 64)  # Default fallback
        else:
            expected_size = (64, 64)  # Default fallback
        
        print(f"   🖼️ Setting up transforms for size: {expected_size}")
        
        # Standard preprocessing for the model's expected input format
        self.preprocess = transforms.Compose([
            transforms.Resize(expected_size),  # Ensure consistent size
            transforms.ToTensor(),
            # Note: Normalization should match training preprocessing
            # Assuming images are in [0, 1] range after ToTensor()
        ])
        
        # Inverse transform for visualization
        self.postprocess = transforms.Compose([
            transforms.ToPILImage(),
        ])
        
    def encode_images(self, images: Union[torch.Tensor, List[Image.Image]], 
                     config: Optional[Union[InferenceConfig, DictConfig, dict]] = None) -> Dict[str, torch.Tensor]:
        """
        Encode images to latent space.
        
        Args:
            images: Input images (tensor [B, C, H, W] or list of PIL Images)
            config: Inference configuration
            
        Returns:
            Dictionary containing latent representations and metadata
        """
        if config is None:
            config = InferenceConfig()
        else:
            config = dictconfig_to_inference_config(config)
        
        # Convert PIL images to tensor if needed
        if isinstance(images, list):
            images = self._pil_to_tensor(images)
        
        # Ensure tensor is on correct device
        images = images.to(self.device)
        
        # Handle different sequence modes
        if images.dim() == 5:  # [B, T, C, H, W]
            if config.sequence_mode == "first_frame":
                images = images[:, 0]  # Take first frame
            elif config.sequence_mode == "single":
                raise ValueError("Got sequence input but sequence_mode is 'single'")
            # For sequence mode, keep as is
        elif images.dim() == 4:  # [B, C, H, W]
            if config.sequence_mode == "sequence":
                images = images.unsqueeze(1)  # Add time dimension
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {images.dim()}D")
        
        batch_size = images.shape[0]
        print(f"🔍 Encoding {batch_size} images to latent space...")
        
        # Process in batches
        all_results = []
        num_batches = (batch_size + config.batch_size - 1) // config.batch_size
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * config.batch_size
                end_idx = min(start_idx + config.batch_size, batch_size)
                
                batch_images = images[start_idx:end_idx]
                batch_result = self._encode_batch(batch_images, config)
                all_results.append(batch_result)
                
                if config.clear_cache:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Concatenate results
        combined_result = self._combine_batch_results(all_results)
        
        print(f"   ✅ Encoded {batch_size} images successfully")
        
        return combined_result
    
    def _encode_batch(self, images: torch.Tensor, config: InferenceConfig) -> Dict[str, torch.Tensor]:
        """Encode a single batch of images."""
        # Handle sequence vs single image encoding
        if images.dim() == 5:  # [B, T, C, H, W] - sequence
            batch_size, seq_len, channels, height, width = images.shape
            
            # Encode first frame to get latent parameters
            x_0 = images[:, 0]
            encoder_out = self.model.encoder(x_0)
            mu = encoder_out.embedding
            log_var = encoder_out.log_covariance
            
            if config.use_mean:
                # Use posterior mean
                z_0 = mu
            else:
                # Sample from posterior
                if hasattr(self.model, 'samplers') and config.sampling_method in ['geodesic', 'enhanced']:
                    # Use advanced sampling if available
                    try:
                        sampler = getattr(self.model, 'sampler', None)
                        if sampler:
                            z_0 = sampler.sample_riemannian_latents(mu, log_var, method=config.sampling_method)
                        else:
                            # Fallback to standard reparameterization
                            eps = torch.randn_like(mu)
                            z_0 = mu + eps * torch.exp(0.5 * log_var)
                    except:
                        # Fallback to standard reparameterization
                        eps = torch.randn_like(mu)
                        z_0 = mu + eps * torch.exp(0.5 * log_var)
                else:
                    # Standard reparameterization
                    eps = torch.randn_like(mu)
                    z_0 = mu + eps * torch.exp(0.5 * log_var)
            
            # Generate sequence using flows if available
            if self.model.n_flows > 0 and seq_len > 1:
                z_seq = [z_0]
                for t in range(1, seq_len):
                    if hasattr(self.model, 'flow_manager'):
                        z_next, _ = self.model.flow_manager.apply_flows([z_seq[-1]], n_obs=2)
                        z_seq.append(z_next[-1])
                    else:
                        # Fallback: small perturbations
                        z_next = z_seq[-1] + 0.01 * torch.randn_like(z_seq[-1])
                        z_seq.append(z_next)
                
                latents = torch.stack(z_seq, dim=1)  # [B, T, D]
            else:
                # Single frame or no flows
                latents = z_0.unsqueeze(1).repeat(1, seq_len, 1) if seq_len > 1 else z_0.unsqueeze(1)
            
        else:  # [B, C, H, W] - single images
            try:
                encoder_out = self.model.encoder(images)
                
                # Handle different encoder output formats
                if hasattr(encoder_out, 'embedding'):
                    mu = encoder_out.embedding
                    log_var = encoder_out.log_covariance
                elif isinstance(encoder_out, dict):
                    mu = encoder_out.get('embedding') or encoder_out.get('mu')
                    log_var = encoder_out.get('log_covariance') or encoder_out.get('log_var')
                elif isinstance(encoder_out, (tuple, list)) and len(encoder_out) == 2:
                    mu, log_var = encoder_out
                else:
                    # Fallback: assume single output is mu, create dummy log_var
                    mu = encoder_out
                    log_var = torch.zeros_like(mu)
                
                if config.use_mean:
                    latents = mu.unsqueeze(1)  # Add time dimension for consistency
                else:
                    # Sample from posterior
                    eps = torch.randn_like(mu)
                    z = mu + eps * torch.exp(0.5 * log_var)
                    latents = z.unsqueeze(1)
                    
            except Exception as e:
                print(f"   ⚠️ Encoder error: {e}")
                print(f"   🔍 Image shape: {images.shape}")
                print(f"   🔍 Expected encoder input: {getattr(self.model.encoder, 'input_size', 'unknown')}")
                raise e
        
        # Compute uncertainties if requested
        uncertainties = None
        if config.return_uncertainties:
            std = torch.exp(0.5 * log_var)
            uncertainties = {
                'posterior_std': std,
                'posterior_var': torch.exp(log_var),
                'total_uncertainty': torch.mean(std, dim=-1),  # Average across latent dims
                'max_uncertainty': torch.max(std, dim=-1)[0],  # Max across latent dims
            }
        
        # Prepare result
        result = {
            'latents': latents,
            'posterior_mean': mu,
            'posterior_log_var': log_var,
        }
        
        if uncertainties:
            result['uncertainties'] = uncertainties
        
        if config.return_intermediate:
            result['encoder_output'] = encoder_out
        
        return result
    
    def reconstruct_from_latents(self, latents: torch.Tensor,
                                config: Optional[Union[InferenceConfig, DictConfig, dict]] = None) -> Dict[str, torch.Tensor]:
        """
        Reconstruct images from latent codes.
        
        Args:
            latents: Latent codes [B, T, D] or [B, D]
            config: Inference configuration
            
        Returns:
            Dictionary containing reconstructed images and metadata
        """
        if config is None:
            config = InferenceConfig()
        else:
            config = dictconfig_to_inference_config(config)
        
        # Handle different latent formats
        if latents.dim() == 2:
            latents = latents.unsqueeze(1)  # Add time dimension
        elif latents.dim() != 3:
            raise ValueError(f"Expected latents shape [B, T, D] or [B, D], got {latents.shape}")
        
        batch_size, seq_len, latent_dim = latents.shape
        
        print(f"🔄 Reconstructing {batch_size} sequences (length {seq_len})...")
        
        # Process in batches
        all_reconstructions = []
        num_batches = (batch_size + config.batch_size - 1) // config.batch_size
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * config.batch_size
                end_idx = min(start_idx + config.batch_size, batch_size)
                
                batch_latents = latents[start_idx:end_idx]
                batch_recon = self._reconstruct_batch(batch_latents)
                all_reconstructions.append(batch_recon)
                
                if config.clear_cache:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Concatenate results
        reconstructions = torch.cat(all_reconstructions, dim=0)
        
        # Clamp to valid range
        reconstructions = torch.clamp(reconstructions, 0, 1)
        
        print(f"   ✅ Reconstructed {batch_size} sequences successfully")
        
        return {
            'reconstructions': reconstructions,
            'latents': latents,
        }
    
    def _reconstruct_batch(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct a single batch from latents."""
        batch_size, seq_len, latent_dim = latents.shape
        
        # Flatten for decoding
        latents_flat = latents.view(-1, latent_dim)
        
        try:
            # Decode - handle different decoder interfaces
            if hasattr(self.model.decoder, 'decode'):
                # Use decoder's decode method if available
                decoder_out = self.model.decoder.decode(latents_flat)
            else:
                # Direct call to decoder
                decoder_out = self.model.decoder(latents_flat)
            
            # Handle different decoder output formats
            if hasattr(decoder_out, 'reconstruction'):
                reconstructions_flat = decoder_out.reconstruction
            elif isinstance(decoder_out, dict) and "reconstruction" in decoder_out:
                reconstructions_flat = decoder_out["reconstruction"]
            else:
                reconstructions_flat = decoder_out
            
            # Reshape back to sequence format
            img_shape = reconstructions_flat.shape[1:]
            reconstructions = reconstructions_flat.view(batch_size, seq_len, *img_shape)
            
            return reconstructions
            
        except Exception as e:
            print(f"   ⚠️ Decoder error: {e}")
            print(f"   🔍 Latent shape: {latents_flat.shape}")
            print(f"   🔍 Expected decoder input size: {getattr(self.model.decoder, 'input_size', 'unknown')}")
            
            # Try to get more info about the decoder
            if hasattr(self.model.decoder, 'linear_layers'):
                first_layer = self.model.decoder.linear_layers[0]
                expected_input = first_layer.in_features
                print(f"   🔍 Decoder first layer expects: {expected_input}")
            
            # Re-raise the error for debugging
            raise e
    
    def encode_and_reconstruct(self, images: Union[torch.Tensor, List[Image.Image]],
                              config: Optional[Union[InferenceConfig, DictConfig, dict]] = None) -> Dict[str, torch.Tensor]:
        """
        Perform full encode-reconstruct cycle.
        
        Args:
            images: Input images
            config: Inference configuration
            
        Returns:
            Dictionary containing original images, latents, and reconstructions
        """
        if config is None:
            config = InferenceConfig()
        else:
            config = dictconfig_to_inference_config(config)
        
        print("🔄 Performing encode-reconstruct cycle...")
        
        # Convert PIL to tensor if needed
        if isinstance(images, list):
            images = self._pil_to_tensor(images)
        
        # Encode
        encoding_result = self.encode_images(images, config)
        
        # Reconstruct
        reconstruction_result = self.reconstruct_from_latents(
            encoding_result['latents'], config
        )
        
        # Compute reconstruction metrics
        if images.dim() == 4:  # Single images
            images = images.unsqueeze(1)  # Add time dimension
        
        reconstruction_metrics = self._compute_reconstruction_metrics(
            images, reconstruction_result['reconstructions']
        )
        
        return {
            'original_images': images,
            'latents': encoding_result['latents'],
            'reconstructions': reconstruction_result['reconstructions'],
            'posterior_mean': encoding_result['posterior_mean'],
            'posterior_log_var': encoding_result['posterior_log_var'],
            'reconstruction_metrics': reconstruction_metrics,
            'uncertainties': encoding_result.get('uncertainties'),
        }
    
    def analyze_latent_space(self, latents: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze properties of latent representations.
        
        Args:
            latents: Latent codes [B, T, D] or [B, D]
            
        Returns:
            Dictionary containing latent space analysis
        """
        if latents.dim() == 2:
            latents = latents.unsqueeze(1)
        
        batch_size, seq_len, latent_dim = latents.shape
        
        # Flatten for analysis
        latents_flat = latents.view(-1, latent_dim)
        
        print(f"📊 Analyzing latent space ({latents_flat.shape[0]} points, {latent_dim}D)...")
        
        with torch.no_grad():
            # Basic statistics
            mean = torch.mean(latents_flat, dim=0)
            std = torch.std(latents_flat, dim=0)
            
            # Dimensionality analysis
            latents_centered = latents_flat - mean
            cov_matrix = torch.mm(latents_centered.T, latents_centered) / (latents_flat.shape[0] - 1)
            eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)
            
            # Sort eigenvalues in descending order
            eigenvals, indices = torch.sort(eigenvals, descending=True)
            eigenvecs = eigenvecs[:, indices]
            
            # Effective dimensionality (participation ratio)
            eigenvals_normalized = eigenvals / eigenvals.sum()
            effective_dim = 1.0 / torch.sum(eigenvals_normalized ** 2)
            
            # Intrinsic dimensionality estimation (90% variance)
            cumsum_eigenvals = torch.cumsum(eigenvals_normalized, dim=0)
            intrinsic_dim = torch.argmax((cumsum_eigenvals >= 0.9).float()) + 1
        
        analysis = {
            'basic_stats': {
                'mean': mean,
                'std': std,
                'min': torch.min(latents_flat, dim=0)[0],
                'max': torch.max(latents_flat, dim=0)[0],
            },
            'dimensionality': {
                'latent_dim': latent_dim,
                'effective_dim': effective_dim.item(),
                'intrinsic_dim': intrinsic_dim.item(),
                'eigenvalues': eigenvals,
                'eigenvectors': eigenvecs,
                'explained_variance_ratio': eigenvals_normalized,
            },
            'geometry': {
                'covariance_matrix': cov_matrix,
                'condition_number': (eigenvals[0] / eigenvals[-1]).item(),
            }
        }
        
        # Riemannian analysis if metric tensor is available
        if hasattr(self.model, 'G_inv'):
            try:
                riemannian_analysis = self._analyze_riemannian_geometry(latents_flat)
                analysis['riemannian'] = riemannian_analysis
            except Exception as e:
                print(f"   ⚠️ Riemannian analysis failed: {e}")
        
        print("   ✅ Latent space analysis complete")
        
        return analysis
    
    def _analyze_riemannian_geometry(self, latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze Riemannian geometry at latent points."""
        print("   🌀 Analyzing Riemannian geometry...")
        
        # Sample subset for efficiency
        n_samples = min(100, len(latents))
        indices = torch.randperm(len(latents))[:n_samples]
        sample_latents = latents[indices]
        
        # Compute metric tensors
        G_inv_samples = self.model.G_inv(sample_latents)
        
        # Analyze metric properties
        eigenvals = torch.linalg.eigvals(G_inv_samples).real
        condition_numbers = eigenvals.max(dim=-1)[0] / eigenvals.min(dim=-1)[0]
        determinants = torch.linalg.det(G_inv_samples)
        
        return {
            'metric_condition_numbers': condition_numbers,
            'metric_determinants': determinants,
            'metric_eigenvalues': eigenvals,
            'mean_condition_number': torch.mean(condition_numbers),
            'mean_determinant': torch.mean(determinants),
        }
    
    def _compute_reconstruction_metrics(self, original: torch.Tensor, 
                                      reconstructed: torch.Tensor) -> Dict[str, float]:
        """Compute reconstruction quality metrics."""
        # Ensure same shape
        if original.shape != reconstructed.shape:
            raise ValueError(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")
        
        # Ensure both tensors are on the same device
        device = reconstructed.device
        original = original.to(device)
        reconstructed = reconstructed.to(device)
        with torch.no_grad():
            # MSE
            mse = F.mse_loss(reconstructed, original)
            
            # PSNR
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            
            # SSIM (simplified)
            ssim = self._compute_simplified_ssim(original, reconstructed)
            
            # L1 loss
            l1_loss = F.l1_loss(reconstructed, original)
        
        return {
            'mse': mse.item(),
            'psnr': psnr.item(),
            'ssim': ssim.item(),
            'l1_loss': l1_loss.item(),
        }
    
    def _compute_simplified_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute simplified SSIM."""
        # Simplified SSIM computation
        mu1 = torch.mean(img1)
        mu2 = torch.mean(img2)
        
        sigma1_sq = torch.var(img1)
        sigma2_sq = torch.var(img2)
        sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return ssim
    
    def _pil_to_tensor(self, images: List[Image.Image]) -> torch.Tensor:
        """Convert list of PIL images to tensor."""
        tensors = [self.preprocess(img) for img in images]
        return torch.stack(tensors, dim=0)
    
    def _combine_batch_results(self, results: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Combine results from multiple batches."""
        combined = {}
        
        for key in results[0].keys():
            if key == 'uncertainties':
                # Handle nested dictionary
                combined[key] = {}
                for subkey in results[0][key].keys():
                    combined[key][subkey] = torch.cat([r[key][subkey] for r in results], dim=0)
            else:
                combined[key] = torch.cat([r[key] for r in results], dim=0)
        
        return combined
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get statistics about inference capabilities."""
        return {
            'model_type': type(self.model).__name__,
            'latent_dim': self.model.latent_dim,
            'input_dim': self.model.input_dim,
            'n_flows': self.model.n_flows,
            'has_metric_tensor': hasattr(self.model, 'G'),
            'device': str(self.device),
            'supports_sequences': True,
            'supports_uncertainty_quantification': True,
        }


def create_inference_pipeline(model: ModularRiemannianFlowVAE,
                            device: Optional[torch.device] = None,
                            config: Union[DictConfig, dict, InferenceConfig, None] = None) -> RlVAEInferencePipeline:
    """
    Factory function to create inference pipeline.
    Args:
        model: Trained modular RlVAE model
        device: Device for computation
        config: Hydra DictConfig, dict, or InferenceConfig
    Returns:
        Configured inference pipeline
    """
    pipeline = RlVAEInferencePipeline(model=model, device=device)
    pipeline._hydra_config = config
    return pipeline 