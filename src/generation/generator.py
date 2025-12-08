"""
Unified Generation Interface for Modular RlVAE
==============================================

This module provides a clean, unified interface for image generation using the 
modular RlVAE architecture. It integrates with existing sampling methods and 
provides high-level generation functions.

Key Features:
- Multiple sampling strategies (geodesic, enhanced, basic, standard)
- Batch generation with memory management
- Support for different sequence lengths
- Integration with existing flow management
- Comprehensive generation configuration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
from pathlib import Path
import warnings
from dataclasses import dataclass
from omegaconf import DictConfig

from rlvae.models.modular_rlvae import ModularRiemannianFlowVAE
from src.models.samplers import WorkingRiemannianSampler, RiemannianHMCSampler, OfficialRHVAESampler


@dataclass
class GenerationConfig:
    """Configuration for image generation."""
    
    # Generation parameters
    num_samples: int = 64
    sequence_length: int = 16
    batch_size: int = 8
    
    # Sampling method
    sampling_method: str = "geodesic"  # geodesic, enhanced, basic, standard
    sampler_type: str = "working"  # working, hmc, official
    
    # Prior sampling parameters
    temperature: float = 1.0
    noise_scale: float = 0.1
    
    # Flow parameters (if applicable)
    use_flows: bool = True
    flow_steps: Optional[int] = None  # Use model default if None
    
    # Output parameters
    output_format: str = "tensor"  # tensor, numpy, pil
    normalize_output: bool = True
    clamp_output: bool = True
    
    # Memory management
    max_batch_size: int = 32
    clear_cache: bool = True


def dictconfig_to_generation_config(cfg: Union[DictConfig, dict, None]) -> GenerationConfig:
    if cfg is None:
        return GenerationConfig()
    if isinstance(cfg, GenerationConfig):
        return cfg
    # Convert DictConfig or dict to GenerationConfig
    return GenerationConfig(
        num_samples=cfg.get('num_samples', 64),
        sequence_length=cfg.get('sequence_length', 16),
        batch_size=cfg.get('batch_size', 8),
        sampling_method=cfg.get('sampling_method', 'geodesic'),
        sampler_type=cfg.get('sampler_type', 'working'),
        temperature=cfg.get('temperature', 1.0),
        noise_scale=cfg.get('noise_scale', 0.1),
        use_flows=cfg.get('use_flows', True),
        flow_steps=cfg.get('flow_steps', None),
        output_format=cfg.get('output_format', 'tensor'),
        normalize_output=cfg.get('normalize_output', True),
        clamp_output=cfg.get('clamp_output', True),
        max_batch_size=cfg.get('max_batch_size', 32),
        clear_cache=cfg.get('clear_cache', True),
    )

class RlVAEGenerator:
    """
    Unified generation interface for RlVAE models.
    
    Provides high-level methods for generating images using various sampling
    strategies while handling memory management and batch processing.
    """
    
    def __init__(self, model: ModularRiemannianFlowVAE, device: Optional[torch.device] = None):
        """
        Initialize generator with a trained model.
        
        Args:
            model: Trained modular RlVAE model
            device: Device for computation
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Initialize samplers
        self._setup_samplers()
        
        # Generation cache
        self._generation_cache = {}
        
    def _setup_samplers(self):
        """Setup different sampler types."""
        print("🎯 Setting up generation samplers...")
        
        # Initialize all available samplers
        self.samplers = {
            'working': WorkingRiemannianSampler(self.model),
            'hmc': RiemannianHMCSampler(self.model),
        }
        
        # Try to add official sampler if available
        try:
            self.samplers['official'] = OfficialRHVAESampler(self.model)
            print("   ✅ Official RHVAE sampler available")
        except Exception as e:
            print(f"   ⚠️ Official RHVAE sampler not available: {e}")
        
        print(f"   📦 Available samplers: {list(self.samplers.keys())}")
    
    def generate_from_prior(self, config: Union[GenerationConfig, DictConfig, dict, None]) -> Dict[str, torch.Tensor]:
        """
        Generate images by sampling from the learned prior.
        
        Args:
            config: Generation configuration
            
        Returns:
            Dictionary containing generated images and intermediate results
        """
        config = dictconfig_to_generation_config(config)
        print(f"🎨 Generating {config.num_samples} samples using {config.sampling_method} method...")
        
        # Get appropriate sampler
        if config.sampler_type not in self.samplers:
            raise ValueError(f"Sampler type '{config.sampler_type}' not available. "
                           f"Available: {list(self.samplers.keys())}")
        
        sampler = self.samplers[config.sampler_type]
        
        # Generate samples in batches
        all_samples = []
        all_latents = []
        
        num_batches = (config.num_samples + config.batch_size - 1) // config.batch_size
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * config.batch_size
                end_idx = min(start_idx + config.batch_size, config.num_samples)
                batch_size = end_idx - start_idx
                
                print(f"   🔄 Batch {batch_idx + 1}/{num_batches}: {batch_size} samples")
                
                # Sample from prior
                z_prior = sampler.sample_prior(batch_size, method=config.sampling_method)
                
                # Generate sequence if flows are enabled
                if config.use_flows and self.model.n_flows > 0:
                    z_sequence = self._generate_sequence(z_prior, config.sequence_length)
                else:
                    # Single frame generation
                    z_sequence = z_prior.unsqueeze(1)  # [batch_size, 1, latent_dim]
                
                # Decode to images
                batch_images = self._decode_sequence(z_sequence)
                
                all_samples.append(batch_images)
                all_latents.append(z_sequence)
                
                # Clear cache if requested
                if config.clear_cache:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Concatenate all batches
        generated_images = torch.cat(all_samples, dim=0)
        generated_latents = torch.cat(all_latents, dim=0)
        
        # Post-process images
        generated_images = self._postprocess_images(generated_images, config)
        
        print(f"   ✅ Generated {len(generated_images)} image sequences")
        
        return {
            'images': generated_images,
            'latents': generated_latents,
            'config': config,
            'generation_info': {
                'sampling_method': config.sampling_method,
                'sampler_type': config.sampler_type,
                'sequence_length': config.sequence_length,
                'num_flows': self.model.n_flows,
            }
        }
    
    def generate_samples(self, n_samples: int, batch_size: int = 32, 
                        method: str = "geodesic", **kwargs) -> torch.Tensor:
        """
        Simple interface for generating samples (used by evaluation system).
        
        Args:
            n_samples: Number of samples to generate
            batch_size: Batch size for generation
            method: Sampling method to use
            **kwargs: Additional generation parameters
            
        Returns:
            Generated images tensor [n_samples, C, H, W]
        """
        # Accept config overrides from kwargs
        config = GenerationConfig(
            num_samples=n_samples,
            batch_size=batch_size,
            sampling_method=method,
            **kwargs
        )
        return self.generate_from_prior(config)['images']
    
    def generate_from_latents(self, latents: torch.Tensor, 
                             config: Optional[GenerationConfig] = None) -> Dict[str, torch.Tensor]:
        """
        Generate images from given latent codes.
        
        Args:
            latents: Latent codes [batch_size, sequence_length, latent_dim] or [batch_size, latent_dim]
            config: Generation configuration (optional)
            
        Returns:
            Dictionary containing generated images and metadata
        """
        if config is None:
            config = GenerationConfig()
        
        # Handle different latent formats
        if latents.dim() == 2:
            # Single frame: [batch_size, latent_dim] -> [batch_size, 1, latent_dim]
            latents = latents.unsqueeze(1)
        elif latents.dim() != 3:
            raise ValueError(f"Expected latents shape [B, T, D] or [B, D], got {latents.shape}")
        
        batch_size, sequence_length, latent_dim = latents.shape
        
        print(f"🎨 Generating images from {batch_size} latent sequences (length {sequence_length})...")
        
        with torch.no_grad():
            # Process in batches if needed
            if batch_size > config.max_batch_size:
                all_images = []
                num_batches = (batch_size + config.max_batch_size - 1) // config.max_batch_size
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * config.max_batch_size
                    end_idx = min(start_idx + config.max_batch_size, batch_size)
                    
                    batch_latents = latents[start_idx:end_idx]
                    batch_images = self._decode_sequence(batch_latents)
                    all_images.append(batch_images)
                
                generated_images = torch.cat(all_images, dim=0)
            else:
                generated_images = self._decode_sequence(latents)
        
        # Post-process images
        generated_images = self._postprocess_images(generated_images, config)
        
        print(f"   ✅ Generated {len(generated_images)} image sequences")
        
        return {
            'images': generated_images,
            'latents': latents,
            'config': config,
        }
    
    def interpolate(self, latent1: torch.Tensor, latent2: torch.Tensor,
                   num_steps: int = 10, method: str = "linear") -> Dict[str, torch.Tensor]:
        """
        Generate interpolation between two latent points.
        
        Args:
            latent1: First latent point [latent_dim]
            latent2: Second latent point [latent_dim]
            num_steps: Number of interpolation steps
            method: Interpolation method ("linear", "spherical", "geodesic")
            
        Returns:
            Dictionary containing interpolated images and latents
        """
        print(f"🔄 Creating {num_steps}-step {method} interpolation...")
        
        # Ensure latents are 2D
        if latent1.dim() == 1:
            latent1 = latent1.unsqueeze(0)
        if latent2.dim() == 1:
            latent2 = latent2.unsqueeze(0)
        
        with torch.no_grad():
            if method == "linear":
                # Linear interpolation
                alphas = torch.linspace(0, 1, num_steps, device=self.device)
                interpolated_latents = []
                
                for alpha in alphas:
                    latent_interp = (1 - alpha) * latent1 + alpha * latent2
                    interpolated_latents.append(latent_interp)
                
            elif method == "spherical":
                # Spherical linear interpolation (SLERP)
                interpolated_latents = self._slerp(latent1, latent2, num_steps)
                
            elif method == "geodesic" and hasattr(self.model, 'G'):
                # Geodesic interpolation using the learned metric
                interpolated_latents = self._geodesic_interpolation(latent1, latent2, num_steps)
                
            else:
                print(f"   ⚠️ Method '{method}' not available, using linear interpolation")
                alphas = torch.linspace(0, 1, num_steps, device=self.device)
                interpolated_latents = []
                
                for alpha in alphas:
                    latent_interp = (1 - alpha) * latent1 + alpha * latent2
                    interpolated_latents.append(latent_interp)
            
            # Stack latents
            latents = torch.cat(interpolated_latents, dim=0)  # [num_steps, latent_dim]
            
            # Generate images
            config = GenerationConfig(num_samples=num_steps, sequence_length=1)
            result = self.generate_from_latents(latents, config)
        
        print(f"   ✅ Generated {num_steps} interpolation steps")
        
        return {
            'images': result['images'],
            'latents': latents,
            'interpolation_method': method,
            'num_steps': num_steps,
        }
    
    def _generate_sequence(self, z_initial: torch.Tensor, sequence_length: int) -> torch.Tensor:
        """
        Generate a sequence using flows.
        
        Args:
            z_initial: Initial latent codes [batch_size, latent_dim]
            sequence_length: Length of sequence to generate
            
        Returns:
            Generated sequence [batch_size, sequence_length, latent_dim]
        """
        if sequence_length == 1:
            return z_initial.unsqueeze(1)
        
        if not hasattr(self.model, 'flow_manager') or self.model.n_flows == 0:
            # No flows available - replicate initial state
            return z_initial.unsqueeze(1).repeat(1, sequence_length, 1)
        
        # Generate sequence using flows
        z_seq = [z_initial]
        
        for t in range(1, sequence_length):
            z_next, _ = self.model.flow_manager.apply_flows([z_seq[-1]], n_obs=2)
            z_seq.append(z_next[-1])  # Get the last (new) state
        
        return torch.stack(z_seq, dim=1)
    
    def _decode_sequence(self, z_sequence: torch.Tensor) -> torch.Tensor:
        """
        Decode latent sequence to images.
        
        Args:
            z_sequence: Latent sequence [batch_size, sequence_length, latent_dim]
            
        Returns:
            Decoded images [batch_size, sequence_length, channels, height, width]
        """
        batch_size, sequence_length, latent_dim = z_sequence.shape
        
        # Flatten for decoding
        z_flat = z_sequence.view(-1, latent_dim)
        
        # Decode using model decoder
        decoder_out = self.model.decoder(z_flat)
        
        # Handle different decoder output formats
        if hasattr(decoder_out, 'reconstruction'):
            decoded_flat = decoder_out.reconstruction
        elif isinstance(decoder_out, dict) and "reconstruction" in decoder_out:
            decoded_flat = decoder_out["reconstruction"]
        else:
            decoded_flat = decoder_out
        
        # Reshape back to sequence format
        img_dims = decoded_flat.shape[1:]  # Get image dimensions
        decoded_sequence = decoded_flat.view(batch_size, sequence_length, *img_dims)
        
        return decoded_sequence
    
    def _postprocess_images(self, images: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        """
        Post-process generated images.
        
        Args:
            images: Generated images [batch_size, sequence_length, channels, height, width]
            config: Generation configuration
            
        Returns:
            Post-processed images
        """
        if config.clamp_output:
            images = torch.clamp(images, 0, 1)
        
        if config.normalize_output:
            # Ensure images are in [0, 1] range
            images = (images - images.min()) / (images.max() - images.min() + 1e-8)
        
        return images
    
    def _slerp(self, latent1: torch.Tensor, latent2: torch.Tensor, num_steps: int) -> List[torch.Tensor]:
        """Spherical linear interpolation."""
        latent1_norm = F.normalize(latent1, dim=-1)
        latent2_norm = F.normalize(latent2, dim=-1)
        
        # Compute angle
        dot = torch.sum(latent1_norm * latent2_norm, dim=-1, keepdim=True)
        dot = torch.clamp(dot, -1, 1)
        omega = torch.acos(dot)
        
        interpolated = []
        for i in range(num_steps):
            t = i / (num_steps - 1)
            interp = (torch.sin((1 - t) * omega) * latent1_norm + 
                     torch.sin(t * omega) * latent2_norm) / torch.sin(omega)
            interpolated.append(interp)
        
        return interpolated
    
    def _geodesic_interpolation(self, latent1: torch.Tensor, latent2: torch.Tensor, 
                               num_steps: int) -> List[torch.Tensor]:
        """Geodesic interpolation using the learned Riemannian metric."""
        # Simplified geodesic interpolation
        # In practice, this would require solving the geodesic equation
        # For now, use metric-weighted interpolation
        
        interpolated = []
        for i in range(num_steps):
            t = i / (num_steps - 1)
            
            # Linear interpolation as base
            latent_interp = (1 - t) * latent1 + t * latent2
            
            # Apply metric correction (simplified)
            if hasattr(self.model, 'G_inv'):
                try:
                    G_inv = self.model.G_inv(latent_interp)
                    # Use metric for correction (simplified approach)
                    eigenvals, eigenvecs = torch.linalg.eigh(G_inv)
                    # Apply small metric-aware perturbation
                    correction = eigenvecs @ torch.diag_embed(torch.sqrt(eigenvals + 1e-6)) @ eigenvecs.T
                    direction = latent2 - latent1
                    metric_direction = torch.einsum('bij,bj->bi', correction, direction.unsqueeze(0))
                    latent_interp = latent1 + t * metric_direction.squeeze(0)
                except:
                    # Fallback to linear interpolation
                    pass
            
            interpolated.append(latent_interp)
        
        return interpolated
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about the generation capabilities."""
        stats = {
            'model_type': type(self.model).__name__,
            'latent_dim': self.model.latent_dim,
            'input_dim': self.model.input_dim,
            'n_flows': self.model.n_flows,
            'available_samplers': list(self.samplers.keys()),
            'has_metric_tensor': hasattr(self.model, 'G'),
            'device': str(self.device),
        }
        
        # Add sampler-specific info
        for name, sampler in self.samplers.items():
            stats[f'{name}_sampler_methods'] = getattr(sampler, 'available_methods', ['default'])
        
        return stats


def create_generator(model: ModularRiemannianFlowVAE, 
                    device: Optional[torch.device] = None, 
                    config: Union[DictConfig, dict, GenerationConfig, None] = None) -> RlVAEGenerator:
    """
    Factory function to create a generator for a trained model.
    Args:
        model: Trained modular RlVAE model
        device: Device for computation
        config: Hydra DictConfig, dict, or GenerationConfig
    Returns:
        Configured generator instance
    """
    generator = RlVAEGenerator(model=model, device=device)
    # Optionally store config for later use (not required for now)
    generator._hydra_config = config
    return generator 