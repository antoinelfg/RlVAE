"""
Modular Vanilla VAE
==================

Standalone vanilla VAE implementation with modular encoder/decoder support.
Supports MLP, CNN, and ResNet architectures through the modular component system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
from types import SimpleNamespace

from models.components.encoder_manager import EncoderManager
from models.components.decoder_manager import DecoderManager


class ModularVanillaVAE(nn.Module):
    """
    Modular Vanilla VAE with configurable encoder/decoder architectures.
    
    Supports:
    - MLP encoders/decoders (default)
    - CNN encoders/decoders for image data
    - ResNet encoders/decoders for complex image features
    
    This is a pure Vanilla VAE - no flows, no Riemannian geometry, just standard VAE.
    """
    
    def __init__(
        self,
        input_dim: Tuple[int, ...] = (3, 64, 64),
        latent_dim: int = 16,
        encoder_architecture: str = "mlp",
        decoder_architecture: str = "mlp",
        encoder_config: Optional[Dict] = None,
        decoder_config: Optional[Dict] = None,
        beta: float = 1.0,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create encoder using modular manager
        self.encoder_manager = EncoderManager(
            input_dim=input_dim,
            latent_dim=latent_dim,
            architecture=encoder_architecture,
            config=encoder_config or {},
            device=self.device
        )
        self.encoder = self.encoder_manager.encoder
        
        # Create decoder using modular manager
        self.decoder_manager = DecoderManager(
            input_dim=input_dim,
            latent_dim=latent_dim,
            architecture=decoder_architecture,
            config=decoder_config or {},
            device=self.device
        )
        self.decoder = self.decoder_manager.decoder
        
        # Move to device
        self.to(self.device)
        
        print(f"✅ Created ModularVanillaVAE:")
        print(f"   - Input: {input_dim}")
        print(f"   - Latent: {latent_dim}")
        print(f"   - Encoder: {encoder_architecture}")
        print(f"   - Decoder: {decoder_architecture}")
        print(f"   - Beta: {beta}")
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent parameters.
        
        Args:
            x: Input tensor [batch_size, ...input_dim]
            
        Returns:
            mu: Latent mean [batch_size, latent_dim]
            log_var: Latent log variance [batch_size, latent_dim]
        """
        encoder_out = self.encoder(x)
        return encoder_out.embedding, encoder_out.log_covariance
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Latent mean [batch_size, latent_dim]
            log_var: Latent log variance [batch_size, latent_dim]
            
        Returns:
            z: Sampled latents [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to reconstruction.
        
        Args:
            z: Latent codes [batch_size, latent_dim]
            
        Returns:
            recon_x: Reconstructed input [batch_size, ...input_dim]
        """
        decoder_out = self.decoder(z)
        
        # Handle different decoder output formats
        if hasattr(decoder_out, 'reconstruction'):
            # Object with attribute (CNN, ResNet)
            return decoder_out.reconstruction
        elif isinstance(decoder_out, dict) and "reconstruction" in decoder_out:
            # Dictionary format (MLP)
            return decoder_out["reconstruction"]
        elif hasattr(decoder_out, 'recon_x'):
            # Alternative attribute name
            return decoder_out.recon_x
        else:
            # Fallback - assume direct tensor return
            return decoder_out
    
    def forward(self, x: torch.Tensor) -> SimpleNamespace:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor [batch_size, ...input_dim]
            
        Returns:
            SimpleNamespace with:
                - recon_x: Reconstructed input
                - mu: Latent mean
                - log_var: Latent log variance
                - z: Sampled latents
                - loss: Total VAE loss
                - reconstruction_loss: Reconstruction loss
                - reg_loss: KL divergence loss
        """
        # Encode
        mu, log_var = self.encode(x)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        recon_x = self.decode(z)
        
        # Compute losses with 255 scaling (user prefers non-normalized scale)
        # This gives meaningful loss values in the 0-255 range  
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='mean') * 255.0
        
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        
        # Total loss
        total_loss = reconstruction_loss + self.beta * kl_loss
        
        return SimpleNamespace(
            recon_x=recon_x,
            mu=mu,
            log_var=log_var,
            z=z,
            loss=total_loss,
            reconstruction_loss=reconstruction_loss,
            reg_loss=kl_loss
        )
    
    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Generate samples from the VAE.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            samples: Generated samples [num_samples, ...input_dim]
        """
        with torch.no_grad():
            # Sample from standard normal
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            # Decode
            samples = self.decode(z)
            return samples
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation (mean) for input.
        
        Args:
            x: Input tensor [batch_size, ...input_dim]
            
        Returns:
            z_mean: Latent means [batch_size, latent_dim]
        """
        with torch.no_grad():
            mu, _ = self.encode(x)
            return mu
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input through the VAE.
        
        Args:
            x: Input tensor [batch_size, ...input_dim]
            
        Returns:
            recon_x: Reconstructed input [batch_size, ...input_dim]
        """
        with torch.no_grad():
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            recon_x = self.decode(z)
            return recon_x
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get information about model architecture."""
        encoder_info = self.encoder_manager.get_architecture_info()
        decoder_info = self.decoder_manager.get_architecture_info()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'ModularVanillaVAE',
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'beta': self.beta,
            'encoder': encoder_info,
            'decoder': decoder_info,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


def create_cnn_vanilla_vae(
    input_dim: Tuple[int, ...] = (3, 64, 64),
    latent_dim: int = 16,
    beta: float = 1.0
) -> ModularVanillaVAE:
    """
    Create a Vanilla VAE with CNN encoder/decoder.
    
    Args:
        input_dim: Input dimensions
        latent_dim: Latent space dimensionality
        beta: Beta parameter for VAE loss
        
    Returns:
        ModularVanillaVAE with CNN architecture
    """
    encoder_config = {
        'cnn': {
            'hidden_dims': [32, 64, 128, 256],
            'kernel_size': 3,
            'stride': 2,
            'padding': 1,
            'dropout': 0.1
        }
    }
    
    decoder_config = {
        'cnn': {
            'hidden_dims': [256, 128, 64, 32],
            'kernel_size': 3,
            'stride': 2,
            'padding': 1,
            'output_padding': 1,
            'dropout': 0.1
        }
    }
    
    return ModularVanillaVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_architecture="cnn",
        decoder_architecture="cnn",
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        beta=beta
    )


def create_resnet_vanilla_vae(
    input_dim: Tuple[int, ...] = (3, 64, 64),
    latent_dim: int = 16,
    beta: float = 1.0
) -> ModularVanillaVAE:
    """
    Create a Vanilla VAE with ResNet encoder/decoder.
    
    Args:
        input_dim: Input dimensions
        latent_dim: Latent space dimensionality
        beta: Beta parameter for VAE loss
        
    Returns:
        ModularVanillaVAE with ResNet architecture
    """
    encoder_config = {
        'resnet': {
            'hidden_dims': [64, 128, 256],  # Reduced dimensions to match output size
            'num_blocks': 1,  # Reduced blocks to avoid over-upsampling
            'dropout': 0.1
        }
    }
    
    decoder_config = {
        'resnet': {
            'hidden_dims': [256, 128, 64],  # Reduced dimensions to match output size
            'num_blocks': 1,  # Reduced blocks to avoid over-upsampling
            'dropout': 0.1
        }
    }
    
    return ModularVanillaVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_architecture="resnet",
        decoder_architecture="resnet",
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        beta=beta
    )


def create_mlp_vanilla_vae(
    input_dim: Tuple[int, ...] = (3, 64, 64),
    latent_dim: int = 16,
    beta: float = 1.0
) -> ModularVanillaVAE:
    """
    Create a Vanilla VAE with MLP encoder/decoder.
    
    Args:
        input_dim: Input dimensions
        latent_dim: Latent space dimensionality
        beta: Beta parameter for VAE loss
        
    Returns:
        ModularVanillaVAE with MLP architecture
    """
    # More gradual dimension reduction for stability
    # Input: 12288 (64*64*3) -> 4096 -> 1024 -> 512 -> latent_dim
    encoder_config = {
        'mlp': {
            'hidden_dims': [4096, 1024, 512],
            'dropout': 0.2
        }
    }
    
    # Gradual dimension increase for decoder
    # latent_dim -> 512 -> 1024 -> 4096 -> 12288
    decoder_config = {
        'mlp': {
            'hidden_dims': [512, 1024, 4096],
            'dropout': 0.2
        }
    }
    
    return ModularVanillaVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_architecture="mlp",
        decoder_architecture="mlp",
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        beta=beta
    )


# Backward compatibility alias
VanillaVAE = ModularVanillaVAE 