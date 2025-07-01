"""
EncoderManager: Modular Encoder Architecture Handler
===================================================

Handles different encoder architectures with plug-and-play capability.
Supports MLP, CNN, ResNet, and custom architectures via configuration.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union
from omegaconf import DictConfig

from pythae.models.nn import BaseEncoder
from pythae.models.nn.default_architectures import Encoder_VAE_MLP

class EncoderManager(nn.Module):
    def __init__(
        self,
        input_dim: Tuple[int, ...],
        latent_dim: int,
        architecture: str = "mlp",
        config: Optional[DictConfig] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.architecture = architecture
        self.config = config or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create encoder based on architecture
        self.encoder = self._create_encoder()
        self.to(self.device)
        
        print(f"✅ Created {architecture.upper()} encoder: {self._get_parameter_count()} parameters")
    
    def _create_encoder(self) -> BaseEncoder:
        """Create encoder based on architecture type."""
        if self.architecture.lower() == "mlp":
            return self._create_mlp_encoder()
        elif self.architecture.lower() == "cnn":
            return self._create_cnn_encoder()
        elif self.architecture.lower() == "resnet":
            return self._create_resnet_encoder()
        elif self.architecture.lower() == "custom":
            return self._create_custom_encoder()
        else:
            raise ValueError(f"Unknown encoder architecture: {self.architecture}")
    
    def _create_mlp_encoder(self) -> BaseEncoder:
        """Create MLP encoder (default VAE architecture)."""
        from types import SimpleNamespace
        
        encoder_config = SimpleNamespace()
        encoder_config.input_dim = self.input_dim
        encoder_config.latent_dim = self.latent_dim
        
        # Add custom MLP parameters if provided
        if hasattr(self.config, 'mlp'):
            encoder_config.hidden_dims = self.config.mlp.get('hidden_dims', [512, 512, 512])
            encoder_config.dropout = self.config.mlp.get('dropout', 0.1)
        
        return Encoder_VAE_MLP(encoder_config)
    
    def _create_cnn_encoder(self) -> BaseEncoder:
        """Create CNN encoder for image data."""
        class CNNEncoder(BaseEncoder):
            def __init__(self, input_dim, latent_dim, config):
                super().__init__()
                self.input_dim = input_dim
                self.latent_dim = latent_dim
                
                # CNN parameters
                hidden_dims = config.get('hidden_dims', [32, 64, 128, 256])
                kernel_size = config.get('kernel_size', 3)
                stride = config.get('stride', 2)
                padding = config.get('padding', 1)
                dropout = config.get('dropout', 0.1)
                
                # Build CNN layers
                layers = []
                in_channels = input_dim[0] if len(input_dim) == 3 else 1
                
                for h_dim in hidden_dims:
                    layers.extend([
                        nn.Conv2d(in_channels, h_dim, kernel_size, stride, padding),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU(),
                        nn.Dropout2d(dropout)
                    ])
                    in_channels = h_dim
                
                self.cnn = nn.Sequential(*layers)
                
                # Calculate flattened size
                with torch.no_grad():
                    dummy_input = torch.randn(1, *input_dim)
                    dummy_output = self.cnn(dummy_input)
                    flattened_size = dummy_output.view(1, -1).size(1)
                
                # MLP head for latent space
                self.mlp = nn.Sequential(
                    nn.Linear(flattened_size, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                
                # Output layers
                self.embedding = nn.Linear(256, latent_dim)
                self.log_covariance = nn.Linear(256, latent_dim)
            
            def forward(self, x):
                # CNN feature extraction
                features = self.cnn(x)
                features = features.view(features.size(0), -1)
                
                # MLP head
                hidden = self.mlp(features)
                
                # Latent parameters
                embedding = self.embedding(hidden)
                log_covariance = self.log_covariance(hidden)
                
                return type('obj', (object,), {
                    'embedding': embedding,
                    'log_covariance': log_covariance
                })
        
        return CNNEncoder(self.input_dim, self.latent_dim, self.config.get('cnn', {}))
    
    def _create_resnet_encoder(self) -> BaseEncoder:
        """Create ResNet encoder for image data."""
        class ResNetEncoder(BaseEncoder):
            def __init__(self, input_dim, latent_dim, config):
                super().__init__()
                self.input_dim = input_dim
                self.latent_dim = latent_dim
                
                # ResNet parameters
                hidden_dims = config.get('hidden_dims', [64, 128, 256, 512])
                num_blocks = config.get('num_blocks', 2)
                dropout = config.get('dropout', 0.1)
                
                # Initial convolution
                in_channels = input_dim[0] if len(input_dim) == 3 else 1
                self.initial_conv = nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dims[0], 7, 2, 3),
                    nn.BatchNorm2d(hidden_dims[0]),
                    nn.ReLU(),
                    nn.MaxPool2d(3, 2, 1)
                )
                
                # ResNet blocks
                self.resnet_blocks = nn.ModuleList()
                for i in range(len(hidden_dims) - 1):
                    block = self._create_resnet_block(
                        hidden_dims[i], hidden_dims[i+1], num_blocks
                    )
                    self.resnet_blocks.append(block)
                
                # Global average pooling and MLP head
                self.global_pool = nn.AdaptiveAvgPool2d(1)
                
                self.mlp = nn.Sequential(
                    nn.Linear(hidden_dims[-1], 512),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                
                # Output layers
                self.embedding = nn.Linear(256, latent_dim)
                self.log_covariance = nn.Linear(256, latent_dim)
            
            def _create_resnet_block(self, in_channels, out_channels, num_blocks):
                """Create a ResNet block with multiple residual connections."""
                layers = []
                for i in range(num_blocks):
                    stride = 2 if i == 0 and in_channels != out_channels else 1
                    layers.append(self._create_residual_layer(in_channels, out_channels, stride))
                    in_channels = out_channels
                return nn.Sequential(*layers)
            
            def _create_residual_layer(self, in_channels, out_channels, stride):
                """Create a single residual layer."""
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.BatchNorm2d(out_channels)
                )
            
            def forward(self, x):
                # Initial convolution
                x = self.initial_conv(x)
                
                # ResNet blocks
                for block in self.resnet_blocks:
                    x = block(x)
                
                # Global pooling
                x = self.global_pool(x)
                x = x.view(x.size(0), -1)
                
                # MLP head
                hidden = self.mlp(x)
                
                # Latent parameters
                embedding = self.embedding(hidden)
                log_covariance = self.log_covariance(hidden)
                
                return type('obj', (object,), {
                    'embedding': embedding,
                    'log_covariance': log_covariance
                })
        
        return ResNetEncoder(self.input_dim, self.latent_dim, self.config.get('resnet', {}))
    
    def _create_custom_encoder(self) -> BaseEncoder:
        """Create custom encoder from user-provided configuration."""
        if 'custom_encoder' not in self.config:
            raise ValueError("Custom encoder configuration not provided")
        
        # This would be implemented based on user's custom architecture
        # For now, fallback to MLP
        print("⚠️ Custom encoder not implemented, falling back to MLP")
        return self._create_mlp_encoder()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through encoder."""
        return self.encoder(x)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space parameters."""
        output = self.encoder(x)
        return output.embedding, output.log_covariance
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get information about the encoder architecture."""
        return {
            'architecture': self.architecture,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'parameter_count': self._get_parameter_count(),
            'config': dict(self.config) if self.config else {}
        }
    
    def _get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def load_pretrained(self, path: str) -> None:
        """Load pretrained encoder weights with backward compatibility."""
        try:
            weights = torch.load(path, map_location=self.device)
            
            # Handle different weight formats
            if hasattr(weights, 'state_dict'):
                state_dict = weights.state_dict()
            else:
                state_dict = weights
            
            # Try loading with current naming convention first
            try:
                self.load_state_dict(state_dict, strict=True)
                print(f"✅ Loaded pretrained encoder from: {path}")
                return
            except:
                pass
            
            # Try loading directly into the encoder (old naming convention)
            try:
                self.encoder.load_state_dict(state_dict, strict=True)
                print(f"✅ Loaded pretrained encoder from: {path} (legacy format)")
                return
            except:
                pass
            
            # Try with encoder prefix (new naming convention)
            try:
                prefixed_state_dict = {}
                for key, value in state_dict.items():
                    if not key.startswith('encoder.'):
                        prefixed_state_dict[f'encoder.{key}'] = value
                    else:
                        prefixed_state_dict[key] = value
                
                self.load_state_dict(prefixed_state_dict, strict=True)
                print(f"✅ Loaded pretrained encoder from: {path} (with prefix)")
                return
            except:
                pass
            
            # Try removing encoder prefix if present
            try:
                unprefixed_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('encoder.'):
                        unprefixed_state_dict[key[8:]] = value  # Remove 'encoder.' prefix
                    else:
                        unprefixed_state_dict[key] = value
                
                self.encoder.load_state_dict(unprefixed_state_dict, strict=True)
                print(f"✅ Loaded pretrained encoder from: {path} (removed prefix)")
                return
            except:
                pass
            
            # If all attempts fail, try partial loading
            try:
                self.load_state_dict(state_dict, strict=False)
                print(f"⚠️ Loaded pretrained encoder from: {path} (partial load)")
                return
            except Exception as e:
                print(f"❌ Failed to load pretrained encoder: {e}")
                
        except Exception as e:
            print(f"❌ Failed to load pretrained encoder: {e}")
    
    def save_pretrained(self, path: str) -> None:
        """Save encoder weights."""
        try:
            torch.save(self.state_dict(), path)
            print(f"✅ Saved encoder to: {path}")
        except Exception as e:
            print(f"⚠️ Failed to save encoder: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get encoder configuration."""
        return {
            'architecture': self.architecture,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'device': str(self.device),
            'config': dict(self.config) if self.config else {}
        } 