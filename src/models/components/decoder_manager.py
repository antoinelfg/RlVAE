"""
DecoderManager: Modular Decoder Architecture Handler
===================================================

Handles different decoder architectures with plug-and-play capability.
Supports MLP, CNN, ResNet, and custom architectures via configuration.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union
from omegaconf import DictConfig

from pythae.models.nn import BaseDecoder
from pythae.models.nn.default_architectures import Decoder_AE_MLP

class DecoderManager(nn.Module):
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
        
        # Create decoder based on architecture
        self.decoder = self._create_decoder()
        self.to(self.device)
        
        print(f"✅ Created {architecture.upper()} decoder: {self._get_parameter_count()} parameters")
    
    def _create_decoder(self) -> BaseDecoder:
        """Create decoder based on architecture type."""
        if self.architecture.lower() == "mlp":
            return self._create_mlp_decoder()
        elif self.architecture.lower() == "cnn":
            return self._create_cnn_decoder()
        elif self.architecture.lower() == "resnet":
            return self._create_resnet_decoder()
        elif self.architecture.lower() == "custom":
            return self._create_custom_decoder()
        else:
            raise ValueError(f"Unknown decoder architecture: {self.architecture}")
    
    def _create_mlp_decoder(self) -> BaseDecoder:
        """Create MLP decoder (default VAE architecture)."""
        from types import SimpleNamespace
        
        decoder_config = SimpleNamespace()
        decoder_config.input_dim = self.input_dim
        decoder_config.latent_dim = self.latent_dim
        
        # Add custom MLP parameters if provided
        if hasattr(self.config, 'mlp'):
            decoder_config.hidden_dims = self.config.mlp.get('hidden_dims', [512, 512, 512])
            decoder_config.dropout = self.config.mlp.get('dropout', 0.1)
        
        return Decoder_AE_MLP(decoder_config)
    
    def _create_cnn_decoder(self) -> BaseDecoder:
        """Create CNN decoder for image data."""
        class CNNDecoder(BaseDecoder):
            def __init__(self, input_dim, latent_dim, config):
                super().__init__()
                self.input_dim = input_dim
                self.latent_dim = latent_dim
                
                # CNN parameters
                hidden_dims = config.get('hidden_dims', [256, 128, 64, 32])
                kernel_size = config.get('kernel_size', 3)
                stride = config.get('stride', 2)
                padding = config.get('padding', 1)
                output_padding = config.get('output_padding', 1)
                dropout = config.get('dropout', 0.1)
                
                # Calculate initial size
                self.initial_size = self._calculate_initial_size(input_dim, len(hidden_dims))
                
                # MLP to expand latent to initial features
                self.mlp = nn.Sequential(
                    nn.Linear(latent_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, hidden_dims[0] * self.initial_size[0] * self.initial_size[1])
                )
                
                # Transposed CNN layers
                layers = []
                in_channels = hidden_dims[0]
                
                for i, h_dim in enumerate(hidden_dims[1:]):
                    layers.extend([
                        nn.ConvTranspose2d(in_channels, h_dim, kernel_size, stride, padding, output_padding),
                        nn.BatchNorm2d(h_dim),
                        nn.ReLU(),
                        nn.Dropout2d(dropout)
                    ])
                    in_channels = h_dim
                
                # Final output layer
                layers.append(nn.ConvTranspose2d(in_channels, input_dim[0], kernel_size, stride, padding, output_padding))
                
                self.cnn = nn.Sequential(*layers)
            
            def _calculate_initial_size(self, input_dim, num_layers):
                """Calculate the initial feature map size."""
                h, w = input_dim[1], input_dim[2]
                for _ in range(num_layers):
                    h = h // 2
                    w = w // 2
                return (h, w)
            
            def forward(self, z):
                # Expand latent to initial features
                batch_size = z.size(0)
                hidden = self.mlp(z)
                hidden = hidden.view(batch_size, -1, self.initial_size[0], self.initial_size[1])
                
                # Transposed CNN
                reconstruction = self.cnn(hidden)
                
                return type('obj', (object,), {
                    'reconstruction': reconstruction
                })
        
        return CNNDecoder(self.input_dim, self.latent_dim, self.config.get('cnn', {}))
    
    def _create_resnet_decoder(self) -> BaseDecoder:
        """Create ResNet decoder for image data."""
        class ResNetDecoder(BaseDecoder):
            def __init__(self, input_dim, latent_dim, config):
                super().__init__()
                self.input_dim = input_dim
                self.latent_dim = latent_dim
                
                # ResNet parameters
                hidden_dims = config.get('hidden_dims', [512, 256, 128, 64])
                num_blocks = config.get('num_blocks', 2)
                dropout = config.get('dropout', 0.1)
                
                # Calculate initial size
                self.initial_size = self._calculate_initial_size(input_dim, len(hidden_dims))
                
                # MLP to expand latent to initial features
                self.mlp = nn.Sequential(
                    nn.Linear(latent_dim, 1024),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(512, hidden_dims[0] * self.initial_size[0] * self.initial_size[1])
                )
                
                # Initial upsampling
                self.initial_upsample = nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[0], hidden_dims[0], 4, 2, 1),
                    nn.BatchNorm2d(hidden_dims[0]),
                    nn.ReLU()
                )
                
                # ResNet blocks
                self.resnet_blocks = nn.ModuleList()
                for i in range(len(hidden_dims) - 1):
                    block = self._create_resnet_block(
                        hidden_dims[i], hidden_dims[i+1], num_blocks
                    )
                    self.resnet_blocks.append(block)
                
                # Final output layer
                self.final_conv = nn.ConvTranspose2d(hidden_dims[-1], input_dim[0], 4, 2, 1)
            
            def _calculate_initial_size(self, input_dim, num_layers):
                """Calculate the initial feature map size."""
                h, w = input_dim[1], input_dim[2]
                for _ in range(num_layers):
                    h = h // 2
                    w = w // 2
                return (h, w)
            
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
                    nn.ConvTranspose2d(in_channels, out_channels, 3, stride, 1, stride-1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.ConvTranspose2d(out_channels, out_channels, 3, 1, 1),
                    nn.BatchNorm2d(out_channels)
                )
            
            def forward(self, z):
                # Expand latent to initial features
                batch_size = z.size(0)
                hidden = self.mlp(z)
                hidden = hidden.view(batch_size, -1, self.initial_size[0], self.initial_size[1])
                
                # Initial upsampling
                x = self.initial_upsample(hidden)
                
                # ResNet blocks
                for block in self.resnet_blocks:
                    x = block(x)
                
                # Final output
                reconstruction = self.final_conv(x)
                
                return type('obj', (object,), {
                    'reconstruction': reconstruction
                })
        
        return ResNetDecoder(self.input_dim, self.latent_dim, self.config.get('resnet', {}))
    
    def _create_custom_decoder(self) -> BaseDecoder:
        """Create custom decoder from user-provided configuration."""
        if 'custom_decoder' not in self.config:
            raise ValueError("Custom decoder configuration not provided")
        
        # This would be implemented based on user's custom architecture
        # For now, fallback to MLP
        print("⚠️ Custom decoder not implemented, falling back to MLP")
        return self._create_mlp_decoder()
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through decoder."""
        return self.decoder(z)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        output = self.decoder(z)
        return output.reconstruction
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get information about the decoder architecture."""
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
        """Load pretrained decoder weights with backward compatibility."""
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
                print(f"✅ Loaded pretrained decoder from: {path}")
                return
            except:
                pass
            
            # Try loading directly into the decoder (old naming convention)
            try:
                self.decoder.load_state_dict(state_dict, strict=True)
                print(f"✅ Loaded pretrained decoder from: {path} (legacy format)")
                return
            except:
                pass
            
            # Try with decoder prefix (new naming convention)
            try:
                prefixed_state_dict = {}
                for key, value in state_dict.items():
                    if not key.startswith('decoder.'):
                        prefixed_state_dict[f'decoder.{key}'] = value
                    else:
                        prefixed_state_dict[key] = value
                
                self.load_state_dict(prefixed_state_dict, strict=True)
                print(f"✅ Loaded pretrained decoder from: {path} (with prefix)")
                return
            except:
                pass
            
            # Try removing decoder prefix if present
            try:
                unprefixed_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('decoder.'):
                        unprefixed_state_dict[key[8:]] = value  # Remove 'decoder.' prefix
                    else:
                        unprefixed_state_dict[key] = value
                
                self.decoder.load_state_dict(unprefixed_state_dict, strict=True)
                print(f"✅ Loaded pretrained decoder from: {path} (removed prefix)")
                return
            except:
                pass
            
            # If all attempts fail, try partial loading
            try:
                self.load_state_dict(state_dict, strict=False)
                print(f"⚠️ Loaded pretrained decoder from: {path} (partial load)")
                return
            except Exception as e:
                print(f"❌ Failed to load pretrained decoder: {e}")
                
        except Exception as e:
            print(f"❌ Failed to load pretrained decoder: {e}")
    
    def save_pretrained(self, path: str) -> None:
        """Save decoder weights."""
        try:
            torch.save(self.state_dict(), path)
            print(f"✅ Saved decoder to: {path}")
        except Exception as e:
            print(f"⚠️ Failed to save decoder: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get decoder configuration."""
        return {
            'architecture': self.architecture,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'device': str(self.device),
            'config': dict(self.config) if self.config else {}
        } 