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
import os
from pathlib import Path

from pythae.models.nn import BaseEncoder
from pythae.models.nn.default_architectures import Encoder_VAE_MLP
try:
    from src.models.rhvae_experiment import RGBEncoder as RHVAE_RGBEncoder
except Exception:
    RHVAE_RGBEncoder = None

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
        elif self.architecture.lower() == "rhvae_rgb":
            return self._create_rhvae_rgb_encoder()
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
                
                # CNN parameters - use config directly, not config.get('cnn', {})
                hidden_dims = config.get('layers', [32, 64, 128, 256])
                kernel_size = config.get('kernel_size', 3)
                stride = config.get('stride', 2)
                padding = config.get('padding', 1)
                activation = config.get('activation', 'relu')
                batch_norm = config.get('batch_norm', True)
                dropout = config.get('dropout', 0.1)
                
                # Build CNN layers
                layers = []
                in_channels = input_dim[0] if len(input_dim) == 3 else 1
                
                for h_dim in hidden_dims:
                    layers.extend([
                        nn.Conv2d(in_channels, h_dim, kernel_size, stride, padding),
                        nn.BatchNorm2d(h_dim) if batch_norm else nn.Identity(),
                        nn.ReLU() if activation == 'relu' else nn.LeakyReLU(),
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
                features = features.reshape(features.size(0), -1)
                
                # MLP head
                hidden = self.mlp(features)
                
                # Latent parameters
                embedding = self.embedding(hidden)
                log_covariance = self.log_covariance(hidden)
                
                return type('obj', (object,), {
                    'embedding': embedding,
                    'log_covariance': log_covariance
                })
        
        # Use self.config directly, not self.config.get('cnn', {})
        return CNNEncoder(self.input_dim, self.latent_dim, self.config)
    
    def _create_resnet_encoder(self) -> BaseEncoder:
        """Create ResNet encoder for image data."""
        class ResNetEncoder(BaseEncoder):
            def __init__(self, input_dim, latent_dim, config):
                super().__init__()
                self.input_dim = input_dim
                self.latent_dim = latent_dim
                
                # ResNet parameters (support both 'hidden_dims' and 'layers')
                hidden_dims = config.get('hidden_dims', config.get('layers', [64, 128, 256, 512]))
                num_blocks_cfg = config.get('num_blocks', 2)  # can be int or list
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
                    # Determine number of blocks for this stage
                    if isinstance(num_blocks_cfg, (list, tuple)) and len(num_blocks_cfg) > 0:
                        this_num_blocks = num_blocks_cfg[min(i, len(num_blocks_cfg) - 1)]
                    else:
                        this_num_blocks = int(num_blocks_cfg)
                    block = self._create_resnet_block(
                        hidden_dims[i], hidden_dims[i+1], this_num_blocks
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
                x = x.reshape(x.size(0), -1)
                
                # MLP head
                hidden = self.mlp(x)
                
                # Latent parameters
                embedding = self.embedding(hidden)
                log_covariance = self.log_covariance(hidden)
                
                return type('obj', (object,), {
                    'embedding': embedding,
                    'log_covariance': log_covariance
                })
        
        # Use the encoder config directly; do not expect a nested 'resnet' key
        return ResNetEncoder(self.input_dim, self.latent_dim, self.config)
    
    def _create_custom_encoder(self) -> BaseEncoder:
        """Create custom encoder from user-provided configuration."""
        if 'custom_encoder' not in self.config:
            raise ValueError("Custom encoder configuration not provided")
        
        # This would be implemented based on user's custom architecture
        # For now, fallback to MLP
        print("⚠️ Custom encoder not implemented, falling back to MLP")
        return self._create_mlp_encoder()

    def _create_rhvae_rgb_encoder(self) -> BaseEncoder:
        """Create the exact RHVAE RGB encoder used in Stage A."""
        if RHVAE_RGBEncoder is None:
            raise RuntimeError("RHVAE_RGBEncoder not available")
        from types import SimpleNamespace
        args = SimpleNamespace()
        args.input_dim = self.input_dim
        args.latent_dim = self.latent_dim
        return RHVAE_RGBEncoder(args)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        debug_enabled = os.environ.get("RLVAE_DEBUG") == "1"
        if debug_enabled:
            print(f"[ENCODER DEBUG] Input shape: {x.shape}, min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("[ENCODER DEBUG] Input contains NaN or Inf!")
            # Warn if using MLP for image data
            if self.architecture.lower() == "mlp" and (len(self.input_dim) == 3 and min(self.input_dim) > 1):
                print("[ENCODER WARNING] Using MLP encoder for image data! Consider 'cnn' or 'resnet'.")
        output = self.encoder(x)
        if debug_enabled:
            mu = output.embedding
            log_var = output.log_covariance
            print(f"[ENCODER DEBUG] mu: min={mu.min().item():.4f}, max={mu.max().item():.4f}, mean={mu.mean().item():.4f}, std={mu.std().item():.4f}")
            print(f"[ENCODER DEBUG] log_var: min={log_var.min().item():.4f}, max={log_var.max().item():.4f}, mean={log_var.mean().item():.4f}, std={log_var.std().item():.4f}")
            if torch.isnan(mu).any() or torch.isinf(mu).any():
                print("[ENCODER DEBUG] mu contains NaN or Inf!")
            if torch.isnan(log_var).any() or torch.isinf(log_var).any():
                print("[ENCODER DEBUG] log_var contains NaN or Inf!")
        return output
    
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
            p = Path(path)
            if not p.exists() or not p.is_file():
                raise FileNotFoundError(f"Pretrained encoder file not found: {path}")
            weights = torch.load(path, map_location=self.device, weights_only=False)
            
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
            
            # Try extracting sub-dict from a full model checkpoint
            try:
                if 'model_state_dict' in state_dict:
                    model_sd = state_dict['model_state_dict']
                else:
                    model_sd = state_dict
                # Prefer encoder.* keys
                enc_sd = {}
                for k, v in model_sd.items():
                    if k.startswith('encoder.'):
                        enc_sd[k[8:]] = v
                if enc_sd:
                    self.encoder.load_state_dict(enc_sd, strict=False)
                    print(f"✅ Loaded pretrained encoder from: {path} (extracted from model_state_dict)")
                    return
            except Exception:
                pass

            # If all attempts fail, optionally abort instead of partial load
            from os import environ
            if environ.get("RLVAE_STRICT_PRETRAIN", "0") == "1":
                raise RuntimeError(f"Strict encoder load failed for {path}; aborting due to RLVAE_STRICT_PRETRAIN=1")

            # Try partial loading with diagnostics
            try:
                incompatible = self.load_state_dict(state_dict, strict=False)
                missing = list(getattr(incompatible, 'missing_keys', []))
                unexpected = list(getattr(incompatible, 'unexpected_keys', []))
                print(f"⚠️ Loaded pretrained encoder from: {path} (partial load): missing={len(missing)}, unexpected={len(unexpected)}")
                if missing:
                    print(f"  ↪ missing keys (first 5): {missing[:5]}")
                if unexpected:
                    print(f"  ↪ unexpected keys (first 5): {unexpected[:5]}")
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
