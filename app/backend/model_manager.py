"""
Model Manager for Streamlit VAE App
==================================

Handles loading, managing, and interfacing with VAE models and their components.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
from typing import Dict, Any, Optional, List, Tuple
from omegaconf import DictConfig, OmegaConf
import numpy as np
from io import BytesIO
import streamlit as st

# Add src to path for imports
current_dir = Path(__file__).parent.parent.parent.absolute()
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from models.modular_rlvae import ModularRiemannianFlowVAE, ModelFactory
from data.cyclic_dataset import CyclicSpritesDataModule


class ModelManager:
    """Manages VAE models and their components for the Streamlit app."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, ModularRiemannianFlowVAE] = {}
        self.current_model: Optional[ModularRiemannianFlowVAE] = None
        self.current_model_name: Optional[str] = None
        self.data_module: Optional[CyclicSpritesDataModule] = None
        
        # Pre-trained component paths
        self.pretrained_paths = {
            'encoder': current_dir / "data" / "pretrained" / "encoder.pt",
            'decoder': current_dir / "data" / "pretrained" / "decoder.pt", 
            'metric': current_dir / "data" / "pretrained" / "metric.pt",
            'metric_scaled': current_dir / "data" / "pretrained" / "metric_T0.7_scaled.pt"
        }
        
        # Check which pretrained components are available
        self.available_pretrained = {}
        for name, path in self.pretrained_paths.items():
            self.available_pretrained[name] = path.exists()
            
        print(f"🎯 ModelManager initialized on {self.device}")
        print(f"📦 Available pretrained components: {[k for k, v in self.available_pretrained.items() if v]}")
    
    def load_default_model(self) -> bool:
        """Load the default VAE model with pretrained components."""
        try:
            # Create default configuration
            config = self._create_default_config()
            
            # Load model
            model = ModularRiemannianFlowVAE(config)
            model = model.to(self.device)
            model.eval()
            
            # Store model
            model_name = "ModularRiemannianFlowVAE_default"
            self.models[model_name] = model
            self.current_model = model
            self.current_model_name = model_name
            
            # Load data module
            self._load_data_module(config)
            
            print(f"✅ Default model loaded: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load default model: {e}")
            st.error(f"Failed to load default model: {e}")
            return False
    
    def _create_default_config(self) -> DictConfig:
        """Create default configuration with pretrained components."""
        
        config_dict = {
            # Model architecture
            'input_dim': [3, 64, 64],
            'latent_dim': 10,
            'n_flows': 4,
            'flow_hidden_size': 128,
            'flow_n_blocks': 3,
            'flow_n_hidden': 128,
            'epsilon': 1e-6,
            'beta': 1.0,
            'riemannian_beta': 5.0,
            
            # Model components
            'encoder': {'architecture': 'mlp'},
            'decoder': {'architecture': 'mlp'},
            
            # Posterior configuration
            'posterior': {
                'type': 'riemannian_metric'
            },
            
            # Sampling configuration
            'sampling': {
                'use_riemannian': True,
                'method': 'enhanced'
            },
            
            # Loop configuration
            'loop': {
                'mode': 'closed',
                'penalty': 1.0
            },
            
            # Metric configuration
            'metric': {
                'temperature_override': 0.7,
                'regularization_override': None
            },
            
            # Pretrained paths
            'pretrained': {
                'encoder_path': str(self.pretrained_paths['encoder']) if self.available_pretrained['encoder'] else None,
                'decoder_path': str(self.pretrained_paths['decoder']) if self.available_pretrained['decoder'] else None,
                'metric_path': str(self.pretrained_paths['metric_scaled']) if self.available_pretrained['metric_scaled'] else None
            }
        }
        
        return OmegaConf.create(config_dict)
    
    def _load_data_module(self, config: DictConfig):
        """Load the data module for the model."""
        try:
            # Data paths
            data_config = {
                'train_path': str(current_dir / "src" / "datasprites" / "Sprites_train.pt"),
                'test_path': str(current_dir / "src" / "datasprites" / "Sprites_test.pt"),
                'num_workers': 0,  # Disable multiprocessing for Streamlit
                'pin_memory': False,
                'verify_cyclicity': True,
                'cyclicity_threshold': 0.01
            }
            
            data_config = OmegaConf.create(data_config)
            self.data_module = CyclicSpritesDataModule(data_config)
            
            # Setup with minimal training config
            training_config = OmegaConf.create({
                'data': {
                    'batch_size': 8,
                    'num_workers': 0,
                    'pin_memory': False
                },
                'data_splits': {
                    'train': 0.7,
                    'val': 0.3
                }
            })
            
            self.data_module.setup("fit", training_config)
            print("✅ Data module loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Failed to load data module: {e}")
            self.data_module = None
    
    def load_custom_model(self, config: DictConfig, model_name: str) -> bool:
        """Load a custom model from configuration."""
        try:
            model = ModularRiemannianFlowVAE(config)
            model = model.to(self.device)
            model.eval()
            
            self.models[model_name] = model
            self.current_model = model
            self.current_model_name = model_name
            
            print(f"✅ Custom model loaded: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load custom model: {e}")
            st.error(f"Failed to load custom model: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.models.keys())
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different loaded model."""
        if model_name in self.models:
            self.current_model = self.models[model_name]
            self.current_model_name = model_name
            print(f"✅ Switched to model: {model_name}")
            return True
        else:
            print(f"❌ Model not found: {model_name}")
            return False
    
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode input to latent space."""
        if self.current_model is None:
            raise RuntimeError("No model loaded")
        
        x = x.to(self.device)
        self.current_model.eval()
        
        with torch.no_grad():
            if len(x.shape) == 5:  # [B, T, C, H, W]
                x_input = x[:, 0]  # Take first frame
            else:  # [B, C, H, W]
                x_input = x
            
            encoder_out = self.current_model.encoder(x_input)
            mu = encoder_out.embedding
            log_var = encoder_out.log_covariance
            
            # Sample latent
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * log_var)
            
            return {
                'latent': z,
                'mu': mu,
                'log_var': log_var,
                'std': torch.exp(0.5 * log_var)
            }
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to images."""
        if self.current_model is None:
            raise RuntimeError("No model loaded")
        
        z = z.to(self.device)
        self.current_model.eval()
        
        with torch.no_grad():
            decoder_out = self.current_model.decoder(z)
            return decoder_out["reconstruction"]
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full forward pass through the model."""
        if self.current_model is None:
            raise RuntimeError("No model loaded")
        
        x = x.to(self.device)
        self.current_model.eval()
        
        with torch.no_grad():
            output = self.current_model(x, compute_metrics=True)
            return output
    
    def generate_random_samples(self, num_samples: int = 8, sequence_length: int = 16) -> torch.Tensor:
        """Generate random samples from the model."""
        if self.current_model is None:
            raise RuntimeError("No model loaded")
        
        self.current_model.eval()
        
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, self.current_model.latent_dim).to(self.device)
            
            # If model uses flows, we need to handle sequences
            if self.current_model.n_flows > 0:
                # Create a sequence by applying flows
                z_seq = [z]
                
                # Apply flows to create sequence
                for i in range(sequence_length - 1):
                    if hasattr(self.current_model, 'flows') and self.current_model.flows:
                        z_next = z_seq[-1]
                        for flow in self.current_model.flows:
                            z_next, _ = flow(z_next)
                        z_seq.append(z_next)
                    else:
                        # Fallback: small perturbations
                        z_next = z_seq[-1] + 0.1 * torch.randn_like(z_seq[-1])
                        z_seq.append(z_next)
                
                # Stack into tensor [B, T, latent_dim]
                z_sequence = torch.stack(z_seq, dim=1)
                
                # Decode each frame
                z_flat = z_sequence.reshape(-1, self.current_model.latent_dim)
                reconstructions = self.decode(z_flat)
                
                # Reshape back to [B, T, C, H, W]
                batch_size, seq_len = z_sequence.shape[:2]
                recon_shape = reconstructions.shape[1:]
                reconstructions = reconstructions.view(batch_size, seq_len, *recon_shape)
                
            else:
                # Simple case: decode directly
                reconstructions = self.decode(z)
                if len(reconstructions.shape) == 4:  # [B, C, H, W]
                    reconstructions = reconstructions.unsqueeze(1)  # [B, 1, C, H, W]
            
            return reconstructions
    
    def interpolate_latent(self, z1: torch.Tensor, z2: torch.Tensor, steps: int = 10, method: str = 'linear') -> torch.Tensor:
        """Interpolate between two latent vectors."""
        if self.current_model is None:
            raise RuntimeError("No model loaded")
        
        z1, z2 = z1.to(self.device), z2.to(self.device)
        
        if method == 'linear':
            # Linear interpolation
            alphas = torch.linspace(0, 1, steps).view(-1, 1).to(self.device)
            z_interp = alphas * z2 + (1 - alphas) * z1
            
        elif method == 'spherical':
            # Spherical linear interpolation
            z1_norm = z1 / torch.norm(z1, dim=-1, keepdim=True)
            z2_norm = z2 / torch.norm(z2, dim=-1, keepdim=True)
            
            omega = torch.acos(torch.clamp(torch.sum(z1_norm * z2_norm, dim=-1), -1, 1))
            sin_omega = torch.sin(omega)
            
            alphas = torch.linspace(0, 1, steps).to(self.device)
            z_interp = []
            
            for alpha in alphas:
                if sin_omega.abs() < 1e-6:
                    z_alpha = (1 - alpha) * z1 + alpha * z2
                else:
                    z_alpha = (torch.sin((1 - alpha) * omega) * z1 + torch.sin(alpha * omega) * z2) / sin_omega.unsqueeze(-1)
                z_interp.append(z_alpha)
            
            z_interp = torch.stack(z_interp, dim=0)
            
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        return z_interp
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the current model."""
        if self.current_model is None:
            return {"error": "No model loaded"}
        
        try:
            info = {
                'model_name': self.current_model_name,
                'device': str(self.device),
                'model_summary': self.current_model.get_model_summary(),
                'parameters': {
                    'total_params': sum(p.numel() for p in self.current_model.parameters()),
                    'trainable_params': sum(p.numel() for p in self.current_model.parameters() if p.requires_grad)
                },
                'architecture': {
                    'latent_dim': self.current_model.latent_dim,
                    'input_dim': self.current_model.input_dim,
                    'n_flows': self.current_model.n_flows,
                    'has_metric': hasattr(self.current_model, 'G') and self.current_model.G is not None
                },
                'pretrained_components': self.available_pretrained
            }
            
            return info
            
        except Exception as e:
            return {"error": f"Failed to get model info: {e}"}
    
    def get_sample_data(self, split: str = 'train', batch_size: int = 8) -> Optional[torch.Tensor]:
        """Get sample data for visualization."""
        if self.data_module is None:
            return None
        
        try:
            return self.data_module.get_sample_batch(split, batch_size)
        except Exception as e:
            print(f"⚠️ Failed to get sample data: {e}")
            return None
    
    def tensor_to_image_array(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array for display."""
        # Move to CPU and convert to numpy
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        array = tensor.detach().numpy()
        
        # Handle different tensor shapes
        if len(array.shape) == 4:  # [B, C, H, W]
            array = array.transpose(0, 2, 3, 1)  # [B, H, W, C]
        elif len(array.shape) == 5:  # [B, T, C, H, W]
            array = array.transpose(0, 1, 3, 4, 2)  # [B, T, H, W, C]
        elif len(array.shape) == 3:  # [C, H, W]
            array = array.transpose(1, 2, 0)  # [H, W, C]
        
        # Normalize to [0, 1] if needed
        if array.min() < 0 or array.max() > 1:
            array = (array - array.min()) / (array.max() - array.min() + 1e-8)
        
        # Convert to [0, 255]
        array = (array * 255).astype(np.uint8)
        
        return array