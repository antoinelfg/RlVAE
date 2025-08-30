"""
CNN Encoder implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from ...base import Encoder, register
from ...base.mixins import LoggingMixin, DeviceMixin


@register("cnn_encoder")
class CNNEncoder(Encoder, LoggingMixin, DeviceMixin):
    """CNN-based encoder for VAE."""
    
    def __init__(
        self,
        input_dim: Tuple[int, ...],
        latent_dim: int,
        hidden_dims: List[int] = [32, 64, 128, 256],
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        LoggingMixin.__init__(self)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Assume input is (C, H, W) format
        if len(input_dim) != 3:
            raise ValueError(f"Expected 3D input (C, H, W), got {input_dim}")
        
        self.input_channels = input_dim[0]
        
        # Build CNN layers
        layers = []
        in_channels = self.input_channels
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size, stride, padding),
                nn.BatchNorm2d(hidden_dim),
                nn.Dropout2d(dropout),
                self._get_activation(activation)
            ])
            in_channels = hidden_dim
        
        self.cnn = nn.Sequential(*layers)
        
        # Calculate output size after CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_dim)
            dummy_output = self.cnn(dummy_input)
            self.cnn_output_size = dummy_output.numel() // dummy_output.shape[0]
        
        # Output layers for mean and log variance
        self.mu_head = nn.Linear(self.cnn_output_size, latent_dim)
        self.log_var_head = nn.Linear(self.cnn_output_size, latent_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "leaky_relu":
            return nn.LeakyReLU()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Dictionary containing:
            - 'embedding': (B, D) encoded representation
            - 'mu': (B, D) mean of posterior
            - 'log_var': (B, D) log variance of posterior
        """
        # Encode through CNN
        batch_size = x.shape[0]
        hidden = self.cnn(x)
        
        # Flatten
        hidden_flat = hidden.view(batch_size, -1)
        
        # Get mean and log variance
        mu = self.mu_head(hidden_flat)
        log_var = self.log_var_head(hidden_flat)
        
        # Log some statistics
        self.log("mu_mean", mu.mean().item())
        self.log("mu_std", mu.std().item())
        self.log("log_var_mean", log_var.mean().item())
        self.log("log_var_std", log_var.std().item())
        
        return {
            'embedding': mu,  # Use mu as embedding for compatibility
            'mu': mu,
            'log_var': log_var
        }
