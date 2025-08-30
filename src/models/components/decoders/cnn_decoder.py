"""
CNN Decoder implementation.
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from ...base import Decoder, register
from ...base.mixins import LoggingMixin, DeviceMixin


@register("cnn_decoder")
class CNNDecoder(Decoder, LoggingMixin, DeviceMixin):
    """CNN-based decoder for VAE."""
    
    def __init__(
        self,
        input_dim: Tuple[int, ...],
        latent_dim: int,
        hidden_dims: List[int] = [256, 128, 64, 32],
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
        dropout: float = 0.1,
        activation: str = "relu",
        output_activation: str = "sigmoid"
    ):
        super().__init__()
        LoggingMixin.__init__(self)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Assume output is (C, H, W) format
        if len(input_dim) != 3:
            raise ValueError(f"Expected 3D output (C, H, W), got {input_dim}")
        
        self.output_channels = input_dim[0]
        
        # Calculate initial spatial dimensions
        self.initial_height = input_dim[1] // (2 ** len(hidden_dims))
        self.initial_width = input_dim[2] // (2 ** len(hidden_dims))
        
        # Initial projection from latent to spatial features
        self.initial_size = hidden_dims[0] * self.initial_height * self.initial_width
        self.initial_projection = nn.Linear(latent_dim, self.initial_size)
        
        # Build transposed CNN layers
        layers = []
        in_channels = hidden_dims[0]
        
        for i, hidden_dim in enumerate(hidden_dims[1:], 1):
            layers.extend([
                nn.ConvTranspose2d(in_channels, hidden_dim, kernel_size, stride, padding, output_padding),
                nn.BatchNorm2d(hidden_dim),
                nn.Dropout2d(dropout),
                self._get_activation(activation)
            ])
            in_channels = hidden_dim
        
        # Final output layer
        layers.extend([
            nn.ConvTranspose2d(in_channels, self.output_channels, kernel_size, stride, padding, output_padding),
            self._get_activation(output_activation)
        ])
        
        self.cnn = nn.Sequential(*layers)
        
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
        elif activation.lower() == "none":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
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
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.
        
        Args:
            z: Latent tensor of shape (B, D) or (B, T, D)
            
        Returns:
            Reconstructed tensor of shape (B, C, H, W)
        """
        # Handle sequence input
        if z.dim() == 3:
            batch_size, seq_len, latent_dim = z.shape
            z_flat = z.view(batch_size * seq_len, latent_dim)
            is_sequence = True
        else:
            z_flat = z
            is_sequence = False
        
        # Project to spatial features
        hidden = self.initial_projection(z_flat)
        hidden = hidden.view(-1, self.hidden_dims[0], self.initial_height, self.initial_width)
        
        # Decode through transposed CNN
        output = self.cnn(hidden)
        
        # Reshape to original dimensions
        if is_sequence:
            output = output.view(batch_size, seq_len, *self.input_dim)
        else:
            output = output.view(-1, *self.input_dim)
        
        # Log some statistics
        self.log("output_mean", output.mean().item())
        self.log("output_std", output.std().item())
        self.log("output_min", output.min().item())
        self.log("output_max", output.max().item())
        
        return output
