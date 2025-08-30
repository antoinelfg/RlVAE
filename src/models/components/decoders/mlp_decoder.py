"""
MLP Decoder implementation.
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from ...base import Decoder, register
from ...base.mixins import LoggingMixin, DeviceMixin


@register("mlp_decoder")
class MLPDecoder(Decoder, LoggingMixin, DeviceMixin):
    """MLP-based decoder for VAE."""
    
    def __init__(
        self,
        input_dim: Tuple[int, ...],
        latent_dim: int,
        hidden_dims: List[int] = [256, 512, 1024],
        dropout: float = 0.1,
        activation: str = "relu",
        output_activation: str = "sigmoid"
    ):
        super().__init__()
        LoggingMixin.__init__(self)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Calculate output size
        self.output_size = 1
        for dim in input_dim:
            self.output_size *= dim
        
        # Build MLP layers
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Dropout(dropout),
                self._get_activation(activation)
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Output layer
        self.output_head = nn.Linear(prev_dim, self.output_size)
        
        # Output activation
        self.output_activation = self._get_activation(output_activation)
        
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
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.
        
        Args:
            z: Latent tensor of shape (B, D) or (B, T, D)
            
        Returns:
            Reconstructed tensor of shape (B, ...)
        """
        # Handle sequence input
        if z.dim() == 3:
            batch_size, seq_len, latent_dim = z.shape
            z_flat = z.view(batch_size * seq_len, latent_dim)
            is_sequence = True
        else:
            z_flat = z
            is_sequence = False
        
        # Decode through MLP
        hidden = self.mlp(z_flat)
        output_flat = self.output_head(hidden)
        output = self.output_activation(output_flat)
        
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
