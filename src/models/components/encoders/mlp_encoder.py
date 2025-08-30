"""
MLP Encoder implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from ...base import Encoder, register
from ...base.mixins import LoggingMixin, DeviceMixin


@register("mlp_encoder")
class MLPEncoder(Encoder, LoggingMixin, DeviceMixin):
    """MLP-based encoder for VAE."""
    
    def __init__(
        self,
        input_dim: Tuple[int, ...],
        latent_dim: int,
        hidden_dims: List[int] = [1024, 512, 256],
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        LoggingMixin.__init__(self)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Flatten input dimension
        self.input_size = 1
        for dim in input_dim:
            self.input_size *= dim
        
        # Build MLP layers
        layers = []
        prev_dim = self.input_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Dropout(dropout),
                self._get_activation(activation)
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Output layers for mean and log variance
        self.mu_head = nn.Linear(prev_dim, latent_dim)
        self.log_var_head = nn.Linear(prev_dim, latent_dim)
        
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
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (B, ...)
            
        Returns:
            Dictionary containing:
            - 'embedding': (B, D) encoded representation
            - 'mu': (B, D) mean of posterior
            - 'log_var': (B, D) log variance of posterior
        """
        # Flatten input
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        # Encode through MLP
        hidden = self.mlp(x_flat)
        
        # Get mean and log variance
        mu = self.mu_head(hidden)
        log_var = self.log_var_head(hidden)
        
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
