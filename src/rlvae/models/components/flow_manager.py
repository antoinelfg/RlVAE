"""
FlowManager: Modular Normalizing Flow Handler
============================================

Handles initialization, application, inversion, and diagnostics of normalizing flows (e.g., IAF) for Riemannian VAE models.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from pythae.models.normalizing_flows.iaf import IAF, IAFConfig

class FlowManager(nn.Module):
    def __init__(self, latent_dim: int, n_flows: int = 8, flow_hidden_size: int = 256, flow_n_blocks: int = 2, flow_n_hidden: int = 1, device: Optional[torch.device] = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_flows = n_flows
        self.flow_hidden_size = flow_hidden_size
        self.flow_n_blocks = flow_n_blocks
        self.flow_n_hidden = flow_n_hidden
        self.device = device or torch.device('cpu')
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            config = IAFConfig(
                input_dim=(latent_dim,),
                hidden_size=flow_hidden_size,
                n_blocks=flow_n_blocks,
                n_hidden=flow_n_hidden,
            )
            flow = IAF(config)
            self.flows.append(flow)
        self.to(self.device)

    def apply_flows(self, z_sequence, n_obs=None):
        """
        Apply flows to generate temporal sequence.
        
        Args:
            z_sequence: Initial latent codes [batch_size, latent_dim] or list with first element
            n_obs: Number of observations (if None, use z_sequence length)
            
        Returns:
            z_sequence: Updated latent sequence
            log_det_jacobians: List of log determinant Jacobians
        """
        if n_obs is None:
            n_obs = len(z_sequence)
        
        # Ensure z_sequence is a list and has the right number of elements
        if isinstance(z_sequence, list):
            # If it's a list, ensure it has enough elements
            while len(z_sequence) < n_obs:
                z_sequence.append(None)  # Placeholder
        else:
            # If it's a tensor, convert to list
            z_sequence = [z_sequence] + [None] * (n_obs - 1)
        
        batch_size = z_sequence[0].shape[0]
        log_det_jacobians = []
        
        # Apply flows to generate sequence
        for t in range(1, n_obs):
            # Get flow for this timestep (cycle if more timesteps than flows)
            flow_idx = (t - 1) % len(self.flows)
            flow = self.flows[flow_idx]
            
            # Apply flow to previous latent
            flow_result = flow(z_sequence[t-1])
            z_t = flow_result.out
            log_det_jac = flow_result.log_abs_det_jac
            
            # Store results
            z_sequence[t] = z_t
            log_det_jacobians.append(log_det_jac)
            
            # 🔍 DEBUG: Verify Jacobian computation
            if hasattr(self, '_debug_flow_counter'):
                self._debug_flow_counter += 1
            else:
                self._debug_flow_counter = 1
                
            if self._debug_flow_counter % 100 == 0:
                with torch.no_grad():
                    print(f"🔍 FLOW DEBUG (step {t}):")
                    print(f"   Input z range: [{z_sequence[t-1].min():.3f}, {z_sequence[t-1].max():.3f}]")
                    print(f"   Output z range: [{z_t.min():.3f}, {z_t.max():.3f}]")
                    print(f"   Log det Jacobian: {log_det_jac.mean():.3f}")
                    print(f"   Jacobian range: [{log_det_jac.min():.3f}, {log_det_jac.max():.3f}]")
        
        return z_sequence, log_det_jacobians

    def invert_flows(self, z_seq: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Invert flows sequentially (if supported).
        Args:
            z_seq: List of latent tensors (one per timestep)
        Returns:
            z_seq_inv: List of inverted latents
        """
        # NOTE: IAF is not easily invertible; this is a placeholder for future invertible flows
        raise NotImplementedError("Invert flows is not implemented for IAF.")

    def get_log_det_jacobians(self, z_seq: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Get log|det J| for each flow given a sequence of latents.
        Args:
            z_seq: List of latent tensors (one per timestep)
        Returns:
            log_det_jacobians: List of log|det J| for each flow
        """
        _, log_det_jacobians = self.apply_flows(z_seq)
        return log_det_jacobians

    def get_flow_params(self) -> Dict[str, Any]:
        """
        Get flow configuration parameters.
        Returns:
            Dictionary of flow parameters
        """
        return {
            'latent_dim': self.latent_dim,
            'n_flows': self.n_flows,
            'flow_hidden_size': self.flow_hidden_size,
            'flow_n_blocks': self.flow_n_blocks,
            'flow_n_hidden': self.flow_n_hidden
        }

    def diagnose_flows(self) -> Dict[str, Any]:
        """
        Diagnostics for the flows (e.g., parameter count).
        Returns:
            Dictionary of diagnostics
        """
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'total_params': total_params,
            'n_flows': self.n_flows
        } 