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

    def apply_flows(self, z_seq: list, n_obs: int = None):
        """
        Apply flows sequentially to a sequence of latent variables.
        If z_seq is length 1 and n_obs is provided, generate the full sequence of length n_obs.
        Args:
            z_seq: List of latent tensors (one per timestep)
            n_obs: Number of timesteps (optional, for temporal evolution)
        Returns:
            z_seq_out: List of transformed latents
            log_det_jacobians: List of log|det J| for each flow
        """
        if n_obs is not None and len(z_seq) == 1:
            # Temporal evolution: generate sequence
            z_seq_out = [z_seq[0]]
            log_det_jacobians = []
            for t in range(1, n_obs):
                flow = self.flows[t-1] if t-1 < len(self.flows) else self.flows[-1]
                flow_res = flow(z_seq_out[-1])
                z_t = flow_res.out
                log_det = flow_res.log_abs_det_jac
                z_seq_out.append(z_t)
                log_det_jacobians.append(log_det)
            return z_seq_out, log_det_jacobians
        else:
            # Standard: apply flows to provided sequence
            z_seq_out = [z_seq[0]]
            log_det_jacobians = []
            for t in range(1, len(z_seq)):
                flow = self.flows[t-1]
                flow_res = flow(z_seq_out[-1])
                z_t = flow_res.out
                log_det = flow_res.log_abs_det_jac
                z_seq_out.append(z_t)
                log_det_jacobians.append(log_det)
            return z_seq_out, log_det_jacobians

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