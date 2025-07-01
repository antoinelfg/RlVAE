"""
LossManager: Modular Loss Computation Handler
============================================

Handles all loss computations for Riemannian VAE models including:
- Reconstruction loss
- KL divergence loss (standard and Riemannian)
- Flow loss
- Loop penalty loss
- Combined loss computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np

class LossManager(nn.Module):
    def __init__(
        self,
        beta: float = 1.0,
        riemannian_beta: Optional[float] = None,
        loop_penalty_weight: float = 1.0,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.beta = beta
        self.riemannian_beta = riemannian_beta if riemannian_beta is not None else beta
        self.loop_penalty_weight = loop_penalty_weight
        self.device = device or torch.device('cpu')
        self.to(self.device)
        
        # Loss tracking
        self.loss_history = {
            'reconstruction': [],
            'kl_divergence': [],
            'riemannian_kl': [],
            'flow_loss': [],
            'loop_penalty': [],
            'total': []
        }
    
    def compute_reconstruction_loss(self, x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss (MSE for continuous data).
        
        Args:
            x_recon: Reconstructed input [batch_size, *input_shape]
            x: Original input [batch_size, *input_shape]
            
        Returns:
            Reconstruction loss scalar
        """
        return F.mse_loss(x_recon, x, reduction='mean')
    
    def compute_standard_kl_loss(
        self, 
        mu: torch.Tensor, 
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute standard KL divergence loss.
        
        Args:
            mu: Posterior mean [batch_size, latent_dim]
            log_var: Posterior log variance [batch_size, latent_dim]
            
        Returns:
            KL divergence loss scalar
        """
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return kl_loss / mu.shape[0]  # Average over batch
    
    def compute_riemannian_kl_loss(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        z_samples: torch.Tensor,
        metric_tensor: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Compute Riemannian-aware KL divergence loss.
        
        Args:
            mu: Posterior mean [batch_size, latent_dim]
            log_var: Posterior log variance [batch_size, latent_dim]
            z_samples: Sampled latent variables [batch_size, latent_dim]
            metric_tensor: Metric tensor component (optional)
            
        Returns:
            Riemannian KL divergence loss scalar
        """
        if metric_tensor is None:
            # Fallback to standard KL
            return self.compute_standard_kl_loss(mu, log_var)
        
        try:
            # Ensure all tensors are on the same device
            target_device = mu.device
            mu = mu.to(target_device)
            log_var = log_var.to(target_device)
            z_samples = z_samples.to(target_device)
            
            # Ensure metric tensor is on the same device
            if hasattr(metric_tensor, 'to'):
                metric_tensor = metric_tensor.to(target_device)
            
            # Compute metric at posterior mean
            G_inv_mu = metric_tensor.compute_inverse_metric(mu)  # [batch_size, latent_dim, latent_dim]
            
            # Compute log determinant of metric
            log_det_G_inv = metric_tensor.compute_log_det_metric(mu)  # [batch_size]
            
            # Compute Riemannian KL divergence
            # KL = 0.5 * (tr(G_inv * Σ) + (μ-μ_prior)^T * G_inv * (μ-μ_prior) - log|G_inv| - d)
            batch_size, latent_dim = mu.shape
            
            # Prior is standard normal (μ_prior = 0, Σ_prior = I)
            mu_prior = torch.zeros_like(mu)
            sigma_prior = torch.eye(latent_dim, device=target_device).unsqueeze(0).expand(batch_size, -1, -1)
            
            # Posterior covariance
            sigma_post = torch.diag_embed(torch.exp(log_var))  # [batch_size, latent_dim, latent_dim]
            
            # Term 1: tr(G_inv * Σ_post)
            term1 = torch.sum(torch.diagonal(torch.bmm(G_inv_mu, sigma_post), dim1=-2, dim2=-1), dim=-1)
            
            # Term 2: (μ-μ_prior)^T * G_inv * (μ-μ_prior)
            mu_diff = mu - mu_prior  # [batch_size, latent_dim]
            term2 = torch.sum(mu_diff.unsqueeze(1) * torch.bmm(G_inv_mu, mu_diff.unsqueeze(-1)).squeeze(-1), dim=-1)
            
            # Term 3: log|G_inv|
            term3 = log_det_G_inv
            
            # Term 4: -d (dimension)
            term4 = -latent_dim
            
            # Combine terms
            kl_riemannian = 0.5 * (term1 + term2 + term3 + term4)
            
            return torch.mean(kl_riemannian)
            
        except Exception as e:
            print(f"⚠️ Riemannian KL computation failed: {e}, falling back to standard KL")
            return self.compute_standard_kl_loss(mu, log_var)
    
    def compute_flow_loss(
        self, 
        log_det_jacobians: Optional[list] = None
    ) -> torch.Tensor:
        """
        Compute flow loss (negative log determinant of Jacobian).
        
        Args:
            log_det_jacobians: List of log determinants for each flow [n_flows]
            
        Returns:
            Flow loss scalar
        """
        if log_det_jacobians is None or len(log_det_jacobians) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Sum log determinants across flows
        total_log_det = sum(log_det_jacobians)
        return -torch.mean(total_log_det)
    
    def compute_loop_penalty(
        self, 
        z_seq: list, 
        loop_mode: str = "open"
    ) -> torch.Tensor:
        """
        Compute loop penalty for temporal consistency.
        
        Args:
            z_seq: List of latent tensors [n_timesteps]
            loop_mode: "open" or "closed" loop
            
        Returns:
            Loop penalty scalar
        """
        if loop_mode == "open" or len(z_seq) < 2:
            return torch.tensor(0.0, device=self.device)
        
        if loop_mode == "closed":
            # Penalize difference between first and last latent
            z_first = z_seq[0]
            z_last = z_seq[-1]
            penalty = F.mse_loss(z_first, z_last, reduction='mean')
            return penalty * self.loop_penalty_weight
        
        return torch.tensor(0.0, device=self.device)
    
    def compute_total_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        z_samples: torch.Tensor,
        log_det_jacobians: Optional[list] = None,
        z_seq: Optional[list] = None,
        loop_mode: str = "open",
        metric_tensor: Optional[Any] = None,
        use_riemannian_kl: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss combining all components.
        
        Args:
            x: Original input [batch_size, *input_shape]
            x_recon: Reconstructed input [batch_size, *input_shape]
            mu: Posterior mean [batch_size, latent_dim]
            log_var: Posterior log variance [batch_size, latent_dim]
            z_samples: Sampled latent variables [batch_size, latent_dim]
            log_det_jacobians: List of log determinants for flows
            z_seq: List of latent tensors for loop penalty
            loop_mode: Loop mode for temporal consistency
            metric_tensor: Metric tensor component
            use_riemannian_kl: Whether to use Riemannian KL
            
        Returns:
            Dictionary containing all loss components and total
        """
        # Compute individual loss components
        recon_loss = self.compute_reconstruction_loss(x_recon, x)
        
        if use_riemannian_kl and metric_tensor is not None:
            kl_loss = self.compute_riemannian_kl_loss(mu, log_var, z_samples, metric_tensor)
            kl_weight = self.riemannian_beta
        else:
            kl_loss = self.compute_standard_kl_loss(mu, log_var)
            kl_weight = self.beta
        
        flow_loss = self.compute_flow_loss(log_det_jacobians)
        loop_penalty = self.compute_loop_penalty(z_seq, loop_mode)
        
        # Combine losses
        total_loss = recon_loss + kl_weight * kl_loss + flow_loss + loop_penalty
        
        # Store in history
        self.loss_history['reconstruction'].append(recon_loss.item())
        self.loss_history['kl_divergence'].append(kl_loss.item())
        self.loss_history['flow_loss'].append(flow_loss.item())
        self.loss_history['loop_penalty'].append(loop_penalty.item())
        self.loss_history['total'].append(total_loss.item())
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_divergence_loss': kl_loss,
            'flow_loss': flow_loss,
            'loop_penalty': loop_penalty,
            'loss_weights': {
                'beta': self.beta,
                'riemannian_beta': self.riemannian_beta,
                'loop_penalty_weight': self.loop_penalty_weight
            }
        }
    
    def get_loss_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of loss history.
        
        Returns:
            Dictionary with loss statistics
        """
        summary = {}
        for loss_name, history in self.loss_history.items():
            if len(history) > 0:
                summary[f'{loss_name}_mean'] = np.mean(history)
                summary[f'{loss_name}_std'] = np.std(history)
                summary[f'{loss_name}_min'] = np.min(history)
                summary[f'{loss_name}_max'] = np.max(history)
                summary[f'{loss_name}_recent'] = history[-10:] if len(history) >= 10 else history
        
        return summary
    
    def reset_history(self):
        """Reset loss history."""
        for key in self.loss_history:
            self.loss_history[key] = []
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get loss manager configuration.
        
        Returns:
            Dictionary of configuration parameters
        """
        return {
            'beta': self.beta,
            'riemannian_beta': self.riemannian_beta,
            'loop_penalty_weight': self.loop_penalty_weight,
            'device': str(self.device)
        } 