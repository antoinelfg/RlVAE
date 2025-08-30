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
import os

class LossManager(nn.Module):
    def __init__(
        self,
        beta: float = 1.0,
        riemannian_beta: Optional[float] = None,
        loop_penalty_weight: float = 1.0,
        device: Optional[torch.device] = None,
        metric_reg_weight: float = 0.0,  # NEW: weight for metric regularization
        metric_reg_type: str = 'none',   # NEW: type: 'none', 'determinant', 'condition', 'smoothness'
        metric_reg_target: float = 0.0,  # NEW: target value for regularization (e.g., logdet target)
        # Riemannian KL options (to mirror original model behavior)
        kl_use_metric_normalization: bool = True,
        kl_metric_norm_mode: str = 'geomean',   # 'geomean' | 'trace' | 'none'
        kl_amp_safe: bool = True,
        kl_metric_eval_point: str = 'z',  # 'z' or 'mu' (curvature correction if 'z')
    ):
        super().__init__()
        self.beta = beta
        self.riemannian_beta = riemannian_beta if riemannian_beta is not None else beta
        self.loop_penalty_weight = loop_penalty_weight
        self.device = device or torch.device('cpu')
        self.metric_reg_weight = metric_reg_weight
        self.metric_reg_type = metric_reg_type
        self.metric_reg_target = metric_reg_target
        self.kl_use_metric_normalization = kl_use_metric_normalization
        self.kl_metric_norm_mode = kl_metric_norm_mode
        self.kl_amp_safe = kl_amp_safe
        self.kl_metric_eval_point = kl_metric_eval_point
        self.to(self.device)
        
        # Loss tracking
        self.loss_history = {
            'reconstruction': [],
            'kl_divergence': [],
            'riemannian_kl': [],
            'flow_loss': [],
            'loop_penalty': [],
            'total': [],
            'metric_reg': []  # NEW
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
        # CRITICAL FIX: Scale reconstruction loss by 255 (user prefers non-normalized scale)
        # This gives meaningful loss values in the 0-255 range
        loss = F.mse_loss(x_recon, x, reduction='mean') * 255.0
        
        if not torch.isfinite(loss):
            if os.environ.get("RLVAE_DEBUG") == "1":
                print("⚠️ Reconstruction loss is not finite! Clamping to 1.0.")
            loss = torch.tensor(1.0, device=loss.device)
        return loss
    
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
        Compute KL divergence for metric-aware Riemannian posterior:
        KL[q_φ(z_0|x_0) || p(z_0)] = 1/2 E_q[(z_0-μ)^T G(z_0) (z_0-μ)]
        Where:
        - q_φ(z_0|x_0) ∝ [det G(z_0)]^{-1/2} exp(-1/2 (z_0-μ)^T G(z_0) (z_0-μ))
        - p(z_0) ∝ [det G(z_0)]^{-1/2} (uniform Riemannian prior)
        The log det G(z_0) terms cancel out, leaving only the quadratic form.
        Args:
            mu: Posterior mean [batch_size, latent_dim]
            log_var: Posterior log variance (not used)
            z_samples: Samples from posterior [batch_size, latent_dim]
            metric_tensor: Metric tensor component (required)
        Returns:
            KL divergence (scalar)
        """
        if metric_tensor is None:
            # Fallback to standard KL (silenced unless debug)
            if os.environ.get("RLVAE_DEBUG") == "1":
                print("⚠️ DEBUG: Falling back to standard KL")
            log_var_clamped = torch.clamp(log_var, -10.0, 10.0)
            return -0.5 * torch.sum(1 + log_var_clamped - mu.pow(2) - log_var_clamped.exp(), dim=1).mean()
        try:
            # Choose evaluation point for metric: 'z' (samples) or 'mu' (means)
            eval_pts = z_samples if self.kl_metric_eval_point == 'z' else mu
            # Compute inverse metric tensor at chosen points (G̃ = G^{-1})
            if hasattr(metric_tensor, 'compute_inverse_metric'):
                G_inv = metric_tensor.compute_inverse_metric(eval_pts)  # [B, D, D]
            elif hasattr(metric_tensor, 'compute_metric'):
                G = metric_tensor.compute_metric(eval_pts)
                G_inv = torch.linalg.inv(G)
            else:
                # Assume callable returns G; invert
                G = metric_tensor(eval_pts)
                G_inv = torch.linalg.inv(G)

            # Optional normalization of G_inv for scale invariance
            if self.kl_use_metric_normalization:
                d = G_inv.shape[-1]
                if self.kl_metric_norm_mode == 'geomean':
                    sign, logabsdet = torch.slogdet(G_inv)
                    s = torch.exp(logabsdet / d).unsqueeze(-1).unsqueeze(-1)
                    Gtilde = G_inv / (s + 1e-12)
                elif self.kl_metric_norm_mode == 'trace':
                    s = (torch.einsum('bii->b', G_inv) / d).unsqueeze(-1).unsqueeze(-1)
                    Gtilde = G_inv / (s + 1e-12)
                else:
                    Gtilde = G_inv
            else:
                Gtilde = G_inv

            # AMP-safe float32 compute
            mu_f32 = mu.float() if self.kl_amp_safe else mu
            z_f32 = z_samples.float() if self.kl_amp_safe else z_samples
            Gtilde_f32 = Gtilde.float() if self.kl_amp_safe else Gtilde

            diff = z_f32 - mu_f32  # [B, D]
            quadratic_form = torch.einsum('bi,bij,bj->b', diff, Gtilde_f32, diff)
            kl_divergence = 0.5 * quadratic_form.mean()
            return kl_divergence.to(mu.dtype)
        except Exception as e:
            print(f"⚠️ Riemannian KL computation failed: {e}, using standard KL")
            log_var_clamped = torch.clamp(log_var, -10.0, 10.0)
            return -0.5 * torch.sum(1 + log_var_clamped - mu.pow(2) - log_var_clamped.exp(), dim=1).mean()
    
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
        loss = torch.mean(total_log_det).abs()  # Ensure positive loss
        if not torch.isfinite(loss):
            print("⚠️ Flow loss is not finite! Clamping to 0.0.")
            loss = torch.tensor(0.0, device=self.device)
        return loss
    
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
            if not torch.isfinite(penalty):
                print("⚠️ Loop penalty is not finite! Clamping to 0.0.")
                penalty = torch.tensor(0.0, device=self.device)
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
        use_riemannian_kl: bool = True
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
        # Ensure metric_reg is always defined for safe returns
        metric_reg = torch.tensor(0.0, device=x.device)

        # Compute individual loss components
        recon_loss = self.compute_reconstruction_loss(x_recon, x)
        if not torch.isfinite(recon_loss) and os.environ.get("RLVAE_DEBUG") == "1":
            print("[DEBUG] recon_loss is not finite!", recon_loss)
        
        if use_riemannian_kl and metric_tensor is not None:
            kl_loss = self.compute_riemannian_kl_loss(mu, log_var, z_samples, metric_tensor)
            kl_weight = self.riemannian_beta
        else:
            if os.environ.get("RLVAE_DEBUG") == "1":
                print("⚠️ DEBUG: Falling back to STANDARD KL")
            kl_loss = self.compute_standard_kl_loss(mu, log_var)
            kl_weight = self.beta
        if not torch.isfinite(kl_loss) and os.environ.get("RLVAE_DEBUG") == "1":
            print("[DEBUG] kl_loss is not finite!", kl_loss)
        
        flow_loss = self.compute_flow_loss(log_det_jacobians)
        if not torch.isfinite(flow_loss) and os.environ.get("RLVAE_DEBUG") == "1":
            print("[DEBUG] flow_loss is not finite!", flow_loss)
        
        loop_penalty = self.compute_loop_penalty(z_seq, loop_mode)
        if not torch.isfinite(loop_penalty) and os.environ.get("RLVAE_DEBUG") == "1":
            print("[DEBUG] loop_penalty is not finite!", loop_penalty)
        
        # Combine losses
        total_loss = recon_loss + kl_weight * kl_loss + flow_loss + loop_penalty
        if not torch.isfinite(total_loss) and os.environ.get("RLVAE_DEBUG") == "1":
            print("[DEBUG] total_loss is not finite!", total_loss)
            print(f"[DEBUG] recon_loss: {recon_loss}, kl_loss: {kl_loss}, flow_loss: {flow_loss}, loop_penalty: {loop_penalty}")
        
        # If metric_tensor is used, print stats
        if metric_tensor is not None and hasattr(metric_tensor, 'trainable') and os.environ.get("RLVAE_DEBUG") == "1":
            try:
                G = metric_tensor(z_samples)
                print("[DEBUG] Metric tensor stats: min", G.min().item(), "max", G.max().item(), "mean", G.mean().item(), "std", G.std().item())
                eigvals = torch.linalg.eigvals(G[0]).real
                print("[DEBUG] Metric eigenvalues: min", eigvals.min().item(), "max", eigvals.max().item(), "mean", eigvals.mean().item())
            except Exception as e:
                print(f"[DEBUG] Error computing metric tensor stats: {e}")
        
        # If latent z_samples is available, print stats
        if z_samples is not None and os.environ.get("RLVAE_DEBUG") == "1":
            try:
                print("[DEBUG] Latent z stats: min", z_samples.min().item(), "max", z_samples.max().item(), "mean", z_samples.mean().item(), "std", z_samples.std().item())
            except Exception as e:
                print(f"[DEBUG] Error computing latent z stats: {e}")
        
        # Store in history
        self.loss_history['reconstruction'].append(recon_loss.item())
        self.loss_history['kl_divergence'].append(kl_loss.item())
        self.loss_history['flow_loss'].append(flow_loss.item())
        self.loss_history['loop_penalty'].append(loop_penalty.item())
        self.loss_history['total'].append(total_loss.item())
        
        # Metric regularization (only if trainable)
        if metric_tensor is not None and getattr(metric_tensor, 'trainable', False) and self.metric_reg_weight > 0.0:
            z_reg = z_samples.detach() if z_samples is not None else torch.randn(32, mu.shape[-1], device=x.device)
            if self.metric_reg_type == 'determinant':
                G = metric_tensor.compute_metric(z_reg)
                logdet = torch.logdet(G)
                metric_reg = ((logdet - self.metric_reg_target) ** 2).mean() * self.metric_reg_weight
            elif self.metric_reg_type == 'condition':
                G = metric_tensor.compute_metric(z_reg)
                eigvals = torch.linalg.eigvals(G)
                cond = (eigvals.abs().max(dim=-1).values / eigvals.abs().min(dim=-1).values).mean()
                metric_reg = ((cond - self.metric_reg_target) ** 2) * self.metric_reg_weight
            elif self.metric_reg_type == 'smoothness':
                # Smoothness: penalize large changes in G(z) for nearby z
                z2 = z_reg + 0.01 * torch.randn_like(z_reg)
                G1 = metric_tensor.compute_metric(z_reg)
                G2 = metric_tensor.compute_metric(z2)
                metric_reg = ((G1 - G2) ** 2).mean() * self.metric_reg_weight
            self.loss_history['metric_reg'].append(metric_reg.item())
            total_loss = total_loss + metric_reg

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
            },
            'metric_reg': metric_reg # NEW
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
            'device': str(self.device),
            'metric_reg_weight': self.metric_reg_weight,
            'metric_reg_type': self.metric_reg_type,
            'metric_reg_target': self.metric_reg_target,
            'kl_use_metric_normalization': self.kl_use_metric_normalization,
            'kl_metric_norm_mode': self.kl_metric_norm_mode,
            'kl_amp_safe': self.kl_amp_safe,
        }
