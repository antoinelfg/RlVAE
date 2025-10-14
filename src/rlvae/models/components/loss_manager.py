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
        # μ anchoring
        mu_l2_weight: float = 0.0,
        # Prior mode for KL: 'uniform' (default, cancels volume) or 'volume_gaussian'
        kl_prior_mode: str = 'uniform',
        # Riemannian KL options (to mirror original model behavior)
        kl_use_metric_normalization: bool = True,
        kl_metric_norm_mode: str = 'geomean',   # 'geomean' | 'trace' | 'none'
        kl_amp_safe: bool = True,
        kl_metric_eval_point: str = 'z',  # 'z' or 'mu' (curvature correction if 'z')
        # RHMC KL switches
        rhmc_kl_mode: str = 'mc',          # {'mc','jac','bound'}
        rhmc_kl_source: str = 'z0',        # {'z0','zk'}
        rhmc_kl_jacobian: bool = False,
    ):
        super().__init__()
        self.beta = beta
        self.riemannian_beta = riemannian_beta if riemannian_beta is not None else beta
        self.loop_penalty_weight = loop_penalty_weight
        self.device = device or torch.device('cpu')
        self.metric_reg_weight = metric_reg_weight
        self.metric_reg_type = metric_reg_type
        self.metric_reg_target = metric_reg_target
        self.mu_l2_weight = float(mu_l2_weight)
        self.kl_prior_mode = str(kl_prior_mode)
        self.kl_use_metric_normalization = kl_use_metric_normalization
        self.kl_metric_norm_mode = kl_metric_norm_mode
        self.kl_amp_safe = kl_amp_safe
        self.kl_metric_eval_point = kl_metric_eval_point
        # RHMC KL toggles
        self.rhmc_kl_mode = str(rhmc_kl_mode)
        self.rhmc_kl_source = str(rhmc_kl_source)
        self.rhmc_kl_jacobian = bool(rhmc_kl_jacobian)
        self.to(self.device)
        
        # Loss tracking
        self.loss_history = {
            'reconstruction': [],
            'kl_divergence': [],
            'riemannian_kl': [],
            'flow_loss': [],
            'loop_penalty': [],
            'total': [],
            'metric_reg': [],  # NEW
            'mu_l2': []
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
        # DEBUG: Print KL computation details
        if not hasattr(self, '_kl_debug_printed'):
            print(f"[KL DEBUG] kl_metric_eval_point: {self.kl_metric_eval_point}")
            print(f"[KL DEBUG] mu shape: {mu.shape}, mean: {mu.mean().item():.4f}, std: {mu.std().item():.4f}")
            print(f"[KL DEBUG] z_samples shape: {z_samples.shape}, mean: {z_samples.mean().item():.4f}, std: {z_samples.std().item():.4f}")
            self._kl_debug_printed = True
        if metric_tensor is None:
            # Fallback to standard KL (silenced unless debug)
            if os.environ.get("RLVAE_DEBUG") == "1":
                print("⚠️ DEBUG: Falling back to standard KL")
            if os.environ.get('RLVAE_STRICT', '0') == '1':
                raise RuntimeError("LossManager: metric_tensor is None under strict Riemannian mode")
            try:
                if os.environ.get('RLVAE_TRACE', '0') == '1':
                    print(f"TRACE KL source: fallback to standard KL; z_samples shape={tuple(z_samples.shape)}")
            except Exception:
                pass
            log_var_clamped = torch.clamp(log_var, -10.0, 10.0)
            return -0.5 * torch.sum(1 + log_var_clamped - mu.pow(2) - log_var_clamped.exp(), dim=1).mean()
        try:
            try:
                if os.environ.get('RLVAE_TRACE', '0') == '1':
                    print(f"TRACE KL source: using rhmc_z0=True shape={tuple(z_samples.shape)}, using log_q from q_Riem")
            except Exception:
                pass
            # Choose evaluation point for metric: 'z' (samples) or 'mu' (means)
            eval_pts = z_samples if self.kl_metric_eval_point == 'z' else mu
            print(f"[KL DEBUG] Using eval_pts: {self.kl_metric_eval_point} -> shape {eval_pts.shape}, mean {eval_pts.mean().item():.4f}, std {eval_pts.std().item():.4f}")
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

            # Choose what to penalize based on evaluation point
            if self.kl_metric_eval_point == 'mu':
                # Penalize encoder means directly: (μ - 0)ᵀ G(μ) (μ - 0)
                # This pulls encoder means toward the prior center (0)
                diff = mu_f32 - 0.0  # [B, D] - distance from prior center
                print(f"[KL DEBUG] Penalizing encoder means: diff shape {diff.shape}, mean {diff.mean().item():.4f}, std {diff.std().item():.4f}")
            else:
                # Penalize distance between samples and means: (z - μ)ᵀ G(z) (z - μ)
                # This is the standard VAE KL behavior
                diff = z_f32 - mu_f32  # [B, D]
                print(f"[KL DEBUG] Penalizing sample-mean distance: diff shape {diff.shape}, mean {diff.mean().item():.4f}, std {diff.std().item():.4f}")
            
            quadratic_form = torch.einsum('bi,bij,bj->b', diff, Gtilde_f32, diff)
            kl_terms = 0.5 * quadratic_form

            # Optional volume-corrected Gaussian prior at μ: 0.5||μ||^2 - 0.5 log|G(μ)|
            if self.kl_prior_mode == 'volume_gaussian' and self.kl_metric_eval_point == 'mu':
                try:
                    # Ensure we have G at eval points
                    if hasattr(metric_tensor, 'compute_metric'):
                        G_mu = metric_tensor.compute_metric(eval_pts)
                    else:
                        # If only inverse is available, invert (small regularization for stability)
                        if 'Gtilde_f32' in locals():
                            G_mu = torch.linalg.inv(Gtilde_f32.float())
                        else:
                            raise RuntimeError('Metric tensor G not available for volume prior term')
                    sign, logabsdet_G = torch.slogdet(G_mu.float())
                    logabsdet_G = logabsdet_G  # [B]
                    mu_norm_sq = torch.sum(mu_f32 ** 2, dim=-1)  # [B]
                    prior_term = 0.5 * mu_norm_sq - 0.5 * logabsdet_G
                    kl_terms = kl_terms + prior_term
                    print(f"[KL DEBUG] Added volume prior term: mean={prior_term.mean().item():.6f}")
                except Exception as e:
                    print(f"[KL DEBUG] Volume prior term failed: {e}")

            kl_divergence = kl_terms.mean()
            print(f"[KL DEBUG] Final KL: {kl_divergence.item():.6f}, quadratic_form mean: {quadratic_form.mean().item():.6f}")
            return kl_divergence.to(mu.dtype)
        except Exception as e:
            if os.environ.get('RLVAE_STRICT', '0') == '1':
                raise RuntimeError(f"LossManager: Riemannian KL failed under strict mode: {e}")
            print(f"⚠️ Riemannian KL computation failed: {e}, using standard KL")
            try:
                if os.environ.get('RLVAE_TRACE', '0') == '1':
                    print("TRACE KL source: error path -> standard KL")
            except Exception:
                pass
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
        # Sanitize before reduction to avoid NaN propagation from a single outlier
        total_log_det = torch.nan_to_num(total_log_det, nan=0.0, posinf=1e6, neginf=-1e6)
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
        use_riemannian_kl: bool = True,
        # RHMC posterior pathway (MC KL on pushforward)
        rhmc_z0: Optional[torch.Tensor] = None,
        rhmc_zK: Optional[torch.Tensor] = None,
        rhmc_log_q: Optional[torch.Tensor] = None,
        rhmc_traj_info: Optional[dict] = None,
        rhmc_posterior: Optional[Any] = None,
        rhmc_kl_mode: Optional[str] = None,
        rhmc_kl_source: Optional[str] = None,
        rhmc_kl_jacobian: Optional[bool] = None,
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
        
        # Decide KL path
        kl_weight = self.beta
        kl_loss: torch.Tensor

        # Prefer RHMC Monte-Carlo path when posterior extras are provided
        use_rhmc_mc = (
            rhmc_posterior is not None and
            (rhmc_zK is not None or z_samples is not None)
        )

        # Resolve runtime switches (kwargs override ctor defaults)
        _mode = (rhmc_kl_mode or self.rhmc_kl_mode).lower()
        _src = (rhmc_kl_source or self.rhmc_kl_source).lower()
        _with_jac = bool(self.rhmc_kl_jacobian if rhmc_kl_jacobian is None else rhmc_kl_jacobian)

        if use_rhmc_mc and _mode in {"mc", "jac"}:
            # Log config once
            if not hasattr(self, "_loss_cfg_printed"):
                print(f"[LOSS CONFIG] rhmc_kl_mode={_mode}, source={_src}, jacobian={str(_with_jac).lower()}")
                self._loss_cfg_printed = True

            # Ensure zK present
            zK = rhmc_zK if rhmc_zK is not None else z_samples
            assert zK is not None, "zK samples required for RHMC MC KL"

            # log_q from z0 if available and selected; else fallback at zK
            if _src == "z0" and rhmc_log_q is not None:
                log_q = rhmc_log_q
            else:
                # Fallback approximate log q at zK using current Riemannian Gaussian at μ
                try:
                    log_q = rhmc_posterior._compute_log_riemannian_gaussian(zK, mu, log_var)
                except Exception as e:
                    if os.environ.get("RLVAE_DEBUG") == "1":
                        print(f"[KL DEBUG] Fallback log_q@zK failed: {e}; using isotropic Gaussian")
                    diff = zK - mu
                    d = zK.shape[-1]
                    log_q = -0.5 * torch.sum(diff ** 2, dim=-1) - 0.5 * d * np.log(2 * np.pi)

            # log_p under Riemannian volume prior at zK
            try:
                log_p = rhmc_posterior._compute_log_prior(zK)
            except Exception:
                d = zK.shape[-1]
                log_p = -0.5 * torch.sum(zK ** 2, dim=-1) - 0.5 * d * np.log(2 * np.pi)

            # Optional Jacobian correction (placeholder)
            jac_correction = 0.0
            if _mode == "jac" and _with_jac and isinstance(rhmc_traj_info, dict):
                j = rhmc_traj_info.get('jac_logdet', None)
                if isinstance(j, torch.Tensor):
                    jac_correction = j
                elif j is not None:
                    try:
                        jac_correction = torch.as_tensor(j, device=zK.device, dtype=log_q.dtype)
                    except Exception:
                        jac_correction = 0.0

            kl_terms = log_q - log_p - jac_correction
            kl_loss = kl_terms.mean().to(x.dtype)
            kl_weight = self.riemannian_beta
        elif use_riemannian_kl and metric_tensor is not None:
            # Geodesic bound / classical Riemannian KL (bound mode)
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
        
        # Route debugging (one-time): show how total loss is assembled
        if not hasattr(self, "_route_debug_printed"):
            try:
                print(
                    f"[LOSS ROUTE] recon={float(recon_loss):.4f} | kl={float(kl_loss):.4f} (w={float(kl_weight):.2f}) | "
                    f"flow={float(flow_loss):.4f} | loop={float(loop_penalty):.4f} | use_riem_kl={bool(use_riemannian_kl and metric_tensor is not None)}"
                )
            except Exception:
                pass
            self._route_debug_printed = True
        
        # Combine losses
        total_loss = recon_loss + kl_weight * kl_loss + flow_loss + loop_penalty
        # Auxiliary μ L2 anchor (encourages μ toward 0 to align with prior/centroids)
        mu_l2_pen = torch.tensor(0.0, device=x.device)
        if self.mu_l2_weight > 0.0 and isinstance(mu, torch.Tensor):
            mu_l2_pen = (mu.pow(2).sum(dim=-1)).mean() * self.mu_l2_weight
            total_loss = total_loss + mu_l2_pen
            self.loss_history['mu_l2'].append(mu_l2_pen.item())
            if os.environ.get('RLVAE_DEBUG') == '1' and not hasattr(self, '_mu_l2_debugged'):
                print(f"[MU L2] weight={self.mu_l2_weight:.3f}, penalty={mu_l2_pen.item():.6f}")
                self._mu_l2_debugged = True
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
            'mu_l2_penalty': mu_l2_pen,
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
            'mu_l2_weight': self.mu_l2_weight,
            'kl_prior_mode': self.kl_prior_mode,
            'kl_use_metric_normalization': self.kl_use_metric_normalization,
            'kl_metric_norm_mode': self.kl_metric_norm_mode,
            'kl_amp_safe': self.kl_amp_safe,
        }
