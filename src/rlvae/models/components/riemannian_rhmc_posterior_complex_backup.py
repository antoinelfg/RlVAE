"""
Riemannian RHMC Posterior Sampler
=================================

Implements the new posterior type: Riemannian initial sampling + RHMC exploration
without acceptance/rejection to preserve differentiability.

Mathematical formulation:
1. z₀ ~ N_Riem(μ_φ(x), α G(μ_φ(x)))  # Riemannian initial sampling
2. ρ₀ ~ N(0, G(z₀))                    # Momentum sampling
3. (z_K, ρ_K) = Φ^K(z₀, ρ₀)           # K steps of Hamiltonian evolution
4. Return z_K                          # Final position (ignore momentum)

Inspired by pyraug's RHVAE implementation but with Riemannian initial sampling.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import math
# Removed complex constraint imports for baseline version


class RiemannianRHMCPosterior(nn.Module):
    """
    Posterior sampler combining Riemannian initial sampling with RHMC exploration.
    
    This implements the hybrid approach:
    - Initial sampling respects the Riemannian geometry from the start
    - RHMC exploration adds rich dynamics along geodesics
    - No acceptance/rejection to preserve differentiability
    """
    
    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        import weakref
        self._ctx = {'model': weakref.proxy(model)}
        self.device = getattr(model, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # RHMC configuration - BASELINE VERSION
        self.config = config or {}
        self.rhmc_steps = self.config.get('rhmc_steps', 5)
        self.rhmc_step_size = self.config.get('rhmc_step_size', 0.01)
        self.rhmc_alpha = self.config.get('rhmc_alpha', 1.0)
        self.eps_reg = self.config.get('eps_regularization', 1e-6)
        
    def sample_riemannian_rhmc_posterior(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Main sampling method: Riemannian initial + RHMC exploration.
        
        Args:
            mu: Encoder mean [B, D]
            log_var: Encoder log variance [B, D] (may be ignored for Riemannian sampling)
            
        Returns:
            z_K: Final latent samples after RHMC evolution [B, D]
        """
        # Step 1: Riemannian initial sampling
        z0 = self._sample_initial_riemannian(mu, log_var)
        
        # Step 2: RHMC exploration (if steps > 0)
        if self.rhmc_steps > 0:
            z_final = self._rhmc_exploration(z0)
        else:
            z_final = z0
            
        return z_final
    
    def _sample_initial_riemannian(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Step 1: Sample z₀ ~ N_Riem(μ, α G(μ))
        
        This is the key difference from standard RHVAE:
        - RHVAE: z₀ ~ N(μ, σ²I) (Euclidean)
        - Ours: z₀ ~ N_Riem(μ, αG(μ)) (Riemannian from start)
        """
        # Fallback if no metric available
        if not hasattr(self._ctx['model'], 'G'):
            print("⚠️ No metric tensor available, falling back to standard sampling")
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(0.5 * log_var)
        
        mu_f32 = mu.float()
        batch_size, latent_dim = mu_f32.shape
        
        # Compute G(μ) - metric at encoder mean
        try:
            G_mu = self._ctx['model'].G(mu_f32)  # [B, D, D]
        except Exception as e:
            print(f"⚠️ Error computing G(μ): {e}, falling back to standard sampling")
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(0.5 * log_var)
        
        # Covariance: Σ = α G(μ) + ε I
        I = torch.eye(latent_dim, device=G_mu.device, dtype=G_mu.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        Sigma = self.rhmc_alpha * G_mu + self.eps_reg * I
        
        # Enhanced Cholesky decomposition with conditioning check
        try:
            # Check condition number before Cholesky (handle complex eigenvalues)
            eigenvals = torch.linalg.eigvals(Sigma)
            if eigenvals.is_complex():
                eigenvals_real = eigenvals.real
            else:
                eigenvals_real = eigenvals
            
            condition_number = torch.max(eigenvals_real, dim=-1)[0] / torch.clamp(torch.min(eigenvals_real, dim=-1)[0], min=1e-12)
            
            if torch.any(condition_number > self.max_condition_number):
                print(f"⚠️ High condition number detected: {condition_number.max().item():.2e}, using regularized eigendecomposition")
                raise torch.linalg.LinAlgError("High condition number")
                
            L = torch.linalg.cholesky(Sigma)  # [B, D, D]
        except Exception as e:
            print(f"⚠️ Cholesky failed: {e}, using stabilized eigendecomposition")
            # Enhanced eigendecomposition fallback
            eigenvals, eigenvecs = torch.linalg.eigh(Sigma)
            # More aggressive eigenvalue clamping for stability - fix tensor size mismatch
            min_eigenval = torch.clamp(eigenvals.max(dim=-1, keepdim=True)[0] * 1e-6, min=self.eps_reg)
            eigenvals = torch.clamp(eigenvals, min=min_eigenval)
            L = eigenvecs @ torch.diag_embed(torch.sqrt(eigenvals))
        
        # Sample: z₀ = μ + L ε
        eps = torch.randn(batch_size, latent_dim, device=mu.device, dtype=mu.dtype)
        z0 = mu_f32 + torch.einsum('bij,bj->bi', L, eps)
        
        # Apply stabilization if configured
        if self.step_clip_scale is not None:
            # Clip the step from mu to z0
            step = z0 - mu_f32
            step_norm = torch.norm(step, dim=-1, keepdim=True)
            max_step = self.step_clip_scale
            step = torch.where(step_norm > max_step, step * (max_step / step_norm), step)
            z0 = mu_f32 + step
        
        if self.maha_clip is not None:
            # Mahalanobis distance clipping
            diff = z0 - mu_f32
            try:
                maha_dist = torch.sqrt(torch.einsum('bi,bij,bj->b', diff, torch.linalg.inv(Sigma), diff))
                mask = maha_dist > self.maha_clip
                if mask.any():
                    scale = self.maha_clip / maha_dist
                    z0 = torch.where(mask.unsqueeze(-1), mu_f32 + diff * scale.unsqueeze(-1), z0)
            except:
                pass  # Skip Mahalanobis clipping if matrix inversion fails
        
        return z0.to(mu.dtype)
    
    def _rhmc_exploration(self, z0: torch.Tensor) -> torch.Tensor:
        """
        Step 2: RHMC exploration without acceptance/rejection.
        
        Hamiltonian dynamics: H(z, ρ) = U(z) + (1/2) ρᵀ G⁻¹(z) ρ
        where U(z) is the potential energy (negative log prior).
        """
        z = z0.clone()
        
        # Sample initial momentum: ρ₀ ~ N(0, G(z₀))
        rho = self._sample_momentum(z)
        
        # K steps of leapfrog integration - BASELINE VERSION
        for step in range(self.rhmc_steps):
            z, rho = self._leapfrog_step(z, rho, self.rhmc_step_size)
        
        return z
    
    def _sample_momentum(self, z: torch.Tensor) -> torch.Tensor:
        """
        Enhanced momentum sampling with stability checks: ρ ~ N(0, G(z))
        
        This gives the kinetic energy: T(ρ) = (1/2) ρᵀ G⁻¹(z) ρ
        """
        try:
            G_z = self._ctx['model'].G(z)  # [B, D, D]
            batch_size, latent_dim = z.shape
            
            # Enhanced regularization for momentum sampling
            I = torch.eye(latent_dim, device=z.device, dtype=G_z.dtype)
            G_z_reg = G_z + self.eps_reg * I
            
            # Check conditioning before Cholesky (handle complex eigenvalues)
            eigenvals = torch.linalg.eigvals(G_z_reg)
            if eigenvals.is_complex():
                eigenvals_real = eigenvals.real
            else:
                eigenvals_real = eigenvals
            
            condition_numbers = torch.max(eigenvals_real, dim=-1)[0] / torch.clamp(torch.min(eigenvals_real, dim=-1)[0], min=1e-12)
            
            if torch.any(condition_numbers > self.max_condition_number):
                print(f"⚠️ High condition number in momentum sampling: {condition_numbers.max().item():.2e}")
                # Use eigendecomposition for ill-conditioned matrices
                eigenvals, eigenvecs = torch.linalg.eigh(G_z_reg)
                min_eigenval = torch.clamp(eigenvals.max(dim=-1, keepdim=True)[0] * 1e-6, min=self.eps_reg)
                eigenvals = torch.clamp(eigenvals, min=min_eigenval)
                L = eigenvecs @ torch.diag_embed(torch.sqrt(eigenvals))
            else:
                # Standard Cholesky
                L = torch.linalg.cholesky(G_z_reg)
            
            # Sample: ρ = L ε with clipped noise
            eps = torch.randn_like(z)
            eps_norm = torch.norm(eps, dim=-1, keepdim=True)
            eps = torch.where(eps_norm > 3.0, eps * (3.0 / eps_norm), eps)  # Clip to 3σ
            
            rho = torch.einsum('bij,bj->bi', L, eps)
            
            # Final momentum magnitude check
            rho_norm = torch.norm(rho, dim=-1, keepdim=True)
            max_rho = 5.0  # Prevent explosive momentum
            rho = torch.where(rho_norm > max_rho, rho * (max_rho / rho_norm), rho)
            
            return rho
            
        except Exception as e:
            print(f"⚠️ Error sampling momentum: {e}, using scaled identity")
            return torch.randn_like(z) * 0.1  # Reduced scale for fallback
    
    def _leapfrog_step(self, z: torch.Tensor, rho: torch.Tensor, step_size: float) -> tuple:
        """
        Enhanced leapfrog integrator with adaptive step size and stability checks.
        
        Leapfrog scheme:
        1. ρ_{1/2} = ρ₀ - (ε/2) ∇_z U(z₀)
        2. z₁ = z₀ + ε G⁻¹(z₀) ρ_{1/2}  
        3. ρ₁ = ρ_{1/2} - (ε/2) ∇_z U(z₁)
        """
        # Adaptive step size based on momentum magnitude
        if self.adaptive_step_size:
            rho_norm = torch.norm(rho, dim=-1, keepdim=True)
            adaptive_factor = torch.clamp(1.0 / (1.0 + rho_norm * 0.1), min=0.1, max=1.0)
            step_size = step_size * adaptive_factor.mean().item()
            step_size = max(step_size, self.min_step_size)
        
        # Half step for momentum
        grad_U = self._compute_potential_gradient(z)
        rho_half = rho - (step_size / 2) * grad_U
        
        # Full step for position with enhanced stability
        try:
            G_inv_z = self._ctx['model'].G_inv(z)  # [B, D, D]
            
            # Enhanced regularization and conditioning
            batch_size, latent_dim = z.shape
            I = torch.eye(latent_dim, device=z.device, dtype=G_inv_z.dtype).unsqueeze(0).expand(batch_size, -1, -1)
            
            # Check condition number and apply adaptive regularization (handle complex eigenvalues)
            eigenvals = torch.linalg.eigvals(G_inv_z)
            if eigenvals.is_complex():
                eigenvals_real = eigenvals.real
            else:
                eigenvals_real = eigenvals
            
            condition_numbers = torch.max(eigenvals_real, dim=-1)[0] / torch.clamp(torch.min(eigenvals_real, dim=-1)[0], min=1e-12)
            
            # Adaptive regularization based on condition number
            adaptive_reg = self.eps_reg * torch.clamp(condition_numbers / 1e3, min=1.0, max=100.0).unsqueeze(-1).unsqueeze(-1)
            G_inv_z_reg = G_inv_z + adaptive_reg * I
            
            velocity = torch.einsum('bij,bj->bi', G_inv_z_reg, rho_half)
            
            # Enhanced velocity clipping
            vel_norm = torch.norm(velocity, dim=-1, keepdim=True)
            max_vel = 1.0 / step_size  # Prevent explosive steps
            velocity = torch.where(vel_norm > max_vel, velocity * (max_vel / (vel_norm + 1e-12)), velocity)
            
            # Check for NaN/Inf in velocity
            if torch.isnan(velocity).any() or torch.isinf(velocity).any():
                print("⚠️ NaN/Inf in velocity, using fallback")
                velocity = torch.randn_like(velocity) * 0.01
            
            z_new = z + step_size * velocity
            
        except Exception as e:
            print(f"⚠️ Error computing G⁻¹(z): {e}, using identity")
            z_new = z + step_size * rho_half * 0.1  # Reduced step for fallback
        
        # Half step for momentum (at new position)
        grad_U_new = self._compute_potential_gradient(z_new)
        rho_new = rho_half - (step_size / 2) * grad_U_new
        
        # Final stability check
        if torch.isnan(z_new).any() or torch.isinf(z_new).any():
            print("⚠️ NaN/Inf in position, resetting to small perturbation")
            z_new = z + torch.randn_like(z) * 0.01
            
        if torch.isnan(rho_new).any() or torch.isinf(rho_new).any():
            print("⚠️ NaN/Inf in momentum, resetting")
            rho_new = torch.randn_like(rho) * 0.01
        
        return z_new, rho_new
    
    def _compute_potential_gradient(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute ∇_z U(z) where U(z) = -log p(z) is the potential energy.
        
        For Riemannian manifold prior: U(z) = -log p_manifold(z)
        This should incorporate the learned manifold structure via the metric tensor.
        """
        try:
            # Use manifold-aware potential that attracts to high-density regions
            if hasattr(self._ctx['model'], 'centroids_tens') and hasattr(self._ctx['model'], 'G_inv'):
                # Compute attraction to nearest centroids weighted by manifold density
                centroids = self._ctx['model'].centroids_tens  # [K, D]
                
                # Find distances to all centroids
                diff = z.unsqueeze(1) - centroids.unsqueeze(0)  # [B, K, D]
                distances = torch.norm(diff, dim=-1)  # [B, K]
                
                # Use inverse distance weighting to create attractive potential
                weights = torch.exp(-distances / 0.5)  # Temperature parameter
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
                
                # Compute weighted gradient toward centroids
                grad_U = torch.einsum('bk,bkd->bd', weights, diff)  # [B, D]
                
                # Add small regularization toward origin to prevent drift
                grad_U = grad_U + 0.1 * z
                
                return grad_U
            else:
                # Fallback to weak Gaussian prior
                return 0.1 * z.clone()
                
        except Exception as e:
            print(f"⚠️ Error computing manifold potential gradient: {e}, using weak Gaussian")
            return 0.1 * z.clone()
    
    def _stabilize_dynamics(self, z: torch.Tensor, rho: torch.Tensor) -> tuple:
        """
        Apply numerical stabilization to prevent explosion.
        """
        # Gradient clipping for momentum
        rho_norm = torch.norm(rho, dim=-1, keepdim=True)
        rho = torch.where(rho_norm > self.max_grad_norm, rho * (self.max_grad_norm / rho_norm), rho)
        
        # Check for NaN/Inf and reset if needed
        if torch.isnan(z).any() or torch.isinf(z).any():
            print("⚠️ NaN/Inf detected in position, resetting")
            z = torch.randn_like(z) * 0.1  # Small random reset
        
        if torch.isnan(rho).any() or torch.isinf(rho).any():
            print("⚠️ NaN/Inf detected in momentum, resetting")
            rho = torch.randn_like(rho) * 0.1  # Small random reset
        
        return z, rho
    
    def _emergency_divergence_protection(self, z: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Emergency protection against divergence during training.
        
        Checks for extreme values and resets to safe fallback if needed.
        """
        # Check for NaN/Inf
        if torch.isnan(z).any() or torch.isinf(z).any():
            print("🚨 EMERGENCY: NaN/Inf detected in RHMC samples, falling back to encoder means")
            return mu.clone()
        
        # Check for extreme magnitudes (likely divergence)
        z_norms = torch.norm(z, dim=-1)
        max_allowed_norm = 10.0  # Conservative threshold
        
        if z_norms.max() > max_allowed_norm:
            print(f"🚨 EMERGENCY: Extreme sample magnitude detected ({z_norms.max():.2f}), applying emergency clipping")
            
            # Clip extreme samples
            extreme_mask = z_norms > max_allowed_norm
            z_clipped = z.clone()
            
            for i, is_extreme in enumerate(extreme_mask):
                if is_extreme:
                    # Reset to encoder mean with small noise
                    noise = torch.randn_like(mu[i]) * 0.1
                    z_clipped[i] = mu[i] + noise
            
            return z_clipped
        
        # Check distance from encoder means
        distances = torch.norm(z - mu, dim=-1)
        max_allowed_distance = 5.0  # Conservative threshold
        
        if distances.max() > max_allowed_distance:
            print(f"🚨 EMERGENCY: Samples too far from encoder means ({distances.max():.2f}), applying recall")
            
            # Apply strong recall toward encoder means
            far_mask = distances > max_allowed_distance
            z_recalled = z.clone()
            
            for i, is_far in enumerate(far_mask):
                if is_far:
                    # Interpolate back toward encoder mean
                    direction = mu[i] - z[i]
                    z_recalled[i] = z[i] + 0.8 * direction  # Strong recall
            
            return z_recalled
        
        return z
    
    def compute_log_density_correction(self, z_initial: torch.Tensor, z_final: torch.Tensor) -> torch.Tensor:
        """
        Compute the log density correction for the RHMC transformation.
        
        This is needed for the KL divergence computation.
        In practice, this is complex to compute exactly, so we use approximations.
        """
        # Simplified approximation: assume volume preservation (Hamiltonian property)
        # In reality, we'd need to compute the Jacobian of the RHMC transformation
        batch_size = z_initial.shape[0]
        return torch.zeros(batch_size, device=z_initial.device)
    
    def get_config(self) -> Dict[str, Any]:
        """Return current configuration."""
        return {
            'rhmc_steps': self.rhmc_steps,
            'rhmc_step_size': self.rhmc_step_size,
            'rhmc_alpha': self.rhmc_alpha,
            'eps_regularization': self.eps_reg,
            'max_grad_norm': self.max_grad_norm,
            'min_step_size': self.min_step_size
        }
    
    def set_config(self, config: Dict[str, Any]):
        """Update configuration."""
        self.rhmc_steps = config.get('rhmc_steps', self.rhmc_steps)
        self.rhmc_step_size = config.get('rhmc_step_size', self.rhmc_step_size)
        self.rhmc_alpha = config.get('rhmc_alpha', self.rhmc_alpha)
        self.eps_reg = config.get('eps_regularization', self.eps_reg)
        self.max_grad_norm = config.get('max_grad_norm', self.max_grad_norm)
        self.min_step_size = config.get('min_step_size', self.min_step_size)
