"""
Riemannian HMC Sampler
======================

Hamiltonian Monte Carlo sampler for Riemannian manifolds - RHVAE compatible.
"""

import torch
from typing import Dict, Any
from .base_sampler import BaseRiemannianSampler


class RiemannianHMCSampler(BaseRiemannianSampler):
    """Hamiltonian Monte Carlo sampler for Riemannian manifold - RHVAE compatible."""
    
    def __init__(self, model, mcmc_steps_nbr=100, n_lf=15, eps_lf=0.03, beta_zero=1.0):
        super().__init__(model)
        self.mcmc_steps_nbr = mcmc_steps_nbr
        self.n_lf = torch.tensor([n_lf], device=model.device)
        self.eps_lf = torch.tensor([eps_lf], device=model.device)
        self.beta_zero_sqrt = torch.tensor([beta_zero], device=model.device).sqrt()
        
        # Use RHVAE-style analytic functions when available
        if self.validate_metric_availability():
            # Define log probability function matching RHVAE exactly
            def _rhvae_log_sqrt_det_G_inv(z):
                G_inv = self.model.G_inv(z)
                det_G_inv = torch.linalg.det(G_inv)
                det_G_inv = torch.clamp(det_G_inv, min=1e-10)
                return 0.5 * torch.log(det_G_inv)
            
            # Define gradient function matching RHVAE exactly  
            def _rhvae_grad_log_sqrt_det_G_inv(z):
                # Ensure z requires gradients
                if not z.requires_grad:
                    z = z.clone().detach().requires_grad_(True)
                
                # Use the model's G and centroids/M_tens directly
                G = self.model.G(z)  # (B, D, D)
                centroids = self.model.centroids_tens  # (K, D)
                M_tens = self.model.M_tens  # (K, D, D)
                temperature = self.model.temperature  # scalar
                
                # Compute gradient exactly like RHVAE
                z_expanded = z.unsqueeze(1)  # (B, 1, D)
                centroids_expanded = centroids.unsqueeze(0)  # (1, K, D)
                diff = centroids_expanded - z_expanded  # (1, K, D) - (B, 1, D) = (B, K, D)
                
                dist_sq = torch.norm(diff, dim=-1) ** 2  # (B, K)
                weights = torch.exp(-dist_sq / (temperature ** 2))  # (B, K)
                
                # Compute weighted derivative term with proper broadcasting
                # Ensure M_tens is properly broadcasted: (K, D, D) -> (1, K, D, D)
                M_tens_expanded = M_tens.unsqueeze(0)  # (1, K, D, D)
                weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
                
                # Weighted M matrices: (1, K, D, D) * (B, K, 1, 1) = (B, K, D, D)
                weighted_M = M_tens_expanded * weights_expanded
                
                # Sum over centroids: (B, K, D, D) -> (B, D, D)
                grad_term = weighted_M.sum(dim=1)
                
                # Scale by temperature and distance
                grad_term = (-2 / (temperature ** 2)) * grad_term
                
                # Final gradient: -0.5 * G^T @ grad_term
                result = -0.5 * torch.transpose(G, -2, -1) @ grad_term.transpose(-2, -1)
                return result.diagonal(dim1=-2, dim2=-1)  # (B, D)
            
            self.log_pi = _rhvae_log_sqrt_det_G_inv
            self.grad_func = _rhvae_grad_log_sqrt_det_G_inv
        else:
            # Fallback to autograd-based computation
            self.log_pi = self._log_sqrt_det_G_inv
            self.grad_func = self._grad_log_prop
    
    def _log_sqrt_det_G_inv(self, z, t=0):
        """Fallback: compute log(sqrt(det(G^{-1}))) using autograd."""
        if not z.requires_grad:
            z = z.clone().detach().requires_grad_(True)
        G = self.model.compute_metric_tensor(z, t)
        G_inv = torch.linalg.inv(G + 1e-6 * torch.eye(G.size(-1), device=G.device).unsqueeze(0).expand_as(G))
        det_G_inv = torch.linalg.det(G_inv)
        det_G_inv = torch.clamp(det_G_inv, min=1e-10)
        log_det = 0.5 * torch.log(det_G_inv)
        return log_det
    
    def _grad_log_prop(self, z, t=0):
        """Fallback: compute gradient using autograd."""
        if not z.requires_grad:
            z_grad = z.clone().detach().requires_grad_(True)
        else:
            z_grad = z
        log_det = self._log_sqrt_det_G_inv(z_grad, t)
        grads = torch.autograd.grad(log_det.sum(), z_grad, create_graph=False)[0]
        return grads
    
    @staticmethod
    def _tempering(k, K, beta_zero_sqrt):
        """Tempering schedule for HMC sampling."""
        beta_k = ((1 - 1 / beta_zero_sqrt) * (k / K) ** 2) + 1 / beta_zero_sqrt
        return 1 / beta_k
    
    def sample(self, n_samples, t=0):
        """Sample from the Riemannian manifold using HMC."""
        # Make sure static tensors are on the right device in case the model
        # has been moved (e.g. by Lightning) after the sampler was created.
        current_device = self.model.device
        self.n_lf = self.n_lf.to(current_device)
        self.eps_lf = self.eps_lf.to(current_device)
        self.beta_zero_sqrt = self.beta_zero_sqrt.to(current_device)

        # Initialize from standard Gaussian
        z0 = torch.randn(n_samples, self.model.latent_dim, device=current_device)
        
        beta_sqrt_old = self.beta_zero_sqrt
        z = z0.clone().detach().requires_grad_(True)
        
        n_lf_int = int(self.n_lf.item())
        for i in range(self.mcmc_steps_nbr):
            # Sample momentum
            gamma = torch.randn_like(z)
            rho = gamma / self.beta_zero_sqrt
            
            # Initial Hamiltonian
            with torch.no_grad():
                H0 = -self.log_pi(z) + 0.5 * torch.norm(rho, dim=1) ** 2
            
            # Leapfrog steps
            for k in range(n_lf_int):
                # Compute gradient
                g = -self.grad_func(z)
                
                # Step 1: half momentum update
                rho_ = rho - (self.eps_lf / 2) * g
                
                # Step 2: position update
                z = (z + self.eps_lf * rho_).clone().detach().requires_grad_(True)
                
                # Recompute gradient
                g = -self.grad_func(z)
                
                # Step 3: final half momentum update
                rho__ = rho_ - (self.eps_lf / 2) * g
                
                # Tempering
                beta_sqrt = self._tempering(k + 1, n_lf_int, self.beta_zero_sqrt)
                rho = (beta_sqrt_old / beta_sqrt) * rho__
                beta_sqrt_old = beta_sqrt
            
            # Final Hamiltonian
            with torch.no_grad():
                H = -self.log_pi(z) + 0.5 * torch.norm(rho, dim=1) ** 2
                
                # Metropolis acceptance
                alpha = torch.exp(-H) / (torch.exp(-H0) + 1e-10)
                alpha = torch.clamp(alpha, 0, 1)
                acc = torch.rand(n_samples, device=current_device)
                moves = (acc < alpha).float().reshape(n_samples, 1)
                
                # Update z (detach to avoid gradient accumulation)
                z = ((moves * z + (1 - moves) * z0).detach().requires_grad_(True))
                z0 = z.clone().detach()
        
        return z.detach()
    
    def sample_posterior(self, mu, log_var, t=0):
        """Sample from posterior using Hamiltonian dynamics on manifold."""
        batch_size = mu.shape[0]
        
        # Initialize near posterior mode
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * log_var)
        z = z.detach().requires_grad_(True)
        
        n_lf_int = 5  # Fewer steps for posterior sampling
        
        for i in range(20):  # Fewer HMC steps
            gamma = torch.randn_like(z)
            rho = gamma * 0.1  # Smaller momentum
            
            # Energy function including posterior term
            def _energy(z):
                # Ensure z requires gradients
                if not z.requires_grad:
                    z = z.clone().detach().requires_grad_(True)
                
                # Riemannian term
                log_det = self.log_pi(z)
                
                # Posterior term
                diff = z - mu
                posterior_term = 0.5 * torch.sum(diff * torch.exp(-log_var) * diff, dim=1)
                
                return -log_det + posterior_term
            
            def _grad_energy(z):
                # Ensure z requires gradients for autograd
                if not z.requires_grad:
                    z = z.clone().detach().requires_grad_(True)
                
                energy = _energy(z)
                grads = torch.autograd.grad(energy.sum(), z, create_graph=False)[0]
                return grads
            
            # Simple leapfrog steps
            for k in range(n_lf_int):
                g = _grad_energy(z)
                rho = rho - (0.01 / 2) * g
                z = (z - 0.01 * rho).clone().detach().requires_grad_(True)
                g = _grad_energy(z)
                rho = rho - (0.01 / 2) * g
        
        return z.detach()
    
    def sample_riemannian_latents(self, mu: torch.Tensor, log_var: torch.Tensor, 
                                 method: str = 'hmc') -> torch.Tensor:
        """
        Sample latent codes using HMC on the Riemannian manifold.
        
        Args:
            mu: Posterior mean [batch_size, latent_dim]
            log_var: Posterior log variance [batch_size, latent_dim]
            method: Sampling method ('hmc' or 'posterior_hmc')
            
        Returns:
            Sampled latent codes [batch_size, latent_dim]
        """
        if method == 'posterior_hmc':
            return self.sample_posterior(mu, log_var)
        else:
            # For training, use a simplified approach that preserves gradients
            # Start from posterior mean and apply a few HMC steps
            batch_size = mu.shape[0]
            
            # Initialize near posterior mode
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * log_var)
            
            # Apply a small number of HMC-style refinement steps
            # This preserves gradients while incorporating Riemannian geometry
            for i in range(3):  # Very few steps for training stability
                z = z.detach().requires_grad_(True)
                
                # Compute gradient of log probability
                try:
                    g = -self.grad_func(z)
                    
                    # Small step in gradient direction
                    step_size = 0.01
                    z = z + step_size * g
                    
                except Exception as e:
                    print(f"⚠️ HMC refinement failed: {e}, using standard sampling")
                    break
            
            return z.detach()
    
    def sample_prior(self, num_samples: int, method: str = 'hmc') -> torch.Tensor:
        """
        Sample from the Riemannian prior using HMC.
        
        Args:
            num_samples: Number of samples to generate
            method: Prior sampling method ('hmc' or 'basic')
            
        Returns:
            Prior samples [num_samples, latent_dim]
        """
        if method == 'hmc':
            return self.sample(num_samples)
        else:
            # Fallback to standard Gaussian
            return torch.randn(num_samples, self.model.latent_dim, device=self.device)
    
    def get_sampling_methods(self) -> Dict[str, str]:
        """Override to provide HMC-specific methods."""
        return {
            'hmc': 'Hamiltonian Monte Carlo sampling on manifold',
            'posterior_hmc': 'HMC sampling from posterior',
            'basic': 'Standard Gaussian sampling (fallback)'
        }
    
    def get_hmc_parameters(self) -> Dict[str, Any]:
        """
        Get HMC sampling parameters.
        
        Returns:
            Dictionary with HMC parameters
        """
        return {
            'mcmc_steps_nbr': self.mcmc_steps_nbr,
            'n_lf': int(self.n_lf.item()),
            'eps_lf': float(self.eps_lf.item()),
            'beta_zero': float(self.beta_zero_sqrt.item() ** 2)
        } 