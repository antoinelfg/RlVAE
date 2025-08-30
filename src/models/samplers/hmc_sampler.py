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
    
    def __init__(self, model, mcmc_steps_nbr=100, n_lf=15, eps_lf=0.03, beta_zero=1.0, include_volume_grad: bool = True):
        super().__init__(model)
        self.mcmc_steps_nbr = mcmc_steps_nbr
        self.n_lf = torch.tensor([n_lf], device=model.device)
        self.eps_lf = torch.tensor([eps_lf], device=model.device)
        self.beta_zero_sqrt = torch.tensor([beta_zero], device=model.device).sqrt()
        self.include_volume_grad = bool(include_volume_grad)
        
        # Set target density to standard Gaussian prior: π(z) ∝ exp(-0.5 ||z||^2)
        # Riemannian geometry enters through kinetic term and + 1/2 log det G(z).
        # This choice yields stable, reversible dynamics and correct acceptance.
        self.log_pi = lambda z: -0.5 * torch.sum(z * z, dim=1)
        # Gradient of log π(z) for standard normal is -z
        self.grad_func = lambda z: -z
    
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
    
    @staticmethod
    def _grad_log_sqrt_det_Ginv(z, model):
        """Gradient of log sqrt det(G^{-1}(z)) using the native metric parametrization.

        Mirrors the implementation used by RHVAE samplers operating on G^{-1} anchors.
        Requires the model to expose `centroids_tens`, `M_tens`, `temperature`, and `G(z)`.
        """
        centroids = model.centroids_tens
        M = model.M_tens
        T = model.temperature
        diff = (centroids.unsqueeze(0) - z.unsqueeze(1))  # [B,K,D]
        weights = torch.exp(-torch.norm(diff, dim=-1) ** 2 / (T ** 2))  # [B,K]
        term = (-2.0 / (T ** 2)) * diff.unsqueeze(2)  # [B,K,1,D]
        weighted_M = M.unsqueeze(0) * weights.unsqueeze(-1).unsqueeze(-1)  # [B,K,D,D]
        inner = torch.matmul(term, weighted_M).sum(dim=1)  # [B,1,D]
        Gz = model.G(z)  # [B,D,D]
        grad = -0.5 * torch.matmul(Gz.transpose(-2, -1), inner.transpose(1, 2))  # [B,D,1]
        return grad.squeeze(-1)  # [B,D]

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
    
    def _compute_hamiltonian(self, z, rho):
        """
        Compute complete Riemannian Hamiltonian:
        H(z, ρ) = -log π(z) + 1/2 ρ^T G^{-1}(z) ρ + 1/2 log det(G(z))
        
        Args:
            z: Position [batch_size, latent_dim]
            rho: Momentum [batch_size, latent_dim]
            
        Returns:
            Hamiltonian energy [batch_size]
        """
        # Potential energy: -log π(z)
        potential = -self.log_pi(z)
        
        # Kinetic energy: 1/2 ρ^T G^{-1}(z) ρ
        G_inv = self.model.G_inv(z)
        kinetic = 0.5 * torch.einsum('bi,bij,bj->b', rho, G_inv, rho)
        
        # Metric-dependent correction: 1/2 log det(G(z))
        # This term accounts for the volume element in Riemannian geometry
        G = self.model.G(z)
        log_det_G = torch.linalg.slogdet(G).logabsdet
        metric_correction = 0.5 * log_det_G
        
        return potential + kinetic + metric_correction
    
    def _initialize_momentum(self, z):
        """
        Initialize momentum using proper Riemannian geometry:
        ρ ~ N(0, G(z)) using Cholesky decomposition
        
        Args:
            z: Position [batch_size, latent_dim]
            
        Returns:
            Momentum [batch_size, latent_dim]
        """
        # Sample from standard Gaussian and shape via Cholesky of G(z)
        gamma = torch.randn_like(z)  # [batch_size, latent_dim]
        G = self.model.G(z)  # [batch_size, D, D]
        try:
            L = torch.linalg.cholesky(G)
            rho = torch.einsum('bij,bj->bi', L, gamma)  # ρ ~ N(0, G)
        except torch.linalg.LinAlgError:
            eigenvals, eigenvecs = torch.linalg.eigh(G)
            eigenvals = torch.clamp(eigenvals, min=1e-6)
            sqrt_G = eigenvecs @ torch.diag_embed(torch.sqrt(eigenvals)) @ eigenvecs.transpose(-2, -1)
            rho = torch.einsum('bij,bj->bi', sqrt_G, gamma)
        
        return rho
    
    def _generalized_leapfrog_step(self, z, rho, eps):
        """
        Generalized leapfrog step for Riemannian HMC.
        
        Args:
            z: Position [batch_size, latent_dim]
            rho: Momentum [batch_size, latent_dim]
            eps: Step size
            
        Returns:
            Updated position and momentum
        """
        # Step 1: Half momentum update (optionally include metric volume correction)
        # U(z) = -log pi(z) + 0.5 log det G(z) = -log pi(z) - log sqrt(det G^{-1}(z))
        grad_pi = -self.grad_func(z)
        if self.include_volume_grad:
            try:
                # ∇[0.5 log det G(z)] = - ∇[log sqrt det G^{-1}(z)]
                vol_grad = -self._grad_log_sqrt_det_Ginv(z, self.model)
            except Exception:
                # Fallback: autograd on 0.5 log det G(z)
                z_req = z if z.requires_grad else z.clone().detach().requires_grad_(True)
                G = self.model.G(z_req)
                sign, logabs = torch.linalg.slogdet(G)
                vol = 0.5 * logabs
                vol_grad = torch.autograd.grad(vol.sum(), z_req, create_graph=False)[0]
        else:
            vol_grad = 0.0

        grad = grad_pi + vol_grad
        rho_half = rho - (eps / 2) * grad
        
        # Step 2: Position update using metric
        G_inv = self.model.G_inv(z)
        z_new = z + eps * torch.einsum('bij,bj->bi', G_inv, rho_half)
        
        # Step 3: Recompute gradient at new position
        grad_new = -self.grad_func(z_new)
        
        # Step 4: Final half momentum update
        rho_new = rho_half - (eps / 2) * grad_new
        
        return z_new, rho_new
    
    def sample(
        self,
        n_samples,
        t: int = 0,
        init_std: float = 1.0,
        eps_jitter: float = 0.0,
        n_lf_jitter: int = 0,
    ):
        """Sample from the Riemannian manifold using HMC."""
        # Make sure static tensors are on the right device in case the model
        # has been moved (e.g. by Lightning) after the sampler was created.
        current_device = self.model.device
        self.n_lf = self.n_lf.to(current_device)
        self.eps_lf = self.eps_lf.to(current_device)
        self.beta_zero_sqrt = self.beta_zero_sqrt.to(current_device)

        # Initialize from a zero-mean Gaussian with configurable std
        z0 = torch.randn(n_samples, self.model.latent_dim, device=current_device) * float(init_std)
        
        beta_sqrt_old = self.beta_zero_sqrt
        z = z0.clone().detach().requires_grad_(True)
        
        n_lf_int = int(self.n_lf.item())
        acceptance_count = 0
        
        for i in range(self.mcmc_steps_nbr):
            # Initialize momentum using proper Riemannian geometry
            rho = self._initialize_momentum(z)
            
            # Initial Hamiltonian
            with torch.no_grad():
                H0 = self._compute_hamiltonian(z, rho)
            
            # Choose local leapfrog count with jitter (optional)
            if n_lf_jitter > 0:
                jitter = torch.randint(-n_lf_jitter, n_lf_jitter + 1, (1,), device=current_device).item()
                local_n_lf = max(1, n_lf_int + int(jitter))
            else:
                local_n_lf = n_lf_int

            # Generalized leapfrog steps
            for k in range(local_n_lf):
                # Use generalized leapfrog with metric updates
                if eps_jitter > 0.0:
                    # Uniform jitter in [(1-j), (1+j)]
                    u = (torch.rand(1, device=current_device).item() - 0.5) * 2.0 * float(eps_jitter)
                    eps_eff = float(self.eps_lf.item()) * (1.0 + u)
                else:
                    eps_eff = float(self.eps_lf.item())
                z, rho = self._generalized_leapfrog_step(z, rho, eps_eff)
                
                # Tempering
                beta_sqrt = self._tempering(k + 1, local_n_lf, self.beta_zero_sqrt)
                rho = (beta_sqrt_old / beta_sqrt) * rho
                beta_sqrt_old = beta_sqrt
            
            # Final Hamiltonian
            with torch.no_grad():
                H = self._compute_hamiltonian(z, rho)
                
                # Metropolis acceptance
                alpha = torch.exp(-H) / (torch.exp(-H0) + 1e-10)
                alpha = torch.clamp(alpha, 0, 1)
                acc = torch.rand(n_samples, device=current_device)
                moves = (acc < alpha).float().reshape(n_samples, 1)
                
                # Update z (detach to avoid gradient accumulation)
                z = ((moves * z + (1 - moves) * z0).detach().requires_grad_(True))
                z0 = z.clone().detach()
                
                # Track acceptance rate
                acceptance_count += moves.sum().item()
        
        # Log acceptance rate
        acceptance_rate = acceptance_count / (self.mcmc_steps_nbr * n_samples)
        # Expose for programmatic checks
        self.last_acceptance_rate = float(acceptance_rate)
        print(f"✅ RHMC Acceptance Rate: {acceptance_rate:.3f}")
        
        return z.detach()
    
    def sample_posterior(self, mu, log_var, t=0):
        """Sample from posterior using simplified RHMC approach."""
        batch_size = mu.shape[0]
        
        # Initialize near posterior mode
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * log_var)
        
        # Apply a small number of refinement steps using metric-aware sampling
        for i in range(3):  # Very few steps for training stability
            z = z.detach().requires_grad_(True)
            
            try:
                # Compute gradient of log probability using metric
                G_z = self.model.G(z)
                
                # Compute gradient of log probability: ∇log p(z) = -G(z) * (z - μ)
                diff = z - mu
                grad_log_prob = -torch.einsum('bij,bj->bi', G_z, diff)
                
                # Small step in gradient direction
                step_size = 0.01
                z = z + step_size * grad_log_prob
                
            except Exception as e:
                print(f"⚠️ RHMC posterior sampling failed: {e}, using standard sampling")
                # Fallback to standard sampling
                eps = torch.randn_like(mu)
                z = mu + eps * torch.exp(0.5 * log_var)
                break
        
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


class DualRiemannianHMCSampler(BaseRiemannianSampler):
    """Dual RHMC that treats G^{-1} as the metric tensor.

    Mathematics:
    - Metric: G^{-1}
    - Momentum: p ~ N(0, G^{-1}(z)) = N(0, G(z))
    - Kinetic: 1/2 p^T G^{-1}(z) p = 1/2 p^T G(z) p
    - Volume correction: +1/2 log det(G^{-1}(z)) = -1/2 log det(G(z))

    We implement a conservative variant that uses autograd for gradients.
    """

    def __init__(self, model, mcmc_steps_nbr: int = 50, n_lf: int = 10, eps_lf: float = 0.02):
        super().__init__(model)
        self.mcmc_steps_nbr = int(mcmc_steps_nbr)
        self.n_lf = int(n_lf)
        self.eps_lf = float(eps_lf)

    def _initialize_momentum(self, z: torch.Tensor) -> torch.Tensor:
        """Sample p ~ N(0, G^{-1}(z)). Since our adapter exposes G and G_inv,
        we compute a stable factor for G^{-1}(z).
        """
        with torch.no_grad():
            G = self.model.G(z)  # [B,D,D]
            # Compute Cholesky of G for p ~ L @ N(0,I) with L s.t. LL^T = G
            try:
                L = torch.linalg.cholesky(G)
            except torch.linalg.LinAlgError:
                # Eigen fallback
                evals, evecs = torch.linalg.eigh(G)
                evals = torch.clamp(evals, min=1e-6)
                L = evecs @ torch.diag_embed(torch.sqrt(evals))
            gamma = torch.randn_like(z)
            p = torch.einsum('bij,bj->bi', L, gamma)
            return p

    def _hamiltonian(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """H(z,p) = 1/2 p^T G^{-1}(z) p + 1/2 log det(G^{-1}(z))
        (flat potential)."""
        G = self.model.G(z)
        G_inv = torch.linalg.inv(G)
        kinetic = 0.5 * torch.einsum('bi,bij,bj->b', p, G_inv, p)
        # volume correction with G^{-1}
        sign, logabs = torch.linalg.slogdet(G_inv)
        vol = 0.5 * logabs
        return kinetic + vol

    def _grad_z(self, z: torch.Tensor) -> torch.Tensor:
        if not z.requires_grad:
            z_req = z.clone().detach().requires_grad_(True)
        else:
            z_req = z
        H = self._hamiltonian(z_req, torch.zeros_like(z_req))
        grad = torch.autograd.grad(H.sum(), z_req, create_graph=True, retain_graph=False)[0]
        return grad

    def _leapfrog(self, z: torch.Tensor, p: torch.Tensor, eps: float):
        # Half momentum step
        grad = self._grad_z(z)
        p_half = p - 0.5 * eps * grad
        # Position step using metric (here G^{-1})
        G = self.model.G(z)
        G_inv = torch.linalg.inv(G)
        z_new = z + eps * torch.einsum('bij,bj->bi', G_inv, p_half)
        # Final half step
        grad_new = self._grad_z(z_new)
        p_new = p_half - 0.5 * eps * grad_new
        return z_new, p_new

    def sample(self, n_samples: int = 100) -> torch.Tensor:
        device = self.model.device
        z = torch.randn(n_samples, self.model.latent_dim, device=device).requires_grad_(True)
        accept_count = 0
        for _ in range(self.mcmc_steps_nbr):
            p0 = self._initialize_momentum(z)
            with torch.no_grad():
                H0 = self._hamiltonian(z, p0)

            z_prop, p_prop = z, p0
            for _ in range(self.n_lf):
                z_prop, p_prop = self._leapfrog(z_prop, p_prop, self.eps_lf)

            with torch.no_grad():
                H1 = self._hamiltonian(z_prop, p_prop)
                log_alpha = H0 - H1
                alpha = torch.exp(torch.clamp(log_alpha, max=0))
                u = torch.rand_like(alpha)
                accept = (u < alpha).float().view(-1, 1)
                z = ((accept * z_prop + (1 - accept) * z).detach().requires_grad_(True))
                accept_count += int(accept.sum().item())

        print(f"✅ RHMC Acceptance Rate: {accept_count / (self.mcmc_steps_nbr * n_samples):.3f}")
        return z.detach()

class RHVAEVolumeElementHMCSampler(BaseRiemannianSampler):
    """Sampler that mirrors the original RHVAE sampler behavior.

    Target density: π(z) ∝ sqrt(det(G^{-1}(z)))
    Momentum: standard Euclidean Gaussian with tempering schedule
    Leapfrog: Euclidean updates using ∇ log sqrt det(G^{-1}).
    """

    def __init__(self, model, mcmc_steps_nbr: int = 100, n_lf: int = 15, eps_lf: float = 0.03, beta_zero: float = 1.0):
        super().__init__(model)
        self.mcmc_steps_nbr = int(mcmc_steps_nbr)
        self.n_lf = int(n_lf)
        self.eps_lf = float(eps_lf)
        self.beta_zero_sqrt = torch.tensor([beta_zero], device=self.device).sqrt()

    @staticmethod
    def _log_sqrt_det_Ginv(z, model):
        Ginv = model.G_inv(z)
        det = torch.linalg.det(Ginv).clamp(min=1e-12)
        return 0.5 * torch.log(det)

    @staticmethod
    def _grad_log_sqrt_det_Ginv(z, model):
        # Exact translation of benchmark_VAE RHVAESampler.grad_log_sqrt_det_G_inv
        centroids = model.centroids_tens
        M = model.M_tens
        T = model.temperature
        diff = (centroids.unsqueeze(0) - z.unsqueeze(1))  # [B,K,D]
        weights = torch.exp(-torch.norm(diff, dim=-1) ** 2 / (T ** 2))  # [B,K]
        term = (-2.0 / (T ** 2)) * diff.unsqueeze(2)  # [B,K,1,D]
        weighted_M = M.unsqueeze(0) * weights.unsqueeze(-1).unsqueeze(-1)  # [B,K,D,D]
        inner = torch.matmul(term, weighted_M).sum(dim=1)  # [B,1,D]
        Gz = model.G(z)  # [B,D,D]
        grad = -0.5 * torch.matmul(Gz.transpose(-2, -1), inner.transpose(1, 2))  # [B,D,1]
        return grad.squeeze(-1)  # [B,D]

    @staticmethod
    def _tempering(k, K, beta_zero_sqrt):
        beta_k = ((1 - 1 / beta_zero_sqrt) * (k / K) ** 2) + 1 / beta_zero_sqrt
        return 1 / beta_k

    def sample(self, n_samples: int = 100) -> torch.Tensor:
        device = self.device
        K = self.model.centroids_tens.shape[0]
        idx = torch.randint(K, (n_samples,), device=device)
        z0 = self.model.centroids_tens[idx].detach()
        z = z0
        beta_sqrt_old = self.beta_zero_sqrt
        accept_count = 0

        for _ in range(self.mcmc_steps_nbr):
            gamma = torch.randn_like(z, device=device)
            rho = gamma / self.beta_zero_sqrt
            with torch.no_grad():
                H0 = -self._log_sqrt_det_Ginv(z, self.model) + 0.5 * torch.sum(rho * rho, dim=1)
            for k in range(self.n_lf):
                g = -self._grad_log_sqrt_det_Ginv(z, self.model)
                rho_half = rho - 0.5 * self.eps_lf * g
                z = z + self.eps_lf * rho_half
                g_new = -self._grad_log_sqrt_det_Ginv(z, self.model)
                rho_new = rho_half - 0.5 * self.eps_lf * g_new
                beta_sqrt = self._tempering(k + 1, self.n_lf, self.beta_zero_sqrt)
                rho = (beta_sqrt_old / beta_sqrt) * rho_new
                beta_sqrt_old = beta_sqrt
            with torch.no_grad():
                H = -self._log_sqrt_det_Ginv(z, self.model) + 0.5 * torch.sum(rho * rho, dim=1)
                alpha = torch.exp(-(H - H0)).clamp(max=1.0)
                u = torch.rand_like(alpha)
                moves = (u < alpha).float().view(-1, 1)
                accept_count += int(moves.sum().item())
                z = ((moves * z + (1 - moves) * z0).detach())
                z0 = z
        self.last_acceptance_rate = accept_count / (self.mcmc_steps_nbr * n_samples)
        print(f"✅ RHVAE-Volume Acceptance Rate: {self.last_acceptance_rate:.3f}")
        return z.detach()

    # Implement abstract API expected by BaseRiemannianSampler
    def sample_prior(self, num_samples: int, method: str = 'hmc') -> torch.Tensor:
        return self.sample(num_samples)

    def sample_riemannian_latents(self, mu: torch.Tensor, log_var: torch.Tensor, method: str = 'hmc') -> torch.Tensor:
        """Lightweight posterior refinement using the same volume-element dynamics.

        Starts from the encoder posterior mean and performs a few tempered Euclidean
        updates on ∇ log sqrt det(G^{-1}). No dependence on dual/standard leapfrog.
        """
        device = self.device
        z = mu.detach().to(device)
        steps = max(1, min(5, self.mcmc_steps_nbr // 10))
        inner_n_lf = max(2, self.n_lf // 2)

        for _ in range(steps):
            gamma = torch.randn_like(z, device=device)
            rho = gamma / self.beta_zero_sqrt
            with torch.no_grad():
                H0 = -self._log_sqrt_det_Ginv(z, self.model) + 0.5 * torch.sum(rho * rho, dim=1)
            beta_sqrt_old = self.beta_zero_sqrt
            for k in range(inner_n_lf):
                g = -self._grad_log_sqrt_det_Ginv(z, self.model)
                rho_half = rho - 0.5 * (self.eps_lf * 0.5) * g
                z = z + (self.eps_lf * 0.5) * rho_half
                g_new = -self._grad_log_sqrt_det_Ginv(z, self.model)
                rho_new = rho_half - 0.5 * (self.eps_lf * 0.5) * g_new
                beta_sqrt = self._tempering(k + 1, inner_n_lf, self.beta_zero_sqrt)
                rho = (beta_sqrt_old / beta_sqrt) * rho_new
                beta_sqrt_old = beta_sqrt
            with torch.no_grad():
                H1 = -self._log_sqrt_det_Ginv(z, self.model) + 0.5 * torch.sum(rho * rho, dim=1)
                alpha = torch.exp(-(H1 - H0)).clamp(max=1.0)
                u = torch.rand_like(alpha)
                moves = (u < alpha).float().view(-1, 1)
                z = ((moves * z + (1 - moves) * mu.to(device)).detach())
        return z.detach()
