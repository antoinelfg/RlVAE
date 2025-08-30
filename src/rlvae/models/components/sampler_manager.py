"""
SamplerManager: unifies training/posterior sampling choices
==========================================================

Provides a single interface to select among:
- Official RHVAE sampler (posterior sampling, differentiable variant)
- Riemannian HMC sampler (training refinement)
- Working Riemannian samplers: enhanced / geodesic / basic
- Default local metric–aligned Gaussian posterior sampler
"""

from typing import Optional
import torch

from .posterior_sampler import PosteriorSampler
from .riemannian_sampler import RiemannianSampler

try:
    from ...riemannian_flow_vae import OfficialRHVAESampler  # canonical re-export
except Exception:
    OfficialRHVAESampler = None  # optional

try:
    # Top-level HMC sampler (repo-local)
    from src.models.samplers.hmc_sampler import RiemannianHMCSampler
except Exception:
    RiemannianHMCSampler = None


class SamplerManager:
    def __init__(self, model):
        self.model = model
        self.device = model.device if hasattr(model, 'device') else next(model.parameters()).device
        self.posterior_sampler = PosteriorSampler(model)
        self.riemannian_sampler = RiemannianSampler(model)
        self._official_sampler = None
        self._hmc_sampler = None

    def setup_official(self):
        if OfficialRHVAESampler is None:
            raise RuntimeError("OfficialRHVAESampler not available")
        self._official_sampler = OfficialRHVAESampler(self.model)
        # The official sampler will lazy-setup when first used; explicit setup optional
        try:
            self._official_sampler.setup_official_rhvae()
        except Exception:
            # Will attempt setup on first sampling instead
            pass

    def _ensure_hmc(self):
        if self._hmc_sampler is None:
            if RiemannianHMCSampler is None:
                raise RuntimeError("RiemannianHMCSampler not available")
            self._hmc_sampler = RiemannianHMCSampler(self.model)

    def sample_training(self, mu: torch.Tensor, log_var: torch.Tensor, posterior_type: str, method: Optional[str] = None) -> torch.Tensor:
        """Select training-time posterior sampling method.

        Args:
            mu, log_var: encoder outputs
            posterior_type: 'riemannian_metric' | 'gaussian'
            method: 'official' | 'hmc' | 'posterior_hmc' | 'enhanced' | 'geodesic' | 'basic' | None
        """
        if posterior_type == 'riemannian_metric':
            # Always prefer local metric–aligned Gaussian for differentiability
            return self.posterior_sampler.sample_metric_aware_posterior(mu, log_var)

        # Gaussian posterior type with manifold-aware alternatives
        m = (method or '').lower()
        if m == 'official' and self._official_sampler is not None:
            try:
                return self._official_sampler.sample_for_training(mu, log_var)
            except Exception:
                # fallback to local gaussian
                pass
        elif m in ('hmc', 'posterior_hmc'):
            self._ensure_hmc()
            if m == 'posterior_hmc':
                return self._hmc_sampler.sample_posterior(mu, log_var)
            else:
                # small refinement steps
                return self._hmc_sampler.sample_riemannian_latents(mu, log_var, method='hmc')
        elif m in ('enhanced', 'geodesic', 'basic'):
            return self.riemannian_sampler.sample_riemannian_latents(mu, log_var, method=m)

        # Fallback: standard reparameterization
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * log_var)

