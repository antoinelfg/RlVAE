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
        # TRACE: print only once per process
        self._trace_printed = False

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
        # DEBUG: Always print what we're doing
        if not hasattr(self, '_debug_printed'):
            print(f"[SamplerManager DEBUG] posterior_type={posterior_type}, method={method}")
            self._debug_printed = True
        
        if posterior_type == 'riemannian_metric':
            # Always prefer local metric–aligned Gaussian for differentiability
            out = self.posterior_sampler.sample_metric_aware_posterior(mu, log_var)
            try:
                import os
                if os.environ.get('RLVAE_TRACE', '0') == '1' and not self._trace_printed:
                    print('USING LOCAL METRIC-ALIGNED GAUSSIAN (posterior_type=riemannian_metric)')
                    self._trace_printed = True
            except Exception:
                pass
            return out

        # Gaussian posterior type with manifold-aware alternatives
        m = (method or '').lower()
        if m == 'official' and self._official_sampler is not None:
            try:
                return self._official_sampler.sample_for_training(mu, log_var)
            except Exception:
                # fallback to local gaussian
                pass
        elif m in ('hmc', 'posterior_hmc'):
            print(f"[SamplerManager DEBUG] Using HMC sampler with method={m}")
            self._ensure_hmc()
            if m == 'posterior_hmc':
                try:
                    import os
                    if os.environ.get('RLVAE_TRACE', '0') == '1' and not self._trace_printed:
                        print('USING RIEMANNIAN RHMC POSTERIOR (posterior_hmc)')
                        self._trace_printed = True
                except Exception:
                    pass
                return self._hmc_sampler.sample_posterior(mu, log_var)
            else:
                # small refinement steps
                try:
                    import os
                    if os.environ.get('RLVAE_TRACE', '0') == '1' and not self._trace_printed:
                        print('USING RIEMANNIAN RHMC POSTERIOR (training refinement hmc)')
                        self._trace_printed = True
                except Exception:
                    pass
                return self._hmc_sampler.sample_riemannian_latents(mu, log_var, method='hmc')
        elif m in ('enhanced', 'geodesic', 'basic'):
            try:
                import os
                if os.environ.get('RLVAE_TRACE', '0') == '1' and not self._trace_printed:
                    print(f'USING RiemannianSampler ({m})')
                    self._trace_printed = True
            except Exception:
                pass
            return self.riemannian_sampler.sample_riemannian_latents(mu, log_var, method=m)

        # Fallback: standard reparameterization
        print(f"[SamplerManager DEBUG] FALLBACK to standard reparam - method='{method}', posterior_type='{posterior_type}'")
        eps = torch.randn_like(mu)
        try:
            import os
            if os.environ.get('RLVAE_TRACE', '0') == '1' and not self._trace_printed:
                print('⚠️ Riemannian sampling failed or not selected, using STANDARD reparam posterior')
                self._trace_printed = True
        except Exception:
            pass
        return mu + eps * torch.exp(0.5 * log_var)
