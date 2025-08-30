"""
Riemannian Sampler Component (Training + Prior)
===============================================

Modular implementation of the geodesic/enhanced/basic training samplers and
several prior sampling strategies, adapted from the original implementation.
"""

from typing import Optional
import torch
import torch.nn as nn


class RiemannianSampler(nn.Module):
    """Geodesic-aware Riemannian sampler using model.G and model.G_inv."""

    def __init__(self, model: nn.Module):
        super().__init__()
        import weakref
        self._ctx = {'model': weakref.proxy(model)}
        self.device = getattr(model, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # ---------- Training-time posterior sampling ----------
    def sample_riemannian_latents(self, mu: torch.Tensor, log_var: torch.Tensor, method: str = 'enhanced') -> torch.Tensor:
        if method == 'geodesic':
            return self.sample_geodesic_riemannian_latents(mu, log_var)
        elif method == 'enhanced':
            return self.sample_enhanced_riemannian_latents(mu, log_var)
        elif method == 'basic':
            return self.sample_basic_riemannian_latents(mu, log_var)
        # Fallback to standard reparameterization
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * log_var)

    def sample_enhanced_riemannian_latents(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        z_standard = mu + eps * torch.exp(0.5 * log_var)
        if not (hasattr(self._ctx['model'], 'centroids_tens') and hasattr(self._ctx['model'], 'G_inv')):
            return z_standard
        try:
            centroids = self._ctx['model'].centroids_tens  # [K, D]
            mu_exp = mu.unsqueeze(1)               # [B, 1, D]
            cen_exp = centroids.unsqueeze(0)       # [1, K, D]
            distances = torch.norm(mu_exp - cen_exp, dim=-1)  # [B, K]
            _, top2_idx = torch.topk(distances, k=2, dim=-1, largest=False)
            top2_dist = torch.gather(distances, 1, top2_idx)              # [B, 2]
            weights = (1.0 / (top2_dist + 1e-8))
            weights = weights / weights.sum(dim=-1, keepdim=True)         # [B, 2]
            c1 = centroids[top2_idx[:, 0]]                                 # [B, D]
            c2 = centroids[top2_idx[:, 1]]                                 # [B, D]
            virt_c = weights[:, 0:1] * c1 + weights[:, 1:2] * c2           # [B, D]
            G_inv_virtual = self._ctx['model'].G_inv(virt_c)                       # [B, D, D]
            try:
                L = torch.linalg.cholesky(G_inv_virtual + 1e-6 * torch.eye(G_inv_virtual.shape[-1], device=G_inv_virtual.device))
                eps_t = torch.einsum('bij,bj->bi', L, eps)
            except Exception:
                evals, evecs = torch.linalg.eigh(G_inv_virtual)
                evals = torch.clamp(evals, min=1e-6)
                sqrtGinv = evecs @ torch.diag_embed(torch.sqrt(evals)) @ evecs.transpose(-2, -1)
                eps_t = torch.einsum('bij,bj->bi', sqrtGinv, eps)
            influence = 0.15
            return mu + eps_t * torch.exp(0.5 * log_var) * influence + eps * torch.exp(0.5 * log_var) * (1.0 - influence)
        except Exception:
            return z_standard

    def sample_geodesic_riemannian_latents(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        z_standard = mu + eps * torch.exp(0.5 * log_var)
        if not (hasattr(self._ctx['model'], 'centroids_tens') and hasattr(self._ctx['model'], 'G_inv')):
            return z_standard
        try:
            B = mu.shape[0]
            centroids = self._ctx['model'].centroids_tens
            mu_exp = mu.unsqueeze(1)
            cen_exp = centroids.unsqueeze(0)
            distances = torch.norm(mu_exp - cen_exp, dim=-1)
            _, idx = torch.topk(distances, k=2, dim=-1, largest=False)
            c1 = centroids[idx[:, 0]]
            c2 = centroids[idx[:, 1]]
            t = torch.rand(B, 1, device=self.device)
            z_geo = (1 - t) * c1 + t * c2
            dir_vec = (c2 - c1)
            dir_vec = dir_vec / (torch.norm(dir_vec, dim=-1, keepdim=True) + 1e-8)
            mu_to_geo = mu - z_geo
            parallel = torch.sum(mu_to_geo * dir_vec, dim=-1, keepdim=True) * dir_vec
            perp = mu_to_geo - parallel
            G_inv_geo = self._ctx['model'].G_inv(z_geo)
            try:
                G_geo = torch.linalg.inv(G_inv_geo)
                L = torch.linalg.cholesky(G_geo + 1e-6 * torch.eye(G_geo.shape[-1], device=G_geo.device))
                eps_perp = torch.einsum('bij,bj->bi', L, eps)
            except Exception:
                G_geo = torch.linalg.inv(G_inv_geo)
                evals, evecs = torch.linalg.eigh(G_geo)
                evals = torch.clamp(evals, min=1e-6)
                sqrtG = evecs @ torch.diag_embed(torch.sqrt(evals)) @ evecs.transpose(-2, -1)
                eps_perp = torch.einsum('bij,bj->bi', sqrtG, eps)
            geo_scale = 0.3
            z_final = z_geo + geo_scale * eps_perp * torch.exp(0.5 * log_var) + (1.0 - geo_scale) * (mu - z_geo) + 0.1 * parallel
            return z_final
        except Exception:
            return z_standard

    def sample_basic_riemannian_latents(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * log_var)
        if not hasattr(self.model, 'G_inv'):
            return z
        try:
            G_inv_z = self.model.G_inv(z)
            try:
                L = torch.linalg.cholesky(G_inv_z + 1e-6 * torch.eye(G_inv_z.shape[-1], device=G_inv_z.device))
                eps_t = torch.einsum('bij,bj->bi', L, eps)
            except Exception:
                evals, evecs = torch.linalg.eigh(G_inv_z)
                evals = torch.clamp(evals, min=1e-6)
                sqrtGinv = evecs @ torch.diag_embed(torch.sqrt(evals)) @ evecs.transpose(-2, -1)
                eps_t = torch.einsum('bij,bj->bi', sqrtGinv, eps)
            corr = 0.1
            return mu + eps_t * torch.exp(0.5 * log_var) * corr + eps * torch.exp(0.5 * log_var) * (1.0 - corr)
        except Exception:
            return z

    # ---------- Prior sampling (geodesic/centroid/mix/basic) ----------
    @torch.no_grad()
    def sample_prior(self, num_samples: int, method: str = 'geodesic', temperature: float = 1.0) -> torch.Tensor:
        if method == 'geodesic':
            return self.sample_geodesic_prior(num_samples)
        elif method == 'centroid_aware':
            return self.sample_centroid_aware_prior(num_samples)
        elif method == 'weighted_mixture':
            return self.sample_weighted_mixture_prior(num_samples)
        else:
            return self.sample_basic_prior(num_samples)

    @torch.no_grad()
    def sample_geodesic_prior(self, num_samples: int) -> torch.Tensor:
        if not hasattr(self._ctx['model'], 'centroids_tens'):
            return torch.randn(num_samples, self.model.latent_dim, device=self.device) * 0.5
        K = len(self._ctx['model'].centroids_tens)
        start_idx = torch.randint(0, K, (num_samples,), device=self.device)
        end_idx = torch.randint(0, K, (num_samples,), device=self.device)
        start = self._ctx['model'].centroids_tens[start_idx]
        end = self._ctx['model'].centroids_tens[end_idx]
        t = torch.rand(num_samples, 1, device=self.device)
        z_path = (1 - t) * start + t * end
        if hasattr(self._ctx['model'], 'G_inv'):
            G_inv = self._ctx['model'].G_inv(z_path)
            try:
                L = torch.linalg.cholesky(G_inv)
                eps = torch.randn(num_samples, self.model.latent_dim, device=self.device)
                eps_metric = torch.einsum('bij,bj->bi', L, eps)
            except Exception:
                evals, evecs = torch.linalg.eigh(G_inv)
                evals = torch.clamp(evals, min=1e-8)
                sqrtGinv = evecs @ torch.diag_embed(torch.sqrt(evals)) @ evecs.transpose(-2, -1)
                eps = torch.randn(num_samples, self.model.latent_dim, device=self.device)
                eps_metric = torch.einsum('bij,bj->bi', sqrtGinv, eps)
            detGinv = torch.linalg.det(G_inv)
            scale = torch.clamp(detGinv ** (1.0 / (2.0 * self.model.latent_dim)), 0.1, 1.0)
            z = z_path + eps_metric * (0.4 / (scale + 1e-6)).unsqueeze(-1)
            return z
        return z_path

    @torch.no_grad()
    def sample_centroid_aware_prior(self, num_samples: int) -> torch.Tensor:
        if not hasattr(self._ctx['model'], 'centroids_tens'):
            return torch.randn(num_samples, self.model.latent_dim, device=self.device) * 0.5
        idx = torch.randint(0, len(self._ctx['model'].centroids_tens), (num_samples,), device=self.device)
        mu = self._ctx['model'].centroids_tens[idx]
        eps = torch.randn_like(mu) * 0.3
        if hasattr(self._ctx['model'], 'G_inv'):
            G_inv = self._ctx['model'].G_inv(mu)
            try:
                L = torch.linalg.cholesky(G_inv)
                eps_m = torch.einsum('bij,bj->bi', L, eps)
            except Exception:
                evals, evecs = torch.linalg.eigh(G_inv)
                evals = torch.clamp(evals, min=1e-8)
                sqrtGinv = evecs @ torch.diag_embed(torch.sqrt(evals)) @ evecs.transpose(-2, -1)
                eps_m = torch.einsum('bij,bj->bi', sqrtGinv, eps)
            return mu + eps_m
        return mu + eps

    @torch.no_grad()
    def sample_weighted_mixture_prior(self, num_samples: int) -> torch.Tensor:
        if not hasattr(self._ctx['model'], 'centroids_tens'):
            return torch.randn(num_samples, self.model.latent_dim, device=self.device) * 0.5
        norms = torch.norm(self._ctx['model'].centroids_tens, dim=-1)
        w = torch.exp(-norms)
        w = w / (w.sum() + 1e-9)
        idx = torch.multinomial(w, num_samples, replacement=True)
        mu = self._ctx['model'].centroids_tens[idx]
        eps = torch.randn_like(mu) * 0.2
        if hasattr(self._ctx['model'], 'G_inv'):
            G_inv = self._ctx['model'].G_inv(mu)
            try:
                L = torch.linalg.cholesky(G_inv)
                eps_m = torch.einsum('bij,bj->bi', L, eps)
            except Exception:
                evals, evecs = torch.linalg.eigh(G_inv)
                evals = torch.clamp(evals, min=1e-8)
                sqrtGinv = evecs @ torch.diag_embed(torch.sqrt(evals)) @ evecs.transpose(-2, -1)
                eps_m = torch.einsum('bij,bj->bi', sqrtGinv, eps)
            return mu + eps_m
        return mu + eps

    @torch.no_grad()
    def sample_basic_prior(self, num_samples: int) -> torch.Tensor:
        z = torch.randn(num_samples, self.model.latent_dim, device=self.device) * 0.5
        if not hasattr(self.model, 'G_inv'):
            return z
        steps = 10
        for s in range(steps):
            z_t = z.clone().requires_grad_(True)
            G_inv = self.model.G_inv(z_t)
            det = torch.clamp(torch.linalg.det(G_inv), min=1e-10)
            log_det = torch.log(det)
            log_prob = 0.5 * log_det - 0.5 * torch.norm(z_t, dim=1) ** 2
            g = torch.autograd.grad(log_prob.sum(), z_t, create_graph=False)[0]
            step = 0.01 * (1.0 - s / steps)
            z = z + step * g.detach()
        return z.detach()
