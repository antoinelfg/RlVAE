"""
Regularizers and Centroid EMA (modular)
======================================

Ports regularization utilities from the monolithic model into modular, reusable
components:

- Centroid regularizer (Phase 1): encourages μ(x₀) near centroids using G(cₖ)
- Spectral penalty (Phase 2): clamps eigenvalues of G(z) to [λ_min, λ_max]
- Smoothness penalty (Phase 2): penalizes ||∇ₙ G(z)||² via random JVPs
- Anisotropy alignment (Phase 2): aligns mean G(μ) to empirical Σ̂ / α
- Centroid EMA updater: soft-responsibility EMA update of centroids
"""

from typing import Dict, Optional
import torch
import torch.nn as nn


class CentroidEMAUpdater(nn.Module):
    def __init__(self, rate: float = 0.01, update_frequency: int = 10):
        super().__init__()
        self.rate = float(rate)
        self.update_frequency = int(update_frequency)

    @torch.no_grad()
    def update(self, model, mu_batch: torch.Tensor, step: int) -> None:
        """Update centroids with soft-responsibility EMA.

        Args:
            model: object exposing `centroids_tens` and callable `G(z)`
            mu_batch: [B, D]
            step: global training step
        """
        if step % self.update_frequency != 0:
            return
        centroids = getattr(model, 'centroids_tens', None)
        if centroids is None:
            return
        device = mu_batch.device
        B = mu_batch.shape[0]
        K = centroids.shape[0]
        # Responsibilities per-sample
        responsibilities = torch.zeros(B, K, device=device)
        for i in range(B):
            mu_i = mu_batch[i:i+1]
            dists_neg = []
            for k in range(K):
                c_k = centroids[k:k+1]
                G_ck = model.G(c_k)
                diff = (mu_i - c_k).unsqueeze(-1)
                dist_sq = torch.matmul(torch.matmul(diff.transpose(-2, -1), G_ck), diff).squeeze()
                dists_neg.append(-dist_sq)
            responsibilities[i] = torch.softmax(torch.stack(dists_neg), dim=0)
        # EMA update
        for k in range(K):
            w = responsibilities[:, k].unsqueeze(-1)
            if w.sum() > 1e-6:
                weighted_mu = (w * mu_batch).sum(dim=0) / w.sum()
                centroids[k] = (1 - self.rate) * centroids[k] + self.rate * weighted_mu


class RegularizerManager(nn.Module):
    def __init__(
        self,
        model,
        centroid_enabled: bool = False,
        centroid_weight: float = 0.01,
        centroid_t0_only: bool = True,
        spectral_enabled: bool = False,
        spectral_weight: float = 0.1,
        eigenval_min_bound: float = 1e-2,
        eigenval_max_bound: float = 1e2,
        smoothness_enabled: bool = False,
        smoothness_weight: float = 0.01,
        anisotropy_enabled: bool = False,
        anisotropy_weight: float = 0.05,
        ema_enabled: bool = False,
        ema_rate: float = 0.01,
        ema_update_frequency: int = 10,
    ):
        super().__init__()
        import weakref
        # Store weak proxy inside a plain dict to avoid module registration cycles
        self._ctx = {'model': weakref.proxy(model)}
        # Centroid reg
        self.centroid_enabled = bool(centroid_enabled)
        self.centroid_weight = float(centroid_weight)
        self.centroid_t0_only = bool(centroid_t0_only)
        # Spectral
        self.spectral_enabled = bool(spectral_enabled)
        self.spectral_weight = float(spectral_weight)
        self.eig_min = float(eigenval_min_bound)
        self.eig_max = float(eigenval_max_bound)
        # Smoothness
        self.smoothness_enabled = bool(smoothness_enabled)
        self.smoothness_weight = float(smoothness_weight)
        # Anisotropy
        self.anisotropy_enabled = bool(anisotropy_enabled)
        self.anisotropy_weight = float(anisotropy_weight)
        # EMA
        self.ema_enabled = bool(ema_enabled)
        self.ema_updater = CentroidEMAUpdater(rate=ema_rate, update_frequency=ema_update_frequency)

        self._global_step = 0

    def step(self):
        self._global_step += 1

    def compute_centroid_regularizer(self, mu: torch.Tensor, t: int = 0) -> torch.Tensor:
        if not self.centroid_enabled:
            return torch.tensor(0.0, device=mu.device)
        if self.centroid_t0_only and t != 0:
            return torch.tensor(0.0, device=mu.device)
        centroids = getattr(self._ctx['model'], 'centroids_tens', None)
        if centroids is None:
            return torch.tensor(0.0, device=mu.device)
        B = mu.shape[0]
        K = centroids.shape[0]
        min_dists = []
        for i in range(B):
            mu_i = mu[i:i+1]
            dists = []
            for k in range(K):
                c_k = centroids[k:k+1]
                G_ck = self._ctx['model'].G(c_k)
                diff = (mu_i - c_k).unsqueeze(-1)
                dist_sq = torch.matmul(torch.matmul(diff.transpose(-2, -1), G_ck), diff).squeeze()
                dists.append(dist_sq)
            min_dists.append(torch.min(torch.stack(dists)))
        loss = torch.stack(min_dists).mean()
        return self.centroid_weight * loss

    def compute_spectral_penalty(self, z: torch.Tensor) -> torch.Tensor:
        if not self.spectral_enabled:
            return torch.tensor(0.0, device=z.device)
        G = self._ctx['model'].G(z)
        eig = torch.linalg.eigvals(G).real
        lower = torch.relu(self.eig_min - eig)
        upper = torch.relu(eig - self.eig_max)
        return self.spectral_weight * (torch.sum(lower ** 2) + torch.sum(upper ** 2))

    def compute_smoothness_penalty(self, z: torch.Tensor) -> torch.Tensor:
        if not self.smoothness_enabled:
            return torch.tensor(0.0, device=z.device)
        z = z.detach().requires_grad_(True)
        G = self._ctx['model'].G(z)
        B, D = z.shape
        jvp_norm = 0.0
        n_proj = min(5, D)
        for _ in range(n_proj):
            v = torch.randn_like(G)
            jvp = torch.autograd.grad(G, z, v, create_graph=True, retain_graph=True, only_inputs=True)[0]
            jvp_norm += torch.sum(jvp ** 2)
        return self.smoothness_weight * jvp_norm / n_proj

    def compute_anisotropy_alignment(self, mu: torch.Tensor) -> torch.Tensor:
        if not self.anisotropy_enabled:
            return torch.tensor(0.0, device=mu.device)
        B, D = mu.shape
        if B < 2:
            return torch.tensor(0.0, device=mu.device)
        mu_c = mu - mu.mean(dim=0, keepdim=True)
        emp_cov = (mu_c.T @ mu_c) / (B - 1)
        # Obtain current alpha from model if available
        get_alpha = getattr(self._ctx['model'], 'get_current_posterior_alpha', None)
        alpha = float(get_alpha(self._ctx['model']._current_epoch) if callable(get_alpha) else getattr(self._ctx['model'], 'posterior_local_alpha', 0.5))
        target = emp_cov / alpha
        G_mu = self._ctx['model'].G(mu).mean(dim=0)
        diff = G_mu - target
        return self.anisotropy_weight * torch.sum(diff ** 2)

    def maybe_update_centroids_ema(self, mu: torch.Tensor):
        if self.ema_enabled:
            self.ema_updater.update(self.model, mu, self._global_step)

    def compute_all(self, mu: torch.Tensor, z_for_spectral: Optional[torch.Tensor] = None, t: int = 0) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        out['centroid_regularizer'] = self.compute_centroid_regularizer(mu, t=t)
        if z_for_spectral is None:
            z_for_spectral = mu
        out['spectral_penalty'] = self.compute_spectral_penalty(z_for_spectral)
        out['smoothness_penalty'] = self.compute_smoothness_penalty(z_for_spectral)
        out['anisotropy_penalty'] = self.compute_anisotropy_alignment(mu)
        return out
