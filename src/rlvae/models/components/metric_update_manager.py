"""
Metric Update Manager
=====================

Performs periodic K-means based updates of the centroid locations and inverse
metric matrices, adapted for the modular MetricTensor.
"""

from typing import Optional
import torch
import torch.nn as nn


class MetricUpdateManager(nn.Module):
    def __init__(
        self,
        metric_tensor: nn.Module,
        frequency: int = 100,
        regularization: Optional[float] = None,
        temperature: Optional[float] = None,
        min_cluster_size: int = 5,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.metric = metric_tensor
        self.frequency = max(1, int(frequency))
        self.regularization = regularization
        self.temperature = temperature
        self.min_cluster_size = max(1, int(min_cluster_size))
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._mu_buffer = []
        self._step = 0

    def collect(self, mu_batch: torch.Tensor):
        """Collect encoder means for clustering."""
        if mu_batch is None:
            return
        self._mu_buffer.append(mu_batch.detach().to(self.device).float())
        self._step += 1

    def ready(self) -> bool:
        if not getattr(self.metric, 'is_loaded', lambda: False)():
            return False
        if len(self._mu_buffer) == 0:
            return False
        return self._step % self.frequency == 0

    @torch.no_grad()
    def update(self):
        """Run K-means on collected μ and update centroids and inverse metric matrices."""
        if not self._mu_buffer:
            return False
        try:
            from sklearn.cluster import KMeans
        except Exception:
            # If sklearn not available, skip update gracefully
            return False

        mu_data = torch.cat(self._mu_buffer, dim=0)  # [N, D]
        self._mu_buffer = []

        # Determine number of centroids from current metric tensor
        centroids_current = getattr(self.metric, 'centroids', None)
        if centroids_current is None or centroids_current.numel() == 0:
            return False
        n_centroids = centroids_current.shape[0]
        d = centroids_current.shape[1]

        # K-means in numpy for compatibility
        mu_np = mu_data.cpu().numpy()
        kmeans = KMeans(n_clusters=n_centroids, random_state=42, n_init=10)
        kmeans.fit(mu_np)
        new_centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=self.device)
        labels = torch.tensor(kmeans.labels_, dtype=torch.long, device=self.device)

        # Compute per-cluster covariance and invert (with regularization)
        inv_mats = []
        for k in range(n_centroids):
            idx = (labels == k).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() < self.min_cluster_size:
                # Fallback to identity
                inv_mats.append(torch.eye(d, device=self.device, dtype=torch.float32))
                continue
            pts = mu_data[idx]  # [Nk, D]
            mean = pts.mean(dim=0, keepdim=True)
            diffs = pts - mean
            cov = (diffs.t() @ diffs) / max(1, (pts.shape[0] - 1))
            reg = self.regularization
            if reg is None:
                # Use metric's own regularization if available
                reg = float(getattr(self.metric, 'regularization', torch.tensor(0.01)).item())
            cov = cov + float(reg) * torch.eye(d, device=self.device)
            try:
                inv_cov = torch.linalg.inv(cov)
            except Exception:
                inv_cov = torch.eye(d, device=self.device)
            inv_mats.append(inv_cov)

        inv_mats = torch.stack(inv_mats, dim=0)  # [K, D, D]

        # Update metric tensor buffers
        temp = self.temperature
        if temp is None:
            temp = float(getattr(self.metric, 'temperature', torch.tensor(0.5)).item())
        reg_val = self.regularization
        if reg_val is None:
            reg_val = float(getattr(self.metric, 'regularization', torch.tensor(0.01)).item())

        try:
            # Reuse the loader path to validate shapes and register buffers correctly
            self.metric.load_pretrained(new_centroids, inv_mats, temperature=temp, regularization=reg_val)
        except Exception:
            # Direct buffer set if load_pretrained is unavailable
            if hasattr(self.metric, 'register_buffer'):
                self.metric.register_buffer('centroids', new_centroids)
                self.metric.register_buffer('metric_matrices', inv_mats)
                if hasattr(self.metric, 'temperature'):
                    self.metric.register_buffer('temperature', torch.tensor(temp, device=self.device))
                if hasattr(self.metric, 'regularization'):
                    self.metric.register_buffer('regularization', torch.tensor(reg_val, device=self.device))
        return True

