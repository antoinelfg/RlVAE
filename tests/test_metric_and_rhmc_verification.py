import os
from pathlib import Path
import torch


def test_metric_file_loadable(tmp_path: Path):
    # Create a tiny fake metric file in the expected format
    centroids = torch.randn(5, 4)
    M = torch.stack([torch.eye(4) for _ in range(5)], dim=0)
    data = {
        "centroids": centroids,
        "M_matrices": M,
        "temperature": torch.tensor(0.5),
        "regularization": torch.tensor(0.01),
    }
    path = tmp_path / "metric_fake.pt"
    torch.save(data, path)

    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert "centroids" in blob and "M_matrices" in blob
    assert blob["centroids"].shape == (5, 4)
    assert blob["M_matrices"].shape == (5, 4, 4)


def test_determinant_positive_definite():
    # Simple PD check for constructed G_inv
    centroids = torch.randn(3, 2)
    M = torch.stack([torch.eye(2) for _ in range(3)], dim=0)
    temperature = torch.tensor(0.5)
    lbd = torch.tensor(0.01)

    class _Stub:
        def __init__(self):
            self.latent_dim = 2
            self.centroids_tens = centroids
            self.M_tens = M
            self.temperature = temperature
            self.lbd = lbd

        def G_inv(self, z):
            diff = self.centroids_tens.unsqueeze(0) - z.unsqueeze(1)
            weights = torch.exp(-torch.norm(diff, dim=-1) ** 2 / (self.temperature ** 2))
            weighted_M = self.M_tens.unsqueeze(0) * weights.unsqueeze(-1).unsqueeze(-1)
            return weighted_M.sum(dim=1) + self.lbd * torch.eye(self.latent_dim)

    stub = _Stub()
    z = torch.randn(5, 2)
    Ginv = stub.G_inv(z)
    dets = torch.linalg.det(Ginv)
    assert torch.all(dets > 0)

