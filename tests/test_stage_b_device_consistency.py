import torch
import pytest


@pytest.mark.parametrize("num_points,latent_dim,num_centroids", [(32, 5, 7), (10, 3, 3)])
def test_centroid_indexing_across_devices(num_points, latent_dim, num_centroids):
    """
    Ensure centroid indices (on CPU) index latent means on the active device
    without raising device mismatch errors, and that the result stays on the
    same device as the source tensor.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_mus = torch.randn(num_points, latent_dim, device=device)
    centroids_idx = torch.randint(0, num_points, (num_centroids,), device=torch.device("cpu"))

    # This mirrors scripts/train_diverse_metric_vae.py logic
    centroids_mu = all_mus[centroids_idx.to(all_mus.device)]

    assert centroids_mu.shape == (num_centroids, latent_dim)
    assert centroids_mu.device == all_mus.device


def test_distance_computation_same_device():
    """
    Mirror the Stage B inner loop: dists = torch.norm(all_mus - c, dim=1)
    for a centroid vector c. This must not trigger device mismatch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_mus = torch.randn(16, 4, device=device)
    centroids_mu = torch.randn(5, 4, device=device)

    # Emulate fixed code path: centroids_mu = centroids_mu.to(all_mus.device)
    centroids_mu = centroids_mu.to(all_mus.device)

    # Pick a centroid and compute distances
    c = centroids_mu[0]
    dists = torch.norm(all_mus - c, dim=1)

    assert dists.shape == (16,)
    assert dists.device == all_mus.device








