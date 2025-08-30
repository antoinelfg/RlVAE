import torch


def check_centroid_indexing(num_points=32, latent_dim=5, num_centroids=7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_mus = torch.randn(num_points, latent_dim, device=device)
    centroids_idx = torch.randint(0, num_points, (num_centroids,), device=torch.device("cpu"))
    centroids_mu = all_mus[centroids_idx.to(all_mus.device)]
    assert centroids_mu.shape == (num_centroids, latent_dim)
    assert centroids_mu.device == all_mus.device


def check_distance_computation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_mus = torch.randn(16, 4, device=device)
    centroids_mu = torch.randn(5, 4, device=device)
    c = centroids_mu[0]
    dists = torch.norm(all_mus - c, dim=1)
    assert dists.shape == (16,)
    assert dists.device == all_mus.device


if __name__ == "__main__":
    check_centroid_indexing()
    check_distance_computation()
    print("Stage B device-consistency self-check: OK")








