import torch

# Path to your metric file (change if needed)
metric_path = "data/pretrained/metric_diverse_mlp_ld2_20250724_150207.pt"

# Load the metric file (set weights_only=False for PyTorch >=2.6 compatibility)
state = torch.load(metric_path, map_location="cpu", weights_only=False)

print("=== Metric File Inspection ===")
print("Keys in metric file:", list(state.keys()))

# Extract centroids
centroids = state.get("centroids")
if centroids is None:
    centroids = state.get("metric_centroids")
if centroids is not None:
    print(f"Centroids shape: {centroids.shape}")
    print(f"First centroid: {centroids[0]}")
else:
    print("No centroids found!")

# Extract metric matrices
metric_matrices = state.get("metric_matrices")
if metric_matrices is None:
    metric_matrices = state.get("M_matrices")
if metric_matrices is not None:
    print(f"Metric matrices shape: {metric_matrices.shape}")
    print(f"First metric matrix:\n{metric_matrices[0]}")
else:
    print("No metric matrices found!")

# Print temperature and regularization
print("Temperature:", state.get("temperature"))
print("Regularization:", state.get("regularization"))

# Inspect a few metric matrices
if metric_matrices is not None:
    for i in range(min(3, metric_matrices.shape[0])):
        print(f"\nMetric matrix {i}:")
        print(metric_matrices[i])
        is_identity = torch.allclose(metric_matrices[i], torch.eye(metric_matrices.shape[1]), atol=1e-2)
        print("Is close to identity?", is_identity)
        eigvals = torch.linalg.eigvalsh(metric_matrices[i])
        print("Eigenvalues:", eigvals) 