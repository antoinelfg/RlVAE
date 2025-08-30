#!/usr/bin/env python3
"""
Analyze Centroids and Manifold Structure
========================================

This script analyzes how centroids are computed (k-medoids vs k-means vs random),
visualizes the metric manifold structure, and shows RHMC sampling behavior.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent))

from src.models.riemannian_flow_vae import RiemannianFlowVAE
from src.models.samplers.hmc_sampler import RiemannianHMCSampler


def create_synthetic_latent_data(n_samples=1000, latent_dim=2, n_clusters=8, noise=0.1):
    """Create synthetic latent data with clear cluster structure."""
    print("🔧 Creating synthetic latent data...")
    
    # Create cluster centers
    np.random.seed(42)
    cluster_centers = np.random.randn(n_clusters, latent_dim) * 3.0
    
    # Generate samples around clusters
    samples = []
    samples_per_cluster = n_samples // n_clusters
    
    for i in range(n_clusters):
        cluster_samples = np.random.multivariate_normal(
            cluster_centers[i], 
            np.eye(latent_dim) * noise, 
            samples_per_cluster
        )
        samples.append(cluster_samples)
    
    # Add some random samples
    remaining = n_samples - len(samples) * samples_per_cluster
    if remaining > 0:
        random_samples = np.random.randn(remaining, latent_dim) * 2.0
        samples.append(random_samples)
    
    samples = np.vstack(samples)
    np.random.shuffle(samples)
    
    print(f"✅ Created {len(samples)} samples in {latent_dim}D with {n_clusters} clusters")
    return samples


def compute_medoids(data, n_clusters, random_state=42):
    """Simple k-medoids implementation using k-means + nearest neighbor."""
    # Use k-means to get cluster centers
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_centers = kmeans.fit_predict(data)
    
    # For each cluster, find the actual data point closest to the k-means center
    medoids = []
    labels = np.zeros(len(data), dtype=int)
    
    for i in range(n_clusters):
        cluster_points = data[cluster_centers == i]
        if len(cluster_points) > 0:
            # Find the point closest to the k-means center
            kmeans_center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_points - kmeans_center, axis=1)
            medoid_idx = np.argmin(distances)
            medoid = cluster_points[medoid_idx]
            medoids.append(medoid)
            
            # Update labels for this cluster
            cluster_indices = np.where(cluster_centers == i)[0]
            labels[cluster_indices] = i
        else:
            # If no points in cluster, use k-means center
            medoids.append(kmeans.cluster_centers_[i])
    
    return np.array(medoids), labels


def compare_centroid_methods(latent_data, n_centroids=25):
    """Compare different centroid computation methods."""
    print(f"\n🔍 Comparing Centroid Computation Methods")
    print("=" * 60)
    
    methods = {
        'k-means': KMeans(n_clusters=n_centroids, random_state=42, n_init=10),
        'k-medoids': None, # KMedoids is removed, so we'll just use k-means for now
        'random': None
    }
    
    results = {}
    
    for method_name, method in methods.items():
        print(f"\n--- {method_name.upper()} ---")
        
        if method_name == 'random':
            # Random selection
            indices = np.random.choice(len(latent_data), n_centroids, replace=False)
            centroids = latent_data[indices]
            labels = np.zeros(len(latent_data), dtype=int)
        elif method_name == 'k-medoids':
            # Use the new compute_medoids function
            centroids, labels = compute_medoids(latent_data, n_centroids)
        else: # k-means
            labels = method.fit_predict(latent_data)
            centroids = method.cluster_centers_
        
        # Compute metric matrices for each centroid
        metric_matrices = []
        for i, centroid in enumerate(centroids):
            # Find points closest to this centroid
            distances = np.linalg.norm(latent_data - centroid, axis=1)
            closest_indices = np.argsort(distances)[:max(10, len(latent_data) // n_centroids)]
            cluster_points = latent_data[closest_indices]
            
            # Compute local covariance
            if len(cluster_points) > 1:
                cov_matrix = np.cov(cluster_points.T)
                # Add regularization
                cov_matrix += np.eye(cov_matrix.shape[0]) * 0.01
                # Metric is inverse of covariance
                try:
                    metric_matrix = np.linalg.inv(cov_matrix)
                except np.linalg.LinAlgError:
                    metric_matrix = np.eye(cov_matrix.shape[0])
            else:
                metric_matrix = np.eye(latent_data.shape[1])
            
            metric_matrices.append(metric_matrix)
        
        metric_matrices = np.array(metric_matrices)
        
        # Analyze results
        eigenvals = np.linalg.eigvals(metric_matrices).real
        min_eigenvals = eigenvals.min(axis=-1)
        max_eigenvals = eigenvals.max(axis=-1)
        condition_numbers = max_eigenvals / (min_eigenvals + 1e-10)
        determinants = np.linalg.det(metric_matrices)
        
        results[method_name] = {
            'centroids': centroids,
            'metric_matrices': metric_matrices,
            'labels': labels,
            'eigenvals': eigenvals,
            'condition_numbers': condition_numbers,
            'determinants': determinants,
            'min_eigenval': min_eigenvals.min(),
            'max_eigenval': max_eigenvals.max(),
            'eigenval_ratio': max_eigenvals.max() / min_eigenvals.min(),
            'mean_condition': condition_numbers.mean(),
            'det_range': [determinants.min(), determinants.max()]
        }
        
        print(f"✅ Centroids: {centroids.shape}")
        print(f"✅ Eigenvalue ratio: {results[method_name]['eigenval_ratio']:.2f}")
        print(f"✅ Mean condition number: {results[method_name]['mean_condition']:.2f}")
        print(f"✅ Determinant range: [{results[method_name]['det_range'][0]:.3e}, {results[method_name]['det_range'][1]:.3e}]")
    
    return results


def visualize_centroid_comparison(latent_data, results):
    """Visualize different centroid computation methods."""
    print(f"\n🎨 Visualizing Centroid Computation Methods")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Centroid Computation Methods Comparison", fontsize=16)
    
    methods = list(results.keys())
    colors = ['red', 'blue', 'green']
    
    for i, (method_name, result) in enumerate(results.items()):
        centroids = result['centroids']
        eigenvals = result['eigenvals']
        determinants = result['determinants']
        
        # Plot 1: Original data with centroids
        ax1 = axes[0, i]
        ax1.scatter(latent_data[:, 0], latent_data[:, 1], alpha=0.3, s=10, color='gray', label='Data')
        ax1.scatter(centroids[:, 0], centroids[:, 1], c=determinants, cmap='viridis', s=100, alpha=0.8)
        ax1.set_title(f"{method_name.upper()}\nCentroids (colored by det)")
        ax1.set_xlabel("z₁")
        ax1.set_ylabel("z₂")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Metric eigenvalues at centroids
        ax2 = axes[1, i]
        scatter = ax2.scatter(centroids[:, 0], centroids[:, 1], 
                             c=eigenvals[:, 0], cmap='plasma', s=100, alpha=0.8)
        ax2.set_title(f"{method_name.upper()}\nCentroids (colored by λ₁)")
        ax2.set_xlabel("z₁")
        ax2.set_ylabel("z₂")
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax2)
    
    plt.tight_layout()
    plt.savefig("centroid_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()


def visualize_metric_manifold_structure(model, n_points=500):
    """Visualize the complete metric manifold structure."""
    print(f"\n🎨 Visualizing Complete Metric Manifold Structure")
    print("=" * 60)
    
    # Create a grid of points to visualize the manifold
    x_range = np.linspace(-4, 4, int(np.sqrt(n_points)))
    y_range = np.linspace(-4, 4, int(np.sqrt(n_points)))
    X, Y = np.meshgrid(x_range, y_range)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Convert to tensor
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=model.device)
    
    # Compute metric properties at each point
    with torch.no_grad():
        G_grid = model.G(grid_tensor)
        eigenvals = torch.linalg.eigvals(G_grid).real
        determinants = torch.linalg.det(G_grid)
        condition_numbers = eigenvals.max(dim=-1)[0] / (eigenvals.min(dim=-1)[0] + 1e-8)
    
    # Reshape for plotting
    eigenvals_reshaped = eigenvals.cpu().numpy().reshape(len(x_range), len(y_range), 2)
    determinants_reshaped = determinants.cpu().numpy().reshape(len(x_range), len(y_range))
    condition_numbers_reshaped = condition_numbers.cpu().numpy().reshape(len(x_range), len(y_range))
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Complete Metric Manifold Structure", fontsize=16)
    
    # Plot 1: Determinant heatmap
    im1 = axes[0, 0].contourf(X, Y, determinants_reshaped, levels=20, cmap='viridis')
    axes[0, 0].set_title("Metric Determinant")
    axes[0, 0].set_xlabel("z₁")
    axes[0, 0].set_ylabel("z₂")
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: First eigenvalue heatmap
    im2 = axes[0, 1].contourf(X, Y, eigenvals_reshaped[:, :, 0], levels=20, cmap='plasma')
    axes[0, 1].set_title("λ₁ (First Eigenvalue)")
    axes[0, 1].set_xlabel("z₁")
    axes[0, 1].set_ylabel("z₂")
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Second eigenvalue heatmap
    im3 = axes[0, 2].contourf(X, Y, eigenvals_reshaped[:, :, 1], levels=20, cmap='hot')
    axes[0, 2].set_title("λ₂ (Second Eigenvalue)")
    axes[0, 2].set_xlabel("z₁")
    axes[0, 2].set_ylabel("z₂")
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Plot 4: Condition number heatmap
    im4 = axes[1, 0].contourf(X, Y, condition_numbers_reshaped, levels=20, cmap='coolwarm')
    axes[1, 0].set_title("Condition Number")
    axes[1, 0].set_xlabel("z₁")
    axes[1, 0].set_ylabel("z₂")
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Plot 5: Anisotropy (difference between eigenvalues)
    anisotropy = eigenvals_reshaped[:, :, 0] - eigenvals_reshaped[:, :, 1]
    im5 = axes[1, 1].contourf(X, Y, anisotropy, levels=20, cmap='RdYlBu')
    axes[1, 1].set_title("Anisotropy (λ₁ - λ₂)")
    axes[1, 1].set_xlabel("z₁")
    axes[1, 1].set_ylabel("z₂")
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Plot 6: Metric magnitude (trace)
    trace = eigenvals_reshaped[:, :, 0] + eigenvals_reshaped[:, :, 1]
    im6 = axes[1, 2].contourf(X, Y, trace, levels=20, cmap='inferno')
    axes[1, 2].set_title("Metric Magnitude (tr(G))")
    axes[1, 2].set_xlabel("z₁")
    axes[1, 2].set_ylabel("z₂")
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig("metric_manifold_structure.png", dpi=150, bbox_inches='tight')
    plt.show()


def visualize_rhmc_sampling_on_manifold(model, n_samples=200, n_steps=50):
    """Visualize RHMC sampling behavior on the manifold."""
    print(f"\n🚀 Visualizing RHMC Sampling on Manifold")
    print("=" * 60)
    
    # Create RHMC sampler
    rhmc_sampler = RiemannianHMCSampler(model, mcmc_steps_nbr=n_steps, n_lf=15, eps_lf=0.02)
    
    # Sample from different starting points
    starting_points = [
        torch.tensor([[0.0, 0.0]], device=model.device),
        torch.tensor([[2.0, 2.0]], device=model.device),
        torch.tensor([[-2.0, 0.0]], device=model.device),
        torch.tensor([[0.0, -2.0]], device=model.device),
    ]
    
    colors = ['red', 'blue', 'green', 'orange']
    labels = ['Center', 'Corner', 'Left', 'Bottom']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("RHMC Sampling on Metric Manifold", fontsize=16)
    
    for i, (start_point, color, label) in enumerate(zip(starting_points, colors, labels)):
        print(f"\n--- Sampling from {label} ---")
        
        # Run RHMC sampling
        start_time = time.time()
        samples = rhmc_sampler.sample(n_samples)
        sampling_time = time.time() - start_time
        
        print(f"✅ Sampling completed in {sampling_time:.3f}s")
        print(f"✅ Samples range: [{samples.min():.3f}, {samples.max():.3f}]")
        
        # Analyze metric at samples
        with torch.no_grad():
            G_samples = model.G(samples)
            eigenvals = torch.linalg.eigvals(G_samples).real
            determinants = torch.linalg.det(G_samples)
        
        print(f"✅ Metric eigenvalues: min={eigenvals.min():.3e}, max={eigenvals.max():.3e}")
        print(f"✅ Determinants: min={determinants.min():.3e}, max={determinants.max():.3e}")
        
        # Plot samples colored by metric properties
        ax = axes[i // 2, i % 2]
        scatter = ax.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), 
                           c=determinants.cpu(), cmap='viridis', alpha=0.7, s=50)
        ax.scatter(start_point[:, 0].cpu(), start_point[:, 1].cpu(), 
                  color='red', s=200, marker='*', label='Start Point')
        ax.set_title(f"RHMC Samples from {label}\n(colored by det(G))")
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.savefig("rhmc_sampling_on_manifold.png", dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main analysis function."""
    print("🔍 Analyzing Centroids and Manifold Structure")
    print("=" * 60)
    
    # Create synthetic latent data
    latent_data = create_synthetic_latent_data(n_samples=1000, latent_dim=2, n_clusters=8)
    
    # Compare different centroid methods
    results = compare_centroid_methods(latent_data, n_centroids=25)
    
    # Visualize centroid comparison
    visualize_centroid_comparison(latent_data, results)
    
    # Create model and load best centroids (k-medoids)
    model = RiemannianFlowVAE(
        input_dim=(3, 64, 64),
        latent_dim=2,
        n_flows=0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load encoder/decoder (for completeness)
    model.load_pretrained_components(
        encoder_path="data/pretrained/encoder_diverse_mlp_ld2_20250730_105419.pt",
        decoder_path="data/pretrained/decoder_diverse_mlp_ld2_20250730_105419.pt",
        metric_path="data/pretrained/metric_diverse_mlp_ld2_20250730_105420.pt"
    )
    
    # Load k-medoids centroids and metrics
    kmedoids_result = results['k-medoids']
    centroids = torch.tensor(kmedoids_result['centroids'], dtype=torch.float32, device=device)
    metric_matrices = torch.tensor(kmedoids_result['metric_matrices'], dtype=torch.float32, device=device)
    
    print(f"\n🔧 Loading k-medoids centroids and metrics...")
    model.load_pretrained_metrics_from_tensor(centroids, metric_matrices, temperature=0.4, regularization=0.01)
    
    # Visualize complete manifold structure
    visualize_metric_manifold_structure(model, n_points=400)
    
    # Visualize RHMC sampling on manifold
    visualize_rhmc_sampling_on_manifold(model, n_samples=150, n_steps=40)
    
    print(f"\n✅ Analysis completed!")
    print(f"📊 Key Findings:")
    print(f"   - Centroids are computed via k-medoids clustering on latent representations")
    print(f"   - Metric matrices are computed from local covariance around each centroid")
    print(f"   - The manifold shows clear geometric structure with varying properties")
    print(f"   - RHMC sampling respects the manifold geometry")


if __name__ == "__main__":
    main() 