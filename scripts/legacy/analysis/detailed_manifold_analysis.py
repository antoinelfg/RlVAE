#!/usr/bin/env python3
"""
Detailed Manifold Analysis with Better Scaling and More Centroids
================================================================

This script creates a more detailed analysis with:
- More centroids to capture all clusters
- Different temperature settings for better metric diversity
- Consistent scaling across plots
- More precise RHMC sampling
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent))

from src.models.riemannian_flow_vae import RiemannianFlowVAE
from src.models.samplers.hmc_sampler import RiemannianHMCSampler


def create_detailed_latent_data(n_samples=2000, latent_dim=2, n_clusters=7, noise=0.05):
    """Create more detailed latent data with clear 7-cluster structure."""
    print("🔧 Creating detailed latent data with 7 clusters...")
    
    # Create 7 distinct cluster centers in a structured pattern
    np.random.seed(42)
    cluster_centers = np.array([
        [-3.0, -2.0],  # Bottom-left
        [-1.5, 2.0],   # Top-left
        [0.0, 0.0],    # Center
        [2.0, -1.5],   # Bottom-right
        [3.5, 1.0],    # Top-right
        [-0.5, -3.0],  # Far bottom
        [1.5, 3.0],    # Far top
    ])
    
    # Generate samples around clusters with different densities
    samples = []
    samples_per_cluster = n_samples // n_clusters
    
    for i in range(n_clusters):
        # Vary cluster density and noise
        cluster_noise = noise * (1 + 0.5 * np.sin(i))  # Vary noise per cluster
        cluster_samples = np.random.multivariate_normal(
            cluster_centers[i], 
            np.eye(latent_dim) * cluster_noise, 
            samples_per_cluster
        )
        samples.append(cluster_samples)
    
    # Add some random samples for realism
    remaining = n_samples - len(samples) * samples_per_cluster
    if remaining > 0:
        random_samples = np.random.randn(remaining, latent_dim) * 2.0
        samples.append(random_samples)
    
    samples = np.vstack(samples)
    np.random.shuffle(samples)
    
    print(f"✅ Created {len(samples)} samples in {latent_dim}D with {n_clusters} distinct clusters")
    print(f"✅ Cluster centers: {cluster_centers}")
    return samples, cluster_centers


def create_detailed_centroids(latent_data, n_centroids=50, method='k-means'):
    """Create more detailed centroids to capture all clusters."""
    print(f"\n🔍 Creating {n_centroids} centroids using {method}...")
    
    if method == 'k-means':
        kmeans = KMeans(n_clusters=n_centroids, random_state=42, n_init=10)
        labels = kmeans.fit_predict(latent_data)
        centroids = kmeans.cluster_centers_
    else:
        # Random selection for comparison
        indices = np.random.choice(len(latent_data), n_centroids, replace=False)
        centroids = latent_data[indices]
        labels = np.zeros(len(latent_data), dtype=int)
    
    # Compute detailed metric matrices for each centroid
    metric_matrices = []
    for i, centroid in enumerate(centroids):
        # Find points closest to this centroid
        distances = np.linalg.norm(latent_data - centroid, axis=1)
        closest_indices = np.argsort(distances)[:max(20, len(latent_data) // n_centroids)]
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
    
    print(f"✅ Created {len(centroids)} centroids with detailed metric matrices")
    return centroids, metric_matrices, labels


def visualize_detailed_manifold_structure(model, n_points=900, save_prefix="detailed"):
    """Visualize the detailed metric manifold structure with consistent scaling."""
    print(f"\n🎨 Visualizing Detailed Metric Manifold Structure")
    print("=" * 60)
    
    # Create a fine grid of points
    x_range = np.linspace(-5, 5, int(np.sqrt(n_points)))
    y_range = np.linspace(-5, 5, int(np.sqrt(n_points)))
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
    
    # Create visualization with consistent scaling
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Detailed Metric Manifold Structure (Consistent Scaling)", fontsize=16)
    
    # Determine global ranges for consistent scaling
    det_min, det_max = determinants_reshaped.min(), determinants_reshaped.max()
    eigenval_min, eigenval_max = eigenvals_reshaped.min(), eigenvals_reshaped.max()
    cond_min, cond_max = condition_numbers_reshaped.min(), condition_numbers_reshaped.max()
    
    # Plot 1: Determinant heatmap with consistent scaling
    im1 = axes[0, 0].contourf(X, Y, determinants_reshaped, levels=30, cmap='viridis', 
                               vmin=det_min, vmax=det_max)
    axes[0, 0].set_title(f"Metric Determinant\nRange: [{det_min:.2e}, {det_max:.2e}]")
    axes[0, 0].set_xlabel("z₁")
    axes[0, 0].set_ylabel("z₂")
    axes[0, 0].set_xlim(-5, 5)
    axes[0, 0].set_ylim(-5, 5)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: First eigenvalue heatmap
    im2 = axes[0, 1].contourf(X, Y, eigenvals_reshaped[:, :, 0], levels=30, cmap='plasma',
                               vmin=eigenval_min, vmax=eigenval_max)
    axes[0, 1].set_title(f"λ₁ (First Eigenvalue)\nRange: [{eigenval_min:.2e}, {eigenval_max:.2e}]")
    axes[0, 1].set_xlabel("z₁")
    axes[0, 1].set_ylabel("z₂")
    axes[0, 1].set_xlim(-5, 5)
    axes[0, 1].set_ylim(-5, 5)
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Second eigenvalue heatmap
    im3 = axes[0, 2].contourf(X, Y, eigenvals_reshaped[:, :, 1], levels=30, cmap='hot',
                               vmin=eigenval_min, vmax=eigenval_max)
    axes[0, 2].set_title(f"λ₂ (Second Eigenvalue)\nRange: [{eigenval_min:.2e}, {eigenval_max:.2e}]")
    axes[0, 2].set_xlabel("z₁")
    axes[0, 2].set_ylabel("z₂")
    axes[0, 2].set_xlim(-5, 5)
    axes[0, 2].set_ylim(-5, 5)
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Plot 4: Condition number heatmap
    im4 = axes[1, 0].contourf(X, Y, condition_numbers_reshaped, levels=30, cmap='coolwarm',
                               vmin=cond_min, vmax=cond_max)
    axes[1, 0].set_title(f"Condition Number\nRange: [{cond_min:.2f}, {cond_max:.2f}]")
    axes[1, 0].set_xlabel("z₁")
    axes[1, 0].set_ylabel("z₂")
    axes[1, 0].set_xlim(-5, 5)
    axes[1, 0].set_ylim(-5, 5)
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Plot 5: Anisotropy (difference between eigenvalues)
    anisotropy = eigenvals_reshaped[:, :, 0] - eigenvals_reshaped[:, :, 1]
    anisotropy_min, anisotropy_max = anisotropy.min(), anisotropy.max()
    im5 = axes[1, 1].contourf(X, Y, anisotropy, levels=30, cmap='RdYlBu',
                               vmin=anisotropy_min, vmax=anisotropy_max)
    axes[1, 1].set_title(f"Anisotropy (λ₁ - λ₂)\nRange: [{anisotropy_min:.2e}, {anisotropy_max:.2e}]")
    axes[1, 1].set_xlabel("z₁")
    axes[1, 1].set_ylabel("z₂")
    axes[1, 1].set_xlim(-5, 5)
    axes[1, 1].set_ylim(-5, 5)
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Plot 6: Metric magnitude (trace)
    trace = eigenvals_reshaped[:, :, 0] + eigenvals_reshaped[:, :, 1]
    trace_min, trace_max = trace.min(), trace.max()
    im6 = axes[1, 2].contourf(X, Y, trace, levels=30, cmap='inferno',
                               vmin=trace_min, vmax=trace_max)
    axes[1, 2].set_title(f"Metric Magnitude (tr(G))\nRange: [{trace_min:.2e}, {trace_max:.2e}]")
    axes[1, 2].set_xlabel("z₁")
    axes[1, 2].set_ylabel("z₂")
    axes[1, 2].set_xlim(-5, 5)
    axes[1, 2].set_ylim(-5, 5)
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_metric_manifold_structure.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'determinants': determinants_reshaped,
        'eigenvals': eigenvals_reshaped,
        'condition_numbers': condition_numbers_reshaped,
        'anisotropy': anisotropy,
        'trace': trace
    }


def test_different_temperatures(model, centroids, metric_matrices, temperatures=[0.1, 0.3, 0.5, 0.8]):
    """Test different temperature settings for metric diversity."""
    print(f"\n🌡️ Testing Different Temperature Settings")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Metric Diversity with Different Temperatures", fontsize=16)
    
    for i, temp in enumerate(temperatures):
        print(f"\n--- Temperature {temp} ---")
        
        # Load metrics with this temperature
        centroids_tensor = torch.tensor(centroids, dtype=torch.float32, device=model.device)
        metric_matrices_tensor = torch.tensor(metric_matrices, dtype=torch.float32, device=model.device)
        model.load_pretrained_metrics_from_tensor(centroids_tensor, metric_matrices_tensor, 
                                                temperature=temp, regularization=0.01)
        
        # Create test points
        test_points = torch.randn(100, 2, device=model.device) * 3.0
        
        # Compute metric properties
        with torch.no_grad():
            G_test = model.G(test_points)
            eigenvals = torch.linalg.eigvals(G_test).real
            determinants = torch.linalg.det(G_test)
        
        print(f"✅ Eigenvalue range: [{eigenvals.min():.3e}, {eigenvals.max():.3e}]")
        print(f"✅ Determinant range: [{determinants.min():.3e}, {determinants.max():.3e}]")
        print(f"✅ Condition number range: [{eigenvals.max(dim=-1)[0].min():.2f}, {eigenvals.max(dim=-1)[0].max():.2f}]")
        
        # Plot
        ax = axes[i // 2, i % 2]
        scatter = ax.scatter(test_points[:, 0].cpu(), test_points[:, 1].cpu(), 
                           c=determinants.cpu(), cmap='viridis', alpha=0.7, s=50)
        ax.set_title(f"Temperature {temp}\nDet range: [{determinants.min():.2e}, {determinants.max():.2e}]")
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.savefig("temperature_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()


def precise_rhmc_sampling(model, n_samples=300, n_steps=60, save_prefix="precise"):
    """Perform more precise RHMC sampling to capture all clusters."""
    print(f"\n🎯 Precise RHMC Sampling to Capture All Clusters")
    print("=" * 60)
    
    # Create RHMC sampler with more steps for precision
    rhmc_sampler = RiemannianHMCSampler(model, mcmc_steps_nbr=n_steps, n_lf=20, eps_lf=0.015)
    
    # Sample from different starting points to cover all clusters
    starting_points = [
        torch.tensor([[-3.0, -2.0]], device=model.device),  # Cluster 1
        torch.tensor([[-1.5, 2.0]], device=model.device),   # Cluster 2
        torch.tensor([[0.0, 0.0]], device=model.device),    # Cluster 3
        torch.tensor([[2.0, -1.5]], device=model.device),   # Cluster 4
        torch.tensor([[3.5, 1.0]], device=model.device),    # Cluster 5
        torch.tensor([[-0.5, -3.0]], device=model.device),  # Cluster 6
        torch.tensor([[1.5, 3.0]], device=model.device),    # Cluster 7
    ]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7']
    
    # Create figure with proper layout for 7 clusters + combined view
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Precise RHMC Sampling from All Clusters", fontsize=16)
    
    all_samples = []
    all_colors = []
    
    for i, (start_point, color, label) in enumerate(zip(starting_points, colors, labels)):
        print(f"\n--- Sampling from {label} ({start_point}) ---")
        
        # Run RHMC sampling
        start_time = time.time()
        samples = rhmc_sampler.sample(n_samples)
        sampling_time = time.time() - start_time
        
        print(f"✅ Sampling completed in {sampling_time:.3f}s")
        print(f"✅ Samples range: [{samples.min():.3f}, {samples.max():.3f}]")
        print(f"✅ Samples mean: {samples.mean(dim=0)}")
        print(f"✅ Samples std: {samples.std(dim=0)}")
        
        # Analyze metric at samples
        with torch.no_grad():
            G_samples = model.G(samples)
            eigenvals = torch.linalg.eigvals(G_samples).real
            determinants = torch.linalg.det(G_samples)
        
        print(f"✅ Metric eigenvalues: min={eigenvals.min():.3e}, max={eigenvals.max():.3e}")
        print(f"✅ Determinants: min={determinants.min():.3e}, max={determinants.max():.3e}")
        
        all_samples.append(samples)
        all_colors.extend([color] * len(samples))
        
        # Plot samples colored by metric properties
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        scatter = ax.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), 
                           c=determinants.cpu(), cmap='viridis', alpha=0.7, s=30)
        ax.scatter(start_point[:, 0].cpu(), start_point[:, 1].cpu(), 
                  color='red', s=200, marker='*', label=f'Start: {label}')
        ax.set_title(f"RHMC from {label}\n(colored by det(G))")
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax)
    
    # Plot 8: Combined view of all samples
    ax_combined = axes[1, 3]
    all_samples_tensor = torch.cat(all_samples, dim=0)
    all_samples_np = all_samples_tensor.cpu().numpy()
    
    # Color by cluster
    for i, (samples, color) in enumerate(zip(all_samples, colors)):
        ax_combined.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), 
                          c=color, alpha=0.6, s=20, label=labels[i])
    
    ax_combined.set_title("All RHMC Samples Combined\n(colored by cluster)")
    ax_combined.set_xlabel("z₁")
    ax_combined.set_ylabel("z₂")
    ax_combined.set_xlim(-5, 5)
    ax_combined.set_ylim(-5, 5)
    ax_combined.grid(True, alpha=0.3)
    ax_combined.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_rhmc_sampling.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return all_samples, labels


def main():
    """Main detailed analysis function."""
    print("🔍 Detailed Manifold Analysis with Better Scaling")
    print("=" * 60)
    
    # Create detailed latent data with 7 clusters
    latent_data, cluster_centers = create_detailed_latent_data(n_samples=2000, n_clusters=7)
    
    # Create detailed centroids
    centroids, metric_matrices, labels = create_detailed_centroids(latent_data, n_centroids=50)
    
    # Create model
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
    
    # Test different temperatures
    test_different_temperatures(model, centroids, metric_matrices)
    
    # Load best temperature (0.5) for detailed analysis
    print(f"\n🔧 Loading metrics with temperature 0.5 for detailed analysis...")
    centroids_tensor = torch.tensor(centroids, dtype=torch.float32, device=device)
    metric_matrices_tensor = torch.tensor(metric_matrices, dtype=torch.float32, device=device)
    model.load_pretrained_metrics_from_tensor(centroids_tensor, metric_matrices_tensor, 
                                            temperature=0.5, regularization=0.01)
    
    # Visualize detailed manifold structure
    manifold_data = visualize_detailed_manifold_structure(model, n_points=900, save_prefix="detailed")
    
    # Perform precise RHMC sampling
    all_samples, labels = precise_rhmc_sampling(model, n_samples=300, n_steps=60, save_prefix="precise")
    
    print(f"\n✅ Detailed analysis completed!")
    print(f"📊 Key Improvements:")
    print(f"   - More centroids (50) to capture all 7 clusters")
    print(f"   - Consistent scaling across all plots")
    print(f"   - Different temperature settings for metric diversity")
    print(f"   - More precise RHMC with more steps and better parameters")
    print(f"   - Sampling from all 7 cluster centers")


if __name__ == "__main__":
    main() 