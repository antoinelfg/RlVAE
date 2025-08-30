#!/usr/bin/env python3
"""
RHVAE MNIST Metric Visualization
================================

This script creates comprehensive visualizations for the trained RHVAE model on MNIST:
- Metric determinant heatmaps
- Latent space visualization
- Geodesic paths
- Metric tensor analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pythae.models import AutoModel
from pythae.samplers import RHVAESampler, RHVAESamplerConfig
import torchvision.datasets as datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# Create output directory
output_dir = Path("rhvae_mnist_metric_analysis")
output_dir.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def load_trained_model():
    """Load the trained RHVAE model."""
    model_dir = "my_model_mnist/RHVAE_training_2025-08-07_16-36-13/final_model"
    trained_model = AutoModel.load_from_folder(model_dir)
    trained_model = trained_model.to(device)
    trained_model.eval()
    print(f"✅ Loaded trained model from {model_dir}")
    return trained_model

def load_mnist_data():
    """Load MNIST data for analysis."""
    mnist_trainset = datasets.MNIST(root='data', train=True, download=True, transform=None)
    train_dataset = mnist_trainset.data[:-10000].reshape(-1, 1, 28, 28) / 255.
    eval_dataset = mnist_trainset.data[-10000:].reshape(-1, 1, 28, 28) / 255.
    
    print(f"📊 MNIST data loaded: train={train_dataset.shape}, eval={eval_dataset.shape}")
    return train_dataset, eval_dataset

def compute_metric_determinant_heatmap(model, latent_points, resolution=50):
    """Compute metric determinant heatmap over latent space."""
    print("🔍 Computing metric determinant heatmap...")
    
    # Create a grid of latent points
    z_min, z_max = latent_points.min(dim=0)[0], latent_points.max(dim=0)[0]
    z_range = z_max - z_min
    z_min = z_min - 0.1 * z_range
    z_max = z_max + 0.1 * z_range
    
    # Create 2D grid for visualization (using first 2 dimensions)
    x = torch.linspace(z_min[0], z_max[0], resolution)
    y = torch.linspace(z_min[1], z_max[1], resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Create full latent points (set other dimensions to 0)
    grid_points = torch.zeros(resolution, resolution, model.latent_dim, device=device)
    grid_points[:, :, 0] = X
    grid_points[:, :, 1] = Y
    
    # Compute metric determinant for each point
    det_values = torch.zeros(resolution, resolution, device=device)
    
    with torch.no_grad():
        for i in range(resolution):
            for j in range(resolution):
                z = grid_points[i, j].unsqueeze(0)
                try:
                    # Compute metric tensor
                    G_inv = model.G_inv(z)
                    det_G_inv = torch.det(G_inv)
                    det_G = 1.0 / det_G_inv
                    det_values[i, j] = det_G.item()
                except:
                    det_values[i, j] = np.nan
    
    return X.cpu().numpy(), Y.cpu().numpy(), det_values.cpu().numpy()

def visualize_latent_space(model, eval_data, n_samples=1000):
    """Visualize latent space with TSNE and PCA."""
    print("🔍 Computing latent representations...")
    
    # Encode data using a simpler approach
    with torch.no_grad():
        encoded_data = []
        labels = []
        
        for i in range(0, min(n_samples, len(eval_data)), 64):
            batch = eval_data[i:i+64].to(device)
            
            # Use encoder directly to get latent representations
            encoder_output = model.encoder(batch)
            mu = encoder_output.embedding
            encoded_data.append(mu.cpu())
            labels.extend(range(i, min(i+64, n_samples)))
    
    latent_points = torch.cat(encoded_data, dim=0)
    print(f"📊 Encoded {len(latent_points)} points to latent space")
    
    # TSNE visualization
    print("🔍 Computing TSNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_tsne = tsne.fit_transform(latent_points.numpy())
    
    # PCA visualization
    print("🔍 Computing PCA...")
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_points.numpy())
    
    return latent_points, latent_tsne, latent_pca, labels

def compute_geodesic_paths(model, latent_points, n_paths=10):
    """Compute geodesic paths between random pairs of points."""
    print("🔍 Computing geodesic paths...")
    
    # Sample random pairs
    n_points = len(latent_points)
    path_pairs = []
    
    for _ in range(n_paths):
        i, j = np.random.choice(n_points, 2, replace=False)
        path_pairs.append((latent_points[i], latent_points[j]))
    
    # Compute geodesic paths using simple interpolation for now
    geodesic_paths = []
    
    for start_z, end_z in path_pairs:
        try:
            # Create a path by interpolating between points
            path_points = []
            for t in np.linspace(0, 1, 20):
                interpolated_z = (1-t) * start_z + t * end_z
                path_points.append(interpolated_z)
            
            geodesic_paths.append(torch.stack(path_points))
        except Exception as e:
            print(f"⚠️ Failed to compute geodesic path: {e}")
            continue
    
    return geodesic_paths

def create_comprehensive_visualization(model, eval_data):
    """Create comprehensive visualization of RHVAE metric analysis."""
    print("🎨 Creating comprehensive visualization...")
    
    # 1. Get latent representations
    latent_points, latent_tsne, latent_pca, labels = visualize_latent_space(model, eval_data)
    
    # 2. Compute metric determinant heatmap
    X, Y, det_values = compute_metric_determinant_heatmap(model, latent_points)
    
    # 3. Compute geodesic paths
    geodesic_paths = compute_geodesic_paths(model, latent_points)
    
    # 4. Create comprehensive plot
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: TSNE with geodesic paths
    ax1 = plt.subplot(2, 3, 1)
    plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], alpha=0.6, s=10)
    for path in geodesic_paths:
        if len(path) > 0:
            # Use smaller perplexity for small paths
            n_samples = len(path)
            perplexity = min(5, n_samples - 1)  # Ensure perplexity < n_samples
            if perplexity > 0:
                path_tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(path.numpy())
                plt.plot(path_tsne[:, 0], path_tsne[:, 1], 'r-', alpha=0.7, linewidth=2)
    plt.title('Latent Space (TSNE) with Geodesic Paths', fontsize=14)
    plt.xlabel('TSNE 1')
    plt.ylabel('TSNE 2')
    
    # Plot 2: PCA with geodesic paths
    ax2 = plt.subplot(2, 3, 2)
    plt.scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.6, s=10)
    for path in geodesic_paths:
        if len(path) > 0:
            path_pca = PCA(n_components=2).fit_transform(path.numpy())
            plt.plot(path_pca[:, 0], path_pca[:, 1], 'r-', alpha=0.7, linewidth=2)
    plt.title('Latent Space (PCA) with Geodesic Paths', fontsize=14)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    
    # Plot 3: Metric determinant heatmap
    ax3 = plt.subplot(2, 3, 3)
    im = plt.imshow(det_values.T, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                    origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax3, label='det(G)')
    plt.title('Metric Determinant Heatmap', fontsize=14)
    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    
    # Plot 4: Log metric determinant heatmap
    ax4 = plt.subplot(2, 3, 4)
    log_det_values = np.log(det_values + 1e-8)  # Add small epsilon to avoid log(0)
    im = plt.imshow(log_det_values.T, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                    origin='lower', cmap='plasma', aspect='auto')
    plt.colorbar(im, ax=ax4, label='log(det(G))')
    plt.title('Log Metric Determinant Heatmap', fontsize=14)
    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    
    # Plot 5: Latent space with metric determinant overlay
    ax5 = plt.subplot(2, 3, 5)
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], 
                         c=latent_points[:, 0], alpha=0.6, s=10, cmap='viridis')
    plt.colorbar(scatter, ax=ax5, label='Latent Dim 1')
    plt.title('Latent Space Colored by First Dimension', fontsize=14)
    plt.xlabel('TSNE 1')
    plt.ylabel('TSNE 2')
    
    # Plot 6: Metric determinant distribution
    ax6 = plt.subplot(2, 3, 6)
    valid_det = det_values[~np.isnan(det_values)]
    plt.hist(valid_det.flatten(), bins=50, alpha=0.7, density=True)
    plt.xlabel('det(G)')
    plt.ylabel('Density')
    plt.title('Metric Determinant Distribution', fontsize=14)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rhvae_mnist_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved comprehensive analysis to {output_dir / 'rhvae_mnist_comprehensive_analysis.png'}")

def create_metric_tensor_analysis(model, latent_points):
    """Create detailed metric tensor analysis."""
    print("🔍 Creating metric tensor analysis...")
    
    # Sample points for analysis
    n_analysis_points = 100
    analysis_points = latent_points[:n_analysis_points]
    
    # Compute metric tensors
    metric_tensors = []
    metric_determinants = []
    metric_eigenvalues = []
    
    with torch.no_grad():
        for z in analysis_points:
            z = z.unsqueeze(0).to(device)
            try:
                G_inv = model.G_inv(z)
                G = torch.inverse(G_inv)
                
                metric_tensors.append(G.cpu())
                det_G = torch.det(G)
                metric_determinants.append(det_G.cpu())
                
                # Compute eigenvalues
                eigenvals = torch.linalg.eigvals(G)
                metric_eigenvalues.append(eigenvals.cpu())
                
            except Exception as e:
                print(f"⚠️ Failed to compute metric for point: {e}")
                continue
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Metric determinant vs position
    det_values = torch.stack(metric_determinants).numpy()
    axes[0, 0].hist(det_values, bins=30, alpha=0.7)
    axes[0, 0].set_xlabel('det(G)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Metric Determinant Distribution')
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Log metric determinant
    log_det_values = np.log(det_values + 1e-8)
    axes[0, 1].hist(log_det_values, bins=30, alpha=0.7)
    axes[0, 1].set_xlabel('log(det(G))')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Log Metric Determinant Distribution')
    
    # Plot 3: Eigenvalue distribution
    all_eigenvals = torch.cat(metric_eigenvalues, dim=0).numpy()
    axes[0, 2].hist(all_eigenvals.real, bins=30, alpha=0.7)
    axes[0, 2].set_xlabel('Eigenvalue (Real Part)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Metric Eigenvalue Distribution')
    
    # Plot 4: Metric determinant vs latent dimension 1
    latent_dim1 = analysis_points[:len(det_values), 0].numpy()
    axes[1, 0].scatter(latent_dim1, det_values, alpha=0.6)
    axes[1, 0].set_xlabel('Latent Dimension 1')
    axes[1, 0].set_ylabel('det(G)')
    axes[1, 0].set_title('Metric Determinant vs Latent Dim 1')
    
    # Plot 5: Metric determinant vs latent dimension 2
    latent_dim2 = analysis_points[:len(det_values), 1].numpy()
    axes[1, 1].scatter(latent_dim2, det_values, alpha=0.6)
    axes[1, 1].set_xlabel('Latent Dimension 2')
    axes[1, 1].set_ylabel('det(G)')
    axes[1, 1].set_title('Metric Determinant vs Latent Dim 2')
    
    # Plot 6: Metric tensor heatmap (average)
    if metric_tensors:
        avg_metric = torch.stack(metric_tensors).mean(dim=0).numpy()
        # Remove the batch dimension if it exists
        if avg_metric.ndim == 3 and avg_metric.shape[0] == 1:
            avg_metric = avg_metric.squeeze(0)
        im = axes[1, 2].imshow(avg_metric, cmap='viridis')
        plt.colorbar(im, ax=axes[1, 2])
        axes[1, 2].set_title('Average Metric Tensor')
        axes[1, 2].set_xlabel('Dimension')
        axes[1, 2].set_ylabel('Dimension')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rhvae_mnist_metric_tensor_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved metric tensor analysis to {output_dir / 'rhvae_mnist_metric_tensor_analysis.png'}")

def create_geodesic_visualization(model, latent_points):
    """Create detailed geodesic visualization."""
    print("🔍 Creating geodesic visualization...")
    
    # Sample points for geodesic computation
    n_geodesics = 20
    geodesic_pairs = []
    
    for _ in range(n_geodesics):
        i, j = np.random.choice(len(latent_points), 2, replace=False)
        geodesic_pairs.append((latent_points[i], latent_points[j]))
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: TSNE with geodesics
    tsne = TSNE(n_components=2, random_state=42)
    latent_tsne = tsne.fit_transform(latent_points.numpy())
    
    axes[0, 0].scatter(latent_tsne[:, 0], latent_tsne[:, 1], alpha=0.6, s=10)
    
    # Compute and plot geodesics
    sampler_config = RHVAESamplerConfig(mcmc_steps_nbr=30, n_lf=3, eps_lf=0.01)
    sampler = RHVAESampler(sampler_config=sampler_config, model=model)
    
    for i, (start_z, end_z) in enumerate(geodesic_pairs[:10]):  # Plot first 10
        try:
            # Create geodesic path using simple interpolation
            path_points = []
            for t in np.linspace(0, 1, 15):
                interpolated_z = (1-t) * start_z + t * end_z
                path_points.append(interpolated_z)
            
            path_tensor = torch.stack(path_points)
            # Use smaller perplexity for small paths
            n_samples = len(path_tensor)
            perplexity = min(5, n_samples - 1)
            if perplexity > 0:
                path_tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(path_tensor.numpy())
                axes[0, 0].plot(path_tsne[:, 0], path_tsne[:, 1], 'r-', alpha=0.7, linewidth=1)
            
        except Exception as e:
            print(f"⚠️ Failed to compute geodesic {i}: {e}")
            continue
    
    axes[0, 0].set_title('Geodesic Paths in Latent Space (TSNE)', fontsize=14)
    axes[0, 0].set_xlabel('TSNE 1')
    axes[0, 0].set_ylabel('TSNE 2')
    
    # Plot 2: PCA with geodesics
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_points.numpy())
    
    axes[0, 1].scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.6, s=10)
    
    for i, (start_z, end_z) in enumerate(geodesic_pairs[:10]):
        try:
            path_points = []
            for t in np.linspace(0, 1, 15):
                interpolated_z = (1-t) * start_z + t * end_z
                path_points.append(interpolated_z)
            
            path_tensor = torch.stack(path_points)
            path_pca = pca.transform(path_tensor.numpy())
            axes[0, 1].plot(path_pca[:, 0], path_pca[:, 1], 'r-', alpha=0.7, linewidth=1)
            
        except Exception as e:
            continue
    
    axes[0, 1].set_title('Geodesic Paths in Latent Space (PCA)', fontsize=14)
    axes[0, 1].set_xlabel('PCA 1')
    axes[0, 1].set_ylabel('PCA 2')
    
    # Plot 3: Geodesic length distribution
    geodesic_lengths = []
    for start_z, end_z in geodesic_pairs:
        try:
            # Compute geodesic length (approximate)
            path_points = []
            for t in np.linspace(0, 1, 20):
                interpolated_z = (1-t) * start_z + t * end_z
                path_points.append(interpolated_z)
            
            path_tensor = torch.stack(path_points)
            # Compute approximate length
            length = torch.norm(path_tensor[1:] - path_tensor[:-1], dim=1).sum().item()
            geodesic_lengths.append(length)
            
        except Exception as e:
            continue
    
    if geodesic_lengths:
        axes[1, 0].hist(geodesic_lengths, bins=15, alpha=0.7)
        axes[1, 0].set_xlabel('Geodesic Length (Approximate)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Geodesic Length Distribution')
    
    # Plot 4: Metric determinant along geodesic
    if geodesic_pairs:
        start_z, end_z = geodesic_pairs[0]
        try:
            det_values = []
            t_values = np.linspace(0, 1, 20)
            
            for t in t_values:
                interpolated_z = (1-t) * start_z + t * end_z
                z = interpolated_z.unsqueeze(0).to(device)
                G_inv = model.G_inv(z)
                G = torch.inverse(G_inv)
                det_G = torch.det(G)
                det_values.append(det_G.item())
            
            axes[1, 1].plot(t_values, det_values, 'b-', linewidth=2)
            axes[1, 1].set_xlabel('Interpolation Parameter t')
            axes[1, 1].set_ylabel('det(G)')
            axes[1, 1].set_title('Metric Determinant Along Geodesic')
            
        except Exception as e:
            print(f"⚠️ Failed to compute metric along geodesic: {e}")
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rhvae_mnist_geodesic_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved geodesic analysis to {output_dir / 'rhvae_mnist_geodesic_analysis.png'}")

def main():
    """Main function to create all visualizations."""
    print("🎨 RHVAE MNIST Metric Visualization")
    print("=" * 50)
    
    # Load model and data
    model = load_trained_model()
    train_data, eval_data = load_mnist_data()
    
    # Get latent representations
    latent_points, latent_tsne, latent_pca, labels = visualize_latent_space(model, eval_data)
    
    # Create all visualizations
    create_comprehensive_visualization(model, eval_data)
    create_metric_tensor_analysis(model, latent_points)
    create_geodesic_visualization(model, latent_points)
    
    print(f"\n✅ All visualizations completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"   - rhvae_mnist_comprehensive_analysis.png")
    print(f"   - rhvae_mnist_metric_tensor_analysis.png")
    print(f"   - rhvae_mnist_geodesic_analysis.png")

if __name__ == "__main__":
    main() 