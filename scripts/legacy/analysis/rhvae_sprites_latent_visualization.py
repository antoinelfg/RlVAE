#!/usr/bin/env python3
"""
RHVAE Sprites Latent Space Visualization
========================================

Visualize the latent space with centroids and the learned metric tensor.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pythae.models import AutoModel
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_trained_model():
    """Load the trained RHVAE model."""
    model_dir = "rhvae_sprites_balanced_model/RHVAE_training_*/final_model"
    import glob
    model_paths = glob.glob(model_dir)
    if not model_paths:
        raise FileNotFoundError("No trained model found!")
    
    latest_model = sorted(model_paths)[-1]
    trained_model = AutoModel.load_from_folder(latest_model)
    trained_model = trained_model.to(device)
    trained_model.eval()
    print(f"✅ Loaded trained model from {latest_model}")
    return trained_model

def load_sprites_data():
    """Load sprites data for encoding."""
    print("📂 Loading sprites data...")
    sprites_data = torch.load('/home/alaforgu/scratch/longitudinal_experiments/RlVAE/data/processed/Sprites_train_cyclic.pt')
    
    # Process data: take first frame from each sequence and convert to grayscale
    first_frame = sprites_data[:, 0, :, :, :]  # [batch, 3, 64, 64]
    
    # Convert RGB to grayscale
    grayscale = first_frame[:, 0, :, :] * 0.299 + first_frame[:, 1, :, :] * 0.587 + first_frame[:, 2, :, :] * 0.114
    grayscale = grayscale.unsqueeze(1)  # [batch, 1, 64, 64]
    
    # Resize to 28x28 to match MNIST encoder
    import torch.nn.functional as F
    processed_data = F.interpolate(grayscale, size=(28, 28), mode='bilinear', align_corners=False)
    
    print(f"📊 Processed data shape: {processed_data.shape}")
    return processed_data

def encode_data_to_latent(model, data, batch_size=64):
    """Encode data to latent space."""
    print("🔍 Encoding data to latent space...")
    latent_points = []
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size].to(device)
            # Get latent representations
            encoder_output = model.encoder(batch)
            mu = encoder_output.embedding
            latent_points.append(mu.cpu())
            
            if i % 500 == 0:
                print(f"  Encoded {i}/{len(data)} samples...")
    
    return torch.cat(latent_points, dim=0).numpy()

def compute_centroids(latent_points, n_clusters=8):
    """Compute centroids using K-means clustering."""
    print(f"🎯 Computing {n_clusters} centroids...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_points)
    centroids = kmeans.cluster_centers_
    
    print(f"✅ Found {len(centroids)} centroids")
    return centroids, cluster_labels

def compute_metric_at_points(model, points, n_samples=100):
    """Compute metric tensor at given points."""
    print("📏 Computing metric tensors...")
    metric_determinants = []
    metric_traces = []
    
    with torch.no_grad():
        for i, point in enumerate(points):
            try:
                z = torch.tensor(point, dtype=torch.float32).unsqueeze(0).to(device)
                G_inv = model.G_inv(z)
                G_inv = G_inv.squeeze(0)
                
                det_G_inv = torch.det(G_inv)
                trace_G_inv = torch.trace(G_inv)
                
                metric_determinants.append(1.0 / det_G_inv.item() if det_G_inv.item() != 0 else 0.0)
                metric_traces.append(trace_G_inv.item())
                
            except Exception as e:
                print(f"⚠️ Failed to compute metric for point {i}: {e}")
                metric_determinants.append(0.0)
                metric_traces.append(0.0)
    
    return np.array(metric_determinants), np.array(metric_traces)

def create_comprehensive_visualization(model, latent_points, centroids, cluster_labels, metric_dets, metric_traces):
    """Create comprehensive latent space visualization."""
    print("🎨 Creating comprehensive visualization...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. TSNE visualization with centroids
    ax1 = plt.subplot(2, 3, 1)
    print("  Computing TSNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_points)//4))
    
    # Combine data and centroids for TSNE
    combined_data = np.vstack([latent_points, centroids])
    combined_tsne = tsne.fit_transform(combined_data)
    
    # Split back into data and centroids
    latent_tsne = combined_tsne[:len(latent_points)]
    centroid_tsne = combined_tsne[len(latent_points):]
    
    # Plot points colored by cluster
    scatter = ax1.scatter(latent_tsne[:, 0], latent_tsne[:, 1], 
                         c=cluster_labels, cmap='tab10', alpha=0.6, s=20)
    # Plot centroids
    ax1.scatter(centroid_tsne[:, 0], centroid_tsne[:, 1], 
                c='red', s=200, marker='*', edgecolors='black', linewidth=2, label='Centroids')
    ax1.set_title('Latent Space (TSNE) with Centroids', fontsize=14, fontweight='bold')
    ax1.set_xlabel('TSNE 1')
    ax1.set_ylabel('TSNE 2')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Cluster')
    
    # 2. PCA visualization with centroids
    ax2 = plt.subplot(2, 3, 2)
    print("  Computing PCA...")
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_points)
    centroid_pca = pca.transform(centroids)
    
    scatter = ax2.scatter(latent_pca[:, 0], latent_pca[:, 1], 
                         c=cluster_labels, cmap='tab10', alpha=0.6, s=20)
    ax2.scatter(centroid_pca[:, 0], centroid_pca[:, 1], 
                c='red', s=200, marker='*', edgecolors='black', linewidth=2, label='Centroids')
    ax2.set_title('Latent Space (PCA) with Centroids', fontsize=14, fontweight='bold')
    ax2.set_xlabel('PCA 1')
    ax2.set_ylabel('PCA 2')
    ax2.legend()
    plt.colorbar(scatter, ax=ax2, label='Cluster')
    
    # 3. Metric determinant heatmap
    ax3 = plt.subplot(2, 3, 3)
    # Sample a grid of points for metric visualization
    x_min, x_max = latent_points[:, 0].min(), latent_points[:, 0].max()
    y_min, y_max = latent_points[:, 1].min(), latent_points[:, 1].max()
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20), np.linspace(y_min, y_max, 20))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Compute metric at grid points (use first 2 dimensions)
    grid_metric_dets, _ = compute_metric_at_points(model, grid_points)
    grid_metric_dets = grid_metric_dets.reshape(xx.shape)
    
    im = ax3.imshow(grid_metric_dets, extent=[x_min, x_max, y_min, y_max], 
                    origin='lower', cmap='viridis', aspect='auto')
    ax3.scatter(latent_points[:, 0], latent_points[:, 1], c='white', s=10, alpha=0.5)
    ax3.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='*', 
                edgecolors='black', linewidth=2)
    ax3.set_title('Metric Determinant Heatmap', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Latent Dim 1')
    ax3.set_ylabel('Latent Dim 2')
    plt.colorbar(im, ax=ax3, label='det(G)')
    
    # 4. Metric determinant distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(metric_dets, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.set_title('Metric Determinant Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('det(G)')
    ax4.set_ylabel('Count')
    ax4.set_yscale('log')
    
    # 5. Metric trace distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(metric_traces, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax5.set_title('Metric Trace Distribution', fontsize=14, fontweight='bold')
    ax5.set_xlabel('trace(G_inv)')
    ax5.set_ylabel('Count')
    
    # 6. Centroid metric comparison
    ax6 = plt.subplot(2, 3, 6)
    centroid_metric_dets, centroid_metric_traces = compute_metric_at_points(model, centroids)
    
    x_pos = np.arange(len(centroids))
    bars1 = ax6.bar(x_pos - 0.2, centroid_metric_dets, 0.4, label='det(G)', alpha=0.7)
    ax6_twin = ax6.twinx()
    bars2 = ax6_twin.bar(x_pos + 0.2, centroid_metric_traces, 0.4, label='trace(G_inv)', 
                          color='orange', alpha=0.7)
    
    ax6.set_title('Metric Properties at Centroids', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Centroid Index')
    ax6.set_ylabel('det(G)', color='blue')
    ax6_twin.set_ylabel('trace(G_inv)', color='orange')
    
    # Add legends
    ax6.legend(loc='upper left')
    ax6_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('rhvae_sprites_latent_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Comprehensive visualization saved to rhvae_sprites_latent_comprehensive.png")

def create_metric_heatmap_2d(model, latent_points, centroids):
    """Create a detailed 2D metric heatmap."""
    print("🔥 Creating detailed metric heatmap...")
    
    # Use first two dimensions for 2D visualization
    x_min, x_max = latent_points[:, 0].min(), latent_points[:, 0].max()
    y_min, y_max = latent_points[:, 1].min(), latent_points[:, 1].max()
    
    # Create finer grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # For points beyond 2D, use zeros for other dimensions
    full_grid_points = np.zeros((len(grid_points), latent_points.shape[1]))
    full_grid_points[:, :2] = grid_points
    
    # Compute metric at grid points
    grid_metric_dets, grid_metric_traces = compute_metric_at_points(model, full_grid_points)
    grid_metric_dets = grid_metric_dets.reshape(xx.shape)
    grid_metric_traces = grid_metric_traces.reshape(xx.shape)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Metric determinant heatmap
    im1 = axes[0].imshow(grid_metric_dets, extent=[x_min, x_max, y_min, y_max], 
                          origin='lower', cmap='viridis', aspect='auto')
    axes[0].scatter(latent_points[:, 0], latent_points[:, 1], c='white', s=5, alpha=0.3)
    axes[0].scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='*', 
                    edgecolors='black', linewidth=2)
    axes[0].set_title('Metric Determinant Heatmap', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Latent Dim 1')
    axes[0].set_ylabel('Latent Dim 2')
    plt.colorbar(im1, ax=axes[0], label='det(G)')
    
    # Plot 2: Metric trace heatmap
    im2 = axes[1].imshow(grid_metric_traces, extent=[x_min, x_max, y_min, y_max], 
                          origin='lower', cmap='plasma', aspect='auto')
    axes[1].scatter(latent_points[:, 0], latent_points[:, 1], c='white', s=5, alpha=0.3)
    axes[1].scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='*', 
                    edgecolors='black', linewidth=2)
    axes[1].set_title('Metric Trace Heatmap', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Latent Dim 1')
    axes[1].set_ylabel('Latent Dim 2')
    plt.colorbar(im2, ax=axes[1], label='trace(G_inv)')
    
    plt.tight_layout()
    plt.savefig('rhvae_sprites_metric_heatmap_2d.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Detailed metric heatmap saved to rhvae_sprites_metric_heatmap_2d.png")

def main():
    """Main visualization function."""
    print("🎨 RHVAE Sprites Latent Space Visualization")
    print("=" * 50)
    
    # Load model and data
    model = load_trained_model()
    data = load_sprites_data()
    
    # Encode data to latent space
    latent_points = encode_data_to_latent(model, data)
    print(f"📊 Encoded {len(latent_points)} points to {latent_points.shape[1]}-dimensional latent space")
    
    # Compute centroids
    centroids, cluster_labels = compute_centroids(latent_points, n_clusters=8)
    
    # Compute metric at all points and centroids
    metric_dets, metric_traces = compute_metric_at_points(model, latent_points)
    
    # Create visualizations
    create_comprehensive_visualization(model, latent_points, centroids, cluster_labels, metric_dets, metric_traces)
    create_metric_heatmap_2d(model, latent_points, centroids)
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print(f"   - Latent space dimension: {latent_points.shape[1]}")
    print(f"   - Number of data points: {len(latent_points)}")
    print(f"   - Number of centroids: {len(centroids)}")
    print(f"   - Metric determinant range: [{metric_dets.min():.2e}, {metric_dets.max():.2e}]")
    print(f"   - Metric trace range: [{metric_traces.min():.2f}, {metric_traces.max():.2f}]")
    print(f"   - Coefficient of variation (det): {metric_dets.std()/metric_dets.mean():.3f}")
    
    print("\n✅ Visualization completed!")
    print("📁 Files created:")
    print("   - rhvae_sprites_latent_comprehensive.png")
    print("   - rhvae_sprites_metric_heatmap_2d.png")

if __name__ == "__main__":
    main() 