#!/usr/bin/env python3
"""
RHVAE Sprites Precise Analysis
==============================

High-precision analysis with more centroids and detailed metric computation
to fit the data more closely.
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
    """Load the fixed RHVAE model."""
    model_dir = "rhvae_sprites_metric_fixed/RHVAE_training_*/final_model"
    import glob
    model_paths = glob.glob(model_dir)
    if not model_paths:
        raise FileNotFoundError("No trained model found!")
    
    latest_model = sorted(model_paths)[-1]
    trained_model = AutoModel.load_from_folder(latest_model)
    trained_model = trained_model.to(device)
    trained_model.eval()
    print(f"✅ Loaded fixed model from {latest_model}")
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

def compute_precise_centroids(latent_points, n_clusters=16):
    """Compute more centroids for precise analysis."""
    print(f"🎯 Computing {n_clusters} precise centroids...")
    
    # Try different numbers of clusters to find optimal
    inertias = []
    K_range = range(8, 25, 2)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(latent_points)
        inertias.append(kmeans.inertia_)
    
    # Find elbow point for optimal number of clusters (simplified)
    try:
        # Simple elbow detection: find point with maximum curvature
        inertias_np = np.array(inertias)
        inertias_diff = np.diff(inertias_np)
        inertias_diff2 = np.diff(inertias_diff)
        elbow_idx = np.argmax(np.abs(inertias_diff2)) + 1
        optimal_k = K_range[elbow_idx] if elbow_idx < len(K_range) else n_clusters
        print(f"📊 Optimal number of clusters (elbow method): {optimal_k}")
    except:
        optimal_k = n_clusters
        print(f"📊 Using default number of clusters: {optimal_k}")
    
    # Compute final clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=15)
    cluster_labels = kmeans.fit_predict(latent_points)
    centroids = kmeans.cluster_centers_
    
    print(f"✅ Found {len(centroids)} precise centroids")
    print(f"📊 Cluster sizes: {np.bincount(cluster_labels)}")
    
    return centroids, cluster_labels, optimal_k

def compute_precise_metric_at_points(model, points, precision='double'):
    """Compute metric tensor with high precision."""
    print(f"📏 Computing precise metric tensors (precision: {precision})...")
    metric_determinants = []
    metric_traces = []
    metric_condition_numbers = []
    metric_eigenvalues = []
    
    with torch.no_grad():
        for i, point in enumerate(points):
            try:
                # Use higher precision if requested
                if precision == 'double':
                    z = torch.tensor(point, dtype=torch.float64).unsqueeze(0).to(device)
                else:
                    z = torch.tensor(point, dtype=torch.float32).unsqueeze(0).to(device)
                
                G_inv = model.G_inv(z)
                G_inv = G_inv.squeeze(0)
                
                # Compute with higher precision
                det_G_inv = torch.det(G_inv)
                trace_G_inv = torch.trace(G_inv)
                
                # Compute eigenvalues and condition number
                eigenvals = torch.linalg.eigvals(G_inv)
                eigenvals_real = eigenvals.real
                condition_number = torch.max(eigenvals_real) / torch.min(eigenvals_real)
                
                metric_determinants.append(det_G_inv.item())
                metric_traces.append(trace_G_inv.item())
                metric_condition_numbers.append(condition_number.item())
                metric_eigenvalues.append(eigenvals_real.cpu())
                
                if i % 100 == 0:
                    print(f"  Processed {i}/{len(points)} points...")
                
            except Exception as e:
                print(f"⚠️ Failed to compute metric for point {i}: {e}")
                continue
    
    return (np.array(metric_determinants), np.array(metric_traces), 
            np.array(metric_condition_numbers), metric_eigenvalues)

def create_precise_visualization(model, latent_points, centroids, cluster_labels, 
                                metric_dets, metric_traces, metric_conditions, n_clusters):
    """Create precise visualization with more centroids and detailed analysis."""
    print("🎨 Creating precise visualization...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(24, 20))
    
    # 1. TSNE visualization with more centroids
    ax1 = plt.subplot(3, 4, 1)
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
                         c=cluster_labels, cmap='tab20', alpha=0.6, s=15)
    # Plot centroids
    ax1.scatter(centroid_tsne[:, 0], centroid_tsne[:, 1], 
                c='red', s=150, marker='*', edgecolors='black', linewidth=2, label=f'{n_clusters} Centroids')
    ax1.set_title(f'Latent Space (TSNE) with {n_clusters} Centroids\n(Precise Analysis)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('TSNE 1')
    ax1.set_ylabel('TSNE 2')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Cluster')
    
    # 2. PCA visualization with more centroids
    ax2 = plt.subplot(3, 4, 2)
    print("  Computing PCA...")
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_points)
    centroid_pca = pca.transform(centroids)
    
    scatter = ax2.scatter(latent_pca[:, 0], latent_pca[:, 1], 
                         c=cluster_labels, cmap='tab20', alpha=0.6, s=15)
    ax2.scatter(centroid_pca[:, 0], centroid_pca[:, 1], 
                c='red', s=150, marker='*', edgecolors='black', linewidth=2, label=f'{n_clusters} Centroids')
    ax2.set_title(f'Latent Space (PCA) with {n_clusters} Centroids\n(Precise Analysis)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('PCA 1')
    ax2.set_ylabel('PCA 2')
    ax2.legend()
    plt.colorbar(scatter, ax=ax2, label='Cluster')
    
    # 3. High-resolution metric determinant heatmap
    ax3 = plt.subplot(3, 4, 3)
    print("  Computing high-resolution metric heatmap...")
    x_min, x_max = latent_points[:, 0].min(), latent_points[:, 0].max()
    y_min, y_max = latent_points[:, 1].min(), latent_points[:, 1].max()
    
    # Higher resolution grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # For points beyond 2D, use zeros for other dimensions
    full_grid_points = np.zeros((len(grid_points), latent_points.shape[1]))
    full_grid_points[:, :2] = grid_points
    
    # Compute metric at grid points
    grid_metric_dets, _, _, _ = compute_precise_metric_at_points(model, full_grid_points)
    grid_metric_dets = grid_metric_dets.reshape(xx.shape)
    
    im = ax3.imshow(grid_metric_dets, extent=[x_min, x_max, y_min, y_max], 
                    origin='lower', cmap='viridis', aspect='auto')
    ax3.scatter(latent_points[:, 0], latent_points[:, 1], c='white', s=3, alpha=0.3)
    ax3.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, marker='*', 
                edgecolors='black', linewidth=1)
    ax3.set_title('High-Resolution Metric Determinant\n(100x100 Grid)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Latent Dim 1')
    ax3.set_ylabel('Latent Dim 2')
    plt.colorbar(im, ax=ax3, label='det(G_inv)')
    
    # 4. Precise metric determinant distribution
    ax4 = plt.subplot(3, 4, 4)
    ax4.hist(metric_dets, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax4.set_title('Precise Metric Determinant Distribution\n(100 Bins)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('det(G_inv)')
    ax4.set_ylabel('Count')
    ax4.set_yscale('log')
    
    # 5. Metric trace distribution
    ax5 = plt.subplot(3, 4, 5)
    ax5.hist(metric_traces, bins=100, alpha=0.7, color='green', edgecolor='black')
    ax5.set_title('Precise Metric Trace Distribution\n(100 Bins)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('trace(G_inv)')
    ax5.set_ylabel('Count')
    
    # 6. Condition number distribution
    ax6 = plt.subplot(3, 4, 6)
    ax6.hist(metric_conditions, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax6.set_title('Precise Condition Number Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Condition Number')
    ax6.set_ylabel('Count')
    ax6.axvline(x=100, color='red', linestyle='--', alpha=0.7, label='Threshold (100)')
    ax6.legend()
    
    # 7. Centroid metric analysis
    ax7 = plt.subplot(3, 4, 7)
    centroid_metric_dets, centroid_metric_traces, centroid_metric_conditions, _ = compute_precise_metric_at_points(model, centroids)
    
    x_pos = np.arange(len(centroids))
    bars1 = ax7.bar(x_pos - 0.2, centroid_metric_dets, 0.4, label='det(G_inv)', alpha=0.7)
    ax7_twin = ax7.twinx()
    bars2 = ax7_twin.bar(x_pos + 0.2, centroid_metric_traces, 0.4, label='trace(G_inv)', 
                          color='orange', alpha=0.7)
    
    ax7.set_title(f'Metric Properties at {n_clusters} Centroids', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Centroid Index')
    ax7.set_ylabel('det(G_inv)', color='blue')
    ax7_twin.set_ylabel('trace(G_inv)', color='orange')
    ax7.legend(loc='upper left')
    ax7_twin.legend(loc='upper right')
    
    # 8. Cluster size distribution
    ax8 = plt.subplot(3, 4, 8)
    cluster_sizes = np.bincount(cluster_labels)
    ax8.bar(range(len(cluster_sizes)), cluster_sizes, alpha=0.7, color='teal')
    ax8.set_title(f'Cluster Size Distribution\n({n_clusters} Clusters)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Cluster Index')
    ax8.set_ylabel('Number of Points')
    
    # 9. Metric determinant vs cluster
    ax9 = plt.subplot(3, 4, 9)
    cluster_avg_dets = []
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        if np.sum(cluster_mask) > 0:
            cluster_avg_dets.append(np.mean(metric_dets[cluster_mask]))
        else:
            cluster_avg_dets.append(0)
    
    ax9.bar(range(n_clusters), cluster_avg_dets, alpha=0.7, color='coral')
    ax9.set_title('Average Metric Determinant per Cluster', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Cluster Index')
    ax9.set_ylabel('Average det(G_inv)')
    
    # 10. Metric trace vs cluster
    ax10 = plt.subplot(3, 4, 10)
    cluster_avg_traces = []
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        if np.sum(cluster_mask) > 0:
            cluster_avg_traces.append(np.mean(metric_traces[cluster_mask]))
        else:
            cluster_avg_traces.append(0)
    
    ax10.bar(range(n_clusters), cluster_avg_traces, alpha=0.7, color='gold')
    ax10.set_title('Average Metric Trace per Cluster', fontsize=12, fontweight='bold')
    ax10.set_xlabel('Cluster Index')
    ax10.set_ylabel('Average trace(G_inv)')
    
    # 11. Condition number vs cluster
    ax11 = plt.subplot(3, 4, 11)
    cluster_avg_conditions = []
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        if np.sum(cluster_mask) > 0:
            cluster_avg_conditions.append(np.mean(metric_conditions[cluster_mask]))
        else:
            cluster_avg_conditions.append(0)
    
    ax11.bar(range(n_clusters), cluster_avg_conditions, alpha=0.7, color='lightgreen')
    ax11.set_title('Average Condition Number per Cluster', fontsize=12, fontweight='bold')
    ax11.set_xlabel('Cluster Index')
    ax11.set_ylabel('Average Condition Number')
    ax11.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Threshold (100)')
    ax11.legend()
    
    # 12. Metric variation analysis
    ax12 = plt.subplot(3, 4, 12)
    cluster_cvs = []
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        if np.sum(cluster_mask) > 1:
            cluster_dets = metric_dets[cluster_mask]
            cv = cluster_dets.std() / cluster_dets.mean()
            cluster_cvs.append(cv)
        else:
            cluster_cvs.append(0)
    
    ax12.bar(range(n_clusters), cluster_cvs, alpha=0.7, color='lightblue')
    ax12.set_title('Coefficient of Variation per Cluster', fontsize=12, fontweight='bold')
    ax12.set_xlabel('Cluster Index')
    ax12.set_ylabel('CV of det(G_inv)')
    
    plt.tight_layout()
    plt.savefig('rhvae_sprites_precise_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Precise analysis saved to rhvae_sprites_precise_analysis.png")

def create_ultra_precise_heatmaps(model, latent_points, centroids, n_clusters):
    """Create ultra-precise metric heatmaps."""
    print("🔥 Creating ultra-precise metric heatmaps...")
    
    # Use first two dimensions for 2D visualization
    x_min, x_max = latent_points[:, 0].min(), latent_points[:, 0].max()
    y_min, y_max = latent_points[:, 1].min(), latent_points[:, 1].max()
    
    # Create ultra-fine grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # For points beyond 2D, use zeros for other dimensions
    full_grid_points = np.zeros((len(grid_points), latent_points.shape[1]))
    full_grid_points[:, :2] = grid_points
    
    # Compute metric at grid points
    grid_metric_dets, grid_metric_traces, grid_metric_conditions, _ = compute_precise_metric_at_points(model, full_grid_points)
    grid_metric_dets = grid_metric_dets.reshape(xx.shape)
    grid_metric_traces = grid_metric_traces.reshape(xx.shape)
    grid_metric_conditions = grid_metric_conditions.reshape(xx.shape)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Ultra-precise metric determinant heatmap
    im1 = axes[0, 0].imshow(grid_metric_dets, extent=[x_min, x_max, y_min, y_max], 
                              origin='lower', cmap='viridis', aspect='auto')
    axes[0, 0].scatter(latent_points[:, 0], latent_points[:, 1], c='white', s=2, alpha=0.2)
    axes[0, 0].scatter(centroids[:, 0], centroids[:, 1], c='red', s=60, marker='*', 
                        edgecolors='black', linewidth=1)
    axes[0, 0].set_title(f'Ultra-Precise Metric Determinant\n(150x150 Grid, {n_clusters} Centroids)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Latent Dim 1')
    axes[0, 0].set_ylabel('Latent Dim 2')
    plt.colorbar(im1, ax=axes[0, 0], label='det(G_inv)')
    
    # Plot 2: Ultra-precise metric trace heatmap
    im2 = axes[0, 1].imshow(grid_metric_traces, extent=[x_min, x_max, y_min, y_max], 
                              origin='lower', cmap='plasma', aspect='auto')
    axes[0, 1].scatter(latent_points[:, 0], latent_points[:, 1], c='white', s=2, alpha=0.2)
    axes[0, 1].scatter(centroids[:, 0], centroids[:, 1], c='red', s=60, marker='*', 
                        edgecolors='black', linewidth=1)
    axes[0, 1].set_title(f'Ultra-Precise Metric Trace\n(150x150 Grid, {n_clusters} Centroids)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Latent Dim 1')
    axes[0, 1].set_ylabel('Latent Dim 2')
    plt.colorbar(im2, ax=axes[0, 1], label='trace(G_inv)')
    
    # Plot 3: Ultra-precise condition number heatmap
    im3 = axes[1, 0].imshow(grid_metric_conditions, extent=[x_min, x_max, y_min, y_max], 
                              origin='lower', cmap='RdYlBu_r', aspect='auto')
    axes[1, 0].scatter(latent_points[:, 0], latent_points[:, 1], c='white', s=2, alpha=0.2)
    axes[1, 0].scatter(centroids[:, 0], centroids[:, 1], c='red', s=60, marker='*', 
                        edgecolors='black', linewidth=1)
    axes[1, 0].set_title(f'Ultra-Precise Condition Number\n(150x150 Grid, {n_clusters} Centroids)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Latent Dim 1')
    axes[1, 0].set_ylabel('Latent Dim 2')
    plt.colorbar(im3, ax=axes[1, 0], label='Condition Number')
    
    # Plot 4: Centroid positions with metric values
    centroid_metric_dets, _, _, _ = compute_precise_metric_at_points(model, centroids)
    scatter = axes[1, 1].scatter(centroids[:, 0], centroids[:, 1], 
                                 c=centroid_metric_dets, s=100, cmap='viridis', 
                                 edgecolors='black', linewidth=1)
    axes[1, 1].set_title(f'Centroid Positions with Metric Determinant\n({n_clusters} Centroids)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Latent Dim 1')
    axes[1, 1].set_ylabel('Latent Dim 2')
    plt.colorbar(scatter, ax=axes[1, 1], label='det(G_inv) at Centroids')
    
    plt.tight_layout()
    plt.savefig('rhvae_sprites_ultra_precise_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Ultra-precise heatmaps saved to rhvae_sprites_ultra_precise_heatmaps.png")

def main():
    """Main precise analysis function."""
    print("🎯 RHVAE Sprites Precise Analysis")
    print("=" * 40)
    
    # Load model and data
    model = load_trained_model()
    data = load_sprites_data()
    
    # Encode data to latent space
    latent_points = encode_data_to_latent(model, data)
    print(f"📊 Encoded {len(latent_points)} points to {latent_points.shape[1]}-dimensional latent space")
    
    # Compute precise centroids with optimal number
    centroids, cluster_labels, n_clusters = compute_precise_centroids(latent_points, n_clusters=16)
    
    # Compute precise metric at all points
    metric_dets, metric_traces, metric_conditions, metric_eigenvals = compute_precise_metric_at_points(model, latent_points)
    
    # Create precise visualizations
    create_precise_visualization(model, latent_points, centroids, cluster_labels, 
                                metric_dets, metric_traces, metric_conditions, n_clusters)
    create_ultra_precise_heatmaps(model, latent_points, centroids, n_clusters)
    
    # Print precise summary statistics
    print("\n📊 Precise Summary Statistics:")
    print(f"   - Latent space dimension: {latent_points.shape[1]}")
    print(f"   - Number of data points: {len(latent_points)}")
    print(f"   - Number of centroids: {n_clusters}")
    print(f"   - Metric determinant range: [{metric_dets.min():.2e}, {metric_dets.max():.2e}]")
    print(f"   - Metric trace range: [{metric_traces.min():.2f}, {metric_traces.max():.2f}]")
    print(f"   - Condition number range: [{metric_conditions.min():.2f}, {metric_conditions.max():.2f}]")
    print(f"   - Coefficient of variation (det): {metric_dets.std()/metric_dets.mean():.3f}")
    print(f"   - Cluster sizes: {np.bincount(cluster_labels)}")
    
    print("\n✅ Precise analysis completed!")
    print("📁 Files created:")
    print("   - rhvae_sprites_precise_analysis.png")
    print("   - rhvae_sprites_ultra_precise_heatmaps.png")

if __name__ == "__main__":
    main() 