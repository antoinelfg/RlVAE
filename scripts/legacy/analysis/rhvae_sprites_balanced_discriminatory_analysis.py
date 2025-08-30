#!/usr/bin/env python3
"""
RHVAE Sprites Balanced Discriminatory Analysis
==============================================

Analyze the balanced discriminatory RHVAE model to show smooth geometry
with discriminatory behavior.
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
    """Load the balanced discriminatory RHVAE model."""
    model_dir = "rhvae_sprites_balanced_discriminatory/RHVAE_training_*/final_model"
    import glob
    model_paths = glob.glob(model_dir)
    if not model_paths:
        raise FileNotFoundError("No trained model found!")
    
    latest_model = sorted(model_paths)[-1]
    trained_model = AutoModel.load_from_folder(latest_model)
    trained_model = trained_model.to(device)
    trained_model.eval()
    print(f"✅ Loaded balanced discriminatory model from {latest_model}")
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

def compute_balanced_metric_at_points(model, points):
    """Compute metric tensor with balanced analysis."""
    print("📏 Computing balanced metric tensors...")
    metric_determinants = []
    metric_traces = []
    metric_condition_numbers = []
    metric_distances = []  # Distance to nearest centroid
    
    with torch.no_grad():
        for i, point in enumerate(points):
            try:
                z = torch.tensor(point, dtype=torch.float32).unsqueeze(0).to(device)
                G_inv = model.G_inv(z)
                G_inv = G_inv.squeeze(0)
                
                det_G_inv = torch.det(G_inv)
                trace_G_inv = torch.trace(G_inv)
                
                # Compute eigenvalues and condition number
                eigenvals = torch.linalg.eigvals(G_inv)
                eigenvals_real = eigenvals.real
                condition_number = torch.max(eigenvals_real) / torch.min(eigenvals_real)
                
                # Compute distance to nearest centroid (for discrimination analysis)
                if hasattr(model, 'centroids_tens'):
                    centroids_tens = model.centroids_tens.to(z.device)
                    distances = torch.norm(centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1)
                    min_distance = torch.min(distances)
                else:
                    min_distance = torch.tensor(0.0, device=z.device)
                
                metric_determinants.append(det_G_inv.item())
                metric_traces.append(trace_G_inv.item())
                metric_condition_numbers.append(condition_number.item())
                metric_distances.append(min_distance.item())
                
                if i % 100 == 0:
                    print(f"  Processed {i}/{len(points)} points...")
                
            except Exception as e:
                print(f"⚠️ Failed to compute metric for point {i}: {e}")
                continue
    
    return (np.array(metric_determinants), np.array(metric_traces), 
            np.array(metric_condition_numbers), np.array(metric_distances))

def create_balanced_visualization(model, latent_points, metric_dets, metric_traces, metric_conditions, metric_distances):
    """Create visualization showing balanced discriminatory behavior."""
    print("🎨 Creating balanced discriminatory visualization...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. TSNE visualization
    ax1 = plt.subplot(2, 3, 1)
    print("  Computing TSNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_points)//4))
    latent_tsne = tsne.fit_transform(latent_points)
    
    scatter = ax1.scatter(latent_tsne[:, 0], latent_tsne[:, 1], 
                         c=metric_distances, cmap='viridis', alpha=0.6, s=20)
    ax1.set_title('Latent Space (TSNE) Colored by Distance to Centroids\n(Balanced Discriminatory Model)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('TSNE 1')
    ax1.set_ylabel('TSNE 2')
    plt.colorbar(scatter, ax=ax1, label='Distance to Nearest Centroid')
    
    # 2. Metric determinant vs distance
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(metric_distances, metric_dets, alpha=0.6, s=10)
    ax2.set_xlabel('Distance to Nearest Centroid')
    ax2.set_ylabel('det(G_inv)')
    ax2.set_title('Balanced Metric Determinant vs Distance\n(Smooth Discrimination)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    
    # 3. Metric trace vs distance
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(metric_distances, metric_traces, alpha=0.6, s=10)
    ax3.set_xlabel('Distance to Nearest Centroid')
    ax3.set_ylabel('trace(G_inv)')
    ax3.set_title('Balanced Metric Trace vs Distance\n(Smooth Discrimination)', fontsize=14, fontweight='bold')
    
    # 4. High-resolution metric determinant heatmap
    ax4 = plt.subplot(2, 3, 4)
    print("  Computing balanced metric heatmap...")
    x_min, x_max = latent_points[:, 0].min(), latent_points[:, 0].max()
    y_min, y_max = latent_points[:, 1].min(), latent_points[:, 1].max()
    
    # Create grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # For points beyond 2D, use zeros for other dimensions
    full_grid_points = np.zeros((len(grid_points), latent_points.shape[1]))
    full_grid_points[:, :2] = grid_points
    
    # Compute metric at grid points
    grid_metric_dets, _, _, _ = compute_balanced_metric_at_points(model, full_grid_points)
    grid_metric_dets = grid_metric_dets.reshape(xx.shape)
    
    im = ax4.imshow(grid_metric_dets, extent=[x_min, x_max, y_min, y_max], 
                    origin='lower', cmap='viridis', aspect='auto')
    ax4.scatter(latent_points[:, 0], latent_points[:, 1], c='white', s=3, alpha=0.3)
    ax4.set_title('Balanced Metric Determinant\n(Smooth Variation)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Latent Dim 1')
    ax4.set_ylabel('Latent Dim 2')
    plt.colorbar(im, ax=ax4, label='det(G_inv)')
    
    # 5. Metric determinant distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(metric_dets, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax5.set_title('Balanced Metric Determinant Distribution\n(Smooth Peaks)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('det(G_inv)')
    ax5.set_ylabel('Count')
    ax5.set_yscale('log')
    
    # 6. Distance distribution
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(metric_distances, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax6.set_title('Distance to Nearest Centroid Distribution', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Distance')
    ax6.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('rhvae_sprites_balanced_discriminatory_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Balanced discriminatory analysis saved to rhvae_sprites_balanced_discriminatory_analysis.png")

def create_smooth_heatmaps(model, latent_points):
    """Create heatmaps showing smooth discriminatory behavior."""
    print("🌊 Creating smooth discriminatory heatmaps...")
    
    # Use first two dimensions for 2D visualization
    x_min, x_max = latent_points[:, 0].min(), latent_points[:, 0].max()
    y_min, y_max = latent_points[:, 1].min(), latent_points[:, 1].max()
    
    # Create fine grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # For points beyond 2D, use zeros for other dimensions
    full_grid_points = np.zeros((len(grid_points), latent_points.shape[1]))
    full_grid_points[:, :2] = grid_points
    
    # Compute metric at grid points
    grid_metric_dets, grid_metric_traces, grid_metric_conditions, grid_distances = compute_balanced_metric_at_points(model, full_grid_points)
    grid_metric_dets = grid_metric_dets.reshape(xx.shape)
    grid_metric_traces = grid_metric_traces.reshape(xx.shape)
    grid_metric_conditions = grid_metric_conditions.reshape(xx.shape)
    grid_distances = grid_distances.reshape(xx.shape)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Smooth metric determinant heatmap
    im1 = axes[0, 0].imshow(grid_metric_dets, extent=[x_min, x_max, y_min, y_max], 
                              origin='lower', cmap='viridis', aspect='auto')
    axes[0, 0].scatter(latent_points[:, 0], latent_points[:, 1], c='white', s=2, alpha=0.2)
    axes[0, 0].set_title('Smooth Balanced Metric Determinant\n(150x150 Grid)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Latent Dim 1')
    axes[0, 0].set_ylabel('Latent Dim 2')
    plt.colorbar(im1, ax=axes[0, 0], label='det(G_inv)')
    
    # Plot 2: Smooth metric trace heatmap
    im2 = axes[0, 1].imshow(grid_metric_traces, extent=[x_min, x_max, y_min, y_max], 
                              origin='lower', cmap='plasma', aspect='auto')
    axes[0, 1].scatter(latent_points[:, 0], latent_points[:, 1], c='white', s=2, alpha=0.2)
    axes[0, 1].set_title('Smooth Balanced Metric Trace\n(150x150 Grid)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Latent Dim 1')
    axes[0, 1].set_ylabel('Latent Dim 2')
    plt.colorbar(im2, ax=axes[0, 1], label='trace(G_inv)')
    
    # Plot 3: Distance heatmap
    im3 = axes[1, 0].imshow(grid_distances, extent=[x_min, x_max, y_min, y_max], 
                              origin='lower', cmap='RdYlBu_r', aspect='auto')
    axes[1, 0].scatter(latent_points[:, 0], latent_points[:, 1], c='white', s=2, alpha=0.2)
    axes[1, 0].set_title('Distance to Nearest Centroid\n(150x150 Grid)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Latent Dim 1')
    axes[1, 0].set_ylabel('Latent Dim 2')
    plt.colorbar(im3, ax=axes[1, 0], label='Distance')
    
    # Plot 4: Condition number heatmap
    im4 = axes[1, 1].imshow(grid_metric_conditions, extent=[x_min, x_max, y_min, y_max], 
                              origin='lower', cmap='RdYlBu_r', aspect='auto')
    axes[1, 1].scatter(latent_points[:, 0], latent_points[:, 1], c='white', s=2, alpha=0.2)
    axes[1, 1].set_title('Smooth Balanced Condition Number\n(150x150 Grid)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Latent Dim 1')
    axes[1, 1].set_ylabel('Latent Dim 2')
    plt.colorbar(im4, ax=axes[1, 1], label='Condition Number')
    
    plt.tight_layout()
    plt.savefig('rhvae_sprites_balanced_discriminatory_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Smooth balanced discriminatory heatmaps saved to rhvae_sprites_balanced_discriminatory_heatmaps.png")

def create_comparison_visualization(model, latent_points, metric_dets, metric_traces, metric_conditions, metric_distances):
    """Create comparison visualization showing geometry vs discrimination."""
    print("🔄 Creating geometry vs discrimination comparison...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. TSNE with smooth coloring
    ax1 = axes[0, 0]
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_points)//4))
    latent_tsne = tsne.fit_transform(latent_points)
    
    scatter = ax1.scatter(latent_tsne[:, 0], latent_tsne[:, 1], 
                         c=metric_dets, cmap='viridis', alpha=0.6, s=20)
    ax1.set_title('TSNE Colored by Metric Determinant\n(Smooth Geometry)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('TSNE 1')
    ax1.set_ylabel('TSNE 2')
    plt.colorbar(scatter, ax=ax1, label='det(G_inv)')
    
    # 2. Metric determinant vs distance (smooth relationship)
    ax2 = axes[0, 1]
    ax2.scatter(metric_distances, metric_dets, alpha=0.6, s=10, c='blue')
    ax2.set_xlabel('Distance to Nearest Centroid')
    ax2.set_ylabel('det(G_inv)')
    ax2.set_title('Smooth Discrimination\n(Geometry Preserved)', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    
    # 3. Condition number distribution (smooth)
    ax3 = axes[0, 2]
    ax3.hist(metric_conditions, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.set_title('Smooth Condition Number Distribution\n(Stable Geometry)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Condition Number')
    ax3.set_ylabel('Count')
    
    # 4. High-resolution smooth metric heatmap
    ax4 = axes[1, 0]
    x_min, x_max = latent_points[:, 0].min(), latent_points[:, 0].max()
    y_min, y_max = latent_points[:, 1].min(), latent_points[:, 1].max()
    
    # Create grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # For points beyond 2D, use zeros for other dimensions
    full_grid_points = np.zeros((len(grid_points), latent_points.shape[1]))
    full_grid_points[:, :2] = grid_points
    
    # Compute metric at grid points
    grid_metric_dets, _, _, _ = compute_balanced_metric_at_points(model, full_grid_points)
    grid_metric_dets = grid_metric_dets.reshape(xx.shape)
    
    im = ax4.imshow(grid_metric_dets, extent=[x_min, x_max, y_min, y_max], 
                    origin='lower', cmap='viridis', aspect='auto')
    ax4.scatter(latent_points[:, 0], latent_points[:, 1], c='white', s=3, alpha=0.3)
    ax4.set_title('Smooth Metric Determinant\n(Geometry + Discrimination)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Latent Dim 1')
    ax4.set_ylabel('Latent Dim 2')
    plt.colorbar(im, ax=ax4, label='det(G_inv)')
    
    # 5. Metric trace vs determinant (smooth relationship)
    ax5 = axes[1, 1]
    ax5.scatter(metric_dets, metric_traces, alpha=0.6, s=10, c='red')
    ax5.set_xlabel('det(G_inv)')
    ax5.set_ylabel('trace(G_inv)')
    ax5.set_title('Smooth Metric Properties\n(Consistent Geometry)', fontsize=12, fontweight='bold')
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    
    # 6. Distance vs condition number (smooth relationship)
    ax6 = axes[1, 2]
    ax6.scatter(metric_distances, metric_conditions, alpha=0.6, s=10, c='purple')
    ax6.set_xlabel('Distance to Nearest Centroid')
    ax6.set_ylabel('Condition Number')
    ax6.set_title('Smooth Geometry vs Distance\n(Balanced Approach)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('rhvae_sprites_balanced_geometry_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Geometry comparison saved to rhvae_sprites_balanced_geometry_comparison.png")

def main():
    """Main balanced discriminatory analysis function."""
    print("🎯 RHVAE Sprites Balanced Discriminatory Analysis")
    print("=" * 50)
    
    # Load model and data
    model = load_trained_model()
    data = load_sprites_data()
    
    # Encode data to latent space
    latent_points = encode_data_to_latent(model, data)
    print(f"📊 Encoded {len(latent_points)} points to {latent_points.shape[1]}-dimensional latent space")
    
    # Compute balanced metric at all points
    metric_dets, metric_traces, metric_conditions, metric_distances = compute_balanced_metric_at_points(model, latent_points)
    
    # Create balanced discriminatory visualizations
    create_balanced_visualization(model, latent_points, metric_dets, metric_traces, metric_conditions, metric_distances)
    create_smooth_heatmaps(model, latent_points)
    create_comparison_visualization(model, latent_points, metric_dets, metric_traces, metric_conditions, metric_distances)
    
    # Print balanced summary statistics
    print("\n📊 Balanced Discriminatory Summary Statistics:")
    print(f"   - Latent space dimension: {latent_points.shape[1]}")
    print(f"   - Number of data points: {len(latent_points)}")
    print(f"   - Metric determinant range: [{metric_dets.min():.2e}, {metric_dets.max():.2e}]")
    print(f"   - Metric trace range: [{metric_traces.min():.2f}, {metric_traces.max():.2f}]")
    print(f"   - Condition number range: [{metric_conditions.min():.2f}, {metric_conditions.max():.2f}]")
    print(f"   - Distance range: [{metric_distances.min():.2f}, {metric_distances.max():.2f}]")
    print(f"   - Coefficient of variation (det): {metric_dets.std()/metric_dets.mean():.3f}")
    
    # Analyze balanced behavior
    print("\n🎯 Balanced Behavior Analysis:")
    print(f"   - Points close to centroids (dist < 0.5): {np.sum(metric_distances < 0.5)}")
    print(f"   - Points far from centroids (dist > 2.0): {np.sum(metric_distances > 2.0)}")
    print(f"   - Smoothness ratio (far/near determinant): {np.mean(metric_dets[metric_distances > 2.0]) / np.mean(metric_dets[metric_distances < 0.5]):.2f}")
    print(f"   - Geometry stability (condition number CV): {metric_conditions.std()/metric_conditions.mean():.3f}")
    
    print("\n✅ Balanced discriminatory analysis completed!")
    print("📁 Files created:")
    print("   - rhvae_sprites_balanced_discriminatory_analysis.png")
    print("   - rhvae_sprites_balanced_discriminatory_heatmaps.png")
    print("   - rhvae_sprites_balanced_geometry_comparison.png")

if __name__ == "__main__":
    main() 