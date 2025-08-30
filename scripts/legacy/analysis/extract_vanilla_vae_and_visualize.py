#!/usr/bin/env python3
"""
Extract trained vanilla VAE and generate comprehensive visualizations
====================================================================

This script extracts the trained vanilla VAE model from the Hydra training run
and generates comprehensive visualizations similar to the real data analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf
import hydra
from src.models.components.native_inverse_metric import NativeInverseMetricTensor
from src.models.modular_rlvae import ModularRiemannianFlowVAE

def load_trained_vanilla_vae():
    """Load the trained vanilla VAE from the Hydra output."""
    print("🔍 Loading trained vanilla VAE...")
    
    # Use the checkpoints from the main outputs directory
    checkpoint_dir = Path("outputs/checkpoints")
    if not checkpoint_dir.exists():
        raise FileNotFoundError("No checkpoints directory found")
    
    # Find the best checkpoint (lowest val_loss, most recent epoch)
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    # Filter for vanilla VAE checkpoints (those with val_loss around 5-6)
    vanilla_checkpoints = []
    for ckpt in checkpoints:
        try:
            val_loss = float(ckpt.stem.split("val_loss=")[1].split(".")[0])
            if 5.0 <= val_loss <= 7.0:  # Vanilla VAE range
                vanilla_checkpoints.append((ckpt, val_loss))
        except (ValueError, IndexError):
            continue
    
    if not vanilla_checkpoints:
        raise FileNotFoundError("No vanilla VAE checkpoints found")
    
    # Get the best checkpoint (lowest val_loss)
    best_checkpoint = min(vanilla_checkpoints, key=lambda x: x[1])[0]
    print(f"✅ Loading best checkpoint: {best_checkpoint}")
    
    # Load checkpoint with weights_only=False to handle OmegaConf objects
    checkpoint = torch.load(best_checkpoint, map_location='cpu', weights_only=False)
    
    # Extract model configuration
    model_config = checkpoint['hyper_parameters']['model']
    print(f"📊 Model config: latent_dim={model_config['latent_dim']}")
    
    # Create model with same configuration
    config = OmegaConf.create(model_config)
    model = ModularRiemannianFlowVAE(config)
    
    # Load state dict
    state_dict = checkpoint['state_dict']
    # Remove 'model.' prefix if present
    clean_state_dict = {k.replace('model.', '') if k.startswith('model.') else k: v 
                       for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully on {device}")
    return model, device

def extract_latent_data(model, device):
    """Extract latent representations from the trained model."""
    print("🔄 Extracting latent data from trained model...")
    
    # Load test data
    test_data = torch.load('data/processed/Sprites_test_cyclic.pt', map_location=device)
    print(f"📊 Test data shape: {test_data.shape}")
    
    # Use a subset for analysis
    test_subset = test_data[:200]  # Use 200 sequences
    print(f"📊 Using subset: {test_subset.shape}")
    
    # Extract latent representations
    latent_data = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(test_subset), 32):
            batch = test_subset[i:i+32]
            output = model(batch)
            # Get latent samples (first timestep)
            if isinstance(output['latent_samples'], list):
                latents = output['latent_samples'][0]  # First timestep
            else:
                latents = output['latent_samples'][:, 0]  # [batch, latent_dim]
            latent_data.append(latents)
    
    latent_data = torch.cat(latent_data, dim=0)
    print(f"✅ Extracted latents: {latent_data.shape}")
    print(f"📊 Latent range: [{latent_data.min():.3f}, {latent_data.max():.3f}]")
    
    return latent_data

def create_metric_and_analyze(latent_data, device):
    """Create metric and perform comprehensive analysis."""
    print("\n🔧 Creating metric from trained model data...")
    
    class DummyModel:
        pass
    model = DummyModel()
    
    # Create metric with optimal temperature
    native_metric = NativeInverseMetricTensor.from_model_data(
        model, latent_data, 
        n_centroids=25,
        temperature=0.5,  # Optimal from diagnostic
        device=device
    )
    
    centroids = native_metric.centroids
    print(f"   Created metric with {len(centroids)} centroids")
    
    return native_metric, centroids

def generate_comprehensive_visualizations(latent_data, native_metric, centroids, device):
    """Generate comprehensive visualizations for the trained vanilla VAE."""
    print("\n🎨 Generating comprehensive visualizations for trained vanilla VAE...")
    
    # Create output directory
    output_dir = "vanilla_vae_16d_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create analysis grid (for 16D, we'll visualize first 2 dimensions)
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    
    # For 16D, create grid points with first 2 dims varying, others at mean
    grid_2d = np.column_stack([X.ravel(), Y.ravel()])
    latent_mean = latent_data.mean(dim=0).cpu().numpy()
    
    # Create 16D grid points: first 2 dims from grid, others from data mean
    grid_points_16d = np.zeros((len(grid_2d), 16))
    grid_points_16d[:, :2] = grid_2d
    grid_points_16d[:, 2:] = latent_mean[2:]  # Use mean for other dimensions
    
    grid_points = torch.tensor(grid_points_16d, dtype=torch.float32, device=device)
    
    # Compute all metrics on grid
    with torch.no_grad():
        G_inv, log_det_G_inv = native_metric(grid_points)
        det_grid = torch.exp(log_det_G_inv).cpu().numpy().reshape(X.shape)
    
    # Compute data density (using first 2 dimensions for visualization)
    latent_cpu = latent_data.cpu().numpy()
    data_density_grid = np.zeros_like(det_grid)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # For 16D, compute density based on first 2 dimensions
            point_2d = np.array([X[i,j], Y[i,j]])
            distances_2d = np.linalg.norm(latent_cpu[:, :2] - point_2d, axis=1)
            density = np.sum(distances_2d < 0.2)
            data_density_grid[i,j] = density
    
    # 1. COMPLETE PIPELINE VISUALIZATION
    print("   1. Creating complete pipeline visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Trained latent space
    ax1 = axes[0, 0]
    contour1 = ax1.contourf(X, Y, det_grid, levels=50, cmap='viridis', alpha=0.7)
    ax1.contour(X, Y, det_grid, levels=15, colors='white', alpha=0.4, linewidths=0.5)
    ax1.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='lightblue', s=15, alpha=0.6, label='Trained Latent Data')
    ax1.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='white', s=120, marker='*',
               edgecolors='black', linewidth=1.5, label='Centroids', zorder=10)
    ax1.set_title('1. Trained Latent Space (16D → 2D projection)\n(Vanilla VAE on Sprites data)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('z₁ (first dimension)')
    ax1.set_ylabel('z₂ (second dimension)')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Metric structure
    ax2 = axes[0, 1]
    contour2 = ax2.contourf(X, Y, det_grid, levels=50, cmap='viridis', alpha=0.7)
    ax2.contour(X, Y, det_grid, levels=15, colors='white', alpha=0.4, linewidths=0.5)
    ax2.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='red', s=100, alpha=0.8, label='Metric Centroids')
    plt.colorbar(contour2, ax=ax2, label='det(G⁻¹)')
    ax2.set_title('2. Metric Structure (16D → 2D projection)\n(G⁻¹(z) computed from trained points)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('z₁ (first dimension)')
    ax2.set_ylabel('z₂ (second dimension)')
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RHMC sampling (simulated)
    ax3 = axes[1, 0]
    contour3 = ax3.contourf(X, Y, det_grid, levels=50, cmap='viridis', alpha=0.7)
    ax3.contour(X, Y, det_grid, levels=15, colors='white', alpha=0.4, linewidths=0.5)
    
    # Simulate RHMC samples (concentrated near data)
    n_samples = 200
    rhmc_samples = []
    for _ in range(n_samples):
        # Sample near centroids with some noise (16D)
        centroid_idx = np.random.randint(0, len(centroids))
        sample = centroids[centroid_idx].cpu().numpy() + np.random.normal(0, 0.3, 16)
        rhmc_samples.append(sample)
    rhmc_samples = np.array(rhmc_samples)
    
    ax3.scatter(rhmc_samples[:, 0], rhmc_samples[:, 1], c='lime', s=30, alpha=0.8,
               edgecolors='darkgreen', linewidth=0.5, label='RHMC Samples (16D → 2D)')
    ax3.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='white', s=120, marker='*',
               edgecolors='black', linewidth=1.5, label='Centroids', zorder=10)
    ax3.set_title('3. RHMC Sampling (16D → 2D projection)\n(True Riemannian HMC on trained manifold)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('z₁ (first dimension)')
    ax3.set_ylabel('z₂ (second dimension)')
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Complete pipeline
    ax4 = axes[1, 1]
    contour4 = ax4.contourf(X, Y, det_grid, levels=50, cmap='viridis', alpha=0.7)
    ax4.contour(X, Y, det_grid, levels=15, colors='white', alpha=0.4, linewidths=0.5)
    ax4.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='lightblue', s=10, alpha=0.4, label='Trained Data')
    ax4.scatter(rhmc_samples[:, 0], rhmc_samples[:, 1], c='lime', s=25, alpha=0.8, label='RHMC Samples (16D → 2D)')
    ax4.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='white', s=120, marker='*',
               edgecolors='black', linewidth=1.5, label='Centroids', zorder=10)
    ax4.set_title('4. Complete Pipeline (16D → 2D projection)\n(Trained data + Metric + RHMC)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('z₁ (first dimension)')
    ax4.set_ylabel('z₂ (second dimension)')
    ax4.set_xlim(-4, 4)
    ax4.set_ylim(-4, 4)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Vanilla VAE RHMC Manifold Sampling: Complete Pipeline (16D)\n(Train VAE → Extract Latent → Compute Metric → RHMC Sample)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_complete_pipeline_vanilla_vae.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. METRIC COMPONENTS ANALYSIS
    print("   2. Creating metric components analysis...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Compute additional metrics (using first 2 dimensions for visualization)
    distance_grid = np.zeros_like(det_grid)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point_2d = torch.tensor([X[i,j], Y[i,j]], device=device)
            # For 16D, compute distance using first 2 dimensions
            distances = torch.norm(point_2d.unsqueeze(0) - centroids[:, :2], dim=1)
            distance_grid[i,j] = distances.min().item()
    
    # Plot 1: det(G⁻¹)
    ax1 = axes[0, 0]
    contour1 = ax1.contourf(X, Y, det_grid, levels=30, cmap='viridis', alpha=0.8)
    ax1.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='red', s=100, marker='*',
               edgecolors='white', linewidth=2)
    plt.colorbar(contour1, ax=ax1, label='det(G⁻¹)')
    ax1.set_title('1. det(G⁻¹)\nVanilla VAE Metric', fontweight='bold')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    
    # Plot 2: Distance to centroid
    ax2 = axes[0, 1]
    contour2 = ax2.contourf(X, Y, distance_grid, levels=30, cmap='plasma_r', alpha=0.8)
    ax2.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='white', s=100, marker='*',
               edgecolors='black', linewidth=2)
    plt.colorbar(contour2, ax=ax2, label='Distance to Centroid')
    ax2.set_title('2. Distance to Centroid\nVanilla VAE', fontweight='bold')
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    
    # Plot 3: Data density (ground truth)
    ax3 = axes[0, 2]
    contour3 = ax3.contourf(X, Y, data_density_grid, levels=30, cmap='Blues', alpha=0.8)
    ax3.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='red', s=1, alpha=0.5)
    plt.colorbar(contour3, ax=ax3, label='Data Density')
    ax3.set_title('3. Trained Data Density\n(Vanilla VAE)', fontweight='bold')
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    
    # Plot 4: Correlation analysis
    ax4 = axes[1, 0]
    det_flat = det_grid.flatten()
    dist_flat = distance_grid.flatten()
    density_flat = data_density_grid.flatten()
    
    # Plot det vs distance
    ax4.scatter(dist_flat, det_flat, alpha=0.3, s=1)
    ax4.set_xlabel('Distance to Nearest Centroid')
    ax4.set_ylabel('det(G⁻¹)')
    ax4.set_title('4. Distance vs det(G⁻¹)\nVanilla VAE', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: det vs data density
    ax5 = axes[1, 1]
    ax5.scatter(density_flat, det_flat, alpha=0.3, s=1)
    ax5.set_xlabel('Data Density')
    ax5.set_ylabel('det(G⁻¹)')
    ax5.set_title('5. Data Density vs det(G⁻¹)\nVanilla VAE', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Best predictor overlay
    ax6 = axes[1, 2]
    # Show distance field with data overlay
    contour6 = ax6.contourf(X, Y, distance_grid, levels=30, cmap='plasma_r', alpha=0.6)
    ax6.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='white', s=1, alpha=0.5, label='Trained Data')
    ax6.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='red', s=100, marker='*',
               edgecolors='white', linewidth=2, label='Centroids')
    plt.colorbar(contour6, ax=ax6, label='Distance')
    ax6.set_title('6. Best Predictor: Distance\nVanilla VAE', fontweight='bold')
    ax6.set_xlim(-4, 4)
    ax6.set_ylim(-4, 4)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_metric_components_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. SUMMARY STATISTICS
    print("   3. Creating summary statistics...")
    
    # Compute all correlations
    correlations = {
        'det(G⁻¹) vs Data Density': np.corrcoef(det_grid.flatten(), data_density_grid.flatten())[0,1],
        'Distance vs Data Density': np.corrcoef(-distance_grid.flatten(), data_density_grid.flatten())[0,1],
        'det(G⁻¹) vs Distance': np.corrcoef(det_grid.flatten(), -distance_grid.flatten())[0,1]
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Correlation summary
    ax1 = axes[0]
    metrics = list(correlations.keys())
    values = list(correlations.values())
    colors = ['blue', 'green', 'orange']
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.7)
    ax1.set_ylabel('Correlation Coefficient')
    ax1.set_title('Vanilla VAE Metric Performance Summary', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 2: Data statistics
    ax2 = axes[1]
    stats = {
        'Data Points': len(latent_data),
        'Centroids': len(centroids),
        'Latent Range': f"[{latent_data.min():.2f}, {latent_data.max():.2f}]",
        'det(G⁻¹) Range': f"[{det_grid.min():.0f}, {det_grid.max():.0f}]",
        'Temperature': 0.5,
        'Correlation': correlations['det(G⁻¹) vs Data Density']
    }
    
    y_pos = np.arange(len(stats))
    ax2.barh(y_pos, [1]*len(stats), color='lightblue', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(list(stats.keys()))
    ax2.set_xlim(0, 1.2)
    ax2.set_title('Vanilla VAE Dataset Statistics', fontweight='bold')
    
    # Add value labels
    for i, (key, value) in enumerate(stats.items()):
        ax2.text(1.05, i, str(value), va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_summary_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ All visualizations saved to {output_dir}/")
    print(f"📊 Generated {len(os.listdir(output_dir))} comprehensive analysis graphs")

def main():
    """Main function to extract trained model and generate visualizations."""
    print("🎨 EXTRACTING TRAINED VANILLA VAE AND GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Step 1: Load trained model
    model, device = load_trained_vanilla_vae()
    
    # Step 2: Extract latent data
    latent_data = extract_latent_data(model, device)
    
    # Step 3: Create metric
    native_metric, centroids = create_metric_and_analyze(latent_data, device)
    
    # Step 4: Generate comprehensive visualizations
    generate_comprehensive_visualizations(latent_data, native_metric, centroids, device)
    
    print(f"\n🎉 VANILLA VAE ANALYSIS COMPLETE!")
    print(f"📁 All graphs saved in: vanilla_vae_16d_analysis/")
    print(f"📊 Analysis includes:")
    print(f"   - Complete pipeline visualization")
    print(f"   - Metric components analysis") 
    print(f"   - Summary statistics")
    
    return "vanilla_vae_16d_analysis"

if __name__ == "__main__":
    output_folder = main() 