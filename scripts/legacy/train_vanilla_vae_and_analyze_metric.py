#!/usr/bin/env python3
"""
Train Vanilla VAE and Analyze Metric with retrieveG
==================================================

Train a vanilla VAE using Hydra on real Sprites data, then use retrieveG 
to perform comprehensive metric analysis with WandB logging.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add the benchmark_VAE to the path
sys.path.append('benchmark_VAE/src')

from pythae.models import VAE, VAEConfig
from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST
import wandb
from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate

def create_rgb_encoder_decoder(input_dim, latent_dim):
    """Create encoder and decoder for RGB data."""
    import torch.nn as nn
    from pythae.models.base_architectures import BaseEncoder, BaseDecoder
    
    class RGBEncoder(BaseEncoder):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            
            # Simple CNN for RGB
            self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)  # 64x64 -> 32x32
            self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)  # 32x32 -> 16x16
            self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)  # 16x16 -> 8x8
            self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)  # 8x8 -> 4x4
            
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(256 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, latent_dim)
            self.fc3 = nn.Linear(512, latent_dim)
            
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.relu(self.conv4(x))
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            
            mu = self.fc2(x)
            log_var = self.fc3(x)
            
            return {"embedding": mu, "log_covariance": log_var}
    
    class RGBDecoder(BaseDecoder):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            
            self.fc1 = nn.Linear(latent_dim, 512)
            self.fc2 = nn.Linear(512, 256 * 4 * 4)
            
            self.unflatten = nn.Unflatten(1, (256, 4, 4))
            self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 4x4 -> 8x8
            self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 8x8 -> 16x16
            self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)    # 16x16 -> 32x32
            self.deconv4 = nn.ConvTranspose2d(32, 3, 4, 2, 1)     # 32x32 -> 64x64
            
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, z):
            x = self.relu(self.fc1(z))
            x = self.relu(self.fc2(x))
            x = self.unflatten(x)
            x = self.relu(self.deconv1(x))
            x = self.relu(self.deconv2(x))
            x = self.relu(self.deconv3(x))
            x = self.sigmoid(self.deconv4(x))
            
            return {"reconstruction": x}
    
    return RGBEncoder(input_dim, latent_dim), RGBDecoder(input_dim, latent_dim)

def train_vanilla_vae(cfg):
    """Train vanilla VAE using Hydra configuration."""
    print("🚀 Training Vanilla VAE with Hydra")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize WandB
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        tags=cfg.wandb.tags,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # Load Sprites data
    print("📂 Loading Sprites data...")
    sprites_data = torch.load('data/processed/Sprites_train_cyclic.pt', map_location=device)
    print(f"   Loaded Sprites: {sprites_data.shape}")
    
    # Resize from 28x28 to 64x64
    if sprites_data.shape[-1] == 28:
        import torch.nn.functional as F
        sprites_data = F.interpolate(sprites_data.view(-1, *sprites_data.shape[2:]), 
                                   size=(64, 64), mode='bilinear', align_corners=False)
        sprites_data = sprites_data.view(sprites_data.shape[0], -1, *sprites_data.shape[1:])
        print(f"   Resized to: {sprites_data.shape}")
    
    # Use subset for training
    sprites_subset = sprites_data[:cfg.max_train_samples//8]  # Account for 8 frames per sprite
    flattened = sprites_subset.view(sprites_subset.shape[0] * sprites_subset.shape[1], *sprites_subset.shape[2:])
    print(f"   Training data: {flattened.shape}")
    
    # Create VAE configuration
    model_config = VAEConfig(
        input_dim=(3, 64, 64),
        latent_dim=cfg.latent_dim,
        beta=cfg.beta
    )
    
    # Create custom encoder and decoder for RGB
    encoder, decoder = create_rgb_encoder_decoder((3, 64, 64), cfg.latent_dim)
    
    # Create VAE model
    model = VAE(
        model_config=model_config,
        encoder=encoder, 
        decoder=decoder
    )
    model.to(device)
    
    print(f"✅ VAE model created")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset wrapper
    class SpritesDataset:
        def __init__(self, data):
            self.data = data
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = SpritesDataset(flattened)
    
    # Training loop
    print(f"\n🎓 Training for {cfg.max_epochs} epochs...")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    model.train()
    
    for epoch in range(cfg.max_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        n_batches = 0
        
        for i in range(0, len(dataset), cfg.batch_size):
            batch = dataset.data[i:i+cfg.batch_size]
            if len(batch) < 2:
                continue
                
            optimizer.zero_grad()
            
            # Forward pass
            inputs = {"data": batch}
            output = model(inputs)
            
            # Compute losses
            recon_loss = output.recon_loss
            kl_loss = output.kl_loss
            total_loss_batch = output.loss
            
            # Backward pass
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            n_batches += 1
        
        # Log metrics
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        avg_recon_loss = total_recon_loss / n_batches if n_batches > 0 else 0
        avg_kl_loss = total_kl_loss / n_batches if n_batches > 0 else 0
        
        wandb.log({
            "epoch": epoch,
            "train/loss": avg_loss,
            "train/recon_loss": avg_recon_loss,
            "train/kl_loss": avg_kl_loss
        })
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{cfg.max_epochs}, Loss: {avg_loss:.6f}, Recon: {avg_recon_loss:.6f}, KL: {avg_kl_loss:.6f}")
    
    print(f"✅ Training completed")
    
    return model, dataset, device

def manual_retrieveG(model, dataset, device, num_centroids=50, T_multiplier=1.0):
    """Manual implementation of retrieveG functionality."""
    print(f"\n🔧 Computing metric with manual retrieveG...")
    
    # Set model to eval mode
    model.eval()
    
    # Get latent representations
    with torch.no_grad():
        latent_data = []
        for i in range(0, len(dataset), 256):
            batch = dataset.data[i:i+256].to(device)
            output = model.encoder(batch)
            latent_data.append(output.embedding)
        
        latent_data = torch.cat(latent_data, dim=0)
        print(f"   Extracted latent data: {latent_data.shape}")
        
        # Compute centroids using k-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_centroids, random_state=42, n_init=10)
        kmeans.fit(latent_data.cpu().numpy())
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
        
        # Compute temperature
        T_is = []
        for i in range(len(centroids)-1):
            mask = torch.tensor([k for k in range(len(centroids)) if k != i])
            dist = torch.norm(centroids[i].unsqueeze(0) - centroids[mask], dim=-1)
            T_i = torch.min(dist, dim=0)[0]
            T_is.append(T_i.item())
        
        T = np.max(T_is) * T_multiplier
        print(f"   Computed temperature: {T:.3f}")
        
        # Create metric matrices for each centroid
        metric_matrices = []
        for i, centroid in enumerate(centroids):
            distances = torch.norm(latent_data - centroid, dim=1)
            closest_indices = torch.argsort(distances)[:100]
            cluster_points = latent_data[closest_indices]
            
            if len(cluster_points) > 1:
                cov_matrix = torch.cov(cluster_points.T)
                cov_matrix += torch.eye(cov_matrix.shape[0], device=device) * 0.01
                try:
                    metric_matrix = torch.linalg.inv(cov_matrix)
                except:
                    metric_matrix = torch.eye(cov_matrix.shape[0], device=device)
            else:
                metric_matrix = torch.eye(latent_data.shape[1], device=device)
            
            metric_matrices.append(metric_matrix)
        
        metric_matrices = torch.stack(metric_matrices)
        
        # Create G function
        def G_sampl(z):
            # z shape: (batch_size, latent_dim)
            batch_size = z.shape[0]
            G = torch.zeros(batch_size, 16, 16, device=device)
            
            for i in range(batch_size):
                z_i = z[i:i+1]  # (1, latent_dim)
                
                # Compute distances to centroids
                distances = torch.norm(z_i.unsqueeze(1) - centroids.unsqueeze(0), dim=2)  # (1, num_centroids)
                
                # Compute weights
                weights = torch.exp(-distances**2 / (T**2))  # (1, num_centroids)
                weights = weights / weights.sum()  # Normalize
                
                # Interpolate metric matrices
                G_i = torch.zeros(16, 16, device=device)
                for j in range(len(centroids)):
                    G_i += weights[0, j] * metric_matrices[j]
                
                # Add regularization
                G_i += torch.eye(16, device=device) * 0.01
                
                G[i] = G_i
            
            return G
        
        print(f"✅ Manual retrieveG completed successfully")
        print(f"   Centroids shape: {centroids.shape}")
        print(f"   Metric matrices shape: {metric_matrices.shape}")
        print(f"   Temperature: {T:.3f}")
        
        return G_sampl, centroids, latent_data

def compute_g_inverse_determinant_grid(G_sampl, device, x_range=(-4, 4), y_range=(-4, 4), n_points=100):
    """Compute G⁻¹ determinant across a grid."""
    print(f"\n🔍 Computing G⁻¹ determinant grid")
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # For 16D, create grid points with first 2 dims varying, others at mean
    grid_2d = np.column_stack([X.ravel(), Y.ravel()])
    
    # Use zeros for other dimensions (simplified)
    grid_points_16d = np.zeros((len(grid_2d), 16))
    grid_points_16d[:, :2] = grid_2d
    
    grid_points = torch.tensor(grid_points_16d, dtype=torch.float32, device=device)
    
    # Compute G(z) for all grid points
    with torch.no_grad():
        if G_sampl is not None:
            # Use the G_sampl function from retrieveG
            G_z = G_sampl(grid_points)
        else:
            # Fallback: use simple metric
            G_z = torch.eye(16, device=device).unsqueeze(0).repeat(len(grid_points), 1, 1)
        
        # Compute G⁻¹ and its determinant
        G_inv = torch.linalg.inv(G_z)
        det_G_inv = torch.linalg.det(G_inv)
        
        # Reshape back to grid
        det_G_inv_grid = det_G_inv.cpu().numpy().reshape(X.shape)
    
    print(f"✅ G⁻¹ determinant computed for {n_points}x{n_points} grid")
    print(f"✅ G⁻¹ determinant range: [{det_G_inv_grid.min():.3e}, {det_G_inv_grid.max():.3e}]")
    
    return X, Y, det_G_inv_grid

def compute_anisotropy_grid(G_sampl, device, x_range=(-4, 4), y_range=(-4, 4), n_points=100):
    """Compute anisotropy (λ₁ - λ₂) across a grid."""
    print(f"\n🔍 Computing anisotropy grid")
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # Create 16D grid points (same as above)
    grid_2d = np.column_stack([X.ravel(), Y.ravel()])
    grid_points_16d = np.zeros((len(grid_2d), 16))
    grid_points_16d[:, :2] = grid_2d
    
    grid_points = torch.tensor(grid_points_16d, dtype=torch.float32, device=device)
    
    # Compute G(z) for all grid points
    with torch.no_grad():
        if G_sampl is not None:
            G_z = G_sampl(grid_points)
        else:
            G_z = torch.eye(16, device=device).unsqueeze(0).repeat(len(grid_points), 1, 1)
        
        # Compute eigenvalues
        eigenvals = torch.linalg.eigvals(G_z).real
        
        # For 16D, compute anisotropy using first 2 eigenvalues
        anisotropy = eigenvals[:, 0] - eigenvals[:, 1]
        
        # Reshape back to grid
        anisotropy_grid = anisotropy.cpu().numpy().reshape(X.shape)
    
    print(f"✅ Anisotropy computed for {n_points}x{n_points} grid")
    print(f"✅ Anisotropy range: [{anisotropy_grid.min():.3f}, {anisotropy_grid.max():.3f}]")
    
    return X, Y, anisotropy_grid

def run_sampling_with_metric(G_sampl, latent_data, device, n_samples=500):
    """Run sampling using the computed metric."""
    print(f"\n🎯 Running sampling with computed metric")
    
    # Sample from the latent space using the metric
    samples = []
    det_G_inv_samples = []
    
    with torch.no_grad():
        for i in range(n_samples):
            # Sample near real data points
            idx = np.random.randint(0, len(latent_data))
            base_point = latent_data[idx]
            
            # Add noise based on metric
            if G_sampl is not None:
                G_z = G_sampl(base_point.unsqueeze(0))
                G_inv = torch.linalg.inv(G_z)
                # Sample from N(0, G_inv)
                noise = torch.randn(1, 16, device=device)
                sample = base_point + torch.linalg.cholesky(G_inv).squeeze(0) @ noise.squeeze(0) * 0.1
            else:
                # Simple sampling
                sample = base_point + torch.randn(16, device=device) * 0.1
            
            samples.append(sample)
            
            # Compute G⁻¹ determinant at sample point
            if G_sampl is not None:
                G_z_sample = G_sampl(sample.unsqueeze(0))
                G_inv_sample = torch.linalg.inv(G_z_sample)
                det_G_inv = torch.linalg.det(G_inv_sample)
            else:
                det_G_inv = torch.tensor(1.0, device=device)
            
            det_G_inv_samples.append(det_G_inv)
    
    samples = torch.stack(samples)
    det_G_inv_samples = torch.stack(det_G_inv_samples)
    
    print(f"✅ Sampling completed")
    print(f"✅ Generated {len(samples)} samples")
    print(f"✅ G⁻¹ determinant at samples: [{det_G_inv_samples.min():.3e}, {det_G_inv_samples.max():.3e}]")
    
    return samples.cpu().numpy(), det_G_inv_samples.cpu().numpy()

def create_comprehensive_visualization(latent_data, centroids, samples, det_G_inv_samples, 
                                     X_det, Y_det, det_G_inv_grid, X_aniso, Y_aniso, anisotropy_grid):
    """Create comprehensive visualization with all four plots."""
    print(f"\n🎨 Creating comprehensive visualization")
    
    # Set consistent color scales
    det_vmin, det_vmax = det_G_inv_grid.min(), det_G_inv_grid.max()
    aniso_vmin, aniso_vmax = anisotropy_grid.min(), anisotropy_grid.max()
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Vanilla VAE Metric Analysis with retrieveG: Real Sprites Data", 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Centroids with data (2D projection)
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(latent_data[:, 0].cpu(), latent_data[:, 1].cpu(), 
                           c='lightblue', alpha=0.3, s=10, label='Real Data')
    ax1.scatter(centroids[:, 0].cpu(), centroids[:, 1].cpu(), 
                c='red', s=100, marker='*', label='Centroids', zorder=5)
    ax1.set_title("1. Centroids Computation\n(Real Sprites Data + retrieveG)", fontweight='bold')
    ax1.set_xlabel("z₁ (first dimension)")
    ax1.set_ylabel("z₂ (second dimension)")
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: G⁻¹ Determinant
    ax2 = axes[0, 1]
    contour2 = ax2.contourf(X_det, Y_det, det_G_inv_grid, levels=50, 
                            cmap='viridis', vmin=det_vmin, vmax=det_vmax)
    ax2.set_title("2. G⁻¹ Determinant\n(16D → 2D projection)", fontweight='bold')
    ax2.set_xlabel("z₁ (first dimension)")
    ax2.set_ylabel("z₂ (second dimension)")
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    plt.colorbar(contour2, ax=ax2, label='det(G⁻¹)')
    
    # Plot 3: Samples colored by G⁻¹ determinant
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(samples[:, 0], samples[:, 1], 
                           c=det_G_inv_samples, cmap='viridis', 
                           vmin=det_vmin, vmax=det_vmax, alpha=0.7, s=30)
    ax3.set_title("3. Metric-Based Sampling\n(Colored by det(G⁻¹))", fontweight='bold')
    ax3.set_xlabel("z₁ (first dimension)")
    ax3.set_ylabel("z₂ (second dimension)")
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='det(G⁻¹)')
    
    # Plot 4: Anisotropy
    ax4 = axes[1, 1]
    contour4 = ax4.contourf(X_aniso, Y_aniso, anisotropy_grid, levels=50, 
                            cmap='coolwarm', vmin=aniso_vmin, vmax=aniso_vmax)
    ax4.set_title("4. Anisotropy (λ₁ - λ₂)\n(16D → 2D projection)", fontweight='bold')
    ax4.set_xlabel("z₁ (first dimension)")
    ax4.set_ylabel("z₂ (second dimension)")
    ax4.set_xlim(-4, 4)
    ax4.set_ylim(-4, 4)
    plt.colorbar(contour4, ax=ax4, label='Anisotropy')
    
    # Add some sample points to anisotropy plot for reference
    ax4.scatter(samples[::10, 0], samples[::10, 1], 
                c='white', s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save locally
    plt.savefig("vanilla_vae_metric_analysis_real_data.png", dpi=150, bbox_inches='tight')
    
    # Log to WandB
    wandb.log({"metric_analysis/comprehensive_visualization": wandb.Image(fig)})
    
    plt.show()
    
    print("✅ Comprehensive visualization saved as 'vanilla_vae_metric_analysis_real_data.png'")
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"   Centroids: {len(centroids)} points")
    print(f"   Data points: {len(latent_data)}")
    print(f"   Samples: {len(samples)}")
    print(f"   G⁻¹ determinant range: [{det_vmin:.3e}, {det_vmax:.3e}]")
    print(f"   Anisotropy range: [{aniso_vmin:.3f}, {aniso_vmax:.3f}]")
    
    # Log metrics to WandB
    wandb.log({
        "metric_analysis/num_centroids": len(centroids),
        "metric_analysis/num_data_points": len(latent_data),
        "metric_analysis/num_samples": len(samples),
        "metric_analysis/det_g_inv_min": det_vmin,
        "metric_analysis/det_g_inv_max": det_vmax,
        "metric_analysis/anisotropy_min": aniso_vmin,
        "metric_analysis/anisotropy_max": aniso_vmax
    })

@hydra.main(config_path="conf", config_name="training/vanilla_vae_sprites_metric_analysis")
def main(cfg):
    """Main function to train VAE and analyze metric."""
    print("🚀 Vanilla VAE Training and Metric Analysis with Real Sprites Data")
    print("=" * 70)
    
    # Step 1: Train vanilla VAE
    model, dataset, device = train_vanilla_vae(cfg)
    
    # Step 2: Compute metric with manual retrieveG
    G_sampl, centroids, latent_data = manual_retrieveG(
        model, dataset, device, 
        num_centroids=cfg.metric_analysis.num_centroids,
        T_multiplier=cfg.metric_analysis.temperature_multiplier
    )
    
    # Step 3: Compute G⁻¹ determinant grid
    X_det, Y_det, det_G_inv_grid = compute_g_inverse_determinant_grid(
        G_sampl, device, n_points=cfg.metric_analysis.grid_points
    )
    
    # Step 4: Compute anisotropy grid
    X_aniso, Y_aniso, anisotropy_grid = compute_anisotropy_grid(
        G_sampl, device, n_points=cfg.metric_analysis.grid_points
    )
    
    # Step 5: Run sampling with metric
    samples, det_G_inv_samples = run_sampling_with_metric(
        G_sampl, latent_data, device, n_samples=cfg.metric_analysis.n_samples
    )
    
    # Step 6: Create comprehensive visualization
    create_comprehensive_visualization(latent_data, centroids, samples, det_G_inv_samples,
                                     X_det, Y_det, det_G_inv_grid, X_aniso, Y_aniso, anisotropy_grid)
    
    print(f"\n✅ Vanilla VAE training and metric analysis completed!")
    wandb.finish()

if __name__ == "__main__":
    main() 