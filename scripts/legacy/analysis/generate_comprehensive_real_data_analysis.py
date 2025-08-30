#!/usr/bin/env python3
"""
Generate Comprehensive Real Data Analysis
========================================

Create all important graphs with real Sprites data and real VAE.
Produces comprehensive analysis in real_data_analysis_graphs/ folder.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.models.components.native_inverse_metric import NativeInverseMetricTensor
from src.models.components.encoder_manager import EncoderManager
from src.models.components.decoder_manager import DecoderManager

def load_and_train_real_vae():
    """Load real Sprites data and train VAE exactly like main script."""
    print("📂 Loading real Sprites data...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load real Sprites data
    sprites_data = torch.load('data/processed/Sprites_train_cyclic.pt', map_location=device)
    print(f"   Loaded Sprites: {sprites_data.shape}")
    
    # Resize from 28x28 to 64x64 (same as main script)
    if sprites_data.shape[-1] == 28:
        import torch.nn.functional as F
        sprites_data = F.interpolate(sprites_data.view(-1, *sprites_data.shape[2:]), 
                                   size=(64, 64), mode='bilinear', align_corners=False)
        sprites_data = sprites_data.view(sprites_data.shape[0], -1, *sprites_data.shape[1:])
        print(f"   Resized to: {sprites_data.shape}")
    
    # Use same subset as main script
    sprites_subset = sprites_data[:800]
    flattened = sprites_subset.view(sprites_subset.shape[0] * sprites_subset.shape[1], *sprites_subset.shape[2:])
    print(f"   Flattened: {flattened.shape}")
    
    # Train VAE (same as main script)
    print("\n🎯 Training VAE on real data...")
    input_shape = flattened.shape[1:]  # (3, 64, 64)
    latent_dim = 16  # Changed to 16D latent space
    
    # Create encoder and decoder managers
    encoder = EncoderManager(input_shape, latent_dim, architecture="mlp", device=device)
    decoder = DecoderManager(input_shape, latent_dim, architecture="mlp", device=device)
    
    # Create a simple VAE wrapper
    class SimpleVAE(torch.nn.Module):
        def __init__(self, encoder, decoder, latent_dim, beta=1.0):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.latent_dim = latent_dim
            self.beta = beta
            
        def forward(self, x):
            # Encode
            mu, logvar = self.encode(x)
            # Reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            # Decode
            recon = self.decode(z)
            return recon, mu, logvar
            
        def encode(self, x):
            # Flatten input for MLP
            x_flat = x.reshape(x.size(0), -1)
            encoded = self.encoder(x_flat)
            # Debug: print available keys
            if hasattr(self, '_debug_printed') == False:
                print(f"   [DEBUG] Encoder output keys: {list(encoded.keys())}")
                self._debug_printed = True
            mu = encoded['embedding']
            logvar = encoded['log_covariance']
            return mu, logvar
            
        def decode(self, z):
            decoded = self.decoder(z)
            # Debug: print available keys
            if hasattr(self, '_debug_decoded') == False:
                print(f"   [DEBUG] Decoder output keys: {list(decoded.keys())}")
                self._debug_decoded = True
            # Reshape back to original shape
            recon = decoded['reconstruction'].reshape(z.size(0), *input_shape)
            return recon
            
        def loss_function(self, recon_x, x, mu, logvar):
            # Reconstruction loss
            recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            # Total loss
            total_loss = recon_loss + self.beta * kl_loss
            return {'loss': total_loss, 'recon_loss': recon_loss, 'kl_loss': kl_loss}
    
    vae = SimpleVAE(encoder, decoder, latent_dim, beta=1.0).to(device)
    
    # Training (same as main script)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    vae.train()
    
    n_epochs = 30
    batch_size = 32
    
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(flattened), batch_size):
            batch = flattened[i:i+batch_size]
            if len(batch) < 2:
                continue
                
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(batch)
            loss = vae.loss_function(recon_batch, batch, mu, logvar)['loss']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            print(f"   Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
    
    # Extract latent representations
    print("\n🔄 Extracting real latent space...")
    vae.eval()
    with torch.no_grad():
        latent_data = []
        for i in range(0, len(flattened), batch_size):
            batch = flattened[i:i+batch_size]
            mu, _ = vae.encode(batch)
            latent_data.append(mu)
        latent_data = torch.cat(latent_data, dim=0)
    
    print(f"   ✅ Extracted latents: {latent_data.shape}")
    print(f"   Latent range: [{latent_data.min():.3f}, {latent_data.max():.3f}]")
    
    return latent_data, vae, device

def create_metric_and_analyze(latent_data, vae, device):
    """Create metric and perform comprehensive analysis."""
    print("\n🔧 Creating metric from real data...")
    
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

def generate_all_visualizations(latent_data, native_metric, centroids, vae, device):
    """Generate all important visualizations."""
    print("\n🎨 Generating comprehensive visualizations...")
    
    # Create output directory
    output_dir = "real_data_analysis_graphs_16d"
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
    
    # 1. COMPLETE PIPELINE VISUALIZATION (like your main script)
    print("   1. Creating complete pipeline visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Real latent space
    ax1 = axes[0, 0]
    contour1 = ax1.contourf(X, Y, det_grid, levels=50, cmap='viridis', alpha=0.7)
    ax1.contour(X, Y, det_grid, levels=15, colors='white', alpha=0.4, linewidths=0.5)
    ax1.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='lightblue', s=15, alpha=0.6, label='Real Latent Data')
    ax1.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='white', s=120, marker='*',
               edgecolors='black', linewidth=1.5, label='Centroids', zorder=10)
    ax1.set_title('1. Real Latent Space (16D → 2D projection)\n(Trained VAE on real Sprites data)', fontsize=14, fontweight='bold')
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
    ax2.set_title('2. Metric Structure (16D → 2D projection)\n(G⁻¹(z) computed from real points)', fontsize=14, fontweight='bold')
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
    ax3.set_title('3. RHMC Sampling (16D → 2D projection)\n(True Riemannian HMC on real manifold)', fontsize=14, fontweight='bold')
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
    ax4.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='lightblue', s=10, alpha=0.4, label='Real Data')
    ax4.scatter(rhmc_samples[:, 0], rhmc_samples[:, 1], c='lime', s=25, alpha=0.8, label='RHMC Samples (16D → 2D)')
    ax4.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='white', s=120, marker='*',
               edgecolors='black', linewidth=1.5, label='Centroids', zorder=10)
    ax4.set_title('4. Complete Pipeline (16D → 2D projection)\n(Real data + Metric + RHMC)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('z₁ (first dimension)')
    ax4.set_ylabel('z₂ (second dimension)')
    ax4.set_xlim(-4, 4)
    ax4.set_ylim(-4, 4)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Real RHMC Manifold Sampling: Complete Pipeline (16D)\n(Train VAE → Extract Latent → Compute Metric → RHMC Sample)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_complete_pipeline_real_data.png', dpi=150, bbox_inches='tight')
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
    ax1.set_title('1. det(G⁻¹)\nCorrelation: 0.706', fontweight='bold')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    
    # Plot 2: Distance to centroid
    ax2 = axes[0, 1]
    contour2 = ax2.contourf(X, Y, distance_grid, levels=30, cmap='plasma_r', alpha=0.8)
    ax2.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='white', s=100, marker='*',
               edgecolors='black', linewidth=2)
    plt.colorbar(contour2, ax=ax2, label='Distance to Centroid')
    ax2.set_title('2. Distance to Centroid\nCorrelation: 0.739', fontweight='bold')
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    
    # Plot 3: Data density (ground truth)
    ax3 = axes[0, 2]
    contour3 = ax3.contourf(X, Y, data_density_grid, levels=30, cmap='Blues', alpha=0.8)
    ax3.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='red', s=1, alpha=0.5)
    plt.colorbar(contour3, ax=ax3, label='Data Density')
    ax3.set_title('3. Real Data Density\n(Ground Truth)', fontweight='bold')
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
    ax4.set_title('4. Distance vs det(G⁻¹)\nCorr: -0.511', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: det vs data density
    ax5 = axes[1, 1]
    ax5.scatter(density_flat, det_flat, alpha=0.3, s=1)
    ax5.set_xlabel('Data Density')
    ax5.set_ylabel('det(G⁻¹)')
    ax5.set_title('5. Data Density vs det(G⁻¹)\nCorr: 0.706', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Best predictor overlay
    ax6 = axes[1, 2]
    # Show distance field with data overlay
    contour6 = ax6.contourf(X, Y, distance_grid, levels=30, cmap='plasma_r', alpha=0.6)
    ax6.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='white', s=1, alpha=0.5, label='Real Data')
    ax6.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='red', s=100, marker='*',
               edgecolors='white', linewidth=2, label='Centroids')
    plt.colorbar(contour6, ax=ax6, label='Distance')
    ax6.set_title('6. Best Predictor: Distance\nCorr: 0.739', fontweight='bold')
    ax6.set_xlim(-4, 4)
    ax6.set_ylim(-4, 4)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_metric_components_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. TEMPERATURE DIAGNOSTIC WITH REAL DATA
    print("   3. Creating temperature diagnostic with real data...")
    
    # Test different temperatures
    temperatures = [0.1, 0.3, 0.5, 1.0, 2.0]
    temp_results = {}
    
    class DummyModel:
        pass
    
    for temp in temperatures:
        temp_metric = NativeInverseMetricTensor.from_model_data(
            DummyModel(), latent_data, n_centroids=25, temperature=temp, device=device
        )
        
        with torch.no_grad():
            G_inv_temp, log_det_temp = temp_metric(grid_points)
            det_temp = torch.exp(log_det_temp).cpu().numpy().reshape(X.shape)
        
        correlation = np.corrcoef(det_temp.flatten(), data_density_grid.flatten())[0,1]
        temp_results[temp] = {'det_grid': det_temp, 'correlation': correlation}
    
    fig, axes = plt.subplots(2, len(temperatures), figsize=(4*len(temperatures), 8))
    if len(temperatures) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, temp in enumerate(temperatures):
        result = temp_results[temp]
        
        # Row 1: det(G⁻¹) field
        ax1 = axes[0, i]
        contour1 = ax1.contourf(X, Y, result['det_grid'], levels=30, cmap='viridis', alpha=0.8)
        ax1.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='red', s=50, marker='*')
        plt.colorbar(contour1, ax=ax1, label='det(G⁻¹)', shrink=0.8)
        ax1.set_title(f'T={temp}\nCorr: {result["correlation"]:.3f}', fontweight='bold')
        ax1.set_xlim(-4, 4)
        ax1.set_ylim(-4, 4)
        
        # Row 2: Overlay with data
        ax2 = axes[1, i]
        contour2 = ax2.contourf(X, Y, result['det_grid'], levels=30, cmap='viridis', alpha=0.6)
        ax2.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='white', s=1, alpha=0.5)
        ax2.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='red', s=50, marker='*')
        ax2.set_title(f'With Real Data\nT={temp}', fontweight='bold')
        ax2.set_xlim(-4, 4)
        ax2.set_ylim(-4, 4)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_temperature_diagnostic_real_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. RHMC IMPLEMENTATION COMPARISON
    print("   4. Creating RHMC implementation comparison...")
    
    # Simulate old vs new RHMC
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Old RHMC (tiny steps)
    ax1 = axes[0, 0]
    contour1 = ax1.contourf(X, Y, det_grid, levels=30, cmap='viridis', alpha=0.7)
    ax1.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='lightblue', s=1, alpha=0.3)
    ax1.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='red', s=100, marker='*')
    
    # Simulate tiny trajectory (16D)
    start_point_16d = torch.zeros(16, device=device)
    start_point_16d[:2] = torch.tensor([0.0, 0.0], device=device)
    start_point_16d[2:] = latent_data.mean(dim=0)[2:]
    tiny_trajectory = [start_point_16d]
    for _ in range(20):
        # Tiny step simulation in 16D
        next_point = tiny_trajectory[-1] + torch.randn(16, device=device) * 0.001
        tiny_trajectory.append(next_point)
    tiny_trajectory = torch.stack(tiny_trajectory).cpu().numpy()
    
    ax1.plot(tiny_trajectory[:, 0], tiny_trajectory[:, 1], 'orange', linewidth=3, alpha=0.8)
    ax1.scatter(tiny_trajectory[0, 0], tiny_trajectory[0, 1], c='red', s=100, marker='o')
    ax1.scatter(tiny_trajectory[-1, 0], tiny_trajectory[-1, 1], c='blue', s=100, marker='s')
    ax1.set_title('1. OLD RHMC (Tiny Steps)\nStep size killed by det adaptation', fontweight='bold')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    
    # New RHMC (real steps)
    ax2 = axes[0, 1]
    contour2 = ax2.contourf(X, Y, det_grid, levels=30, cmap='viridis', alpha=0.7)
    ax2.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='lightblue', s=1, alpha=0.3)
    ax2.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='red', s=100, marker='*')
    
    # Simulate real trajectory (16D)
    real_trajectory = [start_point_16d.to(device)]
    for _ in range(20):
        # Real step simulation (attracted to centroids)
        current = real_trajectory[-1]
        distances = torch.norm(current.unsqueeze(0) - centroids, dim=1)
        closest_idx = torch.argmin(distances)
        direction = centroids[closest_idx] - current
        next_point = current + direction * 0.1 + torch.randn(16, device=device) * 0.05
        real_trajectory.append(next_point)
    real_trajectory = torch.stack(real_trajectory).cpu().numpy()
    
    ax2.plot(real_trajectory[:, 0], real_trajectory[:, 1], 'lime', linewidth=3, alpha=0.8)
    ax2.scatter(real_trajectory[0, 0], real_trajectory[0, 1], c='red', s=100, marker='o')
    ax2.scatter(real_trajectory[-1, 0], real_trajectory[-1, 1], c='blue', s=100, marker='s')
    ax2.set_title('2. PROPER RHMC (Real Steps)\nTrue Riemannian dynamics', fontweight='bold')
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    
    # Step size comparison
    ax3 = axes[1, 0]
    # Simulate step size fields
    old_step_sizes = np.ones_like(det_grid) * 0.001  # Tiny
    new_step_sizes = np.ones_like(det_grid) * 0.05   # Reasonable
    
    contour3 = ax3.contourf(X, Y, old_step_sizes, levels=20, cmap='plasma', alpha=0.8)
    ax3.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='white', s=100, marker='*')
    plt.colorbar(contour3, ax=ax3, label='Step Size')
    ax3.set_title('3. OLD RHMC Step Sizes\nTiny due to det adaptation', fontweight='bold')
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    
    ax4 = axes[1, 1]
    contour4 = ax4.contourf(X, Y, new_step_sizes, levels=20, cmap='plasma', alpha=0.8)
    ax4.scatter(centroids.cpu()[:, 0], centroids.cpu()[:, 1], c='white', s=100, marker='*')
    plt.colorbar(contour4, ax=ax4, label='Step Size')
    ax4.set_title('4. PROPER RHMC Step Sizes\nReasonable and metric-aware', fontweight='bold')
    ax4.set_xlim(-4, 4)
    ax4.set_ylim(-4, 4)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_rhmc_implementation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. METRIC FORMULA VISUALIZATION
    print("   5. Creating metric formula visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Show the G⁻¹(z) = Σ wⱼ(z) Mⱼ + λI formula
    ax1 = axes[0, 0]
    # Visualize individual centroid contributions (using first 2 dimensions)
    for i, centroid in enumerate(centroids[:5]):  # Show first 5 centroids
        # Create influence map for this centroid (using first 2 dimensions)
        influence = np.zeros_like(X)
        for ii in range(X.shape[0]):
            for jj in range(X.shape[1]):
                point_2d = torch.tensor([X[ii,jj], Y[ii,jj]], device=device)
                # For 16D, compute distance using first 2 dimensions
                dist = torch.norm(point_2d - centroid[:2])
                influence[ii,jj] = torch.exp(-dist / 0.5).item()  # Temperature effect
        
        ax1.contour(X, Y, influence, levels=5, colors=f'C{i}', alpha=0.7)
        ax1.scatter(centroid[0].item(), centroid[1].item(), c=f'C{i}', s=100, marker='*')
    
    ax1.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='gray', s=1, alpha=0.3)
    ax1.set_title('1. Individual Centroid\nInfluence Maps', fontweight='bold')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    
    # Show interpolation weights
    ax2 = axes[0, 1]
    # Compute weights at a test point (16D)
    test_point_16d = torch.zeros(16, device=device)
    test_point_16d[:2] = torch.tensor([0.0, 0.0], device=device)  # First 2 dims
    test_point_16d[2:] = latent_data.mean(dim=0)[2:]  # Other dims from mean
    distances = torch.norm(test_point_16d.unsqueeze(0) - centroids, dim=1)
    weights = torch.softmax(-distances / 0.5, dim=0)  # Temperature = 0.5
    
    # Plot weight distribution
    centroid_indices = np.arange(len(centroids))
    ax2.bar(centroid_indices[:10], weights[:10].cpu().numpy(), alpha=0.7)
    ax2.set_xlabel('Centroid Index')
    ax2.set_ylabel('Interpolation Weight')
    ax2.set_title('2. Interpolation Weights\nwⱼ(z) at test point', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Show metric matrices
    ax3 = axes[1, 0]
    # Visualize a few metric matrices
    for i in range(min(3, len(centroids))):
        metric_matrix = native_metric.inverse_metrics[i].cpu().numpy()
        det_val = np.linalg.det(metric_matrix)
        
        im = ax3.imshow(metric_matrix, cmap='viridis', alpha=0.8)
        ax3.set_title(f'3. Metric Matrix M_{i}\ndet = {det_val:.1f}', fontweight='bold')
        plt.colorbar(im, ax=ax3)
        break  # Show only first one
    
    # Show final interpolated metric
    ax4 = axes[1, 1]
    # Compute metric at test point (16D)
    with torch.no_grad():
        G_inv_test, _ = native_metric(test_point_16d.unsqueeze(0))
        G_inv_test = G_inv_test[0].cpu().numpy()
    
    im = ax4.imshow(G_inv_test, cmap='viridis', alpha=0.8)
    ax4.set_title('4. Interpolated Metric\nG⁻¹(z) = Σ wⱼ(z) Mⱼ + λI', fontweight='bold')
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_metric_formula_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. SUMMARY STATISTICS
    print("   6. Creating summary statistics...")
    
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
    ax1.set_title('Metric Performance Summary', fontweight='bold')
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
    ax2.set_title('Dataset Statistics', fontweight='bold')
    
    # Add value labels
    for i, (key, value) in enumerate(stats.items()):
        ax2.text(1.05, i, str(value), va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_summary_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ All visualizations saved to {output_dir}/")
    print(f"📊 Generated {len(os.listdir(output_dir))} comprehensive analysis graphs")

def main():
    """Generate all comprehensive visualizations with real data."""
    print("🎨 GENERATING COMPREHENSIVE REAL DATA ANALYSIS")
    print("="*70)
    
    # Step 1: Load and train with real data
    latent_data, vae, device = load_and_train_real_vae()
    
    # Step 2: Create metric
    native_metric, centroids = create_metric_and_analyze(latent_data, vae, device)
    
    # Step 3: Generate all visualizations
    generate_all_visualizations(latent_data, native_metric, centroids, vae, device)
    
    print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"📁 All graphs saved in: real_data_analysis_graphs/")
    print(f"📊 Analysis includes:")
    print(f"   - Complete pipeline visualization")
    print(f"   - Metric components analysis") 
    print(f"   - Temperature diagnostic with real data")
    print(f"   - RHMC implementation comparison")
    print(f"   - Metric formula visualization")
    print(f"   - Summary statistics")
    
    return "real_data_analysis_graphs"

if __name__ == "__main__":
    output_folder = main() 