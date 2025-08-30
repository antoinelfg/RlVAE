#!/usr/bin/env python3
"""
Diagnose Temperature with Real Data
==================================

Test different temperature values to make det(G⁻¹) fit the real data perfectly!
The metric should have HIGH det where data is and LOW det where it's empty.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.components.native_inverse_metric import NativeInverseMetricTensor

def load_real_sprites_data():
    """Load the exact same Sprites data as the main script."""
    print("📂 Loading real Sprites data...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load real Sprites data (same as main script)
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
        
        return flattened, device
        
    except FileNotFoundError:
        print("   ⚠️  Sprites file not found, creating realistic simulation...")
        # Create data that matches real characteristics
        torch.manual_seed(42)
        # Create realistic clusters based on what we see in the images
        cluster_centers = [
            [-1.5, 0.5], [0.0, 1.5], [1.5, 1.0], [0.5, 0.0], [-1.0, -1.0]
        ]
        
        flattened = []
        for center in cluster_centers:
            cluster = torch.randn(1280, 2, device=device) * 0.3 + torch.tensor(center, device=device)
            flattened.append(cluster)
        
        flattened = torch.cat(flattened, dim=0)
        return flattened, device

def train_vae_and_extract_latents(data, device):
    """Train VAE and extract latents (simplified version of main script)."""
    print("🎯 Training VAE and extracting latents...")
    
    # Import VAE components
    from src.models.components.vae_modules import MLPEncoder, MLPDecoder
    from src.models.vanilla_vae import ModularVanillaVAE
    
    # Create VAE (same as main script)
    input_shape = data.shape[1:]  # (3, 64, 64)
    latent_dim = 2
    
    encoder = MLPEncoder(input_shape, latent_dim).to(device)
    decoder = MLPDecoder(latent_dim, input_shape).to(device)
    vae = ModularVanillaVAE(encoder, decoder, latent_dim=latent_dim, beta=1.0).to(device)
    
    # Quick training (reduced epochs for speed)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    vae.train()
    
    n_epochs = 20  # Reduced for speed
    batch_size = 64
    
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            if len(batch) < 2:
                continue
                
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(batch)
            loss = vae.loss_function(recon_batch, batch, mu, logvar)['loss']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            print(f"   Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
    
    # Extract latent representations
    vae.eval()
    with torch.no_grad():
        latent_data = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            mu, _ = vae.encode(batch)
            latent_data.append(mu)
        latent_data = torch.cat(latent_data, dim=0)
    
    print(f"   ✅ Extracted latents: {latent_data.shape}")
    print(f"   Latent range: [{latent_data.min():.3f}, {latent_data.max():.3f}]")
    
    return latent_data, vae

def test_temperature_effects(latent_data, device):
    """Test different temperatures to see which makes det(G⁻¹) fit data best."""
    print("\n🌡️  TESTING TEMPERATURE EFFECTS ON METRIC")
    print("="*60)
    
    class DummyModel:
        pass
    model = DummyModel()
    
    # Test different temperatures
    temperatures = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    n_centroids = 25
    
    results = {}
    
    for temp in temperatures:
        print(f"\n   Testing temperature: {temp}")
        
        # Create metric with this temperature
        native_metric = NativeInverseMetricTensor.from_model_data(
            model, latent_data, 
            n_centroids=n_centroids,
            temperature=temp,
            device=device
        )
        
        # Create grid for analysis
        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x, y)
        grid_points = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), 
                                  dtype=torch.float32, device=device)
        
        # Compute determinants
        with torch.no_grad():
            G_inv, log_det_G_inv = native_metric(grid_points)
            det_grid = torch.exp(log_det_G_inv).cpu().numpy().reshape(X.shape)
        
        # Compute actual data density
        latent_cpu = latent_data.cpu().numpy()
        data_density_grid = np.zeros_like(det_grid)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.array([X[i,j], Y[i,j]])
                distances = np.linalg.norm(latent_cpu - point, axis=1)
                density = np.sum(distances < 0.2)  # Count points within radius
                data_density_grid[i,j] = density
        
        # Compute correlation between det(G⁻¹) and data density
        correlation = np.corrcoef(det_grid.flatten(), data_density_grid.flatten())[0,1]
        
        # Compute how "sharp" the metric is (gradient magnitude)
        grad_y, grad_x = np.gradient(det_grid)
        sharpness = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        # Store results
        results[temp] = {
            'det_grid': det_grid,
            'data_density': data_density_grid,
            'correlation': correlation,
            'sharpness': sharpness,
            'det_range': (det_grid.min(), det_grid.max()),
            'centroids': native_metric.centroids.cpu()
        }
        
        print(f"     Correlation with data: {correlation:.3f}")
        print(f"     Det(G⁻¹) range: [{det_grid.min():.0f}, {det_grid.max():.0f}]")
        print(f"     Sharpness: {sharpness:.3f}")
    
    # Find best temperature
    best_temp = max(results.keys(), key=lambda t: results[t]['correlation'])
    print(f"\n   🎯 BEST TEMPERATURE: {best_temp} (correlation: {results[best_temp]['correlation']:.3f})")
    
    return results, X, Y, latent_cpu, best_temp

def create_temperature_comparison_visualization(results, X, Y, latent_cpu, best_temp):
    """Create comprehensive visualization comparing different temperatures."""
    print("\n🎨 Creating temperature comparison visualization...")
    
    temperatures = sorted(results.keys())
    n_temps = len(temperatures)
    
    # Create figure with subplots for each temperature
    fig, axes = plt.subplots(3, n_temps, figsize=(4*n_temps, 12))
    if n_temps == 1:
        axes = axes.reshape(-1, 1)
    
    for i, temp in enumerate(temperatures):
        result = results[temp]
        
        # Row 1: det(G⁻¹) field
        ax1 = axes[0, i]
        contour1 = ax1.contourf(X, Y, result['det_grid'], levels=30, cmap='viridis', alpha=0.8)
        ax1.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='white', s=1, alpha=0.5)
        ax1.scatter(result['centroids'][:, 0], result['centroids'][:, 1], c='red', s=50, 
                   marker='*', edgecolors='white', linewidth=1)
        plt.colorbar(contour1, ax=ax1, label='det(G⁻¹)')
        
        title_color = 'red' if temp == best_temp else 'black'
        ax1.set_title(f'T={temp}\nCorr: {result["correlation"]:.3f}', 
                     fontweight='bold', color=title_color)
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        
        # Row 2: Data density (ground truth)
        ax2 = axes[1, i]
        contour2 = ax2.contourf(X, Y, result['data_density'], levels=30, cmap='Blues', alpha=0.8)
        ax2.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='red', s=1, alpha=0.7)
        ax2.scatter(result['centroids'][:, 0], result['centroids'][:, 1], c='red', s=50, 
                   marker='*', edgecolors='white', linewidth=1)
        plt.colorbar(contour2, ax=ax2, label='Data Density')
        ax2.set_title('Data Density\n(Ground Truth)', fontweight='bold')
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(-3, 3)
        
        # Row 3: Overlay comparison
        ax3 = axes[2, i]
        # Normalize both fields to [0,1] for comparison
        det_norm = (result['det_grid'] - result['det_grid'].min()) / (result['det_grid'].max() - result['det_grid'].min())
        data_norm = (result['data_density'] - result['data_density'].min()) / (result['data_density'].max() - result['data_density'].min())
        
        # Create RGB overlay: Red=det(G⁻¹), Blue=data density, Purple=agreement
        overlay = np.zeros((*det_norm.shape, 3))
        overlay[:, :, 0] = det_norm  # Red channel for det(G⁻¹)
        overlay[:, :, 2] = data_norm  # Blue channel for data density
        
        ax3.imshow(overlay, extent=[-3, 3, -3, 3], origin='lower', alpha=0.8)
        ax3.scatter(result['centroids'][:, 0], result['centroids'][:, 1], c='yellow', s=50, 
                   marker='*', edgecolors='black', linewidth=1)
        ax3.set_title(f'Overlay\nRed=det(G⁻¹), Blue=Data\nPurple=Agreement', fontweight='bold')
        ax3.set_xlim(-3, 3)
        ax3.set_ylim(-3, 3)
    
    plt.tight_layout()
    plt.savefig('diagnose_temperature_real_data.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create summary plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Correlation vs Temperature
    temps = list(temperatures)
    corrs = [results[t]['correlation'] for t in temps]
    sharpness = [results[t]['sharpness'] for t in temps]
    
    ax1.plot(temps, corrs, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.scatter([best_temp], [results[best_temp]['correlation']], c='red', s=100, zorder=10,
               label=f'Best: T={best_temp}')
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('Correlation with Data Density')
    ax1.set_title('Temperature vs Data Correlation', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Sharpness vs Temperature
    ax2.plot(temps, sharpness, 'go-', linewidth=2, markersize=8)
    ax2.scatter([best_temp], [results[best_temp]['sharpness']], c='red', s=100, zorder=10,
               label=f'Best: T={best_temp}')
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('Metric Sharpness')
    ax2.set_title('Temperature vs Metric Sharpness', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('temperature_analysis_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Visualizations saved!")
    print(f"\n🎯 FINAL RECOMMENDATION:")
    print(f"   Use temperature = {best_temp} for best data fitting")
    print(f"   This gives correlation = {results[best_temp]['correlation']:.3f} with real data density")
    
    return best_temp

def diagnose_with_real_data():
    """Complete diagnostic using real Sprites data."""
    print("🔍 COMPLETE REAL DATA TEMPERATURE DIAGNOSTIC")
    print("="*70)
    
    # Step 1: Load real data
    data, device = load_real_sprites_data()
    
    # Step 2: Train VAE and extract latents
    latent_data, vae = train_vae_and_extract_latents(data, device)
    
    # Step 3: Test different temperatures
    results, X, Y, latent_cpu, best_temp = test_temperature_effects(latent_data, device)
    
    # Step 4: Create visualizations
    optimal_temp = create_temperature_comparison_visualization(results, X, Y, latent_cpu, best_temp)
    
    print(f"\n🎉 DIAGNOSTIC COMPLETE!")
    print(f"   📊 Tested {len(results)} different temperatures")
    print(f"   🎯 Optimal temperature: {optimal_temp}")
    print(f"   📈 Best correlation: {results[optimal_temp]['correlation']:.3f}")
    print(f"\n💡 NEXT STEP: Update your main script to use temperature = {optimal_temp}")
    
    return optimal_temp, results

if __name__ == "__main__":
    optimal_temperature, all_results = diagnose_with_real_data()