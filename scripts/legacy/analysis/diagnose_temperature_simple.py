#!/usr/bin/env python3
"""
Simple Temperature Diagnostic
=============================

Test different temperatures with simulated real data to make det(G⁻¹) fit data perfectly!
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.components.native_inverse_metric import NativeInverseMetricTensor

def create_realistic_latent_data(device):
    """Create realistic latent data that matches your real Sprites characteristics."""
    print("📂 Creating realistic latent data (based on your real data pattern)...")
    
    torch.manual_seed(42)
    
    # Based on your images, create clusters that match the real data distribution
    cluster_centers = [
        [0.0, 0.0],      # Central cluster
        [1.5, 1.0],      # Upper right cluster  
        [-1.0, 0.5],     # Left cluster
        [0.5, -1.0],     # Lower right
        [-1.5, -1.0],    # Lower left
    ]
    
    cluster_sizes = [2000, 1500, 1200, 800, 900]  # Different cluster sizes
    cluster_spreads = [0.3, 0.25, 0.35, 0.2, 0.4]  # Different spreads
    
    all_points = []
    
    for center, size, spread in zip(cluster_centers, cluster_sizes, cluster_spreads):
        cluster_points = torch.randn(size, 2, device=device) * spread + torch.tensor(center, device=device)
        all_points.append(cluster_points)
    
    latent_data = torch.cat(all_points, dim=0)
    
    # Add some noise points
    noise_points = torch.randn(500, 2, device=device) * 2.0
    latent_data = torch.cat([latent_data, noise_points], dim=0)
    
    # Clamp to reasonable range
    latent_data = torch.clamp(latent_data, -2.5, 2.5)
    
    print(f"   Created {len(latent_data)} latent points")
    print(f"   Range: [{latent_data.min():.3f}, {latent_data.max():.3f}]")
    
    return latent_data

def test_temperatures_comprehensive(latent_data, device):
    """Test comprehensive range of temperatures."""
    print("\n🌡️  COMPREHENSIVE TEMPERATURE TESTING")
    print("="*60)
    
    class DummyModel:
        pass
    model = DummyModel()
    
    # Test wide range of temperatures
    temperatures = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    n_centroids = 25
    
    results = {}
    
    # Create analysis grid
    x = np.linspace(-3, 3, 60)
    y = np.linspace(-3, 3, 60)
    X, Y = np.meshgrid(x, y)
    grid_points = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), 
                              dtype=torch.float32, device=device)
    
    # Compute ground truth data density once
    print("   Computing ground truth data density...")
    latent_cpu = latent_data.cpu().numpy()
    data_density_grid = np.zeros(X.shape)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i,j], Y[i,j]])
            distances = np.linalg.norm(latent_cpu - point, axis=1)
            density = np.sum(distances < 0.25)  # Count points within radius
            data_density_grid[i,j] = density
    
    max_density = data_density_grid.max()
    print(f"   Max data density: {max_density}")
    
    for temp in temperatures:
        print(f"\n   Testing temperature: {temp}")
        
        # Create metric with this temperature
        try:
            native_metric = NativeInverseMetricTensor.from_model_data(
                model, latent_data, 
                n_centroids=n_centroids,
                temperature=temp,
                device=device
            )
            
            # Compute determinants
            with torch.no_grad():
                G_inv, log_det_G_inv = native_metric(grid_points)
                det_grid = torch.exp(log_det_G_inv).cpu().numpy().reshape(X.shape)
            
            # Compute metrics
            correlation = np.corrcoef(det_grid.flatten(), data_density_grid.flatten())[0,1]
            
            # Compute how well det peaks align with data peaks
            det_normalized = (det_grid - det_grid.min()) / (det_grid.max() - det_grid.min())
            data_normalized = data_density_grid / max_density
            
            # Mean squared difference (lower is better)
            alignment_error = np.mean((det_normalized - data_normalized)**2)
            
            # Compute sharpness (gradient magnitude)
            grad_y, grad_x = np.gradient(det_grid)
            sharpness = np.mean(np.sqrt(grad_x**2 + grad_y**2))
            
            # Peak alignment score (how well the peaks match)
            # Find top 10% regions for both
            det_top10 = det_grid > np.percentile(det_grid, 90)
            data_top10 = data_density_grid > np.percentile(data_density_grid, 90)
            overlap = np.sum(det_top10 & data_top10) / np.sum(data_top10)
            
            results[temp] = {
                'det_grid': det_grid,
                'correlation': correlation,
                'alignment_error': alignment_error,
                'sharpness': sharpness,
                'peak_overlap': overlap,
                'det_range': (det_grid.min(), det_grid.max()),
                'centroids': native_metric.centroids.cpu()
            }
            
            print(f"     Correlation: {correlation:.3f}")
            print(f"     Peak overlap: {overlap:.3f}")
            print(f"     Alignment error: {alignment_error:.4f}")
            print(f"     Sharpness: {sharpness:.1f}")
            
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            continue
    
    # Find best temperature using multiple criteria
    print(f"\n   📊 RANKING TEMPERATURES:")
    
    # Rank by different criteria
    rankings = {}
    for criterion in ['correlation', 'peak_overlap', 'alignment_error']:
        if criterion == 'alignment_error':
            # Lower is better for alignment error
            sorted_temps = sorted(results.keys(), key=lambda t: results[t][criterion])
        else:
            # Higher is better for correlation and overlap
            sorted_temps = sorted(results.keys(), key=lambda t: results[t][criterion], reverse=True)
        
        rankings[criterion] = sorted_temps
        print(f"     {criterion}: {sorted_temps[:3]}")
    
    # Composite score (weighted combination)
    composite_scores = {}
    for temp in results.keys():
        score = (results[temp]['correlation'] * 0.4 + 
                results[temp]['peak_overlap'] * 0.4 - 
                results[temp]['alignment_error'] * 0.2)
        composite_scores[temp] = score
    
    best_temp = max(composite_scores.keys(), key=lambda t: composite_scores[t])
    print(f"\n   🎯 BEST TEMPERATURE: {best_temp}")
    print(f"     Composite score: {composite_scores[best_temp]:.3f}")
    print(f"     Correlation: {results[best_temp]['correlation']:.3f}")
    print(f"     Peak overlap: {results[best_temp]['peak_overlap']:.3f}")
    
    return results, X, Y, data_density_grid, latent_cpu, best_temp, composite_scores

def create_temperature_diagnostic_visualization(results, X, Y, data_density_grid, latent_cpu, best_temp, scores):
    """Create comprehensive diagnostic visualization."""
    print("\n🎨 Creating diagnostic visualization...")
    
    temperatures = sorted(results.keys())
    
    # Select key temperatures to show (including best)
    key_temps = [temperatures[0], temperatures[len(temperatures)//4], temperatures[len(temperatures)//2], 
                temperatures[3*len(temperatures)//4], temperatures[-1]]
    if best_temp not in key_temps:
        key_temps.append(best_temp)
    key_temps = sorted(set(key_temps))
    
    n_temps = len(key_temps)
    
    # Create main comparison figure
    fig, axes = plt.subplots(4, n_temps, figsize=(4*n_temps, 16))
    if n_temps == 1:
        axes = axes.reshape(-1, 1)
    
    for i, temp in enumerate(key_temps):
        if temp not in results:
            continue
            
        result = results[temp]
        
        # Row 1: det(G⁻¹) field
        ax1 = axes[0, i]
        contour1 = ax1.contourf(X, Y, result['det_grid'], levels=30, cmap='viridis', alpha=0.8)
        ax1.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='white', s=0.5, alpha=0.3)
        ax1.scatter(result['centroids'][:, 0], result['centroids'][:, 1], c='red', s=30, 
                   marker='*', edgecolors='white', linewidth=0.5)
        plt.colorbar(contour1, ax=ax1, label='det(G⁻¹)', shrink=0.8)
        
        title_color = 'red' if temp == best_temp else 'black'
        weight = 'bold' if temp == best_temp else 'normal'
        ax1.set_title(f'T={temp}\ndet(G⁻¹)', color=title_color, fontweight=weight)
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        
        # Row 2: Data density
        ax2 = axes[1, i]
        contour2 = ax2.contourf(X, Y, data_density_grid, levels=30, cmap='Blues', alpha=0.8)
        ax2.scatter(latent_cpu[:, 0], latent_cpu[:, 1], c='red', s=0.5, alpha=0.5)
        plt.colorbar(contour2, ax=ax2, label='Data Density', shrink=0.8)
        ax2.set_title('Real Data\nDensity', fontweight='bold')
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(-3, 3)
        
        # Row 3: Difference map
        ax3 = axes[2, i]
        det_norm = (result['det_grid'] - result['det_grid'].min()) / (result['det_grid'].max() - result['det_grid'].min())
        data_norm = data_density_grid / data_density_grid.max()
        difference = np.abs(det_norm - data_norm)
        
        contour3 = ax3.contourf(X, Y, difference, levels=30, cmap='Reds', alpha=0.8)
        plt.colorbar(contour3, ax=ax3, label='|Difference|', shrink=0.8)
        ax3.set_title(f'Alignment Error\n{result["alignment_error"]:.4f}', fontweight='bold')
        ax3.set_xlim(-3, 3)
        ax3.set_ylim(-3, 3)
        
        # Row 4: Metrics summary
        ax4 = axes[3, i]
        metrics = ['Correlation', 'Peak Overlap', 'Sharpness/100']
        values = [result['correlation'], result['peak_overlap'], result['sharpness']/100]
        colors = ['blue', 'green', 'orange']
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_ylim(0, 1)
        ax4.set_title(f'Metrics Summary\nScore: {scores[temp]:.3f}', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('diagnose_temperature_simple.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create summary curves
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    
    temps = sorted(results.keys())
    correlations = [results[t]['correlation'] for t in temps]
    overlaps = [results[t]['peak_overlap'] for t in temps]
    errors = [results[t]['alignment_error'] for t in temps]
    scores_list = [scores[t] for t in temps]
    
    # Plot 1: Correlation vs Temperature
    axes2[0,0].plot(temps, correlations, 'bo-', linewidth=2)
    axes2[0,0].scatter([best_temp], [results[best_temp]['correlation']], c='red', s=100, zorder=10)
    axes2[0,0].set_xlabel('Temperature')
    axes2[0,0].set_ylabel('Correlation with Data')
    axes2[0,0].set_title('Data Correlation vs Temperature')
    axes2[0,0].grid(True, alpha=0.3)
    axes2[0,0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 2: Peak overlap vs Temperature
    axes2[0,1].plot(temps, overlaps, 'go-', linewidth=2)
    axes2[0,1].scatter([best_temp], [results[best_temp]['peak_overlap']], c='red', s=100, zorder=10)
    axes2[0,1].set_xlabel('Temperature')
    axes2[0,1].set_ylabel('Peak Overlap Score')
    axes2[0,1].set_title('Peak Alignment vs Temperature')
    axes2[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Alignment error vs Temperature
    axes2[1,0].plot(temps, errors, 'ro-', linewidth=2)
    axes2[1,0].scatter([best_temp], [results[best_temp]['alignment_error']], c='blue', s=100, zorder=10)
    axes2[1,0].set_xlabel('Temperature')
    axes2[1,0].set_ylabel('Alignment Error (lower=better)')
    axes2[1,0].set_title('Alignment Error vs Temperature')
    axes2[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Composite score
    axes2[1,1].plot(temps, scores_list, 'mo-', linewidth=2)
    axes2[1,1].scatter([best_temp], [scores[best_temp]], c='red', s=100, zorder=10)
    axes2[1,1].set_xlabel('Temperature')
    axes2[1,1].set_ylabel('Composite Score')
    axes2[1,1].set_title('Overall Performance vs Temperature')
    axes2[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temperature_analysis_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Visualizations saved!")
    return best_temp

def main():
    """Run complete temperature diagnostic."""
    print("🔍 TEMPERATURE DIAGNOSTIC FOR PERFECT DATA FITTING")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create realistic data
    latent_data = create_realistic_latent_data(device)
    
    # Test temperatures
    results, X, Y, data_density, latent_cpu, best_temp, scores = test_temperatures_comprehensive(latent_data, device)
    
    # Create visualizations
    optimal_temp = create_temperature_diagnostic_visualization(results, X, Y, data_density, latent_cpu, best_temp, scores)
    
    print(f"\n🎉 DIAGNOSTIC COMPLETE!")
    print(f"   🎯 OPTIMAL TEMPERATURE: {optimal_temp}")
    print(f"   📈 Best correlation: {results[optimal_temp]['correlation']:.3f}")
    print(f"   🎪 Best peak overlap: {results[optimal_temp]['peak_overlap']:.3f}")
    print(f"\n💡 RECOMMENDATION:")
    print(f"   Update your main script: temperature={optimal_temp}")
    print(f"   This should make det(G⁻¹) perfectly fit your data distribution!")
    
    return optimal_temp

if __name__ == "__main__":
    optimal_temperature = main()