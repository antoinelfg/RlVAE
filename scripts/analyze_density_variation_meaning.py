#!/usr/bin/env python3
"""
Analyze Density Variation Meaning
=================================

Deep analysis of what density variation means and whether 32% is actually problematic.
"""

import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rlvae.models.components.riemannian_rhmc_posterior import RiemannianRHMCPosterior


class AnalysisModel(torch.nn.Module):
    """Model for analyzing density variation characteristics."""
    
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create centroids representing different manifold regions
        self.centroids_tens = torch.tensor([
            [2.0, 2.0],   # Region 1: High curvature
            [-2.0, 2.0],  # Region 2: Medium curvature
            [-2.0, -2.0], # Region 3: Low curvature  
            [2.0, -2.0]   # Region 4: Medium curvature
        ], device=self.device, dtype=torch.float32)
        
    def G(self, z):
        """Metric with realistic curvature variation."""
        batch_size = z.shape[0]
        
        # Distance-based curvature
        distances = torch.cdist(z, self.centroids_tens)
        weights = torch.softmax(-distances / 1.0, dim=-1)
        
        # Different curvatures for different regions
        curvatures = torch.tensor([3.0, 1.5, 0.8, 2.0], device=z.device)
        local_curvature = torch.einsum('bk,k->b', weights, curvatures)
        
        # Create metric matrices
        I = torch.eye(2, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
        G = I * local_curvature.unsqueeze(-1).unsqueeze(-1)
        
        return G
    
    def G_inv(self, z):
        """Inverse metric tensor."""
        return torch.linalg.inv(self.G(z))


def analyze_density_variation_sources():
    """Analyze what causes density variation and whether it's problematic."""
    print("🔍 Analyzing Density Variation Sources")
    print("="*60)
    
    model = AnalysisModel()
    
    # Test different sampling strategies
    strategies = {
        'uniform_grid': create_uniform_grid_samples,
        'centroid_focused': create_centroid_focused_samples,
        'random_exploration': create_random_exploration_samples,
        'encoder_like': create_encoder_like_samples
    }
    
    results = {}
    
    for strategy_name, strategy_func in strategies.items():
        print(f"\n📊 Testing Strategy: {strategy_name}")
        
        samples = strategy_func(model)
        density_analysis = analyze_sample_densities(samples, model, strategy_name)
        results[strategy_name] = density_analysis
        
        print(f"   Density std: {density_analysis['std']:.4f}")
        print(f"   Density range: {density_analysis['range']:.4f}")
        print(f"   Coefficient of variation: {density_analysis['cv']:.4f}")
    
    # Create comprehensive visualization
    create_density_analysis_plots(results, model)
    
    # Conclusions
    print("\n" + "="*60)
    print("DENSITY VARIATION ANALYSIS CONCLUSIONS")
    print("="*60)
    
    # Find the most "natural" variation
    encoder_cv = results['encoder_like']['cv']
    uniform_cv = results['uniform_grid']['cv']
    
    print(f"📈 Natural encoder-like variation: {encoder_cv:.4f}")
    print(f"📈 Uniform exploration variation: {uniform_cv:.4f}")
    
    if encoder_cv < 0.15:
        print("✅ Encoder-like sampling has low variation - this is the target")
        target_std = results['encoder_like']['std']
        print(f"🎯 Recommended target density std: {target_std:.4f}")
    else:
        print("⚠️ Even encoder-like sampling has high variation")
        print("💡 This suggests the manifold itself has high curvature variation")
    
    # Practical recommendations
    print(f"\n💡 PRACTICAL RECOMMENDATIONS:")
    
    if results['encoder_like']['std'] < 0.1:
        print(f"   ✅ Target density std < {results['encoder_like']['std']:.3f} is achievable")
    elif results['encoder_like']['std'] < 0.2:
        print(f"   ⚠️ Target density std < {results['encoder_like']['std']:.3f} is reasonable")
    else:
        print(f"   ❌ Target density std < 0.1 may be too strict for this manifold")
        print(f"   💡 Consider relaxing target to < {results['encoder_like']['std'] * 1.2:.3f}")
    
    return results


def create_uniform_grid_samples(model):
    """Create uniform grid samples for baseline."""
    x = torch.linspace(-3, 3, 20, device=model.device)
    y = torch.linspace(-3, 3, 20, device=model.device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    samples = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    return samples


def create_centroid_focused_samples(model):
    """Create samples focused around centroids (ideal case)."""
    samples = []
    n_per_centroid = 100
    
    for centroid in model.centroids_tens:
        # Gaussian samples around each centroid
        noise = torch.randn(n_per_centroid, 2, device=model.device) * 0.3
        centroid_samples = centroid.unsqueeze(0) + noise
        samples.append(centroid_samples)
    
    return torch.cat(samples, dim=0)


def create_random_exploration_samples(model):
    """Create random exploration samples (worst case)."""
    return torch.randn(400, 2, device=model.device) * 2.5


def create_encoder_like_samples(model):
    """Create samples that mimic encoder output distribution."""
    samples = []
    
    # 70% near centroids (high-confidence encoder outputs)
    n_near = 280
    for i in range(n_near):
        centroid_idx = np.random.randint(0, len(model.centroids_tens))
        centroid = model.centroids_tens[centroid_idx]
        noise = torch.randn(2, device=model.device) * 0.4
        sample = centroid + noise
        samples.append(sample)
    
    # 30% in between regions (uncertain encoder outputs)
    n_between = 120
    for i in range(n_between):
        # Random interpolation between centroids
        idx1, idx2 = np.random.choice(len(model.centroids_tens), 2, replace=False)
        alpha = np.random.uniform(0.2, 0.8)
        interpolated = alpha * model.centroids_tens[idx1] + (1-alpha) * model.centroids_tens[idx2]
        noise = torch.randn(2, device=model.device) * 0.2
        sample = interpolated + noise
        samples.append(sample)
    
    return torch.stack(samples)


def analyze_sample_densities(samples, model, strategy_name):
    """Analyze density statistics for a set of samples."""
    with torch.no_grad():
        try:
            G_inv = model.G_inv(samples)
            log_densities = torch.logdet(G_inv)
            
            std = torch.std(log_densities).item()
            mean = torch.mean(log_densities).item()
            min_val = torch.min(log_densities).item()
            max_val = torch.max(log_densities).item()
            
            return {
                'std': std,
                'mean': mean,
                'min': min_val,
                'max': max_val,
                'range': max_val - min_val,
                'cv': std / abs(mean) if abs(mean) > 1e-8 else float('inf'),  # Coefficient of variation
                'samples': samples.cpu().numpy(),
                'densities': log_densities.cpu().numpy()
            }
        except Exception as e:
            print(f"⚠️ Density analysis failed for {strategy_name}: {e}")
            return {
                'std': float('inf'),
                'mean': 0,
                'min': 0,
                'max': 0,
                'range': 0,
                'cv': float('inf'),
                'samples': samples.cpu().numpy(),
                'densities': np.zeros(len(samples))
            }


def create_density_analysis_plots(results, model):
    """Create comprehensive density analysis visualization."""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('Density Variation Analysis: Understanding the 32% Problem', fontsize=16)
    
    centroids_np = model.centroids_tens.cpu().numpy()
    
    # Plot 1-4: Different sampling strategies
    strategy_names = list(results.keys())
    
    for i, (strategy_name, data) in enumerate(results.items()):
        if i >= 4:
            break
        
        row = i // 2
        col = i % 2
        ax = axes[row, col] if row < 2 else None
        
        if ax is None:
            continue
        
        samples = data['samples']
        densities = data['densities']
        
        # Scatter plot with density coloring
        scatter = ax.scatter(samples[:, 0], samples[:, 1], 
                           c=densities, cmap='viridis', alpha=0.6, s=15)
        
        # Add centroids
        ax.scatter(centroids_np[:, 0], centroids_np[:, 1], 
                  c='red', marker='*', s=200, edgecolors='black', 
                  label='Centroids', alpha=0.9)
        
        plt.colorbar(scatter, ax=ax, label='log det(G⁻¹)')
        
        ax.set_title(f'{strategy_name.replace("_", " ").title()}\n'
                    f'Density Std: {data["std"]:.4f} | CV: {data["cv"]:.3f}')
        ax.set_xlabel('z₁')
        ax.set_ylabel('z₂')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 5: Density distribution comparison
    ax = axes[1, 2]
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, (strategy_name, data) in enumerate(results.items()):
        densities = data['densities']
        if len(densities) > 0 and not np.all(densities == 0):
            ax.hist(densities, bins=30, alpha=0.6, label=strategy_name.replace('_', ' '), 
                   color=colors[i % len(colors)], density=True)
    
    ax.set_title('Density Distribution Comparison')
    ax.set_xlabel('log det(G⁻¹)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Summary metrics
    ax = axes[0, 2]
    
    metrics = ['Std', 'Range', 'CV']
    strategy_data = {}
    
    for strategy_name, data in results.items():
        strategy_data[strategy_name] = [data['std'], data['range'], data['cv']]
    
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, (strategy_name, values) in enumerate(strategy_data.items()):
        # Clip CV for visualization
        clipped_values = [min(val, 2.0) if not np.isinf(val) else 2.0 for val in values]
        ax.bar(x + i * width, clipped_values, width, 
               label=strategy_name.replace('_', ' '), alpha=0.8)
    
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics)
    ax.set_title('Density Variation Metrics')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add target line for std
    ax.axhline(0.1, color='red', linestyle='--', alpha=0.8, label='Target (0.1)')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("outputs/density_variation_analysis.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Density analysis plots saved to {output_path}")
    
    plt.show()


def main():
    """Run density variation analysis."""
    print("🚀 Density Variation Meaning Analysis")
    print("="*60)
    
    try:
        results = analyze_density_variation_sources()
        
        # Determine if 32% is actually problematic
        encoder_std = results['encoder_like']['std']
        
        print(f"\n🎯 FINAL ASSESSMENT:")
        
        if encoder_std <= 0.1:
            print(f"✅ 32% variation IS problematic - encoder-like sampling achieves {encoder_std:.4f}")
            print(f"💡 RHMC should be able to achieve similar low variation")
        elif encoder_std <= 0.2:
            print(f"⚠️ 32% variation is MODERATELY problematic - encoder-like achieves {encoder_std:.4f}")
            print(f"💡 Target should be relaxed to ~{encoder_std * 1.5:.3f}")
        else:
            print(f"❌ 32% variation may be ACCEPTABLE - even encoder-like sampling has {encoder_std:.4f}")
            print(f"💡 The manifold itself has high curvature variation")
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
