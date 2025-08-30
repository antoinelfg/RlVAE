#!/usr/bin/env python3
"""
RHVAE Sprites Metric Analysis - Fixed Model
===========================================

Analyze the fixed RHVAE model's metric tensor and geometry.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pythae.models import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_trained_model():
    """Load the trained RHVAE model."""
    model_dir = "rhvae_sprites_metric_fixed/RHVAE_training_*/final_model"
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

def analyze_metric_tensor(model, n_samples=200):
    """Analyze the metric tensor across the latent space."""
    print("🔍 Analyzing metric tensor...")
    
    # Sample random latent points
    z_samples = torch.randn(n_samples, model.latent_dim).to(device)
    
    # Compute metric tensors
    metric_determinants = []
    metric_eigenvalues = []
    metric_traces = []
    metric_condition_numbers = []
    
    with torch.no_grad():
        for i, z in enumerate(z_samples):
            try:
                # Get the inverse metric tensor directly
                G_inv = model.G_inv(z.unsqueeze(0))  # Shape: [1, latent_dim, latent_dim]
                G_inv = G_inv.squeeze(0)  # Shape: [latent_dim, latent_dim]
                
                # Compute determinant of inverse metric
                det_G_inv = torch.det(G_inv)
                metric_determinants.append(det_G_inv.item())
                
                # Compute trace of inverse metric
                trace_G_inv = torch.trace(G_inv)
                metric_traces.append(trace_G_inv.item())
                
                # Compute eigenvalues and condition number
                eigenvals = torch.linalg.eigvals(G_inv)
                eigenvals_real = eigenvals.real
                metric_eigenvalues.append(eigenvals_real.cpu())
                
                # Condition number (ratio of largest to smallest eigenvalue)
                condition_number = torch.max(eigenvals_real) / torch.min(eigenvals_real)
                metric_condition_numbers.append(condition_number.item())
                
                if i % 50 == 0:
                    print(f"  Processed {i}/{n_samples} points...")
                
            except Exception as e:
                print(f"⚠️ Failed to compute metric for point {i}: {e}")
                continue
    
    if not metric_determinants:
        print("❌ No valid metric computations found!")
        return
    
    # Create comprehensive analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Metric determinant distribution
    det_values = np.array(metric_determinants)
    axes[0, 0].hist(det_values, bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('det(G_inv)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Metric Inverse Determinant Distribution')
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Log metric determinant
    log_det_values = np.log(det_values + 1e-8)
    axes[0, 1].hist(log_det_values, bins=30, alpha=0.7, color='green')
    axes[0, 1].set_xlabel('log(det(G_inv))')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Log Metric Inverse Determinant Distribution')
    
    # Plot 3: Metric trace distribution
    trace_values = np.array(metric_traces)
    axes[0, 2].hist(trace_values, bins=30, alpha=0.7, color='red')
    axes[0, 2].set_xlabel('trace(G_inv)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Metric Inverse Trace Distribution')
    
    # Plot 4: Condition number distribution
    condition_values = np.array(metric_condition_numbers)
    axes[1, 0].hist(condition_values, bins=30, alpha=0.7, color='purple')
    axes[1, 0].set_xlabel('Condition Number')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Metric Condition Number Distribution')
    axes[1, 0].set_yscale('log')
    
    # Plot 5: Eigenvalue distribution
    all_eigenvals = torch.cat(metric_eigenvalues, dim=0).numpy()
    axes[1, 1].hist(all_eigenvals, bins=40, alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Eigenvalue')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Metric Eigenvalue Distribution')
    
    # Plot 6: Determinant vs Trace scatter
    scatter = axes[1, 2].scatter(trace_values, det_values, alpha=0.6, c=condition_values, cmap='viridis')
    axes[1, 2].set_xlabel('trace(G_inv)')
    axes[1, 2].set_ylabel('det(G_inv)')
    axes[1, 2].set_title('Metric Properties Correlation')
    plt.colorbar(scatter, ax=axes[1, 2], label='Condition Number')
    
    plt.tight_layout()
    plt.savefig('rhvae_sprites_metric_fixed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Metric analysis saved to rhvae_sprites_metric_fixed_analysis.png")
    print(f"📊 Metric statistics:")
    print(f"   - det(G_inv) range: [{det_values.min():.2e}, {det_values.max():.2e}]")
    print(f"   - trace(G_inv) range: [{trace_values.min():.2f}, {trace_values.max():.2f}]")
    print(f"   - condition number range: [{condition_values.min():.2f}, {condition_values.max():.2f}]")
    print(f"   - coefficient of variation (det): {det_values.std()/det_values.mean():.3f}")
    
    # Check if metric is well-conditioned
    if condition_values.mean() < 100:
        print("✅ Good! Metric is well-conditioned (condition number < 100)")
    else:
        print("⚠️ Metric is still ill-conditioned (condition number >= 100)")
    
    if det_values.mean() > 1e-10:
        print("✅ Good! Metric determinant is reasonable (> 1e-10)")
    else:
        print("⚠️ Metric determinant is still too small (< 1e-10)")

def main():
    """Main analysis function."""
    print("🔍 RHVAE Sprites Metric Analysis - Fixed Model")
    print("=" * 50)
    
    model = load_trained_model()
    analyze_metric_tensor(model)

if __name__ == "__main__":
    main()
