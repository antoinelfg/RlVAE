#!/usr/bin/env python3
"""
RHVAE Sprites Metric Analysis
=============================

Analyze the trained RHVAE model's metric tensor and geometry.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pythae.models import AutoModel
import torchvision.transforms as transforms

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

def analyze_metric_tensor(model, n_samples=200):
    """Analyze the metric tensor across the latent space."""
    print("🔍 Analyzing metric tensor...")
    
    # Sample random latent points
    z_samples = torch.randn(n_samples, model.latent_dim).to(device)
    
    # Compute metric tensors
    metric_determinants = []
    metric_eigenvalues = []
    metric_traces = []
    metric_inv_determinants = []
    
    with torch.no_grad():
        for i, z in enumerate(z_samples):
            try:
                # Get the inverse metric tensor directly
                G_inv = model.G_inv(z.unsqueeze(0))  # Shape: [1, latent_dim, latent_dim]
                G_inv = G_inv.squeeze(0)  # Shape: [latent_dim, latent_dim]
                
                # Compute determinant of inverse metric
                det_G_inv = torch.det(G_inv)
                metric_inv_determinants.append(det_G_inv.item())
                
                # Compute trace of inverse metric
                trace_G_inv = torch.trace(G_inv)
                metric_traces.append(trace_G_inv.item())
                
                # Try to compute eigenvalues of inverse metric
                try:
                    eigenvals = torch.linalg.eigvals(G_inv)
                    metric_eigenvalues.append(eigenvals.cpu())
                except:
                    print(f"⚠️ Failed to compute eigenvalues for point {i}")
                    continue
                
                # Try to invert to get the actual metric tensor
                try:
                    G = torch.inverse(G_inv)
                    det_G = torch.det(G)
                    metric_determinants.append(det_G.item())
                except:
                    print(f"⚠️ Failed to invert metric for point {i}, using inverse determinant")
                    # If inversion fails, use the inverse of the inverse determinant
                    metric_determinants.append(1.0 / det_G_inv.item() if det_G_inv.item() != 0 else 0.0)
                
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
    axes[0, 0].set_xlabel('det(G)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Metric Determinant Distribution')
    if det_values.min() > 0:
        axes[0, 0].set_yscale('log')
    
    # Plot 2: Log metric determinant
    log_det_values = np.log(det_values + 1e-8)
    axes[0, 1].hist(log_det_values, bins=30, alpha=0.7, color='green')
    axes[0, 1].set_xlabel('log(det(G))')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Log Metric Determinant Distribution')
    
    # Plot 3: Metric trace distribution
    trace_values = np.array(metric_traces)
    axes[0, 2].hist(trace_values, bins=30, alpha=0.7, color='red')
    axes[0, 2].set_xlabel('trace(G_inv)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Metric Inverse Trace Distribution')
    
    # Plot 4: Eigenvalue distribution
    if metric_eigenvalues:
        all_eigenvals = torch.cat(metric_eigenvalues, dim=0).numpy()
        axes[1, 0].hist(all_eigenvals.real, bins=40, alpha=0.7, color='purple')
        axes[1, 0].set_xlabel('Eigenvalue (Real Part)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Metric Inverse Eigenvalue Distribution')
    else:
        axes[1, 0].text(0.5, 0.5, 'No eigenvalues computed', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Metric Eigenvalue Distribution')
    
    # Plot 5: Metric determinant vs latent dimension 1
    latent_dim1 = z_samples[:len(det_values), 0].cpu().numpy()
    scatter = axes[1, 1].scatter(latent_dim1, det_values, alpha=0.6, c=det_values, cmap='viridis')
    axes[1, 1].set_xlabel('Latent Dimension 1')
    axes[1, 1].set_ylabel('det(G)')
    axes[1, 1].set_title('Metric Determinant vs Latent Dim 1')
    plt.colorbar(scatter, ax=axes[1, 1])
    
    # Plot 6: Metric determinant vs latent dimension 2
    latent_dim2 = z_samples[:len(det_values), 1].cpu().numpy()
    scatter = axes[1, 2].scatter(latent_dim2, det_values, alpha=0.6, c=det_values, cmap='viridis')
    axes[1, 2].set_xlabel('Latent Dimension 2')
    axes[1, 2].set_ylabel('det(G)')
    axes[1, 2].set_title('Metric Determinant vs Latent Dim 2')
    plt.colorbar(scatter, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('rhvae_sprites_metric_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Metric analysis saved to rhvae_sprites_metric_analysis.png")
    print(f"📊 Metric determinant stats:")
    print(f"   - min: {det_values.min():.6f}")
    print(f"   - max: {det_values.max():.6f}")
    print(f"   - mean: {det_values.mean():.6f}")
    print(f"   - std: {det_values.std():.6f}")
    if det_values.mean() != 0:
        print(f"   - coefficient of variation: {det_values.std()/det_values.mean():.3f}")
    
    # Check if metric is varying
    if det_values.std() / det_values.mean() > 0.1:
        print("✅ Good! Metric determinant is varying significantly across latent space.")
    else:
        print("⚠️ Metric determinant is too uniform. Consider more aggressive training.")

def main():
    """Main analysis function."""
    print("🔍 RHVAE Sprites Metric Analysis")
    print("=" * 40)
    
    model = load_trained_model()
    analyze_metric_tensor(model)

if __name__ == "__main__":
    main()
