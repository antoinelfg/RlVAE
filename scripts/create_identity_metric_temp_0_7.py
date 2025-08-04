#!/usr/bin/env python3
"""
Create Identity Metric (Temperature 0.7)
======================================

Creates a basic identity metric tensor for testing the RlVAE evaluation system.
This generates the metric file that the system expects.
"""

import torch
import numpy as np
from pathlib import Path


def create_identity_metric(latent_dim: int = 16, n_centroids: int = 100, temperature: float = 0.7):
    """
    Create a simple identity-based metric for testing.
    
    Args:
        latent_dim: Dimensionality of the latent space
        n_centroids: Number of centroids in the metric
        temperature: Temperature parameter for the metric
    
    Returns:
        Dictionary containing metric components
    """
    print(f"🔧 Creating identity metric...")
    print(f"   Latent dim: {latent_dim}")
    print(f"   Centroids: {n_centroids}")
    print(f"   Temperature: {temperature}")
    
    # Create identity matrices for each centroid
    M_matrices = torch.eye(latent_dim).unsqueeze(0).repeat(n_centroids, 1, 1)
    
    # Scale by temperature
    M_matrices = M_matrices / temperature
    
    # Create random centroids in latent space
    centroids = torch.randn(n_centroids, latent_dim) * 0.5
    
    # Create uniform weights
    weights = torch.ones(n_centroids) / n_centroids
    
    metric_data = {
        'M_matrices': M_matrices,  # [n_centroids, latent_dim, latent_dim]
        'centroids': centroids,    # [n_centroids, latent_dim]
        'weights': weights,        # [n_centroids]
        'temperature': temperature,
        'latent_dim': latent_dim,
        'n_centroids': n_centroids,
        'metric_type': 'identity',
        'description': f'Identity metric with temperature {temperature}'
    }
    
    return metric_data


def main():
    """Create and save identity metric."""
    print("🚀 Creating Identity Metric (Temperature 0.7)")
    print("=" * 45)
    
    # Create metric
    metric_data = create_identity_metric(
        latent_dim=16,
        n_centroids=100, 
        temperature=0.7
    )
    
    # Create pretrained directory
    pretrained_dir = Path("data/pretrained")
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metric
    metric_path = pretrained_dir / "metric_T0.7_scaled.pt"
    print(f"\n💾 Saving metric to: {metric_path}")
    torch.save(metric_data, metric_path)
    
    # Verify the saved metric
    print(f"\n✅ Verifying saved metric...")
    loaded = torch.load(metric_path, weights_only=False)
    print(f"   M_matrices shape: {loaded['M_matrices'].shape}")
    print(f"   Centroids shape: {loaded['centroids'].shape}")
    print(f"   Weights shape: {loaded['weights'].shape}")
    print(f"   Temperature: {loaded['temperature']}")
    
    print(f"\n✅ Identity metric created successfully!")


if __name__ == "__main__":
    main() 