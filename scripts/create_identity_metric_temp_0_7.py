#!/usr/bin/env python3

import torch
import numpy as np

def create_identity_metric_temp_07():
    """Create an identity metric with temperature 0.7 for testing."""
    
    # Load the reference metric to get the right structure
    orig = torch.load('src/datasprites/vanilla_vae_components/metric_T0.7_scaled.pt')
    print(f"Original metric shape: {orig['metric_vars'].shape}")
    print(f"Original temperature: {orig['metric_temperature']}")
    
    # Create identity matrices with the same shape
    n_components, latent_dim, _ = orig['metric_vars'].shape
    identity_metric = torch.eye(latent_dim).unsqueeze(0).repeat(n_components, 1, 1)
    
    # Create the new metric dict
    new_metric = {
        'metric_centroids': orig['metric_centroids'].clone(),  # Keep same centroids
        'metric_vars': identity_metric,  # Use identity matrices
        'metric_temperature': 0.7  # Set temperature to 0.7
    }
    
    # Verify it's identity
    print(f"New metric shape: {new_metric['metric_vars'].shape}")
    print(f"Is identity? {torch.allclose(new_metric['metric_vars'], torch.eye(latent_dim))}")
    print(f"New temperature: {new_metric['metric_temperature']}")
    
    # Save
    output_path = 'src/datasprites/vanilla_vae_components/metric_identity_temp_0.7.pt'
    torch.save(new_metric, output_path)
    print(f"✅ Saved identity metric with T=0.7 to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    create_identity_metric_temp_07() 