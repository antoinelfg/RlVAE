#!/usr/bin/env python3
"""
Test Adaptive Pipeline
======================

Quick test of the adaptive RLVAE pipeline with minimal epochs to verify functionality.
"""

import subprocess
import sys
from pathlib import Path

def test_adaptive_pipeline():
    """Test the adaptive pipeline with minimal settings."""
    
    cmd = [
        sys.executable, "scripts/adaptive_global_rlvae_pipeline.py",
        "--architecture", "mlp",
        "--latent-dim", "2", 
        "--vae-epochs", "5",           # Very quick VAE training
        "--rlvae-epochs", "6",         # Quick RLVAE training (divisible by update freq)
        "--centroid-update-freq", "2", # Update every 2 epochs
        "--n-samples-for-centroids", "200",  # Fewer samples for speed
        "--rlvae-batch-size", "4",     # Smaller batches
        "--n-train-samples", "400",    # Fewer training samples
        "--n-val-samples", "100",      # Fewer validation samples
        "--visualization-level", "minimal",  # Minimal visualizations
        "--wandb",                     # Enable wandb
        "--skip-analysis"              # Skip analysis for speed
    ]
    
    print("🧪 Testing Adaptive Pipeline with minimal settings...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Adaptive pipeline test completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline test failed: {e}")
        raise

if __name__ == "__main__":
    test_adaptive_pipeline() 