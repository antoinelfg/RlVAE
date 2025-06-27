#!/usr/bin/env python3
"""
Test Script for Modular RiemannianFlowVAE Components

This script demonstrates and validates the new modular components extracted
from the monolithic riemannian_flow_vae.py implementation.

It tests:
1. MetricTensor class functionality
2. MetricLoader class functionality  
3. Compatibility with existing pretrained metrics
4. Performance comparison with original implementation
"""

import torch
import numpy as np
from pathlib import Path
import time
from typing import Dict, Any

# Import the new modular components
from src.models.components.metric_tensor import MetricTensor
from src.models.components.metric_loader import MetricLoader

# Import original implementation for comparison
from src.models.riemannian_flow_vae import RiemannianFlowVAE


def test_metric_loader():
    """Test the MetricLoader functionality."""
    print("🧪 Testing MetricLoader...")
    
    loader = MetricLoader()
    
    # Test loading pretrained metrics
    metric_path = "data/pretrained/metric.pt"
    if Path(metric_path).exists():
        try:
            data = loader.load_from_file(metric_path, temperature_override=0.7)
            
            print(f"✅ Loaded metric data:")
            print(f"   Centroids: {data['centroids'].shape}")
            print(f"   Matrices: {data['metric_matrices'].shape}")
            print(f"   Temperature: {data['temperature']}")
            print(f"   Regularization: {data['regularization']}")
            
            # Validate the file
            report = loader.validate_metric_file(metric_path)
            print(f"✅ Validation report: {report}")
            
            return data
            
        except Exception as e:
            print(f"❌ MetricLoader test failed: {e}")
            return None
    else:
        print(f"⚠️ Metric file not found: {metric_path}")
        return None


def test_metric_tensor(metric_data: Dict[str, Any]):
    """Test the MetricTensor functionality."""
    print("\n🧪 Testing MetricTensor...")
    
    if metric_data is None:
        print("⚠️ No metric data available, creating synthetic data")
        # Create synthetic data for testing
        latent_dim = 16
        n_centroids = 10
        
        centroids = torch.randn(n_centroids, latent_dim)
        metric_matrices = torch.stack([torch.eye(latent_dim) for _ in range(n_centroids)])
        temperature = 0.1
        regularization = 0.01
        
        metric_data = {
            'centroids': centroids,
            'metric_matrices': metric_matrices, 
            'temperature': temperature,
            'regularization': regularization,
        }
    
    # Create MetricTensor instance
    latent_dim = metric_data['centroids'].shape[1]
    metric_tensor = MetricTensor(latent_dim=latent_dim)
    
    # Load pretrained data
    metric_tensor.load_pretrained(
        centroids=metric_data['centroids'],
        metric_matrices=metric_data['metric_matrices'],
        temperature=metric_data['temperature'],
        regularization=metric_data['regularization']
    )
    
    # Test computations
    batch_size = 8
    z_test = torch.randn(batch_size, latent_dim, device=metric_tensor.device)
    
    print(f"Testing with batch_size={batch_size}, latent_dim={latent_dim}")
    
    # Test inverse metric computation
    start_time = time.time()
    G_inv = metric_tensor.compute_inverse_metric(z_test)
    inv_time = time.time() - start_time
    print(f"✅ G_inv computed: {G_inv.shape}, time: {inv_time:.4f}s")
    
    # Test metric computation
    start_time = time.time()
    G = metric_tensor.compute_metric(z_test)
    metric_time = time.time() - start_time
    print(f"✅ G computed: {G.shape}, time: {metric_time:.4f}s")
    
    # Test log determinant
    start_time = time.time()
    log_det = metric_tensor.compute_log_det_metric(z_test)
    logdet_time = time.time() - start_time
    print(f"✅ log|G| computed: {log_det.shape}, time: {logdet_time:.4f}s")
    
    # Test Riemannian distance
    z1 = torch.randn(batch_size, latent_dim, device=metric_tensor.device)
    z2 = torch.randn(batch_size, latent_dim, device=metric_tensor.device)
    distance_sq = metric_tensor.compute_riemannian_distance_squared(z1, z2)
    print(f"✅ Riemannian distance² computed: {distance_sq.shape}")
    
    # Diagnostic analysis
    diagnostics = metric_tensor.diagnose_metric_properties(z_test, verbose=True)
    
    # Verify G * G_inv ≈ I
    identity_approx = torch.bmm(G, G_inv)
    identity_target = torch.eye(latent_dim, device=metric_tensor.device).unsqueeze(0)
    identity_error = torch.norm(identity_approx - identity_target, dim=(1, 2))
    print(f"✅ G * G_inv ≈ I error: mean={identity_error.mean():.3e}, max={identity_error.max():.3e}")
    
    return metric_tensor


def test_compatibility_with_original(metric_tensor: MetricTensor):
    """Test compatibility with original RiemannianFlowVAE implementation."""
    print("\n🧪 Testing compatibility with original implementation...")
    
    if not Path("data/pretrained/metric.pt").exists():
        print("⚠️ Skipping compatibility test (no pretrained metric file)")
        return
    
    # Create original model
    original_model = RiemannianFlowVAE(
        input_dim=(3, 64, 64),
        latent_dim=16,
        n_flows=8
    )
    
    # Load pretrained components in original model
    try:
        original_model.load_pretrained_components(
            encoder_path="data/pretrained/encoder.pt",
            decoder_path="data/pretrained/decoder.pt", 
            metric_path="data/pretrained/metric.pt",
            temperature_override=0.7
        )
        
        # Test batch
        batch_size = 4
        z_test = torch.randn(batch_size, 16, device=original_model.device)
        
        # Compare metric computations
        print("Comparing metric computations...")
        
        # Original implementation
        start_time = time.time()
        G_orig = original_model.G(z_test)
        G_inv_orig = original_model.G_inv(z_test)
        orig_time = time.time() - start_time
        
        # Modular implementation
        start_time = time.time()
        G_mod = metric_tensor.compute_metric(z_test)
        G_inv_mod = metric_tensor.compute_inverse_metric(z_test)
        mod_time = time.time() - start_time
        
        # Compare results
        G_diff = torch.norm(G_orig - G_mod)
        G_inv_diff = torch.norm(G_inv_orig - G_inv_mod)
        
        print(f"✅ Metric tensor comparison:")
        print(f"   G difference: {G_diff:.3e}")
        print(f"   G_inv difference: {G_inv_diff:.3e}")
        print(f"   Original time: {orig_time:.4f}s")
        print(f"   Modular time: {mod_time:.4f}s")
        print(f"   Speedup: {orig_time/mod_time:.2f}x" if mod_time > 0 else "N/A")
        
        # Test numerical accuracy
        if G_diff < 1e-6 and G_inv_diff < 1e-6:
            print("✅ NUMERICAL ACCURACY: PASSED")
        else:
            print("❌ NUMERICAL ACCURACY: FAILED")
            
    except Exception as e:
        print(f"⚠️ Compatibility test failed: {e}")


def benchmark_performance(metric_tensor: MetricTensor):
    """Benchmark the performance of modular components."""
    print("\n🧪 Benchmarking performance...")
    
    batch_sizes = [1, 4, 16, 64]
    n_iterations = 100
    
    print(f"Running {n_iterations} iterations for each batch size...")
    
    for batch_size in batch_sizes:
        z_test = torch.randn(batch_size, metric_tensor.latent_dim, device=metric_tensor.device)
        
        # Benchmark inverse metric
        start_time = time.time()
        for _ in range(n_iterations):
            G_inv = metric_tensor.compute_inverse_metric(z_test)
        inv_time = (time.time() - start_time) / n_iterations
        
        # Benchmark metric
        start_time = time.time()
        for _ in range(n_iterations):
            G = metric_tensor.compute_metric(z_test)
        metric_time = (time.time() - start_time) / n_iterations
        
        # Benchmark log determinant
        start_time = time.time()
        for _ in range(n_iterations):
            log_det = metric_tensor.compute_log_det_metric(z_test)
        logdet_time = (time.time() - start_time) / n_iterations
        
        print(f"   Batch size {batch_size:2d}: G_inv={inv_time*1000:.2f}ms, G={metric_time*1000:.2f}ms, log|G|={logdet_time*1000:.2f}ms")


def main():
    """Main test function."""
    print("🚀 Testing Modular RiemannianFlowVAE Components")
    print("=" * 60)
    
    # Test metric loader
    metric_data = test_metric_loader()
    
    # Test metric tensor
    metric_tensor = test_metric_tensor(metric_data)
    
    # Test compatibility
    test_compatibility_with_original(metric_tensor)
    
    # Benchmark performance
    benchmark_performance(metric_tensor)
    
    print("\n✅ All tests completed!")
    print("=" * 60)
    
    # Summary
    config = metric_tensor.get_config()
    print("📊 MODULAR COMPONENTS SUMMARY:")
    print(f"   Latent dimension: {config['latent_dim']}")
    print(f"   Number of centroids: {config['n_centroids']}")
    print(f"   Temperature: {config['temperature']}")
    print(f"   Regularization: {config['regularization']}")
    print(f"   Status: {'✅ Loaded' if config['is_loaded'] else '❌ Not loaded'}")


if __name__ == "__main__":
    main() 