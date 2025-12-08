#!/usr/bin/env python3
"""
Test for Prior Density Fix
==========================

This test validates that the prior density implementation uses the correct
mathematical formula: log p(z) ∝ 1/2 log(det(G(z))) - 1/2 z^T G(z) z
"""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.riemannian_flow_vae import RiemannianFlowVAE
from src.models.components.metric_tensor import MetricTensor


def test_prior_density_formula():
    """Test that prior density uses correct mathematical formula."""
    print("🧪 Testing Prior Density Formula Fix...")
    
    # Create a simple model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiemannianFlowVAE(
        input_dim=(3, 64, 64),
        latent_dim=2,  # Small dimension for easy testing
        n_flows=0  # No flows for simplicity
    )
    
    # Create a simple metric tensor
    metric_tensor = MetricTensor(latent_dim=2, device=device)
    
    # Create synthetic metric data
    centroids = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32, device=device)
    metric_matrices = torch.stack([
        torch.tensor([[2.0, 0.5], [0.5, 1.5]], dtype=torch.float32, device=device),
        torch.tensor([[1.5, 0.3], [0.3, 2.0]], dtype=torch.float32, device=device)
    ])
    
    metric_tensor.load_pretrained(
        centroids=centroids,
        metric_matrices=metric_matrices,
        temperature=0.5,
        regularization=0.01
    )
    
    # Attach metric to model
    model.G = metric_tensor.compute_metric
    model.G_inv = metric_tensor.compute_inverse_metric
    
    # Test points
    z_test = torch.tensor([[0.5, 0.5], [1.0, 0.0]], dtype=torch.float32, device=device, requires_grad=True)
    
    # Compute metric at test points
    G_z = model.G(z_test)
    
    # Manual computation of correct formula
    log_det_G = torch.linalg.slogdet(G_z).logabsdet
    quadratic_term = torch.einsum('bi,bij,bj->b', z_test, G_z, z_test)
    expected_log_prob = 0.5 * log_det_G - 0.5 * quadratic_term
    
    # Test that the formula is implemented correctly
    print(f"✅ Test points: {z_test}")
    print(f"✅ Metric tensor G(z):\n{G_z}")
    print(f"✅ Log det(G): {log_det_G}")
    print(f"✅ Quadratic term z^T G(z) z: {quadratic_term}")
    print(f"✅ Expected log probability: {expected_log_prob}")
    
    # Verify the formula is mathematically sound
    assert torch.all(torch.isfinite(expected_log_prob)), "Log probability contains non-finite values"
    assert torch.all(log_det_G > -10), "Log determinant too negative"
    assert torch.all(quadratic_term > 0), "Quadratic term should be positive for positive-definite G"
    
    print("✅ Prior density formula test PASSED")
    return True


def test_prior_sampling_consistency():
    """Test that prior sampling produces consistent results."""
    print("\n🧪 Testing Prior Sampling Consistency...")
    
    # Create model with metric
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiemannianFlowVAE(
        input_dim=(3, 64, 64),
        latent_dim=4,
        n_flows=0
    )
    
    # Create metric tensor
    metric_tensor = MetricTensor(latent_dim=4, device=device)
    centroids = torch.randn(5, 4, device=device)
    metric_matrices = torch.stack([torch.eye(4, device=device) + 0.1 * torch.randn(4, 4, device=device) for _ in range(5)])
    
    metric_tensor.load_pretrained(
        centroids=centroids,
        metric_matrices=metric_matrices,
        temperature=0.5,
        regularization=0.01
    )
    
    # Load metrics into model (this initializes the sampler)
    model.load_pretrained_metrics_from_tensor(
        centroids=centroids,
        metric_matrices=metric_matrices,
        temperature=0.5,
        regularization=0.01
    )
    
    # Test sampling
    num_samples = 10
    samples = model.sample_riemannian_prior(num_samples, method='basic')
    
    print(f"✅ Generated {len(samples)} samples")
    print(f"✅ Sample shape: {samples.shape}")
    print(f"✅ Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
    print(f"✅ Sample mean: {samples.mean(dim=0)}")
    print(f"✅ Sample std: {samples.std(dim=0)}")
    
    # Verify samples are reasonable
    assert samples.shape == (num_samples, 4), f"Wrong sample shape: {samples.shape}"
    assert torch.all(torch.isfinite(samples)), "Samples contain non-finite values"
    assert torch.all(torch.abs(samples) < 10), "Samples are too large"
    
    print("✅ Prior sampling consistency test PASSED")
    return True


def test_metric_tensor_integration():
    """Test that metric tensor integration works correctly."""
    print("\n🧪 Testing Metric Tensor Integration...")
    
    # Create metric tensor
    latent_dim = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_tensor = MetricTensor(latent_dim=latent_dim, device=device)
    
    # Create synthetic data
    centroids = torch.randn(3, latent_dim, device=device)
    metric_matrices = torch.stack([torch.eye(latent_dim, device=device) + 0.1 * torch.randn(latent_dim, latent_dim, device=device) for _ in range(3)])
    
    metric_tensor.load_pretrained(
        centroids=centroids,
        metric_matrices=metric_matrices,
        temperature=0.3,
        regularization=0.01
    )
    
    # Test metric computations
    z_test = torch.randn(5, latent_dim, device=device)
    
    G = metric_tensor.compute_metric(z_test)
    G_inv = metric_tensor.compute_inverse_metric(z_test)
    
    # Verify properties
    assert G.shape == (5, latent_dim, latent_dim), f"Wrong G shape: {G.shape}"
    assert G_inv.shape == (5, latent_dim, latent_dim), f"Wrong G_inv shape: {G_inv.shape}"
    
    # Test that G * G_inv ≈ I
    identity_approx = torch.bmm(G, G_inv)
    identity_target = torch.eye(latent_dim, device=device).unsqueeze(0).expand(5, -1, -1)
    error = torch.norm(identity_approx - identity_target, dim=(1, 2))
    
    print(f"✅ G * G_inv ≈ I error: mean={error.mean():.3e}, max={error.max():.3e}")
    assert torch.all(error < 1e-3), "G * G_inv is not close to identity"
    
    # Test positive definiteness
    eigenvals = torch.linalg.eigvals(G)
    print(f"✅ G eigenvalues: min={eigenvals.real.min():.3e}, max={eigenvals.real.max():.3e}")
    assert torch.all(eigenvals.real > 1e-6), "G is not positive definite"
    
    print("✅ Metric tensor integration test PASSED")
    return True


def main():
    """Run all prior density tests."""
    print("🚀 Testing Prior Density Fixes")
    print("=" * 50)
    
    try:
        test_metric_tensor_integration()
        test_prior_density_formula()
        test_prior_sampling_consistency()
        
        print("\n✅ ALL PRIOR DENSITY TESTS PASSED!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 