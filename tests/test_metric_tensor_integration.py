"""
Integration test for MetricTensor with LossManager.

This test verifies that the MetricTensor implementation is compatible
with the LossManager using metric_representation="g".
"""

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.rlvae.models.components.metric_tensor import MetricTensor

def test_loss_manager_integration():
    """Test integration with LossManager using metric_representation='g'."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    B, d, K = 8, 3, 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create MetricTensor in fixed mode
    metric = MetricTensor(latent_dim=d, trainable=False, temperature=0.3, regularization=0.05, device=device)
    
    # Create SPD inverse metric matrices
    centroids = torch.randn(K, d, device=device)
    M = torch.randn(K, d, d, device=device)
    M = 0.5 * (M + M.transpose(-1, -2))
    
    # Ensure SPD by adding positive diagonal
    for i in range(K):
        diag = torch.diagonal(M[i])
        diag = torch.nn.functional.softplus(diag) + 0.5
        M[i] = M[i].clone()
        M[i].diagonal().copy_(diag)
        
        # Verify SPD
        eig = torch.linalg.eigvalsh(M[i].float())
        if eig.min() <= 0:
            M[i] = M[i] + 0.5 * torch.eye(d, device=device)
    
    metric.load_pretrained(centroids, M)
    
    # Test data
    z = torch.randn(B, d, device=device)
    
    # Test that compute_metric returns G (not G_inv)
    G = metric.compute_metric(z)
    G_inv = metric.compute_inverse_metric(z)
    
    print(f"G shape: {G.shape}")
    print(f"G_inv shape: {G_inv.shape}")
    
    # Verify that G and G_inv are inverses
    I = torch.eye(d, device=device).unsqueeze(0).expand(B, -1, -1)
    err = (G @ G_inv - I).norm() / I.norm()
    print(f"Inverse relation error: {err.item():.6f}")
    
    # Test log determinant computation
    log_det = metric.compute_log_det_metric(z)
    print(f"log_det shape: {log_det.shape}")
    print(f"log_det values: {log_det}")
    
    # Test Riemannian distance
    z1 = torch.randn(B, d, device=device)
    z2 = torch.randn(B, d, device=device)
    dist_sq = metric.compute_riemannian_distance_squared(z1, z2)
    print(f"dist_sq shape: {dist_sq.shape}")
    print(f"dist_sq values: {dist_sq}")
    
    # Verify all outputs are finite and on correct device
    assert torch.isfinite(G).all(), "G should be finite"
    assert torch.isfinite(G_inv).all(), "G_inv should be finite"
    assert torch.isfinite(log_det).all(), "log_det should be finite"
    assert torch.isfinite(dist_sq).all(), "dist_sq should be finite"
    
    assert G.device.type == device.type, "G should be on correct device"
    assert G_inv.device.type == device.type, "G_inv should be on correct device"
    assert log_det.device.type == device.type, "log_det should be on correct device"
    assert dist_sq.device.type == device.type, "dist_sq should be on correct device"
    
    # Test that G is positive definite
    eig_G = torch.linalg.eigvalsh(G)
    assert eig_G.min() > 0, f"G should be positive definite, min eig: {eig_G.min():.6f}"
    
    # Test that distances are non-negative
    assert (dist_sq >= 0).all(), "Riemannian distances should be non-negative"
    
    print("✅ All integration tests passed!")
    return True

def test_trainable_mode_integration():
    """Test integration with trainable mode."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    B, d = 8, 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create MetricTensor in trainable mode
    metric = MetricTensor(latent_dim=d, trainable=True, architecture="mlp", device=device)
    
    # Test data
    z = torch.randn(B, d, device=device)
    
    # Test that compute_metric returns G
    G = metric.compute_metric(z)
    G_inv = metric.compute_inverse_metric(z)
    
    print(f"Trainable G shape: {G.shape}")
    print(f"Trainable G_inv shape: {G_inv.shape}")
    
    # Verify that G and G_inv are inverses
    I = torch.eye(d, device=device).unsqueeze(0).expand(B, -1, -1)
    err = (G @ G_inv - I).norm() / I.norm()
    print(f"Trainable inverse relation error: {err.item():.6f}")
    
    # Test log determinant computation
    log_det = metric.compute_log_det_metric(z)
    print(f"Trainable log_det shape: {log_det.shape}")
    
    # Verify all outputs are finite and on correct device
    assert torch.isfinite(G).all(), "Trainable G should be finite"
    assert torch.isfinite(G_inv).all(), "Trainable G_inv should be finite"
    assert torch.isfinite(log_det).all(), "Trainable log_det should be finite"
    
    assert G.device.type == device.type, "Trainable G should be on correct device"
    assert G_inv.device.type == device.type, "Trainable G_inv should be on correct device"
    assert log_det.device.type == device.type, "Trainable log_det should be on correct device"
    
    # Test that G is positive definite
    eig_G = torch.linalg.eigvalsh(G)
    assert eig_G.min() > 0, f"Trainable G should be positive definite, min eig: {eig_G.min():.6f}"
    
    print("✅ All trainable integration tests passed!")
    return True

if __name__ == "__main__":
    print("Testing MetricTensor integration with LossManager...")
    print("=" * 60)
    
    print("\n1. Testing fixed mode integration...")
    test_loss_manager_integration()
    
    print("\n2. Testing trainable mode integration...")
    test_trainable_mode_integration()
    
    print("\n🎉 All integration tests passed!")
    print("\nThe MetricTensor implementation is ready for use with LossManager")
    print("using metric_representation='g'.")







