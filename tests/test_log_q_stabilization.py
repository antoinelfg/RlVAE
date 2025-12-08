#!/usr/bin/env python3
"""
Test script to verify log_q_riem stabilization diagnostics.
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.rlvae.models.components.riemannian_rhmc_posterior import log_q_riem

# Enable debug output
os.environ["RLVAE_DEBUG"] = "1"

def test_log_q_riem_no_stabilization():
    """Test when matrix is already well-conditioned (no jitter needed)."""
    print("\n" + "="*80)
    print("TEST 1: Well-conditioned Sigma (no stabilization expected)")
    print("="*80)
    
    # Well-conditioned covariance
    Sigma = torch.tensor([
        [[1.0, 0.0],
         [0.0, 1.0]]
    ])
    mu = torch.zeros(1, 2)
    z = torch.randn(1, 2)
    
    log_q = log_q_riem(z, mu, Sigma, min_eig=1e-3)
    print(f"\nResult: log_q = {log_q.item():.6f}")
    print("Expected: No stabilization message\n")

def test_log_q_riem_with_stabilization():
    """Test when matrix is poorly conditioned (jitter needed)."""
    print("\n" + "="*80)
    print("TEST 2: Poorly-conditioned Sigma (stabilization expected)")
    print("="*80)
    
    # Poorly conditioned covariance (one very small eigenvalue)
    Sigma = torch.tensor([
        [[1.0, 0.999],
         [0.999, 1.0]]
    ])
    mu = torch.zeros(1, 2)
    z = torch.randn(1, 2)
    
    print(f"\nOriginal Sigma eigenvalues: {torch.linalg.eigvalsh(Sigma).numpy()}")
    
    log_q = log_q_riem(z, mu, Sigma, min_eig=1e-3)
    print(f"\nResult: log_q = {log_q.item():.6f}")
    print("Expected: Stabilization should trigger\n")

def test_log_q_riem_large_batch():
    """Test with a realistic batch size."""
    print("\n" + "="*80)
    print("TEST 3: Realistic batch with varied conditioning")
    print("="*80)
    
    B = 8
    D = 2
    
    # Create batch with some well-conditioned and some poorly-conditioned matrices
    Sigma = torch.zeros(B, D, D)
    for i in range(B):
        if i < B // 2:
            # Well-conditioned
            Sigma[i] = torch.eye(D) * (1.0 + 0.1 * i)
        else:
            # Poorly conditioned
            eigvals = torch.tensor([1e-5 + 0.01 * (i - B//2), 10.0])
            Q, _ = torch.linalg.qr(torch.randn(D, D))
            Sigma[i] = Q @ torch.diag(eigvals) @ Q.T
    
    mu = torch.randn(B, D)
    z = mu + 0.1 * torch.randn(B, D)
    
    log_q = log_q_riem(z, mu, Sigma, min_eig=1e-3)
    print(f"\nResult: log_q mean = {log_q.mean().item():.6f}, std = {log_q.std().item():.6f}")
    print("Expected: Stabilization details for the batch\n")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING LOG_Q_RIEM STABILIZATION DIAGNOSTICS")
    print("="*80)
    
    test_log_q_riem_no_stabilization()
    test_log_q_riem_with_stabilization()
    test_log_q_riem_large_batch()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80 + "\n")

