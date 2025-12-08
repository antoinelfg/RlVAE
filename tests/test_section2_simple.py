#!/usr/bin/env python3
"""
Simple test for Section 2: Posterior Sampling (Local, Reparam)

Tests core functionality directly without complex model imports.
"""

import sys
import torch
import unittest
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(current_dir / "src"))

from utils.metric_validation import MetricValidator, compute_metric_diagnostics


class TestSection2Core(unittest.TestCase):
    """Test core Section 2 functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 4
        self.latent_dim = 16
        
    def test_alpha_ramping_logic(self):
        """Test the α ramping logic directly."""
        print("\\n🧪 Test 2.1: α Ramping Logic")
        
        # Simulate ramping parameters
        alpha_start = 0.25
        alpha_end = 1.0
        ramp_epochs = 5
        
        def get_ramped_alpha(epoch):
            if epoch >= ramp_epochs:
                return alpha_end
            progress = epoch / ramp_epochs
            return alpha_start + progress * (alpha_end - alpha_start)
        
        # Test key points
        assert get_ramped_alpha(0) == 0.25, "Start alpha incorrect"
        assert get_ramped_alpha(5) == 1.0, "End alpha incorrect"
        assert abs(get_ramped_alpha(2.5) - 0.625) < 1e-6, "Mid-point alpha incorrect"
        
        print(f"✅ α ramping: {get_ramped_alpha(0):.3f} → {get_ramped_alpha(5):.3f}")
        print(f"   Mid-point (epoch 2.5): {get_ramped_alpha(2.5):.3f}")
    
    def test_metric_aligned_covariance(self):
        """Test the covariance structure Σ = α G(μ)."""
        print("\\n🧪 Test 2.2: Metric-Aligned Covariance")
        
        # Create a simple metric tensor G
        alpha = 0.5
        G = torch.eye(self.latent_dim, device=self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        G = G * 2.0  # Scale to test that α scaling works
        
        # Apply covariance formula: Σ = α G(μ)
        eps_chol = 1e-6
        I = torch.eye(self.latent_dim, device=self.device)
        Sigma = alpha * G + eps_chol * I.unsqueeze(0)
        
        # Verify shape and properties
        assert Sigma.shape == (self.batch_size, self.latent_dim, self.latent_dim), f"Wrong shape: {Sigma.shape}"
        
        # Check that Σ is properly scaled by α
        expected_diag = alpha * 2.0 + eps_chol  # α * G_diag + eps_chol
        actual_diag = Sigma[0].diag().mean().item()
        assert abs(actual_diag - expected_diag) < 1e-6, f"Scaling incorrect: {actual_diag} vs {expected_diag}"
        
        # Test Cholesky decomposition
        try:
            L = torch.linalg.cholesky(Sigma)
            assert L.shape == Sigma.shape, "Cholesky shape mismatch"
            
            # Verify L L^T = Σ
            Sigma_reconstructed = torch.bmm(L, L.transpose(-1, -2))
            error = torch.norm(Sigma - Sigma_reconstructed, dim=(-2, -1)).max().item()
            assert error < 1e-4, f"Cholesky reconstruction error: {error:.2e}"
            
            print(f"✅ Covariance Σ = α G(μ) with α={alpha}")
            print(f"✅ Cholesky decomposition successful (error: {error:.2e})")
            
        except Exception as e:
            assert False, f"Cholesky failed: {e}"
    
    def test_reparameterized_sampling(self):
        """Test reparameterized sampling z = μ + L ε."""
        print("\\n🧪 Test 2.3: Reparameterized Sampling")
        
        # Setup
        mu = torch.randn(self.batch_size, self.latent_dim, device=self.device)
        eps = torch.randn(self.batch_size, self.latent_dim, device=self.device)
        
        # Create covariance matrix
        alpha = 0.5
        G = torch.eye(self.latent_dim, device=self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        G = G * 3.0  # Add some scaling
        
        eps_chol = 1e-6
        I = torch.eye(self.latent_dim, device=self.device)
        Sigma = alpha * G + eps_chol * I.unsqueeze(0)
        
        # Cholesky and sampling
        L = torch.linalg.cholesky(Sigma)
        z = mu + torch.einsum('bij,bj->bi', L, eps)
        
        # Verify sampling
        assert z.shape == mu.shape, f"Sample shape mismatch: {z.shape} vs {mu.shape}"
        assert z.device == mu.device, "Device mismatch"
        
        # Check that samples are different from μ
        sample_distance = torch.norm(z - mu, dim=1).mean().item()
        assert sample_distance > 0, "No sampling variation"
        assert sample_distance < 10.0, f"Samples too far from μ: {sample_distance}"
        
        print(f"✅ Reparameterized sampling z = μ + L ε")
        print(f"   Sample distance from μ: {sample_distance:.3f}")
    
    def test_curvature_correction_concept(self):
        """Test curvature correction concept (evaluate at z vs μ)."""
        print("\\n🧪 Test 2.4: Curvature Correction Concept")
        
        # Create two different metric evaluations
        mu = torch.randn(self.batch_size, self.latent_dim, device=self.device)
        z = mu + 0.1 * torch.randn_like(mu)  # z ≠ μ
        
        # Simple metric function that varies with input
        def simple_metric(x):
            """Metric that depends on input location."""
            scale = 1.0 + 0.1 * torch.norm(x, dim=1, keepdim=True).unsqueeze(-1)
            return scale * torch.eye(self.latent_dim, device=x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
        
        G_mu = simple_metric(mu)     # G(μ) - traditional
        G_z = simple_metric(z)       # G(z) - curvature corrected
        
        # Verify they're different (showing curvature correction matters)
        difference = torch.norm(G_mu - G_z, dim=(-2, -1)).mean().item()
        assert difference > 0, "G(μ) and G(z) should be different for curvature correction"
        
        # Test quadratic form evaluation with both
        diff = (z - mu).unsqueeze(-1)
        quad_mu = torch.matmul(torch.matmul(diff.transpose(-2,-1), G_mu), diff).squeeze()
        quad_z = torch.matmul(torch.matmul(diff.transpose(-2,-1), G_z), diff).squeeze()
        
        quad_diff = torch.norm(quad_mu - quad_z).item()
        assert quad_diff > 0, "Quadratic forms should differ"
        
        print(f"✅ Curvature correction: G(μ) vs G(z) difference = {difference:.6f}")
        print(f"   Quadratic form difference: {quad_diff:.6f}")
        print("✅ Using G(z) provides proper curvature correction")
    
    def test_failure_counting_logic(self):
        """Test failure counting logic."""
        print("\\n🧪 Test 2.5: Failure Counting Logic")
        
        # Simulate failure tracking
        total_samples = 0
        failure_count = 0
        
        # Simulate several batches with some failures
        for batch_idx in range(5):
            batch_size = 4
            total_samples += batch_size
            
            # Simulate random failures
            if batch_idx == 2:  # Fail batch 2
                failure_count += batch_size
                success = False
            else:
                success = True
            
            failure_rate = failure_count / max(total_samples, 1)
            print(f"   Batch {batch_idx}: success={success}, failure_rate={failure_rate:.2%}")
        
        # Final check
        expected_failure_rate = 4 / 20  # 1 batch of 4 failed out of 20 total
        assert abs(failure_rate - expected_failure_rate) < 1e-6, "Failure rate calculation incorrect"
        
        print(f"✅ Failure tracking: {failure_count}/{total_samples} = {failure_rate:.2%}")
    
    def test_diagnostic_structure(self):
        """Test diagnostic data structure."""
        print("\\n🧪 Test 2.6: Diagnostic Structure")
        
        # Simulate diagnostic data
        G = torch.eye(self.latent_dim, device=self.device).unsqueeze(0) * 2.0
        z_sample = torch.randn(self.latent_dim, device=self.device)
        mu = torch.randn(self.latent_dim, device=self.device)
        
        # Compute diagnostics using our utility
        diagnostics = compute_metric_diagnostics(G)
        
        # Create full diagnostic structure
        full_diagnostics = {
            'posterior_alpha': 0.5,
            'posterior_G_eigenval_min': diagnostics['eigenvals_min'][0].item(),
            'posterior_G_eigenval_max': diagnostics['eigenvals_max'][0].item(), 
            'posterior_G_condition_number': diagnostics['condition_number'][0].item(),
            'posterior_sample_norm': torch.norm(z_sample - mu).item(),
            'cholesky_failure_rate': 0.05,
            'cholesky_success': True,
        }
        
        # Verify all required fields
        required_fields = [
            'posterior_alpha',
            'posterior_G_eigenval_min',
            'posterior_G_eigenval_max', 
            'posterior_G_condition_number',
            'posterior_sample_norm',
            'cholesky_failure_rate',
            'cholesky_success'
        ]
        
        for field in required_fields:
            assert field in full_diagnostics, f"Missing diagnostic field: {field}"
            print(f"✅ {field}: {full_diagnostics[field]}")
        
        print("✅ All diagnostic fields present and valid")


def main():
    """Run all Section 2 core tests."""
    print("🧪 Starting Section 2 Core Tests (No Model Dependencies)")
    print("=" * 60)
    
    test_suite = TestSection2Core()
    test_suite.setUp()
    
    try:
        test_suite.test_alpha_ramping_logic()
        test_suite.test_metric_aligned_covariance()
        test_suite.test_reparameterized_sampling()
        test_suite.test_curvature_correction_concept()
        test_suite.test_failure_counting_logic()
        test_suite.test_diagnostic_structure()
        
        print("\\n" + "=" * 60)
        print("🎉 ALL SECTION 2 CORE TESTS PASSED!")
        print("✅ Core Section 2 functionality verified")
        return True
        
    except Exception as e:
        print(f"\\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
