"""
Identity Metric Mode for RlVAE - Sanity Check Utility

This module provides functionality to run the entire RlVAE model with identity
metric tensors G(z) = I for all z. This is useful for:
- Sanity checks and debugging
- Baseline comparisons  
- Verifying that the model reduces to standard VAE when G = I
- Testing metric-dependent code paths
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable, Tuple


class IdentityMetricWrapper:
    """
    Wrapper that provides identity metric functions G(z) = I and G_inv(z) = I.
    """
    
    def __init__(self, latent_dim: int, device: Optional[torch.device] = None):
        self.latent_dim = latent_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pre-compute identity matrix for efficiency
        self.identity = torch.eye(latent_dim, device=self.device, dtype=torch.float32)
        
        print(f"🔧 IdentityMetricWrapper initialized for latent_dim={latent_dim}")
        print(f"   Device: {self.device}")
        print(f"   Identity matrix shape: {self.identity.shape}")
    
    def G(self, z: torch.Tensor) -> torch.Tensor:
        """
        Identity metric tensor: G(z) = I for all z.
        
        Args:
            z: Latent coordinates [batch_size, latent_dim]
            
        Returns:
            Identity matrices [batch_size, latent_dim, latent_dim]
        """
        batch_size = z.shape[0]
        return self.identity.unsqueeze(0).expand(batch_size, -1, -1)
    
    def G_inv(self, z: torch.Tensor) -> torch.Tensor:
        """
        Identity inverse metric tensor: G^{-1}(z) = I for all z.
        
        Args:
            z: Latent coordinates [batch_size, latent_dim]
            
        Returns:
            Identity matrices [batch_size, latent_dim, latent_dim]
        """
        batch_size = z.shape[0]
        return self.identity.unsqueeze(0).expand(batch_size, -1, -1)
    
    def log_det_G(self, z: torch.Tensor) -> torch.Tensor:
        """
        Log determinant of identity matrix: log|I| = 0.
        
        Args:
            z: Latent coordinates [batch_size, latent_dim]
            
        Returns:
            Zeros [batch_size]
        """
        batch_size = z.shape[0]
        return torch.zeros(batch_size, device=self.device)
    
    def get_diagnostics(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get metric diagnostics for identity case.
        
        Args:
            z: Latent coordinates [batch_size, latent_dim]
            
        Returns:
            Dict with diagnostic information
        """
        batch_size = z.shape[0]
        
        return {
            'eigenvals_min': torch.ones(batch_size, device=self.device),
            'eigenvals_max': torch.ones(batch_size, device=self.device),
            'condition_number': torch.ones(batch_size, device=self.device),
            'trace': torch.full((batch_size,), self.latent_dim, device=self.device, dtype=torch.float32),
            'det': torch.ones(batch_size, device=self.device),
            'log_det': torch.zeros(batch_size, device=self.device),
            'frobenius_norm': torch.full((batch_size,), torch.sqrt(torch.tensor(self.latent_dim)), device=self.device)
        }


def verify_identity_metric_equivalence(
    model: nn.Module,
    test_data: torch.Tensor,
    tolerance: float = 1e-5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Verify that model with identity metric behaves like standard VAE.
    
    Args:
        model: RlVAE model with identity metric enabled
        test_data: Test input data [batch_size, *input_shape]
        tolerance: Numerical tolerance for comparisons
        verbose: Whether to print detailed results
        
    Returns:
        Dict with verification results
    """
    model.eval()
    
    with torch.no_grad():
        # Forward pass - handle variable return values
        output = model(test_data)
        if hasattr(output, 'recon_x'):
            recon = output.recon_x
            mu = output.mu if hasattr(output, 'mu') else None
            log_var = output.log_var if hasattr(output, 'log_var') else None
            z_sample = output.z if hasattr(output, 'z') else None
        else:
            # Handle tuple return
            recon = output[0] if len(output) > 0 else None
            mu = output[1] if len(output) > 1 else None
            log_var = output[2] if len(output) > 2 else None
            z_sample = output[3] if len(output) > 3 else None
        
        # Get metric values - use test data if z_sample is None
        if z_sample is not None and len(z_sample.shape) == 2:
            batch_size, latent_dim = z_sample.shape
            metric_z = z_sample
        else:
            batch_size, latent_dim = test_data.shape[0], 16  # Default values
            metric_z = torch.randn(batch_size, latent_dim, device=test_data.device)
            
        G_z = model.G(metric_z)
        G_inv_z = model.G_inv(metric_z)
        
        identity_expected = torch.eye(latent_dim, device=metric_z.device)
        identity_batch = identity_expected.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Check G(z) = I
        G_error = torch.norm(G_z - identity_batch, dim=(-2, -1)).max().item()
        G_is_identity = G_error < tolerance
        
        # Check G_inv(z) = I  
        G_inv_error = torch.norm(G_inv_z - identity_batch, dim=(-2, -1)).max().item()
        G_inv_is_identity = G_inv_error < tolerance
        
        # Check G @ G_inv = I
        product = torch.bmm(G_z, G_inv_z)
        product_error = torch.norm(product - identity_batch, dim=(-2, -1)).max().item()
        inverse_consistent = product_error < tolerance
        
        # Expected KL for identity metric (should match standard VAE)
        # KL = 0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        expected_kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        
        results = {
            'G_is_identity': G_is_identity,
            'G_inv_is_identity': G_inv_is_identity,
            'inverse_consistent': inverse_consistent,
            'G_error': G_error,
            'G_inv_error': G_inv_error,
            'product_error': product_error,
            'expected_kl': expected_kl.item(),
            'tolerance': tolerance,
            'all_checks_passed': G_is_identity and G_inv_is_identity and inverse_consistent
        }
        
        if verbose:
            print(f"\n🔍 Identity Metric Verification:")
            print(f"   G(z) = I: {'✅' if G_is_identity else '❌'} (error: {G_error:.2e})")
            print(f"   G_inv(z) = I: {'✅' if G_inv_is_identity else '❌'} (error: {G_inv_error:.2e})")
            print(f"   G @ G_inv = I: {'✅' if inverse_consistent else '❌'} (error: {product_error:.2e})")
            print(f"   Expected KL: {expected_kl.item():.4f}")
            print(f"   Tolerance: {tolerance:.2e}")
            print(f"   Overall: {'✅ PASSED' if results['all_checks_passed'] else '❌ FAILED'}")
        
        return results


def create_identity_metric_test_suite(
    model_factory: Callable,
    config: Dict[str, Any],
    test_batch_size: int = 16,
    latent_dims: list = [8, 16, 32],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Create comprehensive test suite for identity metric mode.
    
    Args:
        model_factory: Function to create model instances
        config: Base configuration for models
        test_batch_size: Batch size for testing
        latent_dims: List of latent dimensions to test
        verbose: Whether to print detailed results
        
    Returns:
        Dict with test results for each latent dimension
    """
    results = {}
    
    for latent_dim in latent_dims:
        if verbose:
            print(f"\n🧪 Testing identity metric with latent_dim={latent_dim}")
        
        # Create model with identity metric
        test_config = config.copy()
        test_config['latent_dim'] = latent_dim
        test_config['identity_metric_mode'] = True
        
        model = model_factory(test_config)
        
        # Generate test data
        input_shape = test_config.get('input_dim', [3, 64, 64])
        test_data = torch.randn(test_batch_size, *input_shape)
        
        # Run verification
        verification_results = verify_identity_metric_equivalence(
            model, test_data, verbose=verbose
        )
        
        results[f'latent_dim_{latent_dim}'] = verification_results
    
    # Summary
    all_passed = all(result['all_checks_passed'] for result in results.values())
    
    if verbose:
        print(f"\n📊 Identity Metric Test Suite Summary:")
        print(f"   Tested latent dimensions: {latent_dims}")
        print(f"   Overall result: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
        
        for latent_dim in latent_dims:
            key = f'latent_dim_{latent_dim}'
            status = '✅' if results[key]['all_checks_passed'] else '❌'
            print(f"   - {latent_dim}D: {status}")
    
    results['summary'] = {
        'all_passed': all_passed,
        'tested_dimensions': latent_dims,
        'total_tests': len(latent_dims)
    }
    
    return results


class IdentityMetricConfig:
    """
    Configuration helper for identity metric mode.
    """
    
    @staticmethod
    def enable_identity_mode(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enable identity metric mode in configuration.
        
        Args:
            config: Original configuration
            
        Returns:
            Modified configuration with identity metric enabled
        """
        config = config.copy()
        config['identity_metric_mode'] = True
        config['metric_validation_enabled'] = True
        config['eps_chol'] = 1e-12  # Very small since identity is well-conditioned
        
        # Disable metric learning components that don't make sense with identity
        config['update_metric_during_training'] = False
        config['metric_update_frequency'] = 0
        
        return config
    
    @staticmethod
    def create_baseline_comparison_config(
        base_config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Create paired configs for identity vs normal metric comparison.
        
        Args:
            base_config: Base configuration
            
        Returns:
            Tuple of (identity_config, normal_config)
        """
        identity_config = IdentityMetricConfig.enable_identity_mode(base_config)
        normal_config = base_config.copy()
        normal_config['identity_metric_mode'] = False
        
        return identity_config, normal_config
