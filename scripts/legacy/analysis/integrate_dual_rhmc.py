#!/usr/bin/env python3
"""
Dual RHMC Integration
=====================

Simple integration script to use dual RHMC (G⁻¹ as metric) in the main pipeline.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.models.riemannian_flow_vae import RiemannianFlowVAE
from dual_rhmc_implementation import DualRiemannianHMCSampler


def create_dual_rhmc_sampler(model, **kwargs):
    """
    Create a dual RHMC sampler with G⁻¹ as metric.
    
    Args:
        model: RiemannianFlowVAE model
        **kwargs: Sampler parameters (mcmc_steps_nbr, n_lf, eps_lf)
    
    Returns:
        DualRiemannianHMCSampler instance
    """
    default_params = {
        'mcmc_steps_nbr': 30,
        'n_lf': 10,
        'eps_lf': 0.02
    }
    default_params.update(kwargs)
    
    return DualRiemannianHMCSampler(model, **default_params)


def sample_with_dual_rhmc(model, n_samples=100, **sampler_kwargs):
    """
    Sample from the manifold using dual RHMC.
    
    Args:
        model: RiemannianFlowVAE model
        n_samples: Number of samples to generate
        **sampler_kwargs: Sampler parameters
    
    Returns:
        samples: Tensor of shape (n_samples, latent_dim)
    """
    sampler = create_dual_rhmc_sampler(model, **sampler_kwargs)
    return sampler.sample(n_samples=n_samples)


def sample_posterior_with_dual_rhmc(model, mu, log_var, **sampler_kwargs):
    """
    Sample from posterior using dual RHMC.
    
    Args:
        model: RiemannianFlowVAE model
        mu: Posterior mean
        log_var: Posterior log variance
        **sampler_kwargs: Sampler parameters
    
    Returns:
        samples: Tensor of shape (batch_size, latent_dim)
    """
    sampler = create_dual_rhmc_sampler(model, **sampler_kwargs)
    return sampler.sample_posterior(mu, log_var)


def compare_standard_vs_dual_rhmc(model, n_samples=100):
    """
    Compare standard RHMC vs dual RHMC performance.
    
    Args:
        model: RiemannianFlowVAE model
        n_samples: Number of samples to generate
    
    Returns:
        dict: Comparison results
    """
    print("🔍 Comparing Standard vs Dual RHMC")
    print("=" * 50)
    
    # Test standard RHMC (if available)
    try:
        from src.lib.src.pythae.samplers.manifold_sampler import RHMCSampler
        standard_sampler = RHMCSampler(model)
        
        print("📊 Standard RHMC (G as metric):")
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        standard_samples = standard_sampler.sample(n_samples=n_samples)
        end_time.record()
        torch.cuda.synchronize()
        
        standard_time = start_time.elapsed_time(end_time) / 1000.0
        print(f"   ✅ Time: {standard_time:.3f}s")
        print(f"   ✅ Samples range: [{standard_samples.min():.3f}, {standard_samples.max():.3f}]")
        
    except Exception as e:
        print(f"   ❌ Standard RHMC not available: {e}")
        standard_samples = None
        standard_time = None
    
    # Test dual RHMC
    print("\n📊 Dual RHMC (G⁻¹ as metric):")
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    dual_sampler = create_dual_rhmc_sampler(model)
    dual_samples = dual_sampler.sample(n_samples=n_samples)
    end_time.record()
    torch.cuda.synchronize()
    
    dual_time = start_time.elapsed_time(end_time) / 1000.0
    print(f"   ✅ Time: {dual_time:.3f}s")
    print(f"   ✅ Samples range: [{dual_samples.min():.3f}, {dual_samples.max():.3f}]")
    
    # Analyze metric properties
    with torch.no_grad():
        G_dual = model.G(dual_samples)
        dual_eigenvals = torch.linalg.eigvals(G_dual).real
        dual_determinants = torch.linalg.det(G_dual)
        
        print(f"   ✅ G eigenvalues: min={dual_eigenvals.min():.3e}, max={dual_eigenvals.max():.3e}")
        print(f"   ✅ G determinants: min={dual_determinants.min():.3e}, max={dual_determinants.max():.3e}")
    
    # Comparison summary
    results = {
        'dual_samples': dual_samples,
        'dual_time': dual_time,
        'dual_eigenvals': dual_eigenvals,
        'dual_determinants': dual_determinants
    }
    
    if standard_samples is not None:
        results['standard_samples'] = standard_samples
        results['standard_time'] = standard_time
        
        with torch.no_grad():
            G_standard = model.G(standard_samples)
            standard_eigenvals = torch.linalg.eigvals(G_standard).real
            standard_determinants = torch.linalg.det(G_standard)
            
            print(f"\n📊 Standard RHMC G properties:")
            print(f"   ✅ G eigenvalues: min={standard_eigenvals.min():.3e}, max={standard_eigenvals.max():.3e}")
            print(f"   ✅ G determinants: min={standard_determinants.min():.3e}, max={standard_determinants.max():.3e}")
        
        results['standard_eigenvals'] = standard_eigenvals
        results['standard_determinants'] = standard_determinants
    
    print(f"\n✅ Comparison completed!")
    return results


def quick_test():
    """Quick test of dual RHMC integration."""
    print("🚀 Quick Test of Dual RHMC Integration")
    print("=" * 50)
    
    # Create model
    model = RiemannianFlowVAE(
        input_dim=(3, 64, 64),
        latent_dim=2,
        n_flows=0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load pretrained components
    model.load_pretrained_components(
        encoder_path="data/pretrained/encoder_diverse_mlp_ld2_20250730_105419.pt",
        decoder_path="data/pretrained/decoder_diverse_mlp_ld2_20250730_105419.pt",
        metric_path="data/pretrained/metric_diverse_mlp_ld2_20250730_105420.pt"
    )
    
    # Test dual RHMC sampling
    print("\n🎯 Testing Dual RHMC Sampling")
    samples = sample_with_dual_rhmc(model, n_samples=50, mcmc_steps_nbr=20)
    print(f"✅ Generated {len(samples)} samples")
    print(f"✅ Samples shape: {samples.shape}")
    print(f"✅ Samples range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    # Test posterior sampling
    print("\n🎯 Testing Dual RHMC Posterior Sampling")
    mu = torch.tensor([[0.0, 0.0]], device=device)
    log_var = torch.tensor([[0.1, 0.1]], device=device)
    
    posterior_samples = sample_posterior_with_dual_rhmc(model, mu, log_var)
    print(f"✅ Posterior samples: {posterior_samples}")
    print(f"✅ Distance to mean: {torch.norm(posterior_samples - mu, dim=1)}")
    
    print(f"\n✅ Quick test completed!")


if __name__ == "__main__":
    quick_test() 