#!/usr/bin/env python3
"""
Diagnose RHMC Posterior Divergence During Training
==================================================

This script loads a trained Stage B metric and tests the RHMC posterior
to identify why samples are diverging during Stage C training.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rlvae.models.components.riemannian_rhmc_posterior import RiemannianRHMCPosterior


def load_stage_b_metric(checkpoint_path: str):
    """Load metric from Stage B checkpoint."""
    print(f"📂 Loading Stage B metric from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract metric tensors
    C = checkpoint['model_state_dict']['C']
    M = checkpoint['model_state_dict']['M']
    
    print(f"   C shape: {C.shape}, M shape: {M.shape}")
    return C, M


class SimpleMetricModel(torch.nn.Module):
    """Wrapper for Stage B metric."""
    
    def __init__(self, C, M):
        super().__init__()
        self.C = torch.nn.Parameter(C, requires_grad=False)
        self.M = torch.nn.Parameter(M, requires_grad=False)
        self.latent_dim = C.shape[0]
        self.device = C.device
        
    def G(self, z):
        """Compute metric tensor G(z) = C + M·h(z)."""
        batch_size = z.shape[0]
        
        # Compute h(z) - temperature-scaled distances to centroids
        # For simplicity, use identity for now
        G = self.C.unsqueeze(0).expand(batch_size, -1, -1)
        return G
    
    def G_inv(self, z):
        """Compute inverse metric."""
        G = self.G(z)
        return torch.linalg.inv(G)


def test_rhmc_posterior_evolution(model, config, n_iterations=10):
    """Test how RHMC samples evolve over multiple sampling iterations."""
    print("\n" + "=" * 80)
    print("Test: RHMC Sample Evolution Over Iterations")
    print("=" * 80)
    
    posterior = RiemannianRHMCPosterior(model, config)
    
    # Initial encoder outputs (simulating training)
    n_samples = 100
    mu = torch.randn(n_samples, model.latent_dim) * 0.5
    log_var = torch.zeros(n_samples, model.latent_dim)
    
    # Track statistics over iterations
    mean_norms = []
    max_norms = []
    mean_dists_from_mu = []
    
    print(f"\n📊 Running {n_iterations} sampling iterations...")
    for i in range(n_iterations):
        z = posterior.sample_riemannian_rhmc_posterior(mu, log_var)
        
        z_norms = torch.norm(z, dim=-1)
        dists_from_mu = torch.norm(z - mu, dim=-1)
        
        mean_norms.append(z_norms.mean().item())
        max_norms.append(z_norms.max().item())
        mean_dists_from_mu.append(dists_from_mu.mean().item())
        
        if i % 2 == 0:
            print(f"   Iter {i}: mean ||z|| = {mean_norms[-1]:.3f}, "
                  f"max ||z|| = {max_norms[-1]:.3f}, "
                  f"mean ||z-μ|| = {mean_dists_from_mu[-1]:.3f}")
    
    # Check for divergence
    early_avg = np.mean(mean_norms[:3])
    late_avg = np.mean(mean_norms[-3:])
    divergence = (late_avg - early_avg) / early_avg
    
    print(f"\n📈 Divergence Analysis:")
    print(f"   Early average ||z||: {early_avg:.3f}")
    print(f"   Late average ||z||: {late_avg:.3f}")
    print(f"   Divergence: {divergence:.1%}")
    
    if abs(divergence) > 0.2:
        print(f"   ⚠️ SIGNIFICANT DIVERGENCE DETECTED!")
        return False, mean_norms, max_norms, mean_dists_from_mu
    else:
        print(f"   ✅ Samples remain stable")
        return True, mean_norms, max_norms, mean_dists_from_mu


def test_parameter_sensitivity(model):
    """Test how different RHMC parameters affect divergence."""
    print("\n" + "=" * 80)
    print("Test: Parameter Sensitivity Analysis")
    print("=" * 80)
    
    # Test configurations
    configs = {
        "baseline": {
            'rhmc_steps': 1,
            'rhmc_step_size': 0.01,
            'rhmc_alpha': 1.0,
            'eps_regularization': 1e-6
        },
        "with_safety": {
            'rhmc_steps': 1,
            'rhmc_step_size': 0.01,
            'rhmc_alpha': 1.0,
            'eps_regularization': 1e-6,
            'max_momentum_norm': 5.0,
            'max_velocity_norm': 2.0,
            'max_position_step': 1.0,
            'max_position_norm': 12.0
        },
        "conservative": {
            'rhmc_steps': 1,
            'rhmc_step_size': 0.005,
            'rhmc_alpha': 0.5,
            'eps_regularization': 1e-4,
            'max_momentum_norm': 3.0,
            'max_velocity_norm': 1.0,
            'max_position_step': 0.5,
            'max_position_norm': 8.0
        }
    }
    
    results = {}
    for name, config in configs.items():
        print(f"\n🔧 Testing configuration: {name}")
        stable, *stats = test_rhmc_posterior_evolution(model, config, n_iterations=5)
        results[name] = {'stable': stable, 'stats': stats}
    
    # Summary
    print("\n" + "=" * 80)
    print("PARAMETER SENSITIVITY SUMMARY")
    print("=" * 80)
    for name, result in results.items():
        status = "✅ STABLE" if result['stable'] else "⚠️ DIVERGING"
        print(f"{name:15s}: {status}")
    
    return results


def visualize_sample_distribution(model, config):
    """Visualize where RHMC samples end up relative to encoder means."""
    print("\n" + "=" * 80)
    print("Visualization: RHMC Sample Distribution")
    print("=" * 80)
    
    posterior = RiemannianRHMCPosterior(model, config)
    
    # Sample
    n_samples = 200
    mu = torch.randn(n_samples, 2) * 0.5
    log_var = torch.zeros(n_samples, 2)
    z = posterior.sample_riemannian_rhmc_posterior(mu, log_var)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(mu[:, 0].numpy(), mu[:, 1].numpy(), 
               c='green', s=50, marker='x', alpha=0.7, label='Encoder means μ')
    ax.scatter(z[:, 0].numpy(), z[:, 1].numpy(), 
               c='blue', s=20, alpha=0.5, label='RHMC samples z')
    
    # Draw lines from mu to z
    for i in range(min(50, n_samples)):
        ax.plot([mu[i, 0].item(), z[i, 0].item()],
                [mu[i, 1].item(), z[i, 1].item()],
                'gray', alpha=0.2, linewidth=0.5)
    
    ax.set_xlabel('z₁')
    ax.set_ylabel('z₂')
    ax.set_title('RHMC Posterior: Samples vs Encoder Means')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Save
    output_path = project_root / 'wandb' / 'plots' / 'rhmc_divergence_diagnosis.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved visualization to: {output_path}")
    plt.close()


def main():
    print("\n" + "🔍 " * 40)
    print("DIAGNOSING RHMC POSTERIOR DIVERGENCE")
    print("🔍 " * 40)
    
    # For now, use a simple metric
    # In practice, you'd load from Stage B checkpoint
    C = torch.eye(2) * 0.1
    M = torch.zeros(2, 2, 300)  # Dummy
    
    model = SimpleMetricModel(C, M)
    
    # Test 1: Evolution over iterations
    baseline_config = {
        'rhmc_steps': 1,
        'rhmc_step_size': 0.01,
        'rhmc_alpha': 1.0,
        'eps_regularization': 1e-6
    }
    
    stable, *_ = test_rhmc_posterior_evolution(model, baseline_config, n_iterations=10)
    
    # Test 2: Parameter sensitivity
    results = test_parameter_sensitivity(model)
    
    # Test 3: Visualization
    visualize_sample_distribution(model, baseline_config)
    
    # Final summary
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
    
    if not stable:
        print("\n⚠️ RHMC POSTERIOR IS DIVERGING!")
        print("\n📋 Recommended fixes:")
        print("   1. Add safety parameters to config:")
        print("      max_momentum_norm: 3.0")
        print("      max_velocity_norm: 1.0")
        print("      max_position_step: 0.5")
        print("      max_position_norm: 8.0")
        print("   2. Reduce step size: 0.01 → 0.005")
        print("   3. Reduce alpha: 1.0 → 0.5")
        print("   4. Increase regularization: 1e-6 → 1e-4")
    else:
        print("\n✅ RHMC POSTERIOR IS STABLE")
        print("   Divergence may be due to other factors (KL, metric quality, etc.)")
    
    return stable


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

