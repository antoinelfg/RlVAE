#!/usr/bin/env python3
"""
Test RHMC Posterior Diagnostics
===============================

Isolated testing of RHMC posterior components to diagnose manifold divergence issues.
Tests both initial Riemannian sampling and RHMC exploration phases.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rlvae.models.components.riemannian_rhmc_posterior import RiemannianRHMCPosterior
from src.rlvae.models.components.metric_tensor import MetricTensor
from sklearn.decomposition import PCA


class MockModel(nn.Module):
    """Mock model for isolated testing of RHMC posterior."""
    
    def __init__(self, latent_dim=2, n_centroids=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create simple test centroids in a circle pattern
        angles = torch.linspace(0, 2*np.pi, n_centroids+1)[:-1]
        self.centroids_tens = torch.stack([
            2.0 * torch.cos(angles),
            2.0 * torch.sin(angles)
        ], dim=1).to(self.device)
        
        # Create simple metric matrices (more anisotropic near centroids)
        self.metric_matrices = torch.eye(latent_dim).unsqueeze(0).repeat(n_centroids, 1, 1).to(self.device)
        for i in range(n_centroids):
            # Make metrics more anisotropic
            self.metric_matrices[i, 0, 0] = 2.0 + i * 0.5
            self.metric_matrices[i, 1, 1] = 1.0 + i * 0.2
        
        self.temperature = 0.7
        self.regularization = 0.01
        
    def G(self, z):
        """Compute metric tensor G(z) using centroid-based interpolation."""
        batch_size = z.shape[0]
        
        # Compute distances to centroids
        diff = z.unsqueeze(1) - self.centroids_tens.unsqueeze(0)  # [B, K, D]
        distances = torch.norm(diff, dim=-1)  # [B, K]
        
        # Softmax weights
        weights = torch.softmax(-distances / self.temperature, dim=-1)  # [B, K]
        
        # Weighted combination of metric matrices
        G = torch.einsum('bk,kij->bij', weights, self.metric_matrices)  # [B, D, D]
        
        # Add regularization
        I = torch.eye(self.latent_dim, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
        G = G + self.regularization * I
        
        return G
    
    def G_inv(self, z):
        """Compute inverse metric tensor G^-1(z)."""
        G = self.G(z)
        return torch.linalg.inv(G)


class RHMCPosteriorDiagnostics:
    """Comprehensive diagnostics for RHMC posterior behavior."""
    
    def __init__(self, latent_dim=2, n_centroids=5):
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create mock model and RHMC posterior
        self.mock_model = MockModel(latent_dim, n_centroids).to(self.device)
        
        # RHMC configuration for testing
        self.rhmc_config = {
            'rhmc_steps': 3,
            'rhmc_step_size': 0.001,
            'rhmc_alpha': 0.1,
            'eps_regularization': 1e-3,
            'max_grad_norm': 3.0,
            'adaptive_step_size': True,
            'volume_force_sign': 1.0,
            'projection_step_scale': 0.0,
            'volume_grad_scale': 1.0,
            'volume_bias_weight': 1.0,
        }
        self.grad_enabled = os.getenv("DIAG_ENABLE_GRADS", "0") == "1"

        alpha_override = os.getenv("DIAG_ALPHA")
        if alpha_override is not None:
            try:
                self.rhmc_config['rhmc_alpha'] = float(alpha_override)
            except ValueError:
                print(f"⚠️ Invalid DIAG_ALPHA={alpha_override}; keeping default {self.rhmc_config['rhmc_alpha']}")
        eps_override = os.getenv("DIAG_EPS_REG")
        if eps_override is not None:
            try:
                self.rhmc_config['eps_regularization'] = float(eps_override)
            except ValueError:
                print(f"⚠️ Invalid DIAG_EPS_REG={eps_override}; keeping default {self.rhmc_config['eps_regularization']}")
        volume_override = os.getenv("DIAG_VOLUME_SIGN")
        if volume_override is not None:
            try:
                self.rhmc_config['volume_force_sign'] = float(volume_override)
            except ValueError:
                print(f"⚠️ Invalid DIAG_VOLUME_SIGN={volume_override}; keeping default {self.rhmc_config['volume_force_sign']}")
        proj_override = os.getenv("DIAG_PROJECTION_SCALE")
        if proj_override is not None:
            try:
                self.rhmc_config['projection_step_scale'] = float(proj_override)
            except ValueError:
                print(f"⚠️ Invalid DIAG_PROJECTION_SCALE={proj_override}; keeping default {self.rhmc_config['projection_step_scale']}")
        grad_scale_override = os.getenv("DIAG_VOLUME_GRAD_SCALE")
        if grad_scale_override is not None:
            try:
                self.rhmc_config['volume_grad_scale'] = float(grad_scale_override)
            except ValueError:
                print(f"⚠️ Invalid DIAG_VOLUME_GRAD_SCALE={grad_scale_override}; keeping default {self.rhmc_config['volume_grad_scale']}")
        bias_override = os.getenv("DIAG_VOLUME_BIAS_WEIGHT")
        if bias_override is not None:
            try:
                self.rhmc_config['volume_bias_weight'] = float(bias_override)
            except ValueError:
                print(f"⚠️ Invalid DIAG_VOLUME_BIAS_WEIGHT={bias_override}; keeping default {self.rhmc_config['volume_bias_weight']}")

        print(
            "[DIAG CONFIG] "
            f"alpha={self.rhmc_config['rhmc_alpha']}, "
            f"eps_reg={self.rhmc_config['eps_regularization']}, "
            f"volume_force_sign={self.rhmc_config['volume_force_sign']}, "
            f"projection_step_scale={self.rhmc_config['projection_step_scale']}, "
            f"volume_grad_scale={self.rhmc_config['volume_grad_scale']}, "
            f"volume_bias_weight={self.rhmc_config['volume_bias_weight']}"
        )
        
        self.rhmc_posterior = RiemannianRHMCPosterior(self.mock_model, self.rhmc_config)
        
    def test_initial_riemannian_sampling(self, n_samples=500, n_test_points=10):
        """Test Phase 1: Initial Riemannian sampling z₀ ~ N_Riem(μ, α·G(μ))"""
        print("🔬 Testing Initial Riemannian Sampling...")
        
        results = {
            'samples': [],
            'mu_points': [],
            'distances_to_centroids': [],
            'density_preservation': [],
            'comparison_gaussian': []
        }
        
        # Test different encoder means around centroids
        test_mus = []
        for i in range(n_test_points):
            # Place test points near centroids with some noise
            centroid_idx = i % len(self.mock_model.centroids_tens)
            base_centroid = self.mock_model.centroids_tens[centroid_idx]
            noise = torch.randn_like(base_centroid) * 0.3
            test_mu = base_centroid + noise
            test_mus.append(test_mu)
        
        test_mus = torch.stack(test_mus).to(self.device)  # [n_test_points, latent_dim]
        log_var = torch.zeros_like(test_mus)  # Not used in Riemannian sampling
        
        context = torch.enable_grad if self.grad_enabled else torch.no_grad
        with context():
            # Sample using RHMC initial sampling only (no exploration)
            old_steps = self.rhmc_posterior.rhmc_steps
            self.rhmc_posterior.rhmc_steps = 0  # Disable exploration
            
            samples = []
            for i in range(n_samples):
                sample = self.rhmc_posterior.sample_riemannian_rhmc_posterior(test_mus, log_var)
                samples.append(sample)
            
            samples = torch.stack(samples, dim=0)  # [n_samples, n_test_points, latent_dim]
            self.rhmc_posterior.rhmc_steps = old_steps  # Restore
            
            # Compute diagnostics
            for i in range(n_test_points):
                mu = test_mus[i]
                sample_set = samples[:, i, :]  # [n_samples, latent_dim]
                
                # 1. Distance to nearest centroid
                centroid_distances = torch.cdist(sample_set, self.mock_model.centroids_tens)
                min_distances = torch.min(centroid_distances, dim=-1)[0]
                results['distances_to_centroids'].append(min_distances.cpu().numpy())
                
                # 2. Density preservation: log det(G^-1(z))
                G_inv_samples = self.mock_model.G_inv(sample_set)
                log_det_G_inv = torch.logdet(G_inv_samples)
                results['density_preservation'].append(log_det_G_inv.cpu().numpy())
                
                # 3. Comparison with standard Gaussian
                eps = torch.randn_like(sample_set)
                gaussian_samples = mu.unsqueeze(0) + eps * 0.1  # Small std for comparison
                results['comparison_gaussian'].append(gaussian_samples.cpu().numpy())
                
                results['samples'].append(sample_set.cpu().numpy())
                results['mu_points'].append(mu.cpu().numpy())
        
        return results
    
    def test_rhmc_exploration(self, n_samples=100, n_test_points=5):
        """Test Phase 2: RHMC exploration dynamics"""
        print("🔬 Testing RHMC Exploration...")
        
        results = {
            'trajectories': [],
            'step_by_step': [],
            'volume_preservation': [],
            'gradient_field': []
        }
        
        # Test points near centroids
        test_mus = self.mock_model.centroids_tens[:n_test_points] + torch.randn(n_test_points, self.latent_dim, device=self.device) * 0.2
        log_var = torch.zeros_like(test_mus)
        
        context = torch.enable_grad if self.grad_enabled else torch.no_grad
        with context():
            for i in range(n_test_points):
                mu = test_mus[i:i+1]  # [1, latent_dim]
                
                # Track step-by-step RHMC evolution
                trajectory = []
                
                # Initial sampling
                z0 = self.rhmc_posterior._sample_initial_riemannian(mu, log_var[:1])
                trajectory.append(z0.cpu().numpy())
                
                # Manual RHMC steps with tracking
                z = z0.clone()
                for step in range(self.rhmc_posterior.rhmc_steps):
                    # Sample momentum
                    rho = self.rhmc_posterior._sample_momentum(z)
                    
                    # Leapfrog step
                    z_new, rho_new = self.rhmc_posterior._leapfrog_step(z, rho, self.rhmc_posterior.rhmc_step_size)
                    
                    trajectory.append(z_new.cpu().numpy())
                    z = z_new
                
                results['trajectories'].append(np.array(trajectory))
        
        # Compute gradient field for visualization
        if self.latent_dim == 2:
            x_range = torch.linspace(-4, 4, 20, device=self.device)
            y_range = torch.linspace(-4, 4, 20, device=self.device)
            X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
            grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
            
            gradients = []
            for point in grid_points:
                grad = self.rhmc_posterior._compute_potential_gradient(point.unsqueeze(0))
                gradients.append(grad.cpu().numpy())
            
            results['gradient_field'] = {
                'grid': grid_points.cpu().numpy(),
                'gradients': np.array(gradients),
                'X': X.cpu().numpy(),
                'Y': Y.cpu().numpy()
            }
        
        return results
    
    def compute_manifold_metrics(self, samples, mu_points):
        """Compute comprehensive manifold adherence metrics."""
        metrics = {}
        
        # 1. Spatial confinement: percentage within 2σ of centroids
        confined_samples = 0
        total_samples = 0
        
        for i, (sample_set, mu) in enumerate(zip(samples, mu_points)):
            sample_tensor = torch.tensor(sample_set, device=self.device)
            
            # Find nearest centroid
            distances = torch.cdist(sample_tensor, self.mock_model.centroids_tens)
            min_distances, nearest_centroids = torch.min(distances, dim=-1)
            
            # Compute 2σ threshold based on metric at nearest centroid
            for j, (sample, nearest_idx) in enumerate(zip(sample_tensor, nearest_centroids)):
                centroid = self.mock_model.centroids_tens[nearest_idx]
                G_inv = self.mock_model.G_inv(sample.unsqueeze(0))
                
                # Mahalanobis distance to centroid
                diff = sample - centroid
                maha_dist = torch.sqrt(torch.einsum('i,ij,j->', diff, G_inv[0], diff))
                
                if maha_dist < 2.0:  # 2σ threshold
                    confined_samples += 1
                total_samples += 1
        
        metrics['spatial_confinement'] = confined_samples / total_samples if total_samples > 0 else 0.0
        
        # 2. Density preservation
        all_samples = np.concatenate(samples, axis=0)
        sample_tensor = torch.tensor(all_samples, device=self.device)
        G_inv_all = self.mock_model.G_inv(sample_tensor)
        log_det_variation = torch.std(torch.logdet(G_inv_all)).item()
        metrics['density_variation'] = log_det_variation
        
        return metrics
    
    def visualize_results(self, initial_results, rhmc_results, save_path=None):
        """Create comprehensive visualization of diagnostic results."""
        if self.latent_dim != 2:
            print("⚠️ Visualization only available for 2D latent space")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RHMC Posterior Diagnostics', fontsize=16)
        
        # Plot 1: Initial Riemannian Sampling
        ax = axes[0, 0]
        centroids = self.mock_model.centroids_tens.cpu().numpy()
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='*', label='Centroids', alpha=0.8)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(initial_results['samples'])))
        for i, (samples, mu, color) in enumerate(zip(initial_results['samples'], initial_results['mu_points'], colors)):
            ax.scatter(samples[:, 0], samples[:, 1], c=[color], alpha=0.6, s=20, label=f'Samples {i}' if i < 3 else "")
            ax.scatter(mu[0], mu[1], c='green', marker='x', s=100, alpha=0.8)
        
        ax.set_title('Initial Riemannian Sampling')
        ax.set_xlabel('z₁')
        ax.set_ylabel('z₂')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: RHMC Trajectories
        ax = axes[0, 1]
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='*', label='Centroids', alpha=0.8)
        
        for i, trajectory in enumerate(rhmc_results['trajectories']):
            traj = trajectory.squeeze()
            ax.plot(traj[:, 0], traj[:, 1], 'o-', alpha=0.7, linewidth=2, markersize=4, label=f'Trajectory {i}' if i < 3 else "")
            ax.scatter(traj[0, 0], traj[0, 1], c='green', marker='s', s=60, alpha=0.8)  # Start
            ax.scatter(traj[-1, 0], traj[-1, 1], c='blue', marker='D', s=60, alpha=0.8)  # End
        
        ax.set_title('RHMC Trajectories')
        ax.set_xlabel('z₁')
        ax.set_ylabel('z₂')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Gradient Field
        ax = axes[0, 2]
        if 'gradient_field' in rhmc_results and rhmc_results['gradient_field']:
            gf = rhmc_results['gradient_field']
            X, Y = gf['X'], gf['Y']
            gradients = gf['gradients'].squeeze()
            U = gradients[:, 0].reshape(X.shape)
            V = gradients[:, 1].reshape(X.shape)
            
            ax.quiver(X, Y, -U, -V, alpha=0.6, scale=20)  # Negative because gradient points toward high density
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='*', label='Centroids')
        
        ax.set_title('Potential Gradient Field')
        ax.set_xlabel('z₁')
        ax.set_ylabel('z₂')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Distance Distribution
        ax = axes[1, 0]
        all_distances = np.concatenate(initial_results['distances_to_centroids'])
        ax.hist(all_distances, bins=30, alpha=0.7, density=True, label='Initial Sampling')
        ax.axvline(np.mean(all_distances), color='red', linestyle='--', label=f'Mean: {np.mean(all_distances):.3f}')
        ax.set_title('Distance to Nearest Centroid')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Density Preservation
        ax = axes[1, 1]
        all_densities = np.concatenate(initial_results['density_preservation'])
        ax.hist(all_densities, bins=30, alpha=0.7, density=True, label='log det(G⁻¹)')
        ax.axvline(np.mean(all_densities), color='red', linestyle='--', label=f'Mean: {np.mean(all_densities):.3f}')
        ax.set_title('Density Preservation')
        ax.set_xlabel('log det(G⁻¹)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Metrics Summary
        ax = axes[1, 2]
        metrics = self.compute_manifold_metrics(initial_results['samples'], initial_results['mu_points'])
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax.bar(range(len(metric_names)), metric_values, alpha=0.7)
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels([name.replace('_', '\n') for name in metric_names], rotation=45)
        ax.set_title('Manifold Adherence Metrics')
        ax.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Diagnostic plots saved to {save_path}")
        
        plt.show()
        
        return fig


def main():
    """Run comprehensive RHMC posterior diagnostics."""
    print("🚀 Starting RHMC Posterior Diagnostics...")
    
    # Initialize diagnostics
    diagnostics = RHMCPosteriorDiagnostics(latent_dim=2, n_centroids=5)
    
    # Phase 1: Test initial Riemannian sampling
    print("\n" + "="*50)
    print("PHASE 1: Initial Riemannian Sampling")
    print("="*50)
    
    init_samples = int(os.getenv("DIAG_INITIAL_SAMPLES", "200"))
    init_points = int(os.getenv("DIAG_INITIAL_TEST_POINTS", "8"))
    initial_results = diagnostics.test_initial_riemannian_sampling(n_samples=init_samples, n_test_points=init_points)
    
    # Compute and display initial metrics
    initial_metrics = diagnostics.compute_manifold_metrics(initial_results['samples'], initial_results['mu_points'])
    print(f"📊 Initial Sampling Metrics:")
    for metric, value in initial_metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # Phase 2: Test RHMC exploration
    print("\n" + "="*50)
    print("PHASE 2: RHMC Exploration")
    print("="*50)
    
    rhmc_samples = int(os.getenv("DIAG_RHMC_SAMPLES", "50"))
    rhmc_points = int(os.getenv("DIAG_RHMC_TEST_POINTS", "5"))
    rhmc_results = diagnostics.test_rhmc_exploration(n_samples=rhmc_samples, n_test_points=rhmc_points)
    
    print(f"📊 RHMC Exploration Results:")
    print(f"   Number of trajectories: {len(rhmc_results['trajectories'])}")
    print(f"   Steps per trajectory: {diagnostics.rhmc_posterior.rhmc_steps}")
    print(f"   Step size: {diagnostics.rhmc_posterior.rhmc_step_size}")
    
    # Visualization
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)
    
    save_path = Path(__file__).parent.parent / "outputs" / "rhmc_diagnostics.png"
    save_path.parent.mkdir(exist_ok=True)
    
    diagnostics.visualize_results(initial_results, rhmc_results, save_path)
    
    # Summary
    print("\n" + "="*50)
    print("DIAGNOSTIC SUMMARY")
    print("="*50)
    
    success_criteria = {
        'spatial_confinement': (initial_metrics.get('spatial_confinement', 0), 0.95, "95% samples within 2σ of centroids"),
        'density_variation': (initial_metrics.get('density_variation', float('inf')), 0.5, "log det(G⁻¹) variation < 50% (excellent if < 32%)")
    }
    
    print("Success Criteria Check:")
    for criterion, (actual, target, description) in success_criteria.items():
        status = "✅ PASS" if (criterion == 'spatial_confinement' and actual >= target) or (criterion == 'density_variation' and actual <= target) else "❌ FAIL"
        print(f"   {status} {description}: {actual:.4f} (target: {target})")
    
    print(f"\n🎯 Diagnostic complete! Results saved to {save_path}")


if __name__ == "__main__":
    main()
