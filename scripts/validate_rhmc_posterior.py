#!/usr/bin/env python3
"""
RHMC Posterior Validation Script
===============================

Comprehensive validation of RHMC posterior using real Stage B metrics.
Tests manifold adherence, stability, and training integration.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
import yaml
from typing import Dict, Any, Optional, Tuple
import warnings

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rlvae.models.components.riemannian_rhmc_posterior import RiemannianRHMCPosterior
from src.rlvae.models.components.metric_tensor import MetricTensor
from src.rlvae.models.components.manifold_constraints import ManifoldConstraints
from sklearn.decomposition import PCA


class RealDataRHMCValidator:
    """Validation using real Stage B metrics from ellipse_sequences."""
    
    def __init__(self, 
                 metric_path: str,
                 rhmc_config: Optional[Dict[str, Any]] = None,
                 device: Optional[torch.device] = None):
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metric_path = Path(metric_path)
        
        # Load Stage B metric
        print(f"📂 Loading Stage B metric from {self.metric_path}")
        self.metric_data = self._load_stage_b_metric()
        
        # Create mock model with real metric
        self.model = self._create_model_with_real_metric()
        
        # RHMC configuration
        self.rhmc_config = rhmc_config or {
            'rhmc_steps': 3,
            'rhmc_step_size': 0.001,
            'rhmc_alpha': 0.1,
            'eps_regularization': 1e-3,
            'max_grad_norm': 3.0,
            'adaptive_step_size': True,
            'use_manifold_constraints': True,
            'adaptive_constraints': True,
            'manifold_constraints': {
                'projection_strength': 0.5,
                'density_threshold': 0.1,
                'elastic_strength': 0.3,
                'max_projection_distance': 1.0
            }
        }
        
        # Create RHMC posterior
        self.rhmc_posterior = RiemannianRHMCPosterior(self.model, self.rhmc_config)
        
        # Comparison: standard metric-aware posterior
        from src.rlvae.models.components.posterior_sampler import PosteriorSampler
        self.standard_posterior = PosteriorSampler(self.model)
        
    def _load_stage_b_metric(self) -> Dict[str, Any]:
        """Load Stage B metric data."""
        if not self.metric_path.exists():
            raise FileNotFoundError(f"Metric file not found: {self.metric_path}")
        
        try:
            data = torch.load(self.metric_path, map_location='cpu', weights_only=False)
            
            required_keys = ['centroids', 'metric_matrices', 'temperature', 'regularization']
            for key in required_keys:
                if key not in data:
                    raise KeyError(f"Missing required key in metric data: {key}")
            
            print(f"✅ Loaded metric with {data['centroids'].shape[0]} centroids")
            print(f"   Latent dim: {data['centroids'].shape[1]}")
            print(f"   Temperature: {data['temperature']}")
            print(f"   Regularization: {data['regularization']}")
            
            return data
            
        except Exception as e:
            raise RuntimeError(f"Failed to load metric data: {e}")
    
    def _create_model_with_real_metric(self) -> nn.Module:
        """Create model with real Stage B metric."""
        
        class RealMetricModel(nn.Module):
            def __init__(self, metric_data, device):
                super().__init__()
                self.device = device
                
                # Load metric components
                self.centroids_tens = metric_data['centroids'].to(device)
                self.metric_matrices = metric_data['metric_matrices'].to(device)
                self.temperature = float(metric_data['temperature'])
                self.regularization = float(metric_data['regularization'])
                
                self.latent_dim = self.centroids_tens.shape[1]
                self.n_centroids = self.centroids_tens.shape[0]
                
            def G(self, z):
                """Compute metric tensor using real Stage B data."""
                batch_size = z.shape[0]
                
                # Compute distances to centroids
                diff = z.unsqueeze(1) - self.centroids_tens.unsqueeze(0)  # [B, K, D]
                distances = torch.norm(diff, dim=-1)  # [B, K]
                
                # Softmax weights based on temperature
                weights = torch.softmax(-distances / self.temperature, dim=-1)  # [B, K]
                
                # Weighted combination of metric matrices
                G = torch.einsum('bk,kij->bij', weights, self.metric_matrices)  # [B, D, D]
                
                # Add regularization
                I = torch.eye(self.latent_dim, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
                G = G + self.regularization * I
                
                return G
            
            def G_inv(self, z):
                """Compute inverse metric tensor."""
                G = self.G(z)
                return torch.linalg.inv(G)
        
        return RealMetricModel(self.metric_data, self.device)
    
    def validate_initial_sampling(self, n_samples: int = 1000, n_test_points: int = 20) -> Dict[str, Any]:
        """Validate initial Riemannian sampling with real data."""
        print(f"🔬 Validating Initial Riemannian Sampling ({n_samples} samples, {n_test_points} test points)")
        
        results = {
            'rhmc_samples': [],
            'standard_samples': [],
            'test_mus': [],
            'metrics': {}
        }
        
        # Create test points around real centroids
        test_mus = []
        for i in range(n_test_points):
            # Select random centroid and add noise
            centroid_idx = np.random.randint(0, self.model.n_centroids)
            base_centroid = self.model.centroids_tens[centroid_idx]
            noise = torch.randn_like(base_centroid) * 0.2
            test_mu = base_centroid + noise
            test_mus.append(test_mu)
        
        test_mus = torch.stack(test_mus).to(self.device)
        log_var = torch.zeros_like(test_mus)
        
        with torch.no_grad():
            # RHMC sampling (initial only)
            old_steps = self.rhmc_posterior.rhmc_steps
            self.rhmc_posterior.rhmc_steps = 0  # Initial sampling only
            
            rhmc_samples = []
            for _ in range(n_samples):
                sample = self.rhmc_posterior.sample_riemannian_rhmc_posterior(test_mus, log_var)
                rhmc_samples.append(sample)
            
            rhmc_samples = torch.stack(rhmc_samples, dim=0)  # [n_samples, n_test_points, latent_dim]
            self.rhmc_posterior.rhmc_steps = old_steps
            
            # Standard posterior sampling for comparison
            standard_samples = []
            for _ in range(n_samples):
                sample = self.standard_posterior.sample_metric_aware_posterior(test_mus, log_var)
                standard_samples.append(sample)
            
            standard_samples = torch.stack(standard_samples, dim=0)
            
            results['rhmc_samples'] = rhmc_samples.cpu().numpy()
            results['standard_samples'] = standard_samples.cpu().numpy()
            results['test_mus'] = test_mus.cpu().numpy()
            
            # Compute validation metrics
            results['metrics'] = self._compute_validation_metrics(
                rhmc_samples, standard_samples, test_mus
            )
        
        return results
    
    def validate_rhmc_exploration(self, n_trajectories: int = 50) -> Dict[str, Any]:
        """Validate full RHMC exploration with real data."""
        print(f"🔬 Validating RHMC Exploration ({n_trajectories} trajectories)")
        
        results = {
            'trajectories': [],
            'constraint_metrics': [],
            'stability_metrics': {}
        }
        
        # Test points near centroids
        test_indices = np.random.choice(self.model.n_centroids, min(n_trajectories, self.model.n_centroids), replace=False)
        test_mus = self.model.centroids_tens[test_indices] + torch.randn(len(test_indices), self.model.latent_dim, device=self.device) * 0.15
        log_var = torch.zeros_like(test_mus)
        
        with torch.no_grad():
            for i, mu in enumerate(test_mus):
                mu_batch = mu.unsqueeze(0)
                log_var_batch = log_var[i:i+1]
                
                # Track trajectory step by step
                trajectory = []
                constraint_metrics = []
                
                # Initial sampling
                z0 = self.rhmc_posterior._sample_initial_riemannian(mu_batch, log_var_batch)
                trajectory.append(z0.cpu().numpy())
                
                # RHMC exploration with constraint tracking
                z = z0.clone()
                rho = self.rhmc_posterior._sample_momentum(z)
                
                for step in range(self.rhmc_posterior.rhmc_steps):
                    z_before = z.clone()
                    
                    # Leapfrog step
                    z, rho, _ = self.rhmc_posterior._leapfrog_step(z, rho, self.rhmc_posterior.rhmc_step_size)
                    
                    # Apply constraints and track metrics
                    if self.rhmc_posterior.use_manifold_constraints:
                        z_after_constraints = self.rhmc_posterior.manifold_constraints.apply_manifold_constraints(
                            z, self.model, mu_reference=mu_batch
                        )
                        
                        # Compute constraint metrics
                        step_metrics = self.rhmc_posterior.manifold_constraints.compute_constraint_metrics(
                            z, z_after_constraints, self.model
                        )
                        constraint_metrics.append(step_metrics)
                        z = z_after_constraints
                    
                    trajectory.append(z.cpu().numpy())
                
                results['trajectories'].append(np.array(trajectory))
                results['constraint_metrics'].append(constraint_metrics)
        
        # Compute stability metrics
        results['stability_metrics'] = self._compute_stability_metrics(results['trajectories'])
        
        return results
    
    def _compute_validation_metrics(self, 
                                  rhmc_samples: torch.Tensor, 
                                  standard_samples: torch.Tensor, 
                                  test_mus: torch.Tensor) -> Dict[str, float]:
        """Compute comprehensive validation metrics."""
        metrics = {}
        
        # Convert to tensors for computation
        rhmc_tensor = torch.tensor(rhmc_samples, device=self.device)
        standard_tensor = torch.tensor(standard_samples, device=self.device)
        
        # 1. Spatial confinement (within 2σ of centroids)
        rhmc_confined = self._compute_spatial_confinement(rhmc_tensor)
        standard_confined = self._compute_spatial_confinement(standard_tensor)
        
        metrics['rhmc_spatial_confinement'] = rhmc_confined
        metrics['standard_spatial_confinement'] = standard_confined
        metrics['confinement_improvement'] = rhmc_confined - standard_confined
        
        # 2. Density preservation
        rhmc_density_var = self._compute_density_variation(rhmc_tensor)
        standard_density_var = self._compute_density_variation(standard_tensor)
        
        metrics['rhmc_density_variation'] = rhmc_density_var
        metrics['standard_density_variation'] = standard_density_var
        
        # 3. Distance to encoder means
        rhmc_distances = self._compute_mean_distances(rhmc_tensor, torch.tensor(test_mus, device=self.device))
        standard_distances = self._compute_mean_distances(standard_tensor, torch.tensor(test_mus, device=self.device))
        
        metrics['rhmc_mean_distance'] = rhmc_distances
        metrics['standard_mean_distance'] = standard_distances
        
        return metrics
    
    def _compute_spatial_confinement(self, samples: torch.Tensor) -> float:
        """Compute percentage of samples within 2σ of nearest centroids."""
        n_samples, n_test_points, latent_dim = samples.shape
        confined_count = 0
        total_count = 0
        
        for i in range(n_test_points):
            sample_set = samples[:, i, :]  # [n_samples, latent_dim]
            
            # Find nearest centroids
            distances = torch.cdist(sample_set, self.model.centroids_tens)
            nearest_indices = torch.argmin(distances, dim=-1)
            
            for j, (sample, nearest_idx) in enumerate(zip(sample_set, nearest_indices)):
                centroid = self.model.centroids_tens[nearest_idx]
                
                # Compute Mahalanobis distance using metric at sample point
                try:
                    G_inv = self.model.G_inv(sample.unsqueeze(0))
                    diff = sample - centroid
                    maha_dist = torch.sqrt(torch.einsum('i,ij,j->', diff, G_inv[0], diff))
                    
                    if maha_dist < 2.0:  # 2σ threshold
                        confined_count += 1
                except:
                    # Fallback to Euclidean distance
                    euclidean_dist = torch.norm(sample - centroid)
                    if euclidean_dist < 1.0:
                        confined_count += 1
                
                total_count += 1
        
        return confined_count / total_count if total_count > 0 else 0.0
    
    def _compute_density_variation(self, samples: torch.Tensor) -> float:
        """Compute variation in log det(G^-1) across samples."""
        n_samples, n_test_points, latent_dim = samples.shape
        all_samples = samples.reshape(-1, latent_dim)
        
        try:
            G_inv_all = self.model.G_inv(all_samples)
            log_det_values = torch.logdet(G_inv_all)
            return torch.std(log_det_values).item()
        except:
            return float('inf')
    
    def _compute_mean_distances(self, samples: torch.Tensor, test_mus: torch.Tensor) -> float:
        """Compute mean distance from samples to their corresponding encoder means."""
        n_samples, n_test_points, latent_dim = samples.shape
        
        total_distance = 0.0
        count = 0
        
        for i in range(n_test_points):
            sample_set = samples[:, i, :]  # [n_samples, latent_dim]
            mu = test_mus[i]
            
            distances = torch.norm(sample_set - mu.unsqueeze(0), dim=-1)
            total_distance += distances.mean().item()
            count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def _compute_stability_metrics(self, trajectories: list) -> Dict[str, float]:
        """Compute stability metrics from RHMC trajectories."""
        metrics = {}
        
        if not trajectories:
            return metrics
        
        # Trajectory lengths and movements
        movements = []
        max_movements = []
        
        for trajectory in trajectories:
            if len(trajectory) > 1:
                traj_movements = []
                for i in range(1, len(trajectory)):
                    movement = np.linalg.norm(trajectory[i] - trajectory[i-1])
                    traj_movements.append(movement)
                
                if traj_movements:
                    movements.extend(traj_movements)
                    max_movements.append(max(traj_movements))
        
        if movements:
            metrics['avg_step_movement'] = np.mean(movements)
            metrics['max_step_movement'] = np.max(movements)
            metrics['movement_std'] = np.std(movements)
        
        if max_movements:
            metrics['avg_max_trajectory_movement'] = np.mean(max_movements)
        
        return metrics
    
    def create_validation_plots(self, 
                              initial_results: Dict[str, Any], 
                              exploration_results: Dict[str, Any],
                              save_path: Optional[Path] = None) -> None:
        """Create comprehensive validation plots."""
        
        if self.model.latent_dim != 2:
            print("⚠️ Visualization only available for 2D latent space")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('RHMC Posterior Validation with Real Stage B Metrics', fontsize=16)
        
        # Plot 1: Initial sampling comparison
        ax = axes[0, 0]
        centroids = self.model.centroids_tens.cpu().numpy()
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=150, marker='*', 
                  label='Real Centroids', alpha=0.8, edgecolors='black')
        
        # RHMC samples
        rhmc_samples = initial_results['rhmc_samples']
        rhmc_flat = rhmc_samples.reshape(-1, 2)
        ax.scatter(rhmc_flat[:, 0], rhmc_flat[:, 1], c='blue', alpha=0.4, s=10, label='RHMC Initial')
        
        # Standard samples
        standard_samples = initial_results['standard_samples']
        standard_flat = standard_samples.reshape(-1, 2)
        ax.scatter(standard_flat[:, 0], standard_flat[:, 1], c='green', alpha=0.4, s=10, label='Standard')
        
        # Test means
        test_mus = initial_results['test_mus']
        ax.scatter(test_mus[:, 0], test_mus[:, 1], c='orange', marker='x', s=100, 
                  label='Test μ', alpha=0.8)
        
        ax.set_title('Initial Sampling Comparison')
        ax.set_xlabel('z₁')
        ax.set_ylabel('z₂')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: RHMC trajectories
        ax = axes[0, 1]
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=150, marker='*', 
                  label='Centroids', alpha=0.8, edgecolors='black')
        
        trajectories = exploration_results['trajectories']
        for i, trajectory in enumerate(trajectories[:10]):  # Show first 10 trajectories
            traj = trajectory.squeeze()
            if len(traj.shape) == 2 and traj.shape[1] == 2:
                ax.plot(traj[:, 0], traj[:, 1], 'o-', alpha=0.7, linewidth=2, markersize=4)
                ax.scatter(traj[0, 0], traj[0, 1], c='green', marker='s', s=60, alpha=0.8)  # Start
                ax.scatter(traj[-1, 0], traj[-1, 1], c='blue', marker='D', s=60, alpha=0.8)  # End
        
        ax.set_title('RHMC Exploration Trajectories')
        ax.set_xlabel('z₁')
        ax.set_ylabel('z₂')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Metrics comparison
        ax = axes[0, 2]
        metrics = initial_results['metrics']
        
        rhmc_metrics = [metrics['rhmc_spatial_confinement'], metrics['rhmc_density_variation'], metrics['rhmc_mean_distance']]
        standard_metrics = [metrics['standard_spatial_confinement'], metrics['standard_density_variation'], metrics['standard_mean_distance']]
        
        x = np.arange(3)
        width = 0.35
        
        ax.bar(x - width/2, rhmc_metrics, width, label='RHMC', alpha=0.8)
        ax.bar(x + width/2, standard_metrics, width, label='Standard', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Spatial\nConfinement', 'Density\nVariation', 'Mean\nDistance'])
        ax.set_title('Metrics Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Constraint effectiveness
        ax = axes[1, 0]
        if exploration_results['constraint_metrics']:
            constraint_data = exploration_results['constraint_metrics']
            
            # Average constraint metrics across trajectories
            avg_metrics = {}
            for traj_metrics in constraint_data:
                for step_metrics in traj_metrics:
                    for key, value in step_metrics.items():
                        if key not in avg_metrics:
                            avg_metrics[key] = []
                        avg_metrics[key].append(value)
            
            # Plot key metrics
            if avg_metrics:
                keys = list(avg_metrics.keys())[:4]  # Show first 4 metrics
                values = [np.mean(avg_metrics[key]) for key in keys]
                
                bars = ax.bar(range(len(keys)), values, alpha=0.7)
                ax.set_xticks(range(len(keys)))
                ax.set_xticklabels([key.replace('_', '\n') for key in keys], rotation=45)
                ax.set_title('Constraint Effectiveness')
                
                # Add value labels
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                           f'{value:.3f}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Stability metrics
        ax = axes[1, 1]
        stability = exploration_results['stability_metrics']
        
        if stability:
            stability_keys = list(stability.keys())
            stability_values = list(stability.values())
            
            bars = ax.bar(range(len(stability_keys)), stability_values, alpha=0.7)
            ax.set_xticks(range(len(stability_keys)))
            ax.set_xticklabels([key.replace('_', '\n') for key in stability_keys], rotation=45)
            ax.set_title('RHMC Stability Metrics')
            
            # Add value labels
            for bar, value in zip(bars, stability_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stability_values)*0.01, 
                       f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Success criteria
        ax = axes[1, 2]
        
        success_criteria = {
            'Spatial Confinement': (metrics['rhmc_spatial_confinement'], 0.95),
            'Density Variation': (metrics['rhmc_density_variation'], 0.1),
            'Mean Distance': (metrics['rhmc_mean_distance'], 0.5)
        }
        
        criteria_names = list(success_criteria.keys())
        actual_values = [criteria[0] for criteria in success_criteria.values()]
        target_values = [criteria[1] for criteria in success_criteria.values()]
        
        x = np.arange(len(criteria_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, actual_values, width, label='Actual', alpha=0.8)
        bars2 = ax.bar(x + width/2, target_values, width, label='Target', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace(' ', '\n') for name in criteria_names])
        ax.set_title('Success Criteria')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Validation plots saved to {save_path}")
        
        plt.show()
    
    def run_full_validation(self, save_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Run complete validation pipeline."""
        print("🚀 Starting Full RHMC Posterior Validation with Real Data")
        print("="*60)
        
        results = {}
        
        # Phase 1: Initial sampling validation
        print("\n📊 Phase 1: Initial Sampling Validation")
        initial_results = self.validate_initial_sampling(n_samples=500, n_test_points=15)
        results['initial_sampling'] = initial_results
        
        # Phase 2: RHMC exploration validation
        print("\n📊 Phase 2: RHMC Exploration Validation")
        exploration_results = self.validate_rhmc_exploration(n_trajectories=30)
        results['rhmc_exploration'] = exploration_results
        
        # Create validation plots
        print("\n📊 Phase 3: Visualization")
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            plot_path = save_dir / "rhmc_validation_real_data.png"
        else:
            plot_path = None
        
        self.create_validation_plots(initial_results, exploration_results, plot_path)
        
        # Summary report
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        metrics = initial_results['metrics']
        
        success_criteria = [
            ('Spatial Confinement', metrics['rhmc_spatial_confinement'], 0.95, '≥'),
            ('Density Variation', metrics['rhmc_density_variation'], 0.1, '≤'),
            ('Mean Distance', metrics['rhmc_mean_distance'], 0.5, '≤')
        ]
        
        print("Success Criteria Check:")
        all_passed = True
        for name, actual, target, operator in success_criteria:
            if operator == '≥':
                passed = actual >= target
            else:  # operator == '≤'
                passed = actual <= target
            
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status} {name}: {actual:.4f} {operator} {target}")
            
            if not passed:
                all_passed = False
        
        print(f"\n🎯 Overall Status: {'✅ ALL CRITERIA PASSED' if all_passed else '❌ SOME CRITERIA FAILED'}")
        
        # Configuration recommendations
        if not all_passed:
            print("\n💡 Configuration Recommendations:")
            if metrics['rhmc_spatial_confinement'] < 0.95:
                print("   - Increase manifold_constraints.projection_strength")
                print("   - Decrease rhmc_step_size for more conservative exploration")
            if metrics['rhmc_density_variation'] > 0.1:
                print("   - Enable adaptive_constraints")
                print("   - Increase manifold_constraints.elastic_strength")
            if metrics['rhmc_mean_distance'] > 0.5:
                print("   - Reduce rhmc_alpha for tighter initial sampling")
                print("   - Increase manifold_constraints.elastic_strength")
        
        results['summary'] = {
            'all_criteria_passed': all_passed,
            'success_criteria': success_criteria,
            'recommendations': not all_passed
        }
        
        return results


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(description='Validate RHMC Posterior with Real Data')
    parser.add_argument('--metric-path', type=str, required=True,
                       help='Path to Stage B metric file')
    parser.add_argument('--output-dir', type=str, default='outputs/rhmc_validation',
                       help='Output directory for results')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to RHMC configuration YAML file')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    rhmc_config = None
    if args.config:
        with open(args.config, 'r') as f:
            rhmc_config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run validation
    try:
        validator = RealDataRHMCValidator(
            metric_path=args.metric_path,
            rhmc_config=rhmc_config
        )
        
        results = validator.run_full_validation(save_dir=output_dir)
        
        # Save results
        results_path = output_dir / "validation_results.yaml"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for YAML serialization
            yaml_results = {}
            for key, value in results.items():
                if key in ['initial_sampling', 'rhmc_exploration']:
                    yaml_results[key] = {'metrics': value.get('metrics', {})}
                else:
                    yaml_results[key] = value
            
            yaml.dump(yaml_results, f, default_flow_style=False)
        
        print(f"\n💾 Results saved to {results_path}")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

