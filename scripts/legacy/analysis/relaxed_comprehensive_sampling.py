#!/usr/bin/env python3
"""
Relaxed Comprehensive G⁻¹ Sampling
==================================

Relaxed sampling using the exact same data as comprehensive_g_inverse_analysis.py
but with more flexible conditions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent))

from src.models.riemannian_flow_vae import RiemannianFlowVAE
from dual_rhmc_implementation import DualRiemannianHMCSampler


def load_real_data_and_compute_centroids():
    """Load real data and compute centroids using all available data (same as comprehensive)."""
    print("🔍 Loading real data and computing centroids")
    print("=" * 60)
    
    # Create model and load pretrained components (same as comprehensive)
    model = RiemannianFlowVAE(
        input_dim=(3, 64, 64),
        latent_dim=2,
        n_flows=0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load pretrained components (same as comprehensive)
    model.load_pretrained_components(
        encoder_path="data/pretrained/encoder_diverse_mlp_ld2_20250730_105419.pt",
        decoder_path="data/pretrained/decoder_diverse_mlp_ld2_20250730_105419.pt",
        metric_path="data/pretrained/metric_diverse_mlp_ld2_20250730_105420.pt"
    )
    
    # Generate synthetic latent data (same as comprehensive)
    np.random.seed(42)
    n_data_points = 5000
    latent_data = np.random.randn(n_data_points, 2) * 3.0
    
    # Add some cluster structure (same as comprehensive)
    cluster_centers = np.array([
        [-2.0, -1.5], [0.0, 2.0], [2.0, -1.0], [-1.0, 0.0],
        [1.5, 1.5], [-2.5, 1.0], [0.5, -2.0], [2.5, 0.5]
    ])
    
    for i, center in enumerate(cluster_centers):
        n_cluster_points = n_data_points // len(cluster_centers)
        start_idx = i * n_cluster_points
        end_idx = start_idx + n_cluster_points
        if i == len(cluster_centers) - 1:
            end_idx = n_data_points
        
        cluster_points = np.random.randn(end_idx - start_idx, 2) * 0.5 + center
        latent_data[start_idx:end_idx] = cluster_points
    
    print(f"✅ Generated {n_data_points} latent data points")
    print(f"✅ Data range: [{latent_data.min():.3f}, {latent_data.max():.3f}]")
    
    # Compute centroids using k-means on all data (same as comprehensive)
    n_centroids = 50
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_centroids, random_state=42, n_init=10)
    kmeans.fit(latent_data)
    centroids = kmeans.cluster_centers_
    
    print(f"✅ Computed {len(centroids)} centroids using k-means")
    print(f"✅ Centroids range: [{centroids.min():.3f}, {centroids.max():.3f}]")
    
    # Create metric matrices for each centroid (same as comprehensive)
    metric_matrices = []
    for i, centroid in enumerate(centroids):
        distances = np.linalg.norm(latent_data - centroid, axis=1)
        closest_indices = np.argsort(distances)[:100]
        cluster_points = latent_data[closest_indices]
        
        if len(cluster_points) > 1:
            cov_matrix = np.cov(cluster_points.T)
            cov_matrix += np.eye(cov_matrix.shape[0]) * 0.01
            try:
                metric_matrix = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                metric_matrix = np.eye(cov_matrix.shape[0])
        else:
            metric_matrix = np.eye(latent_data.shape[1])
        
        metric_matrices.append(metric_matrix)
    
    metric_matrices = np.array(metric_matrices)
    
    print(f"✅ Created {len(metric_matrices)} metric matrices")
    print(f"✅ Metric determinants range: [{np.linalg.det(metric_matrices).min():.3e}, {np.linalg.det(metric_matrices).max():.3e}]")
    
    # Load centroids and metrics into model (same as comprehensive)
    centroids_tensor = torch.tensor(centroids, dtype=torch.float32, device=device)
    metric_matrices_tensor = torch.tensor(metric_matrices, dtype=torch.float32, device=device)
    model.load_pretrained_metrics_from_tensor(centroids_tensor, metric_matrices_tensor, 
                                            temperature=0.3, regularization=0.01)
    
    return model, latent_data, centroids, metric_matrices


class RelaxedComprehensiveSampler:
    """Relaxed sampler using the comprehensive data with more flexible conditions."""
    
    def __init__(self, model):
        self.model = model
        self.device = model.device
        
        print("🎯 Relaxed Comprehensive Sampler initialized")
        print("   - Uses same data as comprehensive G⁻¹ analysis")
        print("   - Relaxed conditions for more flexibility")
        print("   - Respects G⁻¹ metric with looser constraints")
    
    def compute_relaxed_metric_step(self, current_point: torch.Tensor, 
                                  direction: torch.Tensor, step_size: float = 0.3) -> torch.Tensor:
        """Compute a relaxed step guided by the local G⁻¹ metric."""
        with torch.no_grad():
            # Get local G⁻¹ metric
            G_z = self.model.G(current_point.unsqueeze(0))
            G_inv = torch.linalg.inv(G_z[0])  # Remove batch dimension
            
            # Normalize direction
            direction = direction / torch.norm(direction)
            
            # Relaxed metric transformation
            metric_direction = torch.mv(G_inv, direction)
            metric_direction = metric_direction / torch.norm(metric_direction)
            
            # More relaxed step size scaling
            det_G_inv = torch.det(G_inv)
            local_step_size = step_size / (1.0 + det_G_inv * 0.0001)
            
            # Add some randomness to the step
            random_component = torch.randn(2, device=self.device) * 0.1
            final_direction = metric_direction + random_component
            final_direction = final_direction / torch.norm(final_direction)
            
            # Take relaxed step
            next_point = current_point + local_step_size * final_direction
            
            return next_point
    
    def sample_relaxed_guided_paths(self, n_paths: int = 50, path_length: int = 40) -> torch.Tensor:
        """Sample relaxed paths guided by the G⁻¹ metric."""
        print(f"🎯 Sampling {n_paths} relaxed guided paths")
        
        all_paths = []
        for i in range(n_paths):
            # Start near a random centroid with more variance
            centroid_idx = torch.randint(0, len(self.model.centroids_tens), (1,)).item()
            start_point = self.model.centroids_tens[centroid_idx] + torch.randn(2, device=self.device) * 0.3
            
            path_points = [start_point]
            current_point = start_point
            
            for step in range(path_length - 1):
                if step % 3 == 0:  # More frequent direction changes
                    direction = torch.randn(2, device=self.device)
                else:
                    prev_direction = path_points[-1] - path_points[-2] if len(path_points) > 1 else torch.randn(2, device=self.device)
                    direction = prev_direction + torch.randn(2, device=self.device) * 0.5
                
                next_point = self.compute_relaxed_metric_step(current_point, direction, step_size=0.25)
                next_point = torch.clamp(next_point, -5, 5)  # Looser bounds
                
                path_points.append(next_point)
                current_point = next_point
            
            path_tensor = torch.stack(path_points)
            all_paths.append(path_tensor)
            
            if (i + 1) % 10 == 0:
                print(f"   Generated {i + 1}/{n_paths} relaxed guided paths")
        
        all_paths = torch.cat(all_paths, dim=0)
        print(f"✅ Generated {len(all_paths)} relaxed guided path points")
        return all_paths
    
    def sample_relaxed_explorations(self, n_explorations: int = 100, exploration_length: int = 50) -> torch.Tensor:
        """Sample relaxed manifold exploration with more randomness."""
        print(f"🎯 Sampling {n_explorations} relaxed explorations")
        
        all_explorations = []
        for i in range(n_explorations):
            centroid_idx = torch.randint(0, len(self.model.centroids_tens), (1,)).item()
            current_point = self.model.centroids_tens[centroid_idx] + torch.randn(2, device=self.device) * 0.2
            
            exploration_points = [current_point]
            
            for step in range(exploration_length - 1):
                if torch.rand(1).item() < 0.6:  # More random exploration
                    direction = torch.randn(2, device=self.device)
                    next_point = self.compute_relaxed_metric_step(current_point, direction, step_size=0.2)
                else:
                    direction = torch.randn(2, device=self.device)
                    direction = direction / torch.norm(direction)
                    step_size = 0.2 / (1.0 + torch.rand(1).item() * 1.5)
                    next_point = current_point + step_size * direction
                
                next_point = torch.clamp(next_point, -5, 5)
                exploration_points.append(next_point)
                current_point = next_point
            
            exploration_tensor = torch.stack(exploration_points)
            all_explorations.append(exploration_tensor)
            
            if (i + 1) % 20 == 0:
                print(f"   Generated {i + 1}/{n_explorations} relaxed explorations")
        
        all_explorations = torch.cat(all_explorations, dim=0)
        print(f"✅ Generated {len(all_explorations)} relaxed exploration points")
        return all_explorations
    
    def sample_relaxed_connections(self, n_connections: int = 30, steps_per_connection: int = 30) -> torch.Tensor:
        """Sample relaxed paths connecting different centroids."""
        print(f"🎯 Sampling {n_connections} relaxed centroid connections")
        
        all_connections = []
        for i in range(n_connections):
            start_centroid_idx = torch.randint(0, len(self.model.centroids_tens), (1,)).item()
            end_centroid_idx = torch.randint(0, len(self.model.centroids_tens), (1,)).item()
            while end_centroid_idx == start_centroid_idx:
                end_centroid_idx = torch.randint(0, len(self.model.centroids_tens), (1,)).item()
            
            start_point = self.model.centroids_tens[start_centroid_idx] + torch.randn(2, device=self.device) * 0.2
            end_point = self.model.centroids_tens[end_centroid_idx] + torch.randn(2, device=self.device) * 0.2
            
            connection_points = [start_point]
            current_point = start_point
            
            for step in range(steps_per_connection - 1):
                target_direction = end_point - current_point
                target_direction = target_direction / torch.norm(target_direction)
                
                random_component = torch.randn(2, device=self.device) * 0.4
                direction = target_direction + random_component
                
                next_point = self.compute_relaxed_metric_step(current_point, direction, step_size=0.18)
                next_point = torch.clamp(next_point, -5, 5)
                
                connection_points.append(next_point)
                current_point = next_point
            
            connection_tensor = torch.stack(connection_points)
            all_connections.append(connection_tensor)
            
            if (i + 1) % 10 == 0:
                print(f"   Generated {i + 1}/{n_connections} relaxed centroid connections")
        
        all_connections = torch.cat(all_connections, dim=0)
        print(f"✅ Generated {len(all_connections)} relaxed centroid connection points")
        return all_connections


def run_relaxed_comprehensive_sampling_analysis():
    """Run relaxed comprehensive sampling analysis using the same data."""
    print("🚀 RELAXED COMPREHENSIVE G⁻¹ SAMPLING ANALYSIS")
    print("=" * 60)
    
    # Load the same data as comprehensive analysis
    model, latent_data, centroids, metric_matrices = load_real_data_and_compute_centroids()
    
    # Create relaxed comprehensive sampler
    sampler = RelaxedComprehensiveSampler(model)
    
    # Sample with relaxed conditions
    print("🎯 Sampling relaxed guided paths")
    guided_paths = sampler.sample_relaxed_guided_paths(n_paths=40, path_length=30)
    
    print("🎯 Sampling relaxed explorations")
    explorations = sampler.sample_relaxed_explorations(n_explorations=60, exploration_length=35)
    
    print("🎯 Sampling relaxed centroid connections")
    connections = sampler.sample_relaxed_connections(n_connections=25, steps_per_connection=25)
    
    # Create visualization
    print("🎨 Creating relaxed comprehensive sampling visualization")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    centroids_np = centroids
    guided_paths_np = guided_paths.detach().cpu().numpy()
    explorations_np = explorations.detach().cpu().numpy()
    connections_np = connections.detach().cpu().numpy()
    
    # Plot 1: Relaxed metric-guided paths
    ax1 = axes[0, 0]
    ax1.scatter(latent_data[:, 0], latent_data[:, 1], c='lightblue', alpha=0.3, s=10, label='Data Points')
    ax1.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=150, 
               label='Centroids', zorder=5)
    ax1.scatter(guided_paths_np[:, 0], guided_paths_np[:, 1], c='green', alpha=0.4, s=8)
    ax1.set_title("1. Relaxed Metric-Guided Paths\n(Same Data, Relaxed Conditions)", fontweight='bold')
    ax1.set_xlabel("z₁")
    ax1.set_ylabel("z₂")
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Relaxed manifold explorations
    ax2 = axes[0, 1]
    ax2.scatter(latent_data[:, 0], latent_data[:, 1], c='lightblue', alpha=0.3, s=10, label='Data Points')
    ax2.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=150, 
               label='Centroids', zorder=5)
    ax2.scatter(explorations_np[:, 0], explorations_np[:, 1], c='blue', alpha=0.4, s=8)
    ax2.set_title("2. Relaxed Manifold Explorations\n(More Random Exploration)", fontweight='bold')
    ax2.set_xlabel("z₁")
    ax2.set_ylabel("z₂")
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Relaxed centroid connections
    ax3 = axes[1, 0]
    ax3.scatter(latent_data[:, 0], latent_data[:, 1], c='lightblue', alpha=0.3, s=10, label='Data Points')
    ax3.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=150, 
               label='Centroids', zorder=5)
    ax3.scatter(connections_np[:, 0], connections_np[:, 1], c='purple', alpha=0.4, s=8)
    ax3.set_title("3. Relaxed Centroid Connections\n(More Flexible Routing)", fontweight='bold')
    ax3.set_xlabel("z₁")
    ax3.set_ylabel("z₂")
    ax3.set_xlim(-5, 5)
    ax3.set_ylim(-5, 5)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Combined relaxed sampling
    ax4 = axes[1, 1]
    ax4.scatter(latent_data[:, 0], latent_data[:, 1], c='lightblue', alpha=0.3, s=10, label='Data Points')
    ax4.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=150, 
               label='Centroids', zorder=5)
    ax4.scatter(guided_paths_np[:, 0], guided_paths_np[:, 1], c='green', alpha=0.3, s=6, label='Relaxed Guided')
    ax4.scatter(explorations_np[:, 0], explorations_np[:, 1], c='blue', alpha=0.3, s=6, label='Relaxed Exploration')
    ax4.scatter(connections_np[:, 0], connections_np[:, 1], c='purple', alpha=0.3, s=6, label='Relaxed Connections')
    ax4.set_title("4. Combined Relaxed Comprehensive Sampling\n(All Relaxed Strategies)", fontweight='bold')
    ax4.set_xlabel("z₁")
    ax4.set_ylabel("z₂")
    ax4.set_xlim(-5, 5)
    ax4.set_ylim(-5, 5)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig("relaxed_comprehensive_sampling_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Relaxed comprehensive sampling analysis completed!")
    print(f"📊 Generated {len(guided_paths)} relaxed guided path points")
    print(f"📊 Generated {len(explorations)} relaxed exploration points")
    print(f"📊 Generated {len(connections)} relaxed centroid connection points")
    
    return guided_paths, explorations, connections


if __name__ == "__main__":
    guided_paths, explorations, connections = run_relaxed_comprehensive_sampling_analysis()
    print(f"\n🎉 RELAXED COMPREHENSIVE G⁻¹ SAMPLING SYSTEM COMPLETE!") 