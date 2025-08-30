#!/usr/bin/env python3
"""
Manifold-Guided Sampling System
===============================

Sample with flexibility while respecting the manifold structure.
Uses G⁻¹ metric to guide sampling with controlled exploration.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent))

from native_inverse_metric_system import NativeInverseMetricTensor


class ManifoldGuidedSampler:
    """
    Manifold-guided sampler that respects G⁻¹ metric structure with flexibility.
    
    Mathematical Framework:
    - Uses G⁻¹(z) to guide sampling directions
    - Combines metric-aware steps with exploration
    - Respects manifold curvature and structure
    """
    
    def __init__(self, metric_tensor: NativeInverseMetricTensor):
        self.metric_tensor = metric_tensor
        self.device = metric_tensor.centroids.device
        
        print("🎯 Manifold-Guided Sampler initialized")
        print("   - Respects G⁻¹ metric structure")
        print("   - Combines precision with flexibility")
        print("   - Guided exploration along manifold")
    
    def compute_metric_guided_step(self, current_point: torch.Tensor, 
                                 direction: torch.Tensor, step_size: float = 0.1) -> torch.Tensor:
        """
        Compute a step guided by the local G⁻¹ metric.
        
        Args:
            current_point: [latent_dim] current position
            direction: [latent_dim] desired direction
            step_size: step size parameter
            
        Returns:
            next_point: [latent_dim] new position
        """
        with torch.no_grad():
            # Get local G⁻¹ metric
            G_inv, _ = self.metric_tensor(current_point.unsqueeze(0))
            G_inv = G_inv[0]  # Remove batch dimension
            
            # Normalize direction
            direction = direction / torch.norm(direction)
            
            # Transform direction by G⁻¹ (metric-aware step)
            metric_direction = torch.mv(G_inv, direction)
            metric_direction = metric_direction / torch.norm(metric_direction)
            
            # Scale step size based on local metric determinant
            det_G_inv = torch.det(G_inv)
            local_step_size = step_size / (1.0 + det_G_inv * 0.001)  # Smaller steps in high-metric regions
            
            # Take step
            next_point = current_point + local_step_size * metric_direction
            
            return next_point
    
    def sample_metric_guided_paths(self, n_paths: int = 50, path_length: int = 30) -> torch.Tensor:
        """
        Sample paths guided by the G⁻¹ metric.
        
        Args:
            n_paths: number of paths to sample
            path_length: number of steps per path
            
        Returns:
            all_paths: [n_paths * path_length, latent_dim] all path points
        """
        print(f"🎯 Sampling {n_paths} metric-guided paths")
        
        all_paths = []
        
        for i in range(n_paths):
            # Start near a random centroid
            centroid_idx = torch.randint(0, len(self.metric_tensor.centroids), (1,)).item()
            start_point = self.metric_tensor.centroids[centroid_idx] + torch.randn(2, device=self.device) * 0.1
            
            path_points = [start_point]
            current_point = start_point
            
            for step in range(path_length - 1):
                # Choose direction with some randomness but guided by metric
                if step % 5 == 0:  # Every 5 steps, choose a new random direction
                    direction = torch.randn(2, device=self.device)
                else:
                    # Continue in similar direction with small perturbation
                    prev_direction = path_points[-1] - path_points[-2] if len(path_points) > 1 else torch.randn(2, device=self.device)
                    direction = prev_direction + torch.randn(2, device=self.device) * 0.3
                
                # Take metric-guided step
                next_point = self.compute_metric_guided_step(current_point, direction, step_size=0.15)
                
                # Ensure bounds
                next_point = torch.clamp(next_point, -4, 4)
                
                path_points.append(next_point)
                current_point = next_point
            
            path_tensor = torch.stack(path_points)
            all_paths.append(path_tensor)
            
            if (i + 1) % 10 == 0:
                print(f"   Generated {i + 1}/{n_paths} metric-guided paths")
        
        all_paths = torch.cat(all_paths, dim=0)
        print(f"✅ Generated {len(all_paths)} metric-guided path points")
        
        return all_paths
    
    def sample_manifold_exploration(self, n_explorations: int = 100, exploration_length: int = 40) -> torch.Tensor:
        """
        Sample manifold exploration with controlled randomness.
        
        Args:
            n_explorations: number of exploration sequences
            exploration_length: number of steps per exploration
            
        Returns:
            all_explorations: [n_explorations * exploration_length, latent_dim] all exploration points
        """
        print(f"🎯 Sampling {n_explorations} manifold explorations")
        
        all_explorations = []
        
        for i in range(n_explorations):
            # Start at a random centroid
            centroid_idx = torch.randint(0, len(self.metric_tensor.centroids), (1,)).item()
            current_point = self.metric_tensor.centroids[centroid_idx] + torch.randn(2, device=self.device) * 0.1
            
            exploration_points = [current_point]
            
            for step in range(exploration_length - 1):
                # Mix of random exploration and metric guidance
                if torch.rand(1).item() < 0.7:  # 70% metric-guided steps
                    # Metric-guided step
                    direction = torch.randn(2, device=self.device)
                    next_point = self.compute_metric_guided_step(current_point, direction, step_size=0.1)
                else:
                    # Pure random step (exploration)
                    direction = torch.randn(2, device=self.device)
                    direction = direction / torch.norm(direction)
                    step_size = 0.1 / (1.0 + torch.rand(1).item() * 2.0)  # Variable step size
                    next_point = current_point + step_size * direction
                
                # Ensure bounds
                next_point = torch.clamp(next_point, -4, 4)
                
                exploration_points.append(next_point)
                current_point = next_point
            
            exploration_tensor = torch.stack(exploration_points)
            all_explorations.append(exploration_tensor)
            
            if (i + 1) % 20 == 0:
                print(f"   Generated {i + 1}/{n_explorations} manifold explorations")
        
        all_explorations = torch.cat(all_explorations, dim=0)
        print(f"✅ Generated {len(all_explorations)} manifold exploration points")
        
        return all_explorations
    
    def sample_centroid_connections(self, n_connections: int = 30, steps_per_connection: int = 25) -> torch.Tensor:
        """
        Sample paths connecting different centroids with metric guidance.
        
        Args:
            n_connections: number of centroid connections
            steps_per_connection: number of steps per connection
            
        Returns:
            all_connections: [n_connections * steps_per_connection, latent_dim] all connection points
        """
        print(f"🎯 Sampling {n_connections} centroid connections")
        
        all_connections = []
        
        for i in range(n_connections):
            # Choose two different centroids
            start_centroid_idx = torch.randint(0, len(self.metric_tensor.centroids), (1,)).item()
            end_centroid_idx = torch.randint(0, len(self.metric_tensor.centroids), (1,)).item()
            while end_centroid_idx == start_centroid_idx:
                end_centroid_idx = torch.randint(0, len(self.metric_tensor.centroids), (1,)).item()
            
            start_point = self.metric_tensor.centroids[start_centroid_idx] + torch.randn(2, device=self.device) * 0.1
            end_point = self.metric_tensor.centroids[end_centroid_idx] + torch.randn(2, device=self.device) * 0.1
            
            connection_points = [start_point]
            current_point = start_point
            
            for step in range(steps_per_connection - 1):
                # Direction towards end point with metric guidance
                target_direction = end_point - current_point
                target_direction = target_direction / torch.norm(target_direction)
                
                # Add some randomness to avoid straight lines
                random_component = torch.randn(2, device=self.device) * 0.2
                direction = target_direction + random_component
                
                # Take metric-guided step
                next_point = self.compute_metric_guided_step(current_point, direction, step_size=0.12)
                
                # Ensure bounds
                next_point = torch.clamp(next_point, -4, 4)
                
                connection_points.append(next_point)
                current_point = next_point
            
            connection_tensor = torch.stack(connection_points)
            all_connections.append(connection_tensor)
            
            if (i + 1) % 10 == 0:
                print(f"   Generated {i + 1}/{n_connections} centroid connections")
        
        all_connections = torch.cat(all_connections, dim=0)
        print(f"✅ Generated {len(all_connections)} centroid connection points")
        
        return all_connections


def run_manifold_guided_sampling_analysis():
    """Run comprehensive manifold-guided sampling analysis."""
    print("🚀 MANIFOLD-GUIDED SAMPLING ANALYSIS")
    print("=" * 60)
    
    # Create native metric tensor (same as before)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create centroids and metrics (same as comprehensive analysis)
    torch.manual_seed(42)
    latent_data = torch.randn(3000, 2, device=device) * 2.5
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    kmeans.fit(latent_data.detach().cpu().numpy())
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
    
    # Create diverse G⁻¹ metrics
    inverse_metrics = []
    for i in range(len(centroids)):
        base_scale = 200.0 + i * 100.0
        eigenvals = torch.tensor([base_scale, base_scale * 0.7], device=device)
        metric_matrix = torch.diag(eigenvals)
        inverse_metrics.append(metric_matrix)
    
    inverse_metrics = torch.stack(inverse_metrics)
    
    # Create native metric tensor
    metric_tensor = NativeInverseMetricTensor(latent_dim=2)
    metric_tensor.load_inverse_metrics(
        centroids, inverse_metrics,
        temperature=2.0, regularization=1e-4
    )
    
    # Create manifold-guided sampler
    sampler = ManifoldGuidedSampler(metric_tensor)
    
    # Sample different types of paths
    print("🎯 Sampling metric-guided paths")
    guided_paths = sampler.sample_metric_guided_paths(n_paths=40, path_length=25)
    
    print("🎯 Sampling manifold explorations")
    explorations = sampler.sample_manifold_exploration(n_explorations=60, exploration_length=30)
    
    print("🎯 Sampling centroid connections")
    connections = sampler.sample_centroid_connections(n_connections=25, steps_per_connection=20)
    
    # Create visualization
    print("🎨 Creating manifold-guided sampling visualization")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Convert to numpy
    centroids_np = centroids.detach().cpu().numpy()
    guided_paths_np = guided_paths.detach().cpu().numpy()
    explorations_np = explorations.detach().cpu().numpy()
    connections_np = connections.detach().cpu().numpy()
    
    # Plot 1: Metric-guided paths
    ax1 = axes[0, 0]
    ax1.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=150, 
               label='Centroids', zorder=5)
    ax1.scatter(guided_paths_np[:, 0], guided_paths_np[:, 1], c='green', alpha=0.4, s=8)
    ax1.set_title("1. Metric-Guided Paths\n(G⁻¹-Aware Sampling)", fontweight='bold')
    ax1.set_xlabel("z₁")
    ax1.set_ylabel("z₂")
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Manifold explorations
    ax2 = axes[0, 1]
    ax2.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=150, 
               label='Centroids', zorder=5)
    ax2.scatter(explorations_np[:, 0], explorations_np[:, 1], c='blue', alpha=0.4, s=8)
    ax2.set_title("2. Manifold Explorations\n(Flexible Exploration)", fontweight='bold')
    ax2.set_xlabel("z₁")
    ax2.set_ylabel("z₂")
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Centroid connections
    ax3 = axes[1, 0]
    ax3.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=150, 
               label='Centroids', zorder=5)
    ax3.scatter(connections_np[:, 0], connections_np[:, 1], c='purple', alpha=0.4, s=8)
    ax3.set_title("3. Centroid Connections\n(Manifold-Aware Routing)", fontweight='bold')
    ax3.set_xlabel("z₁")
    ax3.set_ylabel("z₂")
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Combined sampling
    ax4 = axes[1, 1]
    ax4.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=150, 
               label='Centroids', zorder=5)
    ax4.scatter(guided_paths_np[:, 0], guided_paths_np[:, 1], c='green', alpha=0.3, s=6, label='Guided')
    ax4.scatter(explorations_np[:, 0], explorations_np[:, 1], c='blue', alpha=0.3, s=6, label='Exploration')
    ax4.scatter(connections_np[:, 0], connections_np[:, 1], c='purple', alpha=0.3, s=6, label='Connections')
    ax4.set_title("4. Combined Manifold Sampling\n(All Strategies)", fontweight='bold')
    ax4.set_xlabel("z₁")
    ax4.set_ylabel("z₂")
    ax4.set_xlim(-4, 4)
    ax4.set_ylim(-4, 4)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig("manifold_guided_sampling_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Manifold-guided sampling analysis completed!")
    print(f"📊 Generated {len(guided_paths)} metric-guided path points")
    print(f"📊 Generated {len(explorations)} manifold exploration points")
    print(f"📊 Generated {len(connections)} centroid connection points")
    
    return guided_paths, explorations, connections


if __name__ == "__main__":
    guided_paths, explorations, connections = run_manifold_guided_sampling_analysis()
    print(f"\n🎉 MANIFOLD-GUIDED SAMPLING SYSTEM COMPLETE!") 