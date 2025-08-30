#!/usr/bin/env python3
"""
Geodesic Sampling System
========================

Sample geodesics that follow the manifold structure with flexibility.
Uses G⁻¹ metric to compute geodesic paths between points.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Tuple, List, Optional
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent))

from native_inverse_metric_system import NativeInverseMetricTensor


class GeodesicSampler:
    """
    Geodesic sampler that follows the manifold structure.
    
    Mathematical Framework:
    - Geodesic equation: d²z/dt² + Γᵢⱼᵏ dzᵢ/dt dzⱼ/dt = 0
    - Christoffel symbols: Γᵢⱼᵏ = ½G⁻¹ᵏˡ(∂ᵢG⁻¹ⱼˡ + ∂ⱼG⁻¹ᵢˡ - ∂ˡG⁻¹ᵢⱼ)
    - Paths follow the manifold's natural geometry
    """
    
    def __init__(self, metric_tensor: NativeInverseMetricTensor):
        self.metric_tensor = metric_tensor
        self.device = metric_tensor.centroids.device
        
        print("🎯 Geodesic Sampler initialized")
        print("   - Follows manifold structure via G⁻¹ metric")
        print("   - Computes geodesic paths between points")
        print("   - Allows flexible exploration along manifold")
    
    def compute_christoffel_symbols(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute Christoffel symbols Γᵢⱼᵏ for geodesic integration.
        
        Args:
            z: [batch_size, latent_dim] points
            
        Returns:
            christoffel: [batch_size, latent_dim, latent_dim, latent_dim] Christoffel symbols
        """
        batch_size, latent_dim = z.shape
        z.requires_grad_(True)
        
        # Compute G⁻¹ and its derivatives
        G_inv, _ = self.metric_tensor(z)
        
        # Compute derivatives ∂G⁻¹/∂z
        grad_G_inv = torch.autograd.grad(
            G_inv.sum(), z, create_graph=True, retain_graph=True
        )[0]  # [batch_size, latent_dim, latent_dim, latent_dim]
        
        # Reshape for easier computation
        grad_G_inv = grad_G_inv.view(batch_size, latent_dim, latent_dim, latent_dim)
        
        # Compute Christoffel symbols: Γᵢⱼᵏ = ½G⁻¹ᵏˡ(∂ᵢG⁻¹ⱼˡ + ∂ⱼG⁻¹ᵢˡ - ∂ˡG⁻¹ᵢⱼ)
        christoffel = torch.zeros(batch_size, latent_dim, latent_dim, latent_dim, device=z.device)
        
        for i in range(latent_dim):
            for j in range(latent_dim):
                for k in range(latent_dim):
                    # Γᵢⱼᵏ = ½G⁻¹ᵏˡ(∂ᵢG⁻¹ⱼˡ + ∂ⱼG⁻¹ᵢˡ - ∂ˡG⁻¹ᵢⱼ)
                    for l in range(latent_dim):
                        term1 = grad_G_inv[:, i, j, l]  # ∂ᵢG⁻¹ⱼˡ
                        term2 = grad_G_inv[:, j, i, l]  # ∂ⱼG⁻¹ᵢˡ
                        term3 = grad_G_inv[:, l, i, j]  # ∂ˡG⁻¹ᵢⱼ
                        
                        christoffel[:, i, j, k] += 0.5 * G_inv[:, k, l] * (term1 + term2 - term3)
        
        return christoffel
    
    def geodesic_ode(self, t: float, state: np.ndarray, christoffel_fn) -> np.ndarray:
        """
        ODE for geodesic integration: d²z/dt² + Γᵢⱼᵏ dzᵢ/dt dzⱼ/dt = 0
        
        Args:
            t: time parameter
            state: [z, dz/dt] concatenated
            christoffel_fn: function to compute Christoffel symbols
            
        Returns:
            derivatives: [dz/dt, d²z/dt²]
        """
        latent_dim = len(state) // 2
        z = state[:latent_dim]
        dz_dt = state[latent_dim:]
        
        # Convert to tensor for computation
        z_tensor = torch.tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Compute Christoffel symbols
        christoffel = christoffel_fn(z_tensor)[0]  # Remove batch dimension
        
        # Compute d²z/dt² = -Γᵢⱼᵏ dzᵢ/dt dzⱼ/dt
        d2z_dt2 = torch.zeros(latent_dim, device=self.device)
        
        for k in range(latent_dim):
            for i in range(latent_dim):
                for j in range(latent_dim):
                    d2z_dt2[k] -= christoffel[i, j, k] * dz_dt[i] * dz_dt[j]
        
        # Return [dz/dt, d²z/dt²]
        return np.concatenate([dz_dt, d2z_dt2.cpu().numpy()])
    
    def compute_geodesic_path(self, start_point: torch.Tensor, end_point: torch.Tensor, 
                             n_steps: int = 100) -> torch.Tensor:
        """
        Compute geodesic path between two points.
        
        Args:
            start_point: [latent_dim] starting point
            end_point: [latent_dim] ending point
            n_steps: number of integration steps
            
        Returns:
            path: [n_steps, latent_dim] geodesic path
        """
        latent_dim = start_point.shape[0]
        
        # Initial velocity (simple linear approximation)
        initial_velocity = (end_point - start_point) / n_steps
        
        # Initial state: [z, dz/dt]
        initial_state = torch.cat([start_point, initial_velocity]).cpu().numpy()
        
        # Time span
        t_span = (0, n_steps)
        t_eval = np.linspace(0, n_steps, n_steps)
        
        # Create Christoffel function
        def christoffel_fn(z_tensor):
            return self.compute_christoffel_symbols(z_tensor)
        
        # Solve geodesic ODE
        solution = solve_ivp(
            fun=lambda t, state: self.geodesic_ode(t, state, christoffel_fn),
            t_span=t_span,
            y0=initial_state,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-8
        )
        
        # Extract position components
        path = torch.tensor(solution.y[:latent_dim, :].T, device=self.device)
        
        return path
    
    def sample_geodesic_paths(self, n_paths: int = 50, n_steps_per_path: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample multiple geodesic paths between random points.
        
        Args:
            n_paths: number of geodesic paths to sample
            n_steps_per_path: number of steps per path
            
        Returns:
            all_paths: [n_paths * n_steps_per_path, latent_dim] all path points
            path_metadata: [n_paths, 4] [start_idx, end_idx, start_point, end_point] for each path
        """
        print(f"🎯 Sampling {n_paths} geodesic paths")
        
        all_paths = []
        path_metadata = []
        
        # Sample random start and end points
        for i in range(n_paths):
            # Sample start point near a random centroid
            centroid_idx = torch.randint(0, len(self.metric_tensor.centroids), (1,)).item()
            start_centroid = self.metric_tensor.centroids[centroid_idx]
            start_point = start_centroid + torch.randn(2, device=self.device) * 0.2
            
            # Sample end point near a different centroid
            end_centroid_idx = torch.randint(0, len(self.metric_tensor.centroids), (1,)).item()
            while end_centroid_idx == centroid_idx:
                end_centroid_idx = torch.randint(0, len(self.metric_tensor.centroids), (1,)).item()
            
            end_centroid = self.metric_tensor.centroids[end_centroid_idx]
            end_point = end_centroid + torch.randn(2, device=self.device) * 0.2
            
            # Compute geodesic path
            try:
                path = self.compute_geodesic_path(start_point, end_point, n_steps_per_path)
                all_paths.append(path)
                
                # Store metadata
                start_idx = i * n_steps_per_path
                end_idx = (i + 1) * n_steps_per_path
                path_metadata.append([start_idx, end_idx, start_point, end_point])
                
                if (i + 1) % 10 == 0:
                    print(f"   Computed {i + 1}/{n_paths} geodesic paths")
                    
            except Exception as e:
                print(f"   Warning: Failed to compute geodesic path {i}: {e}")
                continue
        
        if all_paths:
            all_paths = torch.cat(all_paths, dim=0)
            path_metadata = torch.tensor(path_metadata, device=self.device)
        else:
            all_paths = torch.empty(0, 2, device=self.device)
            path_metadata = torch.empty(0, 4, device=self.device)
        
        print(f"✅ Generated {len(all_paths)} geodesic path points")
        return all_paths, path_metadata
    
    def sample_manifold_walks(self, n_walks: int = 100, walk_length: int = 50) -> torch.Tensor:
        """
        Sample manifold walks - sequences of geodesic steps.
        
        Args:
            n_walks: number of walks to sample
            walk_length: number of steps per walk
            
        Returns:
            walks: [n_walks * walk_length, latent_dim] all walk points
        """
        print(f"🎯 Sampling {n_walks} manifold walks")
        
        all_walks = []
        
        for i in range(n_walks):
            # Start at a random centroid
            start_centroid_idx = torch.randint(0, len(self.metric_tensor.centroids), (1,)).item()
            current_point = self.metric_tensor.centroids[start_centroid_idx] + torch.randn(2, device=self.device) * 0.1
            
            walk_points = [current_point]
            
            for step in range(walk_length - 1):
                # Choose next direction (random but respecting manifold structure)
                direction = torch.randn(2, device=self.device)
                direction = direction / torch.norm(direction)
                
                # Scale step size based on local metric
                with torch.no_grad():
                    G_inv, _ = self.metric_tensor(current_point.unsqueeze(0))
                    local_scale = torch.sqrt(torch.det(G_inv[0]))
                    step_size = 0.1 / (1.0 + local_scale * 0.01)  # Smaller steps in high-metric regions
                
                # Take geodesic step
                next_point = current_point + step_size * direction
                
                # Ensure we stay within reasonable bounds
                next_point = torch.clamp(next_point, -4, 4)
                
                walk_points.append(next_point)
                current_point = next_point
            
            walk_tensor = torch.stack(walk_points)
            all_walks.append(walk_tensor)
            
            if (i + 1) % 20 == 0:
                print(f"   Generated {i + 1}/{n_walks} manifold walks")
        
        all_walks = torch.cat(all_walks, dim=0)
        print(f"✅ Generated {len(all_walks)} manifold walk points")
        
        return all_walks


def run_geodesic_sampling_analysis():
    """Run comprehensive geodesic sampling analysis."""
    print("🚀 GEODESIC SAMPLING ANALYSIS")
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
    
    # Create geodesic sampler
    geodesic_sampler = GeodesicSampler(metric_tensor)
    
    # Sample geodesic paths
    print("🎯 Sampling geodesic paths")
    geodesic_paths, path_metadata = geodesic_sampler.sample_geodesic_paths(n_paths=30, n_steps_per_path=50)
    
    # Sample manifold walks
    print("🎯 Sampling manifold walks")
    manifold_walks = geodesic_sampler.sample_manifold_walks(n_walks=50, walk_length=30)
    
    # Create visualization
    print("🎨 Creating geodesic sampling visualization")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Convert to numpy
    centroids_np = centroids.detach().cpu().numpy()
    geodesic_paths_np = geodesic_paths.detach().cpu().numpy()
    manifold_walks_np = manifold_walks.detach().cpu().numpy()
    
    # Plot 1: Geodesic paths
    ax1 = axes[0, 0]
    ax1.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=150, 
               label='Centroids', zorder=5)
    
    # Plot each geodesic path with different colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(path_metadata)))
    for i, (start_idx, end_idx, start_point, end_point) in enumerate(path_metadata):
        path_points = geodesic_paths_np[start_idx:end_idx]
        ax1.plot(path_points[:, 0], path_points[:, 1], c=colors[i], alpha=0.7, linewidth=1)
    
    ax1.set_title("1. Geodesic Paths\n(Following Manifold Structure)", fontweight='bold')
    ax1.set_xlabel("z₁")
    ax1.set_ylabel("z₂")
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Manifold walks
    ax2 = axes[0, 1]
    ax2.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=150, 
               label='Centroids', zorder=5)
    ax2.scatter(manifold_walks_np[:, 0], manifold_walks_np[:, 1], c='blue', alpha=0.6, s=10)
    ax2.set_title("2. Manifold Walks\n(Flexible Exploration)", fontweight='bold')
    ax2.set_xlabel("z₁")
    ax2.set_ylabel("z₂")
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Combined sampling
    ax3 = axes[1, 0]
    ax3.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=150, 
               label='Centroids', zorder=5)
    ax3.scatter(geodesic_paths_np[:, 0], geodesic_paths_np[:, 1], c='green', alpha=0.4, s=8, label='Geodesic Points')
    ax3.scatter(manifold_walks_np[:, 0], manifold_walks_np[:, 1], c='blue', alpha=0.4, s=8, label='Walk Points')
    ax3.set_title("3. Combined Sampling\n(Geodesics + Walks)", fontweight='bold')
    ax3.set_xlabel("z₁")
    ax3.set_ylabel("z₂")
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Sampling density
    ax4 = axes[1, 1]
    all_points = np.vstack([geodesic_paths_np, manifold_walks_np])
    ax4.hist2d(all_points[:, 0], all_points[:, 1], bins=50, cmap='viridis')
    ax4.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', marker='*', s=150, 
               edgecolors='white', linewidth=1)
    ax4.set_title("4. Sampling Density\n(Heatmap)", fontweight='bold')
    ax4.set_xlabel("z₁")
    ax4.set_ylabel("z₂")
    ax4.set_xlim(-4, 4)
    ax4.set_ylim(-4, 4)
    
    plt.tight_layout()
    plt.savefig("geodesic_sampling_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Geodesic sampling analysis completed!")
    print(f"📊 Generated {len(geodesic_paths)} geodesic path points")
    print(f"📊 Generated {len(manifold_walks)} manifold walk points")
    
    return geodesic_paths, manifold_walks


if __name__ == "__main__":
    geodesic_paths, manifold_walks = run_geodesic_sampling_analysis()
    print(f"\n🎉 GEODESIC SAMPLING SYSTEM COMPLETE!") 