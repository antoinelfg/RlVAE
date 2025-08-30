#!/usr/bin/env python3
"""
Manifold Sampler Component
=========================

Modular manifold sampling component for RlVAE that provides relaxed 
manifold-guided sampling with G⁻¹ metric awareness.

This component integrates seamlessly with the RlVAE pipeline and provides
multiple sampling strategies with configurable parameters.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional, Union, List
from enum import Enum
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from .metric_tensor import MetricTensor


class ManifoldSamplingMethod(Enum):
    """Available manifold sampling methods."""
    RELAXED_GUIDED = "relaxed_guided"
    RELAXED_EXPLORATION = "relaxed_exploration"  
    RELAXED_CONNECTIONS = "relaxed_connections"
    COMBINED = "combined"


class ManifoldSampler(nn.Module):
    """
    Modular manifold sampler that provides relaxed manifold-guided sampling.
    
    This component can work with both traditional MetricTensor and 
    NativeInverseMetricTensor, automatically detecting and adapting to the metric type.
    
    Key Features:
    - Multiple sampling strategies with configurable parameters
    - Native G⁻¹ support for improved geometric fidelity
    - Integration with RlVAE training pipeline
    - WandB logging support for evolution tracking
    - Configurable via Hydra
    """
    
    def __init__(
        self,
        metric_tensor: Union[MetricTensor, nn.Module],
        method: str = "combined",
        step_size_base: float = 0.25,
        exploration_ratio: float = 0.6,
        direction_change_frequency: int = 3,
        random_component_scale: float = 0.1,
        bounds: Tuple[float, float] = (-5.0, 5.0),
        device: Optional[torch.device] = None
    ):
        """
        Initialize the manifold sampler.
        
        Args:
            metric_tensor: Metric tensor component (MetricTensor or NativeInverseMetricTensor)
            method: Sampling method (relaxed_guided, relaxed_exploration, relaxed_connections, combined)
            step_size_base: Base step size for sampling
            exploration_ratio: Ratio of metric-guided vs random steps (0.0-1.0)
            direction_change_frequency: How often to change directions (in steps)
            random_component_scale: Scale of random component in steps
            bounds: Sampling bounds (min, max)
            device: Device for computations
        """
        super().__init__()
        
        self.metric_tensor = metric_tensor
        self.method = ManifoldSamplingMethod(method)
        self.step_size_base = step_size_base
        self.exploration_ratio = exploration_ratio
        self.direction_change_frequency = direction_change_frequency
        self.random_component_scale = random_component_scale
        self.bounds = bounds
        self.device = device or (
            metric_tensor.centroids.device if (hasattr(metric_tensor, 'centroids') and metric_tensor.centroids is not None) 
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Detect metric tensor type - check for native inverse methods
        self.is_native_inverse = hasattr(metric_tensor, 'inverse_metrics') or 'NativeInverse' in str(type(metric_tensor))
        
        print(f"🎯 ManifoldSampler initialized:")
        print(f"   Method: {self.method.value}")
        print(f"   Metric type: {'Native G⁻¹' if self.is_native_inverse else 'Traditional G'}")
        print(f"   Step size: {self.step_size_base}")
        print(f"   Exploration ratio: {self.exploration_ratio}")
        print(f"   Device: {self.device}")
    
    def _get_metric_inverse_and_det(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get G⁻¹ and its determinant, handling both metric types.
        
        Args:
            z: Input points [batch_size, latent_dim]
            
        Returns:
            G_inv: Inverse metric [batch_size, latent_dim, latent_dim]
            det_G_inv: Determinant of G⁻¹ [batch_size]
        """
        if self.is_native_inverse:
            # Native G⁻¹ implementation
            G_inv, log_det_G_inv = self.metric_tensor(z)
            det_G_inv = torch.exp(log_det_G_inv)
        else:
            # Traditional G implementation - compute inverse
            with torch.no_grad():
                if hasattr(self.metric_tensor, 'compute_metric'):
                    G = self.metric_tensor.compute_metric(z)
                elif hasattr(self.metric_tensor, '__call__'):
                    G = self.metric_tensor(z)
                else:
                    # Fallback - assume it's a model with G method
                    G = self.metric_tensor.G(z) if hasattr(self.metric_tensor, 'G') else torch.eye(2, device=z.device).unsqueeze(0).repeat(z.shape[0], 1, 1)
                
                G_inv = torch.linalg.inv(G)
                det_G_inv = torch.linalg.det(G_inv)
        
        return G_inv, det_G_inv
    
    def _compute_relaxed_metric_step(
        self, 
        current_point: torch.Tensor, 
        direction: torch.Tensor, 
        step_size: float = None
    ) -> torch.Tensor:
        """
        Compute a relaxed step guided by the local metric.
        
        Args:
            current_point: [latent_dim] current position
            direction: [latent_dim] desired direction
            step_size: step size parameter (defaults to self.step_size_base)
            
        Returns:
            next_point: [latent_dim] new position
        """
        if step_size is None:
            step_size = self.step_size_base
            
        with torch.no_grad():
            # Get local metric inverse and determinant
            G_inv, det_G_inv = self._get_metric_inverse_and_det(current_point.unsqueeze(0))
            G_inv = G_inv[0]  # Remove batch dimension
            det_G_inv = det_G_inv[0]
            
            # Normalize direction
            direction = direction / torch.norm(direction)
            
            # Transform direction by G⁻¹ (metric-aware step)
            metric_direction = torch.mv(G_inv, direction)
            metric_direction = metric_direction / torch.norm(metric_direction)
            
            # Scale step size based on local metric determinant (relaxed)
            local_step_size = step_size / (1.0 + det_G_inv * 0.0001)
            
            # Add moderate randomness for balanced guided paths
            random_component = torch.randn(2, device=self.device) * (self.random_component_scale * 0.7)
            final_direction = metric_direction + random_component
            final_direction = final_direction / torch.norm(final_direction)
            
            # Take step
            next_point = current_point + local_step_size * final_direction
            
            return next_point
    
    def sample_relaxed_guided_paths(
        self, 
        n_paths: int = 40, 
        path_length: int = 30,
        centroids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample relaxed paths guided by the metric with BALANCED directional consistency.
        
        Args:
            n_paths: Number of paths to sample
            path_length: Number of steps per path
            centroids: Optional centroids to start from (auto-detected if None)
            
        Returns:
            all_paths: [n_paths * path_length, latent_dim] all path points
        """
        if centroids is None:
            if hasattr(self.metric_tensor, 'centroids') and self.metric_tensor.centroids is not None:
                centroids = self.metric_tensor.centroids
            elif hasattr(self.metric_tensor, 'centroids_tens') and self.metric_tensor.centroids_tens is not None:
                centroids = self.metric_tensor.centroids_tens
            else:
                # Generate default centroids
                centroids = torch.randn(20, 2, device=self.device) * 2.0
        
        all_paths = []
        
        for i in range(n_paths):
            # Start near a random centroid with reasonable variance
            centroid_idx = torch.randint(0, len(centroids), (1,)).item()
            start_point = centroids[centroid_idx] + torch.randn(2, device=self.device) * 0.3
            
            path_points = [start_point]
            current_point = start_point
            
            # Initialize with a random direction
            current_direction = torch.randn(2, device=self.device)
            current_direction = current_direction / torch.norm(current_direction)
            
            for step in range(path_length - 1):
                # Balanced direction evolution: maintain direction but allow changes
                if step % self.direction_change_frequency == 0:
                    # Mix current direction with new random direction (not complete change)
                    new_direction = torch.randn(2, device=self.device)
                    new_direction = new_direction / torch.norm(new_direction)
                    # 60% old direction, 40% new direction (balanced)
                    current_direction = 0.6 * current_direction + 0.4 * new_direction
                else:
                    # Add moderate perturbation to maintain some variability
                    perturbation = torch.randn(2, device=self.device) * 0.2
                    current_direction = current_direction + perturbation
                
                # Normalize direction
                current_direction = current_direction / torch.norm(current_direction)
                
                # Take relaxed metric-guided step with proper step size
                next_point = self._compute_relaxed_metric_step(current_point, current_direction, step_size=self.step_size_base * 1.2)
                
                # Apply bounds
                next_point = torch.clamp(next_point, self.bounds[0], self.bounds[1])
                
                path_points.append(next_point)
                current_point = next_point
            
            path_tensor = torch.stack(path_points)
            all_paths.append(path_tensor)
        
        all_paths = torch.cat(all_paths, dim=0)
        return all_paths
    
    def sample_relaxed_explorations(
        self, 
        n_explorations: int = 60, 
        exploration_length: int = 35,
        centroids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample relaxed manifold exploration with controlled randomness.
        
        Args:
            n_explorations: Number of exploration sequences
            exploration_length: Number of steps per exploration
            centroids: Optional centroids to start from
            
        Returns:
            all_explorations: [n_explorations * exploration_length, latent_dim] all exploration points
        """
        if centroids is None:
            if hasattr(self.metric_tensor, 'centroids') and self.metric_tensor.centroids is not None:
                centroids = self.metric_tensor.centroids
            elif hasattr(self.metric_tensor, 'centroids_tens') and self.metric_tensor.centroids_tens is not None:
                centroids = self.metric_tensor.centroids_tens
            else:
                centroids = torch.randn(20, 2, device=self.device) * 2.0
        
        all_explorations = []
        
        for i in range(n_explorations):
            centroid_idx = torch.randint(0, len(centroids), (1,)).item()
            current_point = centroids[centroid_idx] + torch.randn(2, device=self.device) * 0.2
            
            exploration_points = [current_point]
            
            for step in range(exploration_length - 1):
                # Mix of random exploration and metric guidance
                if torch.rand(1).item() < self.exploration_ratio:
                    # Metric-guided step
                    direction = torch.randn(2, device=self.device)
                    next_point = self._compute_relaxed_metric_step(current_point, direction, step_size=0.2)
                else:
                    # Pure random step (exploration)
                    direction = torch.randn(2, device=self.device)
                    direction = direction / torch.norm(direction)
                    step_size = 0.2 / (1.0 + torch.rand(1).item() * 1.5)
                    next_point = current_point + step_size * direction
                
                # Apply bounds
                next_point = torch.clamp(next_point, self.bounds[0], self.bounds[1])
                exploration_points.append(next_point)
                current_point = next_point
            
            exploration_tensor = torch.stack(exploration_points)
            all_explorations.append(exploration_tensor)
        
        all_explorations = torch.cat(all_explorations, dim=0)
        return all_explorations
    
    def sample_relaxed_connections(
        self, 
        n_connections: int = 25, 
        steps_per_connection: int = 25,
        centroids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample relaxed paths connecting different centroids.
        
        Args:
            n_connections: Number of centroid connections
            steps_per_connection: Number of steps per connection
            centroids: Optional centroids to connect
            
        Returns:
            all_connections: [n_connections * steps_per_connection, latent_dim] all connection points
        """
        if centroids is None:
            if hasattr(self.metric_tensor, 'centroids') and self.metric_tensor.centroids is not None:
                centroids = self.metric_tensor.centroids
            elif hasattr(self.metric_tensor, 'centroids_tens') and self.metric_tensor.centroids_tens is not None:
                centroids = self.metric_tensor.centroids_tens
            else:
                centroids = torch.randn(20, 2, device=self.device) * 2.0
        
        all_connections = []
        
        for i in range(n_connections):
            # Choose two different centroids
            start_centroid_idx = torch.randint(0, len(centroids), (1,)).item()
            end_centroid_idx = torch.randint(0, len(centroids), (1,)).item()
            while end_centroid_idx == start_centroid_idx:
                end_centroid_idx = torch.randint(0, len(centroids), (1,)).item()
            
            start_point = centroids[start_centroid_idx] + torch.randn(2, device=self.device) * 0.2
            end_point = centroids[end_centroid_idx] + torch.randn(2, device=self.device) * 0.2
            
            connection_points = [start_point]
            current_point = start_point
            
            for step in range(steps_per_connection - 1):
                # Direction towards end point with metric guidance
                target_direction = end_point - current_point
                target_direction = target_direction / torch.norm(target_direction)
                
                # Add randomness to avoid straight lines
                random_component = torch.randn(2, device=self.device) * 0.4
                direction = target_direction + random_component
                
                # Take relaxed metric-guided step
                next_point = self._compute_relaxed_metric_step(current_point, direction, step_size=0.18)
                
                # Apply bounds
                next_point = torch.clamp(next_point, self.bounds[0], self.bounds[1])
                
                connection_points.append(next_point)
                current_point = next_point
            
            connection_tensor = torch.stack(connection_points)
            all_connections.append(connection_tensor)
        
        all_connections = torch.cat(all_connections, dim=0)
        return all_connections
    
    def sample(
        self, 
        method: Optional[str] = None,
        n_samples: int = 100,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Sample using the specified method or the configured default method.
        
        Args:
            method: Sampling method to use (overrides self.method if provided)
            n_samples: Total number of samples to generate
            **kwargs: Additional arguments for specific sampling methods
            
        Returns:
            Dict with sampled points for each strategy
        """
        sampling_method = ManifoldSamplingMethod(method) if method else self.method
        
        results = {}
        
        if sampling_method == ManifoldSamplingMethod.RELAXED_GUIDED:
            results['guided_paths'] = self.sample_relaxed_guided_paths(
                n_paths=kwargs.get('n_paths', n_samples // 30),
                path_length=kwargs.get('path_length', 30),
                centroids=kwargs.get('centroids')
            )
        elif sampling_method == ManifoldSamplingMethod.RELAXED_EXPLORATION:
            results['explorations'] = self.sample_relaxed_explorations(
                n_explorations=kwargs.get('n_explorations', n_samples // 35),
                exploration_length=kwargs.get('exploration_length', 35),
                centroids=kwargs.get('centroids')
            )
        elif sampling_method == ManifoldSamplingMethod.RELAXED_CONNECTIONS:
            results['connections'] = self.sample_relaxed_connections(
                n_connections=kwargs.get('n_connections', n_samples // 25),
                steps_per_connection=kwargs.get('steps_per_connection', 25),
                centroids=kwargs.get('centroids')
            )
        elif sampling_method == ManifoldSamplingMethod.COMBINED:
            # Sample all strategies with balanced distribution
            results['guided_paths'] = self.sample_relaxed_guided_paths(
                n_paths=kwargs.get('n_paths', 20),
                path_length=kwargs.get('path_length', 30),
                centroids=kwargs.get('centroids')
            )
            results['explorations'] = self.sample_relaxed_explorations(
                n_explorations=kwargs.get('n_explorations', 30),
                exploration_length=kwargs.get('exploration_length', 35),
                centroids=kwargs.get('centroids')
            )
            results['connections'] = self.sample_relaxed_connections(
                n_connections=kwargs.get('n_connections', 15),
                steps_per_connection=kwargs.get('steps_per_connection', 25),
                centroids=kwargs.get('centroids')
            )
        
        return results
    
    def compute_determinant_grid(
        self, 
        x_range: Tuple[float, float] = None, 
        y_range: Tuple[float, float] = None, 
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute metric determinant across a grid for visualization.
        
        Args:
            x_range: X-axis range (defaults to self.bounds)
            y_range: Y-axis range (defaults to self.bounds)
            n_points: Grid resolution
            
        Returns:
            X, Y, det_grid: Meshgrid coordinates and determinant values
        """
        if x_range is None:
            x_range = self.bounds
        if y_range is None:
            y_range = self.bounds
        
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)
        
        # Flatten grid for batch processing
        grid_points = torch.tensor(
            np.column_stack([X.ravel(), Y.ravel()]), 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Compute determinant
        with torch.no_grad():
            _, det_values = self._get_metric_inverse_and_det(grid_points)
            det_grid = det_values.cpu().numpy().reshape(X.shape)
        
        return X, Y, det_grid
    
    def create_visualization(
        self, 
        samples: Dict[str, torch.Tensor],
        latent_data: Optional[torch.Tensor] = None,
        centroids: Optional[torch.Tensor] = None,
        title: str = "Manifold Sampling Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive visualization of manifold sampling.
        
        Args:
            samples: Dictionary of sampled points
            latent_data: Optional background data points
            centroids: Optional centroids to display
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        # Compute determinant grid
        X, Y, det_grid = self.compute_determinant_grid()
        
        # Auto-detect centroids if not provided
        if centroids is None:
            if hasattr(self.metric_tensor, 'centroids'):
                centroids = self.metric_tensor.centroids.detach().cpu().numpy()
            elif hasattr(self.metric_tensor, 'centroids_tens'):
                centroids = self.metric_tensor.centroids_tens.detach().cpu().numpy()
        elif centroids is not None:
            centroids = centroids.detach().cpu().numpy()
        
        # Prepare sample data
        guided_paths = samples.get('guided_paths')
        explorations = samples.get('explorations') 
        connections = samples.get('connections')
        
        # Convert to numpy
        if guided_paths is not None:
            guided_paths = guided_paths.detach().cpu().numpy()
        if explorations is not None:
            explorations = explorations.detach().cpu().numpy()
        if connections is not None:
            connections = connections.detach().cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Plot 1: Guided paths with determinant
        ax1 = axes[0, 0]
        contour1 = ax1.contourf(X, Y, det_grid, levels=20, cmap='viridis', alpha=0.7)
        if latent_data is not None:
            latent_np = latent_data.detach().cpu().numpy()
            ax1.scatter(latent_np[:, 0], latent_np[:, 1], c='lightblue', alpha=0.3, s=10, label='Data')
        if centroids is not None:
            ax1.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=150, label='Centroids', zorder=5)
        if guided_paths is not None:
            ax1.scatter(guided_paths[:, 0], guided_paths[:, 1], c='green', alpha=0.4, s=8, label='Guided Paths')
        ax1.set_title("Relaxed Guided Paths", fontweight='bold')
        ax1.set_xlabel("z₁")
        ax1.set_ylabel("z₂")
        ax1.legend()
        plt.colorbar(contour1, ax=ax1, label='det(G⁻¹)')
        
        # Plot 2: Explorations with determinant
        ax2 = axes[0, 1]
        contour2 = ax2.contourf(X, Y, det_grid, levels=20, cmap='viridis', alpha=0.7)
        if latent_data is not None:
            ax2.scatter(latent_np[:, 0], latent_np[:, 1], c='lightblue', alpha=0.3, s=10, label='Data')
        if centroids is not None:
            ax2.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=150, label='Centroids', zorder=5)
        if explorations is not None:
            ax2.scatter(explorations[:, 0], explorations[:, 1], c='blue', alpha=0.4, s=8, label='Explorations')
        ax2.set_title("Relaxed Explorations", fontweight='bold')
        ax2.set_xlabel("z₁")
        ax2.set_ylabel("z₂")
        ax2.legend()
        plt.colorbar(contour2, ax=ax2, label='det(G⁻¹)')
        
        # Plot 3: Connections with determinant
        ax3 = axes[0, 2]
        contour3 = ax3.contourf(X, Y, det_grid, levels=20, cmap='viridis', alpha=0.7)
        if latent_data is not None:
            ax3.scatter(latent_np[:, 0], latent_np[:, 1], c='lightblue', alpha=0.3, s=10, label='Data')
        if centroids is not None:
            ax3.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=150, label='Centroids', zorder=5)
        if connections is not None:
            ax3.scatter(connections[:, 0], connections[:, 1], c='purple', alpha=0.4, s=8, label='Connections')
        ax3.set_title("Relaxed Connections", fontweight='bold')
        ax3.set_xlabel("z₁")
        ax3.set_ylabel("z₂")
        ax3.legend()
        plt.colorbar(contour3, ax=ax3, label='det(G⁻¹)')
        
        # Plot 4: Determinant level lines
        ax4 = axes[1, 0]
        contour4 = ax4.contour(X, Y, det_grid, levels=15, colors='black', alpha=0.6, linewidths=0.8)
        if centroids is not None:
            ax4.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=150, label='Centroids', zorder=5)
        all_samples = []
        if guided_paths is not None:
            all_samples.append(guided_paths)
            ax4.scatter(guided_paths[:, 0], guided_paths[:, 1], c='green', alpha=0.3, s=6, label='Guided')
        if explorations is not None:
            all_samples.append(explorations)
            ax4.scatter(explorations[:, 0], explorations[:, 1], c='blue', alpha=0.3, s=6, label='Exploration')
        if connections is not None:
            all_samples.append(connections)
            ax4.scatter(connections[:, 0], connections[:, 1], c='purple', alpha=0.3, s=6, label='Connections')
        ax4.set_title("Determinant Level Lines", fontweight='bold')
        ax4.set_xlabel("z₁")
        ax4.set_ylabel("z₂")
        ax4.legend()
        
        # Plot 5: Sampling density vs determinant
        ax5 = axes[1, 1]
        contour5 = ax5.contourf(X, Y, det_grid, levels=20, cmap='viridis', alpha=0.7)
        if all_samples:
            combined_samples = np.vstack(all_samples)
            ax5.hist2d(combined_samples[:, 0], combined_samples[:, 1], bins=50, cmap='Reds', alpha=0.8)
        if centroids is not None:
            ax5.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=150, 
                       edgecolors='white', linewidth=1, zorder=5)
        ax5.set_title("Sampling Density vs det(G⁻¹)", fontweight='bold')
        ax5.set_xlabel("z₁")
        ax5.set_ylabel("z₂")
        plt.colorbar(contour5, ax=ax5, label='det(G⁻¹)')
        
        # Plot 6: Combined view
        ax6 = axes[1, 2]
        contour6 = ax6.contourf(X, Y, det_grid, levels=20, cmap='viridis', alpha=0.6)
        if latent_data is not None:
            ax6.scatter(latent_np[:, 0], latent_np[:, 1], c='lightblue', alpha=0.3, s=10, label='Data')
        if centroids is not None:
            ax6.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=150, label='Centroids', zorder=5)
        if guided_paths is not None:
            ax6.scatter(guided_paths[:, 0], guided_paths[:, 1], c='green', alpha=0.3, s=6, label='Guided')
        if explorations is not None:
            ax6.scatter(explorations[:, 0], explorations[:, 1], c='blue', alpha=0.3, s=6, label='Exploration')
        if connections is not None:
            ax6.scatter(connections[:, 0], connections[:, 1], c='purple', alpha=0.3, s=6, label='Connections')
        ax6.set_title("Combined Manifold Sampling", fontweight='bold')
        ax6.set_xlabel("z₁")
        ax6.set_ylabel("z₂")
        ax6.legend()
        plt.colorbar(contour6, ax=ax6, label='det(G⁻¹)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig