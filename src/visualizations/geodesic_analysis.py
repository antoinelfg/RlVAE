"""
Geodesic Analysis Visualizations
===============================

This module provides geodesic trajectory visualization for Riemannian VAE models.
It uses the geodesic_toolbox to compute and visualize geodesic paths on the learned
Riemannian manifold.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Import geodesic toolbox components
try:
    from geodesic_toolbox import GEORCE, get_mf_image, ShootingSolver
    GEODESIC_TOOLBOX_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Geodesic toolbox not available: {e}")
    GEODESIC_TOOLBOX_AVAILABLE = False

from .base import BaseVisualization
from .adapters import RLVAEGeodesicAdapter
from .adapters.unified_model_adapter import UnifiedModelAdapter


class GeodesicAnalysisVisualizations(BaseVisualization):
    """Geodesic analysis visualization suite for Riemannian VAE models."""
    
    def __init__(self, model, device, config, should_log_to_wandb=True):
        super().__init__(model, device, config, should_log_to_wandb)
        
        # Check if geodesic toolbox is available
        if not GEODESIC_TOOLBOX_AVAILABLE:
            print("⚠️ Geodesic toolbox not available. Geodesic visualizations will be skipped.")
            self.enabled = False
            return
        
        self.enabled = True
        self.adapter = None
        self.unified_adapter = None
        self.cometric = None
        self.last_geodesic_path = None
        
        # Configuration
        self.n_trajectories = getattr(config, 'geodesic_n_trajectories', 5)
        self.trajectory_resolution = getattr(config, 'geodesic_trajectory_resolution', 50)
        self.magnification_resolution = getattr(config, 'geodesic_magnification_resolution', 100)
        self.solver_type = getattr(config, 'geodesic_solver_type', 'georce')  # 'georce' or 'shooting'
        
        print(f"🌐 Initialized geodesic analysis with {self.n_trajectories} trajectories")
    
    def _initialize_adapter(self):
        """Initialize the geodesic adapter if not already done."""
        if self.adapter is None:
            print(f"🔧 Initializing unified and geodesic adapters for model: {type(self.model).__name__}")
            self.unified_adapter = UnifiedModelAdapter(self.model, self.device)
            self.adapter = RLVAEGeodesicAdapter(self.model, self.device)
            self.cometric = self.adapter.create_cometric()
            print("✅ Geodesic adapters initialized")
    
    def _create_geodesic_solver(self):
        """Create appropriate geodesic solver."""
        if self.solver_type.lower() == 'shooting':
            return ShootingSolver(cometric=self.cometric, verbose=False)
        else:
            # Default to GEORCE
            return GEORCE(cometric=self.cometric, pbar=False)
    
    def _sample_trajectory_endpoints(self, x_sample: torch.Tensor, n_trajectories: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample start and end points for geodesic trajectories.
        
        Args:
            x_sample: Input data sample
            n_trajectories: Number of trajectories to create
            
        Returns:
            Tuple of (start_points, end_points) tensors
        """
        self.model.eval()
        with torch.no_grad():
            # Encode sample to get latent points
            try:
                if hasattr(self.model, 'encode'):
                    # For modular models
                    encoder_output = self.model.encode(x_sample[:, 0])  # Use first timestep
                    if hasattr(encoder_output, 'embedding'):
                        latent_points = encoder_output.embedding
                    elif hasattr(encoder_output, 'mu'):
                        latent_points = encoder_output.mu
                    elif isinstance(encoder_output, dict) and 'mu' in encoder_output:
                        latent_points = encoder_output['mu']
                    else:
                        # Fallback: try to get any tensor-like output
                        if isinstance(encoder_output, torch.Tensor):
                            latent_points = encoder_output
                        else:
                            raise ValueError("Could not extract latent points from encoder output")
                else:
                    # For other model types
                    result = self.model(x_sample)
                    
                    # Try different ways to access latent points
                    if hasattr(result, 'mu'):
                        latent_points = result.mu
                    elif hasattr(result, 'z'):
                        latent_points = result.z
                    elif isinstance(result, dict):
                        # Check available keys and pick the most likely one
                        if 'mu' in result:
                            latent_points = result['mu']
                        elif 'z' in result:
                            latent_points = result['z']
                        elif 'latent_samples' in result:
                            latent_points = result['latent_samples']
                        else:
                            # Print available keys for debugging
                            print(f"⚠️ Available keys in model output: {list(result.keys())}")
                            raise KeyError(f"Could not find latent points in model output. Available keys: {list(result.keys())}")
                    else:
                        raise ValueError("Unknown model output format")
                        
            except Exception as e:
                print(f"⚠️ Failed to extract latent points from model: {e}")
                # Fallback: use adapter to sample points
                print("🔄 Using fallback: sampling points from adapter")
                latent_points = self.adapter.sample_latent_points(n_trajectories * 2)
                # Convert to match expected batch format
                if len(latent_points.shape) == 2:
                    latent_points = latent_points  # Already correct format
                else:
                    latent_points = latent_points.view(-1, latent_points.shape[-1])
        
        # Convert to double precision for geodesic computation
        latent_points = latent_points.to(dtype=torch.float64)
        
        # Sample pairs of points for trajectories
        n_available = min(len(latent_points), n_trajectories * 2)
        print(f"🔍 Available latent points: {n_available}, needed: {n_trajectories * 2}")
        print(f"🔍 Latent points shape: {latent_points.shape}")
        
        if n_available < 2:
            # Fallback: use adapter to sample points
            print("🔄 Using adapter fallback for trajectory endpoints")
            sampled_points = self.adapter.sample_latent_points(n_trajectories * 2)
            indices = torch.randperm(len(sampled_points))[:n_trajectories*2]
            start_points = sampled_points[indices[:n_trajectories]]
            end_points = sampled_points[indices[n_trajectories:]]
        else:
            # Ensure we don't exceed available points
            max_pairs = n_available // 2
            actual_trajectories = min(n_trajectories, max_pairs)
            print(f"🔍 Computing {actual_trajectories} trajectories from {n_available} points")
            
            indices = torch.randperm(n_available)[:actual_trajectories*2]
            selected_points = latent_points[indices]
            start_points = selected_points[:actual_trajectories]
            end_points = selected_points[actual_trajectories:]
        
        print(f"🔍 Final trajectory endpoints: start={start_points.shape}, end={end_points.shape}")
        return start_points, end_points
    
    def _compute_geodesic_trajectories(self, start_points: torch.Tensor, end_points: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute geodesic trajectories between point pairs.
        
        Args:
            start_points: Start points tensor (n_trajectories, latent_dim)
            end_points: End points tensor (n_trajectories, latent_dim)
            
        Returns:
            List of trajectory tensors
        """
        trajectories = []
        solver = self._create_geodesic_solver()
        
        print(f"🔍 Computing geodesics for {len(start_points)} trajectory pairs")
        print(f"🔍 Solver type: {type(solver).__name__}")
        
        try:
            for i in range(len(start_points)):
                q0 = start_points[i:i+1]  # Keep batch dimension
                q1 = end_points[i:i+1]
                
                print(f"🔍 Trajectory {i}: q0={q0.shape}, q1={q1.shape}")
                
                try:
                    # Compute geodesic trajectory
                    result = solver.get_trajectories(q0, q1)
                    
                    # Handle different return formats from geodesic solvers
                    if isinstance(result, tuple):
                        # Some solvers return (trajectory, additional_info)
                        trajectory = result[0]
                        print(f"🔍 Solver returned tuple, trajectory shape: {trajectory.shape}")
                    else:
                        # Direct trajectory return
                        trajectory = result
                        print(f"🔍 Solver returned direct trajectory, shape: {trajectory.shape}")
                    
                    # Ensure trajectory has the right shape and squeeze batch dimension
                    if len(trajectory.shape) == 3:  # (1, n_pts, dim)
                        trajectory = trajectory.squeeze(0)  # (n_pts, dim)
                    elif len(trajectory.shape) == 2:  # (n_pts, dim) - already correct
                        pass
                    else:
                        raise ValueError(f"Unexpected trajectory shape: {trajectory.shape}")
                    
                    print(f"✅ Geodesic {i} computed successfully, final shape: {trajectory.shape}")
                    trajectories.append(trajectory)
                    
                except Exception as e:
                    print(f"⚠️ Failed to compute geodesic {i}: {e}")
                    print(f"🔄 Using linear interpolation fallback")
                    # Fallback: linear interpolation
                    try:
                        t = torch.linspace(0, 1, self.trajectory_resolution, device=q0.device, dtype=q0.dtype)
                        q0_flat = q0.squeeze(0)  # Remove batch dimension
                        q1_flat = q1.squeeze(0)  # Remove batch dimension
                        linear_traj = q0_flat.unsqueeze(0) + t.unsqueeze(-1) * (q1_flat.unsqueeze(0) - q0_flat.unsqueeze(0))
                        linear_traj = linear_traj.squeeze(0)  # Remove extra dimension
                        print(f"✅ Linear fallback {i} shape: {linear_traj.shape}")
                        trajectories.append(linear_traj)
                    except Exception as fallback_e:
                        print(f"⚠️ Even linear fallback failed for trajectory {i}: {fallback_e}")
                        continue

        except Exception as e:
            print(f"⚠️ Geodesic computation failed: {e}")
            import traceback
            traceback.print_exc()

        return trajectories
    
    def _compute_magnification_factor(self) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        Compute magnification factor heatmap for the manifold.
        
        Returns:
            Tuple of (magnification_image, bounds)
        """
        try:
            # Get bounds from adapter
            bounds = self.adapter.get_latent_bounds(margin=0.5)
            
            # Compute magnification factor image
            mf_image = get_mf_image(
                self.cometric, 
                self.adapter._centroids,
                bounds=bounds,
                resolution=self.magnification_resolution
            )
            
            return mf_image.detach().cpu().numpy(), bounds
            
        except Exception as e:
            print(f"⚠️ Failed to compute magnification factor: {e}")
            # Return dummy image
            bounds = (-3, 3, -3, 3)
            dummy_image = np.ones((self.magnification_resolution, self.magnification_resolution))
            return dummy_image, bounds
    
    def _plot_geodesics_on_manifold(self, trajectories: List[torch.Tensor], 
                                   start_points: torch.Tensor, end_points: torch.Tensor,
                                   epoch: int) -> plt.Figure:
        """
        Create comprehensive geodesic visualization plot.
        
        Args:
            trajectories: List of computed geodesic trajectories
            start_points: Start points for trajectories
            end_points: End points for trajectories
            epoch: Current training epoch
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Geodesics on magnification factor background
        try:
            mf_image, bounds = self._compute_magnification_factor()
            
            # Show magnification factor as background
            im1 = ax1.imshow(
                np.log(mf_image + 1e-8), 
                extent=bounds, 
                origin='lower', 
                cmap='viridis', 
                alpha=0.7,
                interpolation='bilinear'
            )
            plt.colorbar(im1, ax=ax1, label='Log Magnification Factor')
            
        except Exception as e:
            print(f"⚠️ Failed to plot magnification factor: {e}")
        
        # Plot geodesic trajectories
        colors = plt.cm.Set1(np.linspace(0, 1, len(trajectories)))
        
        for i, (traj, color) in enumerate(zip(trajectories, colors)):
            if len(traj) > 0:
                traj_np = traj.detach().cpu().numpy()
                ax1.plot(traj_np[:, 0], traj_np[:, 1], 
                        color=color, linewidth=2, alpha=0.8, label=f'Geodesic {i+1}')
                
                # Mark start and end points
                ax1.scatter(traj_np[0, 0], traj_np[0, 1], 
                           color=color, s=100, marker='o', edgecolor='white', linewidth=2)
                ax1.scatter(traj_np[-1, 0], traj_np[-1, 1], 
                           color=color, s=100, marker='s', edgecolor='white', linewidth=2)
        
        ax1.set_title(f'Geodesic Trajectories on Learned Manifold (Epoch {epoch})')
        ax1.set_xlabel('Latent Dimension 1')
        ax1.set_ylabel('Latent Dimension 2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Trajectory comparison (geodesic vs straight line)
        for i, (traj, color) in enumerate(zip(trajectories, colors)):
            if len(traj) > 0:
                traj_np = traj.detach().cpu().numpy()
                
                # Geodesic trajectory
                ax2.plot(traj_np[:, 0], traj_np[:, 1], 
                        color=color, linewidth=2, alpha=0.8, label=f'Geodesic {i+1}')
                
                # Straight line for comparison
                start_np = start_points[i].detach().cpu().numpy()
                end_np = end_points[i].detach().cpu().numpy()
                ax2.plot([start_np[0], end_np[0]], [start_np[1], end_np[1]], 
                        color=color, linestyle='--', alpha=0.5, linewidth=1)
                
                # Mark points
                ax2.scatter(start_np[0], start_np[1], 
                           color=color, s=100, marker='o', edgecolor='white', linewidth=2)
                ax2.scatter(end_np[0], end_np[1], 
                           color=color, s=100, marker='s', edgecolor='white', linewidth=2)
        
        ax2.set_title(f'Geodesics vs Straight Lines (Epoch {epoch})')
        ax2.set_xlabel('Latent Dimension 1')
        ax2.set_ylabel('Latent Dimension 2')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _compute_geodesic_metrics(self, trajectories: List[torch.Tensor], 
                                 start_points: torch.Tensor, end_points: torch.Tensor) -> Dict[str, float]:
        """
        Compute quantitative metrics about the geodesics.
        
        Args:
            trajectories: Computed geodesic trajectories
            start_points: Start points
            end_points: End points
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        try:
            geodesic_lengths = []
            euclidean_lengths = []
            
            for i, traj in enumerate(trajectories):
                if len(traj) > 1:
                    # Geodesic length (approximate)
                    traj_np = traj.detach().cpu().numpy()
                    diffs = np.diff(traj_np, axis=0)
                    geodesic_length = np.sum(np.linalg.norm(diffs, axis=1))
                    geodesic_lengths.append(geodesic_length)
                    
                    # Euclidean length
                    start_np = start_points[i].detach().cpu().numpy()
                    end_np = end_points[i].detach().cpu().numpy()
                    euclidean_length = np.linalg.norm(end_np - start_np)
                    euclidean_lengths.append(euclidean_length)
            
            if geodesic_lengths:
                metrics['mean_geodesic_length'] = float(np.mean(geodesic_lengths))
                metrics['mean_euclidean_length'] = float(np.mean(euclidean_lengths))
                metrics['mean_length_ratio'] = float(np.mean(np.array(geodesic_lengths) / np.array(euclidean_lengths)))
                metrics['n_successful_trajectories'] = len(geodesic_lengths)
            
        except Exception as e:
            print(f"⚠️ Failed to compute geodesic metrics: {e}")
            
        return metrics
    
    def create_geodesic_trajectories(self, x_sample: torch.Tensor, epoch: int):
        """
        Main method to create geodesic trajectory visualizations.
        
        Args:
            x_sample: Input data sample
            epoch: Current training epoch
        """
        if not self.enabled:
            return
            
        print(f"🌐 Creating geodesic trajectory visualization for epoch {epoch}")
        print(f"🔍 Input sample shape: {x_sample.shape}")
        print(f"🔍 Model type: {type(self.model).__name__}")
        
        try:
            # Initialize adapter if needed
            self._initialize_adapter()
            
            # Sample trajectory endpoints
            start_points, end_points = self._sample_trajectory_endpoints(x_sample, self.n_trajectories)
            
            # Compute geodesic trajectories
            trajectories = self._compute_geodesic_trajectories(start_points, end_points)
            
            if not trajectories:
                print("⚠️ No geodesic trajectories computed")
                return
            
            # Create visualization
            fig = self._plot_geodesics_on_manifold(trajectories, start_points, end_points, epoch)
            
            # Save figure
            filename = f"geodesic_trajectories_epoch_{epoch:03d}.png"
            saved_file = self._safe_save_plt_figure(filename, dpi=300, bbox_inches='tight')
            self.last_geodesic_path = saved_file
            
            # Log to WandB
            if self.should_log_to_wandb() and saved_file:
                # Compute metrics
                metrics = self._compute_geodesic_metrics(trajectories, start_points, end_points)
                
                # Log visualization
                wandb.log({
                    "geodesic_analysis/trajectories": wandb.Image(saved_file, caption=f"Geodesic Trajectories - Epoch {epoch}"),
                })
                
                # Log metrics
                for key, value in metrics.items():
                    wandb.log({f"geodesic_analysis/metrics/{key}": value})
                
                print(f"✅ Logged geodesic visualization to WandB with {len(trajectories)} trajectories")
            
            plt.close(fig)
            
        except Exception as e:
            print(f"⚠️ Geodesic visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.model.train()
    
    def should_visualize(self, epoch: int, frequency: int = 20) -> bool:
        """
        Determine if geodesic visualization should be created for this epoch.
        
        Args:
            epoch: Current epoch
            frequency: Visualization frequency
            
        Returns:
            True if visualization should be created
        """
        return self.enabled and (epoch == 0 or epoch % frequency == 0)
