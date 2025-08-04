"""
Manifold Evolution Visualizations
=================================

Specialized visualizations for tracking manifold evolution during adaptive centroid training.
Shows how the learned manifold changes over time with periodic centroid updates.
"""

import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wandb
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from .base import BaseVisualization

# Import RHVAE sampler with proper path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "lib" / "src"))
from pythae.samplers.manifold_sampler import RHVAESampler, RHVAESamplerConfig

logger = logging.getLogger(__name__)

class ManifoldEvolutionVisualizations(BaseVisualization):
    """Visualizations for manifold evolution during adaptive training."""
    
    def __init__(self, model, device, config, should_log_to_wandb=True):
        """Initialize manifold evolution visualizations."""
        super().__init__(model, device, config, should_log_to_wandb)
        
        # Track manifold evolution state
        self.centroid_snapshots = []
        self.metric_snapshots = []
        self.evolution_epochs = []
        self.manifold_metrics_history = {
            'centroid_shifts': [],
            'metric_changes': [],
            'coverage_metrics': [],
            'latent_variances': []
        }
        
        # Store original state
        if hasattr(self.model, 'modular_metric') and hasattr(self.model.modular_metric, 'centroids'):
            self.original_centroids = self.model.modular_metric.centroids.clone().detach().cpu().numpy()
            self.original_metric_matrices = self.model.modular_metric.metric_matrices.clone().detach().cpu().numpy()
        elif hasattr(self.model, 'centroids_tens'):
            self.original_centroids = self.model.centroids_tens.clone().detach().cpu().numpy()
            self.original_metric_matrices = self.model.M_tens.clone().detach().cpu().numpy()
        
        # Initialize RHVAE sampler for evolved manifold sampling
        self.rhvae_sampler = None
        self._initialize_rhvae_sampler()
        
        logger.info("🌊 Manifold evolution visualizations initialized")
    
    def _initialize_rhvae_sampler(self):
        """Initialize RHVAE sampler for evolved manifold sampling."""
        try:
            # Create RHVAE-compatible wrapper for the model
            class RHVAECompatibleWrapper:
                def __init__(self, original_model, device):
                    self.original_model = original_model
                    self.device = device
                    
                    # Copy essential attributes from original model
                    if hasattr(original_model, 'centroids_tens'):
                        self.centroids_tens = original_model.centroids_tens.clone()
                        self.M_tens = original_model.M_tens.clone()
                    elif hasattr(original_model, 'modular_metric'):
                        self.centroids_tens = original_model.modular_metric.centroids.clone()
                        self.M_tens = original_model.modular_metric.metric_matrices.clone()
                    
                    # Essential RHVAE attributes
                    self.latent_dim = getattr(original_model, 'latent_dim', 2)
                    self.temperature = getattr(original_model, 'temperature', 1.0)
                    self.lbd = getattr(original_model, 'lbd', 1e-6)
                    
                    # Create metric functions
                    self._create_metric_functions()
                    
                    # Add decoder reference
                    self.decoder = original_model.decoder if hasattr(original_model, 'decoder') else original_model
                
                def eval(self):
                    """Set to evaluation mode (required by RHVAE sampler)."""
                    return self
                
                def to(self, device):
                    """Move model to device (required by RHVAE sampler)."""
                    self.centroids_tens = self.centroids_tens.to(device)
                    self.M_tens = self.M_tens.to(device)
                    return self
                
                def _create_metric_functions(self):
                    """Create G and G_inv functions for RHVAE sampler."""
                    def G_inv(z: torch.Tensor):
                        # Ensure float32 to avoid complex number issues
                        z = z.to(dtype=torch.float32)
                        centroids_real = self.centroids_tens.to(dtype=torch.float32)
                        M_tens_real = self.M_tens.to(dtype=torch.float32)
                        
                        diff = centroids_real.unsqueeze(0) - z.unsqueeze(1)
                        weights = torch.exp(-torch.norm(diff, dim=-1) ** 2 / (self.temperature ** 2))
                        weighted_M = M_tens_real.unsqueeze(0) * weights.unsqueeze(-1).unsqueeze(-1)
                        G_inv = weighted_M.sum(dim=1) + self.lbd * torch.eye(self.latent_dim, device=z.device, dtype=torch.float32)
                        
                        # Add small regularization for stability
                        G_inv = G_inv + 1e-6 * torch.eye(self.latent_dim, device=z.device, dtype=torch.float32)
                        return G_inv.to(dtype=torch.float32)
                    
                    def G(z: torch.Tensor):
                        try:
                            return torch.linalg.inv(G_inv(z))
                        except:
                            return torch.linalg.pinv(G_inv(z))
                    
                    self.G = G
                    self.G_inv = G_inv
            
            # Create wrapper and sampler
            wrapper_model = RHVAECompatibleWrapper(self.model, self.device)
            sampler_config = RHVAESamplerConfig(
                mcmc_steps_nbr=50,  # Moderate steps for visualization
                n_lf=10,           # Fewer leapfrog steps for speed
                eps_lf=0.01,       # Smaller step size for stability
                beta_zero=1.0
            )
            
            self.rhvae_sampler = RHVAESampler(wrapper_model, sampler_config)
            logger.info("✅ RHVAE sampler initialized for manifold evolution tracking")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize RHVAE sampler: {e}")
            self.rhvae_sampler = None
    
    def record_manifold_snapshot(self, epoch: int, latent_data: Optional[np.ndarray] = None) -> None:
        """Record current manifold state for evolution tracking."""
        try:
            if not hasattr(self.model, 'modular_metric'):
                return
            
            # Get current centroids and metrics
            current_centroids = self.model.modular_metric.centroids.clone().detach().cpu().numpy()
            current_metrics = self.model.modular_metric.metric_matrices.clone().detach().cpu().numpy()
            
            # Store snapshots
            self.centroid_snapshots.append(current_centroids.copy())
            self.metric_snapshots.append(current_metrics.copy())
            self.evolution_epochs.append(epoch)
            
            # Compute evolution metrics if latent data provided
            if latent_data is not None:
                self._compute_and_store_evolution_metrics(current_centroids, current_metrics, latent_data)
            
            logger.info(f"📸 Recorded manifold snapshot at epoch {epoch}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to record manifold snapshot: {e}")
    
    def _compute_and_store_evolution_metrics(self, centroids: np.ndarray, 
                                           metrics: np.ndarray, latent_data: np.ndarray) -> None:
        """Compute and store manifold evolution metrics."""
        
        # Centroid shift from original
        centroid_shift = np.mean(np.linalg.norm(centroids - self.original_centroids, axis=1))
        self.manifold_metrics_history['centroid_shifts'].append(centroid_shift)
        
        # Metric change from original
        metric_change = np.mean([
            np.linalg.norm(metrics[i] - self.original_metric_matrices[i]) 
            for i in range(len(metrics))
        ])
        self.manifold_metrics_history['metric_changes'].append(metric_change)
        
        # Coverage metric (how well centroids cover latent space)
        distances_to_nearest = []
        for point in latent_data:
            dists = np.linalg.norm(centroids - point, axis=1)
            distances_to_nearest.append(np.min(dists))
        coverage = np.mean(distances_to_nearest)
        self.manifold_metrics_history['coverage_metrics'].append(coverage)
        
        # Latent variance
        latent_variance = np.var(latent_data, axis=0).mean()
        self.manifold_metrics_history['latent_variances'].append(latent_variance)
    
    def create_manifold_evolution_summary(self, epoch: int) -> None:
        """Create comprehensive manifold evolution visualization."""
        try:
            if len(self.centroid_snapshots) < 2:
                logger.info("📊 Need at least 2 manifold snapshots for evolution visualization")
                return
            
            # Create the evolution visualization
            fig = self._create_evolution_plot()
            
            # Log to wandb
            if self.should_log_to_wandb:
                wandb.log({
                    f"manifold_evolution/summary_epoch_{epoch}": wandb.Html(fig.to_html()),
                    **self._get_latest_metrics_for_wandb()
                })
            
            # Save locally if needed
            output_path = Path("outputs/manifold_evolution") / f"evolution_epoch_{epoch}.html"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            
            logger.info(f"✅ Created manifold evolution summary for epoch {epoch}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create manifold evolution summary: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_evolution_plot(self) -> go.Figure:
        """Create the main manifold evolution plot."""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Centroid Trajectories Over Time",
                "Manifold Metrics Evolution",
                "Sampling Density Evolution", 
                "Current vs Original Centroids",
                "Metric Tensor Changes",
                "Manifold Coverage Over Time"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}], 
                [{"type": "scatter"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.08
        )
        
        # 1. Centroid trajectories
        self._add_centroid_trajectories(fig, row=1, col=1)
        
        # 2. Evolution metrics
        self._add_evolution_metrics(fig, row=1, col=2)
        
        # 3. Sampling density evolution (simplified)
        self._add_sampling_density_evolution(fig, row=2, col=1)
        
        # 4. Current vs original comparison
        self._add_current_vs_original(fig, row=2, col=2)
        
        # 5. Metric tensor changes
        self._add_metric_tensor_changes(fig, row=3, col=1)
        
        # 6. Coverage evolution
        self._add_coverage_evolution(fig, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            height=1200,
            title=dict(
                text="🌊 Manifold Evolution During Adaptive Training<br><sub>Living manifold: How geometry adapts as the model learns</sub>",
                x=0.5,
                font=dict(size=18)
            ),
            showlegend=True
        )
        
        return fig
    
    def _add_centroid_trajectories(self, fig: go.Figure, row: int, col: int) -> None:
        """Add centroid trajectory visualization."""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Show trajectories for first 5 centroids to avoid clutter
        for centroid_idx in range(min(5, len(self.original_centroids))):
            # Extract trajectory for this centroid
            trajectory_x = [snap[centroid_idx, 0] for snap in self.centroid_snapshots]
            trajectory_y = [snap[centroid_idx, 1] for snap in self.centroid_snapshots]
            
            # Add trajectory line
            fig.add_trace(
                go.Scatter(
                    x=trajectory_x, y=trajectory_y,
                    mode='lines+markers',
                    line=dict(color=colors[centroid_idx % len(colors)], width=2),
                    marker=dict(size=6),
                    name=f'Centroid {centroid_idx}',
                    hovertemplate=f"Centroid {centroid_idx}<br>Z1: %{{x:.3f}}<br>Z2: %{{y:.3f}}<extra></extra>"
                ), row=row, col=col
            )
            
            # Mark start and end points
            if len(trajectory_x) >= 2:
                # Start point
                fig.add_trace(
                    go.Scatter(
                        x=[trajectory_x[0]], y=[trajectory_y[0]],
                        mode='markers',
                        marker=dict(size=10, color=colors[centroid_idx % len(colors)], 
                                  symbol='circle', line=dict(width=2, color='white')),
                        name=f'Start {centroid_idx}',
                        showlegend=False,
                        hovertemplate=f"Start Centroid {centroid_idx}<br>Z1: %{{x:.3f}}<br>Z2: %{{y:.3f}}<extra></extra>"
                    ), row=row, col=col
                )
                
                # End point  
                fig.add_trace(
                    go.Scatter(
                        x=[trajectory_x[-1]], y=[trajectory_y[-1]],
                        mode='markers',
                        marker=dict(size=10, color=colors[centroid_idx % len(colors)], 
                                  symbol='diamond', line=dict(width=2, color='white')),
                        name=f'End {centroid_idx}',
                        showlegend=False,
                        hovertemplate=f"Current Centroid {centroid_idx}<br>Z1: %{{x:.3f}}<br>Z2: %{{y:.3f}}<extra></extra>"
                    ), row=row, col=col
                )
    
    def _add_evolution_metrics(self, fig: go.Figure, row: int, col: int) -> None:
        """Add evolution metrics plot."""
        if not self.manifold_metrics_history['centroid_shifts']:
            return
        
        epochs = self.evolution_epochs[:len(self.manifold_metrics_history['centroid_shifts'])]
        
        # Centroid shifts
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=self.manifold_metrics_history['centroid_shifts'],
                mode='lines+markers',
                name='Centroid Shift',
                line=dict(color='blue', width=2),
                hovertemplate="Epoch: %{x}<br>Shift: %{y:.4f}<extra></extra>"
            ), row=row, col=col
        )
        
        # Metric changes (on secondary y-axis if available)
        if len(self.manifold_metrics_history['metric_changes']) == len(epochs):
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=self.manifold_metrics_history['metric_changes'],
                    mode='lines+markers',
                    name='Metric Change', 
                    line=dict(color='red', width=2, dash='dash'),
                    hovertemplate="Epoch: %{x}<br>Metric Change: %{y:.4f}<extra></extra>",
                    yaxis='y2'
                ), row=row, col=col
            )
    
    def _add_sampling_density_evolution(self, fig: go.Figure, row: int, col: int) -> None:
        """Add sampling density evolution visualization."""
        if len(self.centroid_snapshots) < 2:
            return
        
        # Create density comparison between first and last snapshots
        first_centroids = self.centroid_snapshots[0]
        last_centroids = self.centroid_snapshots[-1]
        
        # Original density (simplified visualization)
        fig.add_trace(
            go.Scatter(
                x=first_centroids[:, 0], y=first_centroids[:, 1],
                mode='markers',
                marker=dict(size=8, color='lightblue', opacity=0.6),
                name='Original Density',
                hovertemplate="Original<br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<extra></extra>"
            ), row=row, col=col
        )
        
        # Current density
        fig.add_trace(
            go.Scatter(
                x=last_centroids[:, 0], y=last_centroids[:, 1],
                mode='markers',
                marker=dict(size=8, color='darkblue', opacity=0.8),
                name='🚀 Evolved Density',
                hovertemplate="Current<br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<extra></extra>"
            ), row=row, col=col
        )
    
    def _add_current_vs_original(self, fig: go.Figure, row: int, col: int) -> None:
        """Add current vs original centroids comparison."""
        if not self.centroid_snapshots:
            return
        
        current_centroids = self.centroid_snapshots[-1]
        
        # Original centroids
        fig.add_trace(
            go.Scatter(
                x=self.original_centroids[:, 0], y=self.original_centroids[:, 1],
                mode='markers',
                marker=dict(size=10, color='red', symbol='circle', opacity=0.7),
                name='Original Centroids',
                hovertemplate="Original<br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<extra></extra>"
            ), row=row, col=col
        )
        
        # Current centroids
        fig.add_trace(
            go.Scatter(
                x=current_centroids[:, 0], y=current_centroids[:, 1],
                mode='markers',
                marker=dict(size=10, color='green', symbol='diamond', opacity=0.9),
                name='🚀 Current Centroids',
                hovertemplate="Current<br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<extra></extra>"
            ), row=row, col=col
        )
        
        # Draw arrows showing movement
        for i in range(min(len(self.original_centroids), len(current_centroids))):
            if i < 10:  # Show first 10 for clarity
                fig.add_annotation(
                    x=current_centroids[i, 0], y=current_centroids[i, 1],
                    ax=self.original_centroids[i, 0], ay=self.original_centroids[i, 1],
                    xref=f'x{4}', yref=f'y{4}',
                    axref=f'x{4}', ayref=f'y{4}',
                    arrowhead=2, arrowsize=1, arrowwidth=1,
                    arrowcolor='gray', opacity=0.6
                )
    
    def _add_metric_tensor_changes(self, fig: go.Figure, row: int, col: int) -> None:
        """Add metric tensor change visualization."""
        if not self.manifold_metrics_history['metric_changes']:
            return
        
        epochs = self.evolution_epochs[:len(self.manifold_metrics_history['metric_changes'])]
        
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=self.manifold_metrics_history['metric_changes'],
                mode='lines+markers',
                name='Metric Tensor Change',
                line=dict(color='purple', width=2),
                hovertemplate="Epoch: %{x}<br>Change: %{y:.4f}<extra></extra>"
            ), row=row, col=col
        )
        
        # Add trend line if we have enough points
        if len(self.manifold_metrics_history['metric_changes']) >= 3:
            z = np.polyfit(epochs, self.manifold_metrics_history['metric_changes'], 1)
            trend_line = np.poly1d(z)
            
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=trend_line(epochs),
                    mode='lines',
                    name='Trend',
                    line=dict(color='purple', width=1, dash='dot'),
                    showlegend=False
                ), row=row, col=col
            )
    
    def _add_coverage_evolution(self, fig: go.Figure, row: int, col: int) -> None:
        """Add manifold coverage evolution."""
        if not self.manifold_metrics_history['coverage_metrics']:
            return
        
        epochs = self.evolution_epochs[:len(self.manifold_metrics_history['coverage_metrics'])]
        
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=self.manifold_metrics_history['coverage_metrics'],
                mode='lines+markers',
                name='Manifold Coverage',
                line=dict(color='green', width=2),
                fill='tonexty' if len(epochs) > 1 else None,
                fillcolor='rgba(0,255,0,0.1)',
                hovertemplate="Epoch: %{x}<br>Coverage: %{y:.4f}<extra></extra>"
            ), row=row, col=col
        )
    
    def _get_latest_metrics_for_wandb(self) -> Dict:
        """Get latest metrics for wandb logging."""
        metrics = {}
        
        if self.manifold_metrics_history['centroid_shifts']:
            metrics['manifold_evolution/latest_centroid_shift'] = self.manifold_metrics_history['centroid_shifts'][-1]
        
        if self.manifold_metrics_history['metric_changes']:
            metrics['manifold_evolution/latest_metric_change'] = self.manifold_metrics_history['metric_changes'][-1]
        
        if self.manifold_metrics_history['coverage_metrics']:
            metrics['manifold_evolution/latest_coverage'] = self.manifold_metrics_history['coverage_metrics'][-1]
        
        if self.manifold_metrics_history['latent_variances']:
            metrics['manifold_evolution/latest_latent_variance'] = self.manifold_metrics_history['latent_variances'][-1]
        
        metrics['manifold_evolution/total_snapshots'] = len(self.centroid_snapshots)
        
        return metrics
    
    def create_manifold_0_sampling_visualization(self, epoch: int, 
                                               sampled_points: Optional[np.ndarray] = None) -> None:
        """Create visualization specifically for manifold 0 sampling using evolved RHVAE sampler."""
        try:
            # Get current centroids (handle both model types)
            current_centroids = None
            if hasattr(self.model, 'modular_metric') and hasattr(self.model.modular_metric, 'centroids'):
                current_centroids = self.model.modular_metric.centroids.clone().detach().cpu().numpy()
            elif hasattr(self.model, 'centroids_tens'):
                current_centroids = self.model.centroids_tens.clone().detach().cpu().numpy()
            
            if current_centroids is None:
                logger.warning("⚠️ No centroids found in model")
                return
            
            # ✨ GENERATE RHVAE SAMPLES WITH EVOLVED CENTROIDS ✨
            rhvae_samples = None
            if self.rhvae_sampler is not None:
                try:
                    with torch.no_grad():
                        # Sample from the evolved manifold using RHVAE
                        n_samples = 200  # Dense sampling for visualization
                        rhvae_sample_tensor = self.rhvae_sampler.hmc_sampling(n_samples)
                        rhvae_samples = rhvae_sample_tensor.detach().cpu().numpy()
                        logger.info(f"✅ Generated {n_samples} RHVAE samples with evolved centroids")
                except Exception as e:
                    logger.warning(f"⚠️ RHVAE sampling failed: {e}")
            
            # Create sampling visualization
            fig = go.Figure()
            
            # Add centroids (evolved structure)
            fig.add_trace(
                go.Scatter(
                    x=current_centroids[:, 0], y=current_centroids[:, 1],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='diamond', line=dict(width=2, color='darkred')),
                    name=f'Evolved Centroids (Epoch {epoch})',
                    hovertemplate="Evolved Centroid<br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<extra></extra>"
                )
            )
            
            # Add RHVAE samples (following evolved manifold structure)
            if rhvae_samples is not None:
                fig.add_trace(
                    go.Scatter(
                        x=rhvae_samples[:, 0], y=rhvae_samples[:, 1],
                        mode='markers',
                        marker=dict(size=4, color='blue', opacity=0.7),
                        name='RHVAE Samples (Evolved Manifold)',
                        hovertemplate="RHVAE Sample<br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<extra></extra>"
                    )
                )
            
            # Add original sampled points if provided (for comparison)
            if sampled_points is not None:
                fig.add_trace(
                    go.Scatter(
                        x=sampled_points[:, 0], y=sampled_points[:, 1],
                        mode='markers',
                        marker=dict(size=4, color='green', opacity=0.5),
                        name='Other Samples',
                        hovertemplate="Other Sample<br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<extra></extra>"
                    )
                )
            
            fig.update_layout(
                title=f"🌊 Evolved Manifold Sampling - Epoch {epoch}<br><sup>RHVAE samples follow the evolved centroid structure</sup>",
                xaxis_title="Latent Dimension 1",
                yaxis_title="Latent Dimension 2",
                template="plotly_white",
                showlegend=True
            )
            
            # Log to wandb with enhanced info
            if self.should_log_to_wandb:
                wandb_data = {f"evolved_manifold_sampling/epoch_{epoch}": wandb.Html(fig.to_html())}
                if rhvae_samples is not None:
                    wandb_data[f"evolved_manifold_sampling/n_rhvae_samples"] = len(rhvae_samples)
                    wandb_data[f"evolved_manifold_sampling/rhvae_mean_z1"] = float(np.mean(rhvae_samples[:, 0]))
                    wandb_data[f"evolved_manifold_sampling/rhvae_mean_z2"] = float(np.mean(rhvae_samples[:, 1]))
                    wandb_data[f"evolved_manifold_sampling/rhvae_std_z1"] = float(np.std(rhvae_samples[:, 0]))
                    wandb_data[f"evolved_manifold_sampling/rhvae_std_z2"] = float(np.std(rhvae_samples[:, 1]))
                
                wandb.log(wandb_data)
            
            logger.info(f"🎯 Created evolved manifold sampling visualization for epoch {epoch}")
            if rhvae_samples is not None:
                logger.info(f"   📊 RHVAE samples follow evolved centroid structure!")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create evolved manifold sampling visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def create_sampling_evolution_summary(self, epoch: int) -> None:
        """Create summary showing how sampling on manifold 0 evolves."""
        try:
            if len(self.centroid_snapshots) < 2:
                return
            
            # Create evolution comparison
            fig = make_subplots(
                rows=1, cols=len(self.centroid_snapshots),
                subplot_titles=[f"Epoch {ep}" for ep in self.evolution_epochs],
                horizontal_spacing=0.05
            )
            
            colors = ['red', 'orange', 'green', 'blue', 'purple']
            
            for i, (centroids, ep) in enumerate(zip(self.centroid_snapshots, self.evolution_epochs)):
                col_idx = i + 1
                
                fig.add_trace(
                    go.Scatter(
                        x=centroids[:, 0], y=centroids[:, 1],
                        mode='markers',
                        marker=dict(size=8, color=colors[i % len(colors)], opacity=0.8),
                        name=f'Epoch {ep}',
                        hovertemplate=f"Epoch {ep}<br>Z1: %{{x:.3f}}<br>Z2: %{{y:.3f}}<extra></extra>"
                    ), row=1, col=col_idx
                )
            
            fig.update_layout(
                title="🌊 Manifold 0 Evolution: Sampling Points Over Training",
                height=400
            )
            
            # Log to wandb
            if self.should_log_to_wandb:
                wandb.log({f"manifold_0_evolution/epoch_{epoch}": wandb.Html(fig.to_html())})
            
            logger.info(f"✅ Created sampling evolution summary for epoch {epoch}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create sampling evolution summary: {e}") 