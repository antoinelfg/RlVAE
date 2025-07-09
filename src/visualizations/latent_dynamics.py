"""
Latent Dynamics Visualizations Module
====================================

Advanced latent space dynamics analysis:
- Phase portraits and velocity fields
- Acceleration and energy analysis  
- Attractor identification
- Stability analysis
- Dynamical system characterization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from .base import BaseVisualization

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class LatentDynamicsVisualizations(BaseVisualization):
    """Advanced latent space dynamics visualization suite."""
    
    def create_phase_portrait_analysis(self, x_sample: torch.Tensor, epoch: int):
        """Create comprehensive phase portrait analysis of latent dynamics."""
        print(f"🌀 Creating phase portrait analysis for epoch {epoch}")
        
        self.model.eval()
        with torch.no_grad():
            result = self.model_forward(x_sample)
            z_seq = result['latent_samples'] if isinstance(result, dict) else result.z
            
            batch_size, n_obs, latent_dim = z_seq.shape
            
            # Compute velocities and accelerations
            dynamics_data = self._compute_comprehensive_dynamics(z_seq)
            
            # Create phase portraits
            self._create_phase_portraits(z_seq, dynamics_data, epoch)
            
        self.model.train()
    
    def create_velocity_field_analysis(self, x_sample: torch.Tensor, epoch: int):
        """Create velocity field and flow analysis."""
        print(f"🌊 Creating velocity field analysis for epoch {epoch}")
        
        self.model.eval()
        with torch.no_grad():
            result = self.model_forward(x_sample)
            z_seq = result['latent_samples'] if isinstance(result, dict) else result.z
            
            # Compute dynamics
            dynamics_data = self._compute_comprehensive_dynamics(z_seq)
            
            # Create velocity field visualization
            self._create_velocity_field_plots(z_seq, dynamics_data, epoch)
            
        self.model.train()
    
    def create_energy_landscape_analysis(self, x_sample: torch.Tensor, epoch: int):
        """Create energy landscape and stability analysis."""
        print(f"⚡ Creating energy landscape analysis for epoch {epoch}")
        
        self.model.eval()
        with torch.no_grad():
            result = self.model_forward(x_sample)
            z_seq = result['latent_samples'] if isinstance(result, dict) else result.z
            
            # Compute energy landscapes
            energy_data = self._compute_energy_landscapes(z_seq)
            
            # Create energy visualization
            self._create_energy_landscape_plots(z_seq, energy_data, epoch)
            
        self.model.train()
    
    def create_attractor_analysis(self, x_sample: torch.Tensor, epoch: int):
        """Analyze attractors and stability in latent space."""
        print(f"🎯 Creating attractor analysis for epoch {epoch}")
        
        self.model.eval()
        with torch.no_grad():
            result = self.model_forward(x_sample)
            z_seq = result['latent_samples'] if isinstance(result, dict) else result.z
            
            # Identify attractors and analyze stability
            attractor_data = self._analyze_attractors(z_seq)
            
            # Create attractor visualization
            self._create_attractor_plots(z_seq, attractor_data, epoch)
            
        self.model.train()
    
    def _compute_comprehensive_dynamics(self, z_seq):
        """Compute comprehensive dynamics metrics."""
        batch_size, n_obs, latent_dim = z_seq.shape
        dynamics_data = {
            'velocities': [],
            'accelerations': [],
            'velocity_magnitudes': [],
            'acceleration_magnitudes': [],
            'curvatures': [],
            'angular_velocities': [],
            'kinetic_energies': [],
            'trajectory_lengths': []
        }
        
        for i in range(batch_size):
            z_traj = z_seq[i].cpu().numpy()  # [n_obs, latent_dim]
            
            # Compute velocities (finite differences)
            if n_obs > 1:
                velocities = np.diff(z_traj, axis=0)  # [n_obs-1, latent_dim]
                velocity_mags = np.linalg.norm(velocities, axis=1)
                dynamics_data['velocities'].append(velocities)
                dynamics_data['velocity_magnitudes'].append(velocity_mags)
                
                # Compute accelerations
                if n_obs > 2:
                    accelerations = np.diff(velocities, axis=0)  # [n_obs-2, latent_dim]
                    accel_mags = np.linalg.norm(accelerations, axis=1)
                    dynamics_data['accelerations'].append(accelerations)
                    dynamics_data['acceleration_magnitudes'].append(accel_mags)
                
                # Compute curvatures using consecutive velocity vectors
                curvatures = []
                angular_velocities = []
                
                for j in range(len(velocities) - 1):
                    v1, v2 = velocities[j], velocities[j + 1]
                    v1_norm = np.linalg.norm(v1)
                    v2_norm = np.linalg.norm(v2)
                    
                    if v1_norm > 1e-8 and v2_norm > 1e-8:
                        # Curvature calculation
                        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle_change = np.arccos(cos_angle)
                        curvature = angle_change / max(v1_norm, 1e-8)
                        curvatures.append(curvature)
                        
                        # Angular velocity
                        dt = 1.0  # Assuming unit time steps
                        angular_vel = angle_change / dt
                        angular_velocities.append(angular_vel)
                
                dynamics_data['curvatures'].append(curvatures)
                dynamics_data['angular_velocities'].append(angular_velocities)
                
                # Kinetic energy (1/2 * m * v^2, assuming unit mass)
                kinetic_energy = 0.5 * np.sum(velocity_mags ** 2)
                dynamics_data['kinetic_energies'].append(kinetic_energy)
                
                # Trajectory length
                trajectory_length = np.sum(velocity_mags)
                dynamics_data['trajectory_lengths'].append(trajectory_length)
            
        return dynamics_data
    
    def _create_phase_portraits(self, z_seq, dynamics_data, epoch):
        """Create phase portrait visualizations."""
        batch_size, n_obs, latent_dim = z_seq.shape
        
        # Use PCA for visualization
        z_pca_seq, pca = self._prepare_pca_data(z_seq, n_components=3)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Phase Portrait Analysis - Epoch {epoch}', fontsize=16)
        
        # Plot 1: Position vs Velocity (PC1)
        ax = axes[0, 0]
        for i in range(min(batch_size, 8)):
            if i < len(dynamics_data['velocities']):
                positions = z_pca_seq[i, :-1, 0]  # Exclude last position to match velocity length
                velocities = dynamics_data['velocities'][i][:, 0] if latent_dim > 0 else []
                
                if len(velocities) > 0 and len(positions) == len(velocities):
                    ax.plot(positions, velocities, 'o-', alpha=0.7, markersize=3, linewidth=1)
        
        ax.set_xlabel('Position (PC1)')
        ax.set_ylabel('Velocity (PC1)')
        ax.set_title('Phase Portrait: PC1')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Position vs Velocity (PC2)
        ax = axes[0, 1]
        for i in range(min(batch_size, 8)):
            if i < len(dynamics_data['velocities']) and z_pca_seq.shape[2] > 1:
                positions = z_pca_seq[i, :-1, 1]
                velocities = dynamics_data['velocities'][i][:, 1] if latent_dim > 1 else []
                
                if len(velocities) > 0 and len(positions) == len(velocities):
                    ax.plot(positions, velocities, 'o-', alpha=0.7, markersize=3, linewidth=1)
        
        ax.set_xlabel('Position (PC2)')
        ax.set_ylabel('Velocity (PC2)')
        ax.set_title('Phase Portrait: PC2')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: 3D Phase Portrait (if available)
        ax = axes[0, 2]
        if z_pca_seq.shape[2] >= 3:
            ax = fig.add_subplot(2, 3, 3, projection='3d')
            for i in range(min(batch_size, 6)):
                traj = z_pca_seq[i, :, :3]
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.7, linewidth=2)
                # Mark start and end
                ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color='green', s=50, marker='o')
                ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='red', s=50, marker='s')
            
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.set_title('3D Trajectory')
        else:
            ax.text(0.5, 0.5, '3D visualization\nnot available\n(latent_dim < 3)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('3D Trajectory')
        
        # Plot 4: Velocity magnitude distribution
        ax = axes[1, 0]
        all_vel_mags = []
        for vel_mags in dynamics_data['velocity_magnitudes']:
            all_vel_mags.extend(vel_mags)
        
        if all_vel_mags:
            ax.hist(all_vel_mags, bins=30, alpha=0.7, density=True, color='blue')
            ax.axvline(np.mean(all_vel_mags), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(all_vel_mags):.2e}')
            ax.set_xlabel('Velocity Magnitude')
            ax.set_ylabel('Density')
            ax.set_title('Velocity Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 5: Curvature distribution
        ax = axes[1, 1]
        all_curvatures = []
        for curvs in dynamics_data['curvatures']:
            all_curvatures.extend(curvs)
        
        if all_curvatures:
            ax.hist(all_curvatures, bins=25, alpha=0.7, density=True, color='orange')
            ax.axvline(np.mean(all_curvatures), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(all_curvatures):.3f}')
            ax.set_xlabel('Trajectory Curvature')
            ax.set_ylabel('Density')
            ax.set_title('Curvature Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 6: Kinetic energy vs trajectory length
        ax = axes[1, 2]
        if dynamics_data['kinetic_energies'] and dynamics_data['trajectory_lengths']:
            ax.scatter(dynamics_data['trajectory_lengths'], dynamics_data['kinetic_energies'], 
                      alpha=0.7, s=50)
            
            # Add correlation
            if len(dynamics_data['kinetic_energies']) > 1:
                corr = np.corrcoef(dynamics_data['trajectory_lengths'], 
                                 dynamics_data['kinetic_energies'])[0, 1]
                ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            ax.set_xlabel('Trajectory Length')
            ax.set_ylabel('Kinetic Energy')
            ax.set_title('Energy vs Path Length')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f'phase_portrait_analysis_epoch_{epoch}.png'
        saved_file = self._safe_save_plt_figure(filename, dpi=200, bbox_inches='tight')
        
        if self.should_log_to_wandb() and saved_file:
            wandb.log({
                "dynamics/phase_portrait": wandb.Image(saved_file, caption=f"Phase Portrait Analysis - Epoch {epoch}"),
                "metrics/mean_velocity_magnitude": np.mean(all_vel_mags) if all_vel_mags else 0.0,
                "metrics/mean_trajectory_curvature": np.mean(all_curvatures) if all_curvatures else 0.0,
                "metrics/mean_kinetic_energy": np.mean(dynamics_data['kinetic_energies']) if dynamics_data['kinetic_energies'] else 0.0,
            })
        
        plt.close()
    
    def _create_velocity_field_plots(self, z_seq, dynamics_data, epoch):
        """Create velocity field visualizations."""
        batch_size, n_obs, latent_dim = z_seq.shape
        z_pca_seq, pca = self._prepare_pca_data(z_seq, n_components=2)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Velocity Field Analysis - Epoch {epoch}', fontsize=16)
        
        # Plot 1: Vector field in PCA space
        ax = axes[0, 0]
        for i in range(min(batch_size, 10)):
            if i < len(dynamics_data['velocities']):
                positions = z_pca_seq[i, :-1, :2]  # [n_obs-1, 2]
                velocities_pca = np.diff(z_pca_seq[i, :, :2], axis=0)  # Velocity in PCA space
                
                # Subsample for clarity
                step = max(1, len(positions) // 10)
                pos_sub = positions[::step]
                vel_sub = velocities_pca[::step]
                
                for j in range(len(pos_sub)):
                    if j < len(vel_sub):
                        ax.arrow(pos_sub[j, 0], pos_sub[j, 1], 
                                vel_sub[j, 0]*0.5, vel_sub[j, 1]*0.5,
                                head_width=0.05, head_length=0.05, 
                                fc=f'C{i%10}', ec=f'C{i%10}', alpha=0.7)
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Velocity Vector Field')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Velocity magnitude heatmap
        ax = axes[0, 1]
        all_positions = []
        all_vel_mags = []
        
        for i in range(min(batch_size, 20)):
            if i < len(dynamics_data['velocity_magnitudes']):
                positions = z_pca_seq[i, :-1, :2]
                vel_mags = dynamics_data['velocity_magnitudes'][i]
                
                if len(positions) == len(vel_mags):
                    all_positions.extend(positions)
                    all_vel_mags.extend(vel_mags)
        
        if all_positions and all_vel_mags:
            all_positions = np.array(all_positions)
            all_vel_mags = np.array(all_vel_mags)
            
            scatter = ax.scatter(all_positions[:, 0], all_positions[:, 1], 
                               c=all_vel_mags, cmap='viridis', alpha=0.6, s=20)
            plt.colorbar(scatter, ax=ax, label='Velocity Magnitude')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title('Velocity Magnitude Field')
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Acceleration analysis
        ax = axes[1, 0]
        all_accel_mags = []
        for accel_mags in dynamics_data['acceleration_magnitudes']:
            all_accel_mags.extend(accel_mags)
        
        if all_accel_mags:
            ax.hist(all_accel_mags, bins=25, alpha=0.7, density=True, color='red')
            ax.axvline(np.mean(all_accel_mags), color='blue', linestyle='--', 
                      label=f'Mean: {np.mean(all_accel_mags):.2e}')
            ax.set_xlabel('Acceleration Magnitude')
            ax.set_ylabel('Density')
            ax.set_title('Acceleration Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Angular velocity analysis
        ax = axes[1, 1]
        all_angular_vels = []
        for ang_vels in dynamics_data['angular_velocities']:
            all_angular_vels.extend(ang_vels)
        
        if all_angular_vels:
            ax.hist(all_angular_vels, bins=25, alpha=0.7, density=True, color='purple')
            ax.axvline(np.mean(all_angular_vels), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(all_angular_vels):.3f} rad/step')
            ax.set_xlabel('Angular Velocity (rad/step)')
            ax.set_ylabel('Density')
            ax.set_title('Angular Velocity Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f'velocity_field_analysis_epoch_{epoch}.png'
        saved_file = self._safe_save_plt_figure(filename, dpi=200, bbox_inches='tight')
        
        if self.should_log_to_wandb() and saved_file:
            wandb.log({
                "dynamics/velocity_field": wandb.Image(saved_file, caption=f"Velocity Field Analysis - Epoch {epoch}"),
                "metrics/mean_acceleration_magnitude": np.mean(all_accel_mags) if all_accel_mags else 0.0,
                "metrics/mean_angular_velocity": np.mean(all_angular_vels) if all_angular_vels else 0.0,
            })
        
        plt.close()
    
    def _compute_energy_landscapes(self, z_seq):
        """Compute energy landscapes and potential functions."""
        batch_size, n_obs, latent_dim = z_seq.shape
        
        energy_data = {
            'kinetic_energies': [],
            'potential_energies': [],
            'total_energies': [],
            'energy_conservation': [],
            'lyapunov_estimates': []
        }
        
        for i in range(batch_size):
            z_traj = z_seq[i].cpu().numpy()
            
            if n_obs > 1:
                # Kinetic energy (1/2 * v^2)
                velocities = np.diff(z_traj, axis=0)
                kinetic = 0.5 * np.sum(velocities ** 2, axis=1)
                energy_data['kinetic_energies'].append(kinetic)
                
                # Potential energy (distance from origin, simplified)
                potential = 0.5 * np.sum(z_traj[:-1] ** 2, axis=1)
                energy_data['potential_energies'].append(potential)
                
                # Total energy
                total = kinetic + potential
                energy_data['total_energies'].append(total)
                
                # Energy conservation (variance of total energy)
                energy_conservation = np.var(total)
                energy_data['energy_conservation'].append(energy_conservation)
                
                # Simple Lyapunov estimate (divergence of nearby trajectories)
                if i < batch_size - 1:
                    z_nearby = z_seq[i + 1].cpu().numpy()
                    distances = np.linalg.norm(z_traj - z_nearby, axis=1)
                    if len(distances) > 1:
                        # Log of distance ratio as simple Lyapunov estimate
                        lyap_est = np.mean(np.log(distances[1:] / (distances[:-1] + 1e-8)))
                        energy_data['lyapunov_estimates'].append(lyap_est)
        
        return energy_data
    
    def _create_energy_landscape_plots(self, z_seq, energy_data, epoch):
        """Create energy landscape visualizations."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Energy Landscape Analysis - Epoch {epoch}', fontsize=16)
        
        # Plot 1: Kinetic energy evolution
        ax = axes[0, 0]
        for i, kinetic in enumerate(energy_data['kinetic_energies'][:8]):
            ax.plot(kinetic, alpha=0.7, linewidth=2, label=f'Seq {i}' if len(energy_data['kinetic_energies']) <= 8 else None)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Kinetic Energy')
        ax.set_title('Kinetic Energy Evolution')
        if len(energy_data['kinetic_energies']) <= 8:
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Potential energy evolution
        ax = axes[0, 1]
        for i, potential in enumerate(energy_data['potential_energies'][:8]):
            ax.plot(potential, alpha=0.7, linewidth=2, label=f'Seq {i}' if len(energy_data['potential_energies']) <= 8 else None)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Potential Energy')
        ax.set_title('Potential Energy Evolution')
        if len(energy_data['potential_energies']) <= 8:
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Total energy evolution
        ax = axes[0, 2]
        for i, total in enumerate(energy_data['total_energies'][:8]):
            ax.plot(total, alpha=0.7, linewidth=2, label=f'Seq {i}' if len(energy_data['total_energies']) <= 8 else None)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Total Energy')
        ax.set_title('Total Energy Evolution')
        if len(energy_data['total_energies']) <= 8:
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Energy conservation
        ax = axes[1, 0]
        if energy_data['energy_conservation']:
            ax.hist(energy_data['energy_conservation'], bins=20, alpha=0.7, density=True, color='green')
            ax.axvline(np.mean(energy_data['energy_conservation']), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(energy_data["energy_conservation"]):.2e}')
            ax.set_xlabel('Energy Variance')
            ax.set_ylabel('Density')
            ax.set_title('Energy Conservation')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 5: Lyapunov estimates
        ax = axes[1, 1]
        if energy_data['lyapunov_estimates']:
            ax.hist(energy_data['lyapunov_estimates'], bins=15, alpha=0.7, density=True, color='purple')
            ax.axvline(np.mean(energy_data['lyapunov_estimates']), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(energy_data["lyapunov_estimates"]):.3f}')
            ax.axvline(0, color='black', linestyle=':', alpha=0.7, label='Stable')
            ax.set_xlabel('Lyapunov Estimate')
            ax.set_ylabel('Density')
            ax.set_title('Stability Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 6: Energy phase space
        ax = axes[1, 2]
        if energy_data['kinetic_energies'] and energy_data['potential_energies']:
            for i in range(min(len(energy_data['kinetic_energies']), 10)):
                kinetic = energy_data['kinetic_energies'][i]
                potential = energy_data['potential_energies'][i]
                if len(kinetic) == len(potential):
                    ax.plot(kinetic, potential, 'o-', alpha=0.7, markersize=3, linewidth=1)
            
            ax.set_xlabel('Kinetic Energy')
            ax.set_ylabel('Potential Energy')
            ax.set_title('Energy Phase Space')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f'energy_landscape_analysis_epoch_{epoch}.png'
        saved_file = self._safe_save_plt_figure(filename, dpi=200, bbox_inches='tight')
        
        if self.should_log_to_wandb() and saved_file:
            mean_energy_conservation = np.mean(energy_data['energy_conservation']) if energy_data['energy_conservation'] else 0.0
            mean_lyapunov = np.mean(energy_data['lyapunov_estimates']) if energy_data['lyapunov_estimates'] else 0.0
            
            wandb.log({
                "dynamics/energy_landscape": wandb.Image(saved_file, caption=f"Energy Landscape Analysis - Epoch {epoch}"),
                "metrics/energy_conservation": mean_energy_conservation,
                "metrics/lyapunov_estimate": mean_lyapunov,
            })
        
        plt.close()
    
    def _analyze_attractors(self, z_seq):
        """Analyze attractors and fixed points in the latent space."""
        batch_size, n_obs, latent_dim = z_seq.shape
        
        attractor_data = {
            'fixed_points': [],
            'attractor_strengths': [],
            'basin_sizes': [],
            'convergence_rates': []
        }
        
        # Simple attractor analysis: look for convergence patterns
        for i in range(batch_size):
            z_traj = z_seq[i].cpu().numpy()
            
            if n_obs > 5:
                # Check for convergence to a point (potential fixed point)
                final_positions = z_traj[-5:]  # Last 5 positions
                final_center = np.mean(final_positions, axis=0)
                final_spread = np.std(np.linalg.norm(final_positions - final_center, axis=1))
                
                # If spread is small, consider it an attractor
                if final_spread < 0.1:  # Threshold for convergence
                    attractor_data['fixed_points'].append(final_center)
                    attractor_data['attractor_strengths'].append(1.0 / (final_spread + 1e-8))
                
                # Compute convergence rate
                distances_to_final = np.linalg.norm(z_traj - final_center, axis=1)
                if len(distances_to_final) > 1:
                    # Exponential decay rate estimate
                    log_distances = np.log(distances_to_final + 1e-8)
                    if len(log_distances) > 2:
                        convergence_rate = -(log_distances[-1] - log_distances[0]) / len(log_distances)
                        attractor_data['convergence_rates'].append(convergence_rate)
        
        return attractor_data
    
    def _create_attractor_plots(self, z_seq, attractor_data, epoch):
        """Create attractor analysis visualizations."""
        z_pca_seq, pca = self._prepare_pca_data(z_seq, n_components=2)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Attractor Analysis - Epoch {epoch}', fontsize=16)
        
        # Plot 1: Trajectories with potential attractors
        ax = axes[0, 0]
        batch_size = z_pca_seq.shape[0]
        
        for i in range(min(batch_size, 10)):
            traj = z_pca_seq[i, :, :2]
            ax.plot(traj[:, 0], traj[:, 1], 'o-', alpha=0.7, linewidth=2, markersize=3)
            
            # Mark start and end
            ax.scatter(traj[0, 0], traj[0, 1], color='green', s=60, marker='s', 
                      edgecolor='black', alpha=0.8)
            ax.scatter(traj[-1, 0], traj[-1, 1], color='red', s=60, marker='*', 
                      edgecolor='black', alpha=0.8)
        
        # Mark identified attractors
        if attractor_data['fixed_points']:
            attractor_pca = pca.transform(attractor_data['fixed_points'])
            ax.scatter(attractor_pca[:, 0], attractor_pca[:, 1], 
                      color='red', s=200, marker='X', edgecolor='black', 
                      alpha=0.9, label='Attractors')
            ax.legend()
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Trajectories and Attractors')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Attractor strength distribution
        ax = axes[0, 1]
        if attractor_data['attractor_strengths']:
            ax.hist(attractor_data['attractor_strengths'], bins=15, alpha=0.7, 
                   density=True, color='orange')
            ax.axvline(np.mean(attractor_data['attractor_strengths']), 
                      color='red', linestyle='--', 
                      label=f'Mean: {np.mean(attractor_data["attractor_strengths"]):.2f}')
            ax.set_xlabel('Attractor Strength')
            ax.set_ylabel('Density')
            ax.set_title('Attractor Strength Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No attractors\nidentified', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Attractor Strength Distribution')
        
        # Plot 3: Convergence rate analysis
        ax = axes[1, 0]
        if attractor_data['convergence_rates']:
            ax.hist(attractor_data['convergence_rates'], bins=15, alpha=0.7, 
                   density=True, color='blue')
            ax.axvline(np.mean(attractor_data['convergence_rates']), 
                      color='red', linestyle='--', 
                      label=f'Mean: {np.mean(attractor_data["convergence_rates"]):.3f}')
            ax.axvline(0, color='black', linestyle=':', alpha=0.7, label='No convergence')
            ax.set_xlabel('Convergence Rate')
            ax.set_ylabel('Density')
            ax.set_title('Convergence Rate Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No convergence\ndata available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Convergence Rate Distribution')
        
        # Plot 4: Statistics summary
        ax = axes[1, 1]
        ax.axis('off')
        
        n_attractors = len(attractor_data['fixed_points'])
        mean_strength = np.mean(attractor_data['attractor_strengths']) if attractor_data['attractor_strengths'] else 0.0
        mean_convergence = np.mean(attractor_data['convergence_rates']) if attractor_data['convergence_rates'] else 0.0
        
        stats_text = f"""Attractor Analysis Summary:
        
Number of Attractors: {n_attractors}
Mean Attractor Strength: {mean_strength:.2f}
Mean Convergence Rate: {mean_convergence:.3f}

Interpretation:
- Positive convergence: Stable attractors
- High strength: Strong attraction
- Multiple attractors: Complex dynamics

Sequences Analyzed: {batch_size}
Timesteps: {z_seq.shape[1]}"""
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        ax.set_title('Analysis Summary')
        
        plt.tight_layout()
        
        filename = f'attractor_analysis_epoch_{epoch}.png'
        saved_file = self._safe_save_plt_figure(filename, dpi=200, bbox_inches='tight')
        
        if self.should_log_to_wandb() and saved_file:
            wandb.log({
                "dynamics/attractor_analysis": wandb.Image(saved_file, caption=f"Attractor Analysis - Epoch {epoch}"),
                "metrics/num_attractors": n_attractors,
                "metrics/mean_attractor_strength": mean_strength,
                "metrics/mean_convergence_rate": mean_convergence,
            })
        
        plt.close() 