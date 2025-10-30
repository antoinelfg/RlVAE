#!/usr/bin/env python3
"""
Visualization Script: KL Diagnostics
====================================

Generate visualizations to understand negative KL divergence issues.

This script creates:
1. RHMC trajectory plot in 2D latent space
2. Distance evolution over RHMC steps
3. Mahalanobis heatmap (contribution per eigenvalue)
4. Distribution comparison (empirical vs theoretical)
5. Log-probability breakdown (stacked bar chart)

Usage:
    python scripts/visualize_kl_diagnostics.py --checkpoint /path/to/phase_c_checkpoint.ckpt

Or run in diagnostic mode to collect data:
    RLVAE_DEBUG=1 python scripts/visualize_kl_diagnostics.py --collect-data
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_trajectory_2d(
    z0: np.ndarray,
    zK: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    trajectory: Optional[List[np.ndarray]] = None,
    save_path: str = "trajectory_2d.png"
):
    """
    Plot RHMC trajectory in 2D latent space with confidence ellipse from Σ_μ.
    
    Args:
        z0: Initial samples [B, 2]
        zK: Final samples [B, 2]
        mu: Encoder mean [B, 2] or [1, 2]
        Sigma: Covariance [B, 2, 2] or [1, 2, 2]
        trajectory: Optional list of intermediate z_k [B, 2]
        save_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Use first sample if batch
    if mu.shape[0] > 1:
        mu_plot = mu[0]
        Sigma_plot = Sigma[0]
    else:
        mu_plot = mu.squeeze()
        Sigma_plot = Sigma.squeeze()
    
    # Plot μ
    ax.scatter(mu_plot[0], mu_plot[1], c='red', marker='x', s=200, label='μ (encoder mean)', zorder=10)
    
    # Plot confidence ellipse from Σ_μ (2σ level)
    from matplotlib.patches import Ellipse
    eigvals, eigvecs = np.linalg.eigh(Sigma_plot)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * 2 * np.sqrt(eigvals)  # 2σ ellipse
    
    ellipse = Ellipse(
        mu_plot, width, height, angle=angle,
        facecolor='red', alpha=0.1, edgecolor='red', linewidth=2,
        label='2σ confidence (Σ_μ)'
    )
    ax.add_patch(ellipse)
    
    # Plot z0 (initial samples)
    ax.scatter(z0[:, 0], z0[:, 1], c='blue', marker='o', s=50, alpha=0.6, label='z0 (initial)', zorder=5)
    
    # Plot zK (final samples)
    ax.scatter(zK[:, 0], zK[:, 1], c='green', marker='s', s=50, alpha=0.6, label='zK (after RHMC)', zorder=5)
    
    # Plot trajectories if available
    if trajectory is not None and len(trajectory) > 0:
        for i in range(min(20, z0.shape[0])):  # Plot max 20 trajectories for clarity
            traj_points = np.array([t[i] for t in trajectory])
            ax.plot(traj_points[:, 0], traj_points[:, 1], 'k-', alpha=0.3, linewidth=0.5)
    
    # Connect z0 to zK with arrows
    for i in range(min(20, z0.shape[0])):
        ax.arrow(
            z0[i, 0], z0[i, 1],
            zK[i, 0] - z0[i, 0], zK[i, 1] - z0[i, 1],
            head_width=0.1, head_length=0.05, fc='gray', ec='gray', alpha=0.3, linewidth=0.5
        )
    
    ax.set_xlabel('z[0]')
    ax.set_ylabel('z[1]')
    ax.set_title('RHMC Trajectory in 2D Latent Space')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved trajectory plot to {save_path}")
    plt.close()


def plot_distance_evolution(
    distances_from_mu: List[float],
    distances_from_z0: List[float],
    momentum_norms: List[float],
    save_path: str = "distance_evolution.png"
):
    """
    Plot distance evolution over RHMC steps.
    
    Args:
        distances_from_mu: ||z_k - μ|| at each step
        distances_from_z0: ||z_k - z0|| at each step
        momentum_norms: ||ρ_k|| at each step
        save_path: Output file path
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    steps = list(range(len(distances_from_mu)))
    
    # Distance from μ
    axes[0].plot(steps, distances_from_mu, 'b-o', linewidth=2, markersize=6)
    axes[0].set_ylabel('||z - μ||')
    axes[0].set_title('Distance from Encoder Mean μ')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=distances_from_mu[0], color='r', linestyle='--', alpha=0.5, label='Initial')
    axes[0].legend()
    
    # Distance from z0
    axes[1].plot(steps, distances_from_z0, 'g-o', linewidth=2, markersize=6)
    axes[1].set_ylabel('||z - z0||')
    axes[1].set_title('Drift from Initial Sample z0')
    axes[1].grid(True, alpha=0.3)
    
    # Momentum norm
    axes[2].plot(steps, momentum_norms, 'r-o', linewidth=2, markersize=6)
    axes[2].set_xlabel('RHMC Step k')
    axes[2].set_ylabel('||ρ||')
    axes[2].set_title('Momentum Magnitude')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved distance evolution plot to {save_path}")
    plt.close()


def plot_mahalanobis_heatmap(
    contrib_per_eig: np.ndarray,
    eigenvalues: np.ndarray,
    save_path: str = "mahalanobis_heatmap.png"
):
    """
    Plot heatmap showing contribution of each eigenvalue to Mahalanobis distance.
    
    Args:
        contrib_per_eig: Contributions y²/λ for each dimension [B, D]
        eigenvalues: Eigenvalues of Σ [B, D] or [D]
        save_path: Output file path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean contributions
    mean_contrib = contrib_per_eig.mean(axis=0)
    dimensions = np.arange(len(mean_contrib))
    
    # Bar chart of contributions
    axes[0].bar(dimensions, mean_contrib, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Dimension (Eigenvalue Index)')
    axes[0].set_ylabel('Mean Contribution to Mahalanobis²')
    axes[0].set_title('Contribution per Eigenvalue (y²/λ)')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Eigenvalues
    if eigenvalues.ndim > 1:
        eigvals_plot = eigenvalues[0]
    else:
        eigvals_plot = eigenvalues
    
    axes[1].bar(dimensions, eigvals_plot, color='coral', alpha=0.7)
    axes[1].set_xlabel('Dimension (Eigenvalue Index)')
    axes[1].set_ylabel('Eigenvalue λ')
    axes[1].set_title('Eigenvalues of Σ_μ')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved Mahalanobis heatmap to {save_path}")
    plt.close()


def plot_distribution_comparison(
    samples_diff: np.ndarray,
    Sigma: np.ndarray,
    save_path: str = "distribution_comparison.png"
):
    """
    Overlay empirical (z-μ) distribution with theoretical N(0, Σ).
    
    Args:
        samples_diff: (z - μ) [B, D]
        Sigma: Theoretical covariance [B, D, D] or [D, D]
        save_path: Output file path
    """
    D = samples_diff.shape[1]
    
    if D == 2:
        # 2D case: plot marginals and joint
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Marginal for dim 0
        axes[0].hist(samples_diff[:, 0], bins=30, density=True, alpha=0.6, label='Empirical')
        Sigma_plot = Sigma[0] if Sigma.ndim > 2 else Sigma
        std_0 = np.sqrt(Sigma_plot[0, 0])
        x = np.linspace(samples_diff[:, 0].min(), samples_diff[:, 0].max(), 100)
        from scipy.stats import norm
        axes[0].plot(x, norm.pdf(x, 0, std_0), 'r-', linewidth=2, label='Theoretical N(0, σ₀²)')
        axes[0].set_xlabel('(z - μ)[0]')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Marginal Distribution: Dimension 0')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Marginal for dim 1
        axes[1].hist(samples_diff[:, 1], bins=30, density=True, alpha=0.6, label='Empirical')
        std_1 = np.sqrt(Sigma_plot[1, 1])
        x = np.linspace(samples_diff[:, 1].min(), samples_diff[:, 1].max(), 100)
        axes[1].plot(x, norm.pdf(x, 0, std_1), 'r-', linewidth=2, label='Theoretical N(0, σ₁²)')
        axes[1].set_xlabel('(z - μ)[1]')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Marginal Distribution: Dimension 1')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Joint 2D
        axes[2].scatter(samples_diff[:, 0], samples_diff[:, 1], alpha=0.5, s=20, label='Empirical')
        
        # Confidence ellipse
        from matplotlib.patches import Ellipse
        eigvals, eigvecs = np.linalg.eigh(Sigma_plot)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * 2 * np.sqrt(eigvals)  # 2σ
        
        ellipse = Ellipse(
            (0, 0), width, height, angle=angle,
            facecolor='none', edgecolor='red', linewidth=2,
            label='Theoretical 2σ'
        )
        axes[2].add_patch(ellipse)
        axes[2].set_xlabel('(z - μ)[0]')
        axes[2].set_ylabel('(z - μ)[1]')
        axes[2].set_title('Joint Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].axis('equal')
    else:
        # Higher-dimensional: plot marginals only
        ncols = min(D, 4)
        nrows = (D + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        axes = axes.flatten() if D > 1 else [axes]
        
        Sigma_plot = Sigma[0] if Sigma.ndim > 2 else Sigma
        
        for i in range(D):
            axes[i].hist(samples_diff[:, i], bins=30, density=True, alpha=0.6, label='Empirical')
            std_i = np.sqrt(Sigma_plot[i, i])
            x = np.linspace(samples_diff[:, i].min(), samples_diff[:, i].max(), 100)
            from scipy.stats import norm
            axes[i].plot(x, norm.pdf(x, 0, std_i), 'r-', linewidth=2, label=f'N(0, σ{i}²)')
            axes[i].set_xlabel(f'(z - μ)[{i}]')
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'Dimension {i}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(D, len(axes)):
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved distribution comparison to {save_path}")
    plt.close()


def plot_logprob_breakdown(
    quad_term: float,
    vol_term: float,
    const_term: float,
    log_p: float,
    save_path: str = "logprob_breakdown.png"
):
    """
    Stacked bar chart of log-probability components.
    
    Args:
        quad_term: Quadratic term -½(z-μ)ᵀΣ⁻¹(z-μ)
        vol_term: Volume term -½log|Σ|
        const_term: Constant term -½D log(2π)
        log_p: Prior log p(z)
        save_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Components
    components = {
        'Quadratic\n-½(z-μ)ᵀΣ⁻¹(z-μ)': quad_term,
        'Volume\n-½log|Σ|': vol_term,
        'Constant\n-½D log(2π)': const_term,
    }
    
    log_q = quad_term + vol_term + const_term
    kl = log_q - log_p
    
    # Bar positions
    x_pos = [0, 1, 2, 3.5, 5]
    labels = list(components.keys()) + ['log q\n(total)', 'KL\n(log q - log p)']
    values = list(components.values()) + [log_q, kl]
    colors = ['steelblue', 'coral', 'lightgreen', 'purple', 'red' if kl < 0 else 'green']
    
    bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (pos, val) in enumerate(zip(x_pos, values)):
        ax.text(pos, val + 0.1 if val > 0 else val - 0.3, f'{val:.2f}',
                ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Add log_p reference
    ax.axhline(y=log_p, color='blue', linestyle='--', linewidth=2, label=f'log p(z) = {log_p:.2f}')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Log-Probability')
    ax.set_title('Log-Probability Breakdown\n(KL = log q - log p)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight KL sign
    if kl < 0:
        ax.text(5, kl - 0.5, '⚠️ NEGATIVE KL!', ha='center', fontsize=12, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved log-probability breakdown to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize KL diagnostics')
    parser.add_argument('--checkpoint', type=str, help='Path to Phase C checkpoint')
    parser.add_argument('--collect-data', action='store_true', help='Run diagnostic mode to collect data')
    parser.add_argument('--output-dir', type=str, default='diagnostic_plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("KL DIAGNOSTICS VISUALIZATION")
    print("="*80)
    print()
    
    if args.collect_data:
        print("Data collection mode enabled.")
        print("Run this script with RLVAE_DEBUG=1 during training to collect diagnostic data.")
        print("Then use the generated data to create visualizations.")
        print()
        print("Example workflow:")
        print("  1. RLVAE_DEBUG=1 python run_experiment.py experiment=phase_c > diagnostic_log.txt")
        print("  2. python scripts/visualize_kl_diagnostics.py --parse-log diagnostic_log.txt")
        print()
    else:
        print("Manual visualization mode.")
        print("This script provides visualization functions for diagnostic data.")
        print()
        print("To use interactively, import functions and call with your data:")
        print()
        print("  from scripts.visualize_kl_diagnostics import plot_trajectory_2d, ...")
        print("  plot_trajectory_2d(z0, zK, mu, Sigma, trajectory, save_path='my_plot.png')")
        print()
        print("Or implement custom data collection and call these functions.")
        print()
    
    # Example usage with dummy data
    print("Example: Generating sample plots with dummy data...")
    
    # Generate dummy 2D data
    B, D = 50, 2
    mu = np.zeros((1, D))
    Sigma = np.eye(D)
    Sigma[0, 0] = 2.0
    Sigma[1, 1] = 0.5
    
    z0 = np.random.multivariate_normal(mu[0], Sigma, B)
    zK = z0 + np.random.randn(B, D) * 0.2  # Small drift
    
    # Plot trajectory
    plot_trajectory_2d(z0, zK, mu, np.array([Sigma]), save_path=str(output_dir / "example_trajectory.png"))
    
    # Plot distance evolution
    distances_mu = [1.0, 1.1, 1.15, 1.2, 1.18]
    distances_z0 = [0.0, 0.1, 0.15, 0.2, 0.18]
    momentum = [1.5, 1.4, 1.3, 1.35, 1.3]
    plot_distance_evolution(distances_mu, distances_z0, momentum, save_path=str(output_dir / "example_distance.png"))
    
    # Plot Mahalanobis heatmap
    contrib = np.random.rand(B, D) * 2
    eigvals = np.array([2.0, 0.5])
    plot_mahalanobis_heatmap(contrib, eigvals, save_path=str(output_dir / "example_mahalanobis.png"))
    
    # Plot distribution comparison
    samples_diff = z0  # Already centered at 0 for example
    plot_distribution_comparison(samples_diff, Sigma, save_path=str(output_dir / "example_distribution.png"))
    
    # Plot log-prob breakdown
    quad = -2.5
    vol = -0.5
    const = -1.8
    log_p = 0.5
    plot_logprob_breakdown(quad, vol, const, log_p, save_path=str(output_dir / "example_logprob.png"))
    
    print(f"\n✓ Example plots saved to {output_dir}/")
    print()
    print("="*80)


if __name__ == "__main__":
    main()

