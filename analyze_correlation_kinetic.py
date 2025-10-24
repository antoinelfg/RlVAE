#!/usr/bin/env python3
"""
Analysis script for correlation and kinetic energy patterns in RHMC training.
Checks if correlation climbs > 0.8 and delta_kin settles < 0.2 for posterior-prior resemblance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def analyze_correlation_kinetic_patterns(csv_path):
    """Analyze correlation and kinetic energy patterns from latent diagnostics."""
    
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Extract relevant columns
    steps = df['step'].values
    corr_zS_mu_max_eig = df['corr_zS_mu_max_eig'].values
    delta_kin_mean = -df['delta_kin_mean'].values  # INVERT SIGNS: RHMC steps go opposite direction
    
    # Create the analysis plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Correlation over time
    ax1.plot(steps, corr_zS_mu_max_eig, 'b-', linewidth=2, label='corr(zS-mu, max_eig)')
    ax1.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='Target: > 0.8')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Correlation')
    ax1.set_title('Correlation between ||zS-mu|| and max_eig(G(zS))')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Delta kinetic energy over time (SIGN INVERTED)
    ax2.plot(steps, delta_kin_mean, 'g-', linewidth=2, label='delta_kin_mean (sign inverted)')
    ax2.axhline(y=0.2, color='r', linestyle='--', alpha=0.7, label='Target: < 0.2')
    ax2.axhline(y=-0.2, color='r', linestyle='--', alpha=0.7, label='Target: > -0.2')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Delta Kinetic Energy (Corrected)')
    ax2.set_title('Kinetic Energy Change (delta_kin_mean) - RHMC Sign Corrected')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Combined analysis
    ax3.scatter(corr_zS_mu_max_eig, delta_kin_mean, c=steps, cmap='viridis', alpha=0.7, s=50)
    ax3.axvline(x=0.8, color='r', linestyle='--', alpha=0.7, label='Corr > 0.8')
    ax3.axhline(y=0.2, color='r', linestyle='--', alpha=0.7, label='|Delta_kin| < 0.2')
    ax3.axhline(y=-0.2, color='r', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Correlation')
    ax3.set_ylabel('Delta Kinetic Energy')
    ax3.set_title('Correlation vs Delta Kinetic Energy (colored by step)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add colorbar for the scatter plot
    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label('Training Step')
    
    plt.tight_layout()
    
    # Analysis
    print("=== CORRELATION AND KINETIC ENERGY ANALYSIS ===")
    print(f"Total steps analyzed: {len(steps)}")
    print(f"Step range: {steps[0]} to {steps[-1]}")
    print()
    
    # Check correlation patterns
    high_corr_steps = np.where(corr_zS_mu_max_eig > 0.8)[0]
    print(f"Steps with correlation > 0.8: {len(high_corr_steps)}")
    if len(high_corr_steps) > 0:
        print(f"  First occurrence at step: {steps[high_corr_steps[0]]}")
        print(f"  Last occurrence at step: {steps[high_corr_steps[-1]]}")
        print(f"  Max correlation: {np.max(corr_zS_mu_max_eig):.4f}")
    print()
    
    # Check kinetic energy patterns (SIGN INVERTED)
    low_kin_steps = np.where(np.abs(delta_kin_mean) < 0.2)[0]
    print(f"Steps with |delta_kin| < 0.2 (SIGN CORRECTED): {len(low_kin_steps)}")
    if len(low_kin_steps) > 0:
        print(f"  First occurrence at step: {steps[low_kin_steps[0]]}")
        print(f"  Last occurrence at step: {steps[low_kin_steps[-1]]}")
        print(f"  Min |delta_kin|: {np.min(np.abs(delta_kin_mean)):.4f}")
    print()
    
    # Check for posterior-prior resemblance conditions
    posterior_prior_conditions = (corr_zS_mu_max_eig > 0.8) & (np.abs(delta_kin_mean) < 0.2)
    posterior_prior_steps = np.where(posterior_prior_conditions)[0]
    
    print("=== POSTERIOR-PRIOR RESEMBLANCE ANALYSIS (SIGN CORRECTED) ===")
    print(f"Steps meeting BOTH conditions (corr > 0.8 AND |delta_kin| < 0.2): {len(posterior_prior_steps)}")
    if len(posterior_prior_steps) > 0:
        print(f"  First occurrence at step: {steps[posterior_prior_steps[0]]}")
        print(f"  Last occurrence at step: {steps[posterior_prior_steps[-1]]}")
        print(f"  Percentage of training: {len(posterior_prior_steps)/len(steps)*100:.1f}%")
        
        # Check if conditions are met in the latter part of training
        latter_half = len(steps) // 2
        latter_conditions = posterior_prior_conditions[latter_half:]
        print(f"  In latter half of training: {np.sum(latter_conditions)} steps")
    else:
        print("  No steps meet both conditions simultaneously")
    print()
    
    # Statistical summary
    print("=== STATISTICAL SUMMARY (SIGN CORRECTED) ===")
    print(f"Correlation - Mean: {np.mean(corr_zS_mu_max_eig):.4f}, Std: {np.std(corr_zS_mu_max_eig):.4f}")
    print(f"Correlation - Min: {np.min(corr_zS_mu_max_eig):.4f}, Max: {np.max(corr_zS_mu_max_eig):.4f}")
    print(f"Delta_kin (CORRECTED) - Mean: {np.mean(delta_kin_mean):.4f}, Std: {np.std(delta_kin_mean):.4f}")
    print(f"Delta_kin (CORRECTED) - Min: {np.min(delta_kin_mean):.4f}, Max: {np.max(delta_kin_mean):.4f}")
    print(f"|Delta_kin| (CORRECTED) - Mean: {np.mean(np.abs(delta_kin_mean)):.4f}, Std: {np.std(np.abs(delta_kin_mean)):.4f}")
    
    # Save the plot
    output_path = '/home/alaforgu/scratch/longitudinal_experiments/RlVAE/correlation_kinetic_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    return df, posterior_prior_steps

if __name__ == "__main__":
    csv_path = '/home/alaforgu/scratch/longitudinal_experiments/RlVAE/outputs/probes/latent_diagnostics.csv'
    df, posterior_prior_steps = analyze_correlation_kinetic_patterns(csv_path)
