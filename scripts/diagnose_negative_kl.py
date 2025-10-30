#!/usr/bin/env python3
"""
Diagnostic Script: Negative KL Root Cause Analysis
===================================================

Runs a minimal Phase C training with comprehensive diagnostics to understand
why log_q(z0) is too negative, causing KL divergence < 0.

Usage:
    RLVAE_DEBUG=1 python scripts/diagnose_negative_kl.py

This script:
1. Loads a small batch of data (1-2 batches)
2. Initializes Phase C model with RHMC posterior
3. Runs forward pass with full diagnostic logging enabled
4. Analyzes and summarizes findings
5. Provides recommendations based on observed patterns
"""

import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def analyze_hypothesis(diagnostics: dict) -> dict:
    """
    Analyze diagnostics and test hypotheses about negative KL.
    
    Returns:
        dict with hypothesis test results and recommendations
    """
    results = {
        'hypothesis_a': {'name': 'Σ_μ is too small (under-dispersed)', 'evidence': [], 'verdict': 'UNKNOWN'},
        'hypothesis_b': {'name': 'Σ_μ has wrong shape (anisotropy mismatch)', 'evidence': [], 'verdict': 'UNKNOWN'},
        'hypothesis_c': {'name': 'RHMC pushes z away from μ', 'evidence': [], 'verdict': 'UNKNOWN'},
        'hypothesis_d': {'name': 'Gaussian posterior is fundamentally wrong', 'evidence': [], 'verdict': 'UNKNOWN'},
        'recommendation': 'UNKNOWN'
    }
    
    # Hypothesis A: Check if ||z0-μ|| >> √(tr(Σ))
    if 'distance_ratio' in diagnostics:
        ratio = diagnostics['distance_ratio']
        results['hypothesis_a']['evidence'].append(f"Distance ratio (actual/expected): {ratio:.2f}")
        if ratio > 1.5:
            results['hypothesis_a']['verdict'] = 'LIKELY'
            results['hypothesis_a']['evidence'].append("z0 is FAR from μ (>1.5× expected)")
        elif ratio < 0.5:
            results['hypothesis_a']['verdict'] = 'LIKELY'
            results['hypothesis_a']['evidence'].append("Σ_μ is TOO LARGE (z0 too close)")
        else:
            results['hypothesis_a']['verdict'] = 'UNLIKELY'
    
    # Hypothesis B: Check anisotropy
    if 'anisotropy_ratio' in diagnostics:
        ratio = diagnostics['anisotropy_ratio']
        results['hypothesis_b']['evidence'].append(f"Anisotropy ratio (λ_max/λ_min): {ratio:.2f}")
        if ratio > 10:
            results['hypothesis_b']['verdict'] = 'LIKELY'
            results['hypothesis_b']['evidence'].append("High anisotropy detected")
        else:
            results['hypothesis_b']['verdict'] = 'UNLIKELY'
    
    # Hypothesis C: Check RHMC drift
    if 'rhmc_drift' in diagnostics:
        drift = diagnostics['rhmc_drift']
        results['hypothesis_c']['evidence'].append(f"RHMC net distance change: {drift:+.4f}")
        if drift > 0.1:
            results['hypothesis_c']['verdict'] = 'LIKELY'
            results['hypothesis_c']['evidence'].append("RHMC moves AWAY from μ")
        elif drift < -0.1:
            results['hypothesis_c']['verdict'] = 'UNLIKELY'
            results['hypothesis_c']['evidence'].append("RHMC moves TOWARD μ (good)")
        else:
            results['hypothesis_c']['verdict'] = 'UNLIKELY'
    
    # Hypothesis D: Chi-squared test
    if 'chi_sq_deviation_sigmas' in diagnostics:
        dev_sigmas = diagnostics['chi_sq_deviation_sigmas']
        results['hypothesis_d']['evidence'].append(f"Chi-squared deviation: {dev_sigmas:.2f} σ")
        if abs(dev_sigmas) > 2.0:
            results['hypothesis_d']['verdict'] = 'LIKELY'
            results['hypothesis_d']['evidence'].append("Significant deviation from χ²(D) distribution")
        else:
            results['hypothesis_d']['verdict'] = 'UNLIKELY'
    
    # Generate recommendation
    likely_hypotheses = [k for k, v in results.items() if k.startswith('hypothesis_') and v['verdict'] == 'LIKELY']
    
    if 'hypothesis_a' in likely_hypotheses:
        if diagnostics.get('distance_ratio', 1.0) > 1.5:
            results['recommendation'] = "Increase rhmc_alpha (try 1.0, 2.0, or 5.0) to enlarge Σ_μ"
        else:
            results['recommendation'] = "Decrease rhmc_alpha (try 0.2, 0.1) to shrink Σ_μ"
    elif 'hypothesis_b' in likely_hypotheses:
        results['recommendation'] = "Try sigma_normalization_mode: 'none' or 'trace' to adjust Σ_μ shape"
    elif 'hypothesis_c' in likely_hypotheses:
        results['recommendation'] = "Reduce rhmc_step_size or rhmc_steps to limit RHMC drift"
    elif 'hypothesis_d' in likely_hypotheses:
        results['recommendation'] = "Consider non-Gaussian posterior (flow-based q) or different prior"
    else:
        results['recommendation'] = "No clear hypothesis. Review full diagnostics for subtle issues."
    
    return results


def extract_diagnostics_from_logs(log_file: str = None) -> dict:
    """
    Parse terminal output to extract diagnostic values.
    
    In real usage, this would parse the logs. For now, we return a placeholder
    that will be populated by the actual training run.
    """
    # This is a placeholder - actual implementation would parse logs
    # For now, users should manually review the terminal output
    return {}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main diagnostic routine.
    """
    print("="*80)
    print("NEGATIVE KL DIAGNOSTIC SCRIPT")
    print("="*80)
    print()
    
    # Force debug mode
    os.environ['RLVAE_DEBUG'] = '1'
    print("✓ Enabled RLVAE_DEBUG=1 for comprehensive diagnostics")
    
    # Check if we're using RHMC
    if 'riemannian_rhmc' not in cfg.model.get('posterior', {}).get('type', ''):
        print("⚠️  WARNING: Model is not using riemannian_rhmc posterior!")
        print("    This diagnostic script is designed for RHMC-based Phase C training.")
        print()
    
    # Print current RHMC configuration
    print("\n" + "="*80)
    print("CURRENT RHMC CONFIGURATION")
    print("="*80)
    
    if 'posterior' in cfg.model:
        posterior_cfg = cfg.model.posterior
        print(f"  rhmc_alpha:                {posterior_cfg.get('rhmc_alpha', 'N/A')}")
        print(f"  rhmc_steps:                {posterior_cfg.get('rhmc_steps', 'N/A')}")
        print(f"  rhmc_step_size:            {posterior_cfg.get('rhmc_step_size', 'N/A')}")
        print(f"  rhmc_eps_reg:              {posterior_cfg.get('rhmc_eps_reg', 'N/A')}")
        print(f"  sigma_normalization_mode:  {posterior_cfg.get('sigma_normalization_mode', 'N/A')}")
        print(f"  initial_target_radius:     {posterior_cfg.get('initial_target_radius', 'N/A')}")
    
    print("\n" + "="*80)
    print("INSTRUCTIONS")
    print("="*80)
    print()
    print("1. This script will attempt to run 1-2 batches of Phase C training")
    print("2. Full diagnostics will be printed to terminal")
    print("3. Look for the following diagnostic sections:")
    print("   - [INITIAL SAMPLING DIAGNOSTICS]: z0 properties and distances")
    print("   - [RHMC TRAJECTORY DIAGNOSTICS]: How RHMC moves from z0 to zK")
    print("   - [LOG_Q_RIEM FULL DECOMPOSITION]: Breakdown of log_q components")
    print()
    print("4. Key metrics to watch:")
    print("   - Distance ratio (actual ||z0-μ|| / expected): Should be ~1.0")
    print("   - Mahalanobis²: Should be close to latent_dim for χ²(D) fit")
    print("   - RHMC drift: Positive means moving away from μ (BAD)")
    print("   - log_q total: Should be ~-2 to -4 for 2D latent space")
    print()
    print("5. After reviewing diagnostics, this script will provide recommendations")
    print()
    print("="*80)
    print()
    
    # Try to run minimal training
    try:
        import pytorch_lightning as pl
        from src.training.lightning_module import RLVAELightningModule
        from src.data.data_module import get_data_module
        
        print("Loading data...")
        data_module = get_data_module(cfg)
        data_module.setup('fit')
        
        # Get one batch
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        
        print(f"Loaded 1 batch: {batch[0].shape if isinstance(batch, tuple) else batch.shape}")
        
        # Initialize model
        print("\nInitializing Phase C model with RHMC...")
        model = RLVAELightningModule(cfg)
        
        # Run one forward pass
        print("\nRunning forward pass with full diagnostics...\n")
        print("="*80)
        print("DIAGNOSTIC OUTPUT (review carefully)")
        print("="*80)
        
        with torch.no_grad():
            if isinstance(batch, tuple):
                x, *_ = batch
            else:
                x = batch
            
            # Move to device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            x = x.to(device)
            model = model.to(device)
            
            # Forward pass - this will trigger all diagnostics
            output = model(x)
        
        print("\n" + "="*80)
        print("DIAGNOSTIC OUTPUT COMPLETE")
        print("="*80)
        print()
        
        print("Manual Analysis Required:")
        print("-------------------------")
        print("Please review the diagnostic output above and identify:")
        print()
        print("1. Is ||z0-μ|| much larger or smaller than expected?")
        print("   → Hypothesis A: Σ_μ scale issue")
        print()
        print("2. Is the anisotropy ratio (λ_max/λ_min) very large?")
        print("   → Hypothesis B: Σ_μ shape issue")
        print()
        print("3. Does RHMC move away from μ (positive drift)?")
        print("   → Hypothesis C: RHMC dynamics issue")
        print()
        print("4. Does Mahalanobis² deviate significantly from D?")
        print("   → Hypothesis D: Gaussian posterior mismatch")
        print()
        print("="*80)
        print()
        
        print("Next Steps:")
        print("-----------")
        print("Based on the diagnostics, try one of these fixes:")
        print()
        print("• If z0 is too far from μ:")
        print("    rhmc_alpha: 1.0  # or 2.0, 5.0 (increase)")
        print()
        print("• If z0 is too close to μ:")
        print("    rhmc_alpha: 0.2  # or 0.1 (decrease)")
        print()
        print("• If anisotropy is too high:")
        print("    sigma_normalization_mode: 'trace'  # or 'none'")
        print()
        print("• If RHMC drifts away:")
        print("    rhmc_step_size: 0.01  # reduce from 0.02")
        print("    rhmc_steps: 2  # reduce from 4")
        print()
        print("• If Gaussian is fundamentally wrong:")
        print("    Consider switching to a different prior or posterior architecture")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during diagnostic run: {e}")
        print(f"\nTraceback:")
        import traceback
        traceback.print_exc()
        print()
        print("Note: This is expected if dependencies are missing or config is incomplete.")
        print("      You can still use this script as a template for manual testing.")
    
    print("="*80)
    print("DIAGNOSTIC SCRIPT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

