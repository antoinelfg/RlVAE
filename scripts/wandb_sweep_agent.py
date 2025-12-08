#!/usr/bin/env python3
"""
WandB Sweep Agent for RLVAE Hyperparameter Optimization
========================================================

This script bridges WandB sweeps with Hydra-based run_experiment.py.

How it works:
1. WandB sweep agent passes parameters as key=value pairs (${args_no_hyphens})
2. This script parses those parameters  
3. Maps them to Hydra config overrides
4. Calls run_experiment.py with the overrides
5. run_experiment.py handles its own WandB logging

Usage:
    # Initialize sweep first:
    wandb sweep conf/sweep/stage_c_bayesian.yaml
    
    # Then launch agents:
    wandb agent <entity>/<project>/<sweep_id>
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


def parse_wandb_args() -> Dict[str, Any]:
    """
    Parse sweep parameters passed by WandB via command line.
    
    WandB uses ${args_no_hyphens} which passes params as:
        rhmc_steps=5 n_flows=3 beta=0.01 ...
    """
    params = {}
    
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            # Try to parse as appropriate type
            try:
                # Try int first
                params[key] = int(value)
            except ValueError:
                try:
                    # Try float
                    params[key] = float(value)
                except ValueError:
                    # Keep as string
                    params[key] = value
    
    return params


def build_hydra_overrides(params: Dict[str, Any]) -> list:
    """
    Map parsed sweep parameters to Hydra config overrides.
    Supports the full "chaos sweep" parameter set.
    """
    overrides = []
    
    # ==========================================================================
    # RHMC Configuration
    # ==========================================================================
    if "rhmc_steps" in params:
        overrides.append(f"settings.model.posterior.rhmc_steps={params['rhmc_steps']}")
        overrides.append(f"settings.training.stage_overrides.stage_c.posterior.rhmc_steps={params['rhmc_steps']}")
        
    if "rhmc_step_size" in params:
        overrides.append(f"settings.model.posterior.rhmc_step_size={params['rhmc_step_size']}")
        overrides.append(f"settings.training.stage_overrides.stage_c.posterior.rhmc_step_size={params['rhmc_step_size']}")
        
    if "rhmc_alpha" in params:
        overrides.append(f"settings.model.posterior.rhmc_alpha={params['rhmc_alpha']}")
        overrides.append(f"settings.training.stage_overrides.stage_c.posterior.rhmc_alpha={params['rhmc_alpha']}")
    
    if "rhmc_eps_reg" in params:
        overrides.append(f"settings.model.posterior.rhmc_eps_reg={params['rhmc_eps_reg']}")
        
    if "min_cov_eig" in params:
        overrides.append(f"settings.model.posterior.min_cov_eig={params['min_cov_eig']}")
    
    # ==========================================================================
    # Flow Configuration
    # ==========================================================================
    if "n_flows" in params:
        overrides.append(f"settings.model.n_flows={params['n_flows']}")
        
    if "flow_weight" in params:
        overrides.append(f"settings.model.losses.flow_weight={params['flow_weight']}")
        
    if "flow_loss_mode" in params:
        overrides.append(f"settings.model.losses.flow_loss_mode={params['flow_loss_mode']}")
        
    if "flow_output_clip" in params:
        overrides.append(f"settings.model.flows.output_clip={params['flow_output_clip']}")
    
    # ==========================================================================
    # KL Balance (beta = riemannian_beta)
    # ==========================================================================
    if "beta" in params:
        beta_val = params["beta"]
        overrides.append(f"settings.model.losses.beta={beta_val}")
        overrides.append(f"settings.model.losses.riemannian_beta={beta_val}")
    
    # ==========================================================================
    # Reconstruction Scale
    # ==========================================================================
    if "recon_scale" in params:
        overrides.append(f"settings.model.losses.recon_scale={params['recon_scale']}")
    
    # ==========================================================================
    # Volume/Prior Configuration
    # ==========================================================================
    if "volume_bias_weight" in params:
        overrides.append(f"settings.model.losses.volume_bias_weight={params['volume_bias_weight']}")
        
    if "volume_grad_scale" in params:
        overrides.append(f"settings.model.losses.volume_grad_scale={params['volume_grad_scale']}")
    
    # ==========================================================================
    # Metric Configuration
    # ==========================================================================
    if "temperature_override" in params:
        overrides.append(f"settings.model.metric.temperature_override={params['temperature_override']}")
        
    if "bg_strength" in params:
        overrides.append(f"settings.model.metric.bg_strength={params['bg_strength']}")
        
    if "eig_floor_abs" in params:
        overrides.append(f"settings.model.metric.eig_floor_abs={params['eig_floor_abs']}")
    
    # ==========================================================================
    # Encoder Regularization
    # ==========================================================================
    if "mu_l2_weight" in params:
        overrides.append(f"settings.model.losses.mu_l2_weight={params['mu_l2_weight']}")
        
    if "mu_centroid_weight" in params:
        overrides.append(f"settings.model.losses.mu_centroid_weight={params['mu_centroid_weight']}")
    
    # ==========================================================================
    # Optimizer Configuration
    # ==========================================================================
    if "lr" in params:
        overrides.append(f"settings.training.optimizer.lr={params['lr']}")
        
    if "gradient_clip_val" in params:
        overrides.append(f"settings.training.strategy.gradient_clip_val={params['gradient_clip_val']}")
    
    # ==========================================================================
    # Sweep-specific settings (shorter runs, early stopping)
    # ==========================================================================
    # Shorter epochs for sweep
    overrides.append("settings.training.strategy.max_epochs=75")
    overrides.append("settings.training.stage_overrides.stage_a.epochs=50")
    overrides.append("settings.training.stage_overrides.stage_c.epochs=75")
    
    # Enable early stopping - monitor val_mse (scale-invariant reconstruction)
    overrides.append("settings.training.early_stopping.enabled=true")
    overrides.append("settings.training.early_stopping.patience=20")
    overrides.append("settings.training.early_stopping.monitor=val_mse")
    
    # Ensure WandB is enabled
    overrides.append("wandb.enabled=true")
    overrides.append("wandb.mode=online")
    
    # Add sweep tag
    overrides.append("wandb.tags=[sweep,chaos,v2]")
    
    return overrides


def run_experiment(overrides: list) -> int:
    """
    Run the experiment with the given Hydra overrides.
    """
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_experiment.py"),
    ] + overrides
    
    print(f"\n{'='*80}")
    print("RLVAE Sweep Agent - Running Experiment")
    print(f"{'='*80}")
    print("Hydra overrides:")
    for ov in overrides:
        print(f"  {ov}")
    print(f"{'='*80}\n")
    
    # Run the experiment - inherit environment for WandB sweep context
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=os.environ.copy())
    
    return result.returncode


def main():
    """
    Main entry point for WandB sweep agent.
    """
    print(f"\n{'='*80}")
    print("WandB Sweep Agent Started")
    print(f"{'='*80}")
    print(f"Raw args: {sys.argv[1:]}")
    print(f"WANDB_SWEEP_ID: {os.environ.get('WANDB_SWEEP_ID', 'N/A')}")
    print(f"WANDB_RUN_ID: {os.environ.get('WANDB_RUN_ID', 'N/A')}")
    
    # Parse sweep parameters from command line
    params = parse_wandb_args()
    
    print(f"\nParsed sweep parameters:")
    for k, v in params.items():
        print(f"  {k}: {v} ({type(v).__name__})")
    print(f"{'='*80}\n")
    
    # Build Hydra overrides
    overrides = build_hydra_overrides(params)
    
    # Run the experiment
    return_code = run_experiment(overrides)
    
    if return_code != 0:
        print(f"\n❌ Experiment failed with return code {return_code}")
        sys.exit(return_code)
    
    print("\n✅ Sweep run completed successfully!")


if __name__ == "__main__":
    main()
