#!/usr/bin/env python3
"""
Enhanced Analysis for Existing Checkpoints

This script runs comprehensive enhanced analysis on existing RLVAE checkpoints.
It's useful for analyzing models that were already trained.

Usage:
    python scripts/analyze_existing_checkpoint.py --checkpoint_path path/to/checkpoint.ckpt --config_path path/to/config.yaml
"""

import os
import sys
import argparse
import glob
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def find_latest_checkpoint(run_name_pattern):
    """Find the latest checkpoint for a given run pattern."""
    checkpoint_pattern = f"outputs/{run_name_pattern}/checkpoints/*.ckpt"
    checkpoint_files = sorted(glob.glob(checkpoint_pattern), key=os.path.getmtime, reverse=True)
    
    if not checkpoint_files:
        return None
    
    return checkpoint_files[0]

def find_latest_config(run_name_pattern):
    """Find the latest config for a given run pattern."""
    config_pattern = f"outputs/{run_name_pattern}/configs/*.yaml"
    config_files = sorted(glob.glob(config_pattern), key=os.path.getmtime, reverse=True)
    
    if not config_files:
        return None
    
    return config_files[0]

def main():
    parser = argparse.ArgumentParser(description="Run enhanced analysis on existing checkpoint")
    parser.add_argument("--checkpoint_path", type=str, 
                       help="Path to checkpoint file (or use --run_name to auto-find)")
    parser.add_argument("--config_path", type=str,
                       help="Path to config file (or use --run_name to auto-find)")
    parser.add_argument("--run_name", type=str,
                       help="Run name pattern to auto-find latest checkpoint and config")
    parser.add_argument("--output_dir", type=str, default="enhanced_analysis_outputs",
                       help="Output directory for analysis results")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples for generation analysis")
    parser.add_argument("--num_cycles", type=int, default=50,
                       help="Number of cycles for inference analysis")
    parser.add_argument("--geodesic_steps", type=int, default=20,
                       help="Number of steps for geodesic interpolation")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--log_to_wandb", action="store_true",
                       help="Log results to wandb")
    
    args = parser.parse_args()
    
    # Auto-find checkpoint and config if run_name is provided
    if args.run_name:
        print(f"🔍 Searching for latest checkpoint and config for run: {args.run_name}")
        
        checkpoint_path = find_latest_checkpoint(args.run_name)
        if not checkpoint_path:
            print(f"❌ No checkpoint found for pattern: {args.run_name}")
            return
        
        config_path = find_latest_config(args.run_name)
        if not config_path:
            print(f"❌ No config found for pattern: {args.run_name}")
            return
        
        print(f"✅ Found checkpoint: {checkpoint_path}")
        print(f"✅ Found config: {config_path}")
        
        args.checkpoint_path = checkpoint_path
        args.config_path = config_path
    
    # Validate required arguments
    if not args.checkpoint_path or not args.config_path:
        print("❌ Both --checkpoint_path and --config_path are required (or use --run_name)")
        return
    
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint not found: {args.checkpoint_path}")
        return
    
    if not os.path.exists(args.config_path):
        print(f"❌ Config not found: {args.config_path}")
        return
    
    # Run enhanced analysis
    analysis_cmd = [
        sys.executable, "scripts/run_enhanced_analysis.py",
        "--checkpoint_path", args.checkpoint_path,
        "--config_path", args.config_path,
        "--output_dir", args.output_dir,
        "--num_samples", str(args.num_samples),
        "--num_cycles", str(args.num_cycles),
        "--geodesic_steps", str(args.geodesic_steps),
        "--batch_size", str(args.batch_size),
        "--device", args.device
    ]
    
    if args.log_to_wandb:
        analysis_cmd.append("--log_to_wandb")
    
    # Set PYTHONPATH to project root for subprocess
    import subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent.resolve())
    
    print(f"🚀 Running enhanced analysis...")
    print(f"Command: {' '.join(analysis_cmd)}")
    
    try:
        subprocess.run(analysis_cmd, check=True, env=env)
        print(f"✅ Enhanced analysis completed successfully!")
        print(f"📊 Results saved to: {args.output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Enhanced analysis failed with error: {e}")
        return

if __name__ == "__main__":
    main() 