#!/usr/bin/env python3
"""
Clean Training Script for RlVAE
================================

This script demonstrates the new WandB-only training options that avoid 
local file clutter while maintaining full experiment tracking.

Usage Examples:
--------------

1. WandB-only mode (no local files at all):
   python run_clean_training.py --loop_mode open --wandb_only

2. Disable local visualization files but keep model checkpoints:
   python run_clean_training.py --loop_mode open --disable_local_files

3. Offline WandB mode (no online syncing, no local run folders):
   python run_clean_training.py --loop_mode open --wandb_offline

4. Fully clean mode (no local files + offline WandB):
   python run_clean_training.py --loop_mode open --wandb_only --wandb_offline

Key Benefits:
- No visualization files cluttering your training directory
- No wandb run folders taking up space
- Full experiment tracking still available on WandB web interface
- Faster training (no local I/O overhead)
- Better for cluster/remote training environments
"""

import subprocess
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Clean RlVAE Training - No Local File Clutter')
    
    # Core training arguments
    parser.add_argument('--loop_mode', choices=['open', 'closed'], required=True,
                       help='Loop mode to train: open or closed')
    parser.add_argument('--n_epochs', type=int, default=10, 
                       help='Number of epochs (default: 10 for quick training)')
    parser.add_argument('--n_train_samples', type=int, default=1000, 
                       help='Number of training samples (default: 1000 for quick training)')
    
    # Clean training flags
    parser.add_argument('--wandb_only', action='store_true', default=True,
                       help='Only log to WandB, no local files (default: True)')
    parser.add_argument('--disable_local_files', action='store_true', default=False,
                       help='Disable local visualization files but keep model checkpoints')
    parser.add_argument('--wandb_offline', action='store_true', default=False,
                       help='Run WandB offline to avoid local run storage')
    
    # Performance flags
    parser.add_argument('--disable_curvature_during_training', action='store_true', default=True,
                       help='Disable expensive curvature computation (default: True)')
    
    args = parser.parse_args()
    
    print("🧹 Clean RlVAE Training")
    print("=" * 50)
    print(f"Loop Mode: {args.loop_mode}")
    print(f"Epochs: {args.n_epochs}")
    print(f"Samples: {args.n_train_samples}")
    print()
    
    # File management options
    if args.wandb_only:
        print("💾 File Management: WandB-ONLY mode")
        print("   ✅ No local visualization files")
        print("   ✅ No local model checkpoints") 
        print("   ✅ All data logged to WandB cloud")
    elif args.disable_local_files:
        print("💾 File Management: HYBRID mode")
        print("   ✅ No local visualization files")
        print("   💾 Model checkpoints saved locally")
        print("   ✅ Visualizations logged to WandB")
    else:
        print("💾 File Management: STANDARD mode")
        print("   💾 All files saved locally + WandB")
    
    if args.wandb_offline:
        print("🌐 WandB Mode: OFFLINE")
        print("   ✅ No local wandb run directories")
        print("   ✅ Logs stored in buffer for later sync")
    else:
        print("🌐 WandB Mode: ONLINE")
        print("   ⚠️  Local wandb run directories will be created")
    
    print()
    
    # Build command
    cmd = [
        'python', 'src/training/train_cyclic_loop_comparison.py',
        '--loop_mode', args.loop_mode,
        '--n_epochs', str(args.n_epochs),
        '--n_train_samples', str(args.n_train_samples),
        '--batch_size', '8',
        '--visualization_frequency', '5',
    ]
    
    # Add clean training flags
    if args.wandb_only:
        cmd.append('--wandb_only')
    if args.disable_local_files:
        cmd.append('--disable_local_files')
    if args.wandb_offline:
        cmd.append('--wandb_offline')
    if args.disable_curvature_during_training:
        cmd.append('--disable_curvature_during_training')
    
    print("🚀 Executing command:")
    print(f"   {' '.join(cmd)}")
    print()
    
    # Run the training
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
        
        if args.wandb_only:
            print("\n📊 Results:")
            print("   Check your WandB dashboard for all visualizations")
            print("   No local files were created (clean workspace!)")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main() 