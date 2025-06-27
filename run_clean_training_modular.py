#!/usr/bin/env python3
"""
Modular Clean Training Script for RlVAE
=======================================

This script demonstrates the new modular visualization system with 
configurable complexity levels and better organization.

Usage Examples:
--------------

1. Minimal visualizations (fastest):
   python run_clean_training_modular.py --loop_mode open --viz_level minimal

2. Standard visualizations (recommended):
   python run_clean_training_modular.py --loop_mode open --viz_level standard

3. Advanced visualizations (interactive elements):
   python run_clean_training_modular.py --loop_mode open --viz_level advanced

4. Full visualizations (everything enabled):
   python run_clean_training_modular.py --loop_mode open --viz_level full

5. Custom configuration:
   python run_clean_training_modular.py --loop_mode open --enable_basic --enable_manifold --disable_flow

Key Benefits:
- Modular visualization system with clear separation of concerns
- Configurable complexity levels for different use cases
- Better performance control and resource management
- Organized code structure for easier maintenance
- Dynamic visualization control during training
"""

import subprocess
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Modular Clean RlVAE Training')
    
    # Core training arguments
    parser.add_argument('--loop_mode', choices=['open', 'closed'], default='open',
                       help='Loop mode to train: open or closed')
    parser.add_argument('--n_epochs', type=int, default=10, 
                       help='Number of epochs (default: 10)')
    parser.add_argument('--n_train_samples', type=int, default=1000, 
                       help='Number of training samples (default: 1000)')
    
    # Visualization configuration
    parser.add_argument('--viz_level', choices=['minimal', 'basic', 'standard', 'advanced', 'full'],
                       default='standard', help='Visualization complexity level (default: standard)')
    
    # Module-specific controls
    parser.add_argument('--enable_basic', action='store_true', 
                       help='Enable basic visualizations')
    parser.add_argument('--enable_manifold', action='store_true',
                       help='Enable manifold visualizations')
    parser.add_argument('--enable_interactive', action='store_true',
                       help='Enable interactive visualizations') 
    parser.add_argument('--enable_flow', action='store_true',
                       help='Enable flow analysis visualizations')
    
    # Disable options
    parser.add_argument('--disable_basic', action='store_true',
                       help='Disable basic visualizations')
    parser.add_argument('--disable_manifold', action='store_true',
                       help='Disable manifold visualizations')
    parser.add_argument('--disable_interactive', action='store_true',
                       help='Disable interactive visualizations')
    parser.add_argument('--disable_flow', action='store_true',
                       help='Disable flow analysis visualizations')
    
    # Clean training flags
    parser.add_argument('--wandb_only', action='store_true', default=True,
                       help='Only log to WandB, no local files (default: True)')
    parser.add_argument('--wandb_offline', action='store_true', default=False,
                       help='Run WandB offline to avoid local run storage')
    
    # Performance flags
    parser.add_argument('--disable_curvature_during_training', action='store_true', default=True,
                       help='Disable expensive curvature computation (default: True)')
    
    args = parser.parse_args()
    
    print("🧹 Modular Clean RlVAE Training")
    print("=" * 60)
    print(f"Loop Mode: {args.loop_mode}")
    print(f"Epochs: {args.n_epochs}")
    print(f"Samples: {args.n_train_samples}")
    print(f"Visualization Level: {args.viz_level}")
    print()
    
    # Visualization configuration display
    viz_levels = {
        'minimal': "⚡ MINIMAL - Only basic cyclicity (fastest)",
        'basic': "📊 BASIC - Essential visualizations",
        'standard': "🎨 STANDARD - Most common visualizations (recommended)",
        'advanced': "🌟 ADVANCED - Includes interactive elements",
        'full': "🚀 FULL - All visualizations (slowest, most detailed)"
    }
    
    print("🎨 Visualization Configuration:")
    print(f"   {viz_levels[args.viz_level]}")
    
    # Show module status
    modules = {
        'basic': not args.disable_basic and (args.enable_basic or args.viz_level in ['basic', 'standard', 'advanced', 'full']),
        'manifold': not args.disable_manifold and (args.enable_manifold or args.viz_level in ['basic', 'standard', 'advanced', 'full']),
        'interactive': not args.disable_interactive and (args.enable_interactive or args.viz_level in ['advanced', 'full']),
        'flow': not args.disable_flow and (args.enable_flow or args.viz_level in ['standard', 'advanced', 'full'])
    }
    
    print("   📦 Enabled modules:")
    for module, enabled in modules.items():
        status = "✅" if enabled else "❌"
        print(f"      {status} {module}")
    print()
    
    # File management options
    if args.wandb_only:
        print("💾 File Management: WandB-ONLY mode")
        print("   ✅ No local visualization files")
        print("   ✅ No local model checkpoints") 
        print("   ✅ All data logged to WandB cloud")
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
    
    # Build command - using the existing training script for now
    cmd = [
        'python', 'src/training/train_cyclic_loop_comparison.py',
        '--loop_mode', args.loop_mode,
        '--n_epochs', str(args.n_epochs),
        '--n_train_samples', str(args.n_train_samples),
        '--batch_size', '8',
        '--visualization_frequency', '5'
    ]
    
    # Add module-specific flags
    if args.enable_basic:
        cmd.append('--enable_basic')
    if args.enable_manifold:
        cmd.append('--enable_manifold')
    if args.enable_interactive:
        cmd.append('--enable_interactive')
    if args.enable_flow:
        cmd.append('--enable_flow')
        
    if args.disable_basic:
        cmd.append('--disable_basic')
    if args.disable_manifold:
        cmd.append('--disable_manifold')
    if args.disable_interactive:
        cmd.append('--disable_interactive')
    if args.disable_flow:
        cmd.append('--disable_flow')
    
    # Add clean training flags
    if args.wandb_only:
        cmd.append('--wandb_only')
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
        
        print("\n📊 Results:")
        print("   Check your WandB dashboard for all visualizations")
        print(f"   Visualization level used: {args.viz_level}")
        if args.wandb_only:
            print("   No local files were created (clean workspace!)")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        print("\n💡 Troubleshooting tips:")
        print("   - Try a lower visualization level (--viz_level minimal)")
        print("   - Check if all dependencies are installed")
        print("   - Reduce sample size for faster debugging")
        sys.exit(1)

if __name__ == "__main__":
    main() 