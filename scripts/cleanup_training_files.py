#!/usr/bin/env python3
"""
Cleanup Training Files
======================

This script removes local training files and wandb directories to clean up
your workspace for WandB-only training.

What it removes:
- Visualization files (.png, .html)
- Model checkpoints (.pt)
- WandB run directories
- Temporary files

What it keeps:
- Source code
- Data files
- Configuration files
- README and documentation
"""

import os
import shutil
import glob
from pathlib import Path

def cleanup_training_files():
    """Clean up training files and directories."""
    print("🧹 Cleaning up training files...")
    
    # Current directory
    base_dir = Path(".")
    training_dir = base_dir / "src" / "training"
    
    # Files to remove (patterns)
    file_patterns = [
        "*.png",
        "*.html", 
        "*.pt",
        "*.jpg",
        "*.jpeg",
        "*.svg"
    ]
    
    # Directories to remove
    dir_patterns = [
        "wandb",
        "html_latent_images_*",
        "__pycache__"
    ]
    
    total_removed = 0
    total_size = 0
    
    # Remove files matching patterns
    for pattern in file_patterns:
        # Check in training directory
        files = list(training_dir.glob(pattern))
        for file_path in files:
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"   🗑️  Removing file: {file_path.relative_to(base_dir)} ({size/1024:.1f} KB)")
                file_path.unlink()
                total_removed += 1
                total_size += size
        
        # Check in root directory  
        files = list(base_dir.glob(pattern))
        for file_path in files:
            if file_path.is_file() and file_path.name not in ['requirements.txt', 'setup.py', 'config.py']:
                size = file_path.stat().st_size
                print(f"   🗑️  Removing file: {file_path.relative_to(base_dir)} ({size/1024:.1f} KB)")
                file_path.unlink()
                total_removed += 1
                total_size += size
    
    # Remove directories
    for pattern in dir_patterns:
        # Check in training directory
        dirs = list(training_dir.glob(pattern))
        for dir_path in dirs:
            if dir_path.is_dir():
                size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                print(f"   🗑️  Removing directory: {dir_path.relative_to(base_dir)} ({size/1024/1024:.1f} MB)")
                shutil.rmtree(dir_path)
                total_removed += 1
                total_size += size
        
        # Check in root directory
        dirs = list(base_dir.glob(pattern))
        for dir_path in dirs:
            if dir_path.is_dir():
                size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                print(f"   🗑️  Removing directory: {dir_path.relative_to(base_dir)} ({size/1024/1024:.1f} MB)")
                shutil.rmtree(dir_path)
                total_removed += 1
                total_size += size
    
    print(f"\n✅ Cleanup complete!")
    print(f"   📁 Removed {total_removed} items")
    print(f"   💾 Freed {total_size/1024/1024:.1f} MB of space")
    
    return total_removed, total_size

def show_remaining_files():
    """Show what files remain after cleanup."""
    print("\n📋 Remaining important files:")
    
    base_dir = Path(".")
    important_files = [
        "src/training/train_cyclic_loop_comparison.py",
        "src/models/riemannian_flow_vae.py", 
        "run_clean_training.py",
        "cleanup_training_files.py",
        "data/processed/Sprites_train_cyclic.pt",
        "data/pretrained/encoder.pt",
        "data/pretrained/decoder.pt",
        "data/pretrained/metric_T0.7_scaled.pt",
        "requirements.txt",
        "README.md"
    ]
    
    for file_path in important_files:
        path = base_dir / file_path
        if path.exists():
            if path.is_file():
                size = path.stat().st_size
                print(f"   ✅ {file_path} ({size/1024:.1f} KB)")
            else:
                print(f"   ✅ {file_path} (directory)")
        else:
            print(f"   ❓ {file_path} (not found)")

def main():
    print("🧹 RlVAE Training Files Cleanup")
    print("=" * 40)
    print()
    
    # Ask for confirmation
    response = input("This will remove all training outputs, visualizations, and wandb runs.\nProceed? [y/N]: ")
    
    if response.lower() not in ['y', 'yes']:
        print("❌ Cleanup cancelled.")
        return
    
    print()
    
    # Perform cleanup
    total_removed, total_size = cleanup_training_files()
    
    # Show remaining files
    show_remaining_files()
    
    print(f"\n🎯 Your workspace is now clean!")
    print(f"💡 Use 'python run_clean_training.py --loop_mode open --wandb_only' for clean training")

if __name__ == "__main__":
    main() 