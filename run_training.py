#!/usr/bin/env python3
"""
Launcher script for Riemannian Flow VAE training.
This script sets up the correct paths and runs the training script.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir / "src"
lib_src_dir = src_dir / "lib" / "src"

# Add both src directories to path
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(lib_src_dir))

if __name__ == "__main__":
    # Change to training directory
    training_dir = src_dir / "training"
    os.chdir(training_dir)
    
    # Import and run the training script
    print("🚀 Starting Riemannian Flow VAE Training")
    print(f"📁 Working directory: {training_dir}")
    print(f"🐍 Python path includes: {src_dir}, {lib_src_dir}")
    
    # Import the main training function
    try:
        # Add the training directory to the path for local imports
        sys.path.insert(0, str(training_dir))
        import src.training.train_cyclic_loop_comparison as train_cyclic_loop_comparison
        train_cyclic_loop_comparison.main()
    except ImportError as e:
        print(f"❌ Failed to import training script: {e}")
        print("Please ensure all dependencies are installed and paths are correct.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1) 