#!/usr/bin/env python3
"""
Setup Script for RlVAE Experimental Framework
============================================

This script sets up the complete experimental framework with Hydra configurations.
"""

import os
import subprocess
import sys
from pathlib import Path


def create_directories():
    """Create necessary directories."""
    directories = [
        "conf",
        "conf/model",
        "conf/training", 
        "conf/data",
        "conf/visualization",
        "conf/experiment",
        "outputs",
        "src/models",
        "src/data",
        "src/training",
        "src/visualizations"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")


def test_imports():
    """Test if all required imports work."""
    print("\n🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        import lightning
        print(f"✅ Lightning: {lightning.__version__}")
    except ImportError:
        print("❌ Lightning not found")
        return False
    
    try:
        import hydra
        print(f"✅ Hydra: {hydra.__version__}")
    except ImportError:
        print("❌ Hydra not found")
        return False
    
    try:
        import wandb
        print(f"✅ WandB: {wandb.__version__}")
    except ImportError:
        print("❌ WandB not found")
        return False
    
    try:
        from omegaconf import DictConfig, OmegaConf
        print(f"✅ OmegaConf available")
    except ImportError:
        print("❌ OmegaConf not found")
        return False
    
    return True


def create_init_files():
    """Create __init__.py files for proper imports."""
    init_files = [
        "src/__init__.py",
        "src/models/__init__.py", 
        "src/data/__init__.py",
        "src/training/__init__.py",
        "src/visualizations/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✅ Created: {init_file}")


def create_example_scripts():
    """Create example run scripts."""
    
    # Quick test script
    quick_test = """#!/bin/bash
# Quick test of the experimental framework

echo "🧪 Quick Test - RlVAE Experimental Framework"
echo "=============================================="

echo "📋 Testing with minimal configuration..."
python run_experiment.py training=quick visualization=minimal experiment.name="quick_test" wandb.mode=offline

echo "✅ Quick test completed!"
"""
    
    with open("quick_test.sh", "w") as f:
        f.write(quick_test)
    os.chmod("quick_test.sh", 0o755)
    print("✅ Created: quick_test.sh")
    
    # Comparison study script
    comparison_script = """#!/bin/bash
# Run comparison between Vanilla VAE and Riemannian VAE

echo "🔬 Model Comparison Study"
echo "========================="

echo "🎯 Comparing Vanilla VAE vs Riemannian VAE..."
python run_experiment.py experiment=comparison_study training.n_epochs=20 wandb.mode=online

echo "✅ Comparison study completed!"
"""
    
    with open("run_comparison.sh", "w") as f:
        f.write(comparison_script)
    os.chmod("run_comparison.sh", 0o755)
    print("✅ Created: run_comparison.sh")
    
    # Hyperparameter sweep script
    sweep_script = """#!/bin/bash
# Run hyperparameter sweep

echo "🌊 Hyperparameter Sweep"
echo "======================="

echo "🎯 Running parameter sweep..."
python run_experiment.py experiment=hyperparameter_sweep -m

echo "✅ Hyperparameter sweep completed!"
"""
    
    with open("run_sweep.sh", "w") as f:
        f.write(sweep_script)
    os.chmod("run_sweep.sh", 0o755)
    print("✅ Created: run_sweep.sh")


def main():
    """Main setup function."""
    print("🛠️  Setting up RlVAE Experimental Framework")
    print("=" * 50)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Test imports
    if not test_imports():
        print("\n❌ Some required packages are missing!")
        print("💡 Install them with: pip install -r requirements.txt")
        return
    
    # Create init files
    print("\n📄 Creating __init__.py files...")
    create_init_files()
    
    # Create example scripts
    print("\n📝 Creating example scripts...")
    create_example_scripts()
    
    print("\n✅ Setup completed successfully!")
    print("\n🚀 Next steps:")
    print("1. Make sure your data files are in data/processed/")
    print("2. Run a quick test: ./quick_test.sh")
    print("3. Try a comparison study: ./run_comparison.sh") 
    print("4. Explore configurations in conf/ directory")
    print("\n📖 Usage examples:")
    print("   python run_experiment.py  # Single run")
    print("   python run_experiment.py training=quick  # Quick test")
    print("   python run_experiment.py model=vanilla_vae  # Vanilla VAE")
    print("   python run_experiment.py experiment=comparison_study  # Compare models")


if __name__ == "__main__":
    main() 