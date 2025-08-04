#!/usr/bin/env python3
"""
Demo Script for Enhanced RlVAE Analysis
=======================================

This script demonstrates how to use the enhanced analysis tools with your trained RlVAE model.
It will automatically find the best checkpoint and run comprehensive analysis.

Usage:
    python demo_enhanced_analysis.py
    python demo_enhanced_analysis.py --quick-demo  # For faster testing
"""

import argparse
import sys
from pathlib import Path
import subprocess
import glob

def find_best_checkpoint():
    """Find the best trained model checkpoint."""
    print("🔍 Searching for trained model checkpoints...")
    
    # Common checkpoint locations
    checkpoint_paths = [
        "outputs/checkpoints/*.ckpt",
        "outputs/*/checkpoints/*.ckpt",
        "results/*/checkpoints/*.ckpt", 
        "checkpoints/*.ckpt",
        "*.ckpt"
    ]
    
    all_checkpoints = []
    for pattern in checkpoint_paths:
        all_checkpoints.extend(glob.glob(pattern))
    
    if not all_checkpoints:
        print("❌ No checkpoints found. Please train a model first.")
        return None
    
    # Sort by modification time (most recent first)
    all_checkpoints.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    
    print(f"✅ Found {len(all_checkpoints)} checkpoints")
    best_checkpoint = all_checkpoints[0]
    print(f"📂 Using most recent: {best_checkpoint}")
    
    return best_checkpoint

def run_generation_analysis(checkpoint_path: str, quick: bool = False):
    """Run enhanced generation analysis."""
    print("\n🎨 Running Enhanced Generation Analysis...")
    
    cmd = [
        "python", "enhanced_generation_visualization.py",
        "--model-path", checkpoint_path,
        "--dataset", "dsprites",
        "--num-samples", "32" if quick else "64",
        "--output-dir", "generation_analysis"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ Generation analysis completed successfully!")
            print("📁 Results saved to: generation_analysis/")
        else:
            print("❌ Generation analysis failed:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("⏰ Generation analysis timed out")
    except Exception as e:
        print(f"❌ Error running generation analysis: {e}")

def run_inference_analysis(checkpoint_path: str, quick: bool = False):
    """Run enhanced inference analysis."""
    print("\n🧠 Running Enhanced Inference Analysis...")
    
    cmd = [
        "python", "enhanced_inference_visualization.py",
        "--model-path", checkpoint_path,
        "--data-path", "data",
        "--num-sequences", "10" if quick else "20",
        "--output-dir", "inference_analysis"
    ]
    
    if not quick:
        cmd.append("--sequence-analysis")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ Inference analysis completed successfully!")
            print("📁 Results saved to: inference_analysis/")
        else:
            print("❌ Inference analysis failed:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("⏰ Inference analysis timed out")
    except Exception as e:
        print(f"❌ Error running inference analysis: {e}")

def run_comprehensive_analysis(checkpoint_path: str, quick: bool = False):
    """Run comprehensive analysis suite."""
    print("\n🌐 Running Comprehensive Analysis Suite...")
    
    cmd = [
        "python", "comprehensive_rlvae_analysis.py",
        "--model-path", checkpoint_path,
        "--dataset", "dsprites",
        "--num-samples", "32" if quick else "64",
        "--num-sequences", "10" if quick else "20",
        "--output-dir", "comprehensive_analysis",
        "--full-analysis"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print("✅ Comprehensive analysis completed successfully!")
            print("📁 Results saved to: comprehensive_analysis/")
        else:
            print("❌ Comprehensive analysis failed:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("⏰ Comprehensive analysis timed out")
    except Exception as e:
        print(f"❌ Error running comprehensive analysis: {e}")

def check_requirements():
    """Check if required packages are available."""
    print("🔧 Checking requirements...")
    
    required_packages = [
        'torch', 'torchvision', 'matplotlib', 'seaborn', 
        'sklearn', 'numpy', 'PIL'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("💡 Install with: pip install " + ' '.join(missing))
        return False
    
    print("✅ All required packages available")
    return True

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Demo Enhanced RlVAE Analysis")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint to use")
    parser.add_argument("--quick-demo", action="store_true", help="Run quick demo with smaller samples")
    parser.add_argument("--generation-only", action="store_true", help="Run only generation analysis")
    parser.add_argument("--inference-only", action="store_true", help="Run only inference analysis")
    parser.add_argument("--comprehensive-only", action="store_true", help="Run only comprehensive analysis")
    
    args = parser.parse_args()
    
    print("🚀 Enhanced RlVAE Analysis Demo")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Find or use specified checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not Path(checkpoint_path).exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return
    else:
        checkpoint_path = find_best_checkpoint()
        if not checkpoint_path:
            return
    
    # Check if data is available
    data_path = Path("data/processed/Sprites_test_cyclic.pt")
    if not data_path.exists():
        print("⚠️ dSprites cyclic data not found. Analysis will use CIFAR-10 fallback.")
    else:
        print("✅ dSprites cyclic data found")
    
    # Determine which analyses to run
    if not any([args.generation_only, args.inference_only, args.comprehensive_only]):
        # Run all by default
        run_generation = run_inference = run_comprehensive = True
    else:
        run_generation = args.generation_only
        run_inference = args.inference_only  
        run_comprehensive = args.comprehensive_only
    
    # Run selected analyses
    if run_generation:
        run_generation_analysis(checkpoint_path, args.quick_demo)
    
    if run_inference:
        run_inference_analysis(checkpoint_path, args.quick_demo)
    
    if run_comprehensive:
        run_comprehensive_analysis(checkpoint_path, args.quick_demo)
    
    # Summary
    print(f"\n🎉 Demo Complete!")
    print(f"📊 Check the following directories for results:")
    
    if run_generation:
        print(f"   📈 Generation analysis: generation_analysis/")
    if run_inference:
        print(f"   🧠 Inference analysis: inference_analysis/")
    if run_comprehensive:
        print(f"   🌐 Comprehensive analysis: comprehensive_analysis/")
    
    print(f"\n💡 Tips:")
    print(f"   • Open the PNG files to see beautiful visualizations")
    print(f"   • Check JSON files for detailed statistics")
    print(f"   • Use --quick-demo for faster testing")
    print(f"   • Each script can also be run individually")

if __name__ == "__main__":
    main() 