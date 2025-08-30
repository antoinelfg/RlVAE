#!/usr/bin/env python3
"""
Enhanced RlVAE Streamlit App Launcher
=====================================

Launcher script for the enhanced Streamlit application with proper
environment setup, dependency checking, and configuration management.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import importlib.util

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit',
        'torch',
        'lightning',
        'plotly',
        'numpy',
        'pandas',
        'sklearn',
        'hydra',
        'wandb'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements_streamlit.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_gpu():
    """Check GPU availability and configuration."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"🚀 GPU detected: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
            print(f"   Count: {gpu_count}")
            
            # Set environment variables for optimal GPU usage
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            return True
        else:
            print("⚠️  No GPU detected - will use CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not available - GPU check skipped")
        return False

def setup_environment():
    """Setup environment variables and paths."""
    
    # Add src to Python path
    current_dir = Path(__file__).parent.absolute()
    src_dir = current_dir / "src"
    
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    # Set environment variables
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    # Create output directories
    output_dirs = ['outputs', 'outputs/models', 'outputs/experiments', 'outputs/visualizations']
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Environment setup complete")

def run_streamlit_app(port=8501, host='0.0.0.0', debug=False):
    """Run the Streamlit application."""
    
    print(f"🚀 Starting Enhanced RlVAE Streamlit App...")
    print(f"   URL: http://{host}:{port}")
    print(f"   Debug: {debug}")
    print()
    
    # Build command
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'app.py',
        '--server.port', str(port),
        '--server.address', host,
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false'
    ]
    
    if debug:
        cmd.extend(['--logger.level', 'debug'])
    
    try:
        # Run the app
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit app: {e}")
        return False
    
    return True

def main():
    """Main launcher function."""
    
    parser = argparse.ArgumentParser(
        description="Enhanced RlVAE Streamlit App Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_streamlit.py                    # Run with default settings
  python run_streamlit.py --port 8502        # Run on different port
  python run_streamlit.py --host localhost   # Run on localhost only
  python run_streamlit.py --debug            # Run in debug mode
  python run_streamlit.py --check-only       # Only check dependencies
        """
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=8501,
        help='Port to run the app on (default: 8501)'
    )
    
    parser.add_argument(
        '--host', 
        type=str, 
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Run in debug mode with verbose logging'
    )
    
    parser.add_argument(
        '--check-only', 
        action='store_true',
        help='Only check dependencies and environment, don\'t run app'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("=" * 60)
    print("🧠 Enhanced RlVAE Streamlit App Launcher")
    print("=" * 60)
    print()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Setup environment
    setup_environment()
    
    if args.check_only:
        print("\n✅ Environment check complete!")
        return
    
    # Run the app
    print("\n" + "=" * 60)
    success = run_streamlit_app(
        port=args.port,
        host=args.host,
        debug=args.debug
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()