#!/usr/bin/env python3
"""
Monitor Enhanced KL Experiment Progress
"""

import os
import time
import subprocess
from pathlib import Path

def check_experiment_status():
    """Check the status of the enhanced KL experiment."""
    
    print("🔍 Monitoring Enhanced KL Experiment")
    print("=" * 50)
    
    # Check if process is running
    try:
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        enhanced_kl_processes = [
            line for line in result.stdout.split('\n') 
            if 'run_experiment.py' in line and 'enhanced_kl_experiment' in line
        ]
        
        if enhanced_kl_processes:
            print("✅ Enhanced KL experiment is running:")
            for process in enhanced_kl_processes:
                print(f"   {process.strip()}")
        else:
            print("❌ Enhanced KL experiment is not running")
            return False
            
    except subprocess.CalledProcessError:
        print("❌ Could not check process status")
        return False
    
    # Check for output directories
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        # Look for recent directories
        recent_dirs = []
        for item in outputs_dir.iterdir():
            if item.is_dir() and "2025-08-25" in str(item):
                recent_dirs.append(item)
        
        if recent_dirs:
            print(f"\n📁 Found {len(recent_dirs)} recent output directories:")
            for dir_path in recent_dirs:
                print(f"   {dir_path}")
                
                # Check for log files
                log_files = list(dir_path.rglob("*.log"))
                if log_files:
                    print(f"   📄 Log files: {len(log_files)}")
                    for log_file in log_files[-3:]:  # Show last 3
                        print(f"      {log_file.name}")
                        
                        # Show last few lines
                        try:
                            with open(log_file, 'r') as f:
                                lines = f.readlines()
                                if lines:
                                    print(f"      Last line: {lines[-1].strip()}")
                        except Exception as e:
                            print(f"      Could not read log: {e}")
        else:
            print("\n📁 No recent output directories found yet")
    
    # Check WandB for logging
    print("\n🌐 Checking WandB for experiment logging...")
    try:
        # This would require wandb CLI to be installed
        result = subprocess.run(
            ["wandb", "status"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0:
            print("✅ WandB is available")
        else:
            print("⚠️ WandB status unclear")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️ WandB CLI not available or timed out")
    
    return True

def main():
    """Main monitoring function."""
    
    print("🚀 Enhanced KL Experiment Monitor")
    print("=" * 50)
    
    while True:
        is_running = check_experiment_status()
        
        if not is_running:
            print("\n❌ Experiment appears to have stopped")
            break
            
        print(f"\n⏰ Next check in 30 seconds... (Press Ctrl+C to stop)")
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            break

if __name__ == "__main__":
    main()

