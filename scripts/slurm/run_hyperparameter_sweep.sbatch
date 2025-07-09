#!/bin/bash
#SBATCH --job-name=rlvae_hyperparam_sweep
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --time=47:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu

# Hyperparameter Sweep Runner for RLVAE
# =====================================
# This script runs comprehensive hyperparameter optimization sweeps
# for the RLVAE pipeline using WandB and Hydra

# Load modules (no modules needed on this system)
module purge

# Activate conda environment
source /scratch/alaforgu/miniconda3/etc/profile.d/conda.sh
conda activate rlvae_env

# Create logs directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_MODE=online
export WANDB_PROJECT="rlvae-hyperparameter-optimization"

# Parse command line arguments
SWEEP_CONFIG=${1:-"comprehensive_hyperparameter_sweep"}
AGENT_COUNT=${2:-"1"}
MAX_RUNS=${3:-"20"}
ENTITY=${4:-""}

echo "=========================================="
echo "🔬 RLVAE Hyperparameter Sweep"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "Sweep Config: $SWEEP_CONFIG"
echo "Agent Count: $AGENT_COUNT"
echo "Max Runs: $MAX_RUNS"
echo "=========================================="

# Function to run sweep with error handling
run_sweep() {
    local sweep_name=$1
    local agents=$2
    local max_runs=$3
    local entity_flag=""
    
    if [ ! -z "$ENTITY" ]; then
        entity_flag="--entity $ENTITY"
    fi
    
    echo "=========================================="
    echo "🎯 Starting Sweep: $sweep_name"
    echo "Agents: $agents"
    echo "Max runs: $max_runs"
    echo "Time: $(date)"
    echo "=========================================="
    
    # Run the sweep
    python scripts/run_sweep.py \
        --sweep-config "$sweep_name" \
        --agent-count "$agents" \
        --max-runs "$max_runs" \
        $entity_flag \
        --verbose
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Sweep $sweep_name completed successfully!"
    else
        echo "❌ Sweep $sweep_name failed with exit code $exit_code"
        echo "Check logs for details"
    fi
    
    echo "=========================================="
    echo ""
    
    return $exit_code
}

# Main execution
echo "🚀 Starting hyperparameter optimization..."

# Check if sweep configuration exists
if [ ! -f "conf/sweep/${SWEEP_CONFIG}.yaml" ]; then
    echo "❌ Sweep configuration not found: conf/sweep/${SWEEP_CONFIG}.yaml"
    echo "Available configurations:"
    ls conf/sweep/*.yaml 2>/dev/null || echo "No sweep configurations found"
    exit 1
fi

# System information
echo "📊 System Information:"
echo "   Python: $(python --version)"
echo "   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
echo "   CUDA: $(nvcc --version | grep 'release' | awk '{print $6}' | cut -c2-)"
echo "   Memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "   CPUs: $SLURM_CPUS_PER_TASK"
echo ""

# Verify packages
echo "🔍 Verifying required packages..."
python -c "import wandb, torch, hydra, omegaconf; print('✅ All packages available')" || {
    echo "❌ Missing required packages"
    exit 1
}

# Check WandB login
echo "🔐 Checking WandB authentication..."
python -c "import wandb; wandb.login()" || {
    echo "❌ WandB authentication failed"
    echo "Please run 'wandb login' or set WANDB_API_KEY"
    exit 1
}

# Run the specified sweep
run_sweep "$SWEEP_CONFIG" "$AGENT_COUNT" "$MAX_RUNS"
sweep_exit_code=$?

# Cleanup and summary
echo "=========================================="
echo "🏁 Hyperparameter Sweep Complete"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Duration: $SECONDS seconds"
echo "End time: $(date)"

if [ $sweep_exit_code -eq 0 ]; then
    echo "✅ Sweep completed successfully!"
else
    echo "❌ Sweep failed (exit code: $sweep_exit_code)"
fi

echo "=========================================="

# Generate summary report
echo "📊 Generating summary report..."
python -c "
import wandb
import sys
from datetime import datetime

try:
    api = wandb.Api()
    project = api.project('rlvae-hyperparameter-optimization')
    
    print(f'📈 Project Summary:')
    print(f'   Total runs: {len(list(project.runs()))}')
    print(f'   Project URL: {project.url}')
    
except Exception as e:
    print(f'⚠️ Could not generate summary: {e}')
"

echo "🎯 Check your WandB dashboard for detailed results:"
echo "   https://wandb.ai/your-entity/rlvae-hyperparameter-optimization"

exit $sweep_exit_code 