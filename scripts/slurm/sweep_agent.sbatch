#!/bin/bash
#SBATCH --job-name=rlvae-sweep
#SBATCH --output=sweep_logs/%x_%j.out
#SBATCH --error=sweep_logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu

# =============================================================================
# RLVAE Sweep Agent - SLURM Job Script
# =============================================================================
#
# Usage:
#   1. First, initialize the sweep locally:
#      wandb sweep conf/sweep/stage_c_bayesian.yaml
#
#   2. Then submit multiple agents:
#      for i in {1..10}; do
#        sbatch scripts/slurm/sweep_agent.sbatch <sweep_id>
#      done
#
# =============================================================================

# Check sweep path argument
if [ -z "$1" ]; then
    echo "Usage: sbatch sweep_agent.sbatch <entity/project/sweep_id>"
    echo ""
    echo "Example: sbatch sweep_agent.sbatch alaforgu/RlVAE-posterior/abc12345"
    echo ""
    echo "Get the full path from 'wandb sweep' output or your WandB dashboard."
    exit 1
fi

SWEEP_PATH="$1"

# Validate format (should contain at least one /)
if [[ ! "$SWEEP_PATH" == *"/"* ]]; then
    echo "ERROR: Sweep path must include entity and project."
    echo "Format: <entity>/<project>/<sweep_id>"
    echo "Example: alaforgu/RlVAE-posterior/abc12345"
    exit 1
fi

# Project settings
PROJECT_ROOT="/home/alaforgu/scratch/longitudinal_experiments/RlVAE"

# Create log directory
mkdir -p "$PROJECT_ROOT/sweep_logs"

# Navigate to project
cd "$PROJECT_ROOT"

# Load modules (adjust for your cluster)
module load python/3.10 cuda/11.8 2>/dev/null || true

# Activate environment
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=0

# Print job info
echo "=============================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Sweep Path: $SWEEP_PATH"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=============================================="

# Run the sweep agent with full path
wandb agent "$SWEEP_PATH"

echo "=============================================="
echo "Sweep agent finished at: $(date)"
echo "=============================================="

