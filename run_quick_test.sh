#!/bin/bash
#SBATCH --job-name=rlvae_test
#SBATCH --output=logs/quick_test_%j.out
#SBATCH --error=logs/quick_test_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu

# Quick Test Script for RlVAE
# ===========================
# This script runs a quick validation test to ensure everything works
# before running the full weekend experiment suite

set -e

echo "🧪 Starting RlVAE Quick Test"
echo "============================"
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo ""

# Create logs directory
mkdir -p logs

# Activate conda environment
source /scratch/alaforgu/miniconda3/etc/profile.d/conda.sh
conda activate longitudinal_env

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export WANDB_MODE=online

echo "✅ Environment setup complete"
echo ""

# Run validation tests
echo "🔍 Running validation tests..."
python test_hybrid_model.py > logs/validation_hybrid_${SLURM_JOB_ID}.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Hybrid model validation passed"
else
    echo "❌ Hybrid model validation failed"
    exit 1
fi

python test_modular_components.py > logs/validation_components_${SLURM_JOB_ID}.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Modular components validation passed"
else
    echo "❌ Modular components validation failed"
    exit 1
fi

echo "✅ All validation tests passed"
echo ""

# Run quick training experiment
echo "🚀 Running quick training experiment..."
python run_experiment.py \
    model=hybrid_rlvae \
    training=quick \
    visualization=minimal \
    experiment.name=quick_test_${SLURM_JOB_ID} \
    > logs/quick_training_${SLURM_JOB_ID}.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Quick training completed successfully"
else
    echo "❌ Quick training failed"
    exit 1
fi

echo ""
echo "🎉 Quick test completed successfully!"
echo "Check logs/ directory for detailed results"
echo "Time: $(date)" 