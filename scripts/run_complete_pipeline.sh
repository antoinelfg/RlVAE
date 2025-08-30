#!/bin/bash

# Simple wrapper script to submit the complete RLVAE pipeline experiment
# Usage: ./run_complete_pipeline.sh

echo "🚀 Submitting Complete RLVAE Pipeline Experiment to SLURM"
echo "=========================================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Submit the job
if sbatch scripts/slurm/run_complete_pipeline.sbatch; then
    echo ""
    echo "✅ Job submitted successfully!"
    echo ""
    echo "📊 Monitor progress with:"
    echo "   squeue -u $USER"
    echo "   tail -f logs/rlvae_complete_pipeline_*.out"
    echo ""
    echo "🌐 Track in WandB:"
    echo "   https://wandb.ai/your-username/rlvae-hyperparameter-optimization"
    echo ""
    echo "📁 Results will be saved in: outputs/"
else
    echo ""
    echo "❌ Job submission failed!"
    echo "💡 Check:"
    echo "   - SLURM is available: sinfo"
    echo "   - Correct partition name"
    echo "   - Account permissions"
    exit 1
fi 