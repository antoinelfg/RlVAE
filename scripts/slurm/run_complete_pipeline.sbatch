#!/bin/bash
#SBATCH --job-name=rlvae_complete_pipeline
#SBATCH --output=logs/rlvae_complete_pipeline_%j.out
#SBATCH --error=logs/rlvae_complete_pipeline_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu

# Load modules (adjust for your cluster)
module purge
module load cuda/11.8
module load python/3.10

# Activate conda environment (adjust path)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlvae_env

# Create logs directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export WANDB_MODE=online
export WANDB_PROJECT="rlvae-hyperparameter-optimization"

echo "=============================================================="
echo "🚀 Starting Complete RLVAE Pipeline Experiment (OPTIMIZED)"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"  
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "Memory: 64GB"
echo "CPUs: 8"
echo "=============================================================="

# Change to project directory
cd $SLURM_SUBMIT_DIR || exit 1

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')"

# Print configuration
echo ""
echo "📋 Experiment Configuration:"
echo "- Data: colored_circles (10 timesteps)"
echo "- Latent Dimension: 2"  
echo "- Stage 1: Vanilla VAE (200 epochs)"
echo "- Stage 2: RLVAE (200 epochs, 9 flows)"
echo "- Beta values: 1.0 (balanced losses)"
echo "- Visualization: standard level"
echo "- Loss scaling: 255x (optimized)"
echo ""

# Define the complete command with all optimized parameters
EXPERIMENT_CMD="python run_experiment.py \
experiment=global_vanilla_rlvae_pipeline \
data=colored_circles \
model.latent_dim=2 \
experiment.skip_stage1=false \
experiment.stage1.epochs=200 \
experiment.stage2.epochs=200 \
experiment.stage2.n_flows=9 \
experiment.stage2.beta=1.0 \
experiment.stage2.riemannian_beta=1.0 \
visualization.level=standard \
wandb.mode=online"

echo "🎯 Executing experiment..."
echo "Command: $EXPERIMENT_CMD"
echo ""

# Start timing
start_time=$(date +%s)

# Run the experiment with error handling
if eval $EXPERIMENT_CMD; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    hours=$((duration / 3600))
    minutes=$(((duration % 3600) / 60))
    seconds=$((duration % 60))
    
    echo ""
    echo "=============================================================="
    echo "✅ COMPLETE PIPELINE EXPERIMENT COMPLETED SUCCESSFULLY!"
    echo "=============================================================="
    echo "🕐 Total Duration: ${hours}h ${minutes}m ${seconds}s"
    echo "📊 Check WandB for detailed results and visualizations"
    echo "📁 Results saved in: outputs/"
    echo "🏁 Job completed at: $(date)"
    echo "=============================================================="
else
    echo ""
    echo "=============================================================="
    echo "❌ EXPERIMENT FAILED!"
    echo "=============================================================="
    echo "💡 Check error logs above for details"
    echo "🔍 Debug steps:"
    echo "   1. Check CUDA availability"
    echo "   2. Verify data files exist"
    echo "   3. Check conda environment"
    echo "   4. Review configuration parameters"
    echo "🏁 Job failed at: $(date)"
    echo "=============================================================="
    exit 1
fi

# Print final job statistics
echo ""
echo "📈 Job Statistics:"
echo "- Job ID: $SLURM_JOB_ID"
echo "- Node: $SLURM_NODELIST"
echo "- Partition: $SLURM_JOB_PARTITION" 
echo "- QoS: $SLURM_JOB_QOS"
echo "- Account: $SLURM_JOB_ACCOUNT"
echo ""
echo "🎉 Complete pipeline job finished!" 