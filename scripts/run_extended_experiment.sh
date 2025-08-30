#!/bin/bash
#SBATCH --job-name=rlvae_extended
#SBATCH --output=logs/rlvae_extended_%j.out
#SBATCH --error=logs/rlvae_extended_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu

# Load modules (adjust for your system)
module load cuda/11.8
module load python/3.9

# Activate conda environment (adjust path as needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlvae

# Create logs directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the extended experiment
echo "🚀 Starting extended RLVAE experiment with many epochs"
echo "📅 Start time: $(date)"
echo "💻 Node: $(hostname)"
echo "🔧 GPU: $CUDA_VISIBLE_DEVICES"

python run_experiment.py \
    experiment=global_vanilla_rlvae_pipeline \
    model=rhvae_original_with_metric_update \
    data=cyclic_sprites \
    model.latent_dim=16 \
    experiment.skip_stage1=false \
    experiment.stage1.epochs=50 \
    experiment.stage2.epochs=100 \
    experiment.stage2.visualization=minimal \
    model.metric_update_frequency=30 \
    model.riemannian_beta=1.0 \
    wandb.project=rlvae-extended-experiment \
    wandb.tags=[extended,metric_evolution,long_training] \
    2>&1 | tee logs/rlvae_extended_full.log

echo "✅ Experiment completed"
echo "📅 End time: $(date)"
echo "📊 Check logs/rlvae_extended_full.log for full output"

