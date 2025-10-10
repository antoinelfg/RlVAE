#!/bin/bash
#SBATCH --job-name=rlvae_rhmc_200ep
#SBATCH --output=outputs/logs/rlvae_rhmc_%j.out
#SBATCH --error=outputs/logs/rlvae_rhmc_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Load modules if needed
# module load python/3.12
# module load cuda/11.8

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/alaforgu/scratch/longitudinal_experiments/RlVAE:$PYTHONPATH

# Create logs directory
mkdir -p outputs/logs

# Change to project directory
cd /home/alaforgu/scratch/longitudinal_experiments/RlVAE

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Run the experiment with optimized parameters
echo "Starting RlVAE experiment with optimized μ alignment parameters..."
python run_experiment.py \
    experiment=rlvae_three_stage_long_rhmc_modular \
    data=ellipse_sequences \
    seed=42 \
    wandb.mode=online \
    model=riemannian_rhmc_vae

echo "Experiment completed at: $(date)"
echo "Job finished successfully!"
