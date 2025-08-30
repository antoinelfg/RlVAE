#!/bin/bash
#SBATCH --job-name=test_kl_fix
#SBATCH --output=logs/test_kl_fix_%j.out
#SBATCH --error=logs/test_kl_fix_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load modules (adjust for your cluster)
module load cuda/11.8
module load python/3.9

# Set up environment
cd /home/alaforgu/scratch/longitudinal_experiments/RlVAE
source ~/miniconda3/bin/activate rlvae

# Create logs directory
mkdir -p logs

echo "=== Testing KL Divergence Fix ==="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Test 1: Quick test with minimal epochs to check KL computation
echo "=== Test 1: Quick KL Test ==="
python run_experiment.py \
    experiment=rlvae_three_stage_pipeline \
    model=rhvae_test \
    model.latent_dim=16 \
    metric=rhvae \
    sampling=rhmc_default \
    experiment.run_stage_a=false \
    experiment.run_stage_b=false \
    experiment.run_sampling=false \
    experiment.run_stage_c=true \
    training.trainer.accelerator=gpu \
    training.trainer.devices=1 \
    training.data.batch_size=8 \
    experiment.stage_c.epochs=2 \
    wandb.mode=online \
    wandb.project=kl-fix-test \
    wandb.group=test-kl-computation \
    visualization.level=none \
    training.trainer.precision=16-mixed

# Test 2: Full pipeline test if Test 1 succeeds
if [ $? -eq 0 ]; then
    echo "=== Test 1 PASSED - Running Full Pipeline Test ==="
    python run_experiment.py \
        experiment=rlvae_three_stage_pipeline \
        model=rhvae_test \
        model.latent_dim=16 \
        metric=rhvae \
        sampling=rhmc_default \
        experiment.run_stage_a=true \
        experiment.run_stage_b=true \
        experiment.run_sampling=false \
        experiment.run_stage_c=true \
        training.trainer.accelerator=gpu \
        training.trainer.devices=1 \
        training.data.batch_size=16 \
        experiment.stage_a.epochs=5 \
        experiment.stage_c.epochs=10 \
        wandb.mode=online \
        wandb.project=kl-fix-test \
        wandb.group=full-pipeline-test \
        visualization.level=minimal \
        training.trainer.precision=16-mixed
else
    echo "=== Test 1 FAILED - KL computation still broken ==="
    exit 1
fi

echo "=== Test completed at $(date) ==="

