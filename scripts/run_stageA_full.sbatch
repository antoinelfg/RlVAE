#!/bin/bash
#SBATCH --job-name=stageA_full
#SBATCH --output=/home/alaforgu/scratch/longitudinal_experiments/RlVAE/slurm_logs/stageA_full_%j.out
#SBATCH --error=/home/alaforgu/scratch/longitudinal_experiments/RlVAE/slurm_logs/stageA_full_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
module purge || true
module load cuda/12.1 || true

export WANDB_PROJECT=rlvae-three-stage-visuals
export WANDB_DIR=/home/alaforgu/scratch/longitudinal_experiments/RlVAE/wandb

bash /home/alaforgu/scratch/longitudinal_experiments/RlVAE/scripts/run_stageA_full.sh


