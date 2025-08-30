#!/bin/bash
#SBATCH --job-name=overnight_stageC
#SBATCH --output=/home/alaforgu/scratch/longitudinal_experiments/RlVAE/slurm_logs/overnight_stageC_%j.out
#SBATCH --error=/home/alaforgu/scratch/longitudinal_experiments/RlVAE/slurm_logs/overnight_stageC_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

module purge || true
module load cuda/12.1 || true

export WANDB_PROJECT=rlvae-three-stage-visuals
export WANDB_DIR=/home/alaforgu/scratch/longitudinal_experiments/RlVAE/wandb

bash /home/alaforgu/scratch/longitudinal_experiments/RlVAE/scripts/overnight_stageC.sh | tee -a /home/alaforgu/scratch/longitudinal_experiments/RlVAE/overnight_stageC.log





