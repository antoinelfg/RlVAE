#!/bin/bash
#SBATCH --job-name=rlvae_manyseq
#SBATCH --output=logs/rlvae_manyseq_%j.out
#SBATCH --error=logs/rlvae_manyseq_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00


cd /home/alaforgu/scratch/longitudinal_experiments/RlVAE

python run_experiment.py visualization=full visualization.max_sequences=32 +training.batch_size=32 