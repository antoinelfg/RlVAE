#!/bin/bash
#SBATCH --job-name=rlvae_viz32
#SBATCH --output=logs/rlvae_viz32_%j.out
#SBATCH --error=logs/rlvae_viz32_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load your environment here (edit as needed)
# module load anaconda/3
# source activate your_env_name

cd $SLURM_SUBMIT_DIR

python run_experiment.py experiment=global_vanilla_rlvae_pipeline +visualization.max_sequences=32 visualization.disable_curvature=false 