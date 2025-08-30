#!/usr/bin/env bash
set -euo pipefail

cd /home/alaforgu/scratch/longitudinal_experiments/RlVAE

TRAIN=/home/alaforgu/scratch/longitudinal_experiments/RlVAE/data/processed/Sprites_train_cyclic.pt
TEST=/home/alaforgu/scratch/longitudinal_experiments/RlVAE/data/processed/Sprites_test_cyclic.pt
METRIC_DIR=/home/alaforgu/scratch/longitudinal_experiments/RlVAE/outputs/metrics
RUN_NAME="stageA_mlp_ld16_T0p7_full_$(date +%Y%m%d_%H%M%S)"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u run_experiment.py \
  experiment=rlvae_three_stage_pipeline \
  experiment.run_stage_a=true experiment.run_stage_b=false experiment.run_stage_c=false \
  model.encoder.architecture=mlp model.decoder.architecture=mlp \
  model.latent_dim=16 model.beta=1.0 \
  metric.temperature=0.7 +model.metric.temperature_override=0.7 \
  experiment.stage_a.epochs=200 \
  training.trainer.accelerator=gpu training.trainer.devices=1 training.trainer.precision=16-mixed \
  training.data.batch_size=64 training.optimizer.lr=3e-4 \
  data.train_path=$TRAIN data.test_path=$TEST \
  checkpoint.metric_dir=$METRIC_DIR \
  wandb.group=stageA_full +wandb.name=$RUN_NAME


