#!/usr/bin/env bash
set -euo pipefail

cd /home/alaforgu/scratch/longitudinal_experiments/RlVAE

METRIC_DIR=/home/alaforgu/scratch/longitudinal_experiments/RlVAE/outputs/metrics
METRIC_FILE=metric_t0.7.pt
TRAIN=/home/alaforgu/scratch/longitudinal_experiments/RlVAE/data/processed/Sprites_train_cyclic.pt
TEST=/home/alaforgu/scratch/longitudinal_experiments/RlVAE/data/processed/Sprites_test_cyclic.pt
COMMON_BASE="experiment=rlvae_three_stage_pipeline experiment.run_stage_a=false experiment.run_stage_b=false experiment.run_stage_c=true experiment.stage_c.allow_metric_updates=false checkpoint.metric_dir=$METRIC_DIR checkpoint.metric_file=$METRIC_FILE data.train_path=$TRAIN data.test_path=$TEST training.trainer.devices=1 training.trainer.accelerator=auto training.trainer.precision=32 training.data.batch_size=32 training.optimizer.lr=3e-4 visualization.level=minimal evaluation.enabled=false model.encoder.architecture=cnn model.decoder.architecture=cnn +model.riemannian_kl_mode=sample_logq_logp"

run() {
  local NAME="$1"; shift
  echo "==== Launching $NAME ===="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u run_experiment.py \
    $COMMON_BASE \
    wandb.group=overnight_stageC +wandb.name="$NAME" \
    "$@"
}

# 1) CNN, riem_hmc, flows=0, beta=0.005, reg=0.1, recon=all
run C_riemHMC_f0_reg01_all \
  model.posterior.type=riem_hmc model.n_flows=0 model.beta=0.0 model.riemannian_beta=0.005 +model.metric.regularization_override=0.1 +model.reconstruction_mode=all

# 2) CNN, riem_hmc, flows=2, beta=0.005, reg=0.2, recon=all
run C_riemHMC_f2_reg02_all \
  experiment.stage_c.epochs=10 model.posterior.type=riem_hmc model.n_flows=2 model.beta=0.0 model.riemannian_beta=0.005 +model.metric.regularization_override=0.2 +model.reconstruction_mode=all

# 3) CNN, riem_hmc, flows=7, beta=0.005, reg=0.2, recon=t0_weighted 0.9
run C_riemHMC_f7_reg02_t0w0p9 \
  experiment.stage_c.epochs=10 model.posterior.type=riem_hmc model.n_flows=7 model.beta=0.0 model.riemannian_beta=0.005 +model.metric.regularization_override=0.2 +model.reconstruction_mode=t0_weighted +model.reconstruction_weight_t0=0.9 +model.posterior_hmc.n_lf=5 +model.posterior_hmc.eps_lf=0.01

# 4) CNN, riem_hmc, flows=7, beta=0.01, reg=0.2, recon=t0_weighted 0.9
run C_riemHMC_f7_reg02_beta0p01_t0w0p9 \
  experiment.stage_c.epochs=10 model.posterior.type=riem_hmc model.n_flows=7 model.beta=0.0 model.riemannian_beta=0.01 +model.metric.regularization_override=0.2 +model.reconstruction_mode=t0_weighted +model.reconstruction_weight_t0=0.9 +model.posterior_hmc.n_lf=4 +model.posterior_hmc.eps_lf=0.008

# 5) CNN, gaussian, no flows, small riem_beta, reg=0.2, recon=t0_weighted 0.9
run C_gauss_f0_reg02_t0w0p9 \
  experiment.stage_c.epochs=10 model.posterior.type=gaussian model.n_flows=0 model.beta=0.0 model.riemannian_beta=0.005 +model.metric.regularization_override=0.2 +model.reconstruction_mode=t0_weighted +model.reconstruction_weight_t0=0.9

# 6) CNN, gaussian, no flows, small riem_beta, reg=0.1, recon=all
run C_gauss_f0_reg01_all \
  experiment.stage_c.epochs=10 model.posterior.type=gaussian model.n_flows=0 model.beta=0.0 model.riemannian_beta=0.005 +model.metric.regularization_override=0.1 +model.reconstruction_mode=all

# 7) CNN, riem_hmc, flows=7, beta=0.005, reg=0.1, recon=all, slightly stronger RHMC
run C_riemHMC_f7_reg01_all \
  experiment.stage_c.epochs=10 model.posterior.type=riem_hmc model.n_flows=7 model.beta=0.0 model.riemannian_beta=0.005 +model.metric.regularization_override=0.1 +model.reconstruction_mode=all +model.posterior_hmc.n_lf=6 +model.posterior_hmc.eps_lf=0.012

# 8) CNN, riem_hmc, flows=2, beta=0.01, reg=0.1, recon=t0_weighted 0.9
run C_riemHMC_f2_reg01_beta0p01_t0w0p9 \
  experiment.stage_c.epochs=10 model.posterior.type=riem_hmc model.n_flows=2 model.beta=0.0 model.riemannian_beta=0.01 +model.metric.regularization_override=0.1 +model.reconstruction_mode=t0_weighted +model.reconstruction_weight_t0=0.9 +model.posterior_hmc.n_lf=4 +model.posterior_hmc.eps_lf=0.008


