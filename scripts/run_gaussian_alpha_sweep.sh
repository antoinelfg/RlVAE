#!/bin/bash
# Sweep rhmc_alpha for the Gaussian-only baseline (no RHMC dynamics).
# This isolates the effect of Σμ scaling on z0 statistics.
#
# Usage:
#   bash scripts/run_gaussian_alpha_sweep.sh            # run all alphas
#   bash scripts/run_gaussian_alpha_sweep.sh --alpha 1  # run a single alpha

set -euo pipefail

cd /home/alaforgu/scratch/longitudinal_experiments/RlVAE

LOG_DIR="logs/gaussian_alpha_sweep"
mkdir -p "${LOG_DIR}"

ALPHAS=(0.1 0.25 0.5 1.0 2.0 5.0 10.0)

run_one() {
    local alpha=$1
    local alpha_str=${alpha//./p}
    local log_file="${LOG_DIR}/gaussian_alpha_${alpha_str}.log"

    echo "===================================================================="
    echo "Gaussian baseline sweep — alpha=${alpha}"
    echo "  log file: ${log_file}"
    echo "===================================================================="

    RLVAE_DEBUG=1 \
    python run_experiment.py \
        +experiment=rlvae_debug_gaussian \
        model.posterior.rhmc_alpha=${alpha} \
        training.stage_overrides.stage_c.posterior.rhmc_alpha=${alpha} \
        2>&1 | tee "${log_file}"
}

if [[ "${1:-}" == "--alpha" && -n "${2:-}" ]]; then
    run_one "$2"
    exit 0
fi

for alpha in "${ALPHAS[@]}"; do
    run_one "${alpha}"
done
