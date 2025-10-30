#!/bin/bash
# Run all 5 ablation experiments for z0 investigation
# Usage: bash scripts/run_z0_investigation.sh

set -e  # Exit on error

cd /home/alaforgu/scratch/longitudinal_experiments/RlVAE

# Create logs directory
mkdir -p logs/z0_investigation

echo "======================================================================"
echo "Z0 INVESTIGATION: Running 5 Ablation Experiments"
echo "======================================================================"
echo ""
echo "Each experiment will run Stage B + Stage C (5 epochs) with RLVAE_DEBUG=1"
echo "Logs saved to: logs/z0_investigation/"
echo ""

# Experiment 1: Vanilla (K=1, no volume acceptance) - Pure N(mu, Sigma) baseline
echo "[1/5] Running VANILLA (K=1, vol OFF)..."
RLVAE_DEBUG=1 python run_experiment.py \
  +experiment=rlvae_debug_vanilla \
  settings.pipeline.mode=three_stage \
  settings.pipeline.run_stage_a=false \
  settings.pipeline.run_stage_b=true \
  settings.pipeline.run_stage_c=true \
  settings.training.stage_overrides.stage_b.enabled=true \
  2>&1 | tee logs/z0_investigation/debug_vanilla.log

echo ""
echo "[1/5] VANILLA complete."
echo ""

# Experiment 2: No Multi-Try (K=1, volume acceptance ON)
echo "[2/5] Running NOTRY (K=1, vol ON)..."
RLVAE_DEBUG=1 python run_experiment.py \
  +experiment=rlvae_debug_notry \
  settings.pipeline.mode=three_stage \
  settings.pipeline.run_stage_a=false \
  settings.pipeline.run_stage_b=true \
  settings.pipeline.run_stage_c=true \
  settings.training.stage_overrides.stage_b.enabled=true \
  2>&1 | tee logs/z0_investigation/debug_notry.log

echo ""
echo "[2/5] NOTRY complete."
echo ""

# Experiment 3: No Volume Acceptance (K=5, vol OFF)
echo "[3/5] Running NOVOL (K=5, vol OFF)..."
RLVAE_DEBUG=1 python run_experiment.py \
  +experiment=rlvae_debug_novol \
  settings.pipeline.mode=three_stage \
  settings.pipeline.run_stage_a=false \
  settings.pipeline.run_stage_b=true \
  settings.pipeline.run_stage_c=true \
  settings.training.stage_overrides.stage_b.enabled=true \
  2>&1 | tee logs/z0_investigation/debug_novol.log

echo ""
echo "[3/5] NOVOL complete."
echo ""

# Experiment 4: Baseline (K=5, vol ON) - Current problematic setup
echo "[4/5] Running BASELINE (K=5, vol ON) - current problematic setup..."
RLVAE_DEBUG=1 python run_experiment.py \
  +experiment=rlvae_debug_baseline \
  settings.pipeline.mode=three_stage \
  settings.pipeline.run_stage_a=false \
  settings.pipeline.run_stage_b=true \
  settings.pipeline.run_stage_c=true \
  settings.training.stage_overrides.stage_b.enabled=true \
  2>&1 | tee logs/z0_investigation/debug_baseline.log

echo ""
echo "[4/5] BASELINE complete."
echo ""

# Experiment 5: High K Test (K=20, vol ON)
echo "[5/5] Running HIGHTRY (K=20, vol ON)..."
RLVAE_DEBUG=1 python run_experiment.py \
  +experiment=rlvae_debug_hightry \
  settings.pipeline.mode=three_stage \
  settings.pipeline.run_stage_a=false \
  settings.pipeline.run_stage_b=true \
  settings.pipeline.run_stage_c=true \
  settings.training.stage_overrides.stage_b.enabled=true \
  2>&1 | tee logs/z0_investigation/debug_hightry.log

echo ""
echo "[5/5] HIGHTRY complete."
echo ""

echo "======================================================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "======================================================================"
echo ""
echo "Logs saved to: logs/z0_investigation/"
echo ""
echo "Next steps:"
echo "1. Parse logs with: python scripts/analyze_z0_diagnostics.py logs/z0_investigation/debug_*.log"
echo "2. Compare WandB metrics in project: rlvae-z0-investigation"
echo "3. Review console logs for detailed candidate diagnostics"
echo ""

