#!/bin/bash
# Test script to verify KL and flows fixes
# Run this to quickly check if RHMC KL is enabled and flows count is correct

echo "=================================================="
echo "Testing KL and Flows Fix"
echo "=================================================="
echo ""
echo "This will run Stage C for 1 epoch to verify:"
echo "1. RHMC posterior enables Riemannian KL"
echo "2. n_flows = sequence_length - 1 = 7"
echo ""

# Run RHMC experiment for 1 epoch
echo "Running RHMC KL and flows test..."
python /scratch/alaforgu/longitudinal_experiments/RlVAE/run_experiment.py \
  experiment=rlvae_three_stage_long_rhmc_modular \
  data=ellipse_sequences \
  experiment.stage_a.epochs=1 \
  experiment.stage_c.epochs=1 \
  wandb.mode=online \
  seed=42 \
  2>&1 | tee /tmp/kl_flows_test.log

echo ""
echo "=================================================="
echo "Checking logs for fixes..."
echo "=================================================="
echo ""

# Check if posterior type sync was successful
if grep -q "🔒 Forcing posterior type sync: 'riemannian_rhmc'" /tmp/kl_flows_test.log; then
    echo "✅ Posterior type sync: DETECTED"
else
    echo "❌ Posterior type sync: NOT DETECTED"
fi

# Check if flows enforcement was successful
if grep -q "🔧 Enforcing flows count: sequence_length=8 → n_flows=7" /tmp/kl_flows_test.log; then
    echo "✅ Flows enforcement: DETECTED"
else
    echo "❌ Flows enforcement: NOT DETECTED"
fi

echo ""
echo "Final configuration:"
grep -A 10 "Set model parameters:" /tmp/kl_flows_test.log | grep -E "(posterior|sequence_length|n_flows)"

echo ""
echo "Checking for non-zero KL in training logs..."
if grep -q "riemannian_kl.*[1-9]" /tmp/kl_flows_test.log; then
    echo "✅ Non-zero Riemannian KL: DETECTED"
    # Show some KL values
    echo "Sample KL values:"
    grep "riemannian_kl" /tmp/kl_flows_test.log | head -3
else
    echo "⚠️ Non-zero Riemannian KL: NOT YET DETECTED (may need more training steps)"
fi

echo ""
echo "=================================================="
echo "Test complete!"
echo "Full log saved to: /tmp/kl_flows_test.log"
echo "=================================================="
echo ""
echo "Expected results:"
echo "✅ posterior.type: riemannian_rhmc"
echo "✅ sequence_length: 8"
echo "✅ n_flows: 7"
echo "✅ Non-zero val_riemannian_kl in WandB"
