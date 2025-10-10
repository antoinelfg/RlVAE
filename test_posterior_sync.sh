#!/bin/bash
# Test script to verify posterior type synchronization fix
# Run this to quickly check if RHMC posterior is properly configured

echo "=================================================="
echo "Testing Posterior Type Synchronization Fix"
echo "=================================================="
echo ""
echo "This will run Stage C for 1 epoch to verify the posterior type is correctly synced."
echo ""

# Run RHMC experiment for 1 epoch
echo "Running RHMC posterior test..."
python /scratch/alaforgu/longitudinal_experiments/RlVAE/run_experiment.py \
  experiment=rlvae_three_stage_long_rhmc_modular \
  data=ellipse_sequences \
  experiment.stage_a.epochs=1 \
  experiment.stage_c.epochs=1 \
  wandb.mode=online \
  seed=42 \
  2>&1 | tee /tmp/posterior_sync_test.log

echo ""
echo "=================================================="
echo "Checking logs for posterior type sync..."
echo "=================================================="
echo ""

# Check if the sync was successful
if grep -q "🔒 Forcing posterior type sync: 'riemannian_rhmc'" /tmp/posterior_sync_test.log; then
    echo "✅ SUCCESS: Posterior type sync detected!"
    echo ""
    
    # Extract the posterior type lines
    echo "Posterior types after sync:"
    grep -A 3 "Set model parameters:" /tmp/posterior_sync_test.log | grep "posterior"
    
    echo ""
    
    # Check all three should be riemannian_rhmc
    if grep -q "model.posterior.type: riemannian_rhmc" /tmp/posterior_sync_test.log && \
       grep -q "model.posterior_type: riemannian_rhmc" /tmp/posterior_sync_test.log && \
       grep -q "training.model.posterior.type: riemannian_rhmc" /tmp/posterior_sync_test.log; then
        echo "✅✅✅ PERFECT: All three posterior types are 'riemannian_rhmc'!"
        echo ""
        echo "The fix is working correctly. You can now run full experiments."
    else
        echo "⚠️ WARNING: Not all posterior types are synced correctly."
        echo "Check the logs above."
    fi
else
    echo "❌ ERROR: Posterior type sync not detected in logs!"
    echo ""
    echo "Possible issues:"
    echo "1. The fix was not applied correctly"
    echo "2. The experiment config doesn't have posterior.type set"
    echo "3. An error occurred during sync"
    echo ""
    echo "Check the full log at: /tmp/posterior_sync_test.log"
fi

echo ""
echo "=================================================="
echo "Test complete!"
echo "Full log saved to: /tmp/posterior_sync_test.log"
echo "=================================================="

