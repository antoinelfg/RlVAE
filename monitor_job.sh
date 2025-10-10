#!/bin/bash
# Monitor the SLURM job progress

JOB_ID=4284595
LOG_DIR="outputs/logs"

echo "Monitoring SLURM job $JOB_ID..."
echo "Log files will be in: $LOG_DIR/"
echo ""

# Check job status
echo "Current job status:"
squeue -u alaforgu

echo ""
echo "Job details:"
scontrol show job $JOB_ID

echo ""
echo "To monitor the experiment progress, you can:"
echo "1. Check job status: squeue -u alaforgu"
echo "2. View live output: tail -f $LOG_DIR/rlvae_rhmc_${JOB_ID}.out"
echo "3. View errors: tail -f $LOG_DIR/rlvae_rhmc_${JOB_ID}.err"
echo "4. Check WandB dashboard for real-time metrics"
echo ""
echo "Expected runtime: ~8-12 hours for 200 epochs"
echo "The experiment will run with optimized parameters:"
echo "  - riemannian_beta: 32.0 (4x stronger KL anchoring)"
echo "  - mu_l2_weight: 0.5 (5x stronger μ penalty)"
echo "  - rhmc_steps: 4 (better exploration)"
echo "  - rhmc_alpha: 0.2 (larger initial covariance)"
echo "  - kl_prior_mode: volume_gaussian (pulls μ toward RHMC prior)"
