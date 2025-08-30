#!/bin/bash
# RlVAE Pipeline Runner - Local Metric-Aligned Gaussian Implementation
# Two-stage training pipeline for RLVAE with geometry-aware posterior

set -e  # Exit on any error

echo "🚀 RlVAE Local Metric-Aligned Gaussian Pipeline"
echo "==============================================="
echo "$(date): Starting two-stage training pipeline"

# Configuration
LATENT_DIM=${LATENT_DIM:-16}
ARCHITECTURE=${ARCHITECTURE:-"mlp"}
SEED=${SEED:-42}
EPOCHS_STAGE1=${EPOCHS_STAGE1:-30}
EPOCHS_STAGE2=${EPOCHS_STAGE2:-50}
DATA=${DATA:-"cyclic_sprites"}
VISUALIZATION=${VISUALIZATION:-"standard"}

echo "📋 Configuration:"
echo "  - Latent Dim: ${LATENT_DIM}"
echo "  - Architecture: ${ARCHITECTURE}"
echo "  - Seed: ${SEED}"
echo "  - Data: ${DATA}"
echo "  - Stage 1 Epochs: ${EPOCHS_STAGE1}"
echo "  - Stage 2 Epochs: ${EPOCHS_STAGE2}"
echo ""

# Stage 1: Vanilla VAE + Metric Extraction
echo "🏗️  Stage 1: Vanilla VAE Training + Metric Extraction"
echo "---------------------------------------------------"
echo "$(date): Training vanilla VAE for metric extraction..."

python run_experiment.py \
    experiment=vanilla_diverse_metric \
    model.latent_dim=${LATENT_DIM} \
    model.architecture=${ARCHITECTURE} \
    training.max_epochs=${EPOCHS_STAGE1} \
    data=${DATA} \
    visualization=${VISUALIZATION} \
    seed=${SEED} \
    experiment_type="research" \
    wandb.tags="[\"stage1\",\"vanilla_vae\",\"metric_extraction\",\"local_metric_gaussian\"]" \
    wandb.group="local_metric_aligned_pipeline" \
    experiment_name="pipeline_stage1_vanilla_vae_${ARCHITECTURE}_ld${LATENT_DIM}"

if [ $? -ne 0 ]; then
    echo "❌ Stage 1 failed!"
    exit 1
fi

echo "✅ Stage 1 completed successfully!"
echo ""

# Wait a moment between stages
sleep 2

# Stage 2: RlVAE with Local Metric-Aligned Gaussian Posterior
echo "🧠 Stage 2: RlVAE with Local Metric-Aligned Gaussian Posterior"
echo "-------------------------------------------------------------"
echo "$(date): Training RlVAE with local metric-aligned posterior..."

python run_experiment.py \
    experiment=enhanced_kl_experiment \
    model.latent_dim=${LATENT_DIM} \
    model.architecture=${ARCHITECTURE} \
    training.max_epochs=${EPOCHS_STAGE2} \
    data=${DATA} \
    visualization=${VISUALIZATION} \
    seed=${SEED} \
    experiment_type="research" \
    model.kl_use_metric_normalization=true \
    model.kl_metric_norm_mode="geomean" \
    model.posterior_local_alpha=0.5 \
    model.kl_amp_safe=true \
    model.posterior_type="riemannian_metric" \
    wandb.tags="[\"stage2\",\"rlvae\",\"local_metric_gaussian\",\"geometry_aware\"]" \
    wandb.group="local_metric_aligned_pipeline" \
    experiment_name="pipeline_stage2_rlvae_${ARCHITECTURE}_ld${LATENT_DIM}"

if [ $? -ne 0 ]; then
    echo "❌ Stage 2 failed!"
    exit 1
fi

echo "✅ Stage 2 completed successfully!"
echo ""

# Pipeline completion
echo "🎉 Pipeline Complete!"
echo "===================="
echo "$(date): Two-stage training pipeline completed successfully"
echo ""
echo "📊 Key Features Implemented:"
echo "  ✅ Local Metric-Aligned Gaussian Posterior: z = μ + L ε"
echo "  ✅ Covariance: Σ = α G(μ) (proper geometry alignment)"
echo "  ✅ Reparameterized: Fully differentiable pathwise gradients"
echo "  ✅ Volume Element Prior: p(z) ∝ √det(G̃(z))"
echo "  ✅ Clean KL: KL(q||p) = 1/2 E_q[(z-μ)^T G̃(z) (z-μ)]"
echo "  ✅ Metric Normalization: Prevents global scale collapse"
echo "  ✅ Float32 Computation: AMP-safe for mixed precision"
echo ""
echo "🔗 Results logged to WandB with tags: local_metric_gaussian, geometry_aware"
echo "📁 Checkpoints and visualizations saved in outputs/"
echo ""
echo "🧪 To run with custom parameters:"
echo "   LATENT_DIM=32 ARCHITECTURE=cnn EPOCHS_STAGE2=100 ./run.sh"
echo ""
echo "Pipeline completed at: $(date)"


