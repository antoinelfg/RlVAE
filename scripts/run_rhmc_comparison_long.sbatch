#!/bin/bash
#SBATCH --job-name=rhmc_comparison_long
#SBATCH --output=logs/rhmc_comparison_long_%j.out
#SBATCH --error=logs/rhmc_comparison_long_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=antoine.laforgue@mines-paris.org

# Script SLURM pour comparaison approfondie des posteriors
# Comparaison: Standard (riemannian_metric) vs Nouveau (riemannian_rhmc)
# Configuration: 200 epochs Stage A, 200 epochs Stage C

echo "🚀 SLURM Job: Comparaison Posteriors RHMC - Configuration Longue"
echo "================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Setup environment
module load cuda/11.8
module load python/3.9
source ~/miniconda3/bin/activate rlvae

# Navigate to project directory
cd /home/alaforgu/scratch/longitudinal_experiments/RlVAE

# Create logs directory if it doesn't exist
mkdir -p logs

echo "📋 Configuration:"
echo "  - Stage A: 200 epochs"
echo "  - Stage C: 200 epochs" 
echo "  - Data: ellipse_sequences"
echo "  - Latent dim: 2"
echo "  - Comparaison: riemannian_metric vs riemannian_rhmc"
echo ""

# Create experiment configurations with 200 epochs
echo "🔧 Creating long training configurations..."

# Configuration 1: Standard (référence) avec 200 epochs
cat > conf/experiment/rlvae_three_stage_long_standard.yaml << 'EOF'
# @package experiment

# Long training configuration - Standard posterior (référence)
defaults:
  - rlvae_three_stage_pipeline

type: three_stage
name: rlvae_three_stage_long_standard
description: "Long training with standard riemannian_metric posterior (200 epochs A&C)"

# Extended training configuration
stage_a:
  model: vanilla_vae
  epochs: 200              # Extended from 50
  batch_size: 64
  lr: 1e-4
  beta: 1.0
  architecture: mlp
  latent_dim: 2

stage_b:
  implementation: rhvae
  n_centroids: 300
  temperature: 0.7
  regularization: 0.01
  centroid_method: kmedoids
  neighbor_mode: global
  knn_k: 300

stage_c:
  model: rhvae_original
  epochs: 200              # Extended from 30
  batch_size: 64
  riemannian_beta: 8.0
  allow_metric_updates: false
  force_rhvae_components: false
  update_metric_during_training: false

# Model configuration (standard posterior)
model:
  latent_dim: 2
  posterior:
    type: "riemannian_metric"  # Standard posterior
  metric:
    trainable: false
    init_from_fixed: true
    temperature_override: null

training:
  model:
    latent_dim: 2
    posterior:
      type: riemannian_metric

# WandB configuration
wandb:
  project: rlvae-rhmc-comparison-long
  run_name_template: "STANDARD_200ep_{experiment.name}_{data.name}_ld{model.latent_dim}"
  tags: ["long_training", "standard_posterior", "200_epochs"]
EOF

# Configuration 2: RHMC (nouveau) avec 200 epochs
cat > conf/experiment/rlvae_three_stage_long_rhmc.yaml << 'EOF'
# @package experiment

# Long training configuration - RHMC posterior (nouveau)
defaults:
  - rlvae_three_stage_pipeline

type: three_stage
name: rlvae_three_stage_long_rhmc
description: "Long training with new riemannian_rhmc posterior (200 epochs A&C)"

# Extended training configuration
stage_a:
  model: vanilla_vae
  epochs: 200              # Extended from 50
  batch_size: 64
  lr: 1e-4
  beta: 1.0
  architecture: mlp
  latent_dim: 2

stage_b:
  implementation: rhvae
  n_centroids: 300
  temperature: 0.7
  regularization: 0.01
  centroid_method: kmedoids
  neighbor_mode: global
  knn_k: 300

stage_c:
  model: rhvae_original
  epochs: 200              # Extended from 30
  batch_size: 64
  riemannian_beta: 8.0
  allow_metric_updates: false
  force_rhvae_components: false
  update_metric_during_training: false

# Model configuration (RHMC posterior)
model:
  latent_dim: 2
  posterior:
    type: "riemannian_rhmc"  # Nouveau posterior RHMC
    rhmc_steps: 3
    rhmc_step_size: 0.01
    rhmc_alpha: 1.0
    eps_regularization: 1e-6
    max_grad_norm: 5.0
    min_step_size: 1e-4
  metric:
    trainable: false
    init_from_fixed: true
    temperature_override: null

training:
  model:
    latent_dim: 2
    posterior:
      type: riemannian_rhmc

# WandB configuration
wandb:
  project: rlvae-rhmc-comparison-long
  run_name_template: "RHMC_200ep_{experiment.name}_{data.name}_ld{model.latent_dim}"
  tags: ["long_training", "rhmc_posterior", "200_epochs"]
EOF

echo "✅ Configurations créées:"
echo "  - rlvae_three_stage_long_standard.yaml (référence)"
echo "  - rlvae_three_stage_long_rhmc.yaml (RHMC)"
echo ""

# Function to run experiment with error handling
run_experiment() {
    local exp_name=$1
    local exp_type=$2
    
    echo "🚀 Lancement: $exp_type"
    echo "   Expérience: $exp_name"
    echo "   Début: $(date)"
    
    # Run experiment with timeout and error handling
    timeout 20h python -u run_experiment.py \
        experiment=$exp_name \
        data=ellipse_sequences \
        wandb.mode=online \
        seed=42 \
        > logs/${exp_name}_${SLURM_JOB_ID}.log 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $exp_type terminé avec succès"
    elif [ $exit_code -eq 124 ]; then
        echo "⏰ $exp_type interrompu par timeout (20h)"
    else
        echo "❌ $exp_type échoué (code: $exit_code)"
    fi
    
    echo "   Fin: $(date)"
    echo ""
    
    return $exit_code
}

# Launch experiments sequentially (to avoid GPU memory conflicts)
echo "🎯 Lancement des expériences séquentielles..."
echo ""

# Experiment 1: Standard posterior (référence)
run_experiment "rlvae_three_stage_long_standard" "STANDARD (Référence)"
standard_exit=$?

# Small break between experiments
echo "⏸️  Pause de 60 secondes entre expériences..."
sleep 60

# Experiment 2: RHMC posterior (nouveau)
run_experiment "rlvae_three_stage_long_rhmc" "RHMC (Nouveau)"
rhmc_exit=$?

# Summary
echo "📊 RÉSUMÉ FINAL"
echo "==============="
echo "Job ID: $SLURM_JOB_ID"
echo "Fin: $(date)"
echo ""
echo "Résultats:"
if [ $standard_exit -eq 0 ]; then
    echo "  ✅ STANDARD: Succès"
else
    echo "  ❌ STANDARD: Échec (code: $standard_exit)"
fi

if [ $rhmc_exit -eq 0 ]; then
    echo "  ✅ RHMC: Succès"
else
    echo "  ❌ RHMC: Échec (code: $rhmc_exit)"
fi

echo ""
echo "🔗 Résultats WandB:"
echo "   Projet: rlvae-rhmc-comparison-long"
echo "   Tags: long_training, 200_epochs"
echo ""
echo "📁 Logs détaillés:"
echo "   - logs/rlvae_three_stage_long_standard_${SLURM_JOB_ID}.log"
echo "   - logs/rlvae_three_stage_long_rhmc_${SLURM_JOB_ID}.log"
echo ""

# Exit with error if any experiment failed
if [ $standard_exit -ne 0 ] || [ $rhmc_exit -ne 0 ]; then
    echo "❌ Au moins une expérience a échoué"
    exit 1
else
    echo "🎉 Toutes les expériences terminées avec succès!"
    exit 0
fi
