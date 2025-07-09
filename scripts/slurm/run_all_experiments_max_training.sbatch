#!/bin/bash
#SBATCH --job-name=rlvae_max_training
#SBATCH --output=logs/rlvae_max_training_%j.out
#SBATCH --error=logs/rlvae_max_training_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu

# Load modules (adjust for your cluster)
module purge
module load cuda/11.8
module load python/3.10

# Activate conda environment (adjust path)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlvae_env

# Create logs directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export WANDB_MODE=online
export WANDB_PROJECT="rlvae_max_training_experiments"

echo "=========================================="
echo "🚀 Starting Maximum Training Experiments"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "=========================================="

# Function to run experiment with error handling
run_experiment() {
    local exp_name=$1
    local cmd=$2
    local log_file="logs/${exp_name}_${SLURM_JOB_ID}.log"
    
    echo "=========================================="
    echo "🎯 Starting Experiment: $exp_name"
    echo "Command: $cmd"
    echo "Log file: $log_file"
    echo "Time: $(date)"
    echo "=========================================="
    
    # Run the experiment
    eval $cmd > $log_file 2>&1
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Experiment $exp_name completed successfully!"
        echo "Log saved to: $log_file"
    else
        echo "❌ Experiment $exp_name failed with exit code $exit_code"
        echo "Check log file: $log_file"
        echo "Continuing with next experiment..."
    fi
    
    echo "=========================================="
    echo ""
}

# Wait between experiments to avoid GPU memory issues
sleep_between_experiments() {
    echo "⏳ Waiting 30 seconds before next experiment..."
    sleep 30
}

# Clear GPU memory
clear_gpu_memory() {
    echo "🧹 Clearing GPU memory..."
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
}

# ==========================================
# EXPERIMENT 1: MLP Pipeline (Maximum Training)
# ==========================================
echo "🔬 EXPERIMENT 1: MLP Pipeline (Maximum Training)"
run_experiment "mlp_max_training" \
    "WANDB_PROJECT=rlvae_max_training_experiments python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
    experiment.stage1.architecture=mlp \
    experiment.stage1.latent_dim=32 \
    experiment.stage1.epochs=100 \
    experiment.stage1.preset=diverse \
    experiment.stage1.temperature=1.0 \
    experiment.stage1.regularization=0.001 \
    experiment.stage1.n_heatmaps=12 \
    experiment.stage2.model=mlp_rlvae \
    experiment.stage2.visualization=full \
    experiment.log_level=INFO"

sleep_between_experiments
clear_gpu_memory

# ==========================================
# EXPERIMENT 2: ResNet Pipeline (Maximum Training)
# ==========================================
echo "🔬 EXPERIMENT 2: ResNet Pipeline (Maximum Training)"
run_experiment "resnet_max_training" \
    "WANDB_PROJECT=rlvae_max_training_experiments python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
    experiment.stage1.architecture=resnet \
    experiment.stage1.latent_dim=32 \
    experiment.stage1.epochs=100 \
    experiment.stage1.preset=diverse \
    experiment.stage1.temperature=1.0 \
    experiment.stage1.regularization=0.001 \
    experiment.stage1.n_heatmaps=12 \
    experiment.stage2.model=resnet_rlvae \
    experiment.stage2.visualization=full \
    experiment.log_level=INFO"

sleep_between_experiments
clear_gpu_memory

# ==========================================
# EXPERIMENT 3: MLP Pipeline (High Dimensional)
# ==========================================
echo "🔬 EXPERIMENT 3: MLP Pipeline (High Dimensional)"
run_experiment "mlp_high_dim" \
    "WANDB_PROJECT=rlvae_max_training_experiments python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
    experiment.stage1.architecture=mlp \
    experiment.stage1.latent_dim=64 \
    experiment.stage1.epochs=100 \
    experiment.stage1.preset=diverse \
    experiment.stage1.temperature=0.8 \
    experiment.stage1.regularization=0.005 \
    experiment.stage1.n_heatmaps=15 \
    experiment.stage2.model=mlp_rlvae \
    experiment.stage2.visualization=full \
    experiment.log_level=INFO"

sleep_between_experiments
clear_gpu_memory

# ==========================================
# EXPERIMENT 4: ResNet Pipeline (High Dimensional)
# ==========================================
echo "🔬 EXPERIMENT 4: ResNet Pipeline (High Dimensional)"
run_experiment "resnet_high_dim" \
    "WANDB_PROJECT=rlvae_max_training_experiments python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
    experiment.stage1.architecture=resnet \
    experiment.stage1.latent_dim=64 \
    experiment.stage1.epochs=100 \
    experiment.stage1.preset=diverse \
    experiment.stage1.temperature=0.8 \
    experiment.stage1.regularization=0.005 \
    experiment.stage1.n_heatmaps=15 \
    experiment.stage2.model=resnet_rlvae \
    experiment.stage2.visualization=full \
    experiment.log_level=INFO"

sleep_between_experiments
clear_gpu_memory

# ==========================================
# EXPERIMENT 5: MLP Pipeline (Conservative)
# ==========================================
echo "🔬 EXPERIMENT 5: MLP Pipeline (Conservative)"
run_experiment "mlp_conservative" \
    "WANDB_PROJECT=rlvae_max_training_experiments python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
    experiment.stage1.architecture=mlp \
    experiment.stage1.latent_dim=16 \
    experiment.stage1.epochs=100 \
    experiment.stage1.preset=conservative \
    experiment.stage1.temperature=0.3 \
    experiment.stage1.regularization=0.02 \
    experiment.stage1.n_heatmaps=8 \
    experiment.stage2.model=mlp_rlvae \
    experiment.stage2.visualization=standard \
    experiment.log_level=INFO"

sleep_between_experiments
clear_gpu_memory

# ==========================================
# EXPERIMENT 6: ResNet Pipeline (Conservative)
# ==========================================
echo "🔬 EXPERIMENT 6: ResNet Pipeline (Conservative)"
run_experiment "resnet_conservative" \
    "WANDB_PROJECT=rlvae_max_training_experiments python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
    experiment.stage1.architecture=resnet \
    experiment.stage1.latent_dim=16 \
    experiment.stage1.epochs=100 \
    experiment.stage1.preset=conservative \
    experiment.stage1.temperature=0.3 \
    experiment.stage1.regularization=0.02 \
    experiment.stage1.n_heatmaps=8 \
    experiment.stage2.model=resnet_rlvae \
    experiment.stage2.visualization=standard \
    experiment.log_level=INFO"

# ==========================================
# FINAL SUMMARY
# ==========================================
echo "=========================================="
echo "🎉 ALL EXPERIMENTS COMPLETED!"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date -d @$SLURM_JOB_START_TIME)"
echo "End time: $(date)"
echo "Total experiments: 6"
echo "Logs directory: logs/"
echo "=========================================="

# Create summary report
cat > "logs/experiment_summary_${SLURM_JOB_ID}.txt" << EOF
RLVAE Maximum Training Experiments Summary
==========================================
Job ID: $SLURM_JOB_ID
Start Time: $(date -d @$SLURM_JOB_START_TIME)
End Time: $(date)
Node: $SLURM_NODELIST
GPU: $CUDA_VISIBLE_DEVICES

Experiments Run:
1. mlp_max_training - MLP with maximum diversity (32D, 100 epochs)
2. resnet_max_training - ResNet with maximum diversity (32D, 100 epochs)
3. mlp_high_dim - MLP high dimensional (64D, 100 epochs)
4. resnet_high_dim - ResNet high dimensional (64D, 100 epochs)
5. mlp_conservative - MLP conservative (16D, 100 epochs)
6. resnet_conservative - ResNet conservative (16D, 100 epochs)

All experiments use:
- 100 epochs for both vanilla VAE and RLVAE
- Full visualization level for diverse experiments
- Standard visualization for conservative experiments
- Maximum diversity settings where applicable

Log files available in logs/ directory.
EOF

echo "📊 Summary report created: logs/experiment_summary_${SLURM_JOB_ID}.txt"
echo "==========================================" 