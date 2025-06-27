#!/bin/bash
#SBATCH --job-name=rlvae_weekend
#SBATCH --output=logs/weekend_experiments_%j.out
#SBATCH --error=logs/weekend_experiments_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=antoine.laforgue@etu.minesparis.psl.eu
#SBATCH --partition=gpu

# RlVAE Weekend Experiment Suite
# ==============================
# This script runs a comprehensive set of experiments over the weekend
# to validate the new modular architecture and compare model variants

set -e  # Exit on any error

echo "🚀 Starting RlVAE Weekend Experiment Suite"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo ""

# Create logs directory
mkdir -p logs
mkdir -p outputs

# Activate conda environment (adjust path as needed)
source /scratch/alaforgu/miniconda3/etc/profile.d/conda.sh
conda activate longitudinal_env  # Adjust environment name as needed

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export WANDB_MODE=online

echo "✅ Environment setup complete"
echo ""

# Function to run experiment with error handling
run_experiment() {
    local experiment_name=$1
    local command=$2
    local log_file="logs/${experiment_name}_${SLURM_JOB_ID}.log"
    
    echo "🧪 Running experiment: $experiment_name"
    echo "Command: $command"
    echo "Log file: $log_file"
    echo "Start time: $(date)"
    
    # Run experiment with timeout and logging
    timeout 12h bash -c "$command" > "$log_file" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ Experiment $experiment_name completed successfully"
    else
        echo "❌ Experiment $experiment_name failed or timed out"
    fi
    
    echo "End time: $(date)"
    echo "----------------------------------------"
    echo ""
}

# Function to run quick validation
run_validation() {
    echo "🔍 Running validation tests..."
    
    # Test hybrid model
    python test_hybrid_model.py > logs/validation_hybrid_${SLURM_JOB_ID}.log 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Hybrid model validation passed"
    else
        echo "❌ Hybrid model validation failed"
        return 1
    fi
    
    # Test modular components
    python test_modular_components.py > logs/validation_components_${SLURM_JOB_ID}.log 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Modular components validation passed"
    else
        echo "❌ Modular components validation failed"
        return 1
    fi
    
    echo "✅ All validation tests passed"
    echo ""
}

# Main experiment sequence
main() {
    echo "📋 Experiment Plan:"
    echo "1. Validation tests"
    echo "2. Quick training with hybrid model"
    echo "3. Full training with hybrid model"
    echo "4. Model comparison study"
    echo "5. Hyperparameter sweep (if time permits)"
    echo ""
    
    # Step 1: Validation
    run_validation
    
    # Step 2: Quick training with hybrid model
    run_experiment "hybrid_quick" \
        "python run_experiment.py model=hybrid_rlvae training=quick visualization=standard experiment.name=weekend_hybrid_quick"
    
    # Step 3: Full training with hybrid model
    run_experiment "hybrid_full" \
        "python run_experiment.py model=hybrid_rlvae training=full_data visualization=standard experiment.name=weekend_hybrid_full"
    
    # Step 4: Model comparison study
    run_experiment "comparison_study" \
        "python run_experiment.py experiment=comparison_study experiment.name=weekend_comparison"
    
    # Step 5: Hyperparameter sweep (if time permits)
    run_experiment "hyperparameter_sweep" \
        "python run_experiment.py experiment=hyperparameter_sweep -m experiment.name=weekend_sweep"
    
    # Step 6: Performance benchmarks
    run_experiment "performance_benchmark" \
        "python -c \"
import torch
import time
from src.models.hybrid_rlvae import create_hybrid_model
from omegaconf import OmegaConf

# Create config
config = OmegaConf.create({
    'input_dim': [3, 64, 64],
    'latent_dim': 16,
    'n_flows': 8,
    'flow_hidden_size': 256,
    'flow_n_blocks': 2,
    'flow_n_hidden': 1,
    'epsilon': 1e-6,
    'beta': 1.0,
    'riemannian_beta': 1.0,
    'posterior': {'type': 'gaussian'},
    'loop': {'mode': 'open', 'penalty': 1.0},
    'sampling': {'use_riemannian': True, 'method': 'enhanced'},
    'metric': {'temperature_override': 0.7, 'regularization_override': None},
    'pretrained': {
        'encoder_path': 'data/pretrained/encoder.pt',
        'decoder_path': 'data/pretrained/decoder.pt',
        'metric_path': 'data/pretrained/metric.pt'
    }
})

# Create model
model = create_hybrid_model(config)
model = model.cuda()

# Benchmark metric computations
batch_sizes = [1, 4, 16, 64, 128]
n_iterations = 100

print('Performance Benchmark Results:')
print('Batch Size | Time per call (ms) | Throughput (samples/s)')
print('-----------|-------------------|----------------------')

for batch_size in batch_sizes:
    z = torch.randn(batch_size, 16, device='cuda')
    
    # Warmup
    for _ in range(10):
        _ = model.modular_metric.compute_metric(z)
        _ = model.modular_metric.compute_inverse_metric(z)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(n_iterations):
        _ = model.modular_metric.compute_metric(z)
        _ = model.modular_metric.compute_inverse_metric(z)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_iterations * 1000  # ms
    throughput = batch_size / (avg_time / 1000)  # samples/s
    
    print(f'{batch_size:10d} | {avg_time:17.2f} | {throughput:20.0f}')
\" > logs/performance_benchmark_${SLURM_JOB_ID}.log 2>&1"
    
    echo "🎉 All experiments completed!"
    echo "Results saved to: outputs/"
    echo "Logs saved to: logs/"
    echo ""
    
    # Generate summary report
    echo "📊 Generating summary report..."
    python -c "
import os
import glob
from pathlib import Path

# Find all result files
result_files = glob.glob('outputs/*.yaml')
print('📋 Experiment Summary Report')
print('=' * 50)
print(f'Total experiments completed: {len(result_files)}')
print('')

for result_file in sorted(result_files):
    print(f'📄 {Path(result_file).name}')
    try:
        import yaml
        with open(result_file, 'r') as f:
            data = yaml.safe_load(f)
            if 'test_results' in data:
                results = data['test_results']
                print(f'   Test Loss: {results.get(\"test_loss\", \"N/A\"):.2f}')
                print(f'   Recon Loss: {results.get(\"test_recon_loss\", \"N/A\"):.2f}')
                print(f'   KL Loss: {results.get(\"test_kl_loss\", \"N/A\"):.2f}')
            print('')
    except Exception as e:
        print(f'   Error reading file: {e}')
        print('')

print('✅ Summary report generated')
" > logs/summary_report_${SLURM_JOB_ID}.txt 2>&1
    
    echo "📊 Summary report saved to: logs/summary_report_${SLURM_JOB_ID}.txt"
}

# Run main function
main

echo "🏁 Weekend experiment suite completed at $(date)"
echo "Check logs/ directory for detailed results" 