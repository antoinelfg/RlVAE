#!/bin/bash

# RlVAE Experiment Monitor
# =======================
# This script monitors the status of running experiments and provides
# a summary of results

echo "🔍 RlVAE Experiment Monitor"
echo "==========================="
echo "Time: $(date)"
echo ""

# Check running SLURM jobs
echo "📊 Running SLURM Jobs:"
echo "----------------------"
squeue -u $USER --format="%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R" | grep rlvae || echo "No RlVAE jobs running"
echo ""

# Check recent log files
echo "📋 Recent Log Files:"
echo "-------------------"
if [ -d "logs" ]; then
    find logs -name "*.log" -type f -mtime -1 | head -10 | while read file; do
        echo "📄 $(basename $file)"
        echo "   Size: $(du -h $file | cut -f1)"
        echo "   Modified: $(stat -c %y $file | cut -d' ' -f1,2)"
        echo ""
    done
else
    echo "No logs directory found"
fi

# Check output files
echo "📊 Output Files:"
echo "---------------"
if [ -d "outputs" ]; then
    find outputs -name "*.yaml" -type f -mtime -1 | while read file; do
        echo "📄 $(basename $file)"
        if command -v python3 >/dev/null 2>&1; then
            python3 -c "
import yaml
try:
    with open('$file', 'r') as f:
        data = yaml.safe_load(f)
        if 'test_results' in data:
            results = data['test_results']
            print(f'   Test Loss: {results.get(\"test_loss\", \"N/A\"):.2f}')
            print(f'   Recon Loss: {results.get(\"test_recon_loss\", \"N/A\"):.2f}')
            print(f'   KL Loss: {results.get(\"test_kl_loss\", \"N/A\"):.2f}')
        else:
            print('   No test results found')
except Exception as e:
    print(f'   Error reading file: {e}')
"
        fi
        echo ""
    done
else
    echo "No outputs directory found"
fi

# Check wandb status
echo "📈 Weights & Biases Status:"
echo "---------------------------"
if command -v wandb >/dev/null 2>&1; then
    wandb status 2>/dev/null || echo "WandB not logged in or no active runs"
else
    echo "WandB CLI not available"
fi
echo ""

# Check disk usage
echo "💾 Disk Usage:"
echo "-------------"
df -h . | tail -1
echo ""

# Check GPU usage
echo "🎮 GPU Usage:"
echo "------------"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while IFS=, read -r index name util mem_used mem_total; do
        echo "GPU $index ($name): ${util}% util, ${mem_used}MB/${mem_total}MB"
    done
else
    echo "nvidia-smi not available"
fi
echo ""

echo "✅ Monitor complete" 