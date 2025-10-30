#!/bin/bash
# Alpha sweep to test spatial mismatch hypothesis
# Tests α = [0.5, 1.0, 2.0, 5.0, 10.0]
# Hypothesis: Larger α → wider Σ_μ → samples reach high-volume regions → positive KL
# 
# Usage:
#   bash scripts/run_alpha_sweep.sh                    # Run all missing experiments
#   bash scripts/run_alpha_sweep.sh --alpha 2.0        # Run specific alpha
#   bash scripts/run_alpha_sweep.sh --list             # List what's missing

set -e

cd /home/alaforgu/scratch/longitudinal_experiments/RlVAE

# Create logs directory
mkdir -p logs/alpha_sweep

# Define alpha values to test
ALPHAS=(0.5 1.0 2.0 5.0 10.0)

# Function to check if experiment is complete
is_complete() {
    local alpha=$1
    local log_file="logs/alpha_sweep/alpha_${alpha//./p}.log"
    
    if [[ ! -f "$log_file" ]]; then
        return 1  # Not started
    fi
    
    # Check if log contains completion markers
    if grep -q "FINAL KL LOSS:" "$log_file" && grep -q "Epoch 4/5" "$log_file"; then
        return 0  # Complete
    else
        return 1  # Incomplete
    fi
}

# Function to run single alpha experiment
run_alpha() {
    local alpha=$1
    local alpha_str="${alpha//./p}"
    
    echo "[ALPHA ${alpha}] Starting experiment..."
    echo "  Log: logs/alpha_sweep/alpha_${alpha_str}.log"
    echo "  WandB: rlvae-alpha-sweep"
    echo ""
    
    RLVAE_DEBUG=1 python run_experiment.py \
        +experiment=rlvae_debug_alpha_${alpha_str} \
        settings.pipeline.mode=three_stage \
        settings.pipeline.run_stage_a=false \
        settings.pipeline.run_stage_b=true \
        settings.pipeline.run_stage_c=true \
        settings.training.stage_overrides.stage_b.enabled=true \
        2>&1 | tee logs/alpha_sweep/alpha_${alpha_str}.log
    
    echo ""
    echo "[ALPHA ${alpha}] Complete!"
    echo ""
}

# Parse command line arguments
if [[ "$1" == "--list" ]]; then
    echo "======================================================================"
    echo "ALPHA SWEEP STATUS CHECK"
    echo "======================================================================"
    echo ""
    
    missing=()
    complete=()
    
    for alpha in "${ALPHAS[@]}"; do
        if is_complete "$alpha"; then
            complete+=("$alpha")
        else
            missing+=("$alpha")
        fi
    done
    
    echo "✅ COMPLETE (${#complete[@]}/5):"
    for alpha in "${complete[@]}"; do
        echo "   α = $alpha"
    done
    
    echo ""
    echo "❌ MISSING (${#missing[@]}/5):"
    for alpha in "${missing[@]}"; do
        echo "   α = $alpha"
    done
    
    echo ""
    if [[ ${#missing[@]} -eq 0 ]]; then
        echo "🎉 ALL EXPERIMENTS COMPLETE!"
        echo "   Run: python scripts/analyze_alpha_sweep.py"
    else
        echo "🚀 TO RUN MISSING EXPERIMENTS:"
        echo "   bash scripts/run_alpha_sweep.sh"
        echo ""
        echo "🚀 TO RUN SPECIFIC ALPHA:"
        for alpha in "${missing[@]}"; do
            echo "   bash scripts/run_alpha_sweep.sh --alpha $alpha"
        done
    fi
    
    exit 0
fi

if [[ "$1" == "--alpha" && -n "$2" ]]; then
    # Run specific alpha
    alpha=$2
    alpha_str="${alpha//./p}"
    
    echo "======================================================================"
    echo "RUNNING SPECIFIC ALPHA: α = $alpha"
    echo "======================================================================"
    echo ""
    
    if is_complete "$alpha"; then
        echo "✅ α = $alpha is already complete!"
        echo "   Log: logs/alpha_sweep/alpha_${alpha_str}.log"
        exit 0
    fi
    
    run_alpha "$alpha"
    exit 0
fi

# Default: run all missing experiments
echo "======================================================================"
echo "ALPHA SWEEP: Testing Spatial Mismatch Hypothesis"
echo "======================================================================"
echo ""
echo "Hypothesis: Larger α expands Σ_μ, allowing z0 to reach high-volume regions"
echo "Expected: log|G⁻¹(z0)| increases with α, KL becomes positive"
echo ""

# Check what's missing
missing=()
for alpha in "${ALPHAS[@]}"; do
    if ! is_complete "$alpha"; then
        missing+=("$alpha")
    fi
done

if [[ ${#missing[@]} -eq 0 ]]; then
    echo "🎉 ALL EXPERIMENTS ALREADY COMPLETE!"
    echo ""
    echo "Next steps:"
    echo "1. Analyze results: python scripts/analyze_alpha_sweep.py"
    echo "2. Check WandB: project 'rlvae-alpha-sweep'"
    exit 0
fi

echo "Running ${#missing[@]}/5 missing experiments:"
for alpha in "${missing[@]}"; do
    echo "  α = $alpha"
done
echo ""

# Run missing experiments
for alpha in "${missing[@]}"; do
    run_alpha "$alpha"
done

echo "======================================================================"
echo "ALPHA SWEEP COMPLETE!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. Extract metrics: python scripts/analyze_alpha_sweep.py"
echo "2. Compare in WandB: project 'rlvae-alpha-sweep'"
echo "3. Check key metrics:"
echo "   - Δ(z0-μ) in log|G⁻¹|: Should decrease as α increases"
echo "   - Correlation(||z-μ||, log|G⁻¹|): Should become less negative"
echo "   - KL divergence: Should become positive for large α"
echo ""

