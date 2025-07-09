#!/bin/bash

# Extended Hyperparameter Sweep Runner
# ====================================
# This script runs extended sweeps by breaking them into multiple
# 47-hour chunks that fit within the GPU partition time limits

# Configuration
SWEEP_CONFIG=${1:-"comprehensive_hyperparameter_sweep"}
TOTAL_RUNS=${2:-"100"}
RUNS_PER_CHUNK=${3:-"20"}
AGENT_COUNT=${4:-"2"}

# Calculate number of chunks needed
CHUNKS_NEEDED=$(( (TOTAL_RUNS + RUNS_PER_CHUNK - 1) / RUNS_PER_CHUNK ))

echo "=========================================="
echo "🚀 Extended Hyperparameter Sweep Runner"
echo "=========================================="
echo "Sweep Config: $SWEEP_CONFIG"
echo "Total Runs: $TOTAL_RUNS"
echo "Runs per chunk: $RUNS_PER_CHUNK"
echo "Agent count: $AGENT_COUNT"
echo "Chunks needed: $CHUNKS_NEEDED"
echo "=========================================="

# Function to submit a chunk
submit_chunk() {
    local chunk_num=$1
    local max_runs=$2
    
    echo "📊 Submitting chunk $chunk_num with max $max_runs runs..."
    
    # Submit job with dependency on previous chunk (if any)
    if [ $chunk_num -eq 1 ]; then
        # First chunk - no dependency
        JOB_ID=$(sbatch --parsable run_hyperparameter_sweep.sbatch \
            "$SWEEP_CONFIG" "$AGENT_COUNT" "$max_runs")
    else
        # Subsequent chunks - depend on previous chunk
        JOB_ID=$(sbatch --parsable --dependency=afterany:$PREV_JOB_ID \
            run_hyperparameter_sweep.sbatch \
            "$SWEEP_CONFIG" "$AGENT_COUNT" "$max_runs")
    fi
    
    echo "   Job ID: $JOB_ID"
    echo "$JOB_ID" >> sweep_job_ids.txt
    
    PREV_JOB_ID=$JOB_ID
}

# Create file to track job IDs
echo "# Extended sweep job IDs for $SWEEP_CONFIG" > sweep_job_ids.txt
echo "# Started: $(date)" >> sweep_job_ids.txt

# Submit all chunks
REMAINING_RUNS=$TOTAL_RUNS
for (( chunk=1; chunk<=CHUNKS_NEEDED; chunk++ )); do
    if [ $REMAINING_RUNS -gt $RUNS_PER_CHUNK ]; then
        CHUNK_RUNS=$RUNS_PER_CHUNK
    else
        CHUNK_RUNS=$REMAINING_RUNS
    fi
    
    submit_chunk $chunk $CHUNK_RUNS
    REMAINING_RUNS=$((REMAINING_RUNS - CHUNK_RUNS))
done

echo ""
echo "=========================================="
echo "✅ Extended Sweep Submitted Successfully!"
echo "=========================================="
echo "Total jobs submitted: $CHUNKS_NEEDED"
echo "Job IDs saved to: sweep_job_ids.txt"
echo ""
echo "📊 Monitor progress with:"
echo "   squeue -u $USER"
echo "   tail -f logs/sweep_*.out"
echo ""
echo "🎯 WandB dashboard:"
echo "   https://wandb.ai/your-entity/rlvae-hyperparameter-optimization"
echo "=========================================="

# Create monitoring script
cat > monitor_extended_sweep.sh << 'EOF'
#!/bin/bash
echo "🔍 Extended Sweep Monitoring"
echo "=========================="

if [ ! -f "sweep_job_ids.txt" ]; then
    echo "❌ No sweep_job_ids.txt found"
    exit 1
fi

# Read job IDs
JOB_IDS=$(grep -v "^#" sweep_job_ids.txt | tr '\n' ' ')

echo "Monitoring jobs: $JOB_IDS"
echo ""

# Check job status
echo "📊 Job Status:"
squeue -j "$JOB_IDS" --format="%.10i %.9P %.20j %.8u %.8T %.10M %.6D %R" 2>/dev/null || {
    echo "No jobs currently in queue"
}

echo ""
echo "📈 Recent logs:"
echo "==============="
ls -t logs/sweep_*.out 2>/dev/null | head -3 | while read log_file; do
    echo "--- $log_file (last 5 lines) ---"
    tail -5 "$log_file" 2>/dev/null || echo "Cannot read $log_file"
    echo ""
done

echo "🔄 Run 'watch -n 30 ./monitor_extended_sweep.sh' for live updates"
EOF

chmod +x monitor_extended_sweep.sh

echo "📝 Created monitoring script: monitor_extended_sweep.sh"
echo "   Run with: ./monitor_extended_sweep.sh" 