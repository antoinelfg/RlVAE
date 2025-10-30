#!/bin/bash
# Monitor training progress and check for rescaled metric usage

echo "🔍 Monitoring Training Progress"
echo "========================================"
echo ""

# Find the most recent wandb run
LATEST_RUN=$(ls -td /home/alaforgu/wandb/run-* 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "❌ No wandb run found"
    exit 1
fi

echo "📂 Monitoring run: $LATEST_RUN"
echo ""

# Function to check for key patterns
check_pattern() {
    local pattern="$1"
    local description="$2"
    
    if [ -f "$LATEST_RUN/logs/debug.log" ]; then
        result=$(grep "$pattern" "$LATEST_RUN/logs/debug.log" 2>/dev/null | tail -5)
        if [ -n "$result" ]; then
            echo "✅ $description:"
            echo "$result" | sed 's/^/   /'
            echo ""
        fi
    fi
}

# Wait for logs to appear
echo "⏳ Waiting for training logs..."
for i in {1..10}; do
    if [ -f "$LATEST_RUN/logs/debug.log" ]; then
        echo "✅ Logs found!"
        break
    fi
    sleep 2
    echo -n "."
done
echo ""
echo ""

# Check for rescaled metric loading
echo "🔍 Checking for Rescaled Metric..."
echo "----------------------------------------"
check_pattern "metric_rescaled" "Metric file path"
check_pattern "G⁻¹.*eigenvalues" "G⁻¹ eigenvalues"
check_pattern "RESCALED" "Rescaling confirmation"
echo ""

# Check for KL divergence
echo "🔍 Checking KL Divergence..."
echo "----------------------------------------"
check_pattern "FINAL KL_LOSS" "KL Loss"
check_pattern "log_q mean" "Log Q"
echo ""

# Check for errors
echo "🔍 Checking for Errors..."
echo "----------------------------------------"
if [ -f "$LATEST_RUN/logs/debug.log" ]; then
    errors=$(grep -i "error\|exception\|failed" "$LATEST_RUN/logs/debug.log" 2>/dev/null | tail -5)
    if [ -n "$errors" ]; then
        echo "⚠️  Errors found:"
        echo "$errors" | sed 's/^/   /'
    else
        echo "✅ No errors detected"
    fi
fi
echo ""

# Show recent training progress
echo "📊 Recent Training Progress..."
echo "----------------------------------------"
if [ -f "$LATEST_RUN/logs/debug.log" ]; then
    tail -30 "$LATEST_RUN/logs/debug.log" | grep -E "Epoch|train_loss|val_loss|KL" | tail -10
fi
echo ""

echo "========================================"
echo "💡 To see full logs:"
echo "   tail -f $LATEST_RUN/logs/debug.log"

