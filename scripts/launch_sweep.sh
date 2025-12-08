#!/bin/bash
# =============================================================================
# RLVAE Hyperparameter Sweep Launcher
# =============================================================================
#
# Usage:
#   ./scripts/launch_sweep.sh              # Initialize sweep and print instructions
#   ./scripts/launch_sweep.sh <sweep_id>   # Launch agent for existing sweep
#   ./scripts/launch_sweep.sh <sweep_id> N # Launch N agents in background
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SWEEP_CONFIG="$PROJECT_ROOT/conf/sweep/stage_c_bayesian.yaml"

# WandB project settings (modify as needed)
WANDB_PROJECT="${WANDB_PROJECT:-RlVAE-posterior}"
WANDB_ENTITY="${WANDB_ENTITY:-}"  # Leave empty to use default entity

cd "$PROJECT_ROOT"

# Activate conda/venv if needed
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

if [ -z "$1" ]; then
    # No sweep_id provided - initialize new sweep
    echo "=============================================="
    echo "Initializing new WandB sweep..."
    echo "=============================================="
    echo ""
    
    if [ -n "$WANDB_ENTITY" ]; then
        wandb sweep --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" "$SWEEP_CONFIG"
    else
        wandb sweep --project "$WANDB_PROJECT" "$SWEEP_CONFIG"
    fi
    
    echo ""
    echo "=============================================="
    echo "Sweep initialized!"
    echo ""
    echo "To launch agents, run:"
    echo "  ./scripts/launch_sweep.sh <sweep_id>"
    echo ""
    echo "To launch N parallel agents:"
    echo "  ./scripts/launch_sweep.sh <sweep_id> N"
    echo "=============================================="
    
elif [ -z "$2" ]; then
    # sweep_id provided, no count - launch single agent
    SWEEP_ID="$1"
    echo "=============================================="
    echo "Launching single agent for sweep: $SWEEP_ID"
    echo "=============================================="
    
    if [ -n "$WANDB_ENTITY" ]; then
        wandb agent "$WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID"
    else
        wandb agent "$WANDB_PROJECT/$SWEEP_ID"
    fi
    
else
    # sweep_id and count provided - launch N agents in background
    SWEEP_ID="$1"
    N_AGENTS="$2"
    
    echo "=============================================="
    echo "Launching $N_AGENTS agents for sweep: $SWEEP_ID"
    echo "=============================================="
    
    for i in $(seq 1 "$N_AGENTS"); do
        echo "Starting agent $i/$N_AGENTS..."
        if [ -n "$WANDB_ENTITY" ]; then
            nohup wandb agent "$WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID" > "sweep_agent_${i}.log" 2>&1 &
        else
            nohup wandb agent "$WANDB_PROJECT/$SWEEP_ID" > "sweep_agent_${i}.log" 2>&1 &
        fi
        sleep 2  # Small delay between agent launches
    done
    
    echo ""
    echo "All agents launched in background."
    echo "Logs: sweep_agent_*.log"
    echo ""
    echo "To monitor:"
    echo "  tail -f sweep_agent_1.log"
    echo ""
    echo "To stop all agents:"
    echo "  pkill -f 'wandb agent'"
    echo "=============================================="
fi


