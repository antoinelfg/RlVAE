# How to Launch the Chaos Sweep v2

## Step 1: Initialize the Sweep

```bash
cd /home/alaforgu/scratch/longitudinal_experiments/RlVAE

# Initialize the sweep (this creates it in WandB)
wandb sweep conf/sweep/chaos_sweep_v2.yaml
```

**Output will look like:**
```
wandb: Creating sweep from: conf/sweep/chaos_sweep_v2.yaml
wandb: Created sweep with ID: abc12345
wandb: View sweep at: https://wandb.ai/<entity>/<project>/sweeps/abc12345
```

**Save the full sweep path!** It will be something like:
```
<entity>/<project>/abc12345
```

For example: `antoine-laforgue-mines-paris-alumni/RlVAE-posterior/abc12345`

---

## Step 2: Launch Agents

### Option A: SLURM (Recommended for Cluster)

Launch multiple agents on the cluster:

```bash
# Set your sweep path (from Step 1)
SWEEP_PATH="antoine-laforgue-mines-paris-alumni/RlVAE-posterior/abc12345"

# Launch 15 agents (adjust number as needed)
for i in {1..15}; do
  sbatch scripts/slurm/sweep_agent.sbatch "$SWEEP_PATH"
done
```

**Monitor jobs:**
```bash
# Check job status
squeue -u $USER

# Check logs
tail -f sweep_logs/rlvae-sweep_*.out
```

**Stop all agents:**
```bash
# Cancel all sweep jobs
scancel -u $USER -n rlvae-sweep
```

---

### Option B: Local (For Testing)

```bash
# Set your sweep path
SWEEP_PATH="antoine-laforgue-mines-paris-alumni/RlVAE-posterior/abc12345"

# Launch single agent (foreground)
wandb agent "$SWEEP_PATH"

# Or launch multiple agents in background
for i in {1..5}; do
  nohup wandb agent "$SWEEP_PATH" > "sweep_agent_${i}.log" 2>&1 &
done

# Monitor
tail -f sweep_agent_1.log

# Stop all agents
pkill -f 'wandb agent'
```

---

## Step 3: Monitor Progress

1. **WandB Dashboard:**
   - Go to: https://wandb.ai/<entity>/<project>/sweeps/<sweep_id>
   - View parallel coordinates, parameter importance, best runs

2. **Check Run Status:**
   ```bash
   # In WandB UI, filter by:
   # - State: Finished (ignore Crashed/Failed)
   # - Sort by: val_mse (ascending)
   ```

3. **Monitor Logs:**
   ```bash
   # SLURM logs
   tail -f sweep_logs/rlvae-sweep_*.out
   
   # Local logs
   tail -f sweep_agent_*.log
   ```

---

## Quick Reference

| Action | Command |
|--------|---------|
| **Initialize sweep** | `wandb sweep conf/sweep/chaos_sweep_v2.yaml` |
| **Launch 15 SLURM agents** | `for i in {1..15}; do sbatch scripts/slurm/sweep_agent.sbatch "$SWEEP_PATH"; done` |
| **Launch 5 local agents** | `for i in {1..5}; do nohup wandb agent "$SWEEP_PATH" > "sweep_agent_${i}.log" 2>&1 &; done` |
| **Stop all agents** | `scancel -u $USER -n rlvae-sweep` (SLURM) or `pkill -f 'wandb agent'` (local) |
| **Check sweep status** | Visit WandB dashboard URL from Step 1 |

---

## Expected Timeline

- **Total runs:** 500 (set in `run_cap`)
- **With 15 agents:** ~33 runs per agent
- **Time per run:** ~30-60 minutes (75 epochs with early stopping)
- **Total time:** ~10-20 hours for full sweep

---

## Troubleshooting

**Problem:** Agents not starting
- **Check:** WandB login (`wandb login`)
- **Check:** Sweep path format (must be `entity/project/sweep_id`)

**Problem:** Jobs failing immediately
- **Check:** SLURM logs in `sweep_logs/`
- **Check:** Python environment is activated
- **Check:** GPU availability (`squeue` shows job status)

**Problem:** Sweep not progressing
- **Check:** At least one agent is running (`squeue` or `ps aux | grep wandb`)
- **Check:** WandB dashboard shows runs being created

