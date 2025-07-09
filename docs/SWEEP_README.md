# RLVAE Hyperparameter Optimization System

## 🎯 Quick Start

### Run Comprehensive Sweep
```bash
# Local execution (single agent)
python scripts/run_sweep.py --sweep-config comprehensive_hyperparameter_sweep

# SLURM execution (multiple agents)
sbatch run_hyperparameter_sweep.sbatch comprehensive_hyperparameter_sweep 2 50
```

### Run Focused Sweeps
```bash
# Architecture optimization 
python scripts/run_sweep.py --sweep-config architecture_optimization --agent-count 4

# Training optimization
python scripts/run_sweep.py --sweep-config learning_rate_optimization --max-runs 50
```

---

## 📁 File Structure

```
├── conf/
│   └── sweep/
│       ├── comprehensive_hyperparameter_sweep.yaml  # Full optimization
│       ├── architecture_optimization.yaml          # Architecture focus  
│       └── learning_rate_optimization.yaml         # Training focus
├── scripts/
│   └── run_sweep.py                                # Main sweep runner
├── run_hyperparameter_sweep.sbatch                # SLURM batch script
├── docs/
│   └── HYPERPARAMETER_OPTIMIZATION_GUIDE.md       # Detailed documentation
└── SWEEP_README.md                                 # This file
```

---

## 🔧 Updated Configurations

### 1. WandB Project Integration
- **Project Name:** `rlvae-hyperparameter-optimization` (unified across stages)
- **Stage 1 Runs:** `pipeline_stage1_vanilla_vae_...`
- **Stage 2 Runs:** `pipeline_stage2_rlvae_...`

### 2. Visualization Control
- **Training:** Minimal visualizations for speed
- **Final Analysis:** Full visualizations with optimal parameters

---

## 📊 Parameter Coverage

### Comprehensive Sweep (20+ Parameters)
- **Architectures:** MLP, CNN, ResNet
- **Latent Dimensions:** 8, 16, 32, 64
- **Learning Rates:** 1e-5 to 1e-2 (log-uniform)
- **Batch Sizes:** 8, 16, 32, 64
- **Regularization:** Beta, Riemannian beta, weight decay
- **Data Sizes:** 500-5000 training samples
- **Model Structure:** Flow counts, hidden sizes, blocks

### Key Metrics Optimized
- **Primary:** `val_loss` (validation loss)
- **Secondary:** `val_recon_loss`, `val_kl_loss`, `cyclicity_error`

---

## 🚀 Usage Examples

### Basic Commands
```bash
# Dry run (validate config)
python scripts/run_sweep.py --sweep-config comprehensive_hyperparameter_sweep --dry-run

# Single agent
python scripts/run_sweep.py --sweep-config architecture_optimization

# Multiple agents
python scripts/run_sweep.py --sweep-config comprehensive_hyperparameter_sweep --agent-count 4 --max-runs 100

# Resume sweep  
python scripts/run_sweep.py --sweep-id abcd1234 --agent-count 2

# Check status
python scripts/run_sweep.py --sweep-id abcd1234 --status-only
```

### SLURM Commands
```bash
# Submit default sweep
sbatch run_hyperparameter_sweep.sbatch

# Custom configuration
sbatch run_hyperparameter_sweep.sbatch architecture_optimization 4 50 my-wandb-entity

# Multiple sweep submissions
sbatch run_hyperparameter_sweep.sbatch comprehensive_hyperparameter_sweep 2 100
sbatch run_hyperparameter_sweep.sbatch architecture_optimization 3 30
sbatch run_hyperparameter_sweep.sbatch learning_rate_optimization 2 50
```

---

## 📈 Expected Performance

### Resource Requirements
- **GPU Memory:** 8-16GB recommended
- **Training Time:** 1-3 hours per experiment
- **Total Sweep Time:** 24-72 hours for comprehensive optimization

### Typical Results
- **Architecture Sweep:** ~30 runs, identifies best model type
- **Training Sweep:** ~50 runs, optimizes learning dynamics  
- **Comprehensive Sweep:** ~100 runs, finds global optimum

---

## 🛠 Quick Troubleshooting

### Common Issues
```bash
# WandB authentication
wandb login

# Check GPU availability
nvidia-smi

# Validate sweep config
python scripts/run_sweep.py --sweep-config your_config --dry-run

# Monitor running sweep
tail -f logs/sweep_*.out
```

### Debug Single Experiment
```bash
python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
    experiment.stage1.latent_dim=32 \
    model.riemannian_beta=5.0 \
    training.data.batch_size=16
```

---

## 📚 Documentation

- **Detailed Guide:** `docs/HYPERPARAMETER_OPTIMIZATION_GUIDE.md`
- **Sweep Configs:** `conf/sweep/*.yaml`
- **WandB Dashboard:** https://wandb.ai/your-entity/rlvae-hyperparameter-optimization

---

**Ready to optimize!** Start with `architecture_optimization` to find the best model, then run `comprehensive_hyperparameter_sweep` for full optimization. 