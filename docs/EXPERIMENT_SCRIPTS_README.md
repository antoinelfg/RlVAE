# RlVAE Experiment Scripts
## SLURM Batch Scripts for Automated Experimentation

This directory contains SLURM batch scripts for running comprehensive experiments on the cluster.

---

## 📋 Available Scripts

### 1. **Quick Test Script** (`run_quick_test.sh`)
- **Duration**: 2 hours
- **Purpose**: Validate that everything works before running full experiments
- **Resources**: 1 GPU, 32GB RAM, 8 CPUs

```bash
# Submit quick test
sbatch run_quick_test.sh

# Check status
squeue -u $USER | grep rlvae_test
```

**What it does:**
- ✅ Runs validation tests (`test_hybrid_model.py`, `test_modular_components.py`)
- ✅ Runs quick training experiment (20 epochs, small dataset)
- ✅ Validates the complete pipeline works

### 2. **Weekend Experiment Suite** (`run_weekend_experiments.sh`)
- **Duration**: 48 hours (weekend)
- **Purpose**: Comprehensive validation and comparison of all model variants
- **Resources**: 1 GPU, 64GB RAM, 16 CPUs

```bash
# Submit weekend experiments
sbatch run_weekend_experiments.sh

# Check status
squeue -u $USER | grep rlvae_weekend
```

**What it does:**
- ✅ Validation tests
- ✅ Quick training with hybrid model
- ✅ Full training with hybrid model (50 epochs, full dataset)
- ✅ Model comparison study (Hybrid vs Standard vs Vanilla)
- ✅ Hyperparameter sweep
- ✅ Performance benchmarks
- ✅ Automatic summary report generation

### 3. **Experiment Monitor** (`monitor_experiments.sh`)
- **Purpose**: Monitor running experiments and check results
- **No SLURM submission needed**

```bash
# Run monitor
./monitor_experiments.sh
```

**What it shows:**
- 📊 Running SLURM jobs
- 📋 Recent log files
- 📊 Output files with results
- 📈 Weights & Biases status
- 💾 Disk usage
- 🎮 GPU usage

---

## 🚀 How to Use

### Step 1: Quick Test (Recommended First)
```bash
# Submit quick test
sbatch run_quick_test.sh

# Wait for completion, then check results
./monitor_experiments.sh
```

### Step 2: Weekend Experiments (If Quick Test Passes)
```bash
# Submit weekend suite
sbatch run_weekend_experiments.sh

# Monitor progress
./monitor_experiments.sh
```

### Step 3: Check Results
```bash
# View summary report
cat logs/summary_report_*.txt

# View specific experiment logs
cat logs/hybrid_quick_*.log
cat logs/comparison_study_*.log
```

---

## 📊 Expected Results

### Quick Test (2 hours)
- ✅ Validation tests pass
- ✅ Quick training completes (20 epochs)
- ✅ Test loss < 500
- ✅ All components working

### Weekend Suite (48 hours)
- ✅ All validation tests pass
- ✅ Hybrid model quick training: ~10 minutes
- ✅ Hybrid model full training: ~2 hours
- ✅ Model comparison: ~4 hours
- ✅ Hyperparameter sweep: ~8 hours
- ✅ Performance benchmarks: ~30 minutes

### Performance Expectations
| Model | Training Time | Test Loss | Metric Speed |
|-------|---------------|-----------|--------------|
| **Hybrid RlVAE** | **2x faster** | **Same** | **2x faster** |
| Standard RlVAE | Baseline | Same | Baseline |
| Vanilla VAE | Fastest | Different | N/A |

---

## 🔧 Customization

### Modify Experiment Parameters
Edit the scripts to change:
- Training configurations
- Model parameters
- Dataset sizes
- Visualization levels

### Add New Experiments
Add new experiment functions to `run_weekend_experiments.sh`:
```bash
run_experiment "custom_experiment" \
    "python run_experiment.py model=hybrid_rlvae custom_param=value"
```

### Change Resource Requirements
Modify SLURM parameters in scripts:
```bash
#SBATCH --time=24:00:00    # Change time limit
#SBATCH --mem=128G         # Change memory
#SBATCH --gres=gpu:2       # Change GPU count
```

---

## 📈 Monitoring and Debugging

### Check Job Status
```bash
# All your jobs
squeue -u $USER

# Specific job
squeue -j <job_id>

# Job history
sacct -u $USER --starttime=2024-01-01
```

### View Logs
```bash
# SLURM output
cat logs/weekend_experiments_*.out
cat logs/weekend_experiments_*.err

# Experiment logs
cat logs/hybrid_quick_*.log
cat logs/comparison_study_*.log
```

### Debug Issues
```bash
# Check if environment is correct
conda activate rlvae
python -c "import torch; print(torch.cuda.is_available())"

# Test individual components
python test_hybrid_model.py
python test_modular_components.py
```

---

## 🎯 Success Criteria

### Quick Test Success
- [ ] All validation tests pass
- [ ] Quick training completes without errors
- [ ] Test loss is reasonable (< 500)
- [ ] No CUDA out of memory errors

### Weekend Suite Success
- [ ] All experiments complete successfully
- [ ] Hybrid model shows 2x performance improvement
- [ ] Model comparison shows expected results
- [ ] Summary report generated
- [ ] All results saved to `outputs/`

---

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in config
   # Or request more GPU memory
   #SBATCH --mem=128G
   ```

2. **Job Timeout**
   ```bash
   # Increase time limit
   #SBATCH --time=72:00:00
   ```

3. **Environment Issues**
   ```bash
   # Check conda environment
   conda list | grep torch
   conda list | grep hydra
   ```

4. **Missing Dependencies**
   ```bash
   # Install missing packages
   pip install -e .
   ```

### Emergency Stop
```bash
# Cancel all your jobs
scancel -u $USER

# Cancel specific job
scancel <job_id>
```

---

## 📚 Related Documentation

- [`TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md) - Complete training guide
- [`MODULAR_TRAINING_GUIDE.md`](docs/MODULAR_TRAINING_GUIDE.md) - Modular system guide
- [`MODULARIZATION_SUMMARY.md`](MODULARIZATION_SUMMARY.md) - Architecture summary

---

*💡 **Tip**: Always run the quick test first to validate your setup before submitting the full weekend suite!* 