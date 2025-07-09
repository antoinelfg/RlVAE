# RLVAE Hyperparameter Optimization Guide

## Overview

This guide covers comprehensive hyperparameter optimization for the RLVAE pipeline using WandB sweeps and Hydra configuration management. The system optimizes all major parameters across both stages of the pipeline to minimize validation error.

## 🎯 Quick Start

### 1. Basic Sweep
```bash
# Run comprehensive hyperparameter sweep
python scripts/run_sweep.py --sweep-config comprehensive_hyperparameter_sweep

# Run architecture optimization 
python scripts/run_sweep.py --sweep-config architecture_optimization --agent-count 4

# Run training optimization
python scripts/run_sweep.py --sweep-config learning_rate_optimization --max-runs 50
```

### 2. SLURM Batch Execution
```bash
# Submit comprehensive sweep to SLURM
sbatch run_hyperparameter_sweep.sbatch comprehensive_hyperparameter_sweep 2 20

# Submit architecture optimization
sbatch run_hyperparameter_sweep.sbatch architecture_optimization 4 30

# Submit training optimization  
sbatch run_hyperparameter_sweep.sbatch learning_rate_optimization 2 50
```

---

## 📊 Available Sweep Configurations

### 1. Comprehensive Hyperparameter Sweep
**File:** `conf/sweep/comprehensive_hyperparameter_sweep.yaml`
**Purpose:** Optimize all major parameters across both pipeline stages
**Method:** Random search with early termination
**Parameters:** 20+ hyperparameters including architectures, learning rates, regularization

```bash
python scripts/run_sweep.py --sweep-config comprehensive_hyperparameter_sweep --max-runs 100
```

### 2. Architecture Optimization
**File:** `conf/sweep/architecture_optimization.yaml`
**Purpose:** Find optimal architecture combinations (MLP, CNN, ResNet)
**Method:** Grid search for systematic comparison
**Parameters:** Architecture types, latent dimensions, key hyperparameters

```bash
python scripts/run_sweep.py --sweep-config architecture_optimization --agent-count 3
```

### 3. Learning Rate Optimization
**File:** `conf/sweep/learning_rate_optimization.yaml`
**Purpose:** Optimize training dynamics and learning parameters
**Method:** Random search with hyperband early termination
**Parameters:** Learning rates, batch sizes, regularization, training length

```bash
python scripts/run_sweep.py --sweep-config learning_rate_optimization --max-runs 50
```

---

## 🔧 Parameter Categories

### Stage 1 (Vanilla VAE) Parameters
- **Architecture:** `["mlp", "cnn", "resnet"]`
- **Latent Dimension:** `[8, 16, 32, 64]`
- **Training Epochs:** `25-100`
- **Temperature:** `0.1-2.0` (log uniform)
- **Regularization:** `0.001-0.1` (log uniform)
- **Preset:** `["balanced", "conservative", "max_diversity"]`

### Stage 2 (RLVAE) Parameters
- **Number of Flows:** `[4, 6, 8, 12, 16]`
- **Beta (VAE):** `0.5-5.0` (log uniform)
- **Riemannian Beta:** `1.0-20.0` (log uniform)
- **Flow Hidden Size:** `[128, 256, 512]`
- **Flow Blocks:** `[2, 3, 4]`
- **Sampling Method:** `["geodesic", "enhanced", "standard"]`
- **Loop Mode:** `["open", "closed"]`
- **Loop Penalty:** `0.1-10.0` (log uniform)

### Training Parameters
- **Learning Rate:** `1e-5 - 1e-2` (log uniform)
- **Weight Decay:** `1e-6 - 1e-3` (log uniform)
- **Batch Size:** `[8, 16, 32, 64]`
- **Max Epochs:** `30-100`
- **Train Samples:** `[500, 1000, 2000, 3000, 5000]`
- **Val Samples:** `[200, 400, 600, 888]`

### Regularization Parameters
- **Posterior Type:** `["gaussian", "riemannian_metric"]`
- **Metric Temperature Override:** `0.5-5.0` (log uniform)

---

## 🚀 Running Sweeps

### Local Execution

#### Basic Usage
```bash
# Single agent sweep
python scripts/run_sweep.py --sweep-config comprehensive_hyperparameter_sweep

# Multiple agents in parallel
python scripts/run_sweep.py --sweep-config architecture_optimization --agent-count 4

# Limit number of runs
python scripts/run_sweep.py --sweep-config learning_rate_optimization --max-runs 30
```

#### Advanced Options
```bash
# Custom project and entity
python scripts/run_sweep.py \
    --sweep-config comprehensive_hyperparameter_sweep \
    --project my-rlvae-optimization \
    --entity my-wandb-team \
    --agent-count 2 \
    --max-runs 50

# Dry run to validate configuration
python scripts/run_sweep.py --sweep-config comprehensive_hyperparameter_sweep --dry-run

# Resume existing sweep
python scripts/run_sweep.py --sweep-id abcd1234 --agent-count 2

# Check sweep status
python scripts/run_sweep.py --sweep-id abcd1234 --status-only
```

### SLURM Execution

#### Basic SLURM Usage
```bash
# Submit to SLURM with defaults
sbatch run_hyperparameter_sweep.sbatch

# Custom configuration
sbatch run_hyperparameter_sweep.sbatch architecture_optimization 4 50 my-entity

# Multiple sweep types
sbatch run_hyperparameter_sweep.sbatch comprehensive_hyperparameter_sweep 2 100
sbatch run_hyperparameter_sweep.sbatch architecture_optimization 3 30
sbatch run_hyperparameter_sweep.sbatch learning_rate_optimization 2 50
```

#### SLURM Parameters
- **Time Limit:** 72 hours
- **Resources:** 1 GPU, 8 CPUs, 64GB RAM
- **Output:** `logs/sweep_<JOB_ID>.out`
- **Error:** `logs/sweep_<JOB_ID>.err`

---

## 📈 Monitoring and Analysis

### WandB Dashboard
All sweeps log to the unified project: `rlvae-hyperparameter-optimization`

**Key Metrics Tracked:**
- **Primary:** `val_loss`, `val_recon_loss`, `val_kl_loss`, `train_loss`
- **Secondary:** `cyclicity_error`, `riemannian_kl`, `metric_conditioning`
- **Performance:** Training time, GPU memory usage

### Accessing Results
```bash
# View sweep status
python scripts/run_sweep.py --sweep-id <SWEEP_ID> --status-only

# WandB dashboard URL
https://wandb.ai/your-entity/rlvae-hyperparameter-optimization
```

---

## 🔧 Creating Custom Sweeps

### 1. Basic Structure
Create a new file `conf/sweep/my_custom_sweep.yaml`:

```yaml
# @package sweep

name: "my_custom_sweep"
description: "Custom optimization for specific parameters"

method: random  # random, grid, bayes

objective:
  metric: "val_loss"
  goal: minimize

# Parameters to sweep
parameters:
  experiment.stage1.latent_dim:
    values: [16, 32, 64]
  
  model.riemannian_beta:
    distribution: log_uniform_values
    min: 1.0
    max: 10.0

# Fixed parameters
parameters_fixed:
  seed: 42
  training.trainer.max_epochs: 50
```

### 2. Advanced Configuration
```yaml
# Early termination
early_terminate:
  type: hyperband
  max_iter: 30
  eta: 2

# Resource constraints
run_cap: 50
concurrent_runs: 3

# Bayesian optimization
method: bayes
metric:
  name: val_loss
  goal: minimize
```

### 3. Parameter Types
```yaml
parameters:
  # Discrete values
  param1:
    values: [1, 2, 3, 4, 5]
  
  # Uniform distribution
  param2:
    distribution: uniform
    min: 0.1
    max: 1.0
  
  # Log uniform distribution  
  param3:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-2
  
  # Integer uniform
  param4:
    distribution: int_uniform
    min: 10
    max: 100
  
  # Categorical
  param5:
    values: ["option1", "option2", "option3"]
```

---

## ⚡ Performance Tips

### 1. Efficient Sweeping
- **Start small:** Use architecture optimization first to find best base configuration
- **Use early termination:** Hyperband helps eliminate poor runs quickly
- **Parallel agents:** Run multiple agents for faster exploration
- **Focused sweeps:** Target specific parameter groups rather than everything at once

### 2. Resource Management
```bash
# Light sweep for quick iteration
python scripts/run_sweep.py \
    --sweep-config architecture_optimization \
    --max-runs 20 \
    --agent-count 2

# Heavy sweep for thorough optimization
sbatch run_hyperparameter_sweep.sbatch comprehensive_hyperparameter_sweep 4 100
```

### 3. Staged Optimization
1. **Architecture Selection:** Run `architecture_optimization` first
2. **Training Dynamics:** Use best architecture in `learning_rate_optimization`
3. **Full Optimization:** Run `comprehensive_hyperparameter_sweep` with constraints

---

## 🛠 Troubleshooting

### Common Issues

#### 1. WandB Authentication
```bash
# Login to WandB
wandb login

# Or set API key
export WANDB_API_KEY=your_api_key_here
```

#### 2. Hydra Configuration Errors
```bash
# Validate sweep config
python scripts/run_sweep.py --sweep-config your_config --dry-run

# Check available configs
ls conf/sweep/
```

#### 3. GPU Memory Issues
```bash
# Reduce batch size in sweep config
parameters:
  training.data.batch_size:
    values: [8, 16]  # Smaller batch sizes

# Monitor GPU usage
nvidia-smi -l 1
```

#### 4. Failed Experiments
```bash
# Check experiment logs
tail -f logs/sweep_<JOB_ID>.out

# Check WandB run details
# Look for error logs in WandB dashboard
```

### Debug Commands
```bash
# Test single experiment
python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
    experiment.stage1.latent_dim=16 \
    model.riemannian_beta=5.0

# Validate sweep configuration
python scripts/run_sweep.py --sweep-config comprehensive_hyperparameter_sweep --dry-run

# Check system resources
nvidia-smi
free -h
df -h
```

---

## 📊 Expected Results

### Performance Metrics
- **Comprehensive Sweep:** ~100 runs, 24-48 hours, identifies global optimum
- **Architecture Sweep:** ~30 runs, 8-12 hours, finds best architecture
- **Training Sweep:** ~50 runs, 12-24 hours, optimizes training dynamics

### Typical Optimal Ranges
Based on our experiments, optimal parameters typically fall in these ranges:

- **Latent Dimension:** 16-32 for most tasks
- **Riemannian Beta:** 5.0-15.0 for good manifold structure
- **Learning Rate:** 1e-4 to 5e-4 for stable training
- **Architecture:** CNN often performs best for image data
- **Batch Size:** 16-32 balances performance and memory

### Analysis Outputs
- **Best Configuration:** Exported as Hydra config
- **Parameter Importance:** WandB parallel coordinates plots
- **Training Curves:** Loss progression for all runs
- **Resource Usage:** Training time and memory analysis

---

## 📚 Advanced Usage

### 1. Bayesian Optimization
```yaml
method: bayes
metric:
  name: val_loss
  goal: minimize

early_terminate:
  type: hyperband
  max_iter: 10
```

### 2. Multi-Objective Optimization
```yaml
metric:
  name: composite_metric
  goal: minimize

# Define composite metric in training code
# composite_metric = val_loss + 0.1 * training_time + 0.05 * gpu_memory
```

### 3. Conditional Parameters
```yaml
parameters:
  model.posterior.type:
    values: ["gaussian", "riemannian_metric"]
  
  model.riemannian_beta:
    distribution: log_uniform_values
    min: 1.0
    max: 20.0
    # Only used when posterior.type == "riemannian_metric"
```

### 4. Budget-Based Optimization
```yaml
early_terminate:
  type: hyperband
  max_iter: 50
  eta: 3
  s: 2

run_cap: 200  # Maximum total runs
```

---

## 🎯 Best Practices

### 1. Sweep Strategy
1. **Start with architecture optimization** to find the best base model
2. **Use the best architecture** for training parameter optimization  
3. **Run comprehensive sweep** with promising parameter ranges
4. **Analyze results** and create focused follow-up sweeps

### 2. Parameter Selection
- **Include key parameters** that significantly impact performance
- **Use appropriate distributions** (log-uniform for learning rates, uniform for others)
- **Set reasonable bounds** based on domain knowledge
- **Group related parameters** in focused sweeps

### 3. Resource Management
- **Use early termination** to save compute on poor runs
- **Run parallel agents** to explore parameter space faster
- **Monitor resource usage** to avoid timeouts
- **Save intermediate results** in case of interruptions

### 4. Analysis and Iteration
- **Use WandB parallel coordinates** to understand parameter relationships
- **Look for parameter importance** in the results
- **Create follow-up sweeps** around promising regions
- **Document findings** for future reference

---

## 📞 Support and Community

### Getting Help
- **Documentation Issues:** Check this guide and Hydra/WandB docs
- **Technical Problems:** Create GitHub issue with logs and configuration
- **Best Practices:** Discuss in project discussions or forums

### Contributing
- **New Sweep Configs:** Submit PR with new optimization strategies
- **Performance Improvements:** Optimize sweep runner or parameter selection
- **Documentation:** Improve guides and examples

### Resources
- **WandB Sweeps Documentation:** https://docs.wandb.ai/guides/sweeps
- **Hydra Configuration:** https://hydra.cc/docs/intro/
- **RLVAE Paper:** [Link to paper when available]

---

This guide provides comprehensive coverage of hyperparameter optimization for the RLVAE pipeline. Start with the quick start section and gradually explore more advanced features as needed. 