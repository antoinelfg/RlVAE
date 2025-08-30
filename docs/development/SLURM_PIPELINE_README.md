# Complete RLVAE Pipeline - SLURM Execution Guide

## 🎯 Overview

This guide provides scripts to run the **complete optimized RLVAE pipeline** on SLURM clusters with GPU support. The pipeline includes both Stage 1 (Vanilla VAE) and Stage 2 (RLVAE) training with all the latest optimizations.

## 🚀 Quick Start

### Option 1: Simple Submission (Recommended)
```bash
./run_complete_pipeline.sh
```

### Option 2: Direct SLURM Submission
```bash
sbatch scripts/slurm/run_complete_pipeline.sbatch
```

## 📋 Experiment Configuration

### **Stage 1: Vanilla VAE Training**
- **Duration**: 200 epochs
- **Purpose**: Train base VAE and extract diverse metric
- **Architecture**: MLP encoder/decoder
- **Latent Dimension**: 2D
- **Beta**: 1.0 (balanced reconstruction/KL loss)

### **Stage 2: RLVAE Training**  
- **Duration**: 200 epochs
- **Flows**: 9 flows (optimal for 10-timestep sequences)
- **Beta**: 1.0 (reconstruction weight)
- **Riemannian Beta**: 1.0 (geometry weight)
- **Pretrained**: Uses Stage 1 components

### **Key Optimizations Applied**
- ✅ **Positive loss computation** (fixed negative total loss issue)
- ✅ **Working parameter overrides** (no more forced defaults)
- ✅ **Proper loss scaling** (255x factor for meaningful values)
- ✅ **Stable flow dynamics** (9 flows without numerical instability)

## 🖥️ Resource Requirements

### **SLURM Configuration**
```bash
#SBATCH --time=24:00:00          # 24 hours (adjust based on cluster)
#SBATCH --nodes=1                # Single node
#SBATCH --cpus-per-task=8        # 8 CPU cores
#SBATCH --gres=gpu:1             # 1 GPU required
#SBATCH --mem=64G                # 64GB RAM
#SBATCH --partition=gpu          # GPU partition
```

### **Estimated Runtime**
- **Stage 1**: ~8-12 hours (200 epochs vanilla VAE)
- **Stage 2**: ~10-14 hours (200 epochs RLVAE + 9 flows)
- **Total**: ~18-26 hours (depends on cluster speed)

## 📊 Monitoring & Results

### **Real-time Monitoring**
```bash
# Check job status
squeue -u $USER

# Follow logs in real-time
tail -f logs/rlvae_complete_pipeline_*.out

# Check job details
scontrol show job <JOB_ID>
```

### **WandB Tracking**
- **Project**: `rlvae-hyperparameter-optimization`
- **Real-time metrics**: Loss curves, visualizations, FID scores
- **URL**: Check the job output for your specific WandB link

### **Output Location**
- **Results**: `outputs/` directory
- **Logs**: `logs/rlvae_complete_pipeline_*.out`
- **Errors**: `logs/rlvae_complete_pipeline_*.err`

## ⚙️ Customization

### **Modify Resource Requirements**
Edit `scripts/slurm/run_complete_pipeline.sbatch`:
```bash
#SBATCH --time=48:00:00          # Increase time limit
#SBATCH --mem=128G               # Increase memory
#SBATCH --cpus-per-task=16       # More CPU cores
```

### **Adjust Experiment Parameters**
Modify the `EXPERIMENT_CMD` in the script:
```bash
experiment.stage1.epochs=100 \     # Reduce Stage 1 epochs
experiment.stage2.epochs=300 \     # Increase Stage 2 epochs
experiment.stage2.n_flows=6 \      # Fewer flows for speed
```

### **Change Cluster Configuration**
Update module loading for your cluster:
```bash
module load cuda/12.0              # Your CUDA version
module load python/3.11            # Your Python version
conda activate your_env_name       # Your conda environment
```

## 🐛 Troubleshooting

### **Common Issues**

1. **Job fails immediately**
   ```bash
   # Check cluster status
   sinfo
   
   # Verify partition exists
   sinfo -p gpu
   
   # Check account/QoS permissions
   sacctmgr show user $USER
   ```

2. **CUDA not available**
   ```bash
   # Test CUDA in interactive session
   srun --gres=gpu:1 --pty bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Out of memory**
   ```bash
   # Reduce batch size in config
   training.data.batch_size=32
   experiment.stage1.batch_size=32
   experiment.stage2.batch_size=32
   ```

4. **Environment issues**
   ```bash
   # Check conda environment exists
   conda env list
   
   # Recreate environment if needed
   conda create -n rlvae_env python=3.10
   # ... install dependencies
   ```

## 📈 Expected Results

### **Successful Completion Indicators**
- ✅ **Stage 1**: Vanilla VAE converges (loss ~5-15)
- ✅ **Stage 2**: RLVAE trains stably (positive total loss ~5-10)
- ✅ **FID Score**: ~60-150 (depends on model quality)
- ✅ **No NaN values**: All losses remain finite
- ✅ **WandB logs**: Complete visualization suite

### **Performance Metrics**
- **Reconstruction Loss**: ~5-20 (255-scale)
- **KL Loss**: ~0.5-3.0 (healthy latent diversity)
- **Total Loss**: Positive values (fixed negative loss bug)
- **Training Speed**: ~1-3 it/s (depends on GPU)

## 🎉 Success!

When the experiment completes successfully, you'll have:
- 📦 **Fully trained RLVAE model** with temporal dynamics
- 📊 **Comprehensive evaluation metrics** (FID, generation quality)
- 🎨 **Rich visualizations** in WandB (latent space, flows, reconstructions)
- 💾 **Saved checkpoints** for further analysis

**Happy training!** 🚀 