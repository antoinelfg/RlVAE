# Clean Training Guide for RlVAE 🧹

## Overview

Your RlVAE training environment now supports **WandB-only training** that eliminates local file clutter while maintaining full experiment tracking and visualization capabilities.

## Problem Solved

**Before:** Your training directory was cluttered with:
- 🗂️ **791MB** of training files
- 📁 **300+ files** (visualizations, model checkpoints, wandb runs)
- 🔄 Multiple wandb run directories taking up space
- 📊 Visualization files you'll never look at locally

**After:** Clean workspace with full experiment tracking on WandB cloud!

## New Command Line Options

### File Management Options

```bash
# 1. WandB-only mode (no local files at all)
--wandb_only

# 2. Disable local visualization files (keep model checkpoints)
--disable_local_files  

# 3. Offline WandB mode (no local wandb run folders)
--wandb_offline
```

### Performance Options

```bash
# 4. Disable expensive curvature computation during training
--disable_curvature_during_training
```

## Usage Examples

### 🎯 Recommended: Fully Clean Training

```bash
# Maximum cleanliness - no local files, offline WandB
python run_clean_training.py \
    --loop_mode open \
    --wandb_only \
    --wandb_offline \
    --n_epochs 25 \
    --n_train_samples 3000
```

### 🌐 Clean with Online WandB

```bash
# No local files but online WandB syncing
python run_clean_training.py \
    --loop_mode open \
    --wandb_only \
    --n_epochs 25
```

### 📊 Hybrid Mode

```bash
# Keep model checkpoints, disable visualizations
python run_clean_training.py \
    --loop_mode open \
    --disable_local_files \
    --n_epochs 25
```

### 🔧 Direct Training Script

```bash
# Use the training script directly with new flags
python src/training/train_cyclic_loop_comparison.py \
    --loop_mode open \
    --wandb_only \
    --wandb_offline \
    --disable_curvature_during_training \
    --n_epochs 25 \
    --n_train_samples 3000
```

## File Management Comparison

| Mode | Local Visualizations | Model Checkpoints | WandB Runs | Space Usage |
|------|---------------------|------------------|------------|-------------|
| **Standard** | ✅ Saved locally | ✅ Saved locally | 📁 Local folders | ~800MB+ |
| **Hybrid** | ❌ Disabled | ✅ Saved locally | 📁 Local folders | ~600MB |
| **WandB-only** | ❌ Disabled | ❌ Disabled | ☁️ Cloud only | <1MB |
| **WandB-only + Offline** | ❌ Disabled | ❌ Disabled | 💾 Buffer (~60KB) | <1MB |

## Benefits

### 🧹 Cleanliness
- **No visualization files** cluttering your workspace
- **No model checkpoints** taking up 600+ MB
- **No wandb run directories** with hundreds of files

### ⚡ Performance
- **Faster training** (no local I/O overhead)
- **Less memory usage** (no file writing)
- **Optimized curvature computation** (disabled by default)

### ☁️ Full Experiment Tracking
- **All visualizations** available on WandB dashboard
- **Complete training metrics** logged and tracked
- **Interactive plots** and animations preserved
- **Model weights** can be downloaded from WandB if needed

### 🖥️ Better for Clusters
- **Reduced storage usage** on shared systems
- **No local file conflicts** with other users
- **Offline mode** works without internet during training

## Cleanup Existing Files

To clean up your current cluttered workspace:

```bash
# Interactive cleanup (asks for confirmation)
python cleanup_training_files.py

# Will remove:
# - All .png, .html visualization files
# - All .pt model checkpoints  
# - All wandb run directories
# - Temporary __pycache__ folders
#
# Will keep:
# - Source code
# - Data files
# - Configuration files
```

## WandB Access

### Online Mode
- Training metrics appear on your WandB dashboard in real-time
- Visualizations uploaded automatically
- Access at: https://wandb.ai/your-username/your-project

### Offline Mode
- Logs stored locally in small buffer (~60KB)
- Sync later with: `wandb sync wandb/offline-run-*/`
- Or stay offline and check logs locally

## Test the New System

### Quick Test (1 epoch, 10 samples)
```bash
python run_clean_training.py \
    --loop_mode open \
    --n_epochs 1 \
    --n_train_samples 10 \
    --wandb_only \
    --wandb_offline
```

### Full Training (25 epochs, 3000 samples)
```bash
python run_clean_training.py \
    --loop_mode open \
    --n_epochs 25 \
    --n_train_samples 3000 \
    --wandb_only
```

## What Gets Logged to WandB

Even in clean mode, you still get **full experiment tracking**:

- ✅ **Training metrics** (loss, KL, reconstruction, etc.)
- ✅ **Validation metrics** and learning curves  
- ✅ **Cyclicity analysis** visualizations
- ✅ **Sequence trajectory** plots
- ✅ **Comprehensive reconstruction** comparisons
- ✅ **Manifold visualizations** (PCA, heatmaps)
- ✅ **Enhanced geodesic** analysis
- ✅ **Interactive HTML** animations
- ✅ **Curvature analysis** (when enabled)
- ✅ **System metrics** (memory, GPU usage)

## Implementation Details

The training script now includes:

1. **File saving control** via `should_save_locally()` and `should_log_to_wandb()`
2. **Smart image handling** for both local and WandB-only modes
3. **Offline WandB support** to avoid local run directories
4. **Buffer-based visualization** for WandB-only mode
5. **Enhanced error handling** for visualization failures

Your workspace stays clean while maintaining full research capabilities! 🎉 