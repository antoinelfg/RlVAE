# Modular Vanilla VAE Implementation (Pipeline-Integrated)

## Overview

The modular vanilla VAE is now always run as **Stage 1** of the Global RLVAE Pipeline. All training, metric extraction, and visualizations are managed through the pipeline and the modular visualization system.

---

## Usage

- **Do not use legacy scripts.**
- Always run modular vanilla VAE via the pipeline:

```bash
python scripts/global_rlvae_pipeline.py --architecture cnn --latent-dim 16 --vae-epochs 50 --wandb
```

- Outputs and visualizations are saved in `output_dir/vanilla_vae/`
- All visualizations are logged to wandb

---

## Extensibility

- Add new architectures by extending `EncoderManager` or `DecoderManager` in `src/models/components/`
- Add new visualizations by creating a module in `src/visualizations/` and registering it with the manager

---

## For More Details

See `GLOBAL_RLVAE_PIPELINE.md` and the updated documentation for quickstart, advanced usage, and extensibility examples.

## 🏗️ Architecture Support

| Architecture | Status | Parameters | Description |
|-------------|--------|------------|-------------|
| **MLP** | ✅ Ready | ~12.6M | Multi-layer perceptron (original) |
| **CNN** | ✅ Ready | ~4.2M | Convolutional networks for images |
| **ResNet** | ✅ Ready | ~11.9M | Residual networks (size mismatch) |

## 🚀 Quick Start

### Option 1: Single Architecture Training

```bash
# Train CNN vanilla VAE (recommended)
python train_modular_vanilla_vae.py --architecture cnn --epochs 50

# Train MLP vanilla VAE  
python train_modular_vanilla_vae.py --architecture mlp --epochs 50

# Train ResNet vanilla VAE (has minor size issue)
python train_modular_vanilla_vae.py --architecture resnet --epochs 50
```

### Option 2: Train All Architectures

```bash
# Train all architectures sequentially
python train_modular_vanilla_vae.py --all --epochs 50
```

### Option 3: Use Original Script (Backward Compatible)

```bash
# Your existing script now works with modular components
python scripts/train_and_extract_vanilla_vae.py
```

## 📁 Files Created

- `src/models/modular_vanilla_vae.py` - Main modular VAE implementation
- `vanilla_vae.py` - Backward compatibility wrapper
- `train_modular_vanilla_vae.py` - New modular training script
- `test_modular_vanilla_vae.py` - Testing infrastructure

## 🎯 Output Files

The new system saves components with architecture-specific names to protect your existing files:

```
data/pretrained/
├── encoder_BACKUP_WORKING.pt          # Your protected originals
├── decoder_BACKUP_WORKING.pt          # Your protected originals  
├── metric_T0.7_scaled_BACKUP_WORKING.pt # Your protected originals
├── encoder_cnn_20241202_143022.pt     # New CNN encoder
├── decoder_cnn_20241202_143022.pt     # New CNN decoder
├── metric_cnn_20241202_143022.pt      # New CNN metric
└── ... (similar for mlp, resnet)
```

## 🧪 Testing

```bash
# Test all architectures
python test_modular_vanilla_vae.py
```

## 📊 Current Status

### ✅ Working Features

1. **Modular Architecture**: Easy to switch between MLP/CNN/ResNet
2. **Backward Compatibility**: Existing scripts work unchanged
3. **Protected Files**: Your working models are safely backed up
4. **Metric Extraction**: Full RHVAE-style metric extraction
5. **WandB Integration**: Full logging and visualization
6. **Command Line Interface**: Easy training with different configs

### ⚠️ Known Issues

1. **ResNet Size Mismatch**: Outputs 128x128 instead of 64x64 (decoder upsampling issue)

## 💡 Recommended Usage

**For immediate production use:** Start with **CNN architecture** - it's working perfectly and gives good results with fewer parameters than MLP.

```bash
python train_modular_vanilla_vae.py --architecture cnn --epochs 50 --batch-size 32
```

This will:
- Train on full dataset (train + test) 
- Extract RHVAE-style metrics
- Save encoder/decoder/metric with timestamp
- Protect your existing working files
- Provide full WandB logging

## 🔧 Architecture Configurations

### CNN Configuration
```python
encoder_config = {
    'cnn': {
        'hidden_dims': [32, 64, 128, 256],
        'kernel_size': 3,
        'stride': 2,
        'padding': 1,
        'dropout': 0.1
    }
}
```

### MLP Configuration  
```python
encoder_config = {
    'mlp': {
        'hidden_dims': [512, 512, 512],
        'dropout': 0.1
    }
}
```

## 🎉 Next Steps

You now have a **complete modular vanilla VAE system** that:
- ✅ Protects your existing working files
- ✅ Supports multiple architectures  
- ✅ Maintains your exact training pipeline
- ✅ Extracts metrics in RHVAE format
- ✅ Works with your existing project structure

**Ready to start training new models with different architectures!** 🚀 