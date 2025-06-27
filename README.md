# Riemannian Flow VAE (RlVAE) 🧠

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-purple.svg)](https://lightning.ai/)
[![Hydra](https://img.shields.io/badge/Hydra-1.3+-green.svg)](https://hydra.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A **modular, high-performance** implementation of **Riemannian Flow VAE** for longitudinal data modeling. This repository provides a comprehensive experimental framework with **2x performance improvements**, extensive visualization capabilities, and systematic model comparison tools.

## 🚀 Quick Start

### Installation & Basic Usage
```bash
# Clone and install
git clone https://github.com/antoinelfg/RlVAE.git
cd RlVAE
pip install -e .

# Quick test (🔥 RECOMMENDED: Hybrid model with 2x speedup)
python run_experiment.py model=hybrid_rlvae training=quick visualization=minimal

# Full training
python run_experiment.py model=hybrid_rlvae training=full_data visualization=standard

# Compare all models
python run_experiment.py experiment=comparison_study
```

### Validation & Testing
```bash
# Test hybrid model integration
python test_hybrid_model.py

# Test modular components
python test_modular_components.py
```

---

## 🏗️ Architecture Overview

### 🔥 **NEW: Hybrid Model** (RECOMMENDED)
- **File**: `src/models/hybrid_rlvae.py`
- **Performance**: **2x faster** metric computations
- **Accuracy**: Perfect numerical compatibility with original
- **Features**: Enhanced diagnostics + modular components

### 🧩 **Modular Components**
- **MetricTensor** (`src/models/components/metric_tensor.py`): Optimized Riemannian metric computations
- **MetricLoader** (`src/models/components/metric_loader.py`): Flexible pretrained metric loading
- **Comprehensive Testing**: Validated numerical accuracy and performance

### 📊 **Experimental Framework**
- **Hydra Configuration**: Systematic experiment management
- **Multiple Models**: Hybrid RlVAE, Standard RlVAE, Vanilla VAE
- **Visualization Suite**: From minimal to comprehensive analysis
- **Performance Tracking**: Automatic benchmarking and comparison

---

## 📁 Repository Structure

```
RlVAE/
├── 🧠 src/                          # Core implementation
│   ├── models/
│   │   ├── hybrid_rlvae.py          # 🔥 NEW: 2x faster hybrid model
│   │   ├── modular_rlvae.py         # Hydra-compatible model
│   │   ├── riemannian_flow_vae.py   # Original implementation
│   │   └── components/              # 🧩 NEW: Modular components
│   │       ├── metric_tensor.py     #     Optimized metric computations
│   │       └── metric_loader.py     #     Flexible metric loading
│   ├── training/
│   │   ├── lightning_trainer.py     # PyTorch Lightning integration
│   │   └── train_with_modular_visualizations.py
│   ├── visualizations/              # 🎨 NEW: Comprehensive viz suite
│   │   ├── basic.py                 #     Standard plots
│   │   ├── manifold.py              #     Riemannian analysis
│   │   ├── interactive.py           #     Interactive visualizations
│   │   └── flow_analysis.py         #     Flow dynamics
│   └── data/
│       └── cyclic_dataset.py        # Optimized data loading
├── ⚙️ conf/                         # 🔥 NEW: Hydra configurations
│   ├── config.yaml                  # Main configuration
│   ├── model/                       # Model configurations
│   │   ├── hybrid_rlvae.yaml        #   🔥 Hybrid model (recommended)
│   │   ├── riemannian_flow_vae.yaml #   Standard RlVAE
│   │   └── vanilla_vae.yaml         #   Baseline VAE
│   ├── training/                    # Training configurations
│   │   ├── quick.yaml               #   Fast development (20 epochs)
│   │   ├── full_data.yaml           #   Production training (50 epochs)
│   │   └── default.yaml             #   Standard training (30 epochs)
│   ├── visualization/               # Visualization levels
│   │   ├── minimal.yaml             #   Essential plots only
│   │   ├── standard.yaml            #   Balanced analysis
│   │   └── full.yaml                #   Comprehensive diagnostics
│   └── experiment/                  # Experiment types
│       ├── single_run.yaml          #   Single model training
│       ├── comparison_study.yaml    #   Multi-model comparison
│       └── hyperparameter_sweep.yaml #  Parameter optimization
├── 🧪 tests/                        # Testing & validation
│   ├── test_modular_components.py   # Component validation
│   ├── test_hybrid_model.py         # Integration testing
│   └── test_setup.py                # Environment validation
├── 📚 docs/                         # Documentation
│   ├── TRAINING_GUIDE.md            # 🔥 NEW: Complete training guide
│   ├── MODULAR_TRAINING_GUIDE.md    # Modular system guide
│   ├── MODULAR_VISUALIZATION_GUIDE.md # Visualization system
│   ├── RIEMANNIAN_FLOW_VAE_ANALYSIS.md # Architecture analysis
│   └── MODULARIZATION_ROADMAP.md    # Development roadmap
├── 🚀 run_experiment.py             # 🔥 NEW: Main experiment runner
├── 📊 MODULARIZATION_SUMMARY.md     # Executive summary
└── 📄 README_EXPERIMENTAL_FRAMEWORK.md # Framework overview
```

---

## 🎯 Available Models

| Model | Performance | Use Case | Command |
|-------|-------------|----------|---------|
| **🔥 Hybrid RlVAE** | **2x faster** | **Recommended for all experiments** | `model=hybrid_rlvae` |
| Standard RlVAE | Baseline | Legacy compatibility, comparisons | `model=riemannian_flow_vae` |
| Vanilla VAE | Fastest | Baseline comparisons | `model=vanilla_vae` |

---

## ⚙️ Training Configurations

| Configuration | Epochs | Dataset Size | Time (H100) | Use Case |
|---------------|--------|--------------|-------------|----------|
| **Quick** | 20 | 100 sequences | ~10 min | Development, debugging |
| **Default** | 30 | 1000 sequences | ~45 min | Standard experiments |
| **Full Data** | 50 | 3000 sequences | ~2 hours | Production training |

---

## 🎨 Visualization Levels

| Level | Features | Performance | Use Case |
|-------|----------|-------------|----------|
| **Minimal** | Basic plots only | Fastest | Development |
| **Standard** | Manifold analysis | Balanced | Most experiments |
| **Full** | Complete diagnostics | Comprehensive | Paper figures |

---

## 🧪 Experiment Types

### Single Run
```bash
python run_experiment.py model=hybrid_rlvae training=quick visualization=minimal
```

### Model Comparison
```bash
python run_experiment.py experiment=comparison_study
```

### Hyperparameter Sweep
```bash
python run_experiment.py experiment=hyperparameter_sweep -m
```

### Custom Configuration
```bash
python run_experiment.py model=hybrid_rlvae model.latent_dim=32 training.optimizer.lr=0.0005
```

---

## 📈 Performance Benchmarks

### Modular Components Validation
- **Numerical Accuracy**: Perfect (G difference: 9.459e-19)
- **Performance**: **2x speedup** over original implementation
- **Memory**: Same usage, better efficiency
- **Compatibility**: 100% backward compatible

### Training Speed Comparison
| Model | Metric Computation | Overall Training | Memory Usage |
|-------|-------------------|------------------|--------------|
| **Hybrid RlVAE** | **2x faster** | **1.5x faster** | Same |
| Standard RlVAE | Baseline | Baseline | Baseline |
| Vanilla VAE | N/A | Fastest | Lower |

---

## 🛠️ Development Workflows

### For Research Papers
```bash
# 1. Development
python run_experiment.py model=hybrid_rlvae training=quick visualization=minimal

# 2. Validation  
python test_hybrid_model.py && python test_modular_components.py

# 3. Production
python run_experiment.py model=hybrid_rlvae training=full_data visualization=full

# 4. Comparison
python run_experiment.py experiment=comparison_study
```

### For Quick Experiments
```bash
# Test idea
python run_experiment.py model=hybrid_rlvae training=quick

# Scale up if promising
python run_experiment.py model=hybrid_rlvae training=full_data
```

---

## 🎯 Key Features

### 🔥 **New Hybrid Architecture**
- **2x faster** metric tensor computations
- **Perfect numerical accuracy** maintained
- **Enhanced diagnostics** and error handling
- **Modular components** for easy testing and extension

### 🧪 **Comprehensive Experimental Framework**
- **Hydra configuration** management
- **Multiple model variants** with easy switching
- **Systematic comparison** tools
- **Automatic experiment tracking** with wandb

### 🎨 **Advanced Visualization Suite**
- **Modular visualization** system
- **Interactive plots** for exploration
- **Manifold analysis** for Riemannian geometry
- **Flow dynamics** visualization

### 📊 **Robust Testing & Validation**
- **Component-level testing** for modular parts
- **Integration testing** for hybrid model
- **Performance benchmarking** and validation
- **Numerical accuracy** verification

---

## 🚀 Migration from Legacy Code

### Before (Legacy)
```bash
python run_training.py --epochs 20 --batch_size 4
```

### After (New Framework)
```bash
python run_experiment.py model=hybrid_rlvae training=quick
```

### Benefits
- ✅ **2x faster** training with hybrid model
- ✅ **Systematic configuration** management
- ✅ **Comprehensive visualization** out-of-the-box
- ✅ **Easy model comparison** and hyperparameter tuning
- ✅ **Enhanced monitoring** and experiment tracking

---

## 📚 Documentation

- **[TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** - Complete guide to all training options
- **[MODULAR_TRAINING_GUIDE.md](docs/MODULAR_TRAINING_GUIDE.md)** - Detailed modular system guide
- **[MODULAR_VISUALIZATION_GUIDE.md](docs/MODULAR_VISUALIZATION_GUIDE.md)** - Visualization system
- **[RIEMANNIAN_FLOW_VAE_ANALYSIS.md](docs/RIEMANNIAN_FLOW_VAE_ANALYSIS.md)** - Architecture analysis
- **[MODULARIZATION_ROADMAP.md](docs/MODULARIZATION_ROADMAP.md)** - Future development plans

---

## 🔬 Research Applications

This framework has been designed for:
- **Longitudinal data modeling** with temporal consistency
- **Riemannian geometry** in latent spaces
- **Cyclic data analysis** (sprites, medical imaging, time series)
- **Systematic model comparison** studies
- **Ablation studies** on geometric vs. standard VAEs

---

## 📝 Citation

```bibtex
@software{rlvae2024,
  title={RlVAE: Modular Riemannian Flow VAE Framework},
  author={Antoine Laforgue},
  year={2024},
  url={https://github.com/antoinelfg/RlVAE},
  note={High-performance modular implementation with 2x speedup}
}
```

---

## 🤝 Contributing

We welcome contributions! The modular architecture makes it easy to:
- Add new model variants
- Extend visualization capabilities  
- Contribute new sampling methods
- Improve performance optimizations

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PyTorch Lightning** for training infrastructure
- **Hydra** for configuration management
- **Weights & Biases** for experiment tracking
- **RHVAE** implementation from Pythae library

---

*💡 **Tip**: Start with `model=hybrid_rlvae` for all new experiments - same results, 2x performance improvement!* 