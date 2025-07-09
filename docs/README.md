# RlVAE Documentation

Welcome to the RlVAE (Riemannian Flow VAE) documentation. This directory contains comprehensive guides for using, extending, and contributing to the framework.

## 📖 Core Documentation

### Getting Started
- **[Installation Guide](installation.md)** - Complete setup instructions and dependencies
- **[Training Guide](TRAINING_GUIDE.md)** - Comprehensive training workflows and configuration
- **[Global Pipeline Guide](GLOBAL_PIPELINE_GUIDE.md)** - Two-stage training pipeline overview

### Advanced Usage
- **[Hyperparameter Optimization Guide](HYPERPARAMETER_OPTIMIZATION_GUIDE.md)** - WandB sweeps and parameter optimization
- **[Sweep Guide](SWEEP_README.md)** - Large-scale hyperparameter sweeps on SLURM clusters
- **[Visualization Guide](MODULAR_VISUALIZATION_GUIDE.md)** - Comprehensive visualization system

### Development
- **[Contributing Guide](CONTRIBUTING.md)** - Development setup, coding standards, and guidelines
- **[RlVAE Pipeline](GLOBAL_RLVAE_PIPELINE.md)** - Core pipeline architecture and design

## 🗂️ Quick Reference

| Topic | Document | Description |
|-------|----------|-------------|
| **Setup** | [installation.md](installation.md) | Environment setup and dependencies |
| **Training** | [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | Complete training workflows |
| **Pipeline** | [GLOBAL_PIPELINE_GUIDE.md](GLOBAL_PIPELINE_GUIDE.md) | Two-stage training architecture |
| **Optimization** | [HYPERPARAMETER_OPTIMIZATION_GUIDE.md](HYPERPARAMETER_OPTIMIZATION_GUIDE.md) | WandB sweeps and optimization |
| **Sweeps** | [SWEEP_README.md](SWEEP_README.md) | SLURM cluster sweep execution |
| **Visualization** | [MODULAR_VISUALIZATION_GUIDE.md](MODULAR_VISUALIZATION_GUIDE.md) | Visualization system |
| **Development** | [CONTRIBUTING.md](CONTRIBUTING.md) | Contributing guidelines |

## 🏗️ Architecture Overview

The RlVAE framework follows a modular, two-stage architecture:

1. **Stage 1**: Train vanilla VAE for metric extraction
2. **Stage 2**: Train RlVAE using extracted metrics

### Key Components
- **Modular Design**: Plug-and-play components for encoders, decoders, flows
- **Hydra Configuration**: Systematic experiment management
- **WandB Integration**: Comprehensive experiment tracking
- **SLURM Support**: Scalable cluster computing

## 🚀 Quick Start Paths

### For New Users
1. Read [Installation Guide](installation.md)
2. Follow [Training Guide](TRAINING_GUIDE.md) 
3. Explore [Global Pipeline Guide](GLOBAL_PIPELINE_GUIDE.md)

### For Researchers
1. Check [Hyperparameter Optimization Guide](HYPERPARAMETER_OPTIMIZATION_GUIDE.md)
2. Review [Sweep Guide](SWEEP_README.md)
3. Explore [Visualization Guide](MODULAR_VISUALIZATION_GUIDE.md)

### For Developers
1. Read [Contributing Guide](CONTRIBUTING.md)
2. Understand [Global Pipeline](GLOBAL_RLVAE_PIPELINE.md)
3. Review architecture in [Training Guide](TRAINING_GUIDE.md)

## 📁 Additional Resources

- **Archive**: Historical documentation and development notes
- **Guides**: Specialized guides for specific use cases

---

**Need help?** Start with the [Installation Guide](installation.md) or open an issue for support. 