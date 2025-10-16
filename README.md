# RlVAE: Riemannian Flow Variational Autoencoder Research Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-purple.svg)](https://lightning.ai/)
[![Hydra](https://img.shields.io/badge/Hydra-1.3+-green.svg)](https://hydra.cc/)
[![WandB](https://img.shields.io/badge/WandB-Integrated-orange.svg)](https://wandb.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-ready research framework** for **Riemannian Flow Variational Autoencoders** applied to longitudinal data modeling. This framework implements advanced geometric deep learning techniques with **100% modular architecture**, comprehensive hyperparameter optimization, and systematic model comparison capabilities.

## 🎯 Research Focus

**RlVAE** addresses fundamental challenges in modeling temporal/sequential data by:
- **Learning Riemannian Geometry**: Discovering data-dependent geometric structures in latent space
- **Temporal Dynamics**: Modeling evolution along learned geometric manifolds using normalizing flows
- **Metric Learning**: Extracting optimal Riemannian metrics from data for improved representations
- **Systematic Comparison**: Rigorous evaluation of geometric vs. standard VAE approaches

## 🏗️ Architecture: Three-Stage Training Pipeline

### Stage A: Warm VAE + Time Conditioning
```bash
# Train time-conditioned vanilla VAE
python run_experiment.py experiment=rlvae_three_stage_pipeline \
    metric=rhvae sampling.enabled=false
```
- **Purpose**: Learn time-aware latent representations
- **Output**: Time-conditioned encoder, decoder, and initial metric

### Stage B: Metric Learning at t=0
```bash
# Learn Riemannian metric via RHVAE-style or precision
python run_experiment.py experiment=rlvae_three_stage_pipeline \
    metric=rhvae sampling.enabled=true
```
- **Purpose**: Extract optimal Riemannian metric from data
- **Methods**: `rhvae` (RHVAE-style) or `precision` (posterior precision)
- **Optional**: RHMC sampling with learned metric

### Stage C: Full RlVAE Training
```bash
# Train complete RlVAE with metric updates
python run_experiment.py experiment=rlvae_three_stage_pipeline \
    metric=rhvae sampling=rhmc_default
```
- **Purpose**: Full Riemannian Flow VAE with metric adaptation
- **Features**: Metric alternation, geometric constraints, flow dynamics
- **Naming**: `three_stage_pipeline_stageC_phase2_metric_adapt_flows{n}`

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/antoinelfg/RlVAE.git
cd RlVAE
pip install -r requirements.txt
```

### Basic Usage Examples

#### Three-Stage Pipeline (Recommended)
```bash
# Full pipeline with RHMC posterior and modular architecture
python run_experiment.py experiment=rlvae_three_stage_long_rhmc_modular

# Custom latent dimension and epochs
python run_experiment.py experiment=rlvae_three_stage_long_rhmc_modular \
    settings.model.latent_dim=4 settings.training.strategy.max_epochs=100

# Different visualization level
python run_experiment.py experiment=rlvae_three_stage_long_rhmc_modular \
    settings.visualization.level=minimal
```

#### Model Configuration
```bash
# Use riemannian RHMC VAE model
python run_experiment.py model=riemannian_rhmc_vae

# Use standard riemannian flow VAE
python run_experiment.py model=riemannian_flow_vae
```

#### Training Configuration
```bash
# Quick development run
python run_experiment.py settings.training.strategy.max_epochs=20

# Full data training
python run_experiment.py training=full_data

# Custom learning rate and batch size
python run_experiment.py settings.training.optimizer.lr=1e-3 \
    settings.data.common.batch_size=32
```

#### Visualization Configuration
```bash
# Minimal visualization (for sweeps)
python run_experiment.py visualization=minimal

# Standard visualization
python run_experiment.py visualization=standard

# Full visualization (publication quality)
python run_experiment.py visualization=full

# Three-stage optimized (recommended)
python run_experiment.py visualization=three_stage_optimized
```

#### Hyperparameter Optimization
```bash
# Learning rate optimization (50 runs)
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch learning_rate_optimization 4 50

# Architecture optimization (grid search)
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch architecture_optimization 2 20
```

## 🧩 Core Models

### 1. Three-Stage RlVAE Pipeline
**Purpose**: Complete end-to-end training with metric adaptation
- **Stage A**: Time-conditioned vanilla VAE
- **Stage B**: Metric learning (RHVAE or precision)
- **Stage C**: Full RlVAE with metric alternation

### 2. Modular RlVAE (`ModularRiemannianFlowVAE`)
**Purpose**: Full Riemannian Flow VAE with 100% modular architecture
```yaml
model:
  _target_: models.modular_rlvae.ModularRiemannianFlowVAE
  posterior:
    type: "riemannian_metric"  # gaussian, iaf, riemannian_metric
  sampling:
    method: "geodesic"         # standard, basic, enhanced, geodesic
  encoder:
    architecture: "cnn"        # mlp, cnn, resnet
```

### 3. Modular Vanilla VAE (`ModularVanillaVAE`)
**Purpose**: Stage 1 training for metric extraction
- **Architectures**: MLP, CNN, ResNet encoders/decoders
- **Key Features**: Modular components, Hydra configuration

### 4. Hybrid RlVAE (`HybridRiemannianFlowVAE`)
**Purpose**: Performance-optimized version (2x faster metric computations)
- **Use Case**: Production environments requiring speed
- **Compatibility**: Full backward compatibility with existing training

## 🔧 Configuration System (Hydra)

### Unified Settings Structure
The configuration system uses a unified `settings.*` structure for all parameters:

```bash
# All parameters are under settings.* namespace
python run_experiment.py settings.model.latent_dim=4
python run_experiment.py settings.training.optimizer.lr=1e-3
python run_experiment.py settings.data.common.batch_size=32
python run_experiment.py settings.visualization.level=minimal
```

### Experiment Types
```bash
# Three-stage pipeline with RHMC (recommended)
python run_experiment.py experiment=rlvae_three_stage_long_rhmc_modular

# Default configuration
python run_experiment.py
```

### Model Selection
```bash
# Riemannian RHMC VAE (recommended)
python run_experiment.py model=riemannian_rhmc_vae

# Standard riemannian flow VAE
python run_experiment.py model=riemannian_flow_vae
```

### Training Configurations
```bash
# Development (fast iteration)
python run_experiment.py settings.training.strategy.max_epochs=20

# Full data training
python run_experiment.py training=full_data

# Custom training parameters
python run_experiment.py settings.training.optimizer.lr=1e-3 \
    settings.training.strategy.max_epochs=100
```

### Parameter Overrides
```bash
# Architecture parameters
python run_experiment.py settings.model.latent_dim=4 settings.model.n_flows=12

# Training parameters  
python run_experiment.py settings.training.optimizer.lr=0.001 \
    settings.data.common.batch_size=32

# RHMC parameters
python run_experiment.py settings.model.posterior.rhmc_steps=8 \
    settings.model.posterior.rhmc_step_size=0.01

# Visualization
python run_experiment.py settings.visualization.level=full \
    settings.visualization.frequencies.basic=3
```

## 📁 Project Structure

```
RlVAE/
├── 🧠 src/                                # Core implementation
│   ├── models/                            # Model architectures
│   │   ├── modular_rlvae.py               # Full RlVAE (100% modular)
│   │   ├── modular_vanilla_vae.py         # Stage 1: Metric extraction VAE
│   │   ├── hybrid_rlvae.py                # Performance-optimized (2x faster)
│   │   ├── riemannian_flow_vae.py         # Original implementation
│   │   └── components/                    # Modular components
│   │       ├── encoder_manager.py         # Plug-and-play encoders
│   │       ├── decoder_manager.py         # Plug-and-play decoders
│   │       ├── metric_tensor.py           # Optimized Riemannian metrics
│   │       ├── flow_manager.py            # Normalizing flow management
│   │       ├── loss_manager.py            # Modular loss computation
│   │       └── metric_loader.py           # Pretrained component loading
│   ├── training/                          # Training infrastructure
│   │   ├── lightning_trainer.py           # PyTorch Lightning trainer
│   │   └── plugins/                       # Training plugins
│   │       └── metric_alternation.py      # Metric alternation plugin
│   ├── visualizations/                    # Comprehensive visualization suite
│   └── data/                              # Data handling
├── ⚙️ conf/                               # Hydra configuration system
│   ├── experiment/                        # Experiment types
│   │   ├── rlvae_three_stage_pipeline.yaml    # Three-stage pipeline
│   │   ├── global_vanilla_rlvae_pipeline.yaml # Two-stage pipeline
│   │   ├── comparison_study.yaml               # Model comparisons
│   │   └── single_run.yaml                     # Single experiments
│   ├── model/                             # Model configurations
│   ├── training/                          # Training configurations
│   ├── sweep/                             # Hyperparameter optimization
│   └── visualization/                     # Visualization levels
├── 🛠️ scripts/                           # Automation and utilities
│   ├── slurm/                             # SLURM cluster scripts
│   └── orchestrate_three_stage.py         # Pipeline orchestration
├── 📚 docs/                               # Comprehensive documentation
├── 🧪 tests/                              # Test suite
└── 📄 Configuration files
```

## 🔬 Key Features

### Metric Alternation Training
- **Phase 1**: Train VAE with fixed metric (warmup)
- **Phase 2**: Alternating epochs between VAE and metric training
- **Benefits**: Stable training, gradual metric adaptation

### RHMC Sampling
- **Geodesic Sampling**: Sample along learned Riemannian manifolds
- **Metric-Aware**: Uses learned metric for proper geometric sampling
- **Configurable**: Adjustable step size, leapfrog steps, temperature

### Modular Architecture
- **Plug-and-Play**: Easy component swapping and testing
- **Hydra Integration**: Full configuration management
- **Performance**: 2x faster metric computations

## 📊 Visualization System

### Modular Visualization Architecture
- **Manager**: Centralized orchestration, WandB integration
- **Basic**: Training curves, reconstructions, loss plots
- **Manifold**: Latent space analysis, geodesics, curvature  
- **Interactive**: Interactive plots, animations, hover details
- **Flow Analysis**: Normalizing flow diagnostics
- **Dynamics**: Temporal evolution analysis

### Visualization Levels
```bash
# Development: minimal overhead
python run_experiment.py settings.visualization.level=minimal    # Basic plots every 5 epochs

# Research: balanced analysis
python run_experiment.py settings.visualization.level=standard   # Manifold analysis every 3 epochs

# Publication: comprehensive diagnostics  
python run_experiment.py settings.visualization.level=full       # All modules every epoch

# Three-stage optimized (recommended)
python run_experiment.py settings.visualization.level=three_stage_optimized

# Hyperparameter sweeps: efficiency focused
python run_experiment.py settings.visualization.level=final_only # Complete analysis at end only
```

## 🚀 Hyperparameter Optimization System

### WandB Integration
- **Project**: `rlvae-hyperparameter-optimization`
- **Metrics Tracked**: `val_loss`, `reconstruction_loss`, `kl_loss`, `riemannian_kl`
- **Features**: Real-time monitoring, automatic logging, sweep management

### Sweep Configurations

#### 1. Learning Rate Optimization
```bash
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch learning_rate_optimization 4 50
```
- **Focus**: Training dynamics optimization
- **Parameters**: Learning rate, weight decay, batch size, beta parameters

#### 2. Architecture Optimization
```bash
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch architecture_optimization 2 18
```
- **Focus**: Model architecture comparison
- **Parameters**: Architecture combinations, latent dimensions, flow counts

#### 3. Comprehensive Optimization
```bash
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch comprehensive_hyperparameter_sweep 4 100
```
- **Focus**: Full parameter space optimization
- **Parameters**: All major hyperparameters across pipeline stages

## 🧪 Testing & Validation

### Test Structure
```bash
# Run all tests
python -m pytest tests/ -v

# Specific test categories
python -m pytest tests/test_models.py          # Model architecture validation
python -m pytest tests/test_training.py        # Training loop validation  
python -m pytest tests/test_visualizations.py  # Visualization system
```

### Validation Features
- **Component Testing**: Individual module validation
- **Integration Testing**: End-to-end pipeline validation
- **Configuration Testing**: Hydra config validation
- **Performance Testing**: Memory and speed benchmarks

## 📚 Documentation

### Core Documentation
- **[📖 Installation Guide](docs/installation.md)** - Complete setup, dependencies, troubleshooting
- **[🚀 Training Guide](docs/TRAINING_GUIDE.md)** - Comprehensive training workflows
- **[🔄 Pipeline Guide](docs/RLVAE_THREE_STAGE_PIPELINE.md)** - Three-stage training architecture
- **[📊 Visualization Guide](docs/MODULAR_VISUALIZATION_GUIDE.md)** - Complete visualization system

### Advanced Usage  
- **[⚡ Hyperparameter Optimization](docs/HYPERPARAMETER_OPTIMIZATION_GUIDE.md)** - WandB sweeps, SLURM clusters
- **[🔬 Sweep Guide](docs/SWEEP_README.md)** - Large-scale hyperparameter optimization
- **[🤝 Contributing Guide](docs/CONTRIBUTING.md)** - Development setup, coding standards

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for:
- Development environment setup
- Code style and testing guidelines  
- Pull request process
- Research contribution guidelines

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References & Related Work

### Technical References
- **[RHVAE](https://github.com/clementchadebec/benchmark_VAE)** - Original Riemannian VAE implementation
- **[PyTorch Lightning](https://lightning.ai/)** - Training infrastructure
- **[Hydra](https://hydra.cc/)** - Configuration management

---

## ⚠️ Cursor Agent Rules

All contributors and AI agents **must read and follow the rules in [CURSOR_RULES.md](CURSOR_RULES.md)** before making any changes, running experiments, or answering questions. These rules ensure consistency, quality, and maintainability across the project.

**🚀 Ready to explore Riemannian geometry in your data?** Start with our [Installation Guide](docs/installation.md) or dive into a [Quick Training Example](#quick-start)!

**🔬 For comprehensive project context**, see [`.cursor_context.md`](.cursor_context.md) - essential reading for AI assistants and contributors.

**Need help?** Check the [documentation](docs/) or open an issue for support.

## What's New

- [2024-07-30] **Three-Stage Pipeline**: Complete end-to-end training with metric alternation and RHMC sampling
- [2024-07-30] **Metric Alternation**: Stable training with alternating VAE and metric training phases
- [2024-07-30] **Enhanced KL Divergence**: Improved KL computation with proper metric updates
- [2024-07-30] **Modular Architecture**: 100% modular components for easy experimentation
- [2024-06-10] **Pipeline config rule:** All important parameters now interpolate from top-level config
- [2024-06-10] **Robustness improvement:** Automatic model/data consistency validation 