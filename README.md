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

## 🏗️ Architecture: Two-Stage Training Pipeline

### Stage 1: Vanilla VAE + Metric Extraction
```bash
# Train vanilla VAE and extract Riemannian metric
python run_experiment.py experiment=vanilla_diverse_metric \
    model.latent_dim=32 training.max_epochs=50
```
- **Purpose**: Learn data representations and extract geometric structure
- **Output**: Pretrained encoder, decoder, and **Riemannian metric tensor**
- **Naming**: `pipeline_stage1_vanilla_vae_{architecture}_ld{latent_dim}`
- **Architectures**: MLP, CNN, ResNet with modular components

### Stage 2: RlVAE Training with Loaded Components
```bash
# Train RlVAE using extracted components
python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
    model=cnn_rlvae model.latent_dim=32 training.max_epochs=100
```
- **Purpose**: Train full Riemannian Flow VAE using Stage 1 outputs
- **Features**: Geometric constraints, flow dynamics, metric-aware sampling
- **Naming**: `pipeline_stage2_rlvae_{architecture}_ld{latent_dim}`

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/antoinelfg/RlVAE.git
cd RlVAE
pip install -r requirements.txt
```

### Basic Usage Examples

#### Single Model Training
```bash
# Quick development run (20 epochs, small data)
python run_experiment.py experiment=single_run training=quick model=mlp_rlvae

# Production CNN training (50 epochs, full data)
python run_experiment.py experiment=single_run training=full_data model=cnn_rlvae

# Architecture comparison study
python run_experiment.py experiment=comparison_study
```

#### Hyperparameter Optimization
```bash
# Learning rate optimization (50 runs)
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch learning_rate_optimization 4 50

# Architecture optimization (grid search)
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch architecture_optimization 2 20

# Comprehensive optimization (100 runs, 4 parallel agents)
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch comprehensive_hyperparameter_sweep 4 100
```

#### Advanced Configuration
```bash
# Custom parameter overrides
python run_experiment.py model=cnn_rlvae \
    model.latent_dim=64 \
    model.riemannian_beta=10.0 \
    model.n_flows=12 \
    training.optimizer.lr=0.001 \
    visualization=full

# Multiple parallel sweep agents
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch comprehensive_hyperparameter_sweep 4 25
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch comprehensive_hyperparameter_sweep 4 25
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch comprehensive_hyperparameter_sweep 4 25
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch comprehensive_hyperparameter_sweep 4 25
```

## 📁 Project Structure

```
RlVAE/
├── 🧠 src/                                # Core implementation
│   ├── models/                            # Model architectures
│   │   ├── modular_vanilla_vae.py         # Stage 1: Metric extraction VAE
│   │   ├── modular_rlvae.py               # Stage 2: Full RlVAE (100% modular)
│   │   ├── hybrid_rlvae.py                # Performance-optimized (2x faster)
│   │   ├── riemannian_flow_vae.py         # Original implementation
│   │   └── components/                    # Modular components
│   │       ├── encoder_manager.py         # Plug-and-play encoders (MLP/CNN/ResNet)
│   │       ├── decoder_manager.py         # Plug-and-play decoders
│   │       ├── metric_tensor.py           # Optimized Riemannian metrics (2x faster)
│   │       ├── flow_manager.py            # Normalizing flow management
│   │       ├── loss_manager.py            # Modular loss computation
│   │       └── metric_loader.py           # Pretrained component loading
│   ├── visualizations/                    # Comprehensive visualization suite
│   │   ├── manager.py                     # Visualization orchestration
│   │   ├── basic.py                       # Training curves, reconstructions
│   │   ├── manifold.py                    # Latent space analysis, geodesics
│   │   ├── interactive.py                 # Interactive plots, animations
│   │   ├── flow_analysis.py               # Flow dynamics analysis
│   │   └── latent_dynamics.py             # Temporal evolution
│   ├── training/                          # Training infrastructure
│   └── data/                              # Data handling
├── ⚙️ conf/                               # Hydra configuration system
│   ├── experiment/                        # Experiment types
│   │   ├── global_vanilla_rlvae_pipeline.yaml    # Two-stage pipeline
│   │   ├── comparison_study.yaml                 # Model comparisons
│   │   └── single_run.yaml                       # Single experiments
│   ├── model/                             # Model configurations
│   │   ├── vanilla_vae.yaml               # Vanilla VAE baseline
│   │   ├── mlp_rlvae.yaml                 # MLP architecture
│   │   ├── cnn_rlvae.yaml                 # CNN architecture
│   │   └── resnet_rlvae.yaml              # ResNet architecture
│   ├── training/                          # Training configurations
│   │   ├── quick.yaml                     # Development (20 epochs, small data)
│   │   ├── default.yaml                   # Standard (100 epochs)
│   │   └── full_data.yaml                 # Production (50 epochs, full data)
│   ├── sweep/                             # Hyperparameter optimization
│   │   ├── learning_rate_optimization.yaml        # LR/weight decay (50 runs)
│   │   ├── architecture_optimization.yaml         # Architecture comparison
│   │   └── comprehensive_hyperparameter_sweep.yaml # Full optimization (100 runs)
│   └── visualization/                     # Visualization levels
│       ├── minimal.yaml                   # Basic plots (dev)
│       ├── standard.yaml                  # Balanced analysis
│       ├── full.yaml                      # Comprehensive diagnostics
│       ├── final_only.yaml                # End-of-training only
│       └── end_only.yaml                  # No training visuals
├── 🛠️ scripts/                           # Automation and utilities
│   ├── slurm/                             # SLURM cluster scripts
│   │   ├── run_hyperparameter_sweep.sbatch        # Standard sweeps (47h)
│   │   ├── run_hyperparameter_sweep_short.sbatch  # Test sweeps (4h)
│   │   ├── run_extended_sweep.sh                  # Unlimited via chunking
│   │   └── run_experiment_rlvae.sbatch            # Single experiments
│   ├── run_sweep.py                       # Hyperparameter sweep runner
│   ├── global_rlvae_pipeline.py           # Pipeline orchestration
│   └── train_diverse_metric_vae.py        # Metric extraction utilities
├── 📚 docs/                               # Comprehensive documentation
├── 🧪 tests/                              # Test suite
└── 📄 Configuration files (pyproject.toml, requirements.txt, etc.)
```

## 🤖 Model Architectures

### 1. Modular Vanilla VAE (`ModularVanillaVAE`)
**Purpose**: Stage 1 training for metric extraction
```python
# Factory functions for different architectures
vae = create_cnn_vanilla_vae(input_dim=(3, 64, 64), latent_dim=32, beta=1.0)
vae = create_resnet_vanilla_vae(input_dim=(3, 64, 64), latent_dim=32, beta=1.0)
vae = create_mlp_vanilla_vae(input_dim=(3, 64, 64), latent_dim=32, beta=1.0)
```
- **Features**: Modular encoder/decoder, Hydra configuration
- **Architectures**: MLP, CNN, ResNet with configurable depth/width
- **Output**: Trained VAE + extracted Riemannian metric tensor

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
- **Key Features**: Configuration-driven, plug-and-play components
- **Modular Components**: MetricTensor, FlowManager, LossManager, MetricLoader
- **Posterior Types**: Gaussian, IAF, Riemannian metric-aware

### 3. Hybrid RlVAE (`HybridRiemannianFlowVAE`)
**Purpose**: Performance-optimized version (2x faster metric computations)
- **Use Case**: Production environments requiring speed
- **Compatibility**: Full backward compatibility with existing training
- **Performance**: 2x faster metric tensor operations

### 4. Standard RlVAE (`RiemannianFlowVAE`)
**Purpose**: Original implementation for baseline comparisons
- **Features**: Multiple sampling methods, legacy compatibility
- **Use Case**: Research baselines, method validation

| Model | Speed | Modularity | Use Case |
|-------|-------|------------|----------|
| **Modular RlVAE** | Fast | 100% | **Primary recommendation for research** |
| **Hybrid RlVAE** | 2x faster | High | Performance-focused experiments |
| **Standard RlVAE** | Baseline | Limited | Legacy compatibility, baselines |
| **Vanilla VAE** | Fastest | High | Stage 1, comparisons |

## 🔧 Configuration System (Hydra)

### Model Selection
```bash
# Choose model architecture
python run_experiment.py model=vanilla_vae      # Vanilla VAE baseline
python run_experiment.py model=mlp_rlvae        # MLP Riemannian VAE
python run_experiment.py model=cnn_rlvae        # CNN Riemannian VAE
python run_experiment.py model=resnet_rlvae     # ResNet Riemannian VAE
```

### Training Configurations
```bash
# Development (fast iteration)
python run_experiment.py training=quick         # 20 epochs, small data

# Standard research
python run_experiment.py training=default       # 100 epochs, balanced

# Production
python run_experiment.py training=full_data     # 50 epochs, full dataset
```

### Parameter Overrides
```bash
# Architecture parameters
python run_experiment.py model.latent_dim=64 model.n_flows=12

# Training parameters  
python run_experiment.py training.optimizer.lr=0.001 training.data.batch_size=32

# Riemannian parameters
python run_experiment.py model.riemannian_beta=10.0 model.sampling.method=geodesic

# Visualization
python run_experiment.py visualization=full visualization.frequency=5
```

## 🚀 Hyperparameter Optimization System

### WandB Integration
- **Project**: `rlvae-hyperparameter-optimization`
- **Metrics Tracked**: `val_loss`, `reconstruction_loss`, `kl_loss`, `riemannian_kl`
- **Features**: Real-time monitoring, automatic logging, sweep management

### Sweep Configurations

#### 1. Learning Rate Optimization (`learning_rate_optimization.yaml`)
```bash
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch learning_rate_optimization 4 50
```
- **Focus**: Training dynamics optimization
- **Parameters**: Learning rate (1e-5 to 1e-2), weight decay (1e-6 to 1e-3), batch size (8,16,32,64)
- **Additional**: Beta parameters, training epochs, data efficiency
- **Method**: Random search with 50 runs
- **Early Termination**: Hyperband (eta=2, max_iter=30)

#### 2. Architecture Optimization (`architecture_optimization.yaml`)
```bash
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch architecture_optimization 2 18
```
- **Focus**: Model architecture comparison
- **Parameters**: Stage1 arch (mlp,cnn,resnet), Stage2 arch (mlp,cnn,resnet), latent_dim (16,32,64)
- **Additional**: Flow count (6,8,12), Riemannian beta (1,5,10)
- **Method**: Grid search for systematic comparison
- **Total Combinations**: 3×3×3×3×3 = 243 configurations

#### 3. Comprehensive Optimization (`comprehensive_hyperparameter_sweep.yaml`)
```bash
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch comprehensive_hyperparameter_sweep 4 100
```
- **Focus**: Full parameter space optimization
- **Stage 1 Parameters**: Architecture, latent_dim, epochs, temperature, regularization
- **Stage 2 Parameters**: n_flows, beta, riemannian_beta, sampling_method, loop_mode
- **Training Parameters**: lr, weight_decay, batch_size, max_epochs
- **Data Parameters**: n_train_samples, n_val_samples
- **Method**: Random search with 100 runs
- **Early Termination**: Hyperband (eta=3, max_iter=50)

### SLURM Cluster Integration

#### Batch Scripts (`scripts/slurm/`)
```bash
# Standard 47-hour sweeps
run_hyperparameter_sweep.sbatch <sweep_config> <agents> <max_runs>

# Short 4-hour test sweeps  
run_hyperparameter_sweep_short.sbatch <sweep_config> <agents> <max_runs>

# Unlimited duration via automatic chunking
run_extended_sweep.sh <sweep_config> <total_runs>

# Single experiment execution
run_experiment_rlvae.sbatch
```

#### Usage Patterns
```bash
# Single sweep with multiple agents
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch learning_rate_optimization 4 50

# Parallel sweeps for faster completion
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch comprehensive_hyperparameter_sweep 4 25
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch comprehensive_hyperparameter_sweep 4 25  
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch comprehensive_hyperparameter_sweep 4 25
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch comprehensive_hyperparameter_sweep 4 25
# Total: 16 agents working on 100 runs = ~6.25 runs per agent

# Extended sweep for very long optimizations
./scripts/slurm/run_extended_sweep.sh comprehensive_hyperparameter_sweep 500
```

#### Resource Management
- **Time Limits**: 47 hours (GPU partition limit), 4 hours (short tests)
- **GPU Usage**: 1 GPU per agent, automatic allocation
- **Memory**: Optimized for batch sizes 8-64
- **Monitoring**: Automatic log collection, WandB integration

## 📊 Visualization System

### Modular Architecture (`src/visualizations/`)
- **Manager** (`manager.py`): Centralized orchestration, WandB integration
- **Basic** (`basic.py`): Training curves, loss plots, reconstructions
- **Manifold** (`manifold.py`): Latent space analysis, geodesics, curvature  
- **Interactive** (`interactive.py`): Interactive plots, animations, hover details
- **Flow Analysis** (`flow_analysis.py`): Normalizing flow diagnostics
- **Dynamics** (`latent_dynamics.py`): Temporal evolution analysis

### Visualization Levels
```bash
# Development: minimal overhead
python run_experiment.py visualization=minimal    # Basic plots every 5 epochs

# Research: balanced analysis
python run_experiment.py visualization=standard   # Manifold analysis every 3 epochs

# Publication: comprehensive diagnostics  
python run_experiment.py visualization=full       # All modules every epoch

# Hyperparameter sweeps: efficiency focused
python run_experiment.py visualization=final_only # Complete analysis at end only

# No training visuals: maximum performance
python run_experiment.py visualization=end_only   # Disable during training
```

### Key Features
- **WandB Integration**: Automatic logging of all plots and metrics
- **Cluster Analysis**: Sequence clustering with color coding
- **Interactive Elements**: Hover details, zoom, pan capabilities
- **Performance Optimization**: Configurable sequence limits, smart batching
- **Comparative Analysis**: Side-by-side model comparisons

## 🔬 Sampling Methods

### Standard VAE (`method: standard`)
```python
z = mu + eps * torch.exp(0.5 * log_var)  # Standard reparameterization
```

### Riemannian Sampling Methods
```yaml
sampling:
  method: "geodesic"        # Choose method
  use_riemannian: true      # Enable geometric features
```

#### Available Methods:
- **`basic`**: Simple Riemannian correction with metric conditioning
- **`enhanced`**: Advanced Riemannian sampling with geometric constraints  
- **`geodesic`**: Geodesic sampling on learned Riemannian manifold
- **`official`**: Official RHVAE implementation integration

### Performance Comparison
| Method | Speed | Geometric Accuracy | Use Case |
|--------|-------|-------------------|----------|
| `standard` | Fastest | None | Vanilla VAE baseline |
| `basic` | Fast | Low | Simple geometric correction |
| `enhanced` | Medium | Medium | Balanced performance/accuracy |
| `geodesic` | Slower | High | **Best geometric fidelity** |
| `official` | Slowest | High | Validation against RHVAE |

## 📈 Performance & Scalability

### Memory Management
- **Mixed Precision**: 16-bit training for 2x memory efficiency
- **Batch Sizes**: Adaptive sizing (4-64) based on architecture
- **Data Loading**: Parallel workers (4-16), pin memory, persistent workers

### Compute Optimization  
- **Device Management**: Automatic GPU/CPU selection and placement
- **Single GPU**: Optimized for single GPU training (multi-GPU ready)
- **Metric Computation**: 2x faster with modular MetricTensor component
- **Caching**: Intelligent metric tensor and flow caching

### Scalability Features
- **SLURM Native**: Full cluster computing support
- **Parallel Sweeps**: Multiple agents per hyperparameter sweep
- **Extended Runs**: Automatic chunking for unlimited duration experiments
- **Resource Monitoring**: Automatic GPU utilization and memory tracking

## 🧪 Testing & Validation

### Test Structure
```bash
# Run all tests
python -m pytest tests/ -v

# Specific test categories
python -m pytest tests/test_models.py          # Model architecture validation
python -m pytest tests/test_training.py        # Training loop validation  
python -m pytest tests/test_visualizations.py  # Visualization system
python -m pytest tests/test_config.py          # Hydra configuration
```

### Validation Features
- **Component Testing**: Individual module validation
- **Integration Testing**: End-to-end pipeline validation
- **Configuration Testing**: Hydra config validation
- **Performance Testing**: Memory and speed benchmarks

## 📚 Documentation

### Core Documentation
- **[📖 Installation Guide](docs/installation.md)** - Complete setup, dependencies, troubleshooting
- **[🚀 Training Guide](docs/TRAINING_GUIDE.md)** - Comprehensive training workflows and best practices
- **[🔄 Pipeline Guide](docs/GLOBAL_PIPELINE_GUIDE.md)** - Two-stage training architecture
- **[📊 Visualization Guide](docs/MODULAR_VISUALIZATION_GUIDE.md)** - Complete visualization system

### Advanced Usage  
- **[⚡ Hyperparameter Optimization](docs/HYPERPARAMETER_OPTIMIZATION_GUIDE.md)** - WandB sweeps, SLURM clusters
- **[🔬 Sweep Guide](docs/SWEEP_README.md)** - Large-scale hyperparameter optimization
- **[🤝 Contributing Guide](docs/CONTRIBUTING.md)** - Development setup, coding standards

### Research Documentation
- **[🏗️ RlVAE Architecture](docs/GLOBAL_RLVAE_PIPELINE.md)** - Core pipeline architecture and design principles

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

**🚀 Ready to explore Riemannian geometry in your data?** Start with our [Installation Guide](docs/installation.md) or dive into a [Quick Training Example](#quick-start)!

**🔬 For comprehensive project context**, see [`.cursor_context.md`](.cursor_context.md) - essential reading for AI assistants and contributors.

**Need help?** Check the [documentation](docs/) or open an issue for support. 