# Riemannian Flow VAE (RlVAE) 🧠

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-purple.svg)](https://lightning.ai/)
[![Hydra](https://img.shields.io/badge/Hydra-1.3+-green.svg)](https://hydra.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A **modular, high-performance** implementation of **Riemannian Flow VAE** for longitudinal data modeling. This repository provides a comprehensive experimental framework with **fully modular architecture**, extensive visualization capabilities, and systematic model comparison tools.

## 🚀 Quick Start

### Installation & Basic Usage
```bash
# Clone and install
git clone https://github.com/antoinelfg/RlVAE.git
cd RlVAE
pip install -e .

# Quick test (🔥 RECOMMENDED: Fully modular model)
python run_experiment.py model=modular_rlvae training=quick visualization=minimal

# Full training with modular architecture
python run_experiment.py model=modular_rlvae training=full_data visualization=standard

# Compare all models
python run_experiment.py experiment=comparison_study
```

### Validation & Testing
```bash
# Test modular components
python tests/test_modular_components.py

# Test hybrid model integration
python tests/test_hybrid_model.py

# Environment validation
python tests/test_setup.py
```

---

## 🏗️ Architecture Overview

### 🔥 **Modular RlVAE** (PRIMARY RECOMMENDATION)
- **File**: `src/models/modular_rlvae.py`
- **Features**: **100% modular** component architecture
- **Benefits**: Hydra configuration driven, plug-and-play components
- **Research**: Most flexible for experimentation and customization

### ⚡ **Hybrid RlVAE** (PERFORMANCE FOCUSED)
- **File**: `src/models/hybrid_rlvae.py`
- **Performance**: **2x faster** metric computations
- **Use Case**: When you need original compatibility with speed improvements

### 🏛️ **Standard RlVAE** (LEGACY COMPATIBLE)
- **File**: `src/models/riemannian_flow_vae.py`
- **Use Case**: Original implementation for baseline comparisons

### 🧩 **Modular Components**
- **MetricTensor** (`src/models/components/metric_tensor.py`): Optimized Riemannian metric computations
- **MetricLoader** (`src/models/components/metric_loader.py`): Flexible pretrained metric loading
- **FlowManager** (`src/models/components/flow_manager.py`): Temporal flow dynamics
- **LossManager** (`src/models/components/loss_manager.py`): Modular loss computation
- **Samplers** (`src/models/samplers/`): Pluggable sampling strategies

### 📊 **Experimental Framework**
- **Hydra Configuration**: Systematic experiment management
- **Multiple Models**: Modular RlVAE, Hybrid RlVAE, Standard RlVAE, Vanilla VAE
- **Visualization Suite**: From minimal to comprehensive analysis
- **Performance Tracking**: Automatic benchmarking and comparison

---

## 📁 Repository Structure

```
RlVAE/
├── 🧠 src/                          # Core implementation
│   ├── models/
│   │   ├── modular_rlvae.py         # 🔥 PRIMARY: Fully modular model
│   │   ├── hybrid_rlvae.py          # ⚡ 2x faster hybrid model
│   │   ├── riemannian_flow_vae.py   # 🏛️ Original implementation
│   │   ├── components/              # 🧩 Modular components
│   │   │   ├── metric_tensor.py     #     Optimized metric computations
│   │   │   ├── metric_loader.py     #     Flexible metric loading
│   │   │   ├── flow_manager.py      #     Flow dynamics
│   │   │   ├── loss_manager.py      #     Loss computation
│   │   │   ├── encoder_manager.py   #     Pluggable encoders
│   │   │   └── decoder_manager.py   #     Pluggable decoders
│   │   └── samplers/                # 🎯 Sampling strategies
│   │       ├── base_sampler.py      #     Abstract base class
│   │       ├── riemannian_sampler.py#     Enhanced sampling
│   │       ├── hmc_sampler.py       #     HMC sampling
│   │       └── rhvae_sampler.py     #     RHVAE integration
│   ├── training/
│   │   └── lightning_trainer.py     # PyTorch Lightning integration
│   ├── visualizations/              # 🎨 Comprehensive viz suite
│   │   ├── basic.py                 #     Standard plots
│   │   ├── manifold.py              #     Riemannian analysis
│   │   ├── interactive.py           #     Interactive visualizations
│   │   └── flow_analysis.py         #     Flow dynamics
│   └── data/
│       └── cyclic_dataset.py        # Optimized data loading
├── ⚙️ conf/                         # 🔥 Hydra configurations
│   ├── config.yaml                  # Main configuration
│   ├── model/                       # Model configurations
│   │   ├── modular_rlvae.yaml       #   🔥 Modular model (primary)
│   │   ├── hybrid_rlvae.yaml        #   ⚡ Hybrid model
│   │   ├── riemannian_flow_vae.yaml #   🏛️ Standard RlVAE
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
│   ├── TRAINING_GUIDE.md            # Complete training guide
│   ├── MODULAR_TRAINING_GUIDE.md    # Modular system guide
│   ├── MODULAR_VISUALIZATION_GUIDE.md # Visualization system
│   ├── RIEMANNIAN_FLOW_VAE_ANALYSIS.md # Architecture analysis
│   ├── MODULARIZATION_SUMMARY.md    # Executive summary
│   ├── MODULARIZATION_ROADMAP.md    # Development roadmap
│   └── README_EXPERIMENTAL_FRAMEWORK.md # Framework overview
├── 🛠️ scripts/                      # Automation scripts
│   ├── run_weekend_experiments.sh   # Automated experiment suite
│   ├── run_quick_test.sh            # Quick validation
│   ├── monitor_experiments.sh       # Experiment monitoring
│   └── run_manyseq.sbatch          # SLURM batch script
├── 🚀 run_experiment.py             # Main experiment runner
└── Configuration files (pyproject.toml, requirements.txt, etc.)
```

---

## 🎯 Available Models

| Model | Architecture | Use Case | Command |
|-------|-------------|----------|---------|
| **🔥 Modular RlVAE** | **100% modular** | **Primary recommendation for research** | `model=modular_rlvae` |
| ⚡ Hybrid RlVAE | 2x faster | Performance-focused experiments | `model=hybrid_rlvae` |
| 🏛️ Standard RlVAE | Original | Legacy compatibility, comparisons | `model=riemannian_flow_vae` |
| 📊 Vanilla VAE | Baseline | Baseline comparisons | `model=vanilla_vae` |

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

### Single Run (Modular Model)
```bash
python run_experiment.py model=modular_rlvae training=quick visualization=minimal
```

### Model Comparison
```bash
python run_experiment.py experiment=comparison_study
```

### Hyperparameter Sweep
```bash
python run_experiment.py experiment=hyperparameter_sweep -m
```

### Custom Configuration (Modular Focus)
```bash
python run_experiment.py model=modular_rlvae model.latent_dim=32 training.optimizer.lr=0.0005
```

---

## 📈 Performance Benchmarks

### Modular Architecture Benefits
- **Research Flexibility**: 100% configurable components
- **Easy Experimentation**: Plug-and-play architecture
- **Custom Components**: Simple to extend and modify
- **Clean Interfaces**: Well-documented APIs

### Training Speed Comparison
| Model | Architecture | Training Speed | Memory Usage | Flexibility |
|-------|-------------|----------------|--------------|-------------|
| **Modular RlVAE** | **100% modular** | Optimized | Efficient | **Maximum** |
| Hybrid RlVAE | 2x faster metrics | 1.5x faster | Same | High |
| Standard RlVAE | Original | Baseline | Baseline | Limited |
| Vanilla VAE | Simple | Fastest | Lower | Basic |

---

## 🛠️ Development Workflows

### For Research Papers (Modular Focus)
```bash
# 1. Development with modular architecture
python run_experiment.py model=modular_rlvae training=quick visualization=minimal

# 2. Component validation  
python tests/test_modular_components.py

# 3. Full experimentation
python run_experiment.py model=modular_rlvae training=full_data visualization=standard

# 4. Model comparison
python run_experiment.py experiment=comparison_study
```

### For Method Development
```bash
# Modular component development
python run_experiment.py model=modular_rlvae training=quick

# Component testing
python tests/test_modular_components.py

# Integration validation
python tests/test_hybrid_model.py
```

### For Performance Studies
```bash
# Automated experiment suite
scripts/run_weekend_experiments.sh

# Quick validation
scripts/run_quick_test.sh

# Monitoring
scripts/monitor_experiments.sh
```

---

## 🔬 Research Applications

### Longitudinal Data Modeling
- **Medical time series**: Patient progression tracking
- **Financial data**: Market dynamics analysis
- **Scientific data**: Experimental time course analysis

### Riemannian Geometry Research
- **Metric learning**: Custom Riemannian structures
- **Flow dynamics**: Temporal evolution on manifolds
- **Sampling methods**: Advanced MCMC techniques

### Modular Architecture Benefits
- **Component isolation**: Test individual parts separately
- **Easy A/B testing**: Swap components for comparison
- **Custom workflows**: Build your own model variants
- **Research acceleration**: Focus on your contribution

---

## 📚 Documentation

- **[Training Guide](docs/TRAINING_GUIDE.md)**: Complete training workflows
- **[Modular Architecture](docs/MODULARIZATION_SUMMARY.md)**: System architecture details
- **[Experimental Framework](docs/README_EXPERIMENTAL_FRAMEWORK.md)**: Advanced usage patterns
- **[Installation Guide](docs/installation.md)**: Setup and troubleshooting

---

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

For questions, issues, or collaborations, please open an issue on GitHub or contact the maintainers.

---

## 🙏 Acknowledgments

- Original RlVAE research and implementation
- PyTorch Lightning for training infrastructure
- Hydra for configuration management
- The open-source research community

---

**Ready to explore modular Riemannian geometry in your longitudinal data? Start with the modular model!** 🚀

```bash
python run_experiment.py model=modular_rlvae training=quick visualization=minimal
``` 