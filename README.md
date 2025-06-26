# Riemannian Flow VAE (RlVAE) 🧠

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-purple.svg)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A clean, structured implementation of **Riemannian Flow VAE** for longitudinal data modeling using cyclic sprites data. This repository provides a comprehensive framework for training and evaluating Riemannian variational autoencoders with geometric-aware latent spaces.

## 📁 Repository Structure

```
RlVAE/
├── src/                              # Source code
│   ├── models/
│   │   └── riemannian_flow_vae.py     # Main model implementation
│   ├── training/
│   │   └── train_cyclic_loop_comparison.py  # Training script
│   └── lib/                           # Pythae library dependency
├── data/                             # Data files
│   ├── raw/                          # Original sprites data
│   │   ├── Sprites_train.pt
│   │   └── Sprites_test.pt
│   ├── processed/                    # Cyclic sequence data
│   │   ├── Sprites_train_cyclic.pt
│   │   ├── Sprites_test_cyclic.pt
│   │   ├── Sprites_train_cyclic_metadata.pt
│   │   └── Sprites_test_cyclic_metadata.pt
│   └── pretrained/                   # Pretrained components
│       ├── encoder.pt                # Pretrained encoder
│       ├── decoder.pt                # Pretrained decoder
│       ├── metric.pt                 # Original metric tensor
│       └── metric_T0.7_scaled.pt     # Temperature-scaled metric
├── scripts/                          # Utility and data preparation scripts
│   ├── train_and_extract_vanilla_vae.py      # Creates metric.pt
│   ├── create_identity_metric_temp_0_7.py    # Creates metric_T0.7_scaled.pt
│   ├── extract_cyclic_sequences.py           # Creates cyclic data
│   └── cleanup_training_files.py             # Training cleanup utility
├── tests/                            # Testing and validation
│   └── test_setup.py                 # Setup validation script
├── docs/                             # Documentation
│   ├── installation.md               # Installation guide
│   ├── guides/
│   │   └── CLEAN_TRAINING_GUIDE.md   # Training documentation
│   └── GITHUB_READY_CHECKLIST.md     # Repository quality checklist
├── .github/                          # GitHub configuration
│   ├── workflows/
│   │   └── ci.yml                    # CI/CD pipeline
│   ├── ISSUE_TEMPLATE/               # Issue templates
│   └── pull_request_template.md      # PR template
├── run_clean_training.py             # Main training entry point
├── run_training.py                   # Alternative training script
├── config.py                         # Configuration management
├── Makefile                          # Development automation
└── [Standard files: README.md, LICENSE, setup.py, etc.]
```

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/antoinelfg/RlVAE.git
cd RlVAE

# Install the package
pip install -e .

# Verify installation
python tests/test_setup.py
```

### 2. Basic Usage
```bash
# Quick training (clean mode - no local files)
python run_clean_training.py --loop_mode open --n_epochs 10

# Full training with all features
python run_clean_training.py --loop_mode open --n_epochs 25 --n_train_samples 3000
```

### 3. Development Mode
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
python tests/test_setup.py

# Format code
black src/ scripts/
```

## 📊 Model Components

### Core Model (`src/models/riemannian_flow_vae.py`)
- **RiemannianFlowVAE**: Main model class with multiple posterior types
- **WorkingRiemannianSampler**: Custom Riemannian sampling methods
- **OfficialRHVAESampler**: Official RHVAE-compatible sampling
- **RiemannianHMCSampler**: Hamiltonian Monte Carlo sampler

### Training (`src/training/train_cyclic_loop_comparison.py`)
- Complete training pipeline with Lightning
- Multiple sampling method experiments
- Wandb integration for experiment tracking
- Automatic checkpointing and visualization

## 🎯 Key Features

### Posterior Types
- **Gaussian**: Standard VAE posterior with optional Riemannian sampling
- **Riemannian Metric**: Metric-aware posterior sampling
- **IAF**: Inverse Autoregressive Flow (future implementation)

### Sampling Methods
- **Standard**: Basic reparameterization trick
- **Custom Riemannian**: Enhanced geodesic-aware sampling
- **Official RHVAE**: Exact RHVAE sampling compatibility

### Loop Modes
- **Open Loop**: Standard temporal modeling
- **Closed Loop**: Cyclic constraints for periodic data

## 📈 Data Pipeline

### Raw Data (`data/raw/`)
- Original sprites datasets with shape/position/scale parameters

### Processed Data (`data/processed/`)
- Cyclic sequences extracted for temporal modeling
- Metadata with cycle information and transformations

### Pretrained Components (`data/pretrained/`)
- **encoder.pt/decoder.pt**: Vanilla VAE components
- **metric.pt**: Original Riemannian metric tensors
- **metric_T0.7_scaled.pt**: Temperature-scaled metric (T=0.7)

## 🛠️ Scripts Usage

### Create Original Metric
```bash
cd scripts
python train_and_extract_vanilla_vae.py
```

### Create Temperature-Scaled Metric
```bash
cd scripts
python create_identity_metric_temp_0_7.py
```

### Extract Cyclic Sequences
```bash
cd scripts
python extract_cyclic_sequences.py
```

## 🧪 Experiment Configuration

The training script supports multiple experimental modes:
- Posterior type selection
- Sampling method comparison
- Loop mode configuration
- Metric tensor variants

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{rlvae2024,
  title={RlVAE: Riemannian Flow VAE for Longitudinal Data},
  author={Antoine Laforgue},
  year={2024},
  url={https://github.com/antoinelfg/RlVAE}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- RHVAE implementation from the Pythae library
- PyTorch Lightning for training infrastructure
- Weights & Biases for experiment tracking 