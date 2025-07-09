# RlVAE: Riemannian Flow VAE Research Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, extensible research framework for Riemannian Flow Variational Autoencoders (RlVAE) with comprehensive training, visualization, and hyperparameter optimization capabilities.

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/your-username/RlVAE.git
cd RlVAE
pip install -r requirements.txt
```

### Basic Training
```bash
# Train both vanilla VAE (stage 1) and RlVAE (stage 2)
python run_experiment.py experiment=basic_training

# Train with specific configuration
python run_experiment.py experiment=cnn_training model.latent_dim=32
```

### Hyperparameter Optimization
```bash
# Run hyperparameter sweep
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch learning_rate_optimization 4 10
```

## 📁 Project Structure

```
RlVAE/
├── src/                          # Source code
│   ├── models/                   # VAE and RlVAE implementations
│   ├── components/               # Modular components (encoders, decoders, flows)
│   ├── metrics/                  # Evaluation metrics
│   └── visualizations/           # Visualization modules
├── conf/                         # Hydra configuration files
│   ├── experiment/               # Experiment configurations
│   ├── model/                    # Model configurations
│   ├── training/                 # Training configurations
│   └── sweep/                    # Hyperparameter sweep configurations
├── scripts/                      # Utility scripts
│   ├── slurm/                    # SLURM batch scripts
│   └── run_sweep.py              # Hyperparameter sweep runner
├── docs/                         # Documentation
└── tests/                        # Test files
```

## 📖 Documentation

### Core Guides
- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[Training Guide](docs/TRAINING_GUIDE.md)** - Comprehensive training documentation
- **[Pipeline Guide](docs/GLOBAL_PIPELINE_GUIDE.md)** - Two-stage training pipeline
- **[Visualization Guide](docs/MODULAR_VISUALIZATION_GUIDE.md)** - Visualization system

### Advanced Usage
- **[Hyperparameter Optimization](docs/HYPERPARAMETER_OPTIMIZATION_GUIDE.md)** - WandB sweeps and optimization
- **[Sweep Guide](docs/SWEEP_README.md)** - Running large-scale hyperparameter sweeps
- **[Contributing Guide](docs/CONTRIBUTING.md)** - Development guidelines

### SLURM Scripts
All SLURM batch scripts are located in `scripts/slurm/`:
- `run_hyperparameter_sweep.sbatch` - Standard hyperparameter sweeps (47h limit)
- `run_hyperparameter_sweep_short.sbatch` - Short test sweeps (4h limit)  
- `run_extended_sweep.sh` - Extended sweeps with automatic chunking
- `run_experiment_rlvae.sbatch` - Single experiment execution

## 🏗️ Architecture

### Two-Stage Training Pipeline
1. **Stage 1**: Train vanilla VAE for metric extraction [[memory:2691368]]
2. **Stage 2**: Train RlVAE using extracted metrics

### Modular Components
- **Encoders/Decoders**: CNN, MLP architectures
- **Flows**: Normalizing flows for Riemannian manifolds
- **Metrics**: FID, LPIPS, reconstruction metrics
- **Visualizations**: Latent space, reconstructions, training curves

## 🔧 Configuration

The project uses [Hydra](https://hydra.cc/) for configuration management:

```bash
# Override specific parameters
python run_experiment.py model.latent_dim=64 training.optimizer.lr=0.001

# Use different experiment configuration
python run_experiment.py experiment=architecture_comparison
```

## 🚀 Hyperparameter Optimization

### WandB Sweeps [[memory:2741052]]
The project supports large-scale hyperparameter optimization using WandB:

```bash
# Create and run a sweep
python scripts/run_sweep.py --sweep-config learning_rate_optimization --agent-count 4

# Run multiple agents in parallel
sbatch scripts/slurm/run_hyperparameter_sweep.sbatch comprehensive_hyperparameter_sweep 4 20
```

Available sweep configurations:
- `learning_rate_optimization` - Learning rate and weight decay
- `architecture_optimization` - Model architectures and dimensions
- `comprehensive_hyperparameter_sweep` - Full parameter space

## 📊 Monitoring and Visualization

- **WandB Integration**: All experiments logged automatically
- **Real-time Metrics**: Training progress, validation metrics
- **Visualizations**: Latent space evolution, reconstructions
- **Sweep Monitoring**: Hyperparameter optimization progress

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_models.py
python -m pytest tests/test_training.py
```

## 📈 Performance

- **Scalable**: Supports multi-GPU training via SLURM
- **Efficient**: Modular design allows selective component usage
- **Reproducible**: Hydra configuration ensures experiment reproducibility

## 🤝 Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for development setup, coding standards, and contribution guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- [Variational Autoencoders](https://arxiv.org/abs/1312.6114)
- [Normalizing Flows](https://arxiv.org/abs/1505.05770)  
- [Riemannian Manifold Learning](https://arxiv.org/abs/2006.10411)

---

**Need help?** Check the [documentation](docs/) or open an issue for support. 