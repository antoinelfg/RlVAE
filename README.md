# RlVAE: Riemannian Latent Variational Autoencoder

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-purple.svg)](https://lightning.ai/)
[![Hydra](https://img.shields.io/badge/Hydra-1.3+-green.svg)](https://hydra.cc/)
[![WandB](https://img.shields.io/badge/WandB-Integrated-orange.svg)](https://wandb.ai/)

A research framework for **Riemannian Longitudinal Variational Autoencoders** applied to longitudinal data modeling. RlVAE learns latent spaces equipped with Riemannian geometry, enabling more faithful modeling of temporal dynamics along learned manifolds.

## 🎯 Research Focus

- **Riemannian Geometry**: Learning data-dependent geometric structures in latent space
- **Temporal Dynamics**: Modeling evolution using normalizing flows on learned manifolds
- **Metric Learning**: Extracting optimal Riemannian metrics from data
- **RHMC Posterior**: Riemannian Hamiltonian Monte Carlo for geometric sampling

## 🏗️ Three-Stage Training Pipeline

### Stage A: Vanilla VAE Warmup
Train a standard VAE to learn initial encoder/decoder representations.

### Stage B: Metric Learning  
Learn a Riemannian metric from the latent space using RHVAE-style centroid-based metrics.

### Stage C: Full RlVAE Training
Train the complete Riemannian Flow VAE with:
- Normalizing flows for temporal dynamics
- RHMC posterior sampling
- Pushforward metric through flows
- Geometric KL divergence

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/antoinelfg/RlVAE.git
cd RlVAE
pip install -r requirements.txt
```

### Basic Training

```bash
# Full three-stage pipeline (recommended)
python run_experiment.py

# Stage C only (with pretrained encoder/decoder/metric)
python run_experiment.py pipeline.run_stage_a=false pipeline.run_stage_b=false

# Quick test run
python run_experiment.py settings.training.strategy.max_epochs=5
```

### Key Parameters

```bash
# Model parameters
python run_experiment.py settings.model.latent_dim=2 settings.model.n_flows=7

# Training parameters
python run_experiment.py settings.training.optimizer.lr=1e-4

# RHMC posterior
python run_experiment.py settings.model.posterior.rhmc_steps=2 \
    settings.model.posterior.rhmc_alpha=0.1

# Metric parameters
python run_experiment.py settings.model.metric.temperature_override=1.0 \
    settings.model.metric.bg_strength=0.005
```

### Visualization Levels

```bash
python run_experiment.py settings.visualization.level=minimal   # Fast sweeps
python run_experiment.py settings.visualization.level=standard  # Development
python run_experiment.py settings.visualization.level=full      # Publication
python run_experiment.py settings.visualization.level=none      # No viz
```

## 📁 Project Structure

```
RlVAE/
├── run_experiment.py              # Main entry point
├── conf/                          # Hydra configuration
│   ├── config.yaml                # Main config
│   ├── sweep/                     # WandB sweep configs
│   └── ...
├── src/
│   ├── rlvae/                     # Core RLVAE package
│   │   ├── models/
│   │   │   ├── modular_rlvae.py   # Main model
│   │   │   ├── factory.py         # Model factory
│   │   │   ├── base/              # Base classes
│   │   │   └── components/        # Modular components
│   │   │       ├── flow_manager.py
│   │   │       ├── loss_manager.py
│   │   │       ├── metric_tensor.py
│   │   │       ├── riemannian_rhmc_posterior.py
│   │   │       └── ...
│   │   └── utils/                 # Debug utilities
│   ├── training/                  # Lightning trainer
│   ├── visualizations/            # Visualization modules
│   ├── evaluation/                # Evaluation tools
│   ├── data/                      # Data loading
│   └── models/                    # Legacy models (RHVAE)
├── scripts/
│   ├── wandb_sweep_agent.py       # Sweep agent
│   └── slurm/                     # Cluster scripts
├── tests/                         # Test suite
└── docs/                          # Documentation
```

## 🔬 Key Components

### ModularRiemannianFlowVAE
The main model combining:
- **Encoder/Decoder**: MLP or CNN architectures
- **Metric Tensor**: Centroid-based Riemannian metric
- **Flow Manager**: Normalizing flows for temporal dynamics
- **RHMC Posterior**: Geometric posterior sampling
- **Loss Manager**: Modular loss computation

### Riemannian Metric
Learned from data using Gaussian mixture-style centroids:
- Temperature-controlled sharpness
- Background identity for stability
- Eigenvalue clamping for numerical stability

### RHMC Posterior
Riemannian Hamiltonian Monte Carlo sampling:
- Respects local geometry via metric
- Configurable integration steps
- Alpha scaling for initial distribution

## 📊 WandB Integration

All experiments are logged to Weights & Biases:

```bash
# Online logging (default)
python run_experiment.py wandb.mode=online

# Offline logging
python run_experiment.py wandb.mode=offline

# Disable WandB
python run_experiment.py wandb.enabled=false
```

### Hyperparameter Sweeps

```bash
# Create and run sweep
wandb sweep conf/sweep/chaos_sweep_v2.yaml
wandb agent <entity>/<project>/<sweep_id>

# Or use SLURM
sbatch scripts/slurm/sweep_agent.sbatch <sweep_id>
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Specific test categories
python -m pytest tests/stagec/ -v
python -m pytest tests/unit/ -v
```

## 📚 Documentation

- **[Installation Guide](docs/installation.md)**
- **[Training Guide](docs/TRAINING_GUIDE.md)**
- **[Pipeline Guide](docs/guides/RLVAE_THREE_STAGE_PIPELINE.md)**
- **[Sweep Guide](LAUNCH_SWEEP.md)**

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔗 References

- **[RHVAE](https://github.com/clementchadebec/benchmark_VAE)** - Original Riemannian VAE
- **[PyTorch Lightning](https://lightning.ai/)** - Training framework
- **[Hydra](https://hydra.cc/)** - Configuration management
- **[WandB](https://wandb.ai/)** - Experiment tracking
