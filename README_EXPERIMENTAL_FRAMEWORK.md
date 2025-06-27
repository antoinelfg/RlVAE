# 🧪 RlVAE Experimental Framework

A modular, Hydra-powered experimental framework for systematic comparison of Riemannian VAE variants.

## 🚀 Quick Start

### 1. Setup
```bash
# Setup the framework
python setup_experiment_framework.py

# Quick test
./quick_test.sh
```

### 2. Basic Usage
```bash
# Single experiment
python run_experiment.py

# Quick development test
python run_experiment.py training=quick visualization=minimal

# Compare models
python run_experiment.py experiment=comparison_study

# Custom configuration
python run_experiment.py model=vanilla_vae training.n_epochs=50
```

## 🏗️ Architecture

### Modular Components

```
RlVAE/
├── conf/                           # Hydra configurations
│   ├── config.yaml                 # Main config
│   ├── model/
│   │   ├── riemannian_flow_vae.yaml
│   │   └── vanilla_vae.yaml
│   ├── training/
│   │   ├── default.yaml
│   │   └── quick.yaml
│   ├── visualization/
│   │   ├── minimal.yaml
│   │   ├── standard.yaml
│   │   └── full.yaml
│   └── experiment/
│       ├── single_run.yaml
│       ├── comparison_study.yaml
│       └── hyperparameter_sweep.yaml
├── src/
│   ├── models/
│   │   ├── modular_rlvae.py      # Modular model architecture
│   │   └── riemannian_flow_vae.py # Original implementation
│   ├── data/
│   │   └── cyclic_dataset.py      # Lightning data module
│   ├── training/
│   │   └── lightning_trainer.py   # Lightning training wrapper
│   └── visualizations/            # Existing visualization system
├── run_experiment.py              # Main experiment runner
└── outputs/                       # Experiment outputs
```

## 🎯 Experiment Types

### 1. Single Experiment
```bash
python run_experiment.py
```
- Trains one model with current configuration
- Full visualization and logging
- Good for development and testing

### 2. Model Comparison
```bash
python run_experiment.py experiment=comparison_study
```
- Compares Vanilla VAE vs Riemannian VAE
- Standardized metrics for fair comparison
- Automated comparison report

### 3. Hyperparameter Sweep
```bash
python run_experiment.py experiment=hyperparameter_sweep -m
```
- Systematic parameter exploration
- Bayesian optimization support
- Early termination for efficiency

## ⚙️ Configuration System

### Model Configurations

#### Riemannian Flow VAE (Default)
```yaml
# conf/model/riemannian_flow_vae.yaml
posterior:
  type: "riemannian_metric"
sampling:
  method: "geodesic"
  use_riemannian: true
loop:
  mode: "closed"
  penalty: 5.0
```

#### Vanilla VAE (Baseline)
```yaml
# conf/model/vanilla_vae.yaml
posterior:
  type: "gaussian"
sampling:
  method: "standard"
  use_riemannian: false
loop:
  mode: "open"
  penalty: 0.0
```

### Training Configurations

#### Standard Training
```yaml
# conf/training/default.yaml
n_epochs: 25
batch_size: 8
learning_rate: 3e-4
n_train_samples: 1000
```

#### Quick Development
```yaml
# conf/training/quick.yaml
n_epochs: 5
batch_size: 4
n_train_samples: 100
```

### Visualization Levels

| Level | Components | Use Case |
|-------|------------|----------|
| **minimal** | Basic cyclicity only | Quick testing |
| **basic** | + Trajectories, reconstruction | Development |
| **standard** | + Manifold analysis | Regular training |
| **advanced** | + Interactive plots | Research |
| **full** | + Curvature analysis | Publication |

## 🔬 Research-Focused Features

### Standardized Metrics
```python
comparison_metrics = [
    "reconstruction_loss",
    "kl_divergence", 
    "cyclicity_error",
    "latent_space_quality",
    "geodesic_preservation"
]
```

### Automatic Comparison Reports
- Statistical summaries across models
- Visualization of metric evolution
- WandB comparison tables
- Exportable results

### Reproducible Experiments
- Deterministic training with seeds
- Version-controlled configurations
- Complete experiment logging
- Easy result reproduction

## 📊 Usage Examples

### Research Questions

#### Q1: How does Riemannian geometry improve latent spaces?
```bash
python run_experiment.py experiment=comparison_study \
    training.n_epochs=50 \
    visualization=full
```

#### Q2: What's the optimal Riemannian beta value?
```bash
python run_experiment.py experiment=hyperparameter_sweep \
    hydra.sweeper.params="model.riemannian_beta=range(0.5,10,1.5)" \
    -m
```

#### Q3: How do different sampling methods compare?
```bash
python run_experiment.py \
    hydra.sweeper.params="model.sampling.method=geodesic,enhanced,basic,standard" \
    -m
```

### Development Workflows

#### Quick Iteration
```bash
python run_experiment.py training=quick visualization=minimal
```

#### Visualization Testing
```bash
python run_experiment.py training=quick visualization=full
```

#### Model Architecture Testing
```bash
python run_experiment.py model=vanilla_vae training=quick
```

## 🎨 Enhanced Visualizations

### Automatic Generation
- **Cyclicity Analysis**: Track how well models preserve cycles
- **Latent Trajectories**: Visualize temporal evolution in latent space
- **Manifold Quality**: Analyze Riemannian metric properties
- **Interactive Plots**: Explore results dynamically

### Comparison Plots
- Side-by-side model comparisons
- Metric evolution over training
- Statistical significance tests
- Performance benchmarking

## 💾 Output Management

### Structured Outputs
```
outputs/
├── rlvae_comparison/
│   └── 2024-01-15_14-30-00/
│       ├── checkpoints/
│       ├── visualizations/
│       ├── results.yaml
│       └── .hydra/
```

### WandB Integration
- Automatic experiment tracking
- Interactive dashboards
- Model comparison tables
- Visualization galleries

## 🛠️ Advanced Features

### Custom Model Variants
```python
# Easy to add new model configurations
@hydra.main(config_path="conf", config_name="config")
def custom_experiment(cfg):
    # Modify config programmatically
    cfg.model.custom_parameter = value
    runner = ExperimentRunner(cfg)
    runner.run()
```

### Metric Extensions
```python
# Add custom metrics for comparison
class CustomMetricsCollector(MetricsCollector):
    def compute_custom_metric(self, model_output):
        # Your analysis here
        return custom_value
```

### Visualization Extensions
```python
# Add new visualization modules
class CustomVisualization(BaseVisualization):
    def create_custom_plot(self, data):
        # Your visualization here
        pass
```

## 🚨 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Make sure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### CUDA Issues
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

#### Config Errors
```bash
# Validate configuration
python -c "from omegaconf import OmegaConf; print(OmegaConf.load('conf/config.yaml'))"
```

### Performance Tips

#### For Large Experiments
- Use `visualization=minimal` during sweeps
- Enable `wandb.mode=offline` for cluster runs
- Set `training.num_workers=0` if memory constrained

#### For Development
- Use `training=quick` for fast iteration
- Set `experiment.deterministic=false` for speed
- Use smaller `training.n_train_samples`

## 📈 Results Analysis

### Automated Reports
```python
# Generate comparison report
python -c "
from run_experiment import ExperimentRunner
from omegaconf import OmegaConf

config = OmegaConf.load('outputs/rlvae_comparison/latest/results.yaml')
print('Model Performance Summary:')
for model, metrics in config.comparison_summary.items():
    print(f'{model}: {metrics.reconstruction_loss_final:.4f}')
"
```

### Statistical Analysis
- Automatic significance testing
- Confidence intervals
- Effect size calculations
- Power analysis

## 🔄 Migration from Old System

### Automatic Migration
```bash
# Convert old training scripts
python migrate_old_experiments.py
```

### Manual Steps
1. Move configurations to `conf/` directory
2. Update import paths
3. Adapt visualization calls
4. Test with quick experiments

## 🎉 Benefits Summary

### For Research
- **Systematic Comparisons**: Fair, controlled experiments
- **Reproducible Results**: Version-controlled configurations
- **Statistical Rigor**: Automated significance testing
- **Publication Ready**: High-quality visualizations

### For Development
- **Faster Iteration**: Modular, configurable components
- **Better Organization**: Clean separation of concerns
- **Easy Extension**: Plugin-based architecture
- **Performance Control**: Configurable complexity levels

### For Collaboration
- **Shared Configs**: Version-controlled experiment setups
- **Standardized Metrics**: Consistent evaluation protocols
- **Clear Documentation**: Self-documenting configurations
- **Easy Replication**: One-command experiment reproduction 