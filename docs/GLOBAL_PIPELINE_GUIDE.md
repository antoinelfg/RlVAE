# Global Vanilla VAE + RLVAE Pipeline Guide

## Overview

The global pipeline performs a complete two-stage training process:
1. **Stage 1**: Train a vanilla VAE and extract a diverse metric
2. **Stage 2**: Train an RLVAE using the pretrained components from Stage 1

## Basic Usage

```bash
python run_experiment.py experiment=global_vanilla_rlvae_pipeline
```

## Configuration File

The pipeline is configured via `conf/experiment/global_vanilla_rlvae_pipeline.yaml`:

```yaml
# @package experiment
type: "pipeline"
name: "global_vanilla_rlvae_pipeline"
description: "Full pipeline: vanilla VAE with diverse metric extraction, then RLVAE with loaded components."

# Stage 1: Vanilla VAE + Diverse Metric
stage1:
  model: vanilla_vae
  training: full_data
  data: cyclic_sprites
  visualization: minimal
  extract_diverse_metric: true
  # Metric extraction parameters
  architecture: mlp
  latent_dim: 16
  epochs: 50
  temperature: 0.5
  regularization: 0.01
  preset: balanced
  # Metric analysis parameters
  n_heatmaps: 6  # Number of metric matrix heatmaps to show

# Stage 2: RLVAE
stage2:
  model: mlp_rlvae  # or cnn_rlvae, resnet_rlvae, etc.
  training: full_data
  data: cyclic_sprites
  visualization: standard
  load_pretrained_from_stage1: true

# Logging
log_level: "INFO"
log_to_file: true
deterministic: true
benchmark: false
```

## Available Parameters

### Stage 1 Parameters (Vanilla VAE + Metric Extraction)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `architecture` | string | `mlp` | VAE architecture: `mlp`, `cnn`, `resnet`, `pythae` |
| `latent_dim` | int | `16` | Latent space dimension |
| `epochs` | int | `50` | Number of training epochs |
| `temperature` | float | `0.5` | Metric temperature (higher = more diverse eigenvalues) |
| `regularization` | float | `0.01` | Metric regularization (lower = more diverse) |
| `preset` | string | `balanced` | Metric preset: `balanced`, `diverse`, `conservative` |
| `n_heatmaps` | int | `6` | Number of metric matrix heatmaps to display |

### Stage 2 Parameters (RLVAE)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `mlp_rlvae` | RLVAE model: `mlp_rlvae`, `cnn_rlvae`, `resnet_rlvae`, `hybrid_rlvae` |
| `training` | string | `full_data` | Training configuration |
| `data` | string | `cyclic_sprites` | Dataset configuration |
| `visualization` | string | `standard` | Visualization level: `minimal`, `basic`, `standard`, `advanced`, `full` |

### Global Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_level` | string | `INFO` | Logging level |
| `log_to_file` | bool | `true` | Save logs to file |
| `deterministic` | bool | `true` | Enable deterministic training |
| `benchmark` | bool | `false` | Enable CUDA benchmarking |

## Command Line Overrides

You can override any parameter from the command line:

### Stage 1 Overrides

```bash
# Change architecture and latent dimension
python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
  experiment.stage1.architecture=cnn \
  experiment.stage1.latent_dim=32

# Change training parameters
python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
  experiment.stage1.epochs=100 \
  experiment.stage1.temperature=0.7 \
  experiment.stage1.regularization=0.005

# Change metric analysis
python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
  experiment.stage1.n_heatmaps=10
```

### Stage 2 Overrides

```bash
# Change RLVAE model
python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
  experiment.stage2.model=cnn_rlvae

# Change visualization level
python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
  experiment.stage2.visualization=full
```

### Global Overrides

```bash
# Change logging
python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
  experiment.log_level=DEBUG \
  experiment.deterministic=false
```

## Usage Examples

### Basic MLP Pipeline
```bash
python run_experiment.py experiment=global_vanilla_rlvae_pipeline
```

### CNN Pipeline with Custom Parameters
```bash
python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
  experiment.stage1.architecture=cnn \
  experiment.stage1.latent_dim=32 \
  experiment.stage1.epochs=100 \
  experiment.stage1.temperature=0.7 \
  experiment.stage2.model=cnn_rlvae \
  experiment.stage2.visualization=full
```

### High-Dimensional Pipeline
```bash
python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
  experiment.stage1.latent_dim=64 \
  experiment.stage1.n_heatmaps=12 \
  experiment.stage2.model=resnet_rlvae
```

### Fast Pipeline (Minimal Visualizations)
```bash
python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
  experiment.stage1.epochs=25 \
  experiment.stage2.visualization=minimal
```

## Output Files

### Stage 1 Outputs
- **Model files**: `data/pretrained/vae_diverse_<arch>_ld<dim>_<timestamp>.pt`
- **Encoder**: `data/pretrained/encoder_diverse_<arch>_ld<dim>_<timestamp>.pt`
- **Decoder**: `data/pretrained/decoder_diverse_<arch>_ld<dim>_<timestamp>.pt`
- **Metric**: `data/pretrained/metric_diverse_<arch>_ld<dim>_<timestamp>.pt`
- **Results**: `outputs/vanilla_vae_results.yaml`

### Stage 2 Outputs
- **RLVAE model**: `outputs/models/` (best model saved)
- **Visualizations**: `outputs/visualizations/`
- **Logs**: `outputs/logs/`

## WandB Logging

### Stage 1 WandB
- **Project**: `diverse_metric_vae`
- **Run name**: `pipeline_stage1_vanilla_vae_<arch>_ld<dim>`
- **Logs**: Training metrics, reconstructions, metric analysis

### Stage 2 WandB
- **Project**: Based on model config
- **Run name**: Based on model config
- **Logs**: RLVAE training metrics, visualizations

## Metric Analysis

Stage 1 automatically creates and logs:
1. **Eigenvalue distributions** (6 subplots)
2. **Metric matrix heatmaps** (configurable number)
3. **Centroid statistics** (2 subplots)
4. **Metric file as artifact**

## Troubleshooting

### Common Issues

1. **"Unknown experiment type: pipeline"**
   - Ensure you're using the updated `run_experiment.py` with pipeline support

2. **Missing metric heatmaps**
   - Check that matplotlib and seaborn are installed
   - Verify the metric file was created successfully

3. **NaN losses in Stage 2**
   - Check that Stage 1 completed successfully
   - Verify pretrained component compatibility
   - Try different learning rates or architectures

### Debug Mode
```bash
python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
  experiment.log_level=DEBUG
```

## Advanced Configuration

### Custom Model Configs
You can reference custom model configurations:
```bash
python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
  experiment.stage2.model=my_custom_rlvae
```

### Environment Variables
```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py experiment=global_vanilla_rlvae_pipeline
```

### Multi-GPU Training
```bash
python run_experiment.py experiment=global_vanilla_rlvae_pipeline \
  trainer.accelerator=gpu \
  trainer.devices=2
```

## Performance Tips

1. **For faster training**: Use `visualization=minimal` and fewer epochs
2. **For better results**: Use `visualization=full` and more epochs
3. **For debugging**: Use `log_level=DEBUG` and `deterministic=true`
4. **For production**: Use `log_to_file=true` and save all artifacts

## Next Steps

After running the pipeline:
1. Check WandB for training curves and visualizations
2. Inspect the metric analysis plots
3. Use the saved components for further experiments
4. Analyze the RLVAE results and visualizations 