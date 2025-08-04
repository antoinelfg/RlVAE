# Enhanced Analysis System for RlVAE

This document describes the comprehensive enhanced analysis system for RlVAE models, which provides advanced visualization and evaluation capabilities.

## Overview

The enhanced analysis system provides:

- **Enhanced Generation Analysis**: Multiple sampling methods with FID score evaluation
- **Advanced Inference Analysis**: Latent space trajectory analysis and cycle consistency
- **Geodesic Sampling**: Manifold exploration with Riemannian interpolation
- **Master Visualizations**: Comprehensive dashboards combining all analyses
- **Automated Integration**: Seamless integration with the global pipeline

## Architecture

```
src/evaluation/enhanced_analysis.py    # Core analysis module
scripts/run_enhanced_analysis.py       # Standalone analysis runner
scripts/analyze_existing_checkpoint.py # Analysis for existing checkpoints
scripts/global_rlvae_pipeline.py       # Integrated pipeline (Step 4)
```

## Features

### 1. Enhanced Generation Analysis

- **Multiple Sampling Methods**:
  - Random sampling
  - Grid sampling
  - Gaussian sampling (σ=0.5)
  - Uniform sampling
- **FID Score Evaluation**: Quantitative quality assessment
- **Sample Grids**: Visual comparison of generated samples

### 2. Advanced Inference Analysis

- **Latent Space Trajectories**: Visualization of latent paths
- **Cycle Consistency**: Quantitative measure of reconstruction quality
- **Real vs Reconstructed**: Side-by-side comparison
- **Trajectory Analysis**: PCA-based dimensionality reduction for visualization

### 3. Geodesic Sampling

- **Manifold Exploration**: Interpolation along geodesic paths
- **Multiple Paths**: 5 different geodesic trajectories
- **Visualization**: Path visualization in latent space
- **Sample Interpolation**: Generated samples along paths

### 4. Master Visualizations

- **Comprehensive Dashboard**: 2x3 grid combining all analyses
- **FID Score Comparison**: Bar chart of all sampling methods
- **Cycle Consistency Distribution**: Histogram with mean line
- **Latent Space Visualization**: 2D trajectory plots
- **Generated Samples**: Sample grids for visual inspection
- **Real vs Reconstructed**: Comparison visualization
- **Geodesic Interpolation**: Interpolation samples

## Usage

### 1. Integrated with Global Pipeline

The enhanced analysis is automatically integrated into the global pipeline:

```bash
# Run full pipeline with enhanced analysis
python scripts/global_rlvae_pipeline.py \
    --architecture cnn \
    --latent-dim 16 \
    --vae-epochs 50 \
    --rlvae-epochs 100 \
    --wandb

# Skip enhanced analysis if needed
python scripts/global_rlvae_pipeline.py \
    --architecture cnn \
    --latent-dim 16 \
    --skip-analysis
```

### 2. Standalone Analysis

Run enhanced analysis on existing checkpoints:

```bash
# Using specific checkpoint and config
python scripts/run_enhanced_analysis.py \
    --checkpoint_path outputs/my_run/checkpoints/latest.ckpt \
    --config_path outputs/my_run/configs/config.yaml \
    --output_dir my_analysis_results

# Auto-find latest checkpoint for a run
python scripts/analyze_existing_checkpoint.py \
    --run_name "pipeline_stage2_rlvae_cnn_ld16" \
    --output_dir my_analysis_results
```

### 3. Analysis Parameters

```bash
# Customize analysis parameters
python scripts/run_enhanced_analysis.py \
    --checkpoint_path path/to/checkpoint.ckpt \
    --config_path path/to/config.yaml \
    --num_samples 2000 \        # More samples for generation
    --num_cycles 100 \          # More cycles for inference
    --geodesic_steps 30 \       # More steps for geodesic
    --batch_size 64 \           # Larger batch size
    --log_to_wandb             # Log to wandb
```

## Output Structure

```
enhanced_analysis_outputs/
├── master_analysis.png              # Master visualization dashboard
├── comprehensive_report.json        # Detailed analysis report
├── generation_random.png            # Random sampling results
├── generation_grid.png              # Grid sampling results
├── generation_gaussian.png          # Gaussian sampling results
├── generation_uniform.png           # Uniform sampling results
├── inference_analysis.png           # Inference analysis results
└── geodesic_analysis.png            # Geodesic analysis results
```

## Analysis Report

The comprehensive report includes:

```json
{
  "analysis_timestamp": "2024-01-01T12:00:00",
  "model_config": {
    "latent_dim": 16,
    "input_dim": 4096,
    "hidden_dims": [256, 128, 64]
  },
  "results": {
    "generation": {
      "fid_scores": {
        "random": 466.23,
        "grid": 468.45,
        "gaussian": 467.12,
        "uniform": 469.78
      },
      "best_fid_method": "random"
    },
    "inference": {
      "mean_cycle_consistency": 0.0234,
      "cycle_consistency_std": 0.0089
    }
  }
}
```

## WandB Integration

When `--log_to_wandb` is enabled, the following are logged:

- **Metrics**:
  - `fid_random`, `fid_grid`, `fid_gaussian`, `fid_uniform`
  - `mean_cycle_consistency`, `cycle_consistency_std`
- **Visualizations**:
  - `master_analysis`: Master dashboard image
  - Individual analysis plots

## Technical Details

### Sampling Methods

1. **Random**: Standard normal distribution `N(0, 1)`
2. **Grid**: Uniform grid in first two dimensions
3. **Gaussian**: Normal distribution with σ=0.5
4. **Uniform**: Uniform distribution in [-1, 1]

### Cycle Consistency

Measures how well the model maintains consistency when:
1. Encoding data to latent space
2. Creating a cycle in latent space
3. Decoding back to data space
4. Comparing with original data

### Geodesic Interpolation

Linear interpolation in latent space:
```
z(t) = z_start + t * (z_end - z_start)
```
where `t ∈ [0, 1]` with specified number of steps.

## Performance Considerations

- **Memory**: Analysis can be memory-intensive with large models
- **Time**: FID calculation is the most time-consuming part
- **GPU**: Recommended for faster processing
- **Batch Size**: Adjust based on available memory

## Troubleshooting

### Common Issues

1. **Checkpoint Not Found**: Ensure checkpoint path is correct
2. **Config Not Found**: Ensure config file exists and is valid
3. **Memory Errors**: Reduce batch size or number of samples
4. **FID Calculation Slow**: Consider reducing number of samples

### Debug Mode

Add verbose logging to see detailed progress:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- [ ] Support for different datasets
- [ ] Additional sampling methods
- [ ] Interactive visualizations
- [ ] Batch processing for multiple checkpoints
- [ ] Custom metric evaluation
- [ ] Real-time analysis during training 

## [2024-06-09] Debug Features for NaN/Inf Tracing

- Added debug printouts in `CyclicSpritesDataset.__getitem__` to check for NaN/Inf in each data sample returned.
- Added debug printouts in `LightningRlVAETrainer.validation_step` to print batch statistics, model output stats, and loss values, including explicit checks for NaN/Inf in all tensors.
- These features help trace the origin of NaN/Inf values in the data pipeline and model during validation, aiding in debugging training instabilities or data corruption. 