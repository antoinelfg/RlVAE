# Quickstart: Modular Vanilla VAE + Metric Extraction (Pipeline)

## 1. Quickstart

**Train a vanilla VAE and extract a diverse metric using the global pipeline:**

```bash
python scripts/global_rlvae_pipeline.py --architecture cnn --latent-dim 32 --vae-epochs 50 --wandb
```

- Supported architectures: `cnn`, `resnet`, `mlp`
- Latent dimension: any integer (e.g., 8, 16, 32, 64)
- All outputs and visualizations are saved in `output_dir/vanilla_vae/`
- All visualizations are logged to wandb

## 2. Options

- `--architecture`: Model type (`cnn`, `resnet`, `mlp`)
- `--latent-dim`: Latent space dimension
- `--vae-epochs`: Number of training epochs
- `--wandb`: Enable wandb logging
- `--visualization-level`: Visualization complexity (minimal, standard, full)

## 3. Extensibility

- Add new architectures by extending `EncoderManager` or `DecoderManager`
- Add new visualizations by creating a module in `src/visualizations/` and registering it with the manager

## 4. For More Details

See `GLOBAL_RLVAE_PIPELINE.md` and the updated documentation for quickstart, advanced usage, and extensibility examples.

## 5. Outputs

- All model and metric files are saved in `data/pretrained/` with architecture and latent dim in the filename.
- **Wandb logging**: Training/validation losses, reconstructions, and comprehensive metric analysis graphs (eigenvalue distributions, condition numbers, heatmaps, centroid stats).

## 6. Clean Up Old Runs

- All non-original `.pt` files in `data/pretrained/` are automatically removed by the cleanup process.
- Only the original working files (`encoder.pt`, `decoder.pt`, `metric.pt`) are kept.

## 7. File Structure

- `scripts/global_rlvae_pipeline.py`: Main training and metric extraction script
- `scripts/analyze_metric.py`: Standalone metric analysis (optional)
- `src/models/modular_vanilla_vae.py`: Modular VAE architectures
- `scripts/train_and_extract_vanilla_vae.py`: Original reference script
- `data/pretrained/`: Stores all model/metric files

## 8. Requirements

See `requirements.txt` for dependencies. Install with:

```bash
pip install -r requirements.txt
```

## 9. Credits

- Modular VAE and metric extraction by Antoine Laforgue et al.
- For questions, see the original paper or contact the repo maintainer. 