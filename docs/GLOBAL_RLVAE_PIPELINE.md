# Global RLVAE Pipeline (Unified Modular Workflow)

## Overview

The **Global RLVAE Pipeline** is now a single, unified, modular workflow for end-to-end training and analysis:

1. **Stage 1:** Modular Vanilla VAE training + metric extraction
2. **Stage 2:** Modular RLVAE training (with metric and pretrained components)
3. **Comprehensive Visualizations:** All visualizations from both stages are logged to wandb (with options for large files)
4. **Hydra-Ready:** The pipeline is fully configurable and ready for Hydra-based experiment management
5. **Extensible:** Easily add new priors, samplers, flows, or even replace flows with Riemannian diffusion

---

## Quick Start

### Basic Usage

```bash
# Run the full pipeline (modular vanilla VAE + modular RLVAE)
python scripts/global_rlvae_pipeline.py --architecture cnn --latent-dim 16 --vae-epochs 50 --rlvae-epochs 100 --wandb
```

- All outputs and visualizations are organized in `output_dir/vanilla_vae/` and `output_dir/rlvae/`
- All visualizations are logged to wandb (optionally including large files)

### Hydra Integration (Preview)

The pipeline is structured for easy migration to a single Hydra experiment:

```bash
# (Coming soon) Run the full pipeline with Hydra
python run_experiment.py pipeline=global_rlvae architecture=cnn latent_dim=16 vae_epochs=50 rlvae_epochs=100 visualization=full
```

---

## What It Does

### Stage 1: Modular Vanilla VAE Training + Metric Extraction
- Uses `src/models/modular_vanilla_vae.py` (no legacy scripts)
- Trains a vanilla VAE with the specified architecture and latent dimension
- Extracts a metric for RLVAE
- All visualizations (cyclicity, reconstructions, metric analysis, etc.) are logged to wandb
- Outputs saved in `output_dir/vanilla_vae/`

### Stage 2: Modular RLVAE Training
- Uses `src/models/modular_rlvae.py`
- Loads the pretrained encoder, decoder, and metric from Stage 1
- Trains the full RLVAE with flows and metric-aware sampling
- All visualizations (manifold, flows, recon, cyclicity, etc.) are logged to wandb
- Outputs saved in `output_dir/rlvae/`

### Visualization System
- Uses the modular visualization system (`src/visualizations/`)
- All visualizations are triggered via a manager (not scattered calls)
- Visualization level is configurable (minimal, standard, full)
- Option to include/exclude large files (HTML, high-res images) via a flag

---

## Output Structure

```
output_dir/
├── vanilla_vae/
│   ├── encoder.pt
│   ├── decoder.pt
│   ├── metric.pt
│   └── visualizations/   # All vanilla VAE visualizations
├── rlvae/
│   ├── rlvae_model.pt
│   ├── config.yaml
│   └── visualizations/   # All RLVAE visualizations
├── pipeline_summary.json # Complete experiment summary
```

---

## Extensibility

- **Add new priors:** Extend the relevant manager/component in `src/models/`
- **Add new samplers:** Plug into the modular sampling system
- **Add new flows or Riemannian diffusion:** Extend `FlowManager` or replace as needed
- **Add new visualizations:** Add a new module to `src/visualizations/` and register with the manager

---

## Advanced Usage

- **Visualization Level:**
  - `--visualization-level minimal|standard|full` (controls which visualizations are run)
  - `--include-large-files` (optionally log large files to wandb)
- **Hydra Config:**
  - All parameters (architecture, latent dim, epochs, visualization, etc.) are configurable via Hydra config files
- **Wandb Logging:**
  - By default, both stages log to the same wandb run (with clear grouping/tags)
  - Option to split into separate runs/groups if needed

---

## Troubleshooting

- **Missing outputs:** Check that both `output_dir/vanilla_vae/` and `output_dir/rlvae/` exist and contain the expected files
- **Wandb issues:** Ensure you are logged in and have network access; use `--wandb-offline` for local runs
- **Visualization errors:** Use `--visualization-level minimal` to reduce resource usage

---

## Migration Notes

- **Legacy scripts** are no longer recommended; always use the modular pipeline
- **Documentation** is updated to reflect the new workflow
- **Hydra integration** is the next step for full experiment management

---

## Questions?

See the updated documentation or open an issue for help with the new pipeline! 