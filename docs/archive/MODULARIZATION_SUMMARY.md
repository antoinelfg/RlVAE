# RlVAE Modular Architecture: Unified Pipeline Implementation

## Executive Summary

The **Global RLVAE Pipeline** is now the main workflow for all research and experimentation. It leverages the fully modular architecture for both vanilla VAE and RLVAE stages, with comprehensive visualization and wandb integration.

---

## ✅ Unified Modular Pipeline

- **Single entry point:** End-to-end training and analysis (vanilla VAE → RLVAE)
- **Modular components:** Encoder, decoder, metric, flows, loss, sampling
- **Visualization system:** All visualizations are managed and logged via the modular visualization manager
- **Hydra-ready:** All configuration is centralized and ready for Hydra-based experiment management
- **Extensible:** New priors, samplers, flows, and visualizations can be added with minimal changes

---

## Architecture Overview

```
RlVAE/
├── src/models/
│   ├── modular_vanilla_vae.py      # Modular vanilla VAE (Stage 1)
│   ├── modular_rlvae.py            # Modular RLVAE (Stage 2)
│   ├── components/                 # Modular components (metric, flow, loss, etc.)
│   └── samplers/                   # Modular sampling strategies
├── src/visualizations/             # Modular visualization system
├── scripts/global_rlvae_pipeline.py # Unified pipeline script
├── conf/                           # Hydra configs (coming soon)
```

---

## Visualization Integration

- **All visualizations** (cyclicity, manifold, flows, recon, etc.) are managed by the visualization manager
- **Configurable levels:** minimal, standard, full
- **Wandb logging:** All visualizations are logged to wandb, with options for large files

---

## Extensibility

- **Add new priors, samplers, flows:** Extend the relevant manager/component
- **Add new visualizations:** Add a new module to `src/visualizations/` and register it
- **Hydra integration:** The pipeline is ready for full Hydra experiment management

---

## Usage Pattern

- **Always use the global pipeline** for new experiments
- **Configure everything** via the pipeline script or (soon) Hydra configs
- **Legacy scripts** are deprecated and should not be used

---

## For More Details

See `GLOBAL_RLVAE_PIPELINE.md` and the updated documentation for quickstart, advanced usage, and extensibility examples. 