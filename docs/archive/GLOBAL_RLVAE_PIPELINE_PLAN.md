# Global RLVAE Pipeline: Refactor & Documentation Update Plan

## 1. **Vision & Goals**

- **Single, unified pipeline** for end-to-end training: vanilla VAE (metric extraction) → modular RLVAE
- **All visualizations** (from both stages) logged to wandb, with options for large files
- **Separation of outputs**: vanilla VAE and RLVAE results/visualizations are organized in distinct subfolders/groups
- **Hydra-ready**: pipeline will be structured for easy migration to a single Hydra experiment
- **Extensible**: easy to add new priors, samplers, flows, or even replace flows with Riemannian diffusion
- **Documentation**: all docs reflect the new workflow, pipeline, and extensibility

---

## 2. **Pipeline Refactor Plan**

### 2.1. **Pipeline Structure**
- **Stage 1:** Modular Vanilla VAE training + metric extraction
  - Uses `modular_vanilla_vae.py` (not legacy scripts)
  - All visualizations (cyclicity, reconstructions, metric analysis, etc.) logged to wandb
  - Outputs saved in `output_dir/vanilla_vae/`
- **Stage 2:** Modular RLVAE training
  - Uses `modular_rlvae.py`
  - All visualizations (manifold, flows, recon, cyclicity, etc.) logged to wandb
  - Outputs saved in `output_dir/rlvae/`
- **Wandb Logging:**
  - By default, both stages log to the same wandb run (with clear grouping/tags)
  - Option to split into separate runs/groups if needed
  - Option to include/exclude large files (images, HTML, etc.) via a flag
- **Hydra Integration:**
  - Pipeline will be callable as a single Hydra experiment (single config, single run)
  - All parameters (architecture, latent dim, epochs, visualization level, etc.) configurable via Hydra
- **Extensibility:**
  - Clear extension points for new priors, samplers, flows, etc.
  - Modular code structure (no monolithic scripts)

### 2.2. **Visualization System**
- Use the new modular visualization system (`src/visualizations/`)
- All visualizations triggered via a manager or master function (not scattered calls)
- Visualizations for both vanilla and RLVAE stages are comprehensive and consistent
- Option to control which visualizations are run (level: minimal, standard, full)
- All visualizations are wandb-logged (with option to save locally if desired)
- Large files (e.g., interactive HTML) are only logged if a flag is set

### 2.3. **Output Organization**
- `output_dir/vanilla_vae/`:
  - Encoder, decoder, metric files
  - Visualizations (images, plots, HTML)
  - wandb logs (if local saving enabled)
- `output_dir/rlvae/`:
  - Trained RLVAE model, config
  - Visualizations (images, plots, HTML)
  - wandb logs (if local saving enabled)
- `output_dir/pipeline_summary.json` (summary of the whole experiment)

### 2.4. **Hydra Experiment Structure**
- Single entry point (e.g., `run_experiment.py` or `main.py`)
- Configurable via Hydra config files (model, training, visualization, experiment)
- All pipeline steps, outputs, and logging controlled via config
- Ready for sweeps, multi-run, and reproducibility

---

## 3. **Documentation Update Plan**

### 3.1. **Files to Update**
- `GLOBAL_RLVAE_PIPELINE.md`: Rewrite to reflect the new pipeline, visualization, and Hydra integration
- `MODULARIZATION_SUMMARY.md`: Update to clarify the new pipeline as the main workflow
- `MODULAR_TRAINING_GUIDE.md`: Update to show how modular training and visualization are integrated in the pipeline
- `MODULAR_VISUALIZATION_GUIDE.md`: Ensure it describes how visualizations are called from the pipeline
- `README_MODULAR_VANILLA_VAE.md`: Update to clarify that vanilla VAE is always run via the modular system
- `QUICKSTART_METRIC_VAE.md`: Update to point to the new pipeline for metric extraction
- `TRAINING_GUIDE.md`: Update to show the new pipeline as the main entry point
- `README.md`: Update quickstart and architecture overview
- `README_EXPERIMENTAL_FRAMEWORK.md`: Update to show the global pipeline as the main experiment runner
- Remove or archive any docs that reference legacy scripts or workflows

### 3.2. **New/Modified Content**
- **Quickstart:** How to run the full pipeline (with/without Hydra)
- **Visualization:** How all visualizations are logged to wandb, options for large files
- **Extensibility:** How to add new priors, samplers, flows, etc.
- **Hydra Usage:** How to configure and run the pipeline as a single experiment
- **Output Structure:** Clear description of output directories and wandb logging
- **Troubleshooting:** Common issues with visualization/logging

---

## 4. **Implementation Steps**

1. **Read and map the current code for modular vanilla VAE, modular RLVAE, and visualization system**
2. **Refactor the global pipeline script**:
   - Replace legacy calls with modular components
   - Integrate visualization manager for both stages
   - Add wandb logging for all visualizations (with large file flag)
   - Organize outputs as described
   - Add config options for Hydra
3. **Update documentation as per plan**
4. **Test the new pipeline end-to-end**
5. **(Optional) Migrate to full Hydra experiment**

---

## 5. **Open Questions / Decisions**
- [ ] Confirm which visualizations are "large" and should be optional
- [ ] Decide on default wandb logging structure (single run vs. group)
- [ ] Confirm if any legacy scripts should be kept for reference

---

## 6. **Summary Table: Docs Actions**

| File                                 | Action         | Notes                                    |
|--------------------------------------|----------------|------------------------------------------|
| GLOBAL_RLVAE_PIPELINE.md             | Rewrite        | New pipeline, visualization, Hydra        |
| MODULARIZATION_SUMMARY.md            | Update         | Pipeline as main workflow                 |
| MODULAR_TRAINING_GUIDE.md            | Update         | Modular training in pipeline              |
| MODULAR_VISUALIZATION_GUIDE.md       | Update         | Visualization manager, wandb logging      |
| README_MODULAR_VANILLA_VAE.md        | Update         | Modular vanilla VAE only                  |
| QUICKSTART_METRIC_VAE.md             | Update         | Point to new pipeline                    |
| TRAINING_GUIDE.md                    | Update         | Pipeline as main entry                    |
| README.md                            | Update         | Quickstart, architecture                  |
| README_EXPERIMENTAL_FRAMEWORK.md     | Update         | Pipeline as main runner                   |
| Legacy/old docs                      | Remove/archive | No longer referenced                      |

---

## 7. **Next Steps**
- Await user approval or feedback
- Once approved, proceed to code and docs refactor as per this plan 