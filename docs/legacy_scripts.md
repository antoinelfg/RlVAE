# Workflow Migration Overview

The bespoke Hydra scripts have been fully retired. Use the consolidated helper
`scripts/run_workflow.py` (together with `run_experiment.py`) for staged runs and
multiruns, for example:

```
# Stage A only
python scripts/run_workflow.py stage-a

# Stage B only (assuming Stage A artefacts already exist)
python scripts/run_workflow.py stage-b --stage-a-dir outputs/stages/A_RHVAE_mlp_ld16

# Full pipeline
python scripts/run_workflow.py pipeline

# Sweep example
python scripts/run_workflow.py sweep model.latent_dim='[8,16]' seed='range(3)'

# Sampling only
python scripts/run_workflow.py sampling --overrides settings.pipeline.outputs.metric_file=/path/to/metric.pt
```

Diagnostic helpers that do not map cleanly onto the unified configuration have
been removed; build ad-hoc analyses on top of the modern pipeline when needed.

The historical Lightning trainer has also been retired. `src/training/lightning_trainer.py`
is the canonical implementation aligned with the new configuration scheme.
