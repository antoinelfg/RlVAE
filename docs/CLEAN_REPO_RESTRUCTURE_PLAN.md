# RlVAE Clean Repository Restructure Plan

> Goal: converge to one modern, modular package under `src/rlvae/` with a single source of truth for models, training, configs, and tools. Preserve backwards compatibility via thin wrappers until migration is complete. Keep Hydra and Lightning first‑class, make demos/examples lightweight, and tests actionable.

## Guiding Principles

- Single package root: `src/rlvae/` contains all code (no split across multiple top‑level trees).
- One canonical original model and one modular model; same API and loss keys.
- Strict interfaces and routing: trainers and Hydra configs import only from `rlvae.*`.
- Backwards compatibility: legacy `src/models/*` and `original_rlvae/*` are adapters during migration.
- Minimal duplication: shared utilities in `rlvae.utils` and `rlvae.models.components`.
- Tests verify invariants (identity metric, α scaling, geomean invariance) and trainer I/O.
- Docs reflect the new layout; legacy docs archived under `docs/archive/`.

## Final Target Layout

```
src/
  rlvae/
    __init__.py
    utils/
      __init__.py
      reproducibility.py
      metric_validation.py
      identity_metric.py
    data/
      __init__.py
      cyclic_sprites_datamodule.py
    models/
      __init__.py
      original.py                 # Canonical RiemannianFlowVAE (final KL/posterior)
      modular.py                  # Canonical ModularRiemannianFlowVAE
      components/
        __init__.py
        encoder_manager.py
        decoder_manager.py
        metric_tensor.py
        metric_loader.py
        flow_manager.py
        loss_manager.py
        manifold_sampler.py
        native_inverse_metric.py
        working_riemannian_sampler.py
        official_rhvae_sampler.py
    training/
      __init__.py
      lightning_trainer.py
      training_utils.py
    losses/
      __init__.py
      kl_loss.py                  # optional: if still separate, else folded into loss_manager
      reconstruction_loss.py      # optional
      riemannian_loss.py          # optional
      flow_loss.py                # optional
    evaluation/
      __init__.py
      fid_computation.py
      evaluator.py
      generation_methods.py
    visualization/
      __init__.py
      visualization_manager.py
      basic_visualizations.py
      manifold_visualizations.py

conf/
  config.yaml                    # globals, defaults
  experiment/
    global_vanilla_rlvae_pipeline.yaml
    single_run.yaml
    comparison_study.yaml
    three_stage_pipeline.yaml
  model/
    original.yaml                # _target_: rlvae.models.original.RiemannianFlowVAE
    modular.yaml                 # _target_: rlvae.models.modular.ModularRiemannianFlowVAE
  data/
    cyclic_sprites.yaml
  training/
    default.yaml
    full_data.yaml
    quick.yaml
  evaluation/
    default.yaml
    minimal.yaml
  sampling/
    rhmc_default.yaml
  metric/
    rhvae.yaml
    precision.yaml
  sweep/
    ...
  visualization/
    minimal.yaml
    standard.yaml
    full.yaml

scripts/
  run_experiment.py              # main CLI entry
  tools/
    freeze_experiment.py
    convert_checkpoints.py

examples/
  demo_visualizations_clean.py   # imports rlvae.models.*, no gradients
  notebooks/
    kl_divergence_analysis.ipynb

tests/
  unit/
    test_metric_apis_sanity.py
    test_kl_identity_metric.py
    test_kl_scale_invariance.py
    test_posterior_alpha_scaling.py
  integration/
    test_lightning_forward_smoke.py

docs/
  CLEAN_REPO_RESTRUCTURE_PLAN.md (this file)
  GLOBAL_PIPELINE_GUIDE.md
  GLOBAL_RLVAE_PIPELINE.md
  guides/
    CLEAN_TRAINING_GUIDE.md
    RHMC_Manifold_Demo.md
  archive/
    legacy_docs.md

data/                           # (empty, .gitignore; user supplies processed data)
outputs/                        # (gitignored; Hydra/Lightning outputs)
experiments/
  frozen/
    2025MMDD_<tag>/             # frozen config + artifact references
```

## Canonical Classes and Routing

- `rlvae.models.original.RiemannianFlowVAE` (canonical “original”)
  - Finalized posterior: local Gaussian Σ = α G(μ)
  - Finalized KL: 0.5 E[(z−μ)^T G̃(z) (z−μ)] with metric normalization and fp32
  - Returns keys: `reconstruction`, `reconstruction_loss`, `kl_divergence`, `flow_loss`, `loop_penalty`, `total_loss`, `latent_samples`
- `rlvae.models.modular.ModularRiemannianFlowVAE` (canonical “modular”)
  - Extends `original`, delegates loss to `loss_manager` with the same keys
  - Components in `rlvae.models.components.*`
- Samplers
  - Training posterior: Always metric‑aligned Gaussian (differentiable)
  - Prior RHMC: For exploration/visuals only; logs separately

## Hydra Config Standardization

Shared keys across models:
- Posterior: `posterior_local_alpha`, `posterior_alpha_ramp_enabled`, `posterior_alpha_start`, `posterior_alpha_end`, `posterior_alpha_ramp_epochs`
- KL: `kl_use_metric_normalization`, `kl_metric_norm_mode={geomean|trace|none}`, `kl_amp_safe`
- Flows: `n_flows`, `flow_hidden_size`, `flow_n_blocks`, `flow_n_hidden`
- Trainer routes: loop mode/penalty, phase flags if needed

Targets:
- `conf/model/original.yaml` → `_target_: rlvae.models.original.RiemannianFlowVAE`
- `conf/model/modular.yaml` → `_target_: rlvae.models.modular.ModularRiemannianFlowVAE`

## Migration Plan (Phased, Safe)

1) Adapters (current step)
- Keep wrappers at canonical paths that import from legacy locations.
- Update critical configs and trainer to read canonical paths (done for `rhvae_original_with_metric_update.yaml`).

2) Copy / Move (code)
- Move `src/models/modular_rlvae.py` → `src/rlvae/models/modular.py` and fix imports to `rlvae.*`.
- Move `src/models/components/*` → `src/rlvae/models/components/*`; update imports across repo (modular, trainer, losses, viz, eval).
- Move Lightning trainer to `src/rlvae/training/lightning_trainer.py` and update imports (or keep path but re‑export from `rlvae.training`).
- Move utils (reproducibility, metric_validation, identity_metric) to `src/rlvae/utils/`.

3) Replace wrappers
- Legacy `src/models/*` import from `rlvae.models.*` with deprecation warnings.
- Legacy `original_rlvae/src/models/riemannian_flow_vae.py` becomes a thin import of `rlvae.models.original` (or archived).

4) Config migration
- Add `conf/model/original.yaml`, `conf/model/modular.yaml` with standardized flags.
- Point all experiments to these models. Replace references everywhere else.

5) Docs and examples
- Update docs to reflect new tree; archive legacy docs.
- Move `scripts/demo_visualizations_clean.py` to `examples/` and ensure it imports from `rlvae.models.*` and uses `posterior_local_alpha` small default.

6) Tests and CI
- Add unit tests for:
  - Identity metric KL ≈ 0.5 α d (±10%)
  - Geomean normalization scale invariance
  - α scaling linearity with identity metric
  - Posterior sampling μ–z distance controlled by α
- Add integration smoke: Lightning forward on tiny batch; asserts presence of `total_loss` and no NaN.

7) Deletion / Archive
- Once green, remove legacy trees (`src/models/riemannian_flow_vae.py`, heavy `original_rlvae/...`) or move to `docs/archive/`.

## Exhaustive File Mapping (Key Files)

- Models
  - original_rlvae/src/models/riemannian_flow_vae.py → src/rlvae/models/original.py
  - src/models/modular_rlvae.py → src/rlvae/models/modular.py
  - src/models/components/* → src/rlvae/models/components/*
- Training
  - src/training/lightning_trainer.py → src/rlvae/training/lightning_trainer.py
  - src/training/training_utils.py → src/rlvae/training/training_utils.py
- Data
  - src/data/cyclic_sprites_datamodule.py → src/rlvae/data/cyclic_sprites_datamodule.py
- Losses
  - src/losses/* → src/rlvae/losses/* (or fold into loss_manager)
- Evaluation
  - src/evaluation/* → src/rlvae/evaluation/*
- Visualization
  - src/visualization/* → src/rlvae/visualization/*
- Utils
  - src/utils/reproducibility.py → src/rlvae/utils/reproducibility.py
  - src/utils/metric_validation.py → src/rlvae/utils/metric_validation.py
  - src/utils/identity_metric.py → src/rlvae/utils/identity_metric.py
- Scripts
  - run_experiment.py stays at repo root (imports from `rlvae.*`)
  - scripts/demo_visualizations_clean.py → examples/demo_visualizations_clean.py
- Config
  - conf/model/rhvae_original_with_metric_update.yaml → conf/model/original.yaml (canonical)
  - conf/model/mlp_rlvae.yaml → conf/model/modular.yaml (canonical)

## Compatibility & Routing

- Trainer already accepts canonical and legacy targets; prefer canonical.
- For intermediate period, add modules:
  - `src/models/riemannian_flow_vae.py` → `from rlvae.models.original import RiemannianFlowVAE`
  - `original_rlvae/src/models/riemannian_flow_vae.py` → same import or archived
- Keep both until tests and pipelines pass, then remove legacy.

## Validation Checklist

- [ ] Hydra pipeline runs with `model=original` and `model=modular` targets
- [ ] Demos import from `rlvae.models.*` and run with α=0.001 without warnings
- [ ] `total_loss` present in model outputs across both models
- [ ] Identity metric KL test passes
- [ ] Geomean normalization invariance test passes
- [ ] α scaling test passes
- [ ] Lightning smoke test (CPU) passes in CI

## Suggestions & Best Practices

- Keep config keys uniform across models; avoid per‑model conditionals in trainer.
- Centralize posterior sampling/metric normalization logic to prevent drift.
- Prefer adapters over copy‑paste when bridging old and new code.
- Archive, don’t delete, docs/analyses; add short pointers to current implementations.
- Add pre‑commit (ruff/black/isort) with minimal rules to avoid style drift.

---

This document is the source of truth for the restructuring work. Once Phase 2 starts, we will PR changes in logical chunks (models → components → training → utils → tests), ensuring the Hydra pipeline remains runnable throughout.

