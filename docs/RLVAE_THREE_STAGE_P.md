## RLVAE three‑stage pipeline: options, parameters, visuals, and commands

This guide summarizes how to run the new three‑stage pipeline (A: warm VAE, B: metric at t=0, C: full RLVAE), what you can configure via Hydra, what gets visualized/logged, and ready‑to‑use command recipes.

### High‑level scheme

```mermaid
flowchart LR
  subgraph StageA[Stage A — Warm VAE]
    A1[Train vanilla VAE
        - time‑conditioned
        - architecture, d, β]
    A2[Save encoder/decoder]
  end

  subgraph StageB[Stage B — Metric at t=0]
    B1[Encode t=0 → z0]
    B2[Centroids c_i (k‑medoids/FPS/balanced)]
    B3[L_i init (RHVAE/precision)]
    B4[G0^{-1}(z;ψ) build + analyze]
    B5[Save metric artifact]
  end

  subgraph OptionalSampling[Optional RHMC sampling]
    S1[Sample from p_Riem(z0;ψ)
        via RHMC]
  end

  subgraph StageC[Stage C — Full RLVAE]
    C1[Posterior: amortized RHMC]
    C2[Generative: flows g_t]
    C3[Decoder p(x_t|z_t,t)]
    C4[Alternating schedule
       k epochs RLVAE ↔ 1 epoch metric]
  end

  A1 --> A2 --> B1 --> B2 --> B3 --> B4 --> B5 --> C1
  B5 -.-> S1
```

### Where things live
- Pipeline config: `conf/experiment/rlvae_three_stage_pipeline.yaml`
- Global config: `conf/config.yaml`
- Training defaults: `conf/training/default.yaml`
- Lightning trainer (alternating schedule, anchors, viz): `src/training/lightning_trainer.py`
- Stage A/B script (vanilla VAE + diverse metric): `scripts/train_diverse_metric_vae.py`

### Key Hydra parameters (override with command‑line `key=value`)

- Global trainer/data
  - `training.trainer.accelerator`: `gpu|cpu` (default from training config)
  - `training.trainer.devices`: e.g., `1`
  - `training.trainer.precision`: `16-mixed|bf16-mixed|32`
  - `training.data.batch_size`: batch size for Stage C dataloaders
  - `training.optimizer.lr`, `training.optimizer.weight_decay`
  - `training.logging.stage_c_metric_viz_every_n_epochs`: Stage C metric image frequency (epochs)
  - `training.logging.metric_snapshot_every_n_epochs`: Stage C metric weight snapshot frequency
  - `visualization.level`: `none|minimal|standard|full`
  - `visualization.frequency`: create viz every N epochs (if enabled)
  - `evaluation.enabled`: `true|false` (FID/gen evaluation)

- Stage A (warm VAE)
  - `experiment.stage_a.epochs`: number of epochs
  - `experiment.stage_a.batch_size`, `experiment.stage_a.lr`, `experiment.stage_a.beta`
  - `experiment.stage_a.architecture`: e.g., `mlp` (inherits from `model.encoder.architecture`)
  - `experiment.stage_a.latent_dim`: inherits from `model.latent_dim`

- Stage B (metric on t=0)
  - `experiment.stage_b.implementation`: `rhvae|precision`
  - `experiment.stage_b.n_centroids`: e.g., `8|16|32|64|128`
  - `experiment.stage_b.temperature`: kernel T (can override RHVAE metric T)
  - `experiment.stage_b.regularization`: ridge λ, etc.
  - `experiment.stage_b.centroid_method`: `kmedoids|fps|balanced`
  - `experiment.stage_b.neighbor_mode`: `global|local`, `experiment.stage_b.knn_k`
  - `experiment.stage_b.use_timestep`: `0` (t=0 only)
  - `experiment.stage_b.standardize_latents`: `true|false`

- Optional RHMC sampling after Stage B
  - `experiment.sampling.enabled`: `true|false`
  - `experiment.sampling.n_steps`, `experiment.sampling.n_leapfrog`, `experiment.sampling.step_size`

- Stage C (RLVAE with alternating schedule)
  - `experiment.stage_c.epochs`, `experiment.stage_c.batch_size`
  - `experiment.stage_c.riemannian_beta`: weight for Riemannian KL
  - `experiment.stage_c.allow_metric_updates`: `true|false` (enable metric training in Stage C)
  - Alternation knobs (`training.metric_alternation.*`)
    - `enabled`: `true|false`
    - `warmup_epochs`: freeze ψ first N epochs
    - `k_rlvae_epochs`: epochs updating θ, φ, flows with ψ frozen
    - `metric_step_epochs`: epochs updating ψ only
    - `anchor_size`: size of anchor pool |A|
    - `anchor_refresh_frac`: fraction of anchors refreshed each metric epoch
    - `logdet_clip`: clamp for `log det G^{-1}` stability
    - `consistency_weight`: optional pushforward‑consistency penalty

- WandB
  - `wandb.project`: default `rlvae-three-stage-visuals`
  - `wandb.group`: e.g., `three_stage_visuals`
  - `wandb.mode`: `online|offline|disabled`
  - `wandb.pipeline_mode`: `per_stage|single_run` (per‑stage runs + summary vs. single aggregation)

### What gets visualized/logged
- Stage A
  - Recon grids, training/validation losses (per run or summary)
  - Saved encoder/decoder artifacts
- Stage B
  - Metric analysis: eigenvalue histograms, heatmaps, centroid stats, `√det G^{-1}` maps
  - Saved metric artifact (e.g., `outputs/metrics/metric_t0.7.pt`), logged to WandB as `metric` artifact with alias `${wandb.artifacts.aliases.stage_b_latest}`
- Stage C
  - Training/validation/test: total/recon/KL (+ optional Riemannian terms)
  - Flow diagnostics (periodic)
  - Metric tensor snapshot and eigen‑stats every `training.logging.stage_c_metric_viz_every_n_epochs` epochs
  - Latest trained metric snapshot logged as `metric` artifact with alias `${wandb.artifacts.aliases.stage_c_latest}`
  - Optional FID/generation evaluation when `evaluation.enabled: true`

### Command recipes

- Fast CPU smoke test (no WandB, minimal epochs, no viz/eval)

```bash
python run_experiment.py \
  experiment=rlvae_three_stage_pipeline \
  training=default evaluation=disabled visualization=none \
  training.trainer.accelerator=cpu training.trainer.devices=1 \
  training.data.batch_size=4 \
  experiment.stage_a.epochs=1 \
  experiment.stage_b.implementation=rhvae experiment.stage_b.n_centroids=8 \
  sampling.enabled=false \
  experiment.stage_c.epochs=1 \
  training.logging.stage_c_metric_viz_every_n_epochs=999 \
  training.logging.metric_snapshot_every_n_epochs=999 \
  wandb.mode=disabled
```

- Moderate GPU run (per‑stage logging, minimal viz)

```bash
python run_experiment.py \
  experiment=rlvae_three_stage_pipeline \
  training=default evaluation=default visualization=minimal \
  training.trainer.accelerator=gpu training.trainer.devices=1 training.trainer.precision=16-mixed \
  training.data.batch_size=32 \
  experiment.stage_a.epochs=2 experiment.stage_a.batch_size=32 \
  experiment.stage_b.implementation=rhvae experiment.stage_b.n_centroids=16 \
  sampling.enabled=false \
  experiment.stage_c.epochs=3 experiment.stage_c.batch_size=32 \
  training.metric_alternation.enabled=true training.metric_alternation.warmup_epochs=2 \
  training.metric_alternation.k_rlvae_epochs=2 training.metric_alternation.metric_step_epochs=1 \
  training.logging.stage_c_metric_viz_every_n_epochs=5 training.logging.metric_snapshot_every_n_epochs=5 \
  wandb.mode=online +wandb.group=three_stage_visuals
```

- Heavier Stage C with alternating schedule (GPU)

```bash
python run_experiment.py \
  experiment=rlvae_three_stage_pipeline \
  training=default evaluation=default visualization=minimal \
  training.trainer.accelerator=gpu training.trainer.devices=1 training.trainer.precision=16-mixed \
  experiment.stage_a.epochs=5 \
  experiment.stage_b.implementation=rhvae experiment.stage_b.n_centroids=32 \
  sampling.enabled=false \
  experiment.stage_c.epochs=20 \
  training.metric_alternation.enabled=true training.metric_alternation.warmup_epochs=5 \
  training.metric_alternation.k_rlvae_epochs=3 training.metric_alternation.metric_step_epochs=1 \
  training.logging.stage_c_metric_viz_every_n_epochs=5 training.logging.metric_snapshot_every_n_epochs=5 \
  wandb.mode=online +wandb.group=three_stage_visuals
```

### Outputs and checkpoints
- Stage A: `data/pretrained/encoder_*.pt`, `decoder_*.pt` and a combined model file
- Stage B: `outputs/metrics/metric_*.pt` (DIVERSE metric) and analysis figures
- Stage C: metric snapshots under `metric_snapshots/` and images logged to WandB per frequency
- Hydra outputs under `outputs/${experiment_name}/…`

### Tips and safeguards
- Enable alternating schedule for Stage C geometry updates: `training.metric_alternation.enabled=true`
- Use smaller LR for metric via `training.optimizer.metric_lr_scale` (handled internally by the trainer)
- Tune `n_centroids` and `temperature` (T) carefully; start with `K=32..128`, set `T≈median_nn_dist/√d`
- For speed, set `visualization.level=none` and disable evaluation during sweeps
- Keep WandB organized with `wandb.group` and (optionally) a final summary run
- For MLOps readiness, enable artifacts (`wandb.artifacts.enabled=true`) so you can resume from Stage A/B/C by pulling artifacts via `wandb.use_artifact`

### Artifact-based resume recipes (MLOps)

- Resume Stage C from Stage B metric and Stage A encoder/decoder artifacts

```bash
python run_experiment.py \
  experiment=rlvae_three_stage_pipeline \
  training=default evaluation=default visualization=minimal \
  training.trainer.accelerator=gpu training.trainer.devices=1 training.trainer.precision=16-mixed \
  experiment.run_stage_a=false experiment.run_stage_b=false \
  pretrained.encoder_path=$(python - <<'PY'
import wandb, os
run=wandb.init(project=os.getenv('WANDB_PROJECT','rlvae-three-stage-visuals'), mode='offline')
art=wandb.use_artifact('stageA_vae_mlp_ld16:stageA_latest', type='model')
dir=art.download()
print(os.path.join(dir, 'encoder_diverse_mlp_ld16.pt'))
PY
) \
  pretrained.decoder_path=$(python - <<'PY'
import wandb, os
run=wandb.init(project=os.getenv('WANDB_PROJECT','rlvae-three-stage-visuals'), mode='offline')
art=wandb.use_artifact('stageA_vae_mlp_ld16:stageA_latest', type='model')
dir=art.download()
print(os.path.join(dir, 'decoder_diverse_mlp_ld16.pt'))
PY
) \
  checkpoint.metric_dir=outputs/metrics \
  checkpoint.metric_file=metric_t0.7.pt
```

Notes:
- The runner references `stageB_metric_*:${wandb.artifacts.aliases.stage_b_latest}` automatically when artifacts are enabled, so manual download is often unnecessary.


