## RLVAE three-stage pipeline — quickstart, usage, and current status

This guide explains how to run the three-stage Riemannian Latent VAE (RLVAE) pipeline, what’s implemented, and the current status.

### What the pipeline does
- **Stage A (metric learning)**: Train an RHVAE (or base VAE) on the full dataset and export its latent metric tensors `C` (centroids) and `M` (inverse metric blocks). Also saves `t=0` latents for Stage B filtering.
- **Stage B (t=0 metric + visuals)**: Filter the Stage‑A metric to the `t=0` distribution and generate PCA(2) visuals:
  - det(G^{-1}) heatmap in PCA subspace
  - centroids plot (PCA(2))
  - RHMC volume‑element samples overlaid (actual samples; not centroids)
  - concise table summary (sampler, metric, acceptance)
- **Stage C (RLVAE training)**: Train the Flow‑VAE on top of the fixed t=0 metric. Posterior sampling uses the volume‑element RHMC. All logs go to a single W&B run, namespaced `stageA/`, `stageB/`, `stageC/`.

### Requirements and notes
- Python 3.10+; CUDA GPU recommended.
- Weights & Biases login required (`wandb login`).
- Dataset expected at `data/processed/Sprites_*_cyclic.pt`.
- Recommended env var during heavy visuals/sampling:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## How to run
All commands run from the repo root.

Hydra tip: Use `key=value` to override existing keys. Use `+key=value` (or `++key=value`) only for keys that don’t already exist. If you see “Could not append to config… item is already at …”, remove the `+`.

### 1) Full three‑stage run (RHVAE Stage A → Stage B → Stage C)
Example with 10 epochs for Stage A and Stage C, and standard visuals:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u run_experiment.py \
  experiment=rlvae_three_stage_pipeline \
  experiment.stage_a.model=rhvae \
  experiment.stage_a.epochs=10 \
  visualization.level=standard visualization.enable_manifold=true \
  +evaluation.fid.compute_fid=false \
  wandb.group=three_stage_single_run
```
Notes:
- FID is disabled due to a known config mismatch.
- Stage A uses RHVAE via `experiment.stage_a.model=rhvae`.

### 2) Stage B only (recompute visuals from saved metric)
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u run_experiment.py \
  experiment=rlvae_three_stage_pipeline \
  experiment.run_stage_a=false experiment.run_stage_b=true experiment.run_stage_c=false \
  sampling.enabled=true \
  visualization.level=standard visualization.enable_manifold=true \
  wandb.group=stageB_clean_only
```
This will:
- Load `outputs/metrics/metric_t{temperature}.pt` (created by Stage A)
- Filter centroids to `t=0` if `t0_latents` are available
- Log:
  - `stageB/pca/rhmc_overlay_t0`
  - `stageB/pca/det_heatmap_t0`
  - `stageB/pca/centroids_t0`
  - `stageB/summary_table`

### 3) Stage B → Stage C (short run, lower KL pressure)
Lower KL weights first to stabilize:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u run_experiment.py \
  experiment=rlvae_three_stage_pipeline \
  experiment.run_stage_a=false experiment.run_stage_b=true experiment.run_stage_c=true \
  experiment.stage_c.epochs=10 \
  sampling.enabled=true \
  visualization.level=standard visualization.enable_manifold=true \
  +evaluation.fid.compute_fid=false \
  model.beta=0.1 model.riemannian_beta=0.5 \
  wandb.group=three_stage_BC_lowbeta
```
If KL still dominates, try:
```bash
model.beta=0.0 model.riemannian_beta=0.1
```
Posterior sampler in Stage C is RHMC volume‑element by default.

### 4) Stage A RHVAE hyperparameters (optional)
You can pass RHVAE‑specific settings through `model.*`. Use `+` only for new keys:
```bash
# Example: latent_dim, RHMC steps, temperature, etc.
python -u run_experiment.py \
  experiment=rlvae_three_stage_pipeline \
  experiment.stage_a.model=rhvae \
  experiment.stage_a.epochs=10 \
  model.latent_dim=10 \
  +model.n_lf=15 +model.eps_lf=0.00002 \
  model.temperature=0.7 model.regularization=0.00012 \
  +model.weight_kernel=mahalanobis_normed +model.weight_metric_normalization=trace \
  model.normalize_weight_sum=true +model.topk_weights=10 \
  +model.align_with_knn_cov=true +model.knn_k=300 +model.alpha_align=0.6 \
  +model.metric_normalization=none +model.reestimate_metric_from_decoder_jacobian=true \
  +model.jacobian_alpha=0.5 +model.jacobian_h=0.001 +model.jacobian_stride=4 \
  visualization.level=standard visualization.enable_manifold=true \
  +evaluation.fid.compute_fid=false \
  wandb.group=three_stage_single_run
```

---

## Implemented highlights
- **Unified volume‑element RHMC sampler** across visuals and training:
  - Stage A visuals (RHVAE), Stage B overlays, Stage C posterior sampling all use the same sampler.
- **Stage B PCA‑aligned visuals are accurate:**
  - Heatmap computes `det(G^{-1})` in PCA(2) subspace using the t=0‑filtered centroids; overlays use true RHMC samples.
  - Acceptance rate is logged; visuals align with centroids and samples.
- **Unified W&B run:** a single run for the whole pipeline; logs are namespaced `stageA/`, `stageB/`, `stageC/`.
- **Stage A metric payload:** centroids, metric matrices, `t0_latents`, and encoder path. Recon vs real logged.
- **Stage C integration:** loads Stage A metric; sets `model.latent_dim` from the metric; posterior type `riem_hmc`.

---

## Current behavior and known gaps
- **KL term in Stage C (current):** simplified Riemannian posterior KL:
  - `KL ≈ 0.5 * E[(z_0 − μ)^T G(z_0) (z_0 − μ)]`.
  - If RHMC explores broadly or the metric has large eigenvalues, KL can remain high.
- **Full ELBO “Case 2”** (sample‑wise `log q − log p` with flow Jacobians) is planned but not yet wired.
- **Evaluation config:**
  - FID disabled via `+evaluation.fid.compute_fid=false` due to a struct mismatch.
  - Missing `evaluation.sampler_types` at test end is benign for training; we’ll patch the config.

---

## Practical tips and troubleshooting
- **Hydra overrides:**
  - Existing keys: `model.beta=0.1` (no `+`).
  - New keys: `+model.n_lf=15` or `++model.n_lf=15` if Hydra complains.
- **High KL / non‑convergence:**
  - Lower `model.beta` and `model.riemannian_beta` (e.g., `0.0` and `0.1`) for a warm‑up.
  - Reduce Stage‑C posterior RHMC intensity (smaller `n_lf`, smaller `eps_lf`) when exposed via config.
  - Optionally use the “narrow q” approximation `KL ≈ 0.5 μ^T G(μ) μ` for a short warm‑up (planned option).
- **OOM during visuals/sampling:**
  - Chunking and float32 are used already; keep the allocator env var set.
- **Metric/latent_dim mismatch:** auto‑set `config.model.latent_dim` from the Stage A metric before Stage C.

---

## Outputs and artifacts
- Metric checkpoint (Stage A/Stage B): `outputs/metrics/metric_t{temperature}.pt`.
- RHMC samples (Stage B): `outputs/metrics/rhmc_samples_t0.pt`.
- Stage directories: `outputs/stageA`, `outputs/stageB`, `outputs/stageC`.
- W&B media panels:
  - `stageA/recon_vs_real`
  - `stageB/pca/rhmc_overlay_t0`, `stageB/pca/det_heatmap_t0`, `stageB/pca/centroids_t0`, `stageB/summary_table`, `stageB/recon_vs_real`
  - `stageC/*` (training curves; recon grids). A raw (non‑normalized) recon vs real for Stage C is planned.

---

## Roadmap (short‑term)
- Add `riemannian_kl_mode` with options:
  - `quadratic` (current), `sample_logq_logp` (sample‑wise `log q − log p`), `narrow_approx` (0.5 μ^T G(μ) μ).
- Expose Stage‑C posterior HMC params in Hydra (e.g., `model.posterior.hmc.n_lf`, `eps_lf`).
- Add `stageC/recon_vs_real_raw` (non‑normalized side‑by‑side images).
- Patch evaluation config to avoid `sampler_types`/`compute_fid` warnings.

---

## Repro quick‑refs
- Stage B only (clean visuals):
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u run_experiment.py \
  experiment=rlvae_three_stage_pipeline \
  experiment.run_stage_a=false experiment.run_stage_b=true experiment.run_stage_c=false \
  sampling.enabled=true \
  visualization.level=standard visualization.enable_manifold=true \
  wandb.group=stageB_clean_only
```
- Stage B → C (low beta):
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u run_experiment.py \
  experiment=rlvae_three_stage_pipeline \
  experiment.run_stage_a=false experiment.run_stage_b=true experiment.run_stage_c=true \
  experiment.stage_c.epochs=10 \
  sampling.enabled=true \
  visualization.level=standard visualization.enable_manifold=true \
  +evaluation.fid.compute_fid=false \
  model.beta=0.0 model.riemannian_beta=0.1 \
  wandb.group=three_stage_BC_lowbeta
```
