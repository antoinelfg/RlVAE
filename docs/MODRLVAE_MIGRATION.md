ModRLVAE Migration and Usage
============================

This guide shows how to switch between the original monolithic RiemannianFlowVAE
and the fully modular ModRLVAE using Hydra configs, and how to enable Stage C
metric alternation.

Model selection
---------------

Original (monolith):

```
model:
  _target_: rlvae.models.riemannian_flow_vae.RiemannianFlowVAE
  input_dim: [3, 64, 64]
  latent_dim: 8
  n_flows: 7
  posterior_type: riemannian_metric
  beta: 1.0
  riemannian_beta: 1.0
  loop:
    mode: open
    penalty: 0.0
  # posterior / KL options
  kl_use_metric_normalization: true
  kl_metric_norm_mode: geomean
  posterior_local_alpha: 0.5
  use_curvature_correction: true
```

Fully modular (ModRLVAE):

```
model:
  _target_: rlvae.models.modrlvae.ModRLVAE
  input_dim: [3, 64, 64]
  latent_dim: 8
  sequence_length: 8         # auto sets n_flows = sequence_length - 1
  posterior_type: riemannian_metric
  beta: 1.0
  riemannian_beta: 1.0
  loop:
    mode: open
    penalty: 0.0

  # Enc/Dec default to MLP
  encoder:
    architecture: mlp
  decoder:
    architecture: mlp

  # Flows (IAF)
  flow_hidden_size: 64
  flow_n_blocks: 2
  flow_n_hidden: 1

  # Metric
  metric:
    trainable: false
    temperature_override: 0.1
    regularization_override: 0.01
  pretrained:
    metric_path: /path/to/metric.pt    # optional

  # Posterior / KL options
  kl_use_metric_normalization: true
  kl_metric_norm_mode: geomean
  posterior_local_alpha: 0.5
  use_curvature_correction: true       # or kl_metric_eval_point: 'z' | 'mu'

  # Regularizers and EMA (optional)
  phase1_training: true
  centroid_regularizer_enabled: true
  centroid_regularizer_weight: 0.01
  centroid_regularizer_t0_only: true

  phase2_training: true
  spectral_penalty_enabled: true
  spectral_penalty_weight: 0.1
  eigenval_min_bound: 1e-2
  eigenval_max_bound: 1e2
  smoothness_penalty_enabled: true
  smoothness_penalty_weight: 0.01
  anisotropy_alignment_enabled: true
  anisotropy_alignment_weight: 0.05

  centroid_ema_enabled: true
  centroid_ema_rate: 0.01
  centroid_ema_update_frequency: 10
```

Stage C (metric alternation) via plugin
---------------------------------------

```
training:
  metric_alternation:
    enabled: true
    warmup_epochs: 5
    k_rlvae_epochs: 3
    metric_step_epochs: 1
    logdet_clip: 50.0
    anchor_size: 20000
    anchor_refresh_frac: 0.1
    consistency_weight: 0.0
```

Run examples
------------

Original model:

```
python run_experiment.py model._target_=rlvae.models.riemannian_flow_vae.RiemannianFlowVAE \
  model.input_dim=[3,64,64] model.latent_dim=8 model.n_flows=7 \
  model.posterior_type=riemannian_metric model.beta=1.0 model.riemannian_beta=1.0
```

ModRLVAE:

```
python run_experiment.py model._target_=rlvae.models.modrlvae.ModRLVAE \
  model.input_dim=[3,64,64] model.latent_dim=8 model.sequence_length=8 \
  model.posterior_type=riemannian_metric model.beta=1.0 model.riemannian_beta=1.0 \
  training.metric_alternation.enabled=true
```

Notes
-----
- ModRLVAE mirrors the monolith’s math but is fully componentized (encoder/decoder/flows/metric/posterior/loss/regularizers).
- Regularizer terms are logged in Lightning under train_/val_/test_ prefixes: metric_reg, centroid_regularizer, spectral_penalty, smoothness_penalty, anisotropy_penalty.
- Curvature correction is controlled via `use_curvature_correction` or `kl_metric_eval_point`.

