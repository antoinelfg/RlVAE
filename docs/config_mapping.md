# Configuration Mapping Guide

Ce document fait le lien entre l’ancienne arborescence Hydra (multi-fichiers) et
le méga-config unique proposé dans `conf/config.yaml`. Chaque entrée indique les
chemins Hydra historiques (à gauche) et la nouvelle clef centralisée (à droite).

## 1. Métadonnées & sortie globale

| Ancien chemin                                    | Nouveau chemin                          |
|--------------------------------------------------|----------------------------------------|
| `config.project_name`                            | `global.project_name`                   |
| `config.experiment_name`                         | `global.experiment_name`                |
| `config.description`                             | `global.description`                    |
| `config.seed`                                     | `global.seed`                           |
| `config.device`                                   | `global.device`                         |
| `config.output_dir`                               | `output.base_dir`                       |
| `config.wandb.*`                                  | `wandb.*`                               |
| `checkpoint/default.yaml::*`                      | `checkpoint.*` / `pipeline.outputs.*`   |
| `config.hydra.run/sweep.*`                        | `hydra.run.*` / `hydra.sweep.*`         |

## 2. Données

| Ancien chemin (fichier)                                      | Nouveau chemin                                |
|-------------------------------------------------------------|----------------------------------------------|
| `data/<dataset>.yaml: name`                                 | `data.dataset` (identifiant)                 |
| `data/<dataset>.yaml: channels`                             | `data.common.channels`                       |
| `data/<dataset>.yaml: image_size`                           | `data.common.image_size`                     |
| `data/<dataset>.yaml: sequence_length / seq_len`            | `data.common.sequence_length`                |
| `data/<dataset>.yaml: batch_size`                           | `data.common.batch_size`                     |
| `data/<dataset>.yaml: num_workers`                          | `data.common.num_workers`                    |
| `data/<dataset>.yaml: pin_memory`                           | `data.common.pin_memory`                     |
| `data/<dataset>.yaml: shuffle`                              | `data.common.shuffle`                        |
| `data/<dataset>.yaml: train/val/test_ratio`                 | `data.common.splits.train|val|test`          |
| `ellipse_sequences*.yaml: min/max_radius`                   | `data.ellipse_sequences.min_radius` / `max_radius` |
| `ellipse_sequences*.yaml: min/max_eccentricity`             | `data.ellipse_sequences.min_eccentricity` / `max_eccentricity` |
| `ellipse_sequences*.yaml: fix_center/theta/intensity`       | `data.ellipse_sequences.fix_center` …        |
| `ellipse_sequences*.yaml: keep_major_axis_constant`         | `data.ellipse_sequences.keep_major_axis_constant` |
| `ellipse_sequences*.yaml: keep_area_constant`               | `data.ellipse_sequences.keep_area_constant`  |
| `ellipse_sequences*.yaml: center_jitter`                    | `data.ellipse_sequences.center_jitter`       |
| `ellipse_sequences*.yaml: supersample_factor`               | `data.ellipse_sequences.supersample_factor`  |
| `ellipse_sequences*.yaml: outline_only/outline_width`       | `data.ellipse_sequences.outline_only` / `outline_width` |
| `ellipse_sequences*.yaml: antialias`                        | `data.ellipse_sequences.antialias`           |
| `ellipse_sequences_sinusoidal*.yaml: sinusoidal_*`          | `data.ellipse_sequences.sinusoidal_*`        |
| `colored_circles.yaml | cyclic_sprites.yaml: max_*_samples` | `data.colored_circles.max_train_samples` etc. |

## 3. Modèle

| Ancien chemin                                             | Nouveau chemin                                       |
|-----------------------------------------------------------|------------------------------------------------------|
| `model/*.yaml: _target_`                                  | `model.target` (string)                             |
| `model/*.yaml: latent_dim`                                | `model.latent_dim`                                   |
| `model/*.yaml: n_flows`                                   | `model.n_flows`                                      |
| `model/*.yaml: epsilon`                                   | `model.epsilon`                                      |
| `model/*.yaml: encoder.architecture`                      | `model.encoder.architecture`                         |
| `model/*.yaml: encoder.*`                                 | `model.encoder.*` (hidden_layers, dropout, etc.)     |
| `model/*.yaml: decoder.*`                                 | `model.decoder.*`                                    |
| `model/*.yaml: flow_*`                                    | `model.flows.*`                                      |
| `model/*.yaml: beta`                                      | `model.losses.beta`                                  |
| `model/*.yaml: riemannian_beta`                           | `model.losses.riemannian_beta`                       |
| `model/*.yaml: loop.penalty`                              | `model.losses.loop_penalty_weight`                   |
| `model/*.yaml: mu_l2_weight`                              | `model.losses.mu_l2_weight`                          |
| `model/*.yaml: kl_prior_mode / kl_metric_eval_point`      | `model.losses.kl_prior_mode` / `model.losses.kl_metric_eval_point` |
| `riemannian_rhmc_vae.yaml: posterior.*`                   | `model.posterior.*`                                  |
| `riemannian_rhmc_vae.yaml: rhmc_* duplications`           | `model.posterior.rhmc_*`                             |
| `riemannian_rhmc_vae.yaml: rhmc_kl_*`                     | `model.posterior.rhmc_kl_*`                          |
| `riemannian_flow_vae.yaml: sampling.method`               | `model.sampling.method`                             |
| `model/*.yaml: metric.*`                                  | `model.metric.*`                                     |
| `model/*.yaml: loop.mode`                                 | `model.loop.mode`                                    |
| `model/*.yaml: pretrained.paths`                           | `model.pretrained.*`                                 |

## 4. Entraînement

| Ancien chemin                                          | Nouveau chemin                                      |
|--------------------------------------------------------|-----------------------------------------------------|
| `training/*.yaml: trainer.*`                           | `training.strategy.*`                              |
| `training/*.yaml: optimizer.*`                         | `training.optimizer.*`                             |
| `training/*.yaml: scheduler.*`                         | `training.scheduler.*`                             |
| `training/*.yaml: early_stopping.*`                    | `training.early_stopping.*`                        |
| `training/*.yaml: logging.*`                           | `training.logging.*`                               |
| `training/full_data.yaml: n_train_samples`             | `training.n_train_samples` (optionnel si requis)   |
| `training/full_data.yaml: n_val_samples`               | `training.n_val_samples`                           |
| `training/*: data.batch_size / splits`                 | `data.common.*` (absorbés par le bloc données)     |
| `training/*: visualization.*`                          | `visualization.*`                                   |
| `training/stable_training.yaml: enhanced_kl_monitoring`| `training.monitoring.enhanced_kl.*` (si conservé)   |

### Overrides par stage

| Ancien chemin (expériences Stage)                       | Nouveau chemin                                       |
|---------------------------------------------------------|------------------------------------------------------|
| `experiment/...: stage_a.epochs`                        | `training.stage_overrides.stage_a.epochs`            |
| `experiment/...: stage_a.lr`                            | `training.stage_overrides.stage_a.lr`                |
| `experiment/...: stage_b.* (metric RHVAE)`              | `training.stage_overrides.stage_b.*`                 |
| `experiment/...: stage_c.lr / riemannian_beta / ...`    | `training.stage_overrides.stage_c.*`                 |

## 5. Pipeline & orchestration

| Ancien chemin                                       | Nouveau chemin                      |
|-----------------------------------------------------|------------------------------------|
| `experiment/...: type`                              | `pipeline.mode` (ex: `three_stage`) |
| `experiment/...: run_stage_a/b/c/sampling`          | `pipeline.run_stage_a` etc.         |
| `experiment/...: run_sampling`                      | `pipeline.run_sampling`             |
| `checkpoint/default.yaml: stage*_dir`               | `pipeline.outputs.stage_*_dir`      |
| `experiment/...: sampling.* (stage-level)`          | `pipeline.*` ou `sampling.*` selon usage |
| `experiment/...: checkpoint.metric_file`            | `pipeline.outputs.metric_file`      |
| `experiment/...: resume / load_*`                   | `pipeline.resume_from.*`            |

## 6. Évaluation & Visualisation

| Ancien chemin                                  | Nouveau chemin                       |
|------------------------------------------------|--------------------------------------|
| `evaluation/*.yaml: enabled`                   | `evaluation.enabled`                 |
| `evaluation/*.yaml: frequency`                 | `evaluation.frequency`               |
| `evaluation/*.yaml: fid.*`                     | `evaluation.fid.*`                   |
| `evaluation/*.yaml: generation.*`              | `evaluation.generation.*`            |
| `evaluation/*.yaml: inference.*`               | `evaluation.inference.*`             |
| `evaluation/*.yaml: benchmarking.*`            | `evaluation.benchmarking.*`          |
| `visualization/*.yaml: level`                  | `visualization.level`                |
| `visualization/*.yaml: enable_*`               | `visualization.enable_*`             |
| `visualization/*.yaml: frequency/basic_frequency/...` | `visualization.*` (frequences spécifiques) |
| `visualization/ellipse_sequences_optimized: ellipse_specific.*` | `visualization.ellipse_specific.*` |

## 7. Sampling / génération

| Ancien chemin                                | Nouveau chemin              |
|----------------------------------------------|-----------------------------|
| `sampling/rhmc_default.yaml:*`               | `sampling.*`                 |
| `experiment/...: sampling.enabled`           | `sampling.enabled` (global) |
| `generation/*` (si présents)                 | `generation.*`               |

## 8. Checkpoints

| Ancien chemin                                  | Nouveau chemin                  |
|------------------------------------------------|----------------------------------|
| `checkpoint/default.yaml: save/load`           | `checkpoint.save` / `checkpoint.load` |
| `checkpoint/default.yaml: dir`                 | `checkpoint.dir`                  |
| `checkpoint/default.yaml: metric_file`         | `checkpoint.metric_file` (alias `pipeline.outputs.metric_file`) |
| `checkpoint/default.yaml: rhmc_samples`        | `checkpoint.rhmc_samples` (alias `pipeline.outputs.rhmc_samples_file`) |

## 9. Debug / instrumentation

| Ancien chemin    | Nouveau chemin       |
|------------------|----------------------|
| (variables env, flags dispersés) | `debug.*` (ex: `debug.verbose`, `debug.profiler`) |

## Remarques sur les doublons

- Les valeurs dupliquées entre `model.*` et les expériences (ex. `riemannian_beta`) sont
  désormais définies dans `model.losses.*` et, si nécessaire, surchargées via
  `training.stage_overrides.stage_c.*`.
- Les overrides Hydra de la CLI (anciennement `experiment=...`, `model=...`) se font
  désormais via des chemins directs (`model.posterior.rhmc_steps=8`, etc.).
- Les scripts stage-only ou sweep devront pointer sur les booleens de `pipeline.*` et
  les sections `training.stage_overrides.*` plutôt que charger des fichiers séparés.

N’hésite pas à compléter ce tableau si tu retrouves d’autres clefs dispersées (par ex.
des scripts spécifiques dans `scripts/` qui introduisent des paramètres additionnels).
