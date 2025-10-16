# Config Refactor TODO

Cette liste référence les endroits du code qui continuent de modifier la
configuration Hydra (assignations directes, `OmegaConf.set_struct`, appels à
`hydra.compose`, etc.), afin de les migrer vers la nouvelle structure
`settings.*` sans mutations dynamiques.

## run_experiment.py

- ✅ Étape préliminaire : la config Hydra est désormais clonée lors de
  l'initialisation (`self.original_config` vs `self.config`). Stage C travaille
  sur une copie dédiée qui est restaurée ensuite.
- Stage A / pipelines (autour des lignes 340–420) :
  - ✅ Stage A s'exécute désormais sur une copie locale (restaurée avant Stage B).
  - ✅ `hydra.compose` remplacé par des presets internes — réduire la logique à un
    helper configurable si besoin.
  - Assignations `self.config.model.sequence_length`, `n_flows`, `input_dim`,
    `latent_dim`, modifications de `visualization`, `evaluation`, etc.
  - Patches finaux qui resynchronisent `model`/`training.model`.
- Stage B / Stage C orchestration (lignes ~950+, 1300+, 1900+, 2500+) :
  - ✅ Stage B et C utilisent chacun un clone local + `_set_config_value` pour RHMC, KL, n_flows, evaluation/visualization.
  - Reste : quelques traces d'accès direct (`print` de debug, valeurs stage-specific) à simplifier ou supprimer.
- Section pipeline final (lignes ~3000+) : vérifier qu'il ne reste pas de manip directes type `DictConfig(...)` / `setattr`.

## config/synchronizer.py
- ✅ Simplifié : renvoie désormais une simple copie détachée et avertit que le
  module est déprécié (plus de mutations/OmegaConf.set_struct).

## config/validator.py
- ✅ Validation réécrite pour travailler sur une copie python (plus de
  `set_struct`) et ajout de `validate_model_settings` qui consomme directement
  `settings.*` via `build_model_config_from_settings`.

## training/lightning_trainer.py et backup
- ✅ Le Lightning trainer consomme désormais directement `settings.*` via
  `build_model_config_from_settings`; plus aucune mutation du `DictConfig`
  original.
- ⚠️ Le fichier de sauvegarde `lightning_trainer_backup.py` reste en mode legacy
  (à archiver ou migrer plus tard si nécessaire).

## src/rlvae/... (modular_rlvae, components)
- ✅ `modular_rlvae.py` travaille sur une copie locale et n'ajuste plus la
  config transmise (auto `n_flows` géré sans mutation). Les managers n'ont pas
  besoin de changements immédiats.

## Scripts (scripts/*.py)
- ✅ Les anciens scripts Hydra ont été retirés. Utiliser
  `scripts/run_workflow.py` pour orchestrer les workflows (stages isolés,
  sampling, sweeps) et `run_experiment.py` pour les overrides fins.

## Étapes suivantes
1. Finaliser la conversion des derniers patches Stage C (logs et prints restants) si nécessaire.
2. Créer au besoin des notebooks ou utilitaires spécifiques pour remplacer les
   diagnostics retirés (KL sandbox, probes, etc.).

Ce document sert de checklist ; nous le mettrons à jour au fur et à mesure que
les modules sont convertis.
