# 🔧 Fix Appliqué: KL Divergence Négative

**Date**: 28 Octobre 2025  
**Status**: ✅ Prêt à tester

---

## 📋 Résumé (Mise à Jour - Second Round)

Vos diagnostics ont identifié **3 problèmes critiques**:

1. ❌ **RHMC désactivé** (`rhmc_steps: 0`)
2. ❌ **Multi-try sampling biaise la distribution** (Mahal² = 40 au lieu de 2.0)
3. ❌ **RHMC bloqué par scaling factors** (drift = 0.0000 même avec 4 steps!) ← **NOUVEAU**

**Solution**: Configuration corrigée dans `conf/config.yaml`
- **Premier round**: Activation RHMC + désactivation multi-try
- **Second round**: Augmentation step size (0.02 → 0.10) + limits (5-10×)

---

## 🚀 Test Rapide (5 minutes)

### Option A: Script Automatique
```bash
./test_config_fix.sh
```

### Option B: Manuel
```bash
RLVAE_DEBUG=1 python run_experiment.py \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=2
```

---

## ✅ Critères de Succès

Cherchez dans les logs:

### 1. KL Divergence POSITIVE
```
[DEBUG] FINAL KL LOSS: +X.XXXX  ← Doit être POSITIF !
```

### 2. RHMC Actif ET Bouge (CRITIQUE - Fix Second Round)
```
[TRAJECTORY SUMMARY]
  Total drift from z0:   >0.01  ← NOUVEAU: Doit être >0.01 (était 0.0000!)
  Net change in ||·-μ||: ±0.XX

# Devrait voir 4 steps:
[STEP k=1]
[STEP k=2]
[STEP k=3]
[STEP k=4]
```

**Avant**: drift = 0.0000 (bloqué par scaling)  
**Après**: drift > 0.01 (step size augmenté 5×, limits 4-10×)

### 3. Mahalanobis² Normalisé
```
[CHI-SQUARED TEST]
  Observed Mahal²:       ~2.0  ← Pas 40+ !
  Deviation:             <100%  ← Pas 2000% !
```

### 4. Distance Ratio OK
```
[EXPECTED VS ACTUAL]
  Ratio (actual/expected): ~1.0  ← Pas 0.5 !
```

---

## 📝 Changements Appliqués

### Fichier: `conf/config.yaml`

#### ✅ RHMC Activé
```yaml
rhmc_steps: 4              # Était: 0
rhmc_step_size: 0.02       # Était: 0.008
rhmc_alpha: 0.5            # Était: 0.1
```

#### ✅ Multi-Try Désactivé
```yaml
initial_n_candidates: 1         # Nouveau (était 8 par défaut)
initial_volume_tolerance: 0.0   # Nouveau (désactive volume acceptance)
initial_max_retries: 0          # Nouveau
```

#### ✅ Features Instables Désactivées
```yaml
kinetic_grad_enabled: false     # Était: true
projection_step_scale: 0.0      # Était: 0.05
initial_max_norm: 0.0           # Était: 1.5
```

---

## 🎯 Que Faire Après ?

### Si ✅ KL Positive → Training Complet
```bash
# Désactiver debug pour vitesse
python run_experiment.py
```

### Si ⚠️ KL Toujours Négative → Ajuster Alpha

Modifier dans `conf/config.yaml`:

```yaml
# Si distance ratio >1.5 (z0 trop loin)
posterior:
  rhmc_alpha: 1.0  # ou 2.0

# Si distance ratio <0.5 (z0 trop proche)
posterior:
  rhmc_alpha: 0.2  # ou 0.3
```

Puis re-tester.

---

## 📚 Documentation Complète

- **Modifications détaillées**: `CONFIG_FIX_APPLIED.md`
- **Système de diagnostic**: `DIAGNOSTIC_SYSTEM_SUMMARY.md`
- **Guide complet**: `docs/DEEP_KL_DIAGNOSTICS_IMPLEMENTATION.md`

---

## 🔬 Pourquoi Ça Va Marcher

### Problème Avant
- Multi-try sélectionnait le "meilleur" parmi 8 échantillons
- → Distribution ≠ N(μ, Σ_μ)
- → log_q calculé avec N(μ, Σ_μ) mais z0 ne vient pas de là
- → KL = garbage

### Solution Après
- Un seul échantillon, pas de sélection
- → Distribution = exactement N(μ, Σ_μ)
- → log_q correctement calculé
- → KL correcte et positive ✓

---

## ❓ Questions Fréquentes

**Q: Pourquoi `initial_n_candidates: 1` ?**  
R: Pour éviter le biais de sélection. Avec K=8, on garde le "meilleur" échantillon selon log|G⁻¹|, ce qui change la distribution.

**Q: Est-ce que j'explore moins bien l'espace avec K=1 ?**  
R: Non, RHMC (maintenant actif avec 4 steps) s'occupe de l'exploration.

**Q: Que faire si KL reste négative ?**  
R: Ajuster `rhmc_alpha` selon la distance ratio (voir section ci-dessus).

---

**Prêt à tester ?** Lancez `./test_config_fix.sh` ! 🚀

