# Corrections Appliquées à la Configuration

**Date**: 28 Octobre 2025  
**Basé sur**: Diagnostics Deep KL (Chi² déviation +2000-2300%)

---

## 🔴 Problèmes Identifiés (Mise à Jour)

### 1. **RHMC DÉSACTIVÉ** ❌
```yaml
# AVANT:
rhmc_steps: 0  # ← RHMC ne faisait RIEN !
```

**Symptôme**: Drift = 0.0, aucun mouvement de z0 → zK

### 1bis. **RHMC BLOQUÉ (Découvert Après Premier Fix)** ❌
Même après activation (`rhmc_steps=4`):
```yaml
# AVANT:
rhmc_step_size: 0.02          # Trop petit
max_position_step: 0.1        # Trop restrictif
max_velocity_norm: 0.5        # Trop restrictif
```

**Symptômes**:
- RHMC exécute 4 steps mais drift = 0.0000 (aucun mouvement!)
- Scaling combiné: 0.02 × (0.1/v_norm) × sqrt(min_eig(G⁻¹)) ≈ 0.0002
- Mouvement effectif arrondi à 0.0000 dans les logs

### 2. **Multi-Try et Volume Acceptance Biaisent la Distribution** ❌
- Par défaut: `initial_n_candidates: 8` (non spécifié = 8)
- Sélection du meilleur échantillon basée sur `log|G⁻¹(z)|`
- **Résultat**: Distribution ≠ N(μ, Σ_μ)

**Symptômes**:
- Mahalanobis² = 38-47 au lieu de 2.0 (+2000%)
- Distance ratio = 0.5 (z0 trop proche de μ)
- log_q = -20 à -24 (devrait être -3 à -5)

### 3. **Paramètres Sous-Optimaux**
- `rhmc_alpha: 0.1` trop petit
- Features instables activées (kinetic_grad, projection)

---

## ✅ Corrections Appliquées

### Fichier Modifié: `conf/config.yaml`

#### 1. **Activation de RHMC + Augmentation Step Parameters**
```yaml
# Lines 138-142
rhmc_steps: 4                      # FIX: ENABLE RHMC (was 0!)
rhmc_step_size: 0.10               # FIX: INCREASED from 0.02 (combiné avec scaling factors)
rhmc_alpha: 0.5                    # FIX: Increased for better Σ_μ scale
rhmc_eps_reg: 1.0e-4               # FIX: Standard regularization
min_cov_eig: 1.0e-3                # FIX: Minimum eigenvalue for Cholesky

# Lines 155-157
max_velocity_norm: 2.0             # FIX: INCREASED from 0.5 (4×)
max_position_step: 1.0             # FIX: INCREASED from 0.1 (10×)
```

**Justification**: Le step effectif était `0.02 × (0.1/v_norm) × sqrt(min_eig(G⁻¹)) ≈ 0.0002`, maintenant `0.10 × (1.0/v_norm) × sqrt(0.016) ≈ 0.013` (65× plus grand!)

#### 2. **Désactivation Multi-Try et Volume Acceptance** (CRITIQUE)
```yaml
# Lines 145-147
initial_n_candidates: 1            # FIX: No multi-try sampling (was 8 by default)
initial_volume_tolerance: 0.0      # FIX: No volume acceptance
initial_max_retries: 0             # FIX: No retries
```

**Impact Attendu**: 
- Sampling exactement de N(μ, Σ_μ)
- Mahalanobis² devrait passer de 40+ à ~2.0 ✓
- KL devrait devenir positive ✓

#### 3. **Désactivation Features Instables**
```yaml
# Lines 151-154
kinetic_grad_enabled: false        # FIX: Disable for stability
projection_step_scale: 0.0         # FIX: Disable projection
initial_max_norm: 0.0              # FIX: Disable radial capping
```

#### 4. **Override Stage C** (Cohérence)
```yaml
# Lines 267-276
stage_c:
  posterior:
    rhmc_alpha: 0.5                # FIX: Override with diagnostic-based value
    rhmc_steps: 4                  # FIX: Ensure RHMC is active
    rhmc_step_size: 0.10           # FIX: Match increased step size
    max_position_step: 1.0         # FIX: Match increased position limit
    max_velocity_norm: 2.0         # FIX: Match increased velocity limit
    initial_n_candidates: 1        # FIX: Disable multi-try
    initial_volume_tolerance: 0.0  # FIX: Disable volume acceptance
    initial_max_retries: 0         # FIX: Disable retries
```

---

## 🧪 Test de Validation

### Commande de Test

```bash
# Test avec diagnostics complets
RLVAE_DEBUG=1 python run_experiment.py \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=2 \
  > test_fix_$(date +%Y%m%d_%H%M%S).log 2>&1
```

### Métriques à Vérifier

#### ✅ Succès si:

1. **RHMC Actif ET Bouge**:
   ```
   [TRAJECTORY SUMMARY]
     Total drift from z0:   >0.01 (CRITIQUE: devait être 0.0000 avant!)
     Net change in ||·-μ||: ±0.XX (pas 0.0000)
   
   [STEP k=1], [STEP k=2], [STEP k=3], [STEP k=4]
     # Devrait voir les 4 steps maintenant
   ```
   
   **Note**: Avec les nouveaux paramètres, le drift devrait être ~0.01-0.05 au lieu de 0.0000

2. **Mahalanobis² Normalisé**:
   ```
   [CHI-SQUARED TEST]
     Expected Mahal²:       2.0
     Observed Mahal²:       ~2.0 (±40%, i.e. 1.2-2.8)
     Deviation:             <±100% (au lieu de +2000%)
   ```

3. **Distance Ratio Normal**:
   ```
   [EXPECTED VS ACTUAL]
     Ratio (actual/expected): ~1.0 (±30%, i.e. 0.7-1.3)
   ```

4. **log_q Raisonnable**:
   ```
   [STANDARD DECOMPOSITION]
     Quadratic term: -1 à -3 (au lieu de -20 à -24)
     Total log_q:    -3 à -5 (au lieu de -20 à -24)
   ```

5. **KL POSITIVE**:
   ```
   [DEBUG] FINAL KL LOSS: +X.XXXX  ← DOIT ÊTRE POSITIF !
   ```

#### ⚠️ Si Toujours Négatif:

Essayer des ajustements progressifs:

```yaml
# Option A: Réduire alpha si Σ_μ toujours trop grand
posterior:
  rhmc_alpha: 0.2  # ou 0.3

# Option B: Augmenter alpha si Σ_μ trop petit
posterior:
  rhmc_alpha: 1.0  # ou 2.0
```

---

## 📊 Comparaison Avant/Après

| Métrique | Avant (Problème) | Après Fix 1 | Après Fix 2 (Final) | Status |
|----------|------------------|-------------|---------------------|--------|
| rhmc_steps | 0 | 4 | 4 | ✅ |
| rhmc_step_size | 0.02 (default) | 0.02 | 0.10 | ✅ |
| max_position_step | 0.1 | 0.1 | 1.0 | ✅ |
| max_velocity_norm | 0.5 | 0.5 | 2.0 | ✅ |
| initial_n_candidates | 8 (défaut) | 1 | 1 | ✅ |
| Effective step size | ~0.0002 | ~0.0002 | ~0.013 (65×) | ✅ |
| RHMC drift | 0.0 | 0.0 | **>0.01** | ⏳ À tester |
| Mahalanobis² | 38-47 | ?? | ~2.0 | ⏳ À tester |
| Distance ratio | 0.5 | ?? | ~1.0 | ⏳ À tester |
| log_q | -20 à -24 | ?? | -3 à -5 | ⏳ À tester |
| KL divergence | -22 à -25 | **-2 à -3** | **POSITIVE** | ⏳ À tester |

---

## 🎯 Prochaines Étapes

### 1. Test Immédiat (5 min)
```bash
RLVAE_DEBUG=1 python run_experiment.py \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=2
```

### 2. Validation Rapide
Chercher dans les logs:
```bash
grep "FINAL KL LOSS" test_fix_*.log
# Doit montrer: [DEBUG] FINAL KL LOSS: +X.XXXX
```

### 3. Si Succès → Training Complet
```bash
# Désactiver debug pour vitesse
python run_experiment.py
```

### 4. Si Échec → Ajustement Alpha
Voir section "⚠️ Si Toujours Négatif" ci-dessus

---

## 📝 Notes Techniques

### Pourquoi initial_n_candidates: 1 ?

**Avant** (multi-try avec K=8):
1. Génère 8 échantillons: z₁, z₂, ..., z₈ ~ N(μ, Σ_μ)
2. Évalue log|G⁻¹(zᵢ)| pour chaque échantillon
3. Sélectionne z₀ = argmax log|G⁻¹(zᵢ)|
4. **Résultat**: z₀ ne suit PLUS N(μ, Σ_μ) !

**Après** (single sample avec K=1):
1. Génère 1 échantillon: z₀ ~ N(μ, Σ_μ)
2. Pas de sélection
3. **Résultat**: z₀ suit exactement N(μ, Σ_μ) ✓

### Pourquoi initial_volume_tolerance: 0.0 ?

**Avant**:
- Rejette les échantillons avec log|G⁻¹(z)| < log|G⁻¹(μ)| - tol
- **Biais**: Favorise les régions à haute densité volumique

**Après**:
- Pas de rejet
- **Résultat**: Distribution non biaisée ✓

### Impact sur la Performance

- **Avec K=8**: Exploration potentiellement meilleure, MAIS KL incorrecte
- **Avec K=1**: KL correcte, exploration via RHMC (maintenant actif avec 4 steps)

---

## 🔬 Validation Scientifique

### Test Théorique

Pour z ~ N(μ, Σ), on doit avoir:
1. (z-μ)ᵀΣ⁻¹(z-μ) ~ χ²(D)
2. E[||z-μ||²] = tr(Σ)
3. log q(z) = -½(z-μ)ᵀΣ⁻¹(z-μ) - ½log|Σ| - const

**Avec les anciennes settings**:
- ❌ Mahal² = 40 au lieu de 2 (χ²(2))
- ❌ ||z-μ|| = 0.5×√(tr(Σ))
- ❌ log q = -20 (trop négatif)

**Avec les nouvelles settings** (attendu):
- ✅ Mahal² ≈ 2 (±40%)
- ✅ ||z-μ|| ≈ √(tr(Σ)) (±30%)
- ✅ log q ≈ -3 à -5 (raisonnable pour D=2)

---

## 📚 Références

- **Diagnostic complet**: `docs/DEEP_KL_DIAGNOSTICS_IMPLEMENTATION.md`
- **Système de diagnostic**: `DIAGNOSTIC_SYSTEM_SUMMARY.md`
- **Scripts de diagnostic**: `scripts/diagnose_negative_kl.py`
- **Visualisations**: `scripts/visualize_kl_diagnostics.py`

---

**Configuration mise à jour**: ✅ `conf/config.yaml`  
**Prêt pour test**: ✅ Oui  
**Résultat attendu**: KL divergence POSITIVE  

---

**Bonne chance pour le test !** 🚀

Si KL devient positive → Succès ! 🎉  
Si KL reste négative → Ajuster `rhmc_alpha` (0.2, 0.3, 1.0, 2.0) et re-tester.

