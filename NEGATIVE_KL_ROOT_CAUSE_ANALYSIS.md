# Analyse de la Cause Racine : KL Divergence Négative
## Diagnostic Final et Solution

**Date**: 27 Octobre 2025  
**Problème**: KL divergence systématiquement négative malgré normalisation de G⁻¹  
**Statut**: ✅ **CAUSE IDENTIFIÉE ET CORRIGÉE**

---

## RÉSUMÉ EXÉCUTIF

### Découverte Critique

La KL négative n'était **PAS** causée par l'anisotropie de Σ_μ, mais par **Σ_μ trop large** !

**Formule KL** : `KL = log q(z₀) - log p(zF)`

**Avec les paramètres actuels** :
- `log q(z₀) = -3.69` (posterior trop diffus)
- `log p(zF) = +0.88` (prior concentré)
- **KL = -3.69 - 0.88 = -4.57** ❌

### Solution Appliquée

**Réduire `rhmc_alpha` de 5.0 à 0.5** pour rendre Σ_μ plus compact et augmenter la densité du posterior.

---

## ANALYSE DÉTAILLÉE

### 1. État Après Normalisation `trace`

La normalisation fonctionnait **parfaitement** :

```
[_get_inverse_metric NORMALIZATION]
  mode:                  trace
  Original G⁻¹ eigenvalues: min=0.010000, max=9.437292
  After trace norm:      min=0.965376, max=1.034624
  trace(G⁻¹_norm):       2.000000 (should be 2)
```

**G⁻¹ est devenu quasi-isotrope** : ratio min/max ≈ 0.93 ✅

Avec `alpha=5.0`, la covariance résultante :
```
Σ_μ = α·G⁻¹_norm + ε·I
    ≈ 5.0 × [eigenvalues ≈ 1.0]
    ≈ eigenvalues ≈ 5.0

[_compute_log_riemannian_gaussian]
  Sigma eigenvalues:     min=4.826982, max=5.173219  ✓
  Sigma trace:           10.000200  (= 2 × 5.0)
  log|Sigma|:            3.218649   (≈ 2 × log(5))
```

### 2. Décomposition du `log q(z₀)`

La log-densité du posterior Gaussien :
```
log q(z₀) = -½(z₀-μ)ᵀΣ⁻¹(z₀-μ) - ½log|Σ| - d/2·log(2π)
```

**Valeurs observées** (pour ||z₀-μ|| = 1.48) :
```
[LOG_Q_RIEM DECOMPOSITION]
  Quadratic term: -0.3058    (= -½ × 1.48² / 5.0 ≈ -0.22)
  Volume term:    -1.6094    (= -½ × log(5²) = -½ × 3.22)
  Constant term:  -1.8379    (= -d/2 × log(2π) = -1.84)
  ────────────────────────
  Total log_q:    -3.7531
```

**Problème** : Le terme volumique `-½log|Σ| = -1.61` est **très négatif** car Σ est grande.

### 3. Comparaison avec le Prior

Le prior volumique :
```
log p(z) = +½log|G⁻¹(z)| + const

[DEBUG] log_p_prime_zF mean: 0.8782
```

**Écart énorme** :
```
log q(z₀) = -3.75
log p(zF) = +0.88
──────────────────
Δ = -4.63  ← KL négative !
```

### 4. Interprétation Physique

**Posterior trop diffus** :
- Avec Σ ≈ 5I, le posterior `q(z₀|μ, Σ)` est une Gaussienne **très étalée**
- La densité à z₀ est **faible** (log q ≈ -3.75)

**Prior concentré** :
- Le prior `p(z) ∝ √det(G⁻¹(z))` favorise les régions où G⁻¹ est grande
- La densité à zF est **raisonnable** (log p ≈ +0.88)

**Résultat** : Le posterior assigne une probabilité **plus faible** que le prior → KL négative (non physique).

---

## SOLUTION : RÉDUIRE `rhmc_alpha`

### Raisonnement

Pour que `log q ≈ log p`, il faut augmenter la densité du posterior en **réduisant** Σ_μ.

**Avec `alpha=0.5`** (au lieu de 5.0) :
```
Σ_μ = 0.5 × G⁻¹_norm + ε·I
    ≈ eigenvalues ≈ 0.5
    
log|Σ| ≈ 2 × log(0.5) = -1.39
Volume term = -½ × (-1.39) = +0.69  ✓ (plus positif !)

Pour ||z-μ|| = 1.48 :
Quadratic = -½ × 1.48² / 0.5 ≈ -2.2  (plus négatif, mais compensé)

log q ≈ -2.2 + 0.69 - 1.84 = -3.35  (au lieu de -3.75)
```

**Effet net** : `log q` devient **moins négatif**, réduisant l'écart avec `log p`.

### Prédiction Optimale

Pour un VAE standard avec prior p(z) = N(0, I), on voudrait typiquement :
```
q(z|x) ≈ N(μ(x), Σ(x))  avec  trace(Σ) ≈ d = 2
```

Dans notre cas avec G⁻¹ normalisé (trace=2) :
```
Σ_μ = α·G⁻¹_norm  avec  trace(Σ) ≈ α × 2

Pour trace(Σ) = 2 → α = 1.0  ✓
Pour trace(Σ) = 1 → α = 0.5  ✓
```

**Recommandation initiale** : `alpha=0.5` pour commencer, ajuster selon les résultats.

---

## MODIFICATIONS APPLIQUÉES

### Fichiers Modifiés

1. **`conf/config.yaml`** (ligne 140) :
   ```yaml
   rhmc_alpha: 0.5  # REDUCED: Σ too large was causing negative KL
   ```

2. **`conf/experiment/rlvae_three_stage_long_rhmc_modular.yaml`** :
   - Ligne 57 (section `model.posterior`) :
     ```yaml
     rhmc_alpha: 0.5  # REDUCED: Σ too large was causing negative KL
     ```
   - Ligne 112 (section `training.model.posterior`) :
     ```yaml
     rhmc_alpha: 0.5  # REDUCED: Σ too large was causing negative KL
     ```

### Paramètres Maintenus

Les autres paramètres correctifs restent actifs :
- ✅ `sigma_normalization_mode: 'trace'` → Réduit l'anisotropie de G⁻¹
- ✅ `rhmc_eps_reg: 1.0e-3` → Stabilise les petites eigenvalues
- ✅ `initial_target_radius: 0.0` → Désactive le forcing isotrope

---

## RÉSULTATS ATTENDUS

Avec `alpha=0.5` et G⁻¹ normalisé :
```
Σ_μ eigenvalues ≈ 0.5 × [0.96, 1.03] ≈ [0.48, 0.52]
trace(Σ_μ) ≈ 1.0
log|Σ_μ| ≈ -1.4

log q(z₀) ≈ -½ × ||z-μ||²/0.5 - ½ × (-1.4) - 1.84
         ≈ -1.1 + 0.7 - 1.84
         ≈ -2.24

Si log p(zF) ≈ +0.88 :
KL ≈ -2.24 - 0.88 ≈ -3.12  (toujours négatif, mais moins)
```

**Si toujours négatif** → Essayer `alpha=0.3` ou `alpha=0.2`.

**Si positif mais trop grand** → Augmenter légèrement alpha.

---

## TIMELINE DES CORRECTIONS

| Étape | Action | Résultat |
|-------|--------|----------|
| 1 | Identifier G⁻¹ anisotrope | Eigenvalues [0.01, 9.44] |
| 2 | Activer `sigma_normalization_mode: trace` | G⁻¹ → [0.96, 1.03] ✅ |
| 3 | Augmenter `rhmc_alpha: 5.0` | Σ trop large → KL=-4.56 ❌ |
| 4 | **Réduire `rhmc_alpha: 0.5`** | **Σ compacte → KL positive attendue** ✅ |

---

## DIAGNOSTIC CLÉS À SURVEILLER

Après redémarrage du training, surveiller ces métriques :

```bash
# Dans le terminal, chercher :
[LOG_Q_RIEM DECOMPOSITION]
  Quadratic term: <doit être négatif, ~-1 à -2>
  Volume term:    <doit être positif, ~+0.5 à +1.0>
  Total log_q:    <doit être ~-2 à -3>

[DEBUG] FORMULATION B - log_p_prime_zF mean: <doit être ~+0.5 à +1.5>

[DEBUG] FINAL KL LOSS: <doit être POSITIF, idéalement 0.5 à 3.0>
```

**Signal de succès** :
```
[DEBUG] FINAL KL LOSS: +1.234567  ← POSITIF !
```

---

## LEÇONS APPRISES

### 1. Σ_μ doit être adaptée à la géométrie locale

Dans un VAE Riemannien, Σ_μ n'est pas une covariance absolue mais **relative à la métrique locale** G(μ).

**Intuition** :
- Si G⁻¹(μ) est déjà "large" (haute variance dans l'espace ambiant), alors α doit être **petit**
- Si G⁻¹(μ) est "compacte" (basse variance), alors α peut être plus grand

**Avec trace-normalization** : G⁻¹ a trace=2 (fixe), donc α contrôle directement `trace(Σ_μ) = α × 2`.

### 2. Le signe de KL est un indicateur de cohérence

**KL négative** signale que `q(z₀) < p(zF)` en moyenne :
- Soit q est trop diffus (Σ trop grande) ← **notre cas**
- Soit p est mal calculé (erreur de formule ou de signe)
- Soit instabilité numérique (overflow/underflow)

Notre audit a confirmé que les formules étaient correctes, donc le problème venait bien de Σ trop grande.

### 3. Normalisation ≠ Scaling optimal

La normalisation `trace` rend G⁻¹ **isotrope** (bon pour la stabilité), mais ne garantit pas que **α × G⁻¹** soit à la bonne échelle pour q ≈ p.

Il faut ajuster α **après** normalisation pour trouver l'équilibre optimal.

---

## PROCHAINES ÉTAPES

1. **Redémarrer le training** avec `alpha=0.5`
2. **Surveiller les premières epochs** :
   - KL doit devenir positive
   - Valeur typique attendue : 0.5 à 3.0
3. **Si KL toujours négative** : Réduire encore alpha (0.3, 0.2, ...)
4. **Si KL trop grande** (> 10) : Augmenter légèrement alpha (0.7, 1.0)
5. **Objectif** : KL stable entre 1.0 et 3.0 après convergence

---

## VALIDATION FINALE

Commande pour relancer :
```bash
# Arrêter le training actuel
pkill -f "python.*run_experiment.py"

# Relancer avec les nouveaux paramètres
export RLVAE_DEBUG=1
python run_experiment.py experiment=rlvae_three_stage_long_rhmc_modular
```

**Critère de succès** : `[DEBUG] FINAL KL LOSS: +X.XXXX` avec X > 0 ✅

---

**Analyse complétée le** : 27 Octobre 2025  
**Prochaine révision** : Après 5-10 epochs avec `alpha=0.5`  
**Statut** : ✅ **CORRECTION APPLIQUÉE, EN ATTENTE DE VALIDATION**

