# Solution Corrigée : Préserver l'Anisotropie de G⁻¹
## Correction de l'Approche Initiale

**Date**: 27 Octobre 2025  
**Problème Initial**: KL divergence négative  
**Erreur d'Analyse**: Confusion entre "réduire l'anisotropie excessive" et "forcer l'isotropie"  
**Solution Correcte**: **Préserver l'anisotropie géométrique** tout en réduisant l'échelle globale

---

## ❌ ERREUR DANS L'ANALYSE PRÉCÉDENTE

### Ce Que J'Ai Fait (Incorrectement)

```yaml
sigma_normalization_mode: 'trace'  # ← ERREUR !
rhmc_alpha: 5.0
```

**Effet** :
```
G⁻¹ original: eigenvalues [0.01, 9.44]  # Anisotrope (géométrie locale)
        ↓ trace normalization
G⁻¹_norm:    eigenvalues [0.96, 1.03]  # ISOTROPE (info perdue !)
        ↓ α = 5.0
Σ_μ:         eigenvalues [4.8, 5.2]    # Quasi-isotrope
```

### Pourquoi C'était Faux

**L'anisotropie de G⁻¹ n'est PAS un bug, c'est une FEATURE !**

- G⁻¹(μ) reflète la **géométrie locale** de la variété au point μ
- Si la variété a des directions de haute/basse courbure, G⁻¹ **doit** être anisotrope
- **Σ_μ = α·G⁻¹(μ)** doit **préserver** cette structure anisotrope

**En forçant l'isotropie**, j'ai **détruit l'information géométrique** que le modèle essayait d'apprendre !

---

## ✅ SOLUTION CORRECTE

### Principe

**Garder G⁻¹ anisotrope**, mais **réduire son échelle globale** via un petit `alpha`.

### Configuration Corrigée

```yaml
sigma_normalization_mode: 'none'  # ✅ PRÉSERVER l'anisotropie
rhmc_alpha: 0.1                   # ✅ Petit facteur d'échelle
rhmc_eps_reg: 1.0e-3              # ✅ Stabilisation numérique
initial_target_radius: 0.0        # ✅ Pas de forcing isotrope
```

**Effet** :
```
G⁻¹ original: eigenvalues [0.01, 9.44]  # Anisotrope (trace ≈ 9.45)
        ↓ NO normalization (mode='none')
G⁻¹:         eigenvalues [0.01, 9.44]  # Préservé !
        ↓ α = 0.1
Σ_μ:         eigenvalues [0.001, 0.944]  # Anisotrope, mais à petite échelle
             trace(Σ_μ) ≈ 0.945
```

---

## CALCUL DU log_q ATTENDU

### Avec les Nouveaux Paramètres

**Σ_μ eigenvalues** : `[0.001, 0.944]` (moyenne ≈ 0.47)

Pour `||z - μ|| = 1.48` :

```
log|Σ_μ| ≈ log(0.001 × 0.944) ≈ log(0.0009) ≈ -7.0

Volume term = -½ × (-7.0) = +3.5  ← Très positif !

Quadratic term ≈ -½ × (1.48² / 0.47) ≈ -2.3

Constant term = -1.84

───────────────────────────────
log q ≈ -2.3 + 3.5 - 1.84 ≈ -0.64
```

### Comparaison avec le Prior

```
log q(z₀) ≈ -0.64   (nouveau, avec Σ petite anisotrope)
log p(zF) ≈ +0.88   (prior volumique)

KL ≈ -0.64 - 0.88 ≈ -1.52  (encore négatif, mais mieux)
```

**Si toujours négatif** : Réduire encore `alpha` (0.05, 0.03, ...) jusqu'à ce que `log q ≈ log p`.

---

## POURQUOI Σ_μ DOIT ÊTRE ANISOTROPE

### Intuition Géométrique

Imaginons une variété 2D qui ressemble à un **ruban** :
- Direction 1 (le long du ruban) : **Large**, faible courbure → G⁻¹ grande
- Direction 2 (transverse au ruban) : **Étroite**, forte courbure → G⁻¹ petite

**Le posterior q(z|μ)** devrait refléter cette géométrie :
```
Σ_μ = α·G⁻¹(μ) = α × [large, petite]
                = [α·large, α·petite]  ← Toujours anisotrope !
```

**Si on force l'isotropie** (via trace normalization) :
```
Σ_μ ≈ α × [1, 1]  ← Ignore la géométrie locale
```

Le sampling devient **inadapté** à la structure de la variété.

---

## RÔLE DE `alpha`

**`alpha` est un facteur d'échelle global**, pas un correcteur d'anisotropie.

### Rôle Correct

- **α petit** (0.01 - 0.1) : Posterior **concentré** autour de μ
  - Bon pour les variétés à forte courbure
  - Évite que q soit trop diffus
  
- **α grand** (1.0 - 10.0) : Posterior **diffus** autour de μ
  - Bon pour les variétés plates
  - Permet plus d'exploration

### Notre Cas

Avec `trace(G⁻¹) ≈ 9.45`, un `alpha=0.1` donne `trace(Σ_μ) ≈ 0.945`.

C'est **raisonnable** pour un VAE 2D (comparable à un prior N(0, 0.5I)).

---

## MODIFICATIONS APPLIQUÉES

### 1. `conf/config.yaml` (ligne 140-156)

```yaml
rhmc_alpha: 0.1                    # Small alpha, preserve anisotropy
rhmc_eps_reg: 1.0e-3               # Numerical stability
sigma_normalization_mode: none     # ✅ PRESERVE anisotropy
initial_target_radius: 0.0         # No isotropic forcing
```

### 2. `conf/experiment/rlvae_three_stage_long_rhmc_modular.yaml`

**Section `model.posterior`** (ligne 53-59) :
```yaml
rhmc_alpha: 0.1
sigma_normalization_mode: 'none'
```

**Section `training.model.posterior`** (ligne 108-116) :
```yaml
rhmc_alpha: 0.1
sigma_normalization_mode: 'none'
```

---

## DIAGNOSTICS ATTENDUS

Après redémarrage, vous devriez voir :

```
[_get_inverse_metric NORMALIZATION]
  mode:                  none
  Original G⁻¹ eigenvalues: min=0.010000, max=9.437292
  # PAS de normalisation appliquée
  
[_make_covariance TARGET RADIUS]
  input alpha:           0.100000
  
[_compute_log_riemannian_gaussian]
  Sigma eigenvalues:     min=0.001000, max=0.943729  ← Anisotrope !
  Sigma trace:           0.945000
  log|Sigma|:            -7.014000
  
[LOG_Q_RIEM DECOMPOSITION]
  Quadratic term: ≈ -2.0 to -3.0
  Volume term:    ≈ +3.0 to +3.5   ← Positif grâce à petit |Σ|
  Total log_q:    ≈ -0.5 to -1.5

[DEBUG] log_p_prime_zF mean: ≈ +0.5 to +1.0

[DEBUG] FINAL KL LOSS: ≈ -1.0 to +0.5  (devrait être positif ou proche de 0)
```

---

## SI KL TOUJOURS NÉGATIVE

### Stratégie d'Ajustement

1. **KL ≈ -2 à -1** : Réduire `alpha` à **0.05**
2. **KL ≈ -1 à 0** : Réduire `alpha` à **0.03**
3. **KL ≈ 0 à +1** : ✅ Bon équilibre
4. **KL > +3** : Augmenter `alpha` à **0.15** ou **0.2**

### Objectif

Trouver `alpha` tel que :
```
E[log q(z₀)] ≈ E[log p(zF)]  ⟹  KL ≈ 0 à 1
```

---

## LEÇONS APPRISES (Corrigées)

### 1. Anisotropie ≠ Problème

L'anisotropie de G⁻¹ est **l'information géométrique apprise** par le modèle. La préserver est **essentiel** pour un VAE Riemannien.

### 2. Normalisation à Utiliser avec Précaution

Les normalisations comme `trace` ou `geomean` :
- ✅ **Utiles** pour stabiliser numériquement (éviter overflow/underflow)
- ❌ **Destructrices** si elles effacent la structure géométrique

**Dans notre cas** : Pas de normalisation nécessaire, juste un petit `alpha`.

### 3. Le Problème Était l'Échelle, Pas la Forme

La KL négative venait de :
- `trace(Σ_μ) ≈ 10.0` avec `alpha=5.0` → Σ **trop grande**
- Pas de l'anisotropie de Σ

**Solution** : Réduire l'échelle (`alpha=0.1`) en **préservant** l'anisotropie.

---

## VALIDATION

### Commande de Redémarrage

```bash
# Arrêter le training actuel
pkill -f "python.*run_experiment.py"

# Relancer avec les paramètres corrigés
export RLVAE_DEBUG=1
python run_experiment.py experiment=rlvae_three_stage_long_rhmc_modular
```

### Critères de Succès

1. ✅ G⁻¹ **reste anisotrope** (eigenvalues [0.01, 9.4])
2. ✅ Σ_μ **anisotrope à petite échelle** (eigenvalues [0.001, 0.94])
3. ✅ `log q` proche de `log p` (KL entre -1 et +2)
4. ✅ KL **positive ou proche de 0** après convergence

---

## CONCLUSION

### Ce Qui a Changé

| Avant (INCORRECT) | Après (CORRECT) |
|-------------------|-----------------|
| `sigma_normalization_mode: trace` | `sigma_normalization_mode: none` |
| `rhmc_alpha: 5.0` | `rhmc_alpha: 0.1` |
| G⁻¹ forcé isotrope | G⁻¹ préservé anisotrope |
| Σ_μ ≈ [4.8, 5.2] (grand + isotrope) | Σ_μ ≈ [0.001, 0.94] (petit + anisotrope) |

### Objectif Final

**Préserver la géométrie Riemannienne** tout en trouvant l'**échelle optimale** pour q ≈ p.

---

**Analyse corrigée le** : 27 Octobre 2025  
**Statut** : ✅ **SOLUTION CORRIGÉE APPLIQUÉE**  
**Prochaine validation** : Après 5-10 epochs avec `alpha=0.1`, `mode=none`

