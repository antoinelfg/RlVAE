# 🔍 Test des Diagnostics log_q et Σ_μ

## Modifications Apportées

Ajout de diagnostics détaillés dans `loss_manager.py` (lignes 1342-1413) pour décomposer `log_q` et analyser `Σ_μ`.

## Nouveaux Diagnostics Disponibles

Avec `RLVAE_DEBUG=1`, vous verrez maintenant :

### 1. Décomposition Complète de log_q

```
[LOG_Q DECOMPOSITION]
  Quadratic term:   -0.5432 (range: [-1.2345, +0.1234])
  Volume term:      -1.7036 (range: [-2.3456, -0.9876])
  Constant term:    -1.8379
  Total log_q:      -4.0847
  ||z0 - μ||:       1.0012
  log|Σ_μ|:         3.4072
```

**Interprétation** :
- **Quadratic term** : `-0.5 * (z0-μ)ᵀ Σ_μ⁻¹ (z0-μ)`
  - Mesure la distance Mahalanobis
  - Si très négatif → z0 loin de μ dans l'espace métrique
  
- **Volume term** : `-0.5 * log|Σ_μ|`
  - Normalization de la Gaussienne
  - Si très négatif → Σ_μ a un grand déterminant (variances larges)
  - Si moins négatif → Σ_μ a un petit déterminant (variances étroites)
  
- **Constant term** : `-0.5 * D * log(2π)` = -1.8379 pour D=2

### 2. Informations sur Σ_μ

```
  rhmc_alpha:       0.1
  Σ_μ eigenvalues:  min=0.001234, max=0.567890
  Σ_μ trace:        0.569124
```

**Ce qu'on cherche** :
- **rhmc_alpha trop petit** (< 0.1) → Σ_μ trop serrée
- **Eigenvalues très petites** (< 0.01) → Variances très faibles
- **Trace faible** → Covariance globalement petite

### 3. Comparaison avec G⁻¹(μ)

```
  G⁻¹(μ) eigenvalues: min=0.456789, max=21.234567
  G⁻¹(μ) trace:       21.691356
```

**Relation théorique** :
```
Σ_μ = α * G⁻¹(μ) + ε * I
```

Si `α = 0.1` et `G⁻¹(μ)` a des valeurs propres `[0.46, 21.23]` :
```
Σ_μ ≈ [0.046 + ε, 2.123 + ε]
```

Avec `ε = 1e-6` :
```
Σ_μ ≈ [0.046, 2.123]  ← Valeurs attendues
```

## Comment Utiliser Ces Diagnostics

### Étape 1 : Activer le Mode Debug

```bash
export RLVAE_DEBUG=1
python votre_script_entrainement.py
```

### Étape 2 : Examiner la Sortie

Cherchez le bloc `[LOG_Q DECOMPOSITION]` dans les logs.

### Étape 3 : Identifier le Problème

#### Scénario A : Volume term domine (très négatif)
```
  Quadratic term:   -0.2000
  Volume term:      -2.0000  ← COUPABLE
  Total log_q:      -4.0379
  log|Σ_μ|:         4.0000   ← Grand déterminant
```

**Cause** : `Σ_μ` a un grand déterminant (variances larges)  
**Pas un problème** : C'est normal, ce n'est pas ça qui cause KL négatif

#### Scénario B : Quadratic term domine (très négatif)
```
  Quadratic term:   -2.5000  ← COUPABLE
  Volume term:      +0.5000
  Total log_q:      -3.8379
  ||z0 - μ||:       3.4567   ← Grande distance
  Σ_μ eigenvalues:  min=0.0001, max=0.0012  ← TRÈS PETITES
```

**Cause** : `Σ_μ` trop serrée (petites eigenvalues)  
**Solution** : Augmenter `rhmc_alpha`

#### Scénario C : Les deux sont négatifs
```
  Quadratic term:   -1.2000
  Volume term:      -0.8000
  Total log_q:      -3.8379
```

**Analyse nécessaire** : Regarder les valeurs absolues et comparer

### Étape 4 : Ajuster rhmc_alpha

Si les eigenvalues de `Σ_μ` sont trop petites (< 0.01), augmenter `rhmc_alpha` :

```yaml
# Dans conf/model/votre_modele.yaml
rhmc_posterior:
  rhmc_alpha: 0.5  # Au lieu de 0.1
```

Ou dans le code Python si défini là :
```python
rhmc_posterior = RiemannianRHMCPosterior(
    ...
    rhmc_alpha=0.5,  # Augmenter cette valeur
    ...
)
```

## Valeurs Typiques Attendues

### Pour D=2, distance typique ||z0-μ|| ≈ 1.0

| rhmc_alpha | Σ_μ eigenvalues | Quadratic term | Volume term | Total log_q |
|-----------|----------------|----------------|-------------|-------------|
| 0.01      | [0.005, 0.2]   | -5.0           | +1.0        | -5.8        |
| 0.1       | [0.05, 2.0]    | -0.5           | -0.5        | -2.8        |
| 0.5       | [0.25, 10.0]   | -0.1           | -1.5        | -3.4        |
| 1.0       | [0.5, 20.0]    | -0.05          | -2.0        | -3.9        |

**Règle empirique** : 
- Si `log_q < -3.0` et quadratic term < -1.0 → `rhmc_alpha` probablement trop petit
- Valeur recommandée : `rhmc_alpha ∈ [0.3, 0.7]`

## Tests Rapides

### Test 1 : Vérifier que les diagnostics s'affichent

```bash
export RLVAE_DEBUG=1
# Lancer 1 batch d'entraînement
# Chercher "[LOG_Q DECOMPOSITION]" dans la sortie
```

### Test 2 : Comparer avec/sans augmentation de rhmc_alpha

```bash
# Baseline
export RLVAE_DEBUG=1
python script.py --model.rhmc_posterior.rhmc_alpha=0.1

# Test avec alpha plus grand
python script.py --model.rhmc_posterior.rhmc_alpha=0.5
```

Comparer les valeurs de `Total log_q` et vérifier si KL devient positive.

## Résumé

✅ **Diagnostics ajoutés** : Décomposition complète de `log_q`  
✅ **Information clé** : Valeur de `rhmc_alpha`  
✅ **Comparaison** : `Σ_μ` vs `G⁻¹(μ)`  
🎯 **Objectif** : Identifier si `Σ_μ` trop serrée cause `log_q` trop négatif  
🔧 **Solution** : Augmenter `rhmc_alpha` si confirmé

**Prochaine étape** : Lancer l'entraînement avec `RLVAE_DEBUG=1` et examiner les logs !

