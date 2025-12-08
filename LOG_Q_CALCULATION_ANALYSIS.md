# Analyse Détaillée du Calcul de log_q

## Vue d'ensemble du flux

```
μ (encoder mean) 
  ↓
G⁻¹(μ) = model.G_inv(μ)  [via _get_inverse_metric]
  ↓
Normalization (optional): G⁻¹_norm  [via sigma_normalization_mode]
  ↓
Σ_μ = α * G⁻¹_norm + ε * I  [via _make_covariance]
  ↓
z₀ ~ N(μ, Σ_μ)  [sampling]
  ↓
log q(z₀|μ, Σ_μ) = log_q_riem(z₀, μ, Σ_μ)  [via _compute_log_riemannian_gaussian]
```

---

## 1. Construction de G⁻¹(μ) : `_get_inverse_metric`

**Fichier**: `riemannian_rhmc_posterior.py`, lignes 843-873

### Étapes :
1. **Récupération** : `G_inv = model.G_inv(μ)` ou `G_inv = inv(model.G(μ))`
2. **Symétrisation** : `G_inv = (G_inv + G_inv^T) / 2`
3. **Vérification NaN/Inf** : Fallback à identité si détecté
4. **Clamping des eigenvalues** :
   ```python
   evals, evecs = torch.linalg.eigh(G_inv)
   floor = max(self.eps_reg, 1e-6)  # = 1e-3 maintenant
   evals = torch.clamp(evals, min=floor)
   ```
5. **Optional ceiling** : `evals = clamp(evals, max=metric_eig_ceiling)`

### Normalisation (si `sigma_normalization_mode != 'none'`) :

#### Mode `'trace'` (actuellement activé) :
```python
tr = evals.sum(dim=-1)  # trace de G⁻¹
evals_normalized = d * evals / tr  # Normalise pour que trace = d
```

**Effet** :
- Si `tr(G⁻¹) = 100` et `d = 2`, alors `evals_new = 2 * evals / 100`
- Les eigenvalues sont **réduites** si la trace est grande
- Rend G⁻¹ "plus isotrope" tout en préservant les ratios relatifs

#### Mode `'geomean'` (désactivé) :
```python
gm = exp(mean(log(evals)))  # geometric mean
evals_normalized = evals / gm  # Normalise pour que det(G⁻¹) = 1
```

---

## 2. Construction de Σ_μ : `_make_covariance`

**Fichier**: `riemannian_rhmc_posterior.py`, lignes 652-741

### Formule de base :
```python
Σ_μ = α * G⁻¹_norm + ε * I
```

Où :
- `α = rhmc_alpha` (= 5.0 actuellement)
- `ε = rhmc_eps_reg` (= 1e-3 actuellement)
- `G⁻¹_norm` est la métrique inverse normalisée (si normalization activée)

### Ajustement optionnel du radius cible (désactivé) :
Si `initial_target_radius > 0` :
```python
α_eff = (target_r² - d*ε) / trace(G⁻¹_norm)
Σ_μ = α_eff * G⁻¹_norm + ε * I
```
Cela force `trace(Σ_μ) ≈ target_r²`, rendant les échantillons de rayon ~`target_r`.

**Actuellement désactivé** (`initial_target_radius = 0.0`), donc `α_eff = α = 5.0`.

---

## 3. Calcul de log q : `log_q_riem`

**Fichier**: `riemannian_rhmc_posterior.py`, lignes 84-187

### Formule de la log-densité gaussienne :
```
log q(z|μ, Σ) = -½(z-μ)ᵀΣ⁻¹(z-μ) - ½log|Σ| - d/2·log(2π)
              = [quadratic term] + [volume term] + [constant term]
```

### Implémentation numérique :

1. **Cholesky de Σ** (avec stabilisation) :
   ```python
   chol, stabilized_sigma, was_stabilized = _safe_cholesky(Sigma, min_eig)
   # Si échec, ajoute jitter: Sigma_stab = Sigma + min_eig * I
   ```

2. **Terme quadratique** :
   ```python
   diff = z - μ
   sol = cholesky_solve(diff, chol)  # Résout Σ·sol = diff
   quad_form = dot(diff, sol)        # = diffᵀΣ⁻¹diff
   quadratic_term = -0.5 * quad_form
   ```

3. **Terme volumique (log-déterminant)** :
   ```python
   log_det = 2 * sum(log(diag(chol)))  # log|Σ| = 2·log|L| où Σ = LLᵀ
   volume_term = -0.5 * log_det
   ```

4. **Constante** :
   ```python
   const = 0.5 * d * log(2π)
   constant_term = -const
   ```

5. **Somme** :
   ```python
   log_q = quadratic_term + volume_term + constant_term
   ```

---

## 4. Diagnostic : Pourquoi log_q est négatif ?

### Exemple des logs récents (avec α=5.0, mode='none') :

```
Σ_μ eigenvalues: min=0.075808, max=104.733902  (ratio ~1382:1)
||z - μ||:       1.1192

Quadratic term:  -0.8540  (mean), range [-9.98, -0.0001]
Volume term:     -2.3979
Constant term:   -1.8379
Total log_q:     -5.0898
```

### Analyse :

#### A. Terme quadratique : `-0.85` (problématique)

**Formule** : `-(1/2) * (z-μ)ᵀ Σ_μ⁻¹ (z-μ)`

- `||z-μ|| = 1.12` : distance euclidienne modérée
- Mais `Σ_μ⁻¹` a des eigenvalues **très anisotropes** :
  - `λ_min(Σ_μ⁻¹) = 1/104.73 ≈ 0.0095`
  - `λ_max(Σ_μ⁻¹) = 1/0.076 ≈ 13.2`

**Problème** : Si `(z-μ)` a une composante **même petite** dans la direction de `λ_max(Σ_μ⁻¹)` (= direction de `λ_min(Σ_μ)`), cela contribue **énormément** à la forme quadratique.

**Exemple numérique** :
- Direction "large" de Σ_μ (eigenvalue ~100) : composante de 1.0 → contribution ~0.01 à quad_form
- Direction "serrée" de Σ_μ (eigenvalue ~0.08) : composante de 0.2 → contribution ~0.5 à quad_form
- **Résultat** : quad_form peut facilement être > 2, donnant quadratic_term < -1

**Pire cas** : Certains échantillons ont `quad_term = -9.98`, ce qui signifie `quad_form ≈ 20` !

#### B. Terme volumique : `-2.40` (attendu)

**Formule** : `-(1/2) * log|Σ_μ|`

- `log|Σ_μ| = log(∏ eigenvalues) = log(0.076 * 104.73) ≈ 4.80`
- Volume term = `-0.5 * 4.80 = -2.40`

**C'est normal !** Plus Σ_μ est "large" (grand déterminant), plus ce terme est négatif.

#### C. Constante : `-1.84`

**Formule** : `-d/2 * log(2π)` où `d=2`
- Constante = `-1 * log(2π) = -1.84`

**C'est une constante universelle** pour toute gaussienne 2D.

---

## 5. Effet Attendu de `sigma_normalization_mode: 'trace'`

Avec la normalisation `trace` activée :

### Avant (mode='none') :
```
G⁻¹ eigenvalues: [0.01, 14.50]  (trace ≈ 14.51)
Σ_μ = 5.0 * [0.01, 14.50] + 1e-3 = [0.06, 72.50]
log|Σ_μ| ≈ 4.80
```

### Après (mode='trace') :
```
G⁻¹ eigenvalues normalized: 2 * [0.01, 14.50] / 14.51 ≈ [0.0014, 2.00]  (trace = 2.0)
Σ_μ = 5.0 * [0.0014, 2.00] + 1e-3 ≈ [0.008, 10.00]
log|Σ_μ| ≈ log(0.008 * 10) = log(0.08) ≈ -2.53
```

### Impact sur log_q :

1. **Quadratic term** : Devient **moins négatif**
   - Ratio des eigenvalues de Σ_μ réduit de ~1200:1 à ~1250:1 (similaire)
   - **MAIS** les valeurs absolues sont plus petites, donc `Σ_μ⁻¹` est plus "serré"
   - **Attention** : Cela pourrait **augmenter** le terme quadratique si z est loin de μ !

2. **Volume term** : Devient **moins négatif** (ou positif!)
   - `log|Σ_μ|` passe de ~4.80 à ~-2.53
   - Volume term passe de `-2.40` à **+1.27** 🎉

3. **Net** :
   - Si quadratic term reste similaire (≈ -0.85)
   - log_q ≈ (-0.85) + (+1.27) + (-1.84) = **-1.42** (au lieu de -5.09)
   - **Amélioration de ~3.5 !**

---

## 6. Limitation Fondamentale

Même avec la normalisation `trace`, **l'anisotropie relative** de Σ_μ est préservée. La vraie solution serait :

### Option 1 : Ignorer complètement la métrique
```python
Σ_μ = α * I  # Isotrope, pas de géométrie Riemannienne
```

### Option 2 : "Whitening" de G⁻¹
Transformer G⁻¹ pour avoir des eigenvalues uniformes :
```python
evals_whitened = ones_like(evals) * geometric_mean(evals)
G_inv_whitened = evecs @ diag(evals_whitened) @ evecs.T
Σ_μ = α * G_inv_whitened + ε * I
```

### Option 3 : Utiliser une approximation différente de q
Au lieu de `q = N(μ, α*G⁻¹+ε)`, utiliser :
```python
q = N(μ, α*I) * correction_factor(G⁻¹)
```

---

## Conclusion

Le calcul de `log_q` est **mathématiquement correct**, mais la distribution gaussienne `N(μ, α*G⁻¹+ε)` est **inadaptée** quand G⁻¹ est très anisotrope.

**La normalisation `trace`** devrait significativement améliorer la situation en :
1. Réduisant le terme volumique (de négatif à potentiellement positif)
2. Rendant les échantillons plus "typiques" de la distribution

**Test crucial** : Redémarrer l'entraînement avec `sigma_normalization_mode: 'trace'` et vérifier si `log_q` devient moins négatif et si KL devient positive.


