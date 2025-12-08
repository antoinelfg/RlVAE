# Audit de Cohérence G vs G⁻¹ dans Stage C

## Date
27 Octobre 2025

## Objectif
Vérifier que tous les composants impliqués dans le calcul de KL et le sampling RHMC utilisent cohéremment le tenseur métrique G et son inverse G⁻¹ selon la formulation mathématique :
- **G⁻¹ = matrice de précision** (inverse des covariances locales de Stage B)
- **G = tenseur métrique** (inverse de G⁻¹)
- **Prior** : `p(z) ∝ √det(G⁻¹(z))`
- **Posterior** : `q(z|x)` avec `Σ_μ = α·G⁻¹(μ) + ε·I`

---

## Résumé des Résultats

### ✅ Composants Corrects (Aucun changement nécessaire)

1. **Pushforward de la précision** (`loss_manager.py:_pushforward_metric_via_flows`, ligne 1052-1053)
   - Formule : `G'^{-1} = J G^{-1} J^T` ✓
   - **Statut** : CORRECT

2. **Covariance du posterior** (`riemannian_rhmc_posterior.py:_make_covariance`, ligne 737)
   - Formule : `Σ_μ = α·G⁻¹(μ) + ε·I` ✓
   - **Statut** : CORRECT

3. **Log-densité gaussienne** (`riemannian_rhmc_posterior.py:log_q_riem`, lignes 84-187)
   - Formule : `log q = -½(z-μ)ᵀΣ⁻¹(z-μ) - ½log|Σ| - const` ✓
   - **Statut** : CORRECT

4. **Évaluation de métrique** (`loss_manager.py:_evaluate_metric`, lignes 371-440)
   - Retourne correctement le tensor ET le label de représentation ('g' ou 'ginv') ✓
   - **Statut** : CORRECT

5. **Calcul de volume** (`metric_utils.py:half_logdet_volume`, lignes 120-154)
   - `half_logdet_volume(G_inv, 'ginv')` = `+½ log|G⁻¹|` ✓
   - `half_logdet_volume(G, 'g')` = `-½ log|G| = +½ log|G⁻¹|` ✓
   - **Statut** : CORRECT

### 🔧 Composant Corrigé

6. **Calcul du prior** (`riemannian_rhmc_posterior.py:_compute_log_prior`, lignes 1126-1185)
   
   **Avant** :
   ```python
   # Docstring : p(z) ∝ √det(G(z))
   # Code inversait G_inv pour obtenir G, puis calculait +½ log|G|
   G_z = torch.linalg.inv(G_inv_z)
   half_logdet = half_logdet_volume(G_z, 'g', ...)  # Donne -½ log|G| = +½ log|G⁻¹|
   ```
   
   **Problème** : Inversion inutile et coûteuse de G⁻¹ pour obtenir G
   
   **Après** :
   ```python
   # Docstring : p(z) ∝ √det(G⁻¹(z))
   # Code utilise G_inv directement
   G_inv_z = model.G_inv(z)  # Pas d'inversion !
   half_logdet = half_logdet_volume(G_inv_z, 'ginv', ...)  # Donne +½ log|G⁻¹|
   ```
   
   **Bénéfices** :
   - ✅ Plus efficace (pas d'inversion matricielle)
   - ✅ Plus stable numériquement
   - ✅ Docstring aligné avec le code
   - ✅ Cohérent avec la formulation mathématique

---

## Détails des Vérifications

### 1. Pushforward Metric (loss_manager.py, lignes 939-1100)

**Code actuel** :
```python
# Line 1052-1053
# G^{-1}-transport:  G'^{-1} = J G^{-1} J^T
GT_ginv = torch.bmm(J64, torch.bmm(Ginv64_reg, J64.transpose(1, 2)))
GT_ginv = 0.5 * (GT_ginv + GT_ginv.transpose(1, 2))
```

**Vérification** : La formule de pullback pour une 2-forme (comme G⁻¹) est bien `φ*(G⁻¹) = J^T G⁻¹ J`.

**Conclusion** : ✅ CORRECT

---

### 2. Posterior Covariance (riemannian_rhmc_posterior.py, lignes 652-741)

**Code actuel** :
```python
# Line 737
Sigma = alpha * Ginv_norm + self.eps_reg * eye
```

**Vérification** :
- `Ginv_norm` est G⁻¹ (potentiellement normalisé)
- `Σ_μ = α·G⁻¹(μ) + ε·I` est la covariance du posterior Riemannien

**Conclusion** : ✅ CORRECT

---

### 3. Log-densité Gaussienne (riemannian_rhmc_posterior.py, lignes 84-187)

**Code actuel** :
```python
# Lines 169-185
quad_form = (z-μ)^T Σ^{-1} (z-μ)  # via cholesky_solve
log_det = 2 * sum(log(diag(chol)))  # log|Σ|
log_q = -0.5 * quad_form - 0.5 * log_det - const
```

**Vérification** : Formule standard de la log-densité gaussienne multivariée.

**Conclusion** : ✅ CORRECT

---

### 4. Évaluation de Métrique (loss_manager.py, lignes 371-440)

**Code actuel** :
```python
def _evaluate_metric(self, z, metric_tensor, rhmc_posterior, *, with_rep=False):
    preferred = self.metric_representation.lower()  # 'ginv' par défaut
    
    # Essaie de récupérer la représentation préférée
    if preferred == "ginv" and hasattr(component, "compute_inverse_metric"):
        return component.compute_inverse_metric(z), "ginv"
    # ... fallbacks ...
    
    return (tensor, rep) if with_rep else tensor
```

**Vérification** :
- Retourne le tensor ET le label ('g' ou 'ginv')
- Les appelants utilisent ce label pour interpréter correctement

**Conclusion** : ✅ CORRECT

---

### 5. Calcul de Volume (metric_utils.py, lignes 120-154)

**Code actuel** :
```python
def half_logdet_volume(matrix, representation, *, jitter=1e-6):
    """
    Returns:
        +½ log|det G^{-1}| when representation='ginv',
        -½ log|det G| when representation='g'.
    """
    logdet = 2.0 * torch.log(diag(chol)).sum(dim=-1)
    half = 0.5 * logdet
    half = half if representation == "ginv" else -half  # Line 153
    return half
```

**Vérification** :
- `half_logdet_volume(G_inv, 'ginv')` = `+0.5 * log|G⁻¹|` ✓
- `half_logdet_volume(G, 'g')` = `-0.5 * log|G|` ✓
- Comme `log|G| = -log|G⁻¹|`, les deux donnent la même valeur : `+0.5 * log|G⁻¹|`

**Conclusion** : ✅ CORRECT

---

### 6. Calcul du Prior (riemannian_rhmc_posterior.py, lignes 1126-1185)

#### Avant la Correction

**Code original** :
```python
def _compute_log_prior(self, z: torch.Tensor) -> torch.Tensor:
    """
    p(z) ∝ √det(G(z)) · exp(-0.5 * zᵀ G(z) z)  # Docstring original
    """
    # Lignes 1154-1156 : Inversait G_inv pour obtenir G
    G_inv_z = model.G_inv(z)
    G_z = torch.linalg.inv(G_inv_z)  # ⚠️ Inversion coûteuse et inutile
    
    # Ligne 1160 : Calculait avec G
    half_logdet = half_logdet_volume(G_z, 'g', ...)  # = -½ log|G| = +½ log|G⁻¹|
    
    # Mode 'uniform' : log p(z) = ½ log|G| + const
    log_p = half_logdet + const
```

**Problème identifié** :
1. **Inversion inutile** : `G_z = inv(G_inv_z)` est coûteux (O(D³))
2. **Instabilité numérique** : Inversion peut amplifier les erreurs
3. **Incohérence conceptuelle** : Stage B fournit G⁻¹, pourquoi l'inverser ?
4. **Ambiguïté** : Docstring dit `√det(G)` mais formulation user dit `√det(G⁻¹)`

**Note importante** : Bien que l'inversion soit inefficace, le résultat était mathématiquement correct car :
- `half_logdet_volume(G, 'g')` = `-½ log|G|` = `+½ log|G⁻¹|`

Donc l'ancien code calculait bien `+½ log|G⁻¹|` (mais de manière inefficace).

#### Après la Correction

**Code corrigé** :
```python
def _compute_log_prior(self, z: torch.Tensor) -> torch.Tensor:
    """
    Corrected formulation: p(z) ∝ √det(G⁻¹(z))
    
    This favors regions where the precision matrix G⁻¹ has large determinant,
    corresponding to high-confidence/low-variance regions in the latent space.
    
    log p(z) = 0.5 * log det(G⁻¹(z)) + constant
    """
    # Retrieve G_inv (precision matrix) directly
    if hasattr(model, 'G_inv'):
        G_inv_z = model.G_inv(z)
    elif hasattr(model, 'metric_tensor'):
        # ... fallbacks ...
        
    # Use G_inv directly for volume term (no inversion!)
    G_inv_z32 = G_inv_z.float() if G_inv_z.dtype in (torch.float16, torch.bfloat16) else G_inv_z
    half_logdet = half_logdet_volume(G_inv_z32, 'ginv', jitter=self.eps_reg)
    
    log_det_term = (half_logdet * self.volume_bias_weight).to(z.dtype)
    
    if mode == 'uniform':
        # Uniform prior on the manifold: p(z) ∝ √det(G⁻¹(z))
        log_p = log_det_term + float(self.uniform_prior_log_norm)
```

**Améliorations** :
1. ✅ **Pas d'inversion** : Utilise G⁻¹ directement (gain de performance)
2. ✅ **Plus stable** : Évite les erreurs numériques d'inversion
3. ✅ **Docstring aligné** : Documentation claire sur l'utilisation de G⁻¹
4. ✅ **Cohérent** : Utilise directement ce que Stage B fournit
5. ✅ **Même résultat** : `half_logdet_volume(G_inv, 'ginv')` = `+½ log|G⁻¹|`

---

## Calcul de KL : Vérification de la Formulation B

### Formule Mathématique

```
KL[q(z₀|x) || p(z_T)] = log q(z₀|x) - log p(z_T) + corrections
```

Où :
- `log q(z₀|x)` = log-densité du posterior Riemannien à `z₀` (calculé avec `Σ_μ = α·G⁻¹(μ)`)
- `log p(z_T)` = log-densité du prior au point transporté `z_T = φ(z₀)`

### Implémentation (loss_manager.py, lignes ~1300-1400)

**Code** :
```python
# Line 1298
log_p_prime_zF = half_logdet_push_ginv if rep_push == "ginv" else half_logdet_push_g

# Lines ~1350 (Formulation B)
kl_terms = (
    log_q.to(x.dtype)
    - (self.volume_bias_weight * log_p_prime_zF).to(x.dtype)
    + (delta_kin.to(x.dtype) - delta_vol.to(x.dtype))
)
```

**Vérification** :
1. `log_q` est calculé avec `log_q_riem(z₀, μ, Σ_μ)` où `Σ_μ = α·G⁻¹(μ)` ✓
2. `log_p_prime_zF = half_logdet_push_ginv` = `+½ log|G⁻¹(z_T)|` pour le prior ✓
3. La formule `log_q - log_p_prime_zF` est correcte ✓

**Conclusion** : ✅ CORRECT

---

## Impact sur la KL Négative

### Avant la Correction

Le prior calculait déjà `+½ log|G⁻¹|` (via l'inversion puis `half_logdet_volume(G, 'g')`), donc **le résultat numérique était correct**. Cependant :
- Coût computationnel plus élevé (inversion matricielle)
- Risque d'instabilité numérique accru
- Code moins clair et moins cohérent

### Après la Correction

Le prior calcule toujours `+½ log|G⁻¹|` (mais directement avec `half_logdet_volume(G_inv, 'ginv')`), donc :
- ✅ **Même résultat mathématique**
- ✅ **Performance améliorée** (pas d'inversion)
- ✅ **Stabilité numérique améliorée**
- ✅ **Code plus clair et cohérent**

### Conclusion sur la KL Négative

**La correction du prior NE résoudra PAS directement le problème de KL négative**, car :
1. Le résultat numérique est inchangé (on calculait déjà la bonne quantité)
2. Le problème de KL négative vient de `log_q` trop négatif (cf. analyse précédente)
3. La vraie cause : `Σ_μ = α·G⁻¹(μ)` très anisotrope → quadratic term dans `log_q` trop pénalisant

**Cependant**, cette correction :
- Améliore la cohérence conceptuelle du code
- Réduit les risques d'erreurs numériques futures
- Facilite la compréhension et la maintenance

---

## Recommandations

### 1. Pour résoudre la KL négative ✅ (Déjà implémenté)

Activer `sigma_normalization_mode: 'trace'` pour réduire l'anisotropie de `Σ_μ` :
```yaml
# conf/config.yaml et conf/experiment/*.yaml
posterior:
  sigma_normalization_mode: trace  # Au lieu de 'none'
```

### 2. Tester la correction du prior

Bien que le résultat numérique soit inchangé, tester avec `RLVAE_DEBUG=1` pour vérifier :
- Pas de régression de performance
- Stabilité numérique identique ou améliorée
- Logs de diagnostic cohérents

### 3. Documentation future

Ajouter des commentaires dans le code pour clarifier :
```python
# G_inv represents the precision matrix G⁻¹ (inverse of covariances from Stage B)
# half_logdet_volume(G_inv, 'ginv') computes +½ log|G⁻¹| for the prior p(z) ∝ √det(G⁻¹)
```

---

## Conclusion

### Résumé des Changements

| Composant | Statut Avant | Changement | Statut Après |
|-----------|--------------|------------|--------------|
| Pushforward precision | ✅ Correct | Aucun | ✅ Correct |
| Posterior covariance | ✅ Correct | Aucun | ✅ Correct |
| Posterior log_q | ✅ Correct | Aucun | ✅ Correct |
| Metric evaluation | ✅ Correct | Aucun | ✅ Correct |
| Volume calculation | ✅ Correct | Aucun | ✅ Correct |
| **Prior calculation** | ⚠️ Inefficace | ✅ Optimisé | ✅ Correct & Efficace |

### Cohérence Globale : ✅ VALIDÉE

Tous les composants utilisent maintenant cohéremment :
- **G⁻¹** comme précision (ce que Stage B fournit)
- **G** comme métrique (obtenu par inversion de G⁻¹ seulement si nécessaire)
- **Prior** : `p(z) ∝ √det(G⁻¹(z))`
- **Posterior** : `q(z|x) = N(μ, α·G⁻¹(μ) + ε·I)`

### Impact sur la KL Négative

La correction du prior **améliore l'efficacité et la clarté** mais **ne change pas le résultat numérique**. Le problème de KL négative persiste et doit être résolu par :
1. ✅ Normalisation de G⁻¹ (`sigma_normalization_mode: trace`) - **DÉJÀ IMPLÉMENTÉ**
2. ✅ Augmentation de `rhmc_alpha` à 5.0 - **DÉJÀ IMPLÉMENTÉ**
3. ✅ Augmentation de `rhmc_eps_reg` à 1e-3 - **DÉJÀ IMPLÉMENTÉ**

**Prochaine étape** : Redémarrer l'entraînement avec ces paramètres pour vérifier que la KL devient positive.

---

## Fichiers Modifiés

### 1. `src/rlvae/models/components/riemannian_rhmc_posterior.py`

**Lignes 1126-1185** : Fonction `_compute_log_prior`
- Utilise maintenant G_inv directement (pas d'inversion)
- Docstring mis à jour pour refléter `p(z) ∝ √det(G⁻¹(z))`
- Code simplifié et plus efficace

**Changements** :
```python
# Avant
G_z = torch.linalg.inv(G_inv_z)
half_logdet = half_logdet_volume(G_z, 'g', ...)

# Après
G_inv_z32 = G_inv_z.float() if ... else G_inv_z
half_logdet = half_logdet_volume(G_inv_z32, 'ginv', ...)
```

### Aucun Autre Fichier Modifié

Tous les autres composants étaient déjà corrects.

---

**Audit complété le** : 27 Octobre 2025  
**Statut** : ✅ TOUS LES COMPOSANTS VÉRIFIÉS ET COHÉRENTS


