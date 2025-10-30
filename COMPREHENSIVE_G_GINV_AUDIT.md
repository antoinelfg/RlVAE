# Audit Complet de Cohérence G vs G⁻¹
## Fichiers: loss_manager.py & riemannian_rhmc_posterior.py

**Date**: 27 Octobre 2025  
**Convention**: Le modèle travaille principalement avec la **précision** G⁻¹, variable nommée `G_inv` ou `G` avec label `'ginv'`  
**Prior cible**: p(z) ∝ √det(G⁻¹(z)) ⟹ log p(z) = +½ log|G⁻¹(z)| + const

---

## RÉSUMÉ EXÉCUTIF

### ✅ Composants Vérifiés et CORRECTS

| Composant | Fichier | Lignes | Statut |
|-----------|---------|--------|--------|
| `_half_logdet_volume` | loss_manager.py | 585-596 | ✅ CORRECT |
| `half_logdet_volume` (global) | metric_utils.py | 120-154 | ✅ CORRECT |
| `_evaluate_metric` | loss_manager.py | 371-440 | ✅ CORRECT |
| `_pushforward_metric_via_flows` | loss_manager.py | 939-1100 | ✅ CORRECT |
| `_make_covariance` | riemannian_rhmc_posterior.py | 652-741 | ✅ CORRECT |
| `_get_inverse_metric` | riemannian_rhmc_posterior.py | 843-873 | ✅ CORRECT |
| `log_q_riem` | riemannian_rhmc_posterior.py | 84-187 | ✅ CORRECT |
| `_compute_log_prior` | riemannian_rhmc_posterior.py | 1126-1193 | ✅ CORRIGÉ |

### 🔧 Composants Nécessitant Vérification Approfondie

| Composant | Fichier | Lignes | Préoccupation |
|-----------|---------|--------|---------------|
| `_log_kinetic_density` | loss_manager.py | 825-867 | ⚠️ Utilise G, devrait utiliser G⁻¹ |
| `_compute_potential_gradient` | riemannian_rhmc_posterior.py | 1193-1241 | ⚠️ Signe de `volume_force_sign` à vérifier |
| `_sample_momentum` | riemannian_rhmc_posterior.py | ~900-950 | ⚠️ Doit inverser G⁻¹ pour obtenir G |
| `_quad_with_G` | loss_manager.py | 697-783 | ⚠️ Nom ambigu, vérifier usage |

---

## PARTIE 1: FONCTIONS DE BASE (LOSS_MANAGER.PY)

### 1.1 `_half_logdet_volume` (lignes 585-596)

**Objectif**: Calculer +½ log|G⁻¹| indépendamment de la représentation fournie.

**Code**:
```python
def _half_logdet_volume(
    self,
    G_or_Ginv: torch.Tensor,
    rep: Optional[str] = None,
    *,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """
    Return +½ log det G^{-1} regardless of representation supplied.
    """
    rep_effective = (rep or self.metric_representation).lower()
    return _global_half_logdet_volume(G_or_Ginv, rep_effective, jitter=jitter)
```

**Vérification**: ✅ **CORRECT**
- Docstring claire: "Return +½ log det G^{-1}"
- Délègue à la fonction globale `half_logdet_volume` de `metric_utils.py`

**Fonction globale** (`metric_utils.py`, lignes 120-154):
```python
def half_logdet_volume(matrix: torch.Tensor, representation: str, *, jitter: float = 1e-6) -> torch.Tensor:
    """
    Returns:
        +½ log|det G⁻¹| when representation='ginv',
        -½ log|det G| when representation='g'.
    """
    # ... Cholesky computation ...
    logdet = 2.0 * torch.log(diag).sum(dim=-1)
    half = 0.5 * logdet
    half = half if representation == "ginv" else -half  # Line 153
    return half
```

**Vérification**: ✅ **CORRECT**
- `half_logdet_volume(G_inv, 'ginv')` → `+0.5 * log|G⁻¹|` ✓
- `half_logdet_volume(G, 'g')` → `-0.5 * log|G| = +0.5 * log|G⁻¹|` ✓
- Les deux donnent le même résultat final

---

### 1.2 `_evaluate_metric` (lignes 371-440)

**Objectif**: Récupérer le tensor métrique et son label de représentation.

**Code**:
```python
def _evaluate_metric(
    self,
    z: Optional[torch.Tensor],
    metric_tensor: Optional[Any],
    rhmc_posterior: Optional[Any],
    *,
    with_rep: bool = False,
) -> Union[Optional[torch.Tensor], Tuple[Optional[torch.Tensor], Optional[str]]]:
    """
    Returns a symmetrized SPD tensor together with the representation tag
    ('g' for metric, 'ginv' for precision) or (None, None) if no
    metric information is available.
    """
    preferred = self.metric_representation.lower()  # Default: 'ginv'
    
    def _resolve(component: Any) -> tuple[Optional[torch.Tensor], Optional[str]]:
        if preferred == "ginv" and hasattr(component, "compute_inverse_metric"):
            return component.compute_inverse_metric(z), "ginv"
        if preferred == "g" and hasattr(component, "compute_metric"):
            return component.compute_metric(z), "g"
        # ... fallbacks ...
    
    tensor, rep = _resolve(metric_tensor)
    # ... additional fallbacks ...
    
    return (tensor, rep) if with_rep else tensor
```

**Vérification**: ✅ **CORRECT**
- Préfère `'ginv'` par défaut (ligne 389, via `self.metric_representation`)
- Retourne le tensor ET son label
- Les appelants doivent interpréter correctement selon le label

**Exemple d'utilisation correcte**:
```python
G_source, rep_source = self._evaluate_metric(z0, metric_tensor, rhmc_posterior, with_rep=True)
half_logdet = self._half_logdet_volume(G_source, rep_source.lower(), jitter=eps_reg)
```

---

### 1.3 `_pushforward_metric_via_flows` (lignes 939-1100)

**Objectif**: Transporter la métrique/précision à travers les flows normalisants.

**Code critique** (lignes 1044-1054):
```python
# G-transport:    G' = J^{-T} G J^{-1}
J_inv = self._stable_matrix_inverse(J64, jitter=1e-6)
GT_g  = torch.bmm(J_inv.transpose(1, 2), torch.bmm(G064_reg, J_inv))
GT_g  = 0.5 * (GT_g + GT_g.transpose(1, 2))

# G^{-1}-transport:  G'^{-1} = J G^{-1} J^T
GT_ginv = torch.bmm(J64, torch.bmm(Ginv64_reg, J64.transpose(1, 2)))
GT_ginv = 0.5 * (GT_ginv + GT_ginv.transpose(1, 2))
```

**Vérification**: ✅ **CORRECT**
- **Formule pour G**: G' = (J⁻¹)ᵀ G J⁻¹ ✓ (pullback d'une métrique)
- **Formule pour G⁻¹**: (G⁻¹)' = J G⁻¹ Jᵀ ✓ (pullback d'une 2-forme/précision)
- Les deux formules sont mathématiquement correctes

**Calcul des log-déterminants** (lignes 1056-1075):
```python
half_logdet_push_g = self._half_logdet_volume(GT_g, 'g', jitter=1e-6)
half_logdet_push_ginv = self._half_logdet_volume(GT_ginv, 'ginv', jitter=1e-6)
```

**Vérification**: ✅ **CORRECT**
- Utilise les labels corrects pour chaque représentation
- Les deux calculent effectivement +½ log|G⁻¹(zF)|

---

### 1.4 `_log_kinetic_density` (lignes 825-867)

**Objectif**: Calculer log π_kin(ρ|z) = -½ρᵀG⁻¹(z)ρ + ½log|G(z)| - const

**Code**:
```python
def _log_kinetic_density(
    self,
    model: Any,
    z: torch.Tensor,
    rho: torch.Tensor,
    *,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """
    log π_kin(ρ|z) for an RHMC kinetic energy with G(z).
    
    π_kin(ρ|z) = N(ρ; 0, G(z)) 
             ∝ |G(z)|^{1/2} exp(-½ ρᵀ G^{-1}(z) ρ)
    
    so log π_kin(ρ|z) = +½ log|G(z)| - ½ ρᵀ G^{-1}(z) ρ - const
    """
    # G or G^{-1}, plus its representation tag
    G_or_Ginv, rep = self._evaluate_metric(z, model, None, with_rep=True)
    if G_or_Ginv is None or rep is None:
        # Fallback: standard kinetic with identity metric
        quad = torch.sum(rho ** 2, dim=-1)
        const = 0.5 * z.shape[-1] * math.log(2 * math.pi)
        return (-0.5 * quad - const).to(rho.dtype)
    
    rep = rep.lower()
    if rep == "ginv":
        # We have G^{-1}, compute quadratic directly
        rho32 = rho.float()
        quad = torch.einsum('bi,bij,bj->b', rho32, G_or_Ginv.float(), rho32)
        half_logdet = self._half_logdet_volume(G_or_Ginv, 'ginv', jitter=jitter)
        return (-0.5 * quad + half_logdet - const).to(rho.dtype)
    if rep == "g":
        # We have G, need to invert for quadratic
        G_inv = torch.linalg.inv(G_or_Ginv.float())
        rho32 = rho.float()
        quad = torch.einsum('bi,bij,bj->b', rho32, G_inv, rho32)
        half_logdet = self._half_logdet_volume(G_or_Ginv, 'g', jitter=jitter)
        return (-0.5 * quad + half_logdet - const).to(rho.dtype)
```

**PROBLÈME IDENTIFIÉ**: ⚠️ **INCOHÉRENCE DANS LE SIGNE DU TERME VOLUMIQUE**

**Analyse mathématique**:

La densité cinétique est:
```
π_kin(ρ|z) = N(ρ; 0, G(z))
           = (2π)^{-d/2} |G(z)|^{+1/2} exp(-½ρᵀG⁻¹(z)ρ)

log π_kin = +½ log|G(z)| - ½ρᵀG⁻¹ρ - ½d·log(2π)
          = -½ log|G⁻¹(z)| - ½ρᵀG⁻¹ρ - ½d·log(2π)
```

**Code actuel**:
- Ligne 853: `half_logdet = self._half_logdet_volume(G_or_Ginv, 'ginv', ...)` → retourne **+½ log|G⁻¹|**
- Ligne 854: `return (-0.5 * quad + half_logdet - const)` → retourne **-½ quad + ½log|G⁻¹| - const**

**ERREUR**: Le terme devrait être **-½ log|G⁻¹|** et non **+½ log|G⁻¹|** !

**Correction proposée**:
```python
if rep == "ginv":
    # We have G^{-1}, compute quadratic directly
    rho32 = rho.float()
    quad = torch.einsum('bi,bij,bj->b', rho32, G_or_Ginv.float(), rho32)
    # FIX: Kinetic density uses -½ log|G⁻¹| = +½ log|G|
    half_logdet_ginv = self._half_logdet_volume(G_or_Ginv, 'ginv', jitter=jitter)
    half_logdet_g = -half_logdet_ginv  # Convert to +½ log|G|
    const = 0.5 * z.shape[-1] * math.log(2 * math.pi)
    return (-0.5 * quad + half_logdet_g - const).to(rho.dtype)
```

Ou plus simplement:
```python
if rep == "ginv":
    rho32 = rho.float()
    quad = torch.einsum('bi,bij,bj->b', rho32, G_or_Ginv.float(), rho32)
    half_logdet_ginv = self._half_logdet_volume(G_or_Ginv, 'ginv', jitter=jitter)
    const = 0.5 * z.shape[-1] * math.log(2 * math.pi)
    # log π_kin = -½ρᵀG⁻¹ρ - ½log|G⁻¹| - const
    return (-0.5 * quad - half_logdet_ginv - const).to(rho.dtype)
```

**Impact**: Cette erreur affecterait le calcul de `delta_kin` dans la KL, mais seulement si RHMC avec Jacobian est utilisé (ce qui n'est pas le cas actuellement selon la config).

---

### 1.5 `_quad_with_G` (lignes 697-783)

**Objectif**: Calculer forme quadratique avec métrique.

**Code** (lignes 697-708):
```python
def _quad_with_G(
    self,
    z: torch.Tensor,
    G_or_Ginv: torch.Tensor,
    rep: str,
    *,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """
    Compute zᵀ G z given either G or G^{-1}.
    
    Args:
        G_or_Ginv: Either metric (rep='g') or precision (rep='ginv')
        rep: 'g' or 'ginv'
    """
```

**Code** (lignes 714-730):
```python
rep = rep.lower()
if rep == "g":
    # Direct quadratic with G
    quad = torch.einsum('bi,bij,bj->b', z32, G_or_Ginv32, z32)
    return quad.to(z.dtype)

elif rep == "ginv":
    # Need to invert to get G
    try:
        G = torch.linalg.inv(G_or_Ginv32)
        quad = torch.einsum('bi,bij,bj->b', z32, G, z32)
        return quad.to(z.dtype)
    except RuntimeError:
        # Fallback with Cholesky
        chol, _ = self._cholesky_spd(G_or_Ginv32, jitter=jitter)
        sol = torch.cholesky_solve(
            torch.eye(d, device=z.device, dtype=G_or_Ginv32.dtype).unsqueeze(0),
            chol
        )
```

**Vérification**: ✅ **CORRECT (mais nom ambigu)**
- Calcule bien zᵀGz en inversant G⁻¹ si nécessaire
- Utilise Cholesky en fallback pour stabilité

**Recommandation**: ⚠️ Renommer en `_quad_with_metric` pour clarté, car la fonction calcule spécifiquement zᵀ**G**z (pas zᵀG⁻¹z).

---

## PARTIE 2: FONCTIONS RHMC (RIEMANNIAN_RHMC_POSTERIOR.PY)

### 2.1 `_compute_log_prior` (lignes 1126-1193)

**DÉJÀ CORRIGÉ** dans l'audit précédent.

**Code actuel** (après correction):
```python
def _compute_log_prior(self, z: torch.Tensor) -> torch.Tensor:
    """
    Corrected formulation: p(z) ∝ √det(G⁻¹(z))
    
    log p(z) = 0.5 * log det(G⁻¹(z)) + constant
    """
    # Retrieve G_inv (precision matrix) directly
    if hasattr(model, 'G_inv'):
        G_inv_z = model.G_inv(z)
    # ... fallbacks ...
    
    # Use G_inv directly for volume term
    G_inv_z32 = G_inv_z.float() if ... else G_inv_z
    half_logdet = half_logdet_volume(G_inv_z32, 'ginv', jitter=self.eps_reg)
    
    log_det_term = (half_logdet * self.volume_bias_weight).to(z.dtype)
    
    if mode == 'uniform':
        # Uniform prior on manifold: p(z) ∝ √det(G⁻¹(z))
        log_p = log_det_term + float(self.uniform_prior_log_norm)
```

**Vérification**: ✅ **CORRECT**
- Utilise G⁻¹ directement (pas d'inversion inutile)
- Calcule +½ log|G⁻¹| comme requis
- Docstring aligné avec le code

---

### 2.2 `_compute_potential_gradient` (lignes 1193-1241)

**Objectif**: Calculer ∇U(z) où U(z) = -log p(z) pour le leapfrog RHMC.

**Code** (extraits pertinents):
```python
def _compute_potential_gradient(self, z: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of potential energy U(z) = -log p(z).
    
    For p(z) ∝ √det(G⁻¹(z)), we have:
    U(z) = -½ log|G⁻¹(z)| + const
    ∇U(z) = -½ ∇log|G⁻¹(z)|
    """
    try:
        model = self._ctx['model']
        # ... G_inv retrieval ...
        
        # Compute gradient of log|G⁻¹(z)|
        grad_logdet_ginv = torch.autograd.grad(
            logdet_ginv.sum(),
            z_req,
            retain_graph=False,
            create_graph=False,
            allow_unused=True
        )[0]
        
        if grad_logdet_ginv is None:
            grad_logdet_ginv = torch.zeros_like(z_req)
        
        # Potential gradient: ∇U = -½ ∇log|G⁻¹|
        volume_force_sign = float(getattr(self, 'volume_force_sign', -1.0))
        grad_U = volume_force_sign * 0.5 * grad_logdet_ginv
        
        return grad_U.detach()
```

**Analyse**:

Pour p(z) ∝ √det(G⁻¹(z)):
```
log p(z) = +½ log|G⁻¹(z)| + const
U(z) = -log p(z) = -½ log|G⁻¹(z)| + const
∇U(z) = -½ ∇log|G⁻¹(z)|
```

Dans le leapfrog, on met à jour:
```
ρ ← ρ - ε·∇U(z)
  = ρ - ε·(-½ ∇log|G⁻¹|)
  = ρ + ε·½ ∇log|G⁻¹|
```

**Code actuel**:
```python
volume_force_sign = -1.0  # Default
grad_U = volume_force_sign * 0.5 * grad_logdet_ginv
       = -1.0 * 0.5 * ∇log|G⁻¹|
       = -½ ∇log|G⁻¹|
```

**Vérification**: ✅ **CORRECT**
- `volume_force_sign = -1.0` donne le bon signe
- `grad_U = -½ ∇log|G⁻¹|` correspond à ∇U pour p ∝ √det(G⁻¹)

---

### 2.3 `_make_covariance` (lignes 652-741)

**DÉJÀ VÉRIFIÉ** dans l'audit précédent.

**Code**:
```python
def _make_covariance(self, G_inv: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Build SPD covariance Σ = α·Ĝ^{-1} + εI
    """
    # ... normalization ...
    Sigma = alpha * Ginv_norm + self.eps_reg * eye
    return self._stabilize_spd(_symmetrize(Sigma), self.min_cov_eig)
```

**Vérification**: ✅ **CORRECT**
- Formule Σ_μ = α·G⁻¹(μ) + ε·I ✓
- G_inv est bien la précision interpolée

---

### 2.4 `_sample_momentum` (lignes 916-938)

**Objectif**: Échantillonner le momentum ρ ~ N(0, G(z))

**Code**:
```python
def _sample_momentum(self, z: torch.Tensor) -> torch.Tensor:
    """
    Simple momentum sampling: ρ ~ N(0, G(z))
    """
    try:
        G = self._ctx['model'].G(z)  # ← Appelle model.G()
        G = _symmetrize(G)
        L, _, _ = _safe_cholesky(G + self.eps_reg * eye, self.eps_reg)
        eps = torch.randn_like(z, dtype=L.dtype)
        rho32 = torch.einsum('bij,bj->bi', L, eps)  # ρ = L·ε
        rho = rho32.to(z.dtype)
        # ... clipping ...
        return rho
    except:
        # Fallback to isotropic
        rho = torch.randn_like(z)
        # ... clipping ...
        return rho
```

**Analyse**:

Pour échantillonner ρ ~ N(0, G(z)):
1. Décomposer G(z) = LLᵀ (Cholesky)
2. Échantillonner ε ~ N(0, I)
3. Calculer ρ = L·ε

**Problème potentiel**: ⚠️ **DÉPEND DE LA DISPONIBILITÉ DE `model.G()`**

Le code appelle `model.G(z)` (ligne 921), mais le modèle fournit principalement `G_inv(z)`. 

**Vérification nécessaire**:
- Si `model.G()` existe et retourne G (métrique), alors ✅ CORRECT
- Si `model.G()` n'existe pas, le code va dans le fallback (sampling isotrope)
- Si `model.G()` existe mais retourne G⁻¹ par erreur, alors ❌ ERREUR

**Recommandation**: Vérifier l'implémentation de `model.G()`:
```python
# Option 1: model.G() existe et inverse G_inv
def G(self, z):
    G_inv = self.metric_tensor.compute_inverse_metric(z)
    return torch.linalg.inv(G_inv)

# Option 2: model.G() n'existe pas → fallback isotrope
# Option 3 (DANGER): model.G() retourne G_inv par erreur
```

**Note**: Si RHMC n'est pas actuellement utilisé avec leapfrog (rhmc_steps=0), cette fonction n'est pas appelée et l'erreur potentielle n'a pas d'impact.

---

### 2.5 `_leapfrog_step` (lignes 940-1058)

**Objectif**: Intégration leapfrog pour RHMC

**Code** (extrait simplifié):
```python
def _leapfrog_step(self, z, rho, step_size):
    """
    Simple leapfrog: updates (z, ρ) with G(z)-aware dynamics.
    
    Omits kinetic position-dependence term for simplicity.
    """
    # Half step for momentum
    grad_U = self._compute_potential_gradient(z)
    rho = rho - 0.5 * step_size * grad_U
    
    # Full step for position
    z_req = z.clone().requires_grad_(True)
    G_inv = self._get_inverse_metric(z_req)
    # Velocity: v = G⁻¹(z)·ρ
    velocity = torch.einsum('bij,bj->bi', G_inv, rho)
    z = z_req + step_size * velocity
    
    # Half step for momentum
    grad_U = self._compute_potential_gradient(z)
    rho = rho - 0.5 * step_size * grad_U
    
    return z, rho
```

**Analyse**:

Le leapfrog Riemannian utilise:
- **Vitesse**: v = G⁻¹(z)·ρ (car ρ = G·v)
- **Update position**: z ← z + ε·v = z + ε·G⁻¹·ρ
- **Update momentum**: ρ ← ρ - ε·∇U

**Code actuel** (ligne ~970-980):
```python
G_inv = self._get_inverse_metric(z_req)
velocity = torch.einsum('bij,bj->bi', G_inv, rho)
z = z_req + step_size * velocity
```

**Vérification**: ✅ **CORRECT**
- Utilise G⁻¹ pour calculer la vitesse
- Formula v = G⁻¹·ρ est correcte

**Note importante** (commentaire ligne 944-945):
> "Omits the kinetic position-dependence term -0.5 ∇_z [ρᵀ G^{-1}(z) ρ] for simplicity"

Cela signifie que l'implémentation est un **leapfrog simplifié** qui ignore le terme de dérivée de la métrique. C'est une approximation acceptable si:
- G⁻¹(z) varie peu localement
- Les pas de temps sont petits
- Des mécanismes de clipping sont en place

---

## PARTIE 3: RÉSUMÉ DES PROBLÈMES IDENTIFIÉS

### 🔴 ERREUR CRITIQUE #1: `_log_kinetic_density` (loss_manager.py)

**Fichier**: `loss_manager.py`  
**Ligne**: 853-854  
**Problème**: Signe incorrect pour le terme volumique

**Code actuel**:
```python
if rep == "ginv":
    quad = torch.einsum('bi,bij,bj->b', rho32, G_or_Ginv.float(), rho32)
    half_logdet = self._half_logdet_volume(G_or_Ginv, 'ginv', jitter=jitter)
    return (-0.5 * quad + half_logdet - const).to(rho.dtype)
    #                      ^^^^^^^^^^^ ERREUR: devrait être - half_logdet
```

**Formule correcte**:
```
log π_kin(ρ|z) = -½ρᵀG⁻¹ρ - ½log|G⁻¹| - ½d·log(2π)
               = -½ρᵀG⁻¹ρ + ½log|G| - ½d·log(2π)
```

**Correction**:
```python
if rep == "ginv":
    quad = torch.einsum('bi,bij,bj->b', rho32, G_or_Ginv.float(), rho32)
    half_logdet_ginv = self._half_logdet_volume(G_or_Ginv, 'ginv', jitter=jitter)
    const = 0.5 * z.shape[-1] * math.log(2 * math.pi)
    # FIX: Kinetic density uses -½log|G⁻¹| = +½log|G|
    return (-0.5 * quad - half_logdet_ginv - const).to(rho.dtype)
```

**Impact**: 
- Affecte le calcul de `delta_kin` dans la KL (si `rhmc_kl_jacobian=True`)
- Actuellement, `rhmc_kl_jacobian=False` dans la config, donc l'erreur n'a **pas d'impact pratique**
- **Correction recommandée pour la cohérence mathématique**

---

### ✅ VÉRIFIÉ: `_sample_momentum` (riemannian_rhmc_posterior.py)

**Fichier**: `riemannian_rhmc_posterior.py`  
**Ligne**: 921  
**Code**: Appelle `model.G(z)`

**Code actuel**:
```python
G = self._ctx['model'].G(z)  # ← Appelle model.G()
```

**Vérification effectuée**: ✅ `model.G()` **EXISTE ET EST CORRECT**

**Implémentation** (modular_rlvae.py, lignes 337-342):
```python
def _G_impl(z: torch.Tensor) -> torch.Tensor:
    return self.modular_metric.compute_metric(z)

def _Ginv_impl(z: torch.Tensor) -> torch.Tensor:
    return self.modular_metric.compute_inverse_metric(z)

self.G = _G_impl
self.G_inv = _Ginv_impl
```

**Implémentation de `compute_metric`** (metric_tensor.py, lignes 578-604):
```python
def compute_metric(self, z: torch.Tensor) -> torch.Tensor:
    """Compute metric tensor G(z) = [G^{-1}(z)]^{-1}."""
    if self.trainable:
        # Trainable: network outputs G directly
        G = self.metric_net(z)
        # ... enforce SPD ...
        return G
    else:
        # Non-trainable: compute G_inv first, then invert
        G_inv, G = self._compute_precision_components(z, return_metric=True)
        return G  # ← Returns the inverted G_inv
```

**Dans `_compute_precision_components`** (lignes 536-543):
```python
if return_metric:
    G_metric64 = _robust_inverse_from_cholesky(G_inv64)  # ← G = (G⁻¹)⁻¹
    G_metric = G_metric64.to(target_dtype)
    # ...
```

**Conclusion**: ✅ **CORRECT**
- `model.G(z)` existe et retourne correctement **G = (G⁻¹)⁻¹**, la métrique (pas la précision)
- `model.G_inv(z)` retourne correctement **G⁻¹**, la précision interpolée
- Le sampling de momentum ρ ~ N(0, G) utilise donc la bonne matrice
- **Aucun problème ici**

---

### ⚠️ PRÉOCCUPATION #2: Nom ambigu `_quad_with_G` (loss_manager.py)

**Fichier**: `loss_manager.py`  
**Ligne**: 697  
**Problème**: Nom de fonction ambigu

La fonction calcule spécifiquement **zᵀGz** (forme quadratique avec la **métrique** G), pas zᵀG⁻¹z.

**Recommandation**: Renommer en `_quad_with_metric` pour clarté:
```python
def _quad_with_metric(self, z, G_or_Ginv, rep, *, jitter=1e-6):
    """
    Compute zᵀ G(z) z given either G or G⁻¹.
    
    Note: This computes the quadratic form with the METRIC G,
    not with the precision G⁻¹.
    """
```

**Impact**: Aucun impact fonctionnel, uniquement clarté du code.

---

## PARTIE 4: RECOMMANDATIONS DE COMMENTAIRES

### 4.1 Variables ambiguës nécessitant des commentaires

**Dans `loss_manager.py`**:
```python
# Line 371: _evaluate_metric
def _evaluate_metric(...):
    # Returns: (tensor, representation_label)
    # representation_label is 'g' if tensor is metric G(z)
    #                       or 'ginv' if tensor is precision G⁻¹(z)
    pass

# Line 585: _half_logdet_volume
def _half_logdet_volume(self, G_or_Ginv, rep, *, jitter=1e-6):
    """
    Return +½ log det G^{-1} regardless of representation supplied.
    
    Args:
        G_or_Ginv: Either metric G (if rep='g') or precision G⁻¹ (if rep='ginv')
        rep: 'g' for metric, 'ginv' for precision
    """
```

**Dans `riemannian_rhmc_posterior.py`**:
```python
# Line 843: _get_inverse_metric
def _get_inverse_metric(self, pts):
    """Fetch G^{-1}(pts) - the PRECISION matrix (not metric)"""
    # G_inv represents the precision matrix G⁻¹, which is the
    # interpolated inverse covariance from Stage B
    pass

# Line 652: _make_covariance
def _make_covariance(self, G_inv, alpha):
    """
    Build posterior covariance Σ_μ = α·G⁻¹(μ) + ε·I
    
    Args:
        G_inv: Precision matrix G⁻¹(μ) (inverse of metric G)
        alpha: Scaling factor
    
    Returns:
        Sigma: Posterior covariance matrix
    """
```

---

## PARTIE 5: CHECKLIST FINALE

| Vérification | Fichier | Fonction | Statut |
|--------------|---------|----------|--------|
| ✅ Prior: p ∝ √det(G⁻¹) | riemannian_rhmc_posterior.py | `_compute_log_prior` | CORRECT |
| ✅ Posterior: Σ = α·G⁻¹+ε·I | riemannian_rhmc_posterior.py | `_make_covariance` | CORRECT |
| ✅ Potential: ∇U = -½∇log\|G⁻¹\| | riemannian_rhmc_posterior.py | `_compute_potential_gradient` | CORRECT |
| ✅ Volume term: +½log\|G⁻¹\| | metric_utils.py / loss_manager.py | `half_logdet_volume` | CORRECT |
| ✅ Pushforward: G'⁻¹ = JG⁻¹Jᵀ | loss_manager.py | `_pushforward_metric_via_flows` | CORRECT |
| ✅ Metric eval: retourne (tensor, 'g'/'ginv') | loss_manager.py | `_evaluate_metric` | CORRECT |
| 🔴 Kinetic density signe | loss_manager.py | `_log_kinetic_density` | **ERREUR** |
| ✅ Momentum sampling | riemannian_rhmc_posterior.py | `_sample_momentum` | **CORRECT** |
| ⚠️ Nom ambigu | loss_manager.py | `_quad_with_G` | **RENOMMER** |

---

## CONCLUSION

### Cohérence Globale: ✅ LARGEMENT CORRECTE

Le code est **globalement cohérent** dans son utilisation de G et G⁻¹, avec la convention que:
- G⁻¹ = précision (ce que Stage B fournit via interpolation)
- p(z) ∝ √det(G⁻¹(z)) (prior volumique uniforme sur la variété)
- Σ_μ = α·G⁻¹(μ) + ε·I (covariance du posterior)

### Problèmes Identifiés

1. **🔴 ERREUR CRITIQUE** (mais sans impact pratique actuel):
   - `_log_kinetic_density`: Signe incorrect pour le terme volumique
   - **Fix requis** pour cohérence mathématique
   - **Pas d'impact** car `rhmc_kl_jacobian=False` actuellement

2. **⚠️ AMÉLIORATION MINEURE**:
   - `_quad_with_G`: Nom ambigu, devrait être `_quad_with_metric` pour clarté

### Corrections Proposées

Voir les sections individuelles ci-dessus pour les corrections détaillées de code.

### Impact sur la KL Négative

**Aucun de ces problèmes n'explique la KL négative actuelle**, car:
1. L'erreur dans `_log_kinetic_density` n'affecte que le mode Jacobian (désactivé)
2. Tous les composants critiques (prior, posterior, pushforward, momentum sampling) sont corrects

La KL négative vient de **Σ_μ trop anisotrope**, comme analysé précédemment.

---

## ANNEXE: SNIPPET DE CODE POUR CORRECTIONS

### Correction de `_log_kinetic_density`

**Fichier**: `src/rlvae/models/components/loss_manager.py`  
**Lignes**: 825-867

```python
def _log_kinetic_density(
    self,
    model: Any,
    z: torch.Tensor,
    rho: torch.Tensor,
    *,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """
    log π_kin(ρ|z) for an RHMC kinetic energy with G(z).
    
    π_kin(ρ|z) = N(ρ; 0, G(z)) 
             ∝ |G(z)|^{1/2} exp(-½ ρᵀ G^{-1}(z) ρ)
    
    so log π_kin(ρ|z) = +½ log|G(z)| - ½ ρᵀ G^{-1}(z) ρ - const
                      = -½ log|G⁻¹(z)| - ½ ρᵀ G^{-1}(z) ρ - const
    """
    # G or G^{-1}, plus its representation tag
    G_or_Ginv, rep = self._evaluate_metric(z, model, None, with_rep=True)
    if G_or_Ginv is None or rep is None:
        # Fallback: standard kinetic with identity metric
        quad = torch.sum(rho ** 2, dim=-1)
        const = 0.5 * z.shape[-1] * math.log(2 * math.pi)
        return (-0.5 * quad - const).to(rho.dtype)
    
    rep = rep.lower()
    const = 0.5 * z.shape[-1] * math.log(2 * math.pi)
    
    if rep == "ginv":
        # We have G^{-1}, compute quadratic directly
        rho32 = rho.float()
        G_or_Ginv32 = G_or_Ginv.float() if G_or_Ginv.dtype in (torch.float16, torch.bfloat16) else G_or_Ginv
        quad = torch.einsum('bi,bij,bj->b', rho32, G_or_Ginv32, rho32)
        
        # FIX: Kinetic density needs -½log|G⁻¹| = +½log|G|
        # half_logdet_volume(G_inv, 'ginv') returns +½log|G⁻¹|
        # We need to negate it to get -½log|G⁻¹|
        half_logdet_ginv = self._half_logdet_volume(G_or_Ginv, 'ginv', jitter=jitter)
        
        # log π_kin = -½ρᵀG⁻¹ρ - ½log|G⁻¹| - const
        return (-0.5 * quad - half_logdet_ginv - const).to(rho.dtype)
    
    elif rep == "g":
        # We have G, need to invert for quadratic
        G_or_Ginv32 = G_or_Ginv.float() if G_or_Ginv.dtype in (torch.float16, torch.bfloat16) else G_or_Ginv
        try:
            G_inv = torch.linalg.inv(G_or_Ginv32)
        except RuntimeError:
            chol, _ = self._cholesky_spd(G_or_Ginv32, jitter=jitter)
            G_inv = torch.cholesky_inverse(chol)
        
        rho32 = rho.float()
        quad = torch.einsum('bi,bij,bj->b', rho32, G_inv, rho32)
        
        # half_logdet_volume(G, 'g') returns -½log|G| = +½log|G⁻¹|
        # For kinetic density, we need +½log|G| = -½log|G⁻¹|
        # So we can directly use -½log|G| from the function (with opposite sign interpretation)
        half_logdet_g_term = self._half_logdet_volume(G_or_Ginv, 'g', jitter=jitter)
        
        # Note: half_logdet_volume(G, 'g') = -½log|G|, but we want +½log|G|
        # So we negate: +½log|G| = -(-½log|G|) = -half_logdet_g_term
        # log π_kin = -½ρᵀG⁻¹ρ + ½log|G| - const
        return (-0.5 * quad - half_logdet_g_term - const).to(rho.dtype)
    
    else:
        raise ValueError(f"Unknown metric representation: {rep}")
```

**Justification mathématique**:
```
π_kin(ρ|z) = (2π)^{-d/2} |G(z)|^{1/2} exp(-½ρᵀG⁻¹(z)ρ)

log π_kin = -½d·log(2π) + ½log|G(z)| - ½ρᵀG⁻¹(z)ρ
          = -½d·log(2π) - ½log|G⁻¹(z)| - ½ρᵀG⁻¹(z)ρ
          
Si rep='ginv':
    half_logdet_volume(G_inv, 'ginv') retourne +½log|G⁻¹|
    Donc: log π_kin = -½quad - half_logdet_ginv - const ✓

Si rep='g':
    half_logdet_volume(G, 'g') retourne -½log|G| = +½log|G⁻¹|
    Donc: log π_kin = -½quad - half_logdet_g_term - const ✓
```

---

## ADDENDUM: Erreur de Typo dans Diagnostic

**Fichier**: `riemannian_rhmc_posterior.py`  
**Ligne**: 1105  
**Erreur**: Typo `model.Ginv()` au lieu de `model.G_inv()`

**Code original**:
```python
G_inv_raw = self._ctx['model'].Ginv(mu)  # ❌ TYPO
```

**Corrigé**:
```python
G_inv_raw = self._ctx['model'].G_inv(mu)  # ✅ CORRECT
```

**Impact**: ⚠️ **AUCUN IMPACT SUR LA FONCTIONNALITÉ**
- C'est uniquement dans un bloc de diagnostic (conditionné par `RLVAE_DEBUG=1`)
- Encapsulé dans un `try-except` qui capture l'exception silencieusement
- Le vrai calcul de Σ utilise correctement `_get_inverse_metric()` (ligne 1090)
- Cette comparaison diagnostique est **après** le calcul fonctionnel

**Correction appliquée**: ✅ Ligne 1105 corrigée

---

**Audit complété le**: 27 Octobre 2025  
**Fichiers audités**: `loss_manager.py`, `riemannian_rhmc_posterior.py`, `metric_utils.py`, `modular_rlvae.py`, `metric_tensor.py`  
**Statut global**: ✅ COHÉRENT avec corrections mineures appliquées


