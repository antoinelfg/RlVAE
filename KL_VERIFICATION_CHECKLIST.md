# Checklist de Vérification Complète du Calcul de KL
## Formulation B : KL = log q_base - log p_T + Δ_kin - Δ_vol

**Date**: 27 Octobre 2025  
**Objectif**: Vérifier systématiquement chaque composant du calcul de KL pour identifier les problèmes numériques ou d'implémentation

---

## FORMULE ET CONVENTION

```
KL = log q(z_base | μ, Σ_μ) - log p_T(z_T) + Δ_kin - Δ_vol

où:
- z_base = z₀ (échantillon initial du posterior)
- z_T = z_S (échantillon après flows)
- log q_base : densité du posterior Gaussien Riemannien
- log p_T : densité du prior après pushforward via flows
- Δ_kin : correction cinétique RHMC
- Δ_vol : correction volumique RHMC (=0 dans notre cas)
```

---

## 1. VÉRIFICATION DE log q_base

### 1.1 Inputs de Σ_μ

**À vérifier** :
```
[CONFIG OVERRIDE] alpha_override=??, eps_override=??

[_make_covariance TARGET RADIUS]
  input alpha:           ??
  eps_reg:               ??
  
[_compute_log_riemannian_gaussian BEFORE log_q_riem]
  Sigma eigenvalues:     min=??, max=??
  Sigma trace:           ??
  log|Sigma|:            ??
```

**Critères de validation** :
- ✅ `alpha_override` = 0.1 (nouvelle valeur)
- ✅ `eps_override` = 1e-3
- ✅ Eigenvalues **strictement positives** (min > 0)
- ✅ Condition number raisonnable (max/min < 10000)
- ✅ `log|Sigma|` fini

**Questions** :
- Ratio eigenvalues (max/min) = ? (attendu: élevé car anisotrope)
- Trace(Σ) = ? (attendu: ~0.95 avec alpha=0.1, trace(G⁻¹)~9.5)

---

### 1.2 Stabilisation de Σ_μ

**À vérifier** :
```
[LOG_Q_RIEM STABILIZATION]
  min_eig (jitter):     ??
  was_stabilized:       ?? (False attendu)
```

**Critères de validation** :
- ✅ `was_stabilized = False` (pas de jitter additionnel)
- ⚠️ Si `True` : Vérifier que le jitter est petit et acceptable

---

### 1.3 Output log q

**À vérifier** :
```
[LOG_Q_RIEM DECOMPOSITION]
  Quadratic term: mean=??, min=??, max=??
  Volume term:    mean=??
  Constant term:  ??
  ||z - μ||:      mean=??
  Σ eigenvalues:  min=??, max=??
  
[LOG_Q FROM RHMC]
  Total log_q:      ?? (range: [??, ??])
```

**Critères de validation** :
- ✅ `log_q` **fini** (pas de NaN/inf)
- ✅ Somme des termes : `Quadratic + Volume + Constant ≈ Total log_q`
- ✅ Quadratic term négatif (forme quadratique)
- ✅ Volume term cohérent avec `log|Σ|` : Volume = -0.5 × log|Σ|

**Calculs de validation** :
```
Volume term attendu = -0.5 × log|Σ|
Quadratic attendu ≈ -0.5 × ||z-μ||² / (moyenne eigenvalues de Σ)
```

---

## 2. VÉRIFICATION DE log p_T (Pushforward)

### 2.1 Stabilité du Jacobien

**À vérifier** :
```
[DEBUG] --- FLOW JACOBIAN SUMMARY ---
[DEBUG FLOW 0] min=??, max=??, mean=??, std=??
[DEBUG FLOW 1] min=??, max=??, mean=??, std=??
...
[DEBUG FLOW 6] min=??, max=??, mean=??, std=??
```

**Critères de validation** :
- ✅ Tous les log-déterminants **finis**
- ✅ Pas de valeurs extrêmes (|logdet| < 1.0 typiquement pour flows IAF)

---

### 2.2 Stabilité de la Métrique Transportée

**À vérifier** :
```
[PUSH DEBUG] transported metric stats:
  eig_min=??, eig_max=??, 
  cond(G')=??, cond(G'^-1)=??,
  min_sv(J)=??, ||G'·G'^-1-I||_F=??
  
[DEBUG] G_pushforward has NaN: ??
[DEBUG] G_pushforward has inf: ??
[DEBUG] G_pushforward eigenvalues - min: ??, max: ??
[DEBUG] G_pushforward condition number: ??
```

**Critères de validation** :
- ✅ `cond(G')` < 10000 (sinon fallback vers Formulation A)
- ✅ `NaN/inf` = False
- ✅ Eigenvalues positives
- ✅ `||G'·G'^-1-I||_F` < 0.1 (vérification G' × (G')⁻¹ ≈ I)

**Question critique** :
- Y a-t-il eu **fallback vers Formulation A** ? 
  - Si oui, `log_p_prime_zF` utilisera `half_logdet_target_ginv` au lieu du pushforward

---

### 2.3 Output log p_T

**À vérifier** :
```
[DEBUG] half_logdet_push_g mean: ??
[DEBUG] half_logdet_push_ginv mean: ??
[DEBUG] log_p_prime_zF mean: ??, min: ??, max: ??
[DEBUG] log_p_prime_zF has NaN: ??, has inf: ??
[DEBUG] Using rep_push='??', log_p_prime_zF mean: ??
```

**Critères de validation** :
- ✅ `log_p_prime_zF` **fini**
- ✅ `rep_push = 'ginv'` (utilise G⁻¹ transporté)
- ✅ `half_logdet_push_g ≈ half_logdet_push_ginv` (à un signe près dans la formule)

**Calcul de validation** :
```
log p_T(z_T) = +0.5 × log|G⁻¹(z_T)| + const

Si rep='ginv' : log_p_prime_zF = half_logdet_push_ginv
```

---

## 3. VÉRIFICATION DE Δ_kin

### 3.1 Stabilité des Métriques Intermédiaires

**À vérifier** :
```
[METRIC DEBUG] G(z0): 
  rep=??, eig_min=??, eig_max=??, cond(G)=??,
  log|G|=??, log|G⁻¹|=??, ||G·G⁻¹-I||_F=??
  
[METRIC DEBUG] G(zS):
  rep=??, eig_min=??, eig_max=??, cond(G)=??,
  log|G|=??, log|G⁻¹|=??, ||G·G⁻¹-I||_F=??
```

**Critères de validation** :
- ✅ `rep = 'ginv'` (G⁻¹ fourni)
- ✅ Eigenvalues positives
- ✅ `cond(G)` et `cond(G⁻¹)` raisonnables (< 10000)
- ✅ `||G·G⁻¹-I||_F` < 0.01 (vérification inversion correcte)
- ✅ `log|G|` et `log|G⁻¹|` finis et de signes opposés

---

### 3.2 Output Δ_kin

**À vérifier** :
```
[DEBUG] KL CALCULATION - delta_kin mean: ??

[KL DEBUG] kinetic density:
  start_mean=??, end_mean=??, 
  diff_mean=??, delta_kin_residual=??
```

**Critères de validation** :
- ✅ `delta_kin` **fini**
- ✅ Magnitude raisonnable (typiquement petit, |Δ_kin| < 1.0)
- ✅ `delta_kin_residual` petit (< 1e-3) indique cohérence du calcul

**Calcul de validation** :
```
Δ_kin = log π_kin(ρ_base|z_base) - log π_kin(ρ_S|z_S)
      = [+0.5 log|G(z_base)| - 0.5 ρ_base^T G⁻¹(z_base) ρ_base] 
        - [+0.5 log|G(z_S)| - 0.5 ρ_S^T G⁻¹(z_S) ρ_S]

diff_mean ≈ delta_kin (si bien calculé)
```

---

## 4. VÉRIFICATION DE Δ_vol

**À vérifier** :
```
[DEBUG] KL CALCULATION - delta_vol mean: ??

[KL DEBUG] flow stats: delta_vol mean=??
```

**Critères de validation** :
- ✅ `delta_vol = 0.0` (intégrateur traité comme volume-preserving)
- ⚠️ Si non-zéro, vérifier la stabilité du calcul

---

## 5. VÉRIFICATION DE L'ASSEMBLAGE FINAL

### 5.1 Termes Avant Sommation

**À vérifier** :
```
[DEBUG] INTERMEDIATE KL TERMS:
[DEBUG] - log_q has NaN: ??, has inf: ??
[DEBUG] - delta_kin has NaN: ??, has inf: ??
[DEBUG] - delta_vol has NaN: ??, has inf: ??
[DEBUG] - log_p_prime_zF has NaN: ??, has inf: ??
```

**Critères de validation** :
- ✅ **TOUS** les termes sont finis (pas de NaN/inf)

---

### 5.2 Formule et Résultat

**À vérifier** :
```
[DEBUG] FORMULATION B - log_p_prime_zF mean: ??
[DEBUG] FORMULATION B - (volume_bias_weight * log_p_prime_zF) mean: ??
[DEBUG] FORMULATION B - kl_terms mean: ??

[DEBUG] FINAL KL_TERMS CHECK:
[DEBUG] - kl_terms mean: ??, min: ??, max: ??
[DEBUG] - kl_terms has NaN: ??, has inf: ??

[DEBUG] FINAL KL LOSS: ??

[KL VALIDATION] Negative KL detected (??)
[KL VALIDATION] log_q mean=??, volume term mean=??
```

**Critères de validation** :
- ✅ `kl_terms` **fini**
- ✅ `volume_bias_weight = 1.0`

**Calcul de validation MANUEL** :
```
KL_calculé = log_q_mean - (volume_bias_weight × log_p_prime_zF_mean) + delta_kin_mean - delta_vol_mean
           = log_q_mean - log_p_prime_zF_mean + delta_kin_mean - delta_vol_mean

Comparer avec kl_terms_mean reporté
```

---

## 6. VALIDATIONS SUPPLÉMENTAIRES

### 6.1 Comparaison Formulation A vs B

**À vérifier si disponible** :
- KL via Formulation A (quand pushforward réussit)
- KL via Formulation B
- Différence absolue < 0.1

### 6.2 Cohérence Géométrique

**À vérifier** :
```
[KL DEBUG] latent norms:
  ||mu|| mean=??, ||z0-mu|| mean=??, ||zS-mu|| mean=??, ||zS-z0|| mean=??
  
[KL DEBUG] flow stats:
  log_q mean=??, sum_logdet_flow mean=??, delta_kin mean=??, delta_vol mean=??
```

**Questions** :
- `||zS-z0||` petit ? (flows ne devraient pas trop déplacer)
- `sum_logdet_flow` cohérent avec les Jacobiens individuels ?

---

## TEMPLATE DE RAPPORT

### Valeurs du Terminal

```
=== 1. LOG Q BASE ===
[CONFIG OVERRIDE] alpha_override=??, eps_override=??
[_make_covariance] alpha=??, eps=??
[Sigma] eigenvalues: min=??, max=??, trace=??, log|Σ|=??
[LOG_Q_RIEM] was_stabilized=??
[LOG_Q_RIEM DECOMPOSITION] Quad=??, Vol=??, Const=??, ||z-μ||=??
[LOG_Q FROM RHMC] Total log_q=?? (range: [??, ??])

=== 2. LOG P_T (PUSHFORWARD) ===
[PUSH DEBUG] cond(G')=??, cond(G'^-1)=??, min_sv(J)=??, ||G'·G'^-1-I||=??
[DEBUG] G_pushforward: NaN=??, inf=??, eig_min=??, eig_max=??, cond=??
[DEBUG] half_logdet_push_ginv mean=??
[DEBUG] log_p_prime_zF mean=??, min=??, max=??

=== 3. DELTA_KIN ===
[METRIC DEBUG] G(z0): eig_min=??, eig_max=??, cond=??, log|G⁻¹|=??, ||G·G⁻¹-I||=??
[METRIC DEBUG] G(zS): eig_min=??, eig_max=??, cond=??, log|G⁻¹|=??, ||G·G⁻¹-I||=??
[DEBUG] delta_kin mean=??
[KL DEBUG] kinetic: start=??, end=??, diff=??, residual=??

=== 4. DELTA_VOL ===
[DEBUG] delta_vol mean=??

=== 5. ASSEMBLAGE FINAL ===
[DEBUG] INTERMEDIATE KL TERMS: all finite=??
[DEBUG] kl_terms mean=??, min=??, max=??
[DEBUG] FINAL KL LOSS=??

=== CALCUL MANUEL ===
KL_manuel = log_q - log_p_T + delta_kin - delta_vol
          = ?? - ?? + ?? - ??
          = ??
          
Comparaison: KL_reporté=??, KL_manuel=??, Δ=??
```

---

## INSTRUCTIONS

1. **Copiez les valeurs de votre terminal** dans le template ci-dessus
2. **Remplacez tous les `??`** par les valeurs réelles
3. Je vérifierai **systématiquement** chaque point selon les critères de validation
4. J'identifierai **précisément** où se trouve le problème (si présent)

---

**Prêt à analyser les valeurs !** 🔍

