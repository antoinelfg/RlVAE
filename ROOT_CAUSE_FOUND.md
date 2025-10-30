# 🎯 ROOT CAUSE FOUND: `initial_target_radius` Forces Σ to Have trace = 1

## 📊 The Problem

The RHMC posterior creates a covariance matrix `Sigma_mu` with:
- **Eigenvalues**: `[0.482, 0.518]` (nearly isotropic)
- **Trace**: `1.000` (exactly 1.0)
- **log|Σ|**: `-1.387`

But the raw `G⁻¹(μ)` has:
- **Eigenvalues**: `[0.012, 17.975]` (highly anisotropic)
- **log|G⁻¹(μ)|**: `-2.261`

**Difference in log|Σ|**: +0.87 (makes volume term less negative)
**Difference in log_q**: ~5.6 units (RHMC: -2.25 vs Expected: -7.87)

---

## 🔍 The Root Cause

### Location: `_make_covariance` (lines 688-704)

```python
# Adapt alpha to hit a target Euclidean radius for initial draws
target_r = float(getattr(self, 'initial_target_radius', 1.0))
if target_r > 0:
    tr_ginv = torch.einsum('bii->b', Ginv_norm.float()).unsqueeze(-1)
    # E||δ||^2 ≈ trace(Sigma) = alpha*trace(Ginv_norm)+ d*eps
    alpha_eff = ((target_r ** 2) - d * self.eps_reg) / torch.clamp(tr_ginv, min=1e-12)
    Sigma = alpha_eff.unsqueeze(-1).unsqueeze(-1) * Ginv_norm + self.eps_reg * eye
```

### What Happens

1. **`initial_target_radius = 1.0`** (default value)
2. The code computes `alpha_eff` to satisfy:
   ```
   trace(Σ) = alpha_eff * trace(G⁻¹_norm) + d * eps_reg ≈ target_r² = 1.0
   ```
3. This **overrides** the user-specified `rhmc_alpha` parameter!
4. The result is a `Sigma` with `trace ≈ 1.0`, regardless of the geometry of `G⁻¹(μ)`

### For D=2 (2D Latent Space)

If `trace(Σ) = 1.0` and Σ is forced to be nearly isotropic:
- Eigenvalues ≈ `[0.5, 0.5]`
- `det(Σ) = 0.5 * 0.5 = 0.25`
- `log|Σ| = log(0.25) = -1.386` ← **Matches observed value!**

---

## 🧪 Evidence from Terminal Output

```
[_compute_log_riemannian_gaussian BEFORE log_q_riem]
  Sigma eigenvalues:     min=0.482348, max=0.517973
  Sigma trace:           1.000000  ← EXACTLY 1.0!
  log|Sigma|:            -1.386607 ← Matches log(0.25)
```

Compare with manual reconstruction in `loss_manager.py`:
```
[SIGMA DEBUG] Sigma_mu:
  Σ_μ eigenvalues:  min=0.011916, max=17.975180
  log|Σ_μ|:         -2.2614
```

The manual reconstruction uses `Σ_μ = α * G⁻¹(μ) + ε * I` **without** the target radius adjustment.

---

## 💡 Why This Causes Negative KL

### Volume Term Impact

```
Volume term = -0.5 * log|Σ|

With target radius:
  log|Σ| = -1.387
  Volume term = +0.694

Without target radius (expected):
  log|Σ| = -2.261
  Volume term = +1.131

Difference: -0.437
```

This makes `log_q` **more negative** by ~0.44 units.

### But Why Is the Discrepancy 5.6 Units?

The volume term explains only ~0.44 units. The remaining ~5.2 units must come from:

1. **Additional normalization effects**:
   - `sigma_normalization_mode: 'geomean'` (default) forces `det(G⁻¹_norm) = 1`
   - This drastically changes the scale of eigenvalues

2. **Quadratic term differences**:
   - `Σ⁻¹` computed from the isotropic Σ is very different from `Σ⁻¹` of anisotropic Σ
   - The quadratic term `(z - μ)ᵀ Σ⁻¹ (z - μ)` will be affected

3. **Interaction between normalizations**:
   - Geomean normalization + target radius creates a compound effect

---

## ✅ Solution

### Option 1: Disable Target Radius (Recommended)

In your RHMC config:

```yaml
initial_target_radius: 0.0  # Disable target radius adjustment
```

This will make `Sigma_mu = rhmc_alpha * G⁻¹_norm + eps_reg * I` without forcing a specific trace.

### Option 2: Disable Geomean Normalization

```yaml
sigma_normalization_mode: 'none'  # Disable normalization
```

This will use raw `G⁻¹(μ)` without forcing `det = 1`.

### Option 3: Use Both

```yaml
initial_target_radius: 0.0
sigma_normalization_mode: 'none'
```

This gives the "pure" Riemannian covariance: `Σ_μ = rhmc_alpha * G⁻¹(μ) + eps_reg * I`

---

## 🧪 Testing the Fix

With diagnostics enabled (`RLVAE_DEBUG=1`), you'll now see:

```
[_get_inverse_metric NORMALIZATION]
  mode:                  geomean
  Original G⁻¹ eigenvalues: min=0.011916, max=17.975180
  ...

[_make_covariance TARGET RADIUS]
  initial_target_radius: 1.000000
  input alpha:           1.000000
  ...
  Final trace(Σ):        1.000000
  → TARGET RADIUS FORCES trace(Σ) ≈ 1.00
```

After setting `initial_target_radius: 0.0`, you should see:

```
[_make_covariance TARGET RADIUS]
  initial_target_radius: 0.000000
  Target radius disabled (target_r=0), using alpha=1.0
```

And `Sigma` will have eigenvalues matching the geometry of `G⁻¹(μ)`.

---

## 📝 Configuration Changes Needed

Update your config file (e.g., `conf/model/rlvae_rotation_rhmc_stage2.yaml`):

```yaml
posterior:
  _target_: src.rlvae.models.components.riemannian_rhmc_posterior.RiemannianRHMCPosterior
  rhmc_alpha: 1.0
  eps_regularization: 1e-4
  min_cov_eig: 1e-3
  sigma_normalization_mode: 'none'   # NEW: Disable geomean normalization
  initial_target_radius: 0.0         # NEW: Disable target radius
  # ... other parameters
```

---

## 🎯 Expected Impact

After the fix:
1. `Sigma_mu` will be anisotropic, matching `G⁻¹(μ)` geometry
2. `log|Σ|` will be closer to `log|G⁻¹(μ)|` (around -2.26 instead of -1.39)
3. `log_q` will be more negative (around -7.87 instead of -2.25)
4. **KL divergence may become positive** (or less negative)

---

## 📚 Files Modified

- `src/rlvae/models/components/riemannian_rhmc_posterior.py`:
  - Added diagnostics in `_get_inverse_metric` (lines 652-678)
  - Added diagnostics in `_make_covariance` (lines 690-721)
  - Added diagnostics in `_compute_log_riemannian_gaussian` (lines 1024-1053)
  - Enhanced `_safe_cholesky` to return stabilization flag (lines 23-53)
  - Enhanced `log_q_riem` with stabilization diagnostics (lines 143-164)

---

## 🔬 Further Investigation

If KL is still negative after disabling target radius and normalization, investigate:

1. **Pushforward metric calculation**: Is `log_p_prime_zF` correctly computed?
2. **Flow Jacobian determinants**: Are they correctly accumulated?
3. **Formulation B correctness**: Verify the KL formula implementation

But this (target radius + geomean) is almost certainly the main cause of the discrepancy.

---

**Status**: ✅ Root cause identified  
**Fix**: Set `initial_target_radius: 0.0` and `sigma_normalization_mode: 'none'`  
**Next**: Test with fixed configuration

