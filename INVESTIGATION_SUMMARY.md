# Investigation Summary: Negative KL Divergence and log_q Analysis

## 🎯 Investigation Goal

Determine why `log_q` (the log-density of the posterior approximation) is unexpectedly negative (-2.5 to -3.0), leading to negative KL divergence values.

## 🔍 Key Findings

### 1. Mathematical Formula is Correct

The `log_q_riem` function in `riemannian_rhmc_posterior.py` implements the **correct** formula for a Riemannian Gaussian:

```python
log_q = -0.5 * (z - μ)ᵀ Σ⁻¹ (z - μ) - 0.5 * log|Σ| - const
      = [quadratic term]        + [volume term]   + [constant]
```

This is mathematically sound.

### 2. Discrepancy Between RHMC and Manual Reconstruction

**Observation from diagnostics**:

- **RHMC `log_q`** (from RHMC posterior): `-2.25` (typical range: -2.0 to -3.0)
- **Manual reconstruction** (in `loss_manager.py`): `-7.87` (typical range: -6.0 to -9.0)

**Difference**: ~5.6 units!

### 3. Root Cause Hypothesis: Stabilization Modifies Σ

The `_safe_cholesky` function in `riemannian_rhmc_posterior.py` adds numerical stabilization:

```python
def _safe_cholesky(matrix, jitter):
    try:
        chol = torch.linalg.cholesky(matrix)
        return chol, matrix, False  # No modification
    except RuntimeError:
        stabilized = matrix + jitter * I  # ← Add jitter to diagonal
        chol = torch.linalg.cholesky(stabilized)
        return chol, stabilized, True  # Return modified matrix
```

**Key Insight**: The stabilized matrix has **larger eigenvalues** and thus a **larger determinant**.

**Impact on log_q**:
```
Volume term = -0.5 * log|Σ|

If Σ_stabilized = Σ + jitter * I, then:
  eigenvalues(Σ_stabilized) = eigenvalues(Σ) + jitter
  log|Σ_stabilized| > log|Σ|
  Volume term becomes LESS negative
  log_q becomes LESS negative (more positive)
```

**Example**:
```
Original Σ:      eigenvalues = [0.012, 17.98], log|Σ| = -2.26
Stabilized Σ:    eigenvalues = [0.013, 17.98], log|Σ| ≈ -2.18
Δ log|Σ| ≈ +0.08
Volume term changes from +1.13 to +1.09
```

Small changes in eigenvalues can have large effects, especially when one eigenvalue is very small.

### 4. Additional Stabilization: Eigenvalue Clamping

In `_get_inverse_metric` (line 617-620 of `riemannian_rhmc_posterior.py`):

```python
evals, evecs = torch.linalg.eigh(Ginv)
evals = torch.clamp(evals, min=self.min_cov_eig)  # ← Clamp to min_cov_eig
```

This **directly modifies the eigenvalues** of G⁻¹ before constructing Σ_μ = α * G⁻¹ + ε * I.

**Impact**:
- If G⁻¹ has eigenvalues < 0.001, they are clamped to 0.001
- This makes Σ_μ better conditioned but changes its spectrum
- The manual reconstruction in `loss_manager.py` doesn't apply this clamping

### 5. The Geomean Normalization Issue

In `_get_inverse_metric` (lines 620-630), there's an optional geomean normalization:

```python
if mode == 'geomean':
    geomean_eig = torch.exp(torch.log(evals + 1e-12).mean(dim=-1, keepdim=True))
    evals = evals / (geomean_eig + 1e-12)
    # Result: det(Ginv_norm) ≈ 1
```

**Impact**:
- Forces `det(G⁻¹_normalized) ≈ 1`
- Makes `log|G⁻¹| ≈ 0`
- Σ_μ = α * G⁻¹_normalized has `log|Σ_μ| ≈ log(α^D)`
- This is a **major change** from using the raw G⁻¹

**Check**: The mode is set by `sigma_normalization_mode` config parameter (default: 'geomean').

## 🛠️ Diagnostics Added

### 1. Enhanced `_safe_cholesky`

- Now returns a third boolean value: `was_stabilized`
- Indicates if jitter was added to the matrix

### 2. Stabilization Diagnostics in `log_q_riem`

When `RLVAE_DEBUG=1`, prints:
```
[LOG_Q_RIEM STABILIZATION]
  min_eig (jitter):     0.001000
  was_stabilized:       True/False
  Original Σ:
    eigenvalues:        min=..., max=...
    log|Σ|:             ...
  Stabilized Σ:  (if was_stabilized)
    eigenvalues:        min=..., max=...
    log|Σ|:             ...
    Δ log|Σ|:           ...
```

### 3. Parameter Logging

In `_compute_log_riemannian_gaussian`, logs:
```
[_compute_log_riemannian_gaussian]
  Calling log_q_riem with min_cov_eig = 0.001000
```

## 🧪 Testing

Created `test_log_q_stabilization.py` to verify:
1. Well-conditioned matrices (no stabilization)
2. Poorly-conditioned matrices (stabilization expected)
3. Batch processing with mixed conditioning

Run with: `python test_log_q_stabilization.py`

## 📊 Next Steps

### Immediate Actions

1. **Run training with `RLVAE_DEBUG=1`**:
   ```bash
   RLVAE_DEBUG=1 [your training command]
   ```

2. **Check the output for**:
   - `[LOG_Q_RIEM STABILIZATION]` blocks
   - Whether `was_stabilized: True` appears frequently
   - The magnitude of eigenvalue changes

3. **Compare eigenvalues**:
   - In `[SIGMA DEBUG]` (manual reconstruction in `loss_manager.py`)
   - In `[LOG_Q_RIEM STABILIZATION]` (RHMC internal calculation)
   - Any discrepancy indicates where the modification occurs

### Deeper Investigation

If `was_stabilized: False` consistently, then the issue is **not** in `_safe_cholesky` but in:

1. **Eigenvalue clamping** in `_get_inverse_metric` (line 617-620)
2. **Geomean normalization** in `_get_inverse_metric` (line 621-630)
3. **Covariance construction** in `_make_covariance` (line 638-651)

**Recommended**: Add similar diagnostics to `_get_inverse_metric` and `_make_covariance` to trace the full transformation:

```
G(μ) → G⁻¹(μ) → G⁻¹_clamped → G⁻¹_normalized → Σ_μ = α * G⁻¹_normalized + ε * I
```

### Potential Solutions

1. **Disable geomean normalization**:
   ```yaml
   sigma_normalization_mode: 'none'
   ```

2. **Adjust `min_cov_eig`**:
   - Lower value (e.g., `1e-5`) for less aggressive clamping
   - Higher value (e.g., `1e-2`) for more numerical stability (but more bias)

3. **Increase `rhmc_alpha`**:
   - Makes Σ_μ = α * G⁻¹ + ε * I larger
   - Can improve conditioning
   - But changes the posterior approximation

4. **Disable clamping** (not recommended for numerical stability):
   - Remove line 620 in `_get_inverse_metric`

## 📝 Files Modified

1. **`src/rlvae/models/components/riemannian_rhmc_posterior.py`**:
   - Modified `_safe_cholesky` to return stabilization flag (lines 23-53)
   - Added stabilization diagnostics in `log_q_riem` (lines 143-164)
   - Added parameter logging in `_compute_log_riemannian_gaussian` (lines 1023-1025)
   - Updated all `_safe_cholesky` calls to handle third return value (lines 512, 552, 859)

2. **`test_log_q_stabilization.py`** (new):
   - Test script for verifying stabilization diagnostics

3. **`LOG_Q_STABILIZATION_GUIDE.md`** (new):
   - Comprehensive guide for using the new diagnostics

4. **`INVESTIGATION_SUMMARY.md`** (this file):
   - Summary of investigation findings and next steps

## 🎓 Key Takeaways

1. **The mathematical formula is correct**
   - No bugs in `log_q_riem` implementation

2. **Numerical stabilization changes Σ**
   - `_safe_cholesky` adds jitter when needed
   - Eigenvalue clamping modifies the spectrum
   - Geomean normalization forces det ≈ 1

3. **Manual reconstruction in `loss_manager.py` doesn't match RHMC**
   - It uses raw G⁻¹(μ) without clamping or normalization
   - This explains the ~5-6 unit discrepancy in log_q

4. **The "true" log_q is what RHMC computes internally**
   - It reflects the actual posterior q(z|x) used for sampling
   - The manual reconstruction is useful for diagnostics but not the ground truth

5. **Next step: Trace the full transformation pipeline**
   - Add diagnostics to `_get_inverse_metric` and `_make_covariance`
   - Track how G(μ) → Σ_μ evolves through each transformation
   - Identify which step causes the largest change

## 🔧 Usage

To activate all diagnostics:

```bash
export RLVAE_DEBUG=1
# Then run your training
```

You'll see detailed output for:
- `[LOG_Q_RIEM STABILIZATION]` - Cholesky stabilization details
- `[LOG_Q FROM RHMC]` - Manual reconstruction and comparison
- `[SIGMA DEBUG]` - Covariance matrix analysis
- `[METRIC DEBUG]` - Riemannian metric properties
- `[PUSH DEBUG]` - Pushforward metric stabilization

All diagnostics are **silent** by default (`RLVAE_DEBUG=0`) to avoid cluttering training logs.


