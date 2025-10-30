# Diagnostic Update Summary: Complete Σ Tracing

## 🎯 Objective

Add comprehensive diagnostics to trace the exact `Sigma` matrix used by the RHMC posterior when computing `log_q`, and compare it with the expected values from raw `G⁻¹(μ)`.

## 📝 Changes Made

### File: `src/rlvae/models/components/riemannian_rhmc_posterior.py`

#### 1. Enhanced `_safe_cholesky` (Lines 23-53)
- **Added**: Third return value `was_stabilized: bool`
- **Purpose**: Indicates whether jitter was added to stabilize the Cholesky decomposition
- **Impact**: All callers updated to handle the new return signature

#### 2. Stabilization Diagnostics in `log_q_riem` (Lines 143-164)
- **Added**: Detailed logging when `RLVAE_DEBUG=1`
- **Shows**:
  - `min_eig` (jitter value)
  - `was_stabilized` flag
  - Original Σ eigenvalues and log-determinant
  - If stabilized: modified eigenvalues and change in log|Σ|

#### 3. **NEW: Complete Σ Tracing in `_compute_log_riemannian_gaussian` (Lines 1023-1051)**
- **Added**: Comprehensive diagnostic before passing Σ to `log_q_riem`
- **Shows**:
  - Whether `Sigma` was provided or computed
  - `alpha` value (scaling parameter)
  - Σ eigenvalues, trace, and log-determinant
  - Distance `||z - μ||`
  - **Comparison with raw G⁻¹(μ)**:
    - Raw G⁻¹(μ) eigenvalues
    - Expected vs actual log|Σ|
    - Delta showing the transformation effect

## 🔍 What This Diagnostic Will Reveal

### Expected Output During Training

When you run with `RLVAE_DEBUG=1`, you'll now see:

```
[_compute_log_riemannian_gaussian BEFORE log_q_riem]
  min_cov_eig:           0.001000
  covariance_provided:   False
  alpha:                 1.0
  Sigma eigenvalues:     min=0.482533, max=0.517849
  Sigma trace:           1.000382
  log|Sigma|:            -1.386779
  ||z - μ||:             1.000600
  [Comparison with raw G⁻¹(μ)]
    G⁻¹(μ) eigenvalues:  min=0.011916, max=17.975180
    log|G⁻¹(μ)|:         -2.261400
    Expected log|Σ|:     -2.261400 (if α=1, ε≈0)
    Actual log|Σ|:       -1.386779
    Δ log|Σ|:            +0.874621  ← BIG DIFFERENCE!

[LOG_Q_RIEM STABILIZATION]
  min_eig (jitter):     0.001000
  was_stabilized:       False
  Original Σ:
    eigenvalues:        min=0.482533, max=0.517849
    log|Σ|:             -1.386779
```

### Key Insights from Output

1. **If `Δ log|Σ|` is large (> 0.5)**:
   - Σ has been significantly modified between G⁻¹(μ) and the final Σ
   - This happens in `_get_inverse_metric` or `_make_covariance`
   - Check for eigenvalue clamping or geomean normalization

2. **If Σ eigenvalues are nearly equal (e.g., `[0.48, 0.52]`)**:
   - Σ has been regularized to be nearly isotropic
   - This is **very different** from the raw G⁻¹(μ) which is anisotropic
   - Explains why `log_q` is less negative than expected

3. **If `covariance_provided: True`**:
   - Σ came from the `covariance` parameter (e.g., from RHMC sampling)
   - Not computed from G⁻¹(μ) in this call
   - Check where this covariance was created

## 🎯 Hypothesis Being Tested

Based on the terminal output you provided, we suspect:

### **Hypothesis**: The RHMC computes Σ ≈ 0.5 * I (nearly isotropic)

**Evidence**:
- `[LOG_Q_RIEM STABILIZATION]` showed eigenvalues `[0.482, 0.518]` ≈ `[0.5, 0.5]`
- Manual reconstruction showed eigenvalues `[0.012, 17.98]` (anisotropic)
- Difference in log|Σ|: `-1.387` vs `-2.261` ≈ **0.87 units**

**Implications**:
```
Volume term (RHMC):     -0.5 * (-1.387) = +0.694
Volume term (Expected): -0.5 * (-2.261) = +1.131
Difference:             -0.437

This makes log_q MORE NEGATIVE by ~0.44
```

But the actual discrepancy is **~5.6 units** (`-2.25` vs `-7.87`), so there must be **additional factors**.

## 🧪 Next Steps

1. **Run training with `RLVAE_DEBUG=1`**:
   ```bash
   RLVAE_DEBUG=1 [your training command]
   ```

2. **Look for the new diagnostic block**:
   ```
   [_compute_log_riemannian_gaussian BEFORE log_q_riem]
   ```

3. **Check the output**:
   - Is `Δ log|Σ|` large?
   - Are the Σ eigenvalues nearly equal (isotropic)?
   - Is `covariance_provided: True` or `False`?

4. **Based on findings**:
   - If Δ log|Σ| is large → investigate `_get_inverse_metric` and `_make_covariance`
   - If Σ is isotropic → check for regularization or normalization
   - If covariance is provided → trace back where it was created

## 📊 Comparison Table

| Metric | Manual Reconstruction | RHMC (Observed) | Expected if Same Σ |
|--------|----------------------|-----------------|-------------------|
| Σ eigenvalues | `[0.012, 17.98]` | `[0.482, 0.518]` | `[0.012, 17.98]` |
| log\|Σ\| | `-2.261` | `-1.387` | `-2.261` |
| Volume term | `+1.131` | `+0.694` | `+1.131` |
| log_q | `-7.87` | `-2.25` | `-7.87` |

**Discrepancy**: RHMC's `log_q` is **~5.6 units less negative** than expected.

**Volume term explains**: ~0.44 units  
**Remaining unexplained**: ~5.2 units ← **Must be in the quadratic term!**

## 🔧 Additional Diagnostics Needed

If the volume term discrepancy (~0.44) doesn't explain the full ~5.6 difference, we need to investigate the **quadratic term**:

```
Quadratic term = -0.5 * (z - μ)ᵀ Σ⁻¹ (z - μ)
```

**Potential issues**:
1. Different `z` being used (z0 vs zK vs zS)
2. Different `μ` being used
3. Σ⁻¹ computed differently (not just inverse of Σ)

**Recommendation**: Add a diagnostic in `log_q_riem` to print the quadratic term separately:

```python
quad_form = torch.einsum('bij,bij->b', diff32, sol32)
if os.environ.get("RLVAE_DEBUG", "0") == "1":
    print(f"  Quadratic term: {(-0.5 * quad_form).mean().item():.6f}")
    print(f"  Volume term:    {(-0.5 * log_det).mean().item():.6f}")
    print(f"  Constant term:  {-const:.6f}")
```

This will confirm if the discrepancy is in the quadratic term.

## 📚 Related Files

- `INVESTIGATION_SUMMARY.md`: Overall investigation findings
- `LOG_Q_STABILIZATION_GUIDE.md`: How to use stabilization diagnostics
- `NEXT_INVESTIGATION_STEPS.md`: Recommended next steps
- `tests/test_log_q_stabilization.py`: Test script for basic functionality

## ⚙️ Configuration to Check

If normalization is causing issues, check your config:

```yaml
# In riemannian_rhmc_posterior config
sigma_normalization_mode: 'geomean'  # Try 'none' to disable
min_cov_eig: 1e-3                    # Try lowering to 1e-5
rhmc_alpha: 1.0                      # Try adjusting scaling
```

---

**Status**: ✅ Diagnostics implemented and ready for testing  
**Next**: Run training with `RLVAE_DEBUG=1` and analyze output  
**Expected**: Clear identification of where Σ gets modified

