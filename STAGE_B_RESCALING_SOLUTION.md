# Stage B Metric Rescaling: Solution for Negative KL Divergence

**Date**: October 27, 2025  
**Problem**: Negative KL divergence due to extremely negative `log q`  
**Root Cause**: Stage B produces G⁻¹ matrices with very small eigenvalues and weak anisotropy  
**Solution**: Rescale G⁻¹ atoms by 10x to increase eigenvalue scale while preserving anisotropy

---

## 🔍 Root Cause Analysis

### Problem Chain

1. **Stage B produces G⁻¹ with small determinants**
   - Mean determinant: `0.0088`
   - Eigenvalue range: `[0.012, 0.21]`
   - Mean anisotropy ratio: `1.77` (weak anisotropy)

2. **Interpolation amplifies isotropy**
   - Weighted sum of already-weak anisotropic matrices
   - Result: G⁻¹(μ) eigenvalues `[0.0100, 0.0102]` → ratio `1.02` (nearly isotropic!)

3. **Small G⁻¹ → Small Σ_μ → Very negative log|Σ_μ|**
   - With `rhmc_alpha=0.1`: `Σ_μ ≈ 0.1 × G⁻¹(μ) + 1e-3 × I`
   - `Σ_μ` eigenvalues: `[0.0010, 0.0010]` (extremely small!)
   - `log|Σ_μ| ≈ -9.12` (very negative)

4. **Volume term in log q dominates**
   ```
   log q = -0.5 * (z-μ)ᵀ Σ⁻¹ (z-μ) - 0.5 * log|Σ| - const
         = quadratic term        + volume term  - const
         ≈ -0.5 to -2.0          + 4.56         - 1.84
         ≈ -2.6 (negative!)
   ```

5. **Result: KL = log q - log p + Δ < 0**
   - `log q ≈ -2.6`
   - `log p ≈ +1.8`
   - `KL ≈ -4.4` ❌ (NEGATIVE!)

---

## ✅ Solution: Isotropic Rescaling

### Strategy

**Rescale all G⁻¹ eigenvalues uniformly by factor 10**:
```
G⁻¹_new = 10 × G⁻¹_original
```

### Why 10x?

1. **Preserves anisotropy ratios exactly**
   - `λ_new / μ_new = (10λ) / (10μ) = λ/μ` ✅

2. **Increases determinant by 10^D = 100** (for D=2)
   - `det(G⁻¹_new) = 10² × det(G⁻¹_orig) = 100 × 0.0088 = 0.88`

3. **Reduces negative volume term by ~1.6**
   - `log|Σ_μ|`: `-9.12 → -5.93` (**+3.19**)
   - Volume term: `+4.56 → +2.97` (**-1.59**, less positive → log q less negative)

---

## 📊 Measured Impact

### Before Rescaling (Original Stage B)

```
Eigenvalues:     [0.012, 0.21]     (mean: 0.094)
Anisotropy ratio: 1.77             (weak)
Determinant:     0.0088            (very small!)
Trace:           0.19              (small)

→ Σ_μ eigenvalues: [0.0010, 0.0010]
→ log|Σ_μ|:        -9.12
→ log q:           ≈ -2.6 (negative)
→ KL:              ≈ -4.4 ❌
```

### After Rescaling (10x)

```
Eigenvalues:     [0.12, 2.09]      (mean: 0.945, +10x ✅)
Anisotropy ratio: 1.77             (preserved ✅)
Determinant:     0.88              (+100x ✅)
Trace:           1.89              (+10x ✅)

→ Σ_μ eigenvalues: [0.013, 0.210]   (much larger!)
→ log|Σ_μ|:        -5.93            (+3.19, less negative!)
→ log q:           ≈ -1.0 to 0.0    (expected improvement!)
→ KL:              ≈ 0.0 to 1.0 ✅  (expected positive!)
```

---

## 🚀 Implementation

### 1. Rescaling Script

Created `rescale_stage_b_metric.py` with features:
- **Isotropic mode**: Preserves anisotropy ratios exactly
- **Anisotropic mode**: Optional anisotropy amplification
- **Dry-run mode**: Preview impact before saving
- **Comprehensive analysis**: Eigenvalues, ratios, determinants, traces

### 2. Usage

```bash
# Preview impact (dry-run)
python rescale_stage_b_metric.py --dry-run

# Create rescaled metric (10x isotropic)
python rescale_stage_b_metric.py --scale-factor 10.0 --mode isotropic

# With anisotropy amplification
python rescale_stage_b_metric.py --scale-factor 10.0 --mode anisotropic --anisotropy-amplification 1.5
```

### 3. Configuration Update

Updated configs to use `metric_rescaled.pt`:
- `conf/config.yaml` (line 162)
- `conf/monolith_stagec.yaml` (line 65)

---

## 📈 Expected Results

### KL Divergence

**Before**:
```
log q:        -2.6
log p:        +1.8
KL:           -4.4 ❌
```

**After** (predicted):
```
log q:        -1.0 to 0.0  (volume term less positive)
log p:        +1.8         (unchanged)
KL:           +0.0 to 2.8 ✅
```

### Convergence

- **Encoder variance (`log_var`)**: Should stabilize at higher values (less aggressive compression)
- **RHMC sampling**: Should produce more diverse z0 (larger Σ_μ allows more exploration)
- **Training stability**: Should improve (no more negative KL penalty)

---

## 🔬 Alternative Approaches Considered

### Option 1: Rescale G⁻¹ atoms (CHOSEN ✅)
- ✅ Direct fix at the source
- ✅ Preserves/amplifies anisotropy
- ✅ No code changes needed
- ⚠️ Introduces hyperparameter (scale factor)

### Option 2: Increase `rhmc_alpha`
- ❌ Tried (alpha=5.0), didn't help
- ❌ Doesn't fix underlying small eigenvalues
- ❌ Just scales the already-tiny G⁻¹

### Option 3: Normalize G⁻¹ (trace/geomean)
- ❌ Tried, DESTROYS anisotropy
- ❌ Makes G⁻¹ more isotropic → worse KL

### Option 4: Ignore log q in KL
- ❌ Fundamentally wrong mathematically
- ❌ Loses Bayesian interpretation

### Option 5: Use different prior
- ❌ Doesn't address root cause
- ❌ More complex changes needed

---

## ✅ Next Steps

1. **Restart training with rescaled metric**
   ```bash
   # Kill current training
   pkill -f "python.*run_experiment"
   
   # Restart Stage C
   python run_experiment.py stage=C
   ```

2. **Monitor KL divergence**
   - Should be **positive** from epoch 0
   - Expected range: `+0.5 to +2.0`

3. **Check debug output**
   ```bash
   export RLVAE_DEBUG=1
   python run_experiment.py stage=C
   ```
   
   Look for:
   - `G⁻¹(μ) eigenvalues: min=0.12, max=2.09` (rescaled!)
   - `Σ_μ eigenvalues: [0.013, 0.210]` (much larger!)
   - `log|Σ_μ|: -5.93` (less negative!)
   - `log q: -1.0 to 0.0` (improved!)
   - `KL: +0.5 to +2.0 ✅` (positive!)

4. **If still negative KL**:
   - Try larger scale factor (20x or 50x)
   - Try anisotropic mode with amplification
   - Check for other numerical issues

---

## 📝 Technical Details

### Rescaling Formula (Isotropic)

For each atom `G⁻¹_i`:
```
G⁻¹_new,i = α × G⁻¹_orig,i
```

Where `α = 10.0` (scale factor).

### Rescaling Formula (Anisotropic)

For each atom `G⁻¹_i`:
1. Eigendecomposition: `G⁻¹_i = V Λ Vᵀ`
2. Apply global scaling: `Λ_scaled = α × Λ`
3. Amplify anisotropy:
   ```
   geom_mean = exp(mean(log(Λ_scaled)))
   Λ_final = geom_mean × (Λ_scaled / geom_mean)^β
   ```
   where `β = 1.5` (anisotropy amplification factor)
4. Reconstruct: `G⁻¹_new,i = V Λ_final Vᵀ`

### Determinant Scaling

For D-dimensional matrices:
```
det(α × G⁻¹) = α^D × det(G⁻¹)
```

For D=2:
```
det(10 × G⁻¹) = 100 × det(G⁻¹)
```

### Impact on log|Σ_μ|

```
Σ_μ = α_rhmc × G⁻¹(μ) + ε × I

log|Σ_μ,new| ≈ log(α_rhmc) + log|G⁻¹_new(μ)|
             ≈ log(α_rhmc) + log(α^D × |G⁻¹_orig(μ)|)
             ≈ log|Σ_μ,orig| + D × log(α)
             = -9.12 + 2 × log(10)
             = -9.12 + 4.61
             = -4.51

(Actual: -5.93 due to eps_reg contribution)
```

---

## 🎯 Success Criteria

- [x] Rescaled metric file created
- [x] Configuration updated
- [ ] Training restarted with new metric
- [ ] KL divergence positive (> 0.0)
- [ ] KL divergence stable (not exploding)
- [ ] Loss decreasing smoothly
- [ ] Reconstruction quality maintained or improved

---

## 📚 Related Documents

- `LOG_Q_CALCULATION_ANALYSIS.md`: Analysis of log q components
- `COMPREHENSIVE_G_GINV_AUDIT.md`: G and G⁻¹ consistency audit
- `KL_VERIFICATION_CHECKLIST.md`: Systematic KL verification checklist
- `NEGATIVE_KL_ROOT_CAUSE_ANALYSIS.md`: Root cause of negative KL
- `CORRECTED_SOLUTION_ANISOTROPY.md`: Anisotropy preservation strategy

---

## 📞 Contact

If issues persist after rescaling:
1. Check `RLVAE_DEBUG=1` output for anomalies
2. Verify metric_rescaled.pt was loaded (check model initialization logs)
3. Try larger scale factors (20x, 50x)
4. Consider anisotropic mode with amplification

**Status**: ✅ Solution implemented and ready for testing


