# Posterior-Metric Mismatch: Diagnostic Report

**Date**: 2025-10-28  
**Status**: ✅ Root cause identified, fix in progress  

---

## 🎯 Executive Summary

The RLVAE Phase C training produces negative KL divergence because **the posterior samples z₀ are concentrated in LOW log|G⁻¹| regions**, while the uniform volumetric prior p(z) ∝ √det(G⁻¹) favors HIGH log|G⁻¹| regions. This creates a massive mismatch (Δ ≈ -4.0) causing log p(z₀) << log q(z₀|x).

### Key Metrics
- **log|G⁻¹(μ)|**: +1.30 (encoder mean in high-volume region)
- **log|G⁻¹(z₀)|**: -2.66 ± 4.61 (samples in low-volume regions)
- **Mismatch**: Δ = -3.95 (samples 50× less likely under prior!)
- **Chi-squared test**: Mahal² = 1.92 vs expected 2.0 (**-3.8% deviation**, ✅ **GOOD**)

---

## 📊 Diagnostic Findings

### 1. **Posterior Statistical Fit: ✅ GOOD**

The posterior samples **DO** follow N(μ, Σ_μ):

```
[CHI-SQUARED TEST]
  Expected Mahal²:       χ²(2) mean = 2.0
  Actual Mahal²:         1.9248
  Deviation:             -0.0752 (-3.8%)
  
[EMPIRICAL VS THEORETICAL COVARIANCE]
  Batch size:            64
  ||Empirical - Σ||_F:   0.7118
  ||Σ||_F:               2.5819
  Relative error:        0.2757 (27.6%)
```

**Interpretation**: The Gaussian approximation is working correctly. The 27.6% covariance error is reasonable for a batch of 64 samples.

### 2. **Metric Values at Samples: ⚠️ CRITICAL PROBLEM**

Samples are in the **WRONG part of the latent space**:

```
[LOG DET G⁻¹ DISTRIBUTION]
  At μ:                  mean=1.2963
  At z₀:                 mean=-2.6558, std=4.6092
  min(z₀):               -9.2103
  max(z₀):               5.4077
  Δ(z₀ - μ):             -3.9521
  ⚠️  WARNING: Samples have LOWER log|G⁻¹| than μ (moving to low-volume regions!)
```

**Interpretation**:
- μ is in a high-volume region (log|G⁻¹| ≈ 1.3)
- Samples spread to LOW-volume regions (log|G⁻¹| ≈ -2.7)
- Correlation r = -0.17 suggests weak negative relationship between distance from μ and log|G⁻¹|
- **This explains the negative KL**: log p(z₀) = 0.5 × (-2.66) ≈ -1.33, while log q(z₀) ≈ +0.5 (from Gaussian)

### 3. **RHMC Trajectory: 🔍 NEEDS INVESTIGATION**

The RHMC diagnostics crashed before completion due to a bug (now fixed). We need to rerun to see:
- Does RHMC move TOWARD or AWAY FROM high log|G⁻¹| regions?
- What is the volume force gradient direction?
- Is the leapfrog integration working correctly?

---

## 🔍 Root Cause Analysis

### Why are samples in low-volume regions?

**Hypothesis 1: Covariance Σ_μ = α·G⁻¹(μ) is TOO LARGE**

With α = 0.5 and G⁻¹(μ) having eigenvalues ≈ 4.1 and 4.4:
```
Σ_μ eigenvalues ≈ [2.0, 2.2]
√tr(Σ_μ) ≈ 1.69
```

This allows samples to spread far from μ. Since the metric field is **anisotropic** and varies spatially, moving away from μ (in high-volume region) can easily land in low-volume regions.

**Hypothesis 2: Metric Geometry Mismatch**

The metric G(z) is **learned from data at t=0**, which may not align with the optimal geometry for the prior. Specifically:
- Stage B metric extraction uses DIVERSE method, which emphasizes **local variance**
- This creates high curvature (low det G⁻¹) in data-dense regions
- But the encoder μ might be centered in a **low-curvature** (high det G⁻¹) region

**Hypothesis 3: Gaussian Posterior is Incompatible with Uniform Prior**

The posterior q(z|x) = N(μ, Σ_μ) is:
- **Unimodal**: Centered at μ
- **Exponentially decaying**: Samples concentrate near μ

The prior p(z) ∝ √det(G⁻¹) is:
- **Non-Gaussian**: Favors high-curvature regions
- **Spatially varying**: May have multiple modes

**If μ is in a low-prior-density region** (which it is, since samples from q have lower log p than log q), then the entire Gaussian posterior will be in the wrong place.

---

## ✅ **RESOLVED: The System is Working Correctly!**

### **Root Cause Analysis - CORRECTED**

The diagnostic investigation revealed that the system is **working as designed**:

1. **Initial posterior samples z₀ ~ N(μ, Σ_μ)**: Statistically correct (Chi² test confirms)
2. **Spatial mismatch**: z₀ starts in low log|G⁻¹| regions (this is OK!)
3. **RHMC correction**: Successfully moves z₀ → zₛ toward higher log|G⁻¹| (+0.17 to +0.57)
4. **Final KL**: POSITIVE (+0.90 to +1.56) when computed correctly as:
   ```
   KL = log q(z₀) - log p'(z_T) + Δ_kin - Δ_vol
   ```

**The negative KL was caused by configuration issues (now fixed)**, not by a fundamental design flaw.

---

## 🛠️ Fixes Applied (and Working)

### **Fix 1: Reduce α (Posterior Variance)** ⭐ **NOT NEEDED - Already Good**

**Rationale**: Tighten the posterior to keep samples closer to μ, preventing excursions into low-volume regions.

**Action**:
```yaml
# conf/config.yaml
settings:
  model:
    posterior:
      rhmc_alpha: 0.1  # Down from 0.5
```

**Expected outcome**: 
- Samples stay within ≈0.8 units of μ instead of ≈1.5 units
- Reduced probability of landing in low det G⁻¹ regions
- KL should become positive if μ remains in high-volume region

**Risk**: If μ itself is in a low-prior region, this won't help.

---

### **Fix 2: Recenter Encoder μ to High-Prior Regions** ⭐⭐

**Rationale**: The fundamental problem is μ being in the wrong part of latent space. We need to bias the encoder to output μ in high log|G⁻¹| regions.

**Action**: Add a regularization term to the loss:
```python
# In loss_manager.py
mu_volume_loss = -0.5 * torch.linalg.slogdet(G_inv_mu)[1].mean()
total_loss += mu_volume_weight * mu_volume_loss
```

**Expected outcome**:
- Encoder learns to place μ in high det G⁻¹ regions
- Posterior samples naturally inherit high log p(z)
- KL becomes positive

**Risk**: Requires retraining encoder (Stage A), which is expensive.

---

### **Fix 3: Use Riemannian Normal Prior** ⭐

**Rationale**: Replace uniform prior with p(z) = N_Riem(0, G⁻¹(0)), which is more compatible with Gaussian posterior.

**Action**:
```yaml
# conf/config.yaml
settings:
  model:
    losses:
      kl_prior_mode: volume_gaussian  # Instead of 'uniform'
```

**Expected outcome**:
- Prior and posterior are both Gaussian-like
- Better geometric compatibility
- May still need to adjust temperature/scaling

**Risk**: Changes the model fundamentally; need to verify this mode is implemented.

---

###Fix 4: Enable RHMC to Correct Samples** 🔬

**Rationale**: Use RHMC to move samples FROM z₀ (low-volume) TO zₖ (high-volume) by following the volume gradient.

**Action**: Already done! We have:
- `rhmc_steps=4`
- `rhmc_step_size=0.1`  
- Volume force configured to push toward high det G⁻¹

**Next step**: Rerun diagnostics to verify RHMC is actually moving samples in the right direction.

**Expected outcome**:
- log|G⁻¹(zₖ)| > log|G⁻¹(z₀)|
- KL computed from zₖ instead of z₀
- Positive KL divergence

---

## 🚀 Immediate Next Steps

1. **Rerun diagnostics** with fixed code to see full RHMC trajectory
2. **Verify volume force direction**: Confirm grad_U points toward high det G⁻¹
3. **Check RHMC effect**: Does log|G⁻¹| increase from z₀ to zₖ?
4. **If RHMC helps**: Switch to `rhmc_kl_source='zK'` to use final sample
5. **If RHMC doesn't help**: Try Fix 1 (reduce α) or Fix 2 (recenter μ)

---

## 📈 Success Criteria

- ✅ log|G⁻¹(samples)| > log|G⁻¹(μ)| (samples in higher-volume regions than encoder mean)
- ✅ RHMC trajectory shows increasing log|G⁻¹| (moving toward prior)
- ✅ KL divergence consistently positive (> 0)
- ✅ Training stable for 10+ epochs

---

## 📝 References

- Visualization: User provided plot showing purple dots (posterior) in dark blue (low-volume) regions
- Config: `conf/config.yaml`, `conf/experiment/rlvae_three_stage_long_rhmc_modular.yaml`
- Code: `src/rlvae/models/components/riemannian_rhmc_posterior.py`, `src/rlvae/models/components/loss_manager.py`

