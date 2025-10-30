# Negative KL Divergence: Diagnostic Report

**Date**: [To be filled after running diagnostics]  
**Configuration**: [Phase C with RHMC posterior]  
**Objective**: Understand why `log q(z0) = log N(z0 | μ, Σ_μ)` is too negative, causing `KL = log q - log p < 0`

---

## Executive Summary

**Problem**: KL divergence is consistently negative during Phase C training.

**Root Cause**: [To be determined from diagnostics]

**Recommendation**: [To be determined based on hypothesis testing]

---

## Diagnostic Setup

### Configuration Used

```yaml
model:
  posterior:
    type: riemannian_rhmc
    rhmc_alpha: [VALUE]
    rhmc_steps: [VALUE]
    rhmc_step_size: [VALUE]
    rhmc_eps_reg: [VALUE]
    sigma_normalization_mode: [MODE]
    initial_target_radius: [VALUE]
    min_cov_eig: [VALUE]
```

### Data Collection

- **Batch size**: [VALUE]
- **Latent dimension**: [VALUE]
- **Number of batches analyzed**: [VALUE]
- **Diagnostic mode**: RLVAE_DEBUG=1

---

## Diagnostic Results

### 1. Initial Sampling Diagnostics (z_base → z0)

#### Σ_μ Properties

| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| alpha | [VALUE] | - | - |
| Eigenvalue min | [VALUE] | >0 | [✓/⚠️/❌] |
| Eigenvalue max | [VALUE] | - | - |
| Trace | [VALUE] | ~D×alpha | [✓/⚠️/❌] |
| log\|Σ\| | [VALUE] | - | - |
| Condition number | [VALUE] | <1000 | [✓/⚠️/❌] |
| Anisotropy ratio | [VALUE] | 1-10 | [✓/⚠️/❌] |

#### Distance Analysis

| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| \\|\\|z_base - μ\\|\\| | [VALUE] | - | - |
| \\|\\|z0 - μ\\|\\| | [VALUE] | - | - |
| \\|\\|z_base - z0\\|\\| | [VALUE] | small | [✓/⚠️/❌] |
| Volume acceptance effect | [MOVED CLOSER/AWAY/SAME] | - | - |

#### Expected vs Actual

| Metric | Value | Status |
|--------|-------|--------|
| Expected \\|\\|z-μ\\|\\| (√tr(Σ)) | [VALUE] | - |
| Actual \\|\\|z0-μ\\|\\| | [VALUE] | - |
| Ratio (actual/expected) | [VALUE] | Should be ~1.0 |

**Analysis**: 
- [✓] Distance ratio is acceptable (~1.0)
- [⚠️] Distance ratio is [high/low], suggesting [over/under]-dispersed Σ_μ
- [❌] Distance ratio is [>>1.5 or <<0.5], indicating serious mismatch

#### Mahalanobis Analysis

| Metric | Value |
|--------|-------|
| Euclidean dist² | [VALUE] |
| Mahalanobis dist² | [VALUE] |
| Ratio (Mahal²/Euclid²) | [VALUE] |

**Per-eigenvalue contributions** (showing dominant dimensions):

| Dimension | λ | y² | Contribution (y²/λ) |
|-----------|---|----|--------------------|
| [0] | [VALUE] | [VALUE] | [VALUE] |
| [1] | [VALUE] | [VALUE] | [VALUE] |
| ... | ... | ... | ... |

**Chi-squared test**:
- Expected Mahal²: χ²(D) mean = [D]
- Observed Mahal²: [VALUE]
- Deviation: [VALUE] ([PERCENT]%)
- Status: [✓ Within 2σ / ⚠️ Borderline / ❌ Significant deviation]

---

### 2. RHMC Trajectory Diagnostics (z0 → zK)

#### Initial State (k=0)

| Metric | Value |
|--------|-------|
| \\|\\|z0 - μ\\|\\| | [VALUE] |
| \\|\\|ρ0\\|\\| | [VALUE] |

#### Step-by-Step Evolution

| Step k | \\|\\|z_k - μ\\|\\| | \\|\\|z_k - z0\\|\\| | \\|\\|ρ_k\\|\\| |
|--------|----------------|-----------------|-------------|
| 0 | [VALUE] | 0.000 | [VALUE] |
| 1 | [VALUE] | [VALUE] | [VALUE] |
| 2 | [VALUE] | [VALUE] | [VALUE] |
| ... | ... | ... | ... |
| K | [VALUE] | [VALUE] | [VALUE] |

#### Trajectory Summary

| Metric | Value | Analysis |
|--------|-------|----------|
| Initial \\|\\|z0 - μ\\|\\| | [VALUE] | - |
| Final \\|\\|zK - μ\\|\\| | [VALUE] | - |
| Total drift from z0 | [VALUE] | - |
| Net change in \\|\\|·-μ\\|\\| | [VALUE] | [MOVED AWAY/TOWARD/SAME] |
| Monotonicity | [AWAY/TOWARD/OSCILLATING] | - |

**Conclusion**: 
- [✓] RHMC moves toward or stays near μ (good)
- [⚠️] RHMC shows oscillatory behavior
- [❌] RHMC consistently moves away from μ (problematic)

---

### 3. Log_q Decomposition

#### Standard Decomposition

| Component | Value | Expected Range | Status |
|-----------|-------|----------------|--------|
| Quadratic term | [VALUE] | Negative | [✓/⚠️/❌] |
| Volume term | [VALUE] | Negative | [✓/⚠️/❌] |
| Constant term | [VALUE] | Negative | [✓] |
| **Total log_q** | [VALUE] | -2 to -4 (2D) | [✓/⚠️/❌] |

#### Mahalanobis Eigenbasis Decomposition

| Metric | Value |
|--------|-------|
| Euclidean \\|\\|z-μ\\|\\|² | [VALUE] |
| Mahalanobis (z-μ)ᵀΣ⁻¹(z-μ) | [VALUE] |
| Ratio (Mahal/Euclid) | [VALUE] |

**Dominant dimensions** (contributing >10% to Mahalanobis²):

| Dimension | Contribution % | Cumulative % |
|-----------|----------------|--------------|
| [0] | [VALUE]% | [VALUE]% |
| [1] | [VALUE]% | [VALUE]% |

**Chi-squared fit**:
- Dimension D: [VALUE]
- Expected Mahal²: [D]
- Observed Mahal²: [VALUE]
- Deviation: [VALUE] ([PERCENT]%)

---

## Hypothesis Testing

### Hypothesis A: Σ_μ is too small (under-dispersed)

**Evidence**:
- [ ] Distance ratio (actual/expected): [VALUE] [>>1.5 suggests too small]
- [ ] z0 is far from μ (>1.5× expected)
- [ ] Σ eigenvalues are small relative to ||z-μ||²

**Verdict**: [✓ LIKELY / ⚠️ POSSIBLE / ❌ UNLIKELY]

---

### Hypothesis B: Σ_μ has wrong shape (anisotropy mismatch)

**Evidence**:
- [ ] Anisotropy ratio (λ_max/λ_min): [VALUE] [>10 suggests mismatch]
- [ ] Empirical covariance differs significantly from Σ_μ
- [ ] Mahalanobis contributions are highly uneven

**Verdict**: [✓ LIKELY / ⚠️ POSSIBLE / ❌ UNLIKELY]

---

### Hypothesis C: RHMC pushes z away from μ

**Evidence**:
- [ ] RHMC net distance change: [VALUE] [>0.1 suggests moving away]
- [ ] Trajectory is monotonically increasing from μ
- [ ] Final ||zK-μ|| > initial ||z0-μ||

**Verdict**: [✓ LIKELY / ⚠️ POSSIBLE / ❌ UNLIKELY]

---

### Hypothesis D: Gaussian posterior is fundamentally wrong

**Evidence**:
- [ ] Chi-squared deviation: [VALUE] σ [>2 suggests mismatch]
- [ ] Mahalanobis² significantly deviates from χ²(D)
- [ ] Empirical distribution is non-Gaussian

**Verdict**: [✓ LIKELY / ⚠️ POSSIBLE / ❌ UNLIKELY]

---

## Recommended Actions

### Primary Recommendation

Based on the diagnostic findings, the primary issue is: **[HYPOTHESIS X]**

**Action**: [SPECIFIC CONFIG CHANGE]

```yaml
# Recommended configuration change:
model:
  posterior:
    [parameter]: [new_value]  # Reason: [explanation]
```

### Alternative Approaches

If the primary fix doesn't work, try:

1. **[Alternative 1]**
   - Config: `[parameter]: [value]`
   - Rationale: [explanation]

2. **[Alternative 2]**
   - Config: `[parameter]: [value]`
   - Rationale: [explanation]

### Parameter Sweep Suggestions

If diagnosis is inconclusive, perform a systematic sweep:

```yaml
# Sweep configuration
rhmc_alpha: [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
sigma_normalization_mode: ['none', 'trace', 'geomean']
```

---

## Visualizations

### Trajectory Plot
![Trajectory](../diagnostic_plots/trajectory_2d.png)

**Observation**: [Describe what the plot shows]

### Distance Evolution
![Distance](../diagnostic_plots/distance_evolution.png)

**Observation**: [Describe trends]

### Mahalanobis Heatmap
![Mahalanobis](../diagnostic_plots/mahalanobis_heatmap.png)

**Observation**: [Identify dominant dimensions]

### Distribution Comparison
![Distribution](../diagnostic_plots/distribution_comparison.png)

**Observation**: [Assess Gaussian fit quality]

### Log-Probability Breakdown
![LogProb](../diagnostic_plots/logprob_breakdown.png)

**Observation**: [Identify which term dominates negativity]

---

## Next Steps

1. **Implement primary recommendation**: [ACTION]
2. **Re-run diagnostics**: Verify KL becomes positive
3. **If still negative**: Try alternative approaches
4. **If persistently negative**: Consider architectural changes (non-Gaussian posterior or different prior)

---

## Appendix: Full Terminal Output

```
[Paste full diagnostic output from RLVAE_DEBUG=1 run here]
```

---

**Report completed**: [DATE]  
**Analyst**: [NAME]

