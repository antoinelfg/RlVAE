# Deep KL Diagnostics: Implementation Complete

**Date**: October 28, 2025  
**Status**: ✅ **IMPLEMENTED**  
**Purpose**: Comprehensive diagnostic system to understand negative KL divergence in Phase C training

---

## What Was Implemented

### Phase 1: Enhanced RHMC Posterior Diagnostics ✅

**File Modified**: `src/rlvae/models/components/riemannian_rhmc_posterior.py`

#### Added Diagnostic Functions

1. **`_diagnose_initial_sample()`** (lines 904-1045)
   - Comprehensive Σ_μ properties analysis
   - Distance analysis (z_base → z0)
   - Expected vs actual distance comparison
   - Mahalanobis distance decomposition in eigenbasis
   - Chi-squared test for Gaussian fit
   - Empirical vs theoretical covariance comparison

2. **`_diagnose_posterior_mismatch()`** (lines 1047-1108)
   - Statistical tests for posterior mismatch
   - Returns quantitative test results
   - Q-Q plot data generation
   - Covariance mismatch metrics

3. **Enhanced `_rhmc_exploration()`** (lines 1110-1205)
   - Step-by-step trajectory tracking
   - Distance evolution from μ and z0
   - Momentum norm monitoring
   - Drift analysis (toward/away from μ)
   - Monotonicity detection

4. **Enhanced `log_q_riem()`** (lines 173-248)
   - Full decomposition: quadratic + volume + constant
   - Mahalanobis eigenbasis analysis
   - Per-eigenvalue contribution breakdown
   - Dominant dimension identification
   - Enhanced chi-squared fit testing

#### Key Features

- **All diagnostics gated by `RLVAE_DEBUG=1`**: Zero performance impact in production
- **Proper terminology**: Consistently uses z0 (not z_base) for log_q calculations
- **Comprehensive output**: 80-character formatted sections for easy reading
- **Actionable warnings**: Highlights problems with ⚠️ markers

---

### Phase 2: Standalone Diagnostic Script ✅

**File Created**: `scripts/diagnose_negative_kl.py`

#### Features

- **Minimal Phase C training**: Runs 1-2 batches for quick diagnosis
- **Automatic hypothesis testing**: Tests all 4 hypotheses systematically
- **Recommendations**: Provides specific config changes based on findings
- **Hydra integration**: Uses existing config system

#### Usage

```bash
# Run diagnostic with full logging
RLVAE_DEBUG=1 python scripts/diagnose_negative_kl.py experiment=your_phase_c_config

# Or directly
RLVAE_DEBUG=1 python scripts/diagnose_negative_kl.py
```

#### Output

- Terminal diagnostics (all sections)
- Hypothesis test results
- Specific recommendations
- Example config changes

---

### Phase 3: Visualization Script ✅

**File Created**: `scripts/visualize_kl_diagnostics.py`

#### Visualization Functions

1. **`plot_trajectory_2d()`**
   - Shows z0, zK, μ in 2D latent space
   - Confidence ellipse from Σ_μ
   - Trajectory arrows
   - Multi-sample overlay

2. **`plot_distance_evolution()`**
   - ||z_k - μ|| over RHMC steps
   - ||z_k - z0|| drift tracking
   - Momentum norm evolution

3. **`plot_mahalanobis_heatmap()`**
   - Per-eigenvalue contributions
   - Eigenvalue magnitudes
   - Identifies dominant dimensions

4. **`plot_distribution_comparison()`**
   - Empirical (z-μ) vs theoretical N(0, Σ)
   - Marginal distributions
   - Joint 2D visualization (for 2D latent)

5. **`plot_logprob_breakdown()`**
   - Stacked bar chart: quadratic + volume + constant
   - Shows log_q and KL
   - Highlights negative KL

#### Usage

```bash
# Generate example plots
python scripts/visualize_kl_diagnostics.py --output-dir diagnostic_plots

# In code (import and use functions)
from scripts.visualize_kl_diagnostics import plot_trajectory_2d
plot_trajectory_2d(z0, zK, mu, Sigma, trajectory, save_path='my_plot.png')
```

---

### Phase 4: Documentation Template ✅

**File Created**: `docs/NEGATIVE_KL_DIAGNOSTIC_REPORT.md`

#### Sections

1. **Executive Summary**: Problem, root cause, recommendation
2. **Diagnostic Setup**: Configuration and data collection details
3. **Diagnostic Results**: Complete tables for all metrics
4. **Hypothesis Testing**: Structured evidence and verdicts
5. **Recommended Actions**: Primary fix and alternatives
6. **Visualizations**: Embedded plots with observations
7. **Next Steps**: Clear action items
8. **Appendix**: Full terminal output

---

## How to Use This System

### Step 1: Run Diagnostics

```bash
# Set environment variable for full diagnostics
export RLVAE_DEBUG=1

# Option A: Use standalone diagnostic script (fastest)
python scripts/diagnose_negative_kl.py experiment=rlvae_three_stage_long_rhmc_modular

# Option B: Run your normal Phase C training (1-2 epochs)
python run_experiment.py experiment=rlvae_three_stage_long_rhmc_modular

# Save output to file for analysis
python run_experiment.py experiment=your_config > diagnostic_log.txt 2>&1
```

### Step 2: Review Diagnostic Output

Look for these key sections in the terminal output:

#### [INITIAL SAMPLING DIAGNOSTICS]
```
Σ_μ PROPERTIES
  alpha:                 0.200000
  Σ eigenvalues:         min=0.XXXXX, max=X.XXXXX
  Σ trace:               X.XXXXX
  Anisotropy ratio:      X.XX

DISTANCE ANALYSIS
  ||z0 - μ||:            mean=X.XXXX, std=X.XXXX
  
EXPECTED VS ACTUAL
  Expected ||z-μ||:      √(tr(Σ)) = X.XXXX
  Actual ||z0-μ||:       X.XXXX
  Ratio (actual/expected): X.XXXX  ← KEY METRIC!

MAHALANOBIS ANALYSIS
  Mahalanobis dist²:     mean=X.XXXX
  
CHI-SQUARED TEST
  Expected Mahal²:       χ²(D) mean = D
  Actual Mahal²:         X.XXXX  ← Should be ~D
  Deviation:             ±X.XXXX (XX%)  ← Should be <20%
```

#### [RHMC TRAJECTORY DIAGNOSTICS]
```
TRAJECTORY SUMMARY
  Initial ||z0 - μ||:    X.XXXX
  Final ||zK - μ||:      X.XXXX
  Net change in ||·-μ||: ±X.XXXX  ← Positive = moved away (BAD)
  → RHMC MOVED [AWAY/TOWARD/SAME] from μ
```

#### [LOG_Q_RIEM FULL DECOMPOSITION]
```
STANDARD DECOMPOSITION
  Quadratic term: mean=X.XXXX  ← Should be negative
  Volume term:    mean=X.XXXX  ← Should be negative
  Constant term:  X.XXXX       ← Always negative
  Total log_q:    mean=X.XXXX  ← KEY: Is this too negative?

MAHALANOBIS EIGENBASIS DECOMPOSITION
  Per-eigenvalue contributions:
    Dim 0: λ=X.XX, y²/λ=X.XX  ← Which dims dominate?
    Dim 1: λ=X.XX, y²/λ=X.XX
    
CHI-SQUARED FIT
  Expected Mahal²:       D
  Observed Mahal²:       X.XXXX
  Deviation:             ±X.XXXX (XX%)  ← >50% suggests mismatch
```

### Step 3: Analyze Findings

Use this decision tree:

```
1. Check distance ratio (actual/expected)
   ├─ >1.5 → Σ too small → Increase rhmc_alpha
   ├─ <0.5 → Σ too large → Decrease rhmc_alpha
   └─ ~1.0 → Σ scale OK, check other metrics

2. Check anisotropy ratio (λ_max/λ_min)
   ├─ >10 → Try sigma_normalization_mode: 'trace'
   └─ <5  → Anisotropy OK

3. Check RHMC drift
   ├─ >+0.1 → RHMC moving away → Reduce step_size or steps
   └─ <0    → RHMC moving toward μ (good)

4. Check chi-squared deviation
   ├─ >50% → Gaussian posterior mismatch → Consider architectural change
   └─ <20% → Gaussian fit is acceptable
```

### Step 4: Generate Visualizations (Optional)

```bash
# Use the visualization script with your data
python scripts/visualize_kl_diagnostics.py --output-dir my_diagnostics

# Or import functions in your analysis notebook
from scripts.visualize_kl_diagnostics import *
```

### Step 5: Document Findings

Fill in the report template:

```bash
# Copy template
cp docs/NEGATIVE_KL_DIAGNOSTIC_REPORT.md docs/my_diagnostic_report_$(date +%Y%m%d).md

# Edit with your findings
vim docs/my_diagnostic_report_$(date +%Y%m%d).md
```

### Step 6: Implement Fix

Based on your analysis, update the config:

```yaml
# Example fixes:

# Fix 1: Σ too small (distance ratio >1.5)
model:
  posterior:
    rhmc_alpha: 1.0  # Increase from 0.2

# Fix 2: Σ too large (distance ratio <0.5)
model:
  posterior:
    rhmc_alpha: 0.1  # Decrease from 0.5

# Fix 3: High anisotropy (ratio >10)
model:
  posterior:
    sigma_normalization_mode: 'trace'  # Force trace normalization

# Fix 4: RHMC drifts away
model:
  posterior:
    rhmc_step_size: 0.01  # Reduce from 0.02
    rhmc_steps: 2  # Reduce from 4

# Fix 5: Persistent issues
model:
  posterior:
    initial_target_radius: 0.0  # Ensure disabled
    sigma_normalization_mode: 'none'  # Try raw G⁻¹
```

### Step 7: Re-test

```bash
# Re-run with new config
RLVAE_DEBUG=1 python run_experiment.py experiment=your_config_fixed

# Check if KL is now positive
grep "FINAL KL LOSS" training_log.txt
```

---

## Hypothesis → Fix Mapping

| Hypothesis | Evidence | Primary Fix | Alternative Fix |
|------------|----------|-------------|-----------------|
| **A: Σ too small** | Distance ratio >1.5 | `rhmc_alpha: 1.0` (up) | `sigma_normalization_mode: 'none'` |
| **A: Σ too large** | Distance ratio <0.5 | `rhmc_alpha: 0.1` (down) | Check if `initial_target_radius` is forcing scale |
| **B: Anisotropy** | Ratio >10 | `sigma_normalization_mode: 'trace'` | `sigma_normalization_mode: 'none'` |
| **C: RHMC drift** | Net change >+0.1 | `rhmc_step_size: 0.01` (down) | `rhmc_steps: 2` (down) |
| **D: Gaussian wrong** | Chi² dev >50% | Try volume_gaussian prior | Consider flow-based posterior |

---

## Example Workflow

```bash
# Complete diagnostic workflow example

# 1. Run diagnostics (save to file)
RLVAE_DEBUG=1 python scripts/diagnose_negative_kl.py \
  experiment=rlvae_three_stage_long_rhmc_modular \
  > diagnostic_output_$(date +%Y%m%d).txt 2>&1

# 2. Review output
less diagnostic_output_$(date +%Y%m%d).txt

# 3. Extract key metrics
grep "Ratio (actual/expected)" diagnostic_output_*.txt
grep "Net change in" diagnostic_output_*.txt
grep "Deviation:" diagnostic_output_*.txt

# 4. Based on findings, create new config
cp conf/experiment/rlvae_three_stage_long_rhmc_modular.yaml \
   conf/experiment/rlvae_three_stage_fixed.yaml

# Edit the new config with recommended fixes
vim conf/experiment/rlvae_three_stage_fixed.yaml

# 5. Test the fix
RLVAE_DEBUG=1 python run_experiment.py \
  experiment=rlvae_three_stage_fixed \
  > diagnostic_output_fixed_$(date +%Y%m%d).txt 2>&1

# 6. Compare KL values
grep "FINAL KL LOSS" diagnostic_output_*.txt

# 7. If positive, proceed with full training
python run_experiment.py experiment=rlvae_three_stage_fixed
```

---

## Quick Reference: Diagnostic Checklist

- [ ] Enable `RLVAE_DEBUG=1`
- [ ] Run 1-2 batches of Phase C
- [ ] Check [INITIAL SAMPLING DIAGNOSTICS]
  - [ ] Distance ratio (actual/expected): Should be ~1.0
  - [ ] Anisotropy ratio: Should be <10
  - [ ] Chi-squared deviation: Should be <20%
- [ ] Check [RHMC TRAJECTORY DIAGNOSTICS]
  - [ ] Net change in ||·-μ||: Should be ≤0 (toward μ)
  - [ ] Monotonicity: Should not be "AWAY"
- [ ] Check [LOG_Q_RIEM FULL DECOMPOSITION]
  - [ ] Total log_q: Should be reasonable (-2 to -4 for 2D)
  - [ ] Dominant dimensions: Should be balanced
  - [ ] Chi-squared fit: Should be <50% deviation
- [ ] Identify primary hypothesis (A, B, C, or D)
- [ ] Apply corresponding fix
- [ ] Re-run diagnostics to verify

---

## Troubleshooting

### Issue: No diagnostic output

**Cause**: `RLVAE_DEBUG=1` not set or not propagated

**Fix**:
```bash
# Ensure export
export RLVAE_DEBUG=1
python your_script.py

# Or inline
RLVAE_DEBUG=1 python your_script.py
```

### Issue: Too much output

**Cause**: Diagnostics run on every batch

**Solution**: Run for only 1-2 batches
```yaml
trainer:
  max_epochs: 1
  limit_train_batches: 2  # Only 2 batches
```

### Issue: Diagnostics show conflicting evidence

**Cause**: Multiple factors contributing to negative KL

**Solution**: Address issues in order:
1. First fix Σ scale (hypothesis A)
2. Then fix anisotropy (hypothesis B)
3. Then fix RHMC dynamics (hypothesis C)
4. Finally consider architectural changes (hypothesis D)

---

## Files Modified/Created

### Modified
- ✅ `src/rlvae/models/components/riemannian_rhmc_posterior.py`
  - Added `_diagnose_initial_sample()` (140 lines)
  - Added `_diagnose_posterior_mismatch()` (60 lines)
  - Enhanced `_rhmc_exploration()` (95 lines)
  - Enhanced `log_q_riem()` (75 lines)

### Created
- ✅ `scripts/diagnose_negative_kl.py` (300 lines)
  - Standalone diagnostic runner
  - Hypothesis testing
  - Recommendation engine

- ✅ `scripts/visualize_kl_diagnostics.py` (450 lines)
  - 5 visualization functions
  - Example usage
  - Interactive mode

- ✅ `docs/NEGATIVE_KL_DIAGNOSTIC_REPORT.md` (template)
  - Structured report format
  - Tables for all metrics
  - Visualization embeddings

- ✅ `docs/DEEP_KL_DIAGNOSTICS_IMPLEMENTATION.md` (this file)
  - Complete usage guide
  - Workflow examples
  - Quick reference

---

## Performance Impact

- **With `RLVAE_DEBUG=0`** (default): **ZERO overhead**
  - All diagnostics are gated by environment variable
  - No performance impact on production training

- **With `RLVAE_DEBUG=1`**: **~10-20% slowdown**
  - Due to eigendecompositions and printing
  - Only use for diagnosis, not production

---

## Next Steps After Diagnosis

1. **If KL becomes positive**: Continue full training
2. **If KL remains negative**: Try alternative fixes
3. **If all fixes fail**: Document findings and consider:
   - Switching to Volume Gaussian prior
   - Using non-Gaussian posterior (normalizing flows)
   - Adjusting the entire Phase C architecture

---

**Implementation Status**: ✅ **COMPLETE**  
**Ready for Use**: Yes  
**Testing Required**: User should run on actual Phase C configuration  
**Documentation**: Complete

---

For questions or issues, refer to:
- `ROOT_CAUSE_FOUND.md` - Previous investigation
- `NEGATIVE_KL_ROOT_CAUSE_ANALYSIS.md` - Earlier diagnostic work
- `LOG_Q_STABILIZATION_GUIDE.md` - Stabilization techniques

