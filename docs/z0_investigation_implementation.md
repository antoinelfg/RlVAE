# Z0 Investigation Implementation Summary

## Completed Diagnostic Enhancements

### 1. Added Candidate-Level Diagnostics (`_diagnose_candidates`)
**Location:** `src/rlvae/models/components/riemannian_rhmc_posterior.py` (after line 1120)

**Features:**
- Logs properties of all K candidates before multi-try selection
- Computes Euclidean and Mahalanobis distances for each candidate
- Calculates correlation between h_score (volume metric) and Mahalanobis²
- Shows selection bias: Δ Mahal² (selected - pool mean)
- Provides detailed per-candidate dump for first 2-3 batches

**Call sites:** Lines ~628 and ~719 (both multi-try code paths)

### 2. Extended `_diagnose_initial_sample` for Stage Comparison
**Enhancements:**
- Now accepts `z_selected` (after multi-try, before volume acceptance)
- Compares three stages:
  - Raw candidates (analyzed in `_diagnose_candidates`)
  - z_selected (after multi-try selection)
  - z0 (final, after volume acceptance)
- Quantifies bias at each stage: Δ Mahal² for selection and volume acceptance

### 3. Enhanced Chi-Squared Test
**New features:**
- Kolmogorov-Smirnov test statistic and p-value
- Histogram comparison: empirical vs theoretical χ²(D) density
- Detects distribution mismatch quantitatively

## Created Ablation Experiment Configs

All configs in `conf/experiment/`:

1. **rlvae_debug_baseline.yaml**: K=5, vol_tol=0.05 (both ON)
2. **rlvae_debug_notry.yaml**: K=1, vol_tol=0.05 (multi-try OFF)
3. **rlvae_debug_novol.yaml**: K=5, vol_tol=0.0 (volume acc OFF)
4. **rlvae_debug_vanilla.yaml**: K=1, vol_tol=0.0 (both OFF, pure N(μ,Σ))
5. **rlvae_debug_hightry.yaml**: K=20, vol_tol=0.05 (high K test)

All set to 5 epochs for quick testing, WandB project: `rlvae-z0-investigation`

## Analysis Helper Script

**Location:** `scripts/analyze_z0_diagnostics.py`

**Usage:**
```bash
RLVAE_DEBUG=1 python run_experiment.py ... | tee output.log
python scripts/analyze_z0_diagnostics.py output.log
```

**Extracts:**
- All candidates Mahal² mean
- Selected Mahal² mean
- Δ Mahal² (selection bias)
- Correlation(h, Mahal²)
- Chi-squared deviation %
- KS test p-value

## Next Steps

### To Run Experiments:
```bash
# Create logs directory
mkdir -p logs/z0_investigation

# Run each experiment (note: use +experiment= to append to defaults)
for config in vanilla notry novol baseline hightry; do
  RLVAE_DEBUG=1 python run_experiment.py \
    +experiment=rlvae_debug_${config} \
    settings.pipeline.mode=three_stage \
    settings.pipeline.run_stage_a=false \
    settings.pipeline.run_stage_b=true \
    settings.pipeline.run_stage_c=true \
    settings.training.stage_overrides.stage_b.enabled=true \
    | tee logs/z0_investigation/debug_${config}.log
done
```

### To Analyze Results:
1. Parse logs: `python scripts/analyze_z0_diagnostics.py logs/z0_investigation/debug_*.log`
2. Compare WandB metrics across runs
3. Generate plots from diagnostic data
4. Write conclusion document with recommendations

## Expected Outcomes

**If multi-try causes bias:**
- `vanilla` and `notry` should have Mahal² ≈ 2.0, KL > 0
- `baseline`, `novol`, `hightry` should show large deviations
- Correlation(h, Mahal²) should be strongly positive in multi-try runs

**If volume acceptance causes bias:**
- `vanilla` and `novol` should have Mahal² ≈ 2.0, KL > 0
- `baseline`, `notry` should show deviations
- Stage comparison should show large Δ Mahal² (volume acc step)

**If both contribute:**
- Only `vanilla` matches χ²(2)
- Both stages show bias in diagnostics

