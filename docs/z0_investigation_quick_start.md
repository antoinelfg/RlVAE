# Z0 Investigation - Quick Start Guide

## ✅ What's Been Implemented

### 1. Enhanced Diagnostics in `riemannian_rhmc_posterior.py`
- **`_diagnose_candidates()`**: Logs all K candidates before multi-try selection
  - Euclidean & Mahalanobis distances
  - Correlation between h_score and Mahalanobis²
  - Selection bias quantification
- **Enhanced `_diagnose_initial_sample()`**: Stage-by-stage comparison (candidates → selected → final)
- **KS test & histogram comparison**: Quantitative distribution mismatch detection

### 2. Five Ablation Experiment Configs
All configs in `conf/experiment/` (standalone, no broken dependencies):

| Config | K | Vol Acc | Purpose |
|--------|---|---------|---------|
| `rlvae_debug_vanilla` | 1 | OFF | Pure N(μ,Σ) baseline |
| `rlvae_debug_notry` | 1 | ON | Volume acceptance alone |
| `rlvae_debug_novol` | 5 | OFF | Multi-try alone |
| `rlvae_debug_baseline` | 5 | ON | Current problematic setup |
| `rlvae_debug_hightry` | 20 | ON | Test if bias scales with K |

### 3. Analysis Tools
- **`scripts/analyze_z0_diagnostics.py`**: Parses console logs for key metrics
- **`scripts/run_z0_investigation.sh`**: Batch runner for all 5 experiments

---

## 🚀 Running the Investigation

### Option 1: Run All Experiments (Sequential)
```bash
cd /home/alaforgu/scratch/longitudinal_experiments/RlVAE
bash scripts/run_z0_investigation.sh
```

### Option 2: Run Individual Experiments

**Vanilla (currently running):**
```bash
RLVAE_DEBUG=1 python run_experiment.py \
  +experiment=rlvae_debug_vanilla \
  settings.pipeline.mode=three_stage \
  settings.pipeline.run_stage_a=false \
  settings.pipeline.run_stage_b=true \
  settings.pipeline.run_stage_c=true \
  settings.training.stage_overrides.stage_b.enabled=true \
  | tee logs/z0_investigation/debug_vanilla.log
```

**No Multi-Try:**
```bash
RLVAE_DEBUG=1 python run_experiment.py \
  +experiment=rlvae_debug_notry \
  settings.pipeline.mode=three_stage \
  settings.pipeline.run_stage_a=false \
  settings.pipeline.run_stage_b=true \
  settings.pipeline.run_stage_c=true \
  settings.training.stage_overrides.stage_b.enabled=true \
  | tee logs/z0_investigation/debug_notry.log
```

**No Volume Acceptance:**
```bash
RLVAE_DEBUG=1 python run_experiment.py \
  +experiment=rlvae_debug_novol \
  settings.pipeline.mode=three_stage \
  settings.pipeline.run_stage_a=false \
  settings.pipeline.run_stage_b=true \
  settings.pipeline.run_stage_c=true \
  settings.training.stage_overrides.stage_b.enabled=true \
  | tee logs/z0_investigation/debug_novol.log
```

**Baseline (problematic):**
```bash
RLVAE_DEBUG=1 python run_experiment.py \
  +experiment=rlvae_debug_baseline \
  settings.pipeline.mode=three_stage \
  settings.pipeline.run_stage_a=false \
  settings.pipeline.run_stage_b=true \
  settings.pipeline.run_stage_c=true \
  settings.training.stage_overrides.stage_b.enabled=true \
  | tee logs/z0_investigation/debug_baseline.log
```

**High K Test:**
```bash
RLVAE_DEBUG=1 python run_experiment.py \
  +experiment=rlvae_debug_hightry \
  settings.pipeline.mode=three_stage \
  settings.pipeline.run_stage_a=false \
  settings.pipeline.run_stage_b=true \
  settings.pipeline.run_stage_c=true \
  settings.training.stage_overrides.stage_b.enabled=true \
  | tee logs/z0_investigation/debug_hightry.log
```

---

## 📊 Analyzing Results

### Quick Log Analysis
```bash
# Parse each experiment log
for config in vanilla notry novol baseline hightry; do
  echo "=== $config ==="
  python scripts/analyze_z0_diagnostics.py logs/z0_investigation/debug_${config}.log
done
```

### What to Look For

**In Console Logs:**
1. **[CANDIDATE DIAGNOSTICS]** section:
   - All candidates Mahal² mean (should be ~2.0 if unbiased)
   - Selected Mahal² mean
   - Δ Mahal² (selection bias)
   - Corr(h, Mahal²) - positive means high-volume regions are far from μ

2. **[SELECTION STAGE COMPARISON]** section:
   - Δ Mahal² (volume acc) - how much volume acceptance changes distance

3. **[CHI-SQUARED TEST]** and **[KOLMOGOROV-SMIRNOV TEST]**:
   - Deviation % and KS p-value
   - Low p-value (<0.01) = distribution mismatch

**In WandB (project: `rlvae-z0-investigation`):**
- `train/kl_loss` - should be positive for valid configs
- Compare across all 5 runs

---

## 🔍 Expected Findings

### If Multi-Try Selection Causes Bias:
- ✅ `vanilla` (K=1, no vol): Mahal² ≈ 2.0, KL > 0
- ✅ `notry` (K=1, vol ON): Mahal² ≈ 2.0, KL > 0
- ❌ `novol` (K=5, no vol): Mahal² >> 2.0, KL < 0
- ❌ `baseline` (K=5, vol ON): Mahal² >> 2.0, KL < 0
- ❌ `hightry` (K=20, vol ON): Even worse deviation
- **Key**: Positive Corr(h, Mahal²) in multi-try runs

### If Volume Acceptance Causes Bias:
- ✅ `vanilla`: Mahal² ≈ 2.0
- ✅ `novol`: Mahal² ≈ 2.0
- ❌ `notry`: Mahal² >> 2.0
- ❌ `baseline`: Mahal² >> 2.0
- **Key**: Large Δ Mahal² (volume acc) in stage comparison

### If Both Contribute:
- ✅ Only `vanilla` has Mahal² ≈ 2.0
- ❌ All others show deviations
- **Key**: Both stages show significant bias

---

## 📝 Next Steps After Experiments

1. **Parse all logs**: `python scripts/analyze_z0_diagnostics.py logs/z0_investigation/*.log`
2. **Compare WandB metrics**: Check KL divergence trends across 5 runs
3. **Create comparison table**: Summarize Mahal², χ² deviation, KS p-value for each config
4. **Identify root cause**: Multi-try, volume acceptance, or both
5. **Write recommendations**: Document in `docs/z0_investigation_results.md`

---

## 📂 Files Modified/Created

**Modified:**
- `src/rlvae/models/components/riemannian_rhmc_posterior.py`

**Created:**
- `conf/experiment/rlvae_debug_*.yaml` (5 configs)
- `scripts/analyze_z0_diagnostics.py`
- `scripts/run_z0_investigation.sh`
- `docs/z0_investigation_implementation.md`
- `docs/z0_investigation_quick_start.md` (this file)

**Directories:**
- `logs/z0_investigation/` (experiment outputs)

---

## 🐛 Troubleshooting

**If experiments fail:**
- Check that Stage A & B checkpoints exist (run Stage B first if needed)
- Verify RLVAE_DEBUG=1 is set for diagnostic output
- Check WandB login if logging fails

**If diagnostics don't appear:**
- Confirm RLVAE_DEBUG=1 environment variable is set
- Check that the code changes in `riemannian_rhmc_posterior.py` are present

**If configs don't load:**
- Ensure you're using `+experiment=` not `experiment=`
- Configs are now standalone and don't require parent configs

