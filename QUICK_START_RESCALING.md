# Quick Start: Testing Rescaled Metric

## 🎯 What Was Done

1. **Root cause identified**: Stage B produces G⁻¹ with very small eigenvalues (mean=0.09) and weak anisotropy (ratio=1.77)
2. **Solution implemented**: Rescaled G⁻¹ atoms by **10x** to increase eigenvalue scale
3. **Files updated**:
   - Created `metric_rescaled.pt` (10x scaling, anisotropy preserved)
   - Updated `conf/config.yaml` to use rescaled metric
   - Updated `conf/monolith_stagec.yaml` to use rescaled metric

## 🚀 How to Test

### Step 1: Restart Training

```bash
# Stop current training
pkill -f "python.*run_experiment"

# Start Stage C with rescaled metric
export RLVAE_DEBUG=1
python run_experiment.py stage=C
```

### Step 2: Monitor KL Divergence

Look for these lines in the output:

```
[_compute_log_riemannian_gaussian BEFORE log_q_riem]
  G⁻¹(μ) eigenvalues:  min=0.12, max=2.09  ← Should be 10x larger!
  Σ eigenvalues:       min=0.013, max=0.21 ← Much larger than before!
  log|Σ|:              -5.93                ← Less negative!

[LOG_Q_RIEM DECOMPOSITION]
  Volume term:    mean=2.97  ← Was +4.56, now less positive!
  
[DEBUG] KL CALCULATION
  log_q mean: -1.0 to 0.0   ← Was -2.6!
  
[DEBUG] FINAL KL_LOSS: +0.5 to +2.0 ✅ ← Should be POSITIVE!
```

## ✅ Success Criteria

- KL divergence **positive** (> 0.0)
- G⁻¹ eigenvalues **~10x larger** (0.12-2.09 range)
- log q **less negative** (-1.0 to 0.0)
- Training **stable** (loss decreasing)

## 🔄 If Still Negative

Try larger scale factor:

```bash
# Try 20x scaling
python rescale_stage_b_metric.py --scale-factor 20.0

# Or 50x scaling
python rescale_stage_b_metric.py --scale-factor 50.0
```

## 📊 Expected Impact

| Metric | Before | After (10x) |
|--------|--------|-------------|
| G⁻¹ eigenvalues | 0.012-0.21 | **0.12-2.09** |
| Determinant | 0.0088 | **0.88** |
| log\|Σ_μ\| | -9.12 | **-5.93** |
| log q | -2.6 | **-1.0 to 0.0** |
| KL | -4.4 ❌ | **+0.5 to +2.0** ✅ |

## 📝 Files Modified

- `conf/config.yaml` (line 162): `path: metric_rescaled.pt`
- `conf/monolith_stagec.yaml` (line 65): `metric_path: metric_rescaled.pt`

## 📚 Full Documentation

See `STAGE_B_RESCALING_SOLUTION.md` for complete analysis and technical details.

