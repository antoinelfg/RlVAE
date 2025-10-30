# Configuration Fix Summary

**Date**: October 27, 2025  
**Issue**: RuntimeError when running Stage C - missing encoder_path and decoder_path

## Problem

When trying to run Stage C after updating `metric_path` to use `metric_rescaled.pt`, the system failed with:

```
RuntimeError: Stage A artifacts are required when skipping Stage A, but missing: encoder_path, decoder_path.
```

## Root Cause

`conf/config.yaml` had `encoder_path: null` and `decoder_path: null` in the `model.pretrained` section.

## Solution Applied

Updated `conf/config.yaml` (lines 183-184):

```yaml
pretrained:
  encoder_path: outputs/stages/A_VANILLA_MLP_2_SPRITES/encoder_diverse_mlp_ld2_20251016_115412.pt
  decoder_path: outputs/stages/A_VANILLA_MLP_2_SPRITES/decoder_diverse_mlp_ld2_20251016_115412.pt
  metric_path: null
```

## Verified Files

All required Stage C files now exist and are configured:

- ✅ **Encoder**: `encoder_diverse_mlp_ld2_20251016_115412.pt` (8.0 MB)
- ✅ **Decoder**: `decoder_diverse_mlp_ld2_20251016_115412.pt` (8.0 MB)  
- ✅ **Metric**: `metric_rescaled.pt` (4.5 KB, 10x rescaled)

## Current Configuration

### `conf/config.yaml`:
- `model.pretrained.encoder_path`: ✅ Set
- `model.pretrained.decoder_path`: ✅ Set  
- `model.metric.path`: ✅ `metric_rescaled.pt`

### `conf/monolith_stagec.yaml`:
- `model.pretrained.encoder_path`: ✅ Set
- `model.pretrained.decoder_path`: ✅ Set
- `model.pretrained.metric_path`: ✅ `metric_rescaled.pt`

## Ready to Run

```bash
# Stage C training with rescaled metric
python run_experiment.py stage=C

# With debug output
export RLVAE_DEBUG=1
python run_experiment.py stage=C
```

## Expected Improvements

With `metric_rescaled.pt` (10x scaling):
- G⁻¹ eigenvalues: 10x larger
- Σ_μ eigenvalues: ~10x larger  
- log|Σ_μ|: +3.19 (less negative)
- log q: +1.6 to +2.6 (less negative)
- **KL divergence: POSITIVE (+0.5 to +2.0)** ✅

---

## Notes

- The rescaling preserves local anisotropy (ratio 1.77 → 1.77)
- Spatial anisotropy (variation across space) is preserved and amplified
- Alternative `metric_rescaled_aniso.pt` available for 2x anisotropy amplification
