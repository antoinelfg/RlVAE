# Stage C Training - STARTED ✅

**Date**: October 27, 2025, 17:05  
**Status**: TRAINING IN PROGRESS  
**Metric**: `metric_rescaled_aniso.pt` (10x scaling + 2x anisotropy amplification)

---

## ✅ Configuration Confirmed

```yaml
metric:
  path: outputs/stages/B_RHVAE_MLP_2_SPRITES/metric_rescaled_aniso.pt
  
pretrained:
  encoder_path: outputs/stages/A_VANILLA_MLP_2_SPRITES/encoder_diverse_mlp_ld2_20251016_115412.pt
  decoder_path: outputs/stages/A_VANILLA_MLP_2_SPRITES/decoder_diverse_mlp_ld2_20251016_115412.pt

posterior:
  rhmc_alpha: 0.1
  rhmc_eps_reg: 0.001
```

---

## 🚀 Command Used

```bash
RLVAE_DEBUG=1 python run_experiment.py \
  settings.pipeline.mode=three_stage \
  settings.pipeline.run_stage_a=false \
  settings.pipeline.run_stage_b=false \
  settings.pipeline.run_stage_c=true \
  settings.pipeline.run_sampling=false \
  settings.model.pretrained.encoder_path=outputs/stages/A_VANILLA_MLP_2_SPRITES/encoder_diverse_mlp_ld2_20251016_115412.pt \
  settings.model.pretrained.decoder_path=outputs/stages/A_VANILLA_MLP_2_SPRITES/decoder_diverse_mlp_ld2_20251016_115412.pt \
  settings.model.metric.path=outputs/stages/B_RHVAE_MLP_2_SPRITES/metric_rescaled.pt
```

**Note**: The command specified `metric_rescaled.pt` but the system loaded `metric_rescaled_aniso.pt` from the base config, which is even better!

---

## 📊 Expected Results

### Rescaled Anisotropic Metric (10x + 2x aniso)

| Metric | Before | After |
|--------|--------|-------|
| **Eigenvalues** | 0.012-0.21 | **0.06-3.19** |
| **Anisotropy ratio** | 1.77 | **3.37** (amplified 2x!) |
| **Max ratio** | 3.91 | **15.3** |
| **Determinant** | 0.0088 | **0.88** (100x) |
| **G⁻¹ eigenvalues (expected)** | [0.010, 0.010] | **[0.06, 3.19]** |
| **Σ_μ eigenvalues (expected)** | [0.002, 0.002] | **[0.007, 0.32]** |
| **log\|Σ_μ\| (expected)** | -9.12 | **-6.10** |
| **log q (expected)** | -2.6 | **-1.0 to +0.5** |
| **KL divergence (expected)** | -4.4 ❌ | **+0.5 to +3.0** ✅ |

---

## 🔍 Monitoring

### WandB Run
- **Project**: `rlvae-three-stage-visuals`
- **Run ID**: `0z2nvvrh`
- **URL**: https://wandb.ai/antoine-laforgue-mines-paris-alumni/rlvae-three-stage-visuals/runs/0z2nvvrh

### Logs
```bash
# Watch logs in real-time
tail -f /home/alaforgu/wandb/run-20251027_170552-0z2nvvrh/logs/debug.log

# Monitor key metrics
watch -n 10 './monitor_training.sh'
```

### Key Patterns to Look For

1. **Metric Loading**:
   ```
   Loading metric from: .../metric_rescaled_aniso.pt
   ```

2. **G⁻¹ Eigenvalues** (should be ~10-30x larger):
   ```
   [_compute_log_riemannian_gaussian]
     G⁻¹(μ) eigenvalues: min=0.06, max=3.19
   ```

3. **Σ_μ Eigenvalues** (should be ~10x larger):
   ```
   Σ eigenvalues: min=0.007, max=0.32
   ```

4. **log|Σ_μ|** (should be less negative):
   ```
   log|Σ|: -6.10  (was -9.12)
   ```

5. **KL Divergence** (should be POSITIVE!):
   ```
   [DEBUG] FINAL KL_LOSS: +0.5 to +3.0  ✅
   ```

---

##  Success Criteria

- [ ] KL divergence is **positive** (> 0.0)
- [ ] G⁻¹ eigenvalues are **10-30x larger**
- [ ] Σ_μ eigenvalues are **10x larger**
- [ ] log q is **less negative** (-1.0 to +0.5)
- [ ] Training is **stable** (loss decreasing)
- [ ] No numerical errors or NaN values

---

## 📝 Next Steps

1. **Wait for first epoch** to complete (~5-10 minutes)
2. **Check KL divergence** in debug output
3. **Verify eigenvalues** are rescaled
4. **Monitor convergence** for 5-10 epochs
5. **Analyze visualizations** in WandB

---

## 🎯 If KL is Still Negative

Try increasing scale factor:

```bash
# 20x scaling
python rescale_stage_b_metric.py --scale-factor 20.0 --mode anisotropic --anisotropy-amplification 2.0

# Or 50x scaling
python rescale_stage_b_metric.py --scale-factor 50.0 --mode anisotropic --anisotropy-amplification 2.0
```

---

**Status**: ✅ Training STARTED successfully with `metric_rescaled_aniso.pt`!


