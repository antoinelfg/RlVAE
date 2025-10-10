# RHMC Alpha Ramping Bug Fix

## Problem Identified

The RHMC posterior was **not respecting the model's alpha ramping schedule**, causing divergence after epoch 0.

### Symptoms
1. ✅ Clipping works perfectly in epoch 0 (samples bounded to norm ≤ 12)
2. ❌ No clipping in later epochs, massive divergence
3. ❌ Visualization shows "Metric-Aware" instead of "RHMC"

### Root Cause

**Two different alpha values were being used:**

| Component | Alpha Source | Value (Epoch 0) | Value (Epoch 5) |
|-----------|-------------|-----------------|-----------------|
| `PosteriorSampler` (Metric-Aware) | `model.get_current_posterior_alpha()` | 0.25 (ramping) | 0.625 |
| `RiemannianRHMCPosterior` | `config['rhmc_alpha']` | 1.0 (fixed) | 1.0 |

**Alpha ramping schedule (when enabled):**
```python
α(epoch) = (1 - t) * α_start + t * α_end
where t = epoch / ramp_epochs
```

Default ramp: `α_start=0.25 → α_end=1.0` over 10 epochs

**Why this caused divergence:**
- **Epoch 0**: RHMC uses fixed `α=1.0`, but if metric-aware posterior uses `α=0.25`, the **RHMC samples are 2x larger**
- **Later epochs**: As alpha ramps up, **RHMC samples grow exponentially** (Σ = α·G(μ) → larger covariance → larger samples)
- Safety bounds (max_position_norm=12) can't keep up with the growing covariance

## Fix Implemented

### File: `src/rlvae/models/components/riemannian_rhmc_posterior.py`

**Modified `_sample_initial_riemannian()` to use dynamic alpha:**

```python
# Get current alpha (respects ramping schedule if enabled)
current_epoch = getattr(self._ctx['model'], '_current_epoch', None)
if hasattr(self._ctx['model'], 'get_current_posterior_alpha'):
    alpha = self._ctx['model'].get_current_posterior_alpha(current_epoch)
    # Debug: print alpha changes
    if not hasattr(self, '_last_logged_alpha') or abs(alpha - self._last_logged_alpha) > 0.01:
        print(f"🔵 [RHMC] Alpha at epoch {current_epoch}: {alpha:.4f}")
        self._last_logged_alpha = alpha
else:
    alpha = self.rhmc_alpha  # fallback to config value

# Use dynamic alpha
Sigma = alpha * G_mu + self.eps_reg * I
```

**Modified `_compute_log_posterior()` to use same dynamic alpha:**

```python
# Get current alpha (same as sampling)
current_epoch = getattr(self._ctx['model'], '_current_epoch', None)
if hasattr(self._ctx['model'], 'get_current_posterior_alpha'):
    alpha = self._ctx['model'].get_current_posterior_alpha(current_epoch)
else:
    alpha = self.rhmc_alpha

# Use dynamic alpha for covariance
Sigma = alpha * G_mu + self.eps_reg * I
```

## Expected Behavior After Fix

1. **Epoch 0**: 
   - `α = 0.25` (or `posterior_alpha_start`)
   - Small covariance → samples stay close to μ
   - Few/no clipping events

2. **Later epochs** (with ramping):
   - `α` gradually increases
   - Samples explore more, but **growth is controlled**
   - Safety bounds remain effective

3. **Visualizations**:
   - Should now correctly show RHMC posterior (if `sampler_manager` is available)
   - Blue points should track the ramping behavior

## Debug Outputs to Monitor

Look for these messages in logs:

```
🔵 [RHMC] Alpha at epoch 0: 0.2500
🔵 [RHMC] Alpha at epoch 1: 0.3250
🔵 [RHMC] Alpha at epoch 5: 0.6250
🔵 [RHMC] Alpha at epoch 10: 1.0000
🔴 [RHMC] Position norm clipping: X/64 samples exceeded 12.0 (max norm: Y)
```

## Configuration Notes

### To disable alpha ramping (recommended for RHMC stability):

```yaml
# In conf/experiment/*.yaml or conf/model/*.yaml
posterior_alpha_ramp_enabled: false
posterior_local_alpha: 0.5  # or your desired fixed value
```

### To enable controlled ramping:

```yaml
posterior_alpha_ramp_enabled: true
posterior_alpha_start: 0.1    # Start small for stability
posterior_alpha_end: 0.5      # Don't go too large (not 1.0!)
posterior_alpha_ramp_epochs: 20  # Slow ramp
```

## Testing

Run the experiment and check:
1. Alpha values printed match expectations
2. Clipping events are consistent across epochs
3. No sudden divergence after epoch 0
4. Visualizations show correct posterior type

```bash
python /scratch/alaforgu/longitudinal_experiments/RlVAE/run_experiment.py \
  experiment=rlvae_three_stage_long_rhmc_modular \
  data=ellipse_sequences \
  seed=42 \
  model=riemannian_rhmc_vae \
  experiment.run_stage_a=false \
  experiment.run_stage_b=false \
  experiment.run_sampling=false \
  experiment.run_stage_c=true
```

## Related Files

- `src/rlvae/models/components/riemannian_rhmc_posterior.py` (FIXED)
- `src/rlvae/models/components/posterior_sampler.py` (uses `get_current_posterior_alpha`)
- `src/rlvae/models/modrlvae.py` (implements `get_current_posterior_alpha` at line 304)
- `conf/model/riemannian_rhmc_vae.yaml` (RHMC config)
- `conf/experiment/rlvae_three_stage_long_rhmc_modular.yaml` (experiment config)

## Commit Message

```
fix(rhmc): RHMC posterior now respects model alpha ramping schedule

The RHMC posterior was using a fixed alpha value from config while
the rest of the model used a dynamic ramping schedule. This caused
severe divergence after epoch 0 as the effective covariance grew
unchecked.

Now RHMC uses model.get_current_posterior_alpha() to sync with the
global alpha schedule, ensuring consistent behavior across all
posterior samplers.

Fixes #divergence-after-epoch-0
```

