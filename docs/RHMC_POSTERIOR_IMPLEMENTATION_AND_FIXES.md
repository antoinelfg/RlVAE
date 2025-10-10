# RHMC Posterior Implementation and Fixes

**Date**: 2025-10-06  
**Summary**: Fixed posterior type routing and implemented proper RHMC posterior support in ModRLVAE

---

## 🔍 Problem Identified

The experiments `rlvae_three_stage_long_rhmc` and `rlvae_three_stage_long_standard` were producing **identical results** despite being configured with different posterior types:
- **RHMC**: `posterior.type: "riemannian_rhmc"`
- **Standard**: `posterior.type: "riemannian_metric"`

### Root Cause

The `run_experiment.py` Stage C initialization code was **hard-coding** the posterior type to `'riemannian_metric'` on lines 2957 and 2963, overriding any configuration:

```python
# OLD CODE (BROKEN):
if 'type' in self.config.model.posterior:
    self.config.model.posterior.type = 'riemannian_metric'  # ❌ Always overwrites!
```

This meant that **all experiments** were using the same `riemannian_metric` posterior, regardless of configuration.

---

## ✅ Solution Implemented

### 1. Fixed Posterior Type Routing (`run_experiment.py`)

**Changed lines 2951-2972** to preserve the configured posterior type:

```python
# NEW CODE (FIXED):
if 'type' in self.config.model.posterior:
    pass  # ✅ Keep the configured posterior type!
else:
    # Only set default if not already configured
    self.config.model.posterior.type = 'riemannian_metric'

# Sync top-level posterior_type with posterior.type
if hasattr(self.config.model, 'posterior') and hasattr(self.config.model.posterior, 'type'):
    self.config.model.posterior_type = self.config.model.posterior.type
```

### 2. Created New Modular Experiment Configs

Created two new experiment configs that properly use **ModRLVAE** (which supports all posterior types):

#### a) **RHMC Posterior** (`rlvae_three_stage_long_rhmc_modular.yaml`)
```yaml
stage_c:
  model: modular_rlvae  # ✅ Uses ModRLVAE

model:
  posterior:
    type: "riemannian_rhmc"  # ✅ RHMC posterior
    rhmc_steps: 3
    rhmc_step_size: 0.01
    rhmc_alpha: 1.0
  posterior_type: "riemannian_rhmc"
```

#### b) **Standard Posterior** (`rlvae_three_stage_long_standard_modular.yaml`)
```yaml
stage_c:
  model: modular_rlvae  # ✅ Uses ModRLVAE

model:
  posterior:
    type: "riemannian_metric"  # ✅ Standard Riemannian
  posterior_type: "riemannian_metric"
```

---

## 🧠 RHMC Posterior Implementation Details

The RHMC posterior was **already correctly implemented** in:
- `src/rlvae/models/components/riemannian_rhmc_posterior.py`
- `src/rlvae/models/components/sampler_manager.py`

### Mathematical Formulation

The RHMC posterior implements a **two-step process**:

#### Step 1: Riemannian Initial Sampling
```
z₀ ~ N_Riem(μ_φ(x), α G(μ_φ(x)))
```
- **Covariance**: `Σ = α G(μ) + ε I`
- **Cholesky sampling**: `z₀ = μ + L ε` where `L Lᵀ = Σ`

This is **different from standard RHVAE** which uses Euclidean initial sampling:
```
# RHVAE Original: z₀ ~ N(μ, σ²I)  (Euclidean)
# Ours:          z₀ ~ N_Riem(μ, αG(μ))  (Riemannian from start)
```

#### Step 2: RHMC Exploration (K leapfrog steps, NO acceptance/rejection)
```
ρ₀ ~ N(0, G(z₀))                    # Sample momentum
(z_K, ρ_K) = Φ^K(z₀, ρ₀)           # K leapfrog steps
return z_K                          # Final position (differentiable!)
```

**Leapfrog Integration**:
```
1. ρ_{1/2} = ρ₀ - (ε/2) ∇_z U(z₀)     # Half momentum step
2. z₁ = z₀ + ε G⁻¹(z₀) ρ_{1/2}        # Full position step
3. ρ₁ = ρ_{1/2} - (ε/2) ∇_z U(z₁)     # Half momentum step
```

**Key Properties**:
- ✅ **Differentiable**: No acceptance/rejection, all gradients flow
- ✅ **Riemannian-aware**: Uses `G(z)` and `G⁻¹(z)` throughout
- ✅ **Numerically stable**: Gradient clipping, Cholesky fallback
- ✅ **Configurable**: Steps, step size, alpha, regularization

### Configuration Parameters

```yaml
posterior:
  type: "riemannian_rhmc"
  rhmc_steps: 3              # Number of leapfrog steps (0 = just initial sampling)
  rhmc_step_size: 0.01       # Leapfrog step size
  rhmc_alpha: 1.0            # Scaling for Σ = α G(μ)
  eps_regularization: 1e-6   # Numerical stability: Σ = α G + ε I
  max_grad_norm: 5.0         # Gradient clipping for stability
  min_step_size: 1e-4        # Minimum allowed step size
```

---

## 📊 Comparison: Posterior Types

| Posterior Type | Initial Sampling | Exploration | Differentiable | Implementation |
|----------------|------------------|-------------|----------------|----------------|
| **`gaussian`** | `z ~ N(μ, σ²I)` | None | ✅ Yes | Standard VAE |
| **`riemannian_metric`** | `z ~ N_Riem(μ, αG(μ))` | None | ✅ Yes | Current baseline |
| **`riemannian_rhmc`** | `z ~ N_Riem(μ, αG(μ))` | K leapfrog steps | ✅ Yes | **NEW** |
| **RHVAE Original** | `z ~ N(μ, σ²I)` | RHMC + accept/reject | ❌ No (in generation) | Reference |

---

## 🎯 Encoder/Decoder Training in Stage C

### Are They Trained?
**YES** - Both encoder and decoder are trained in Stage C.

### Initialization Process:
1. **Stage A**: Train vanilla VAE → save encoder/decoder weights
2. **Stage B**: Extract metric from Stage A → save metric tensors (`C`, `M`)
3. **Stage C**: 
   - **Load** pretrained encoder/decoder from Stage A
   - **Load** fixed metric from Stage B
   - **Continue training** encoder/decoder with:
     - Riemannian constraints (via posterior and KL)
     - Flow dynamics (temporal evolution)
   - **Metric is frozen** (`trainable: false`)

**Key Point**: The encoder/decoder are **fine-tuned** in Stage C, not frozen!

---

## 🔧 Metric Training Status

### Is Metric Being Trained?
**NO** - The metric is **frozen** in both experiments:

```yaml
stage_c:
  allow_metric_updates: false
  update_metric_during_training: false

model:
  metric:
    trainable: false  # ✅ Metric is frozen
```

The metric tensors (`C`, `M`) from Stage B are loaded as **fixed parameters** and not updated during Stage C training.

---

## 📈 Loss Function Analysis

### KL Divergence Computation

The model uses **Riemannian KL divergence**:

```python
KL[q(z|x) || p_R(z)]
```

Where:
- **Posterior**: `q(z|x) = N_Riem(μ, αG(μ))` (evaluated at encoder mean)
- **Prior**: `p_R(z) ∝ √det(G(z)) exp(-½ zᵀ G(z) z)` (evaluated at sample)

**Components**:
1. **Trace term**: `tr(G(z) * diag(exp(log_var)))`
2. **Quadratic term**: `μᵀ G(z) μ`
3. **Log-determinant**: `log(det(G(z))) - log(det(diag(exp(log_var))))`

### ⚠️ Curvature Effect: G(z) vs G(μ)

This is a **critical design choice** that warrants investigation:

#### The Issue:
- **Posterior sampling** uses `G(μ)` at encoder mean
- **KL computation** uses `G(z)` at sampled point

This creates a **curvature correction** effect where the KL measures the difference between:
1. The metric at the **deterministic encoder output** `μ`
2. The metric at the **sampled latent** `z`

#### Why Investigate This?

**Pros** (intentional feature):
- Encourages posterior samples to stay near encoder mean
- Captures local curvature information
- More expressive than pure Euclidean KL

**Cons** (potential issue):
- **Inconsistency**: Posterior uses `G(μ)`, prior uses `G(z)`
- **Gradient mismatch**: Different evaluation points may cause instability
- **Mathematical validity**: KL formula assumes same covariance point

**Possible Fix**:
```python
# Option 1: Evaluate KL at μ (consistent with posterior)
G_mu = self.G(mu)  # Instead of G(z)
KL = compute_kl_with_G(mu, log_var, z, G_mu)

# Option 2: Evaluate posterior at z (consistent with prior)
z0 = self._sample_at_z(mu, log_var)  # Sample then refine

# Option 3: Use Monte Carlo estimate
KL_mc = (log q(z|x) - log p(z)).mean()  # Explicit densities
```

**Recommendation**: Run ablation study comparing:
- `kl_metric_eval_point: "mu"` (consistent with posterior)
- `kl_metric_eval_point: "z"` (current, with curvature)

---

## 🚀 How to Run New Experiments

### Standard Posterior (Baseline)
```bash
python /scratch/alaforgu/longitudinal_experiments/RlVAE/run_experiment.py \
  experiment=rlvae_three_stage_long_standard_modular \
  data=ellipse_sequences \
  wandb.mode=online \
  seed=42
```

### RHMC Posterior (New)
```bash
python /scratch/alaforgu/longitudinal_experiments/RlVAE/run_experiment.py \
  experiment=rlvae_three_stage_long_rhmc_modular \
  data=ellipse_sequences \
  wandb.mode=online \
  seed=42
```

### Key Differences:
- ✅ **Modular RLVAE**: Supports all posterior types
- ✅ **Proper routing**: Posterior type is preserved
- ✅ **RHMC dynamics**: 3 leapfrog steps for exploration
- ✅ **Same base config**: 200 epochs A&C, frozen metric

---

## 📊 Expected Outcomes

### What Should Be Different Now?

1. **Training Dynamics**:
   - RHMC should show **more exploration** in latent space
   - Standard should be **more localized** around encoder mean

2. **Latent Space Structure**:
   - RHMC: Richer geodesic paths, better coverage
   - Standard: Tighter clusters, faster convergence

3. **Reconstruction Quality**:
   - RHMC: Potentially better generalization (more diverse samples)
   - Standard: Potentially better fit (less variance)

4. **Metrics to Compare**:
   - Reconstruction loss (MSE)
   - KL divergence (should be different!)
   - Latent space coverage (PCA visualization)
   - Sample quality (visual inspection)

---

## 🔬 Future Investigations

### Completed:
- ✅ Fixed posterior type routing
- ✅ Created modular experiment configs
- ✅ Documented RHMC implementation

### Pending:
- 🔍 **Curvature Effect Study**: Impact of `G(z)` vs `G(μ)` evaluation
  - Run ablation with `kl_metric_eval_point: "mu"` vs `"z"`
  - Analyze gradient flow and training stability
  - Measure impact on reconstruction and latent structure

- 🔍 **RHMC Hyperparameter Tuning**:
  - Optimize `rhmc_steps` (1, 3, 5, 10)
  - Optimize `rhmc_step_size` (0.001, 0.01, 0.1)
  - Optimize `rhmc_alpha` (0.5, 1.0, 2.0)

- 🔍 **Metric Adaptation Experiments**:
  - Try `trainable: true` with slow learning rate
  - Compare frozen vs adaptive metric performance
  - Analyze metric evolution over training

---

## ✅ Validation Checklist

Before running experiments, verify:

- [ ] `run_experiment.py` has posterior routing fix (lines 2951-2972)
- [ ] New configs exist: `rlvae_three_stage_long_*_modular.yaml`
- [ ] ModRLVAE is imported correctly
- [ ] RHMC posterior sampler is in `src/rlvae/models/components/`
- [ ] Metric is frozen in both configs
- [ ] WandB project is set correctly
- [ ] Data path points to `ellipse_sequences`

---

## 📝 Notes

- The RHMC implementation is **fully differentiable** (no accept/reject)
- The modular architecture supports easy posterior swapping
- The metric remains frozen as requested
- All experiments use the same Stage A/B configuration for fair comparison

---

**Ready to run!** The experiments should now properly compare RHMC vs standard posteriors.

