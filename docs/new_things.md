# New Developments Log

## 2025-10-06: RHMC Posterior Fix and Modular RLVAE Integration

### 🔧 Critical Bug Fix: Posterior Type Routing
**Fixed in**: `run_experiment.py` (lines 2951-2972, 3012-3084)

**Problem**: Stage C was hard-coding all experiments to use `riemannian_metric` posterior, AND training defaults were overriding experiment config.

**Solution 1 - Preserve configured type** (lines 2951-2972):
- Keeps `posterior.type` from config (e.g., `riemannian_rhmc`)
- Only sets default `riemannian_metric` if not configured
- Syncs `posterior_type` with `posterior.type` automatically

**Solution 2 - Force-sync guard** (lines 3012-3084):
- Reads intended posterior type from experiment config (highest priority)
- Force-syncs to ALL config locations:
  - `model.posterior.type`
  - `model.posterior_type`
  - `training.model.posterior.type`
  - `training.model.posterior_type`
- Prints verification showing all posterior type locations
- Prevents training defaults from overriding experiment config

**Impact**: Experiments now **actually use** the configured posterior type (RHMC, standard, etc.)

---

### ✨ New Experiment Configurations

Created two new modular experiment configs for proper RHMC comparison:

#### 1. **`rlvae_three_stage_long_rhmc_modular.yaml`**
- Uses **ModRLVAE** (supports all posterior types)
- **RHMC posterior**: Riemannian initial + 3 leapfrog steps
- 200 epochs Stage A & C
- Frozen metric

#### 2. **`rlvae_three_stage_long_standard_modular.yaml`**
- Uses **ModRLVAE** (for fair comparison)
- **Standard Riemannian posterior**: No RHMC exploration
- 200 epochs Stage A & C
- Frozen metric

**Usage**:
```bash
# Standard baseline
python run_experiment.py experiment=rlvae_three_stage_long_standard_modular \
    data=ellipse_sequences wandb.mode=online seed=42

# RHMC posterior
python run_experiment.py experiment=rlvae_three_stage_long_rhmc_modular \
    data=ellipse_sequences wandb.mode=online seed=42
```

---

### 📚 Documentation Added

**New file**: `docs/RHMC_POSTERIOR_IMPLEMENTATION_AND_FIXES.md`

Comprehensive documentation covering:
- Root cause analysis of the bug
- RHMC posterior mathematical formulation
- Comparison table of all posterior types
- Encoder/decoder training details
- Metric training status
- Loss function analysis with curvature effect discussion
- Usage instructions and expected outcomes

---

### 🧠 RHMC Posterior Details

The RHMC posterior (`riemannian_rhmc`) implements:

1. **Riemannian Initial Sampling**: `z₀ ~ N_Riem(μ, αG(μ))`
   - Uses metric at encoder mean for initial sample

2. **RHMC Exploration**: K leapfrog steps
   - Hamiltonian dynamics: `H(z, ρ) = U(z) + ½ ρᵀ G⁻¹(z) ρ`
   - No acceptance/rejection (fully differentiable)
   - Configurable steps, step size, and regularization

**Key Difference from Standard RHVAE**:
- RHVAE: Euclidean initial (`z ~ N(μ, σ²I)`) + RHMC + accept/reject
- Ours: Riemannian initial (`z ~ N_Riem(μ, αG(μ))`) + RHMC + no accept/reject

---

### 🎯 Investigation Findings

#### Encoder/Decoder Training (Stage C)
- ✅ **ARE trained** (not frozen)
- Initialized from Stage A checkpoints
- Fine-tuned with Riemannian constraints
- Flow dynamics added for temporal evolution

#### Metric Training (Stage C)
- ❌ **NOT trained** (frozen as requested)
- Loaded from Stage B checkpoints
- Tensors `C` (centroids) and `M` (metric matrices) are fixed

#### Prior Implementation
- **Riemannian Gaussian**: `p(z) ∝ √det(G(z)) exp(-½ zᵀ G(z) z)`
- Volume element: `√det(G(z))` encourages high-curvature regions
- Quadratic form: `zᵀ G(z) z` uses metric for distance

#### Loss Function
- **Reconstruction**: MSE × 255 (non-normalized scale)
- **KL Divergence**: Riemannian KL with curvature correction
- **Flow Loss**: Log-determinant of flow Jacobians
- **Loop Penalty**: Optional cycle consistency (currently disabled)

---

### ⚠️ Open Question: Curvature Effect

**Issue**: KL computation evaluates metric at **different points** than posterior:
- **Posterior sampling**: Uses `G(μ)` at encoder mean
- **KL divergence**: Uses `G(z)` at sampled point

**Why this matters**:
- Creates curvature correction effect
- May cause gradient inconsistencies
- Mathematical validity of KL formula unclear

**Recommendation**: Run ablation study with:
- `kl_metric_eval_point: "mu"` (consistent with posterior)
- `kl_metric_eval_point: "z"` (current, with curvature)

See documentation for full analysis and proposed fixes.

---

### 🚀 Next Steps

1. **Run new experiments**: Compare RHMC vs standard modular configs
2. **Analyze results**: Latent space, reconstruction, KL divergence
3. **Investigate curvature**: Ablation study on `G(z)` vs `G(μ)`
4. **Tune RHMC hyperparameters**: Steps, step size, alpha
5. **Document findings**: Update based on experimental results

---

### ✅ Files Modified

- `run_experiment.py`: Fixed posterior type routing (lines 2951-2972)

### ✅ Files Created

- `conf/experiment/rlvae_three_stage_long_rhmc_modular.yaml`
- `conf/experiment/rlvae_three_stage_long_standard_modular.yaml`
- `docs/RHMC_POSTERIOR_IMPLEMENTATION_AND_FIXES.md`

---

**Status**: Ready for experimentation! 🎉
