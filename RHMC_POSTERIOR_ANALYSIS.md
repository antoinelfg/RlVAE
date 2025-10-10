# RHMC Posterior Divergence Analysis

## Current Implementation Recap

### Algorithm Overview
The RHMC posterior uses a two-stage sampling process:

1. **Initial Riemannian Sampling**: Sample z₀ ~ N_Riem(μ, α·G(μ))
   - Compute metric tensor at encoder mean: `G_mu = G(μ)`
   - Build covariance: `Σ = α·G(μ) + ε·I`
   - Cholesky decomposition: `L = chol(Σ)`
   - Sample: `z₀ = μ + L·ε` where `ε ~ N(0,I)`

2. **RHMC Exploration** (if rhmc_steps > 0):
   - Sample momentum: `ρ ~ N(0, G(z))`
   - Leapfrog integration (without accept/reject):
     - Half momentum step: `ρ ← ρ - 0.5·h·∇U(z)`
     - Full position step: `z ← z + h·G⁻¹(z)·ρ`
     - Half momentum step: `ρ ← ρ - 0.5·h·∇U(z)`

### Current Parameters (Baseline Config)
```yaml
rhmc_steps: 1                    # Single leapfrog step
rhmc_step_size: 0.01            # Step size h
rhmc_alpha: 1.0                 # Scaling for initial covariance
eps_regularization: 1e-6        # Numerical stability

# Safety bounds (added)
max_momentum_norm: 5.0          # Clip ||ρ||
max_velocity_norm: 2.0          # Clip ||v|| = ||G⁻¹·ρ||
max_position_step: 1.0          # Bound ||Δz||
max_position_norm: 12.0         # Clip ||z||
```

## Critical Issues Identified

### 1. **RHMC Design Constraint: No Accept/Reject (BY DESIGN for Backprop)**
- **Requirement**: We MUST use leapfrog output directly WITHOUT accept/reject to maintain differentiability
- **Why**: Accept/reject breaks the gradient flow needed for backpropagation through the sampling process
- **Trade-off**: We sacrifice distributional guarantees for gradient-based training
- **Solution**: Use manifold-aware potential to guide samples correctly WITHOUT accept/reject
- **Key insight**: The volume correction term `∇ log det(G(z))` acts as a "soft guidance" replacing the hard accept/reject step

### 2. **Potential Gradient Mismatch**
```python
def _compute_potential_gradient(self, z: torch.Tensor) -> torch.Tensor:
    return z.clone()  # ∇U(z) = z assumes N(0,I) prior
```
- **Assumption**: Gaussian prior p(z) = N(0, I) → U(z) = 0.5·||z||² → ∇U(z) = z
- **Problem**: This ignores the Riemannian metric entirely!
- **Correct form**: ∇U(z) should include metric-dependent terms
- **Mathematical issue**: For Riemannian manifolds, the potential should be:
  ```
  U(z) = 0.5·z^T·z + 0.5·log det(G(z))  [volume correction]
  ∇U(z) = z + 0.5·∇_z log det(G(z))
  ```

### 3. **Metric Evaluation Instability**
- **G(μ)**: Evaluated at encoder mean (relatively stable)
- **G(z)**: Evaluated at RHMC-evolved position (potentially far from manifold)
- **Issue**: As z drifts away from training data region, G(z) becomes:
  - Ill-conditioned (large condition number)
  - Extrapolated (neural network out of training distribution)
  - Unstable (small eigenvalues → large G⁻¹ values)

### 4. **Velocity Explosion Mechanism**
```python
velocity = torch.einsum('bij,bj->bi', G_inv_reg, rho)
```
- If G(z) has small eigenvalues → G⁻¹(z) has large eigenvalues
- Even with clipped momentum ||ρ|| ≤ 5, velocity can be huge
- Example: if λ_min(G) = 1e-3, then λ_max(G⁻¹) = 1e3
  - ||v|| = ||G⁻¹·ρ|| could be ~1000 even with ||ρ|| = 1
- Current clip at ||v|| ≤ 2 helps but is a band-aid

### 5. **Progressive Drift Accumulation**
- **Epoch 0-3**: z stays near μ, G(z) ≈ G(μ), relatively stable
- **Epoch 4-5**: Small drifts accumulate, z starts exploring low-density regions
- **Epoch 6+**: G(z) becomes ill-conditioned → velocity explosions → hard clipping kicks in → unnatural dynamics → divergence

### 6. **Lack of Manifold Awareness**
- No check if z remains on learned manifold
- No density-based rejection or correction
- No projection back to high-density regions
- Clipping is Euclidean (||z|| ≤ 12) but manifold is non-Euclidean

## Hypothesis Analysis

### ❌ Incorrect Hypotheses
1. **"Clipping harder will fix it"**: No - clipping is a symptom treatment, not root cause fix
2. **"Smaller step size will stabilize"**: Partially - delays divergence but doesn't prevent it
3. **"More regularization (larger ε) helps"**: Marginally - masks metric issues but degrades quality

### ✅ Core Problems
1. **Missing accept/reject → no distributional guarantees**
2. **Wrong potential gradient → incorrect dynamics**
3. **Metric extrapolation → numerical instability**
4. **No manifold constraint → unbounded drift**

## Proposed Solutions (Ranked by Impact)

### Option A: **Manifold-Aware Potential** (RECOMMENDED)
**Goal**: Keep samples on the learned manifold by modifying the potential

```python
def _compute_potential_gradient(self, z: torch.Tensor) -> torch.Tensor:
    """
    Manifold-aware potential with volume correction and elastic recall
    """
    # Standard Gaussian prior term
    grad = z.clone()
    
    # Volume correction: -0.5·∇ log det(G(z))
    # This attracts samples to high-density regions
    try:
        G = self._ctx['model'].G(z)
        G_inv = torch.linalg.inv(G + self.eps_reg * torch.eye(...))
        
        # Compute ∂G/∂z via finite differences (approximate)
        delta = 1e-4
        grad_log_det = torch.zeros_like(z)
        for i in range(z.shape[-1]):
            z_plus = z.clone()
            z_plus[..., i] += delta
            G_plus = self._ctx['model'].G(z_plus)
            dG_dzi = (G_plus - G) / delta
            grad_log_det[..., i] = torch.einsum('bij,bij->b', G_inv, dG_dzi)
        
        grad = grad - 0.5 * grad_log_det
    except:
        pass  # Fall back to standard Gaussian
    
    return grad
```

**Pros**: Theoretically correct, attracts to manifold
**Cons**: Expensive (requires metric gradients), approximate

### Option B: **Adaptive Step Size Based on Density**
**Goal**: Reduce step size in low-density regions

```python
def _adaptive_step_size(self, z: torch.Tensor, base_step: float) -> torch.Tensor:
    """Reduce step size when log det(G⁻¹) is low (far from manifold)"""
    try:
        G_inv = self._ctx['model'].G_inv(z)
        log_det = torch.logdet(G_inv)
        # If log_det < threshold, reduce step size
        density_score = torch.sigmoid(log_det + 5.0)  # Normalize around -5
        adaptive_step = base_step * density_score.unsqueeze(-1)
        return adaptive_step.clamp(min=1e-4)
    except:
        return torch.full_like(z[..., :1], base_step)
```

### Option C: **Projection to Nearest High-Density Point**
**Goal**: After each leapfrog step, project z back to manifold

```python
def _project_to_manifold(self, z: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Project z towards encoder mean if it drifts too far"""
    # Compute distance in metric space
    delta = z - mu
    G_mu = self._ctx['model'].G(mu)
    mahalanobis_dist = torch.sqrt(torch.einsum('bi,bij,bj->b', delta, G_mu, delta))
    
    # If distance > threshold, pull back
    threshold = 3.0  # ~3 standard deviations
    scale = torch.where(
        mahalanobis_dist > threshold,
        threshold / (mahalanobis_dist + 1e-12),
        torch.ones_like(mahalanobis_dist)
    )
    return mu + scale.unsqueeze(-1) * delta
```

### Option D: **Reduce to Zero RHMC Steps** (SAFEST, FALLBACK)
**Goal**: Disable RHMC entirely, use only Riemannian initial sampling

```yaml
posterior:
  rhmc_steps: 0  # No RHMC exploration
  rhmc_alpha: 1.0
```

**Pros**: Guaranteed stability, still uses metric
**Cons**: Less exploration, might underestimate posterior variance

### Option E: **Hybrid: Initial Only + Noise Injection**
**Goal**: Add controlled noise instead of deterministic RHMC

```python
def sample_riemannian_rhmc_posterior(self, mu, log_var):
    # Riemannian initial sample
    z0 = self._sample_initial_riemannian(mu, log_var)
    
    # Instead of RHMC, add small metric-aware noise
    if self.rhmc_steps > 0:
        G_z = self._ctx['model'].G(z0)
        L = torch.linalg.cholesky(0.1 * G_z + self.eps_reg * I)
        noise = torch.einsum('bij,bj->bi', L, torch.randn_like(z0))
        z_final = z0 + noise
    else:
        z_final = z0
    
    return z_final
```

## Immediate Action Plan

### Phase 1: Emergency Stabilization (NOW)
1. **Set `rhmc_steps: 0`** in config → disable RHMC completely
2. **Test if divergence stops** → confirms RHMC is the issue
3. **Monitor KL divergence** → should be stable

### Phase 2: Implement Manifold-Aware Potential (NEXT)
1. Add volume correction term to `_compute_potential_gradient`
2. Use finite differences for ∇ log det(G)
3. Test with `rhmc_steps: 1`, small `step_size: 0.005`

### Phase 3: Add Safety Layers
1. Implement adaptive step size based on density
2. Add projection to high-density regions
3. Monitor: std(log det(G⁻¹)) for posterior samples

### Phase 4: Validation
1. Check spatial confinement: mean ||z - μ||_G ≤ 2.0
2. Check density variation: std(log det(G⁻¹(z))) ≤ 0.5
3. Visualize: posterior samples stay near centroids/encoder means

## Key Metrics to Monitor

1. **Spatial Confinement**: `mean(||z_posterior - μ||_G)` should be ≤ 2.0
2. **Density Variation**: `std(log det(G⁻¹(z_posterior)))` should be ≤ 0.5
3. **Condition Number**: `cond(G(z))` should be < 100
4. **KL Divergence**: Should remain bounded (< 20)
5. **Sample Magnitude**: `max(||z||)` should be < 10

## What Could Be Made Better

### Short Term
1. ✅ Disable RHMC (set steps=0) for immediate stability
2. ✅ Fix potential gradient to include volume correction
3. ✅ Add manifold projection after sampling

### Medium Term
1. ✅ **Manifold-aware potential with volume correction** (implemented - replaces accept/reject!)
2. Use adaptive step size based on local curvature/density
3. Add tempering schedule for exploration vs exploitation
4. Monitor and log manifold quality metrics during training

### Long Term
1. Learn optimal RHMC parameters via meta-learning
2. Hybrid approach: Use normalizing flows for posterior approximation
3. Implement geodesic-based sampling on manifold
4. **Note**: Accept/reject methods are NOT compatible with gradient-based training

## Conclusion

**Root Cause**: RHMC without accept/reject + incorrect potential gradient + metric extrapolation

**Immediate Fix**: Set `rhmc_steps: 0` (disable RHMC)

**Proper Fix**: Implement manifold-aware potential with volume correction

**Best Practice**: Use Riemannian initial sampling only, or implement full Riemannian HMC with accept/reject

