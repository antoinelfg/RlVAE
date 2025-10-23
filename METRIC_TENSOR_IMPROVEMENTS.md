# MetricTensor Critical Improvements

## Overview
Applied 6 critical tweaks to fix metric behavior and ensure compatibility with LossManager expectations, including surgical fixes for "funnels to infinity" issues.

## Changes Made

### 1. **Representation Clarification**
- **Issue**: Fixed mode mixes G⁻¹ atoms, but LossManager needs to know this
- **Solution**: Added `self.atoms_are = "ginv"` flag for defensive programming
- **Action Required**: Ensure LossManager uses `metric_representation="ginv"` (not "g")

### 2. **Atom Normalization** 
- **Issue**: Global log|G⁻¹| scale was drifting due to absolute scales of atoms
- **Solution**: Normalize each centroid matrix to unit geometric mean determinant
- **Implementation**: Added atom normalization in `load_pretrained()`:
  ```python
  # Normalize atoms to unit geomean determinant
  with torch.no_grad():
      L = _cholesky_spd(M)  # [K, d, d]
      logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1).abs() + 1e-18).sum(-1)  # [K]
      d = M.shape[-1]
      s = torch.exp(-logdet / d)  # [K]
      M = M * s.view(-1, 1, 1)
  ```
- **Result**: Removes global-scale bias, makes volume term more informative

### 3. **Softmax Weighting**
- **Issue**: Manual normalization was numerically unstable
- **Solution**: Use proper softmax for weights
- **Implementation**: Replaced manual normalization with:
  ```python
  weights = torch.softmax(-distances_sq / (temp2 + 1e-12), dim=-1)
  ```
- **Result**: More stable gradients and numerics, prevents weird plateaus

### 4. **Relative Regularization**
- **Issue**: Fixed λ was flattening variation (low contrast in heatmaps)
- **Solution**: Scale regularization relative to atom strength
- **Implementation**: 
  ```python
  # Scale regularization relative to atom strength
  d = G_inv.shape[-1]
  mean_trace = torch.diagonal(G_inv, dim1=-2, dim2=-1).sum(-1).mean()
  alpha = self.regularization.to(z.dtype)  # use as α not λ
  lambda_eff = (alpha * (mean_trace / d)).clamp_min(1e-9)
  ```
- **Result**: Prevents λ from dominating, maintains field structure

### 5. **Background Identity Mixing** ⭐ **NEW**
- **Issue**: "Funnels to infinity" in log|det(G⁻¹)| field - metric becomes too anisotropic far from centroids
- **Solution**: Blend centroid mixture with identity precision in the tails
- **Implementation**: Distance-aware blending with identity:
  ```python
  # Background identity mixing (distance-aware)
  dmin_sq = dist_sq.min(dim=-1)[0]  # [B]
  bg_r2 = (4.0 * self.temperature.to(z.dtype)) ** 2  # decay radius
  beta = torch.exp(-dmin_sq / (bg_r2 + 1e-12))  # [B] in [0,1]
  beta = torch.clamp(beta, min=self.bg_floor)  # avoid pure mixture
  
  # Blend: near centroids beta≈1 → mixture; far beta→bg_floor → mostly identity
  G_inv = beta.view(-1, 1, 1) * G_inv_mix + (1.0 - beta).view(-1, 1, 1) * (self.bg_strength * I)
  ```
- **Result**: Prevents "teleport lanes" to infinity, smooth basins around centroids

### 6. **Spectral Clamping** ⭐ **NEW**
- **Issue**: Convex mixtures of SPD matrices can have tiny λ_min, creating low-volume corridors
- **Solution**: Clip eigenvalues to sane band before adding regularization
- **Implementation**: 
  ```python
  # Spectral clamping BEFORE regularization
  evals, evecs = torch.linalg.eigh(G_inv)  # [B, d], [B, d, d]
  floor = torch.tensor(self.eig_floor_abs, device=z.device, dtype=z.dtype)
  evals = torch.clamp(evals, min=floor)
  if self.eig_ceiling is not None:
      ceil = torch.tensor(self.eig_ceiling, device=z.device, dtype=z.dtype)
      evals = torch.clamp(evals, max=ceil)
  G_inv = (evecs @ (evals.unsqueeze(-1) * evecs.transpose(-1, -2)))  # reconstruct
  ```
- **Result**: Guarantees λ_min(G⁻¹) ≥ eig_floor_abs everywhere, eliminates razor-thin tunnels

## Nice-to-Have Improvements

### 7. **Eigenvalue Computation**
- **Change**: Use `torch.linalg.eigvalsh()` instead of `torch.linalg.eigvals()`
- **Benefit**: Faster, more stable for symmetric matrices

### 8. **Representation Flag**
- **Added**: `self.atoms_are = "ginv"` for defensive programming
- **Benefit**: Clear documentation of what the module returns

### 9. **Enhanced Diagnostics**
- **Added**: `lambda_min_Ginv_batch_min` to diagnostics
- **Benefit**: Spot eigenvalue collapses immediately, monitor spectral health

## Expected Results

With these changes, you should see:

1. **Better Volume Field Contrast**: 
   - Volume heatmaps should show clear structure (not flat)
   - `preview/half_logdet_volume_*` should have meaningful variation

2. **Stable KL Terms**:
   - `diagnostics/pushforward_consistency_mean` should drop near 0
   - `preview/sum_logdet_flow` should be more stable

3. **Reduced Drift**:
   - Fewer `[WARN] Abnormal logdet magnitude` events
   - More stable `jacobian_min_singular_mean` (no collapse to ~0)

4. **No More "Funnels to Infinity"**: ⭐ **NEW**
   - Smooth basins around centroids instead of razor-thin tunnels
   - No low-volume "highways" that RHMC can exploit
   - Well-behaved geodesics and logdet fields

## Integration Requirements

**CRITICAL**: The LossManager must use `metric_representation="ginv"` because:
- Fixed mode mixes G⁻¹ atoms: `G_inv = weighted_matrices.sum(dim=1) + regularization`
- This means `compute_inverse_metric()` returns G⁻¹
- LossManager needs to know this to compute correct volume terms

## Testing

All tests pass:
- ✅ SPD and inverse consistency
- ✅ Logdet relations  
- ✅ Trainable mode SPD
- ✅ Distance properties
- ✅ Weight normalization
- ✅ Numerical stability
- ✅ Device/dtype handling
- ✅ Volume field contrast

## Usage

```python
# Correct usage with LossManager
metric = MetricTensor(latent_dim=d, trainable=False, ...)
# LossManager must use metric_representation="ginv"
loss_manager = LossManager(metric_representation="ginv", ...)
```

The MetricTensor is now production-ready and should eliminate representation mismatch issues in your LossManager/Flow stack!
