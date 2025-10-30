# Next Investigation Steps: Tracing the Full Σ_μ Construction Pipeline

## 🎯 Current Status

We've confirmed that:
1. ✅ `log_q_riem` formula is mathematically correct
2. ✅ There's a ~5-6 unit discrepancy between RHMC's `log_q` and manual reconstruction
3. ✅ Stabilization diagnostics are in place for `_safe_cholesky`
4. ❓ **Open Question**: Where in the pipeline does Σ_μ get modified?

## 🔍 Suspected Transformation Points

The construction of Σ_μ involves several transformations:

```
1. G(μ)                        ← Riemannian metric at μ
     ↓
2. G⁻¹(μ) = inverse(G(μ))     ← Inverse metric
     ↓
3. G⁻¹_clamped                 ← Clamp eigenvalues to min_cov_eig
     ↓  (line 617-620 of riemannian_rhmc_posterior.py)
4. G⁻¹_normalized              ← Optional geomean normalization (det ≈ 1)
     ↓  (line 621-630)
5. Σ_μ = α * G⁻¹_norm + ε * I  ← Final covariance
     ↓  (line 638-651)
6. Σ_stabilized                ← Optional Cholesky jitter
     ↓  (in _safe_cholesky if Cholesky fails)
7. Used in log_q_riem
```

## 🛠️ Recommended Diagnostics to Add

### 1. In `_get_inverse_metric` (lines ~615-650)

Add diagnostics to track eigenvalue clamping and normalization:

```python
def _get_inverse_metric(self, mu: torch.Tensor) -> torch.Tensor:
    """Get G^{-1}(μ) with optional normalization."""
    Ginv = self._ctx['model'].Ginv(mu)
    Ginv = _symmetrize(Ginv)
    
    # Clamp spectrum for robustness
    try:
        evals, evecs = torch.linalg.eigh(Ginv.float())
        evals_orig = evals.clone()  # Save original for diagnostics
        evals = torch.clamp(evals, min=self.min_cov_eig)
        
        # DIAGNOSTIC: Track clamping
        if os.environ.get("RLVAE_DEBUG", "0") == "1":
            n_clamped = (evals_orig < self.min_cov_eig).sum().item()
            if n_clamped > 0:
                print(f"\n[GINV CLAMPING]")
                print(f"  {n_clamped} eigenvalues clamped to min_cov_eig={self.min_cov_eig:.6f}")
                print(f"  Original eigenvalues: min={evals_orig.min().item():.6e}, max={evals_orig.max().item():.6f}")
                print(f"  Clamped eigenvalues:  min={evals.min().item():.6e}, max={evals.max().item():.6f}")
        
        # Optional normalization by geometric mean so det(Ginv_norm)=1
        mode = str(getattr(self, 'sigma_normalization_mode', 'geomean')).lower()
        if mode == 'geomean':
            geomean_eig = torch.exp(torch.log(evals + 1e-12).mean(dim=-1, keepdim=True))
            evals_norm = evals / (geomean_eig + 1e-12)
            
            # DIAGNOSTIC: Track normalization
            if os.environ.get("RLVAE_DEBUG", "0") == "1":
                logdet_before = torch.log(evals).sum(dim=-1).mean().item()
                logdet_after = torch.log(evals_norm).sum(dim=-1).mean().item()
                print(f"\n[GINV NORMALIZATION]")
                print(f"  mode: {mode}")
                print(f"  Eigenvalues before normalization: {evals[0].tolist()}")
                print(f"  Eigenvalues after normalization:  {evals_norm[0].tolist()}")
                print(f"  log|G⁻¹| before: {logdet_before:.6f}")
                print(f"  log|G⁻¹| after:  {logdet_after:.6f}")
                print(f"  Δ log|G⁻¹|:     {logdet_after - logdet_before:+.6f}")
            
            evals = evals_norm
        
        Ginv = torch.einsum('...ij,...j,...kj->...ik', evecs, evals, evecs).to(Ginv.dtype)
        # ... rest of the function
```

### 2. In `_make_covariance` (lines ~638-651)

Add diagnostics to track the final Σ_μ construction:

```python
def _make_covariance(self, Ginv: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Construct Σ_μ = α·G^{-1}(μ) + ε·I
    """
    d = Ginv.shape[-1]
    eye = torch.eye(d, device=Ginv.device, dtype=Ginv.dtype).unsqueeze(0)
    
    # Scale for standard Riemannian Gaussian proposal or for RHMC kinetic
    if getattr(self, 'use_metric_basis_norm', False):
        Ginv_norm = Ginv / torch.linalg.det(Ginv).unsqueeze(-1).unsqueeze(-1).pow(1.0 / d)
    else:
        Ginv_norm = Ginv
    
    # DIAGNOSTIC: Track components
    if os.environ.get("RLVAE_DEBUG", "0") == "1":
        eigvals_ginv = torch.linalg.eigvalsh(Ginv_norm)
        logdet_ginv = torch.linalg.slogdet(Ginv_norm)[1]
        
        alpha_term = alpha * Ginv_norm
        eps_term = self.eps_reg * eye
        Sigma = alpha_term + eps_term
        
        eigvals_alpha = torch.linalg.eigvalsh(alpha_term)
        eigvals_sigma = torch.linalg.eigvalsh(Sigma)
        logdet_sigma = torch.linalg.slogdet(Sigma)[1]
        
        print(f"\n[MAKE COVARIANCE]")
        print(f"  alpha:                {alpha:.6f}")
        print(f"  eps_reg:              {self.eps_reg:.6e}")
        print(f"  G⁻¹_norm eigenvalues: min={eigvals_ginv.min().item():.6e}, max={eigvals_ginv.max().item():.6f}")
        print(f"  log|G⁻¹_norm|:        {logdet_ginv.mean().item():.6f}")
        print(f"  α·G⁻¹ eigenvalues:    min={eigvals_alpha.min().item():.6e}, max={eigvals_alpha.max().item():.6f}")
        print(f"  Σ_μ eigenvalues:      min={eigvals_sigma.min().item():.6e}, max={eigvals_sigma.max().item():.6f}")
        print(f"  log|Σ_μ|:             {logdet_sigma.mean().item():.6f}")
        print(f"  ε term contribution:  {(eigvals_sigma - eigvals_alpha).mean().item():.6e}")
    
    if hasattr(self, '_stabilize_spd'):
        Sigma = self._stabilize_spd(_symmetrize(Sigma), self.min_cov_eig)
    else:
        Sigma = alpha * Ginv_norm + self.eps_reg * eye
    
    return self._stabilize_spd(_symmetrize(Sigma), self.min_cov_eig)
```

## 📊 Expected Output

With these diagnostics, you'll see a complete trace:

```
[GINV CLAMPING]
  2 eigenvalues clamped to min_cov_eig=0.001000
  Original eigenvalues: min=5.234e-05, max=18.234
  Clamped eigenvalues:  min=1.000e-03, max=18.234

[GINV NORMALIZATION]
  mode: geomean
  Eigenvalues before normalization: [0.001, 18.234]
  Eigenvalues after normalization:  [0.0074, 135.45]  ← Notice the scale change!
  log|G⁻¹| before: 2.903
  log|G⁻¹| after:  0.000  ← Forced to ~0 by geomean
  Δ log|G⁻¹|:     -2.903

[MAKE COVARIANCE]
  alpha:                1.000000
  eps_reg:              1.000e-04
  G⁻¹_norm eigenvalues: min=7.400e-03, max=135.450
  log|G⁻¹_norm|:        0.000
  α·G⁻¹ eigenvalues:    min=7.400e-03, max=135.450
  Σ_μ eigenvalues:      min=7.500e-03, max=135.451  ← Small ε contribution
  log|Σ_μ|:             -2.587  ← This is what goes into log_q_riem
  ε term contribution:  1.000e-04

[_compute_log_riemannian_gaussian]
  Calling log_q_riem with min_cov_eig = 0.001000

[LOG_Q_RIEM STABILIZATION]
  min_eig (jitter):     0.001000
  was_stabilized:       False
  Original Σ:
    eigenvalues:        min=0.007500, max=135.451
    log|Σ|:             -2.587  ← Matches [MAKE COVARIANCE]
```

## 🎯 What This Will Tell Us

1. **If `[GINV CLAMPING]` appears frequently**:
   - G⁻¹(μ) has very small eigenvalues
   - Clamping is changing the spectrum significantly
   - This is a likely source of discrepancy

2. **If `[GINV NORMALIZATION]` shows large Δ log|G⁻¹|**:
   - Geomean normalization is drastically changing the determinant
   - Disabling normalization (`sigma_normalization_mode: 'none'`) may help
   - Or the normalization is working as intended to avoid volume issues

3. **If eigenvalues at each step differ greatly from manual reconstruction**:
   - We've identified the exact transformation causing the issue
   - Can then decide if it's a bug or an intentional stabilization

## 🧪 Testing Strategy

1. **Add the diagnostics** as shown above
2. **Run training** with `RLVAE_DEBUG=1` for 1-2 batches
3. **Compare outputs**:
   - `[SIGMA DEBUG]` in `loss_manager.py` (manual reconstruction)
   - `[GINV CLAMPING]`, `[GINV NORMALIZATION]`, `[MAKE COVARIANCE]` (RHMC internal)
4. **Identify the step** with the largest change in eigenvalues or log-determinant

## 🔧 Potential Fixes (Based on Findings)

### If Geomean Normalization is the Issue

**Config change**:
```yaml
sigma_normalization_mode: 'none'  # or 'trace' or 'maxeig'
```

### If Clamping is Too Aggressive

**Config change**:
```yaml
min_cov_eig: 1e-5  # Lower threshold (current default: 1e-3)
```

### If α Scaling is the Issue

**Config change**:
```yaml
rhmc_alpha: 0.5  # Try different values (current default: 1.0)
```

## 📝 Implementation Checklist

- [ ] Add diagnostics to `_get_inverse_metric` (eigenvalue clamping)
- [ ] Add diagnostics to `_get_inverse_metric` (geomean normalization)
- [ ] Add diagnostics to `_make_covariance` (Σ_μ construction)
- [ ] Run training with `RLVAE_DEBUG=1` for 1-2 batches
- [ ] Analyze output to identify transformation with largest impact
- [ ] Test potential fixes (disable normalization, adjust min_cov_eig, etc.)
- [ ] Document findings and update configuration accordingly

## 🎓 Key Questions to Answer

1. **Is geomean normalization active?**
   - Check if `sigma_normalization_mode: 'geomean'` in config
   - If yes, this forces det(G⁻¹_norm) ≈ 1, which drastically changes log|Σ_μ|

2. **How many eigenvalues are being clamped?**
   - If many, this indicates G⁻¹(μ) is poorly conditioned
   - May need to revisit the metric tensor architecture

3. **What's the contribution of α and ε to Σ_μ?**
   - α * G⁻¹ should dominate
   - ε * I is just for numerical stability
   - If ε is significant, α might be too small or G⁻¹ has tiny eigenvalues

4. **Does _stabilize_spd add additional modifications?**
   - Check the implementation of `_stabilize_spd` if it exists
   - It might apply spectral shift or other regularization

## 📚 Related Documentation

- `INVESTIGATION_SUMMARY.md`: Current findings and hypothesis
- `LOG_Q_STABILIZATION_GUIDE.md`: How to use the existing diagnostics
- `TEST_LOG_Q_DIAGNOSTICS.md`: Documentation of log_q decomposition
- `STABILIZATION_SUMMARY.md`: Pushforward metric stabilization

---

**Last Updated**: October 27, 2025  
**Status**: Ready for next phase of investigation  
**Blocker**: Need to add diagnostics to `_get_inverse_metric` and `_make_covariance`

