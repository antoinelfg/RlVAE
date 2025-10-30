# Guide: log_q Stabilization Diagnostics

## 📋 Overview

This guide documents the enhanced diagnostics added to `log_q_riem` to investigate why `log_q` values are unexpectedly negative, leading to negative KL divergence.

## 🔍 What Was Added

### 1. Enhanced `_safe_cholesky` Function

**File**: `src/rlvae/models/components/riemannian_rhmc_posterior.py` (lines 23-53)

The `_safe_cholesky` function now returns a third value indicating whether stabilization (jitter) was applied:

```python
def _safe_cholesky(matrix: torch.Tensor, jitter: float) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Returns:
        chol: Cholesky factor
        stabilized: Stabilized matrix (original or with jitter added)
        was_stabilized: True if jitter was added
    """
```

**Key Changes**:
- Returns `(chol, matrix, False)` if Cholesky succeeds without modification
- Returns `(chol, stabilized, True)` if jitter was added (`stabilized = matrix + jitter * I`)

### 2. Stabilization Diagnostics in `log_q_riem`

**File**: `src/rlvae/models/components/riemannian_rhmc_posterior.py` (lines 143-164)

Added detailed diagnostics controlled by `RLVAE_DEBUG=1`:

```python
if os.environ.get("RLVAE_DEBUG", "0") == "1":
    print(f"\n[LOG_Q_RIEM STABILIZATION]")
    print(f"  min_eig (jitter):     {min_eig:.6f}")
    print(f"  was_stabilized:       {was_stabilized}")
    print(f"  Original Σ:")
    print(f"    eigenvalues:        min={eigvals_orig.min():.6f}, max={eigvals_orig.max():.6f}")
    print(f"    log|Σ|:             {logdet_orig.mean():.6f}")
    if was_stabilized:
        print(f"  Stabilized Σ:")
        print(f"    eigenvalues:        min={eigvals_stab.min():.6f}, max={eigvals_stab.max():.6f}")
        print(f"    log|Σ|:             {logdet_stab.mean():.6f}")
        print(f"    Δ log|Σ|:           {(logdet_stab - logdet_orig).mean():+.6f}")
```

**What It Shows**:
- The `min_eig` parameter (jitter value) used for stabilization
- Whether `_safe_cholesky` actually applied jitter
- Original Σ eigenvalues and log-determinant
- If stabilized: the modified eigenvalues, log-determinant, and the change in log|Σ|

### 3. Parameter Logging in `_compute_log_riemannian_gaussian`

**File**: `src/rlvae/models/components/riemannian_rhmc_posterior.py` (lines 1023-1025)

Added logging of the `min_cov_eig` parameter:

```python
if os.environ.get("RLVAE_DEBUG", "0") == "1":
    print(f"\n[_compute_log_riemannian_gaussian]")
    print(f"  Calling log_q_riem with min_cov_eig = {self.min_cov_eig:.6f}")
```

**What It Shows**:
- The exact `min_cov_eig` value being passed to `log_q_riem`
- This value is typically `1e-3` by default (line 209 of `riemannian_rhmc_posterior.py`)

## 🚀 Usage

### Enable Full Diagnostics

Set the environment variable before running your training:

```bash
export RLVAE_DEBUG=1
# Then run your training command
```

Or inline:

```bash
RLVAE_DEBUG=1 python your_training_script.py
```

### Expected Output

During training, you'll now see output like:

```
[_compute_log_riemannian_gaussian]
  Calling log_q_riem with min_cov_eig = 0.001000

[LOG_Q_RIEM STABILIZATION]
  min_eig (jitter):     0.001000
  was_stabilized:       False
  Original Σ:
    eigenvalues:        min=0.011916, max=17.975180
    log|Σ|:             -2.263456
```

If stabilization is triggered:

```
[LOG_Q_RIEM STABILIZATION]
  min_eig (jitter):     0.001000
  was_stabilized:       True
  Original Σ:
    eigenvalues:        min=0.000050, max=20.500000
    log|Σ|:             -7.824563
  Stabilized Σ:
    eigenvalues:        min=0.001050, max=20.501000
    log|Σ|:             -4.952341
    Δ log|Σ|:           +2.872222
```

## 🔍 Interpretation

### Understanding the Discrepancy

The diagnostics help identify why the RHMC-computed `log_q` differs from a manual reconstruction:

1. **No Stabilization** (`was_stabilized: False`):
   - Σ_μ is already well-conditioned
   - The eigenvalues are above `min_cov_eig`
   - No jitter was added
   - The original Σ is used for log_q calculation

2. **Stabilization Applied** (`was_stabilized: True`):
   - Σ_μ had very small eigenvalues or failed Cholesky
   - Jitter was added: `Σ_stabilized = Σ + min_cov_eig * I`
   - The modified Σ_stabilized is used for log_q calculation
   - **This changes the log-determinant**, making the volume term less negative
   - **Result**: `log_q` is less negative than expected

### Key Diagnostic Points

1. **Check `min_cov_eig` Value**:
   - Default is `1e-3` (0.001)
   - Configured via `min_cov_eig` in RHMC config
   - Always >= `eps_reg` (enforced in line 210-211)

2. **Compare Eigenvalues**:
   - If `min(eigvals_orig) < min_cov_eig`: Stabilization is likely
   - If `min(eigvals_orig) >> min_cov_eig`: No stabilization expected

3. **Analyze Δ log|Σ|**:
   - Positive Δ log|Σ| → log|Σ_stabilized| > log|Σ_original|
   - Volume term = `-0.5 * log|Σ|`
   - Larger log|Σ| → less negative volume term
   - Less negative volume term → less negative log_q
   - Less negative log_q → **more negative KL divergence**

## 🧪 Testing

A test script is provided: `test_log_q_stabilization.py`

Run it with:

```bash
python test_log_q_stabilization.py
```

This tests:
1. Well-conditioned Σ (no stabilization)
2. Poorly-conditioned Σ (stabilization expected)
3. Batch with mixed conditioning

## 🎯 Next Steps for Investigation

Based on the diagnostic output, you can:

1. **If `was_stabilized: True` frequently**:
   - The covariance matrices Σ_μ are poorly conditioned
   - Consider adjusting `rhmc_alpha` to scale Σ_μ = α * G⁻¹(μ) + ε * I
   - Consider increasing `eps_reg` to improve base conditioning

2. **If `was_stabilized: False` but log_q still unexpectedly negative**:
   - The issue is not in `_safe_cholesky` stabilization
   - Check the covariance construction in `_make_covariance`
   - Verify the eigenvalue clamping in `_get_inverse_metric` (line 617-620)

3. **Compare with Manual Reconstruction**:
   - The `[LOG_Q FROM RHMC]` diagnostics in `loss_manager.py` reconstruct Σ_μ
   - Compare the eigenvalues and log|Σ| between:
     - Manual reconstruction (in loss_manager diagnostics)
     - RHMC internal computation (these new diagnostics)
   - Any discrepancy indicates where the issue lies

## 📝 Related Files

- `src/rlvae/models/components/riemannian_rhmc_posterior.py`: RHMC posterior implementation
- `src/rlvae/models/components/loss_manager.py`: KL divergence calculation and diagnostics
- `TEST_LOG_Q_DIAGNOSTICS.md`: Documentation of the log_q decomposition diagnostics
- `STABILIZATION_SUMMARY.md`: Documentation of pushforward metric stabilization

## ⚙️ Configuration Parameters

Relevant RHMC parameters that affect stabilization:

```yaml
rhmc_alpha: 1.0           # Scaling for Σ_μ = α * G⁻¹(μ) + ε * I
eps_regularization: 1e-4  # Base regularization epsilon
min_cov_eig: 1e-3        # Minimum eigenvalue for Cholesky (jitter value)
```

In your config file (e.g., `conf/model/rlvae_rotation_rhmc_stage2.yaml`).

