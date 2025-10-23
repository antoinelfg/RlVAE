# Metric Representation Audit Checklist

This checklist codifies the sanity tests and logging practices that keep the distinction between the metric `G(z)` and its precision `G⁻¹(z)` explicit throughout the RF‑VAE pipeline.

## 1. Representation Tracking
- `_evaluate_metric(..., with_rep=True)` **must** return `(tensor, rep)` where `rep ∈ {'g','ginv'}`.
- Helpers `_half_logdet_volume`, `_quad_with_G`, `_log_kinetic_density`, `_pushforward_metric_via_flows`, `_resolve_sigma_mu` must accept a representation tag and avoid implicit inversion.
- Grep the codebase for `torch.linalg.inv` and confirm it only appears in verified, local contexts (e.g., explicit diagnostics) and never as “blind” conversion.

## 2. Volume Identities
- **Local consistency:** For random `z`, check `logdet(G(z)) + logdet(G⁻¹(z)) ≈ 0` and `‖G(z)·G⁻¹(z) − I‖_F` is small (<1e-5).
- **Flow push-forward:** Verify
  ```
  half_logdet_push_ginv  ≈ half_logdet_source + sum log|det J|
  ```
  Use the precision branch (`ginv`) for the comparison regardless of the “active” representation.

## 3. Stage‑C Uniform KL
- Ensure the implementation matches
  ```
  KL = E_q[ log q(z₀)  - ½ log det G⁻¹(z_S)  - Σ log|det J|  + (Δ_kin - Δ_vol) ].
  ```
- Confirm `delta_kin` is supplied by the RHMC sampler; when the fallback path runs, it should agree with the sampler value within 1e-3.

## 4. Stage‑B Local Surrogate
- For `kl_metric_eval_point='z'`, penalise `(z - μ)` using whichever representation is available (solve for `G⁻¹` or multiply by `G`).
- For `'mu'`, penalise `μ` with `G(μ)` and add the optional volume term `+½ log det G⁻¹(μ)` without extra conversions.

## 5. Kinetic Term
- `_log_kinetic_density(ρ, z)` must compute `-½ ρᵀG⁻¹ρ + ½ log det G⁻¹ - d/2 log(2π)` either from precision directly or via a local solve when only `G` is available.

## 6. Representation Toggle Test
- Run the same batch twice with `metric_representation='g'` and `'ginv'`. Differences are expected, but:
  - Volume identity (section 2) must continue to hold.
  - No NaN/Inf should appear.
  - Gradient norms should remain finite and similar.

## 7. Suggested Unit Tests
- `test_metric_pair_identity`: probes `logdet(G) + logdet(G⁻¹)` and `G·G⁻¹`.
- `test_pushforward_identity`: asserts `½ log det G'⁻¹ ≈ ½ log det G⁻¹ + Σ log|det J|`.
- `test_kinetic_sign`: compares the fallback kinetic density with the sampler-provided value.
- `test_stageB_quadratic_rep`: ensures solving with the explicit representation yields consistent scalars (tolerance ≤ 1e-5).

## 8. Diagnostics / Logging Guidelines
- Rename volume logs to make the representation explicit (e.g., `half_logdet_ginv`, `neg_half_logdet_g`).
- When printing diagnostics, include the representation tag: `[VOL rep=ginv] half_logdet=…`.
- Track fallbacks in `_robust_inverse_from_cholesky` (e.g., increment `self._inverse_fallback_count` or emit a warning) to surface degenerate metrics.
