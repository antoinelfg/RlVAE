# RHMC Posterior Overview

The RHMC posterior used in the modular RLVAE family pushes a Riemannian normal through a leapfrog flow:

\[
q^{\text{RHMC}}(z_K \mid x) = \big(\Phi^K\big)_\# \, \mathcal{N}_{\text{Riem}}\!\left(\mu_\phi(x), \Sigma_\phi(\mu)\right)
\quad\text{with}\quad
\Sigma_\phi(\mu) = \alpha\,\widehat{G}^{-1}(\mu) + \varepsilon I.
\]

Sampling proceeds in two stages:

1. Draw \(z_0 \sim \mathcal{N}_{\text{Riem}}(\mu, \Sigma_\phi(\mu))\) using the current metric estimate.
2. Roll out \(K\) RHMC steps without accept/reject to obtain \(z_K\). The differentiable flow keeps gradients intact for training.

The sampler API now exposes both the initial samples and diagnostics:

```python
zK, log_q0, z0, traj = sample_riemannian_rhmc_posterior(
    mu,
    log_var,
    return_log_prob=True,
    return_initial=True,
    return_traj=True,
)
```

- `zK`: final RHMC samples used in the decoder.
- `log_q0`: log-density of the Riemannian normal at \(z_0\).
- `z0`: initial samples prior to the RHMC flow.
- `traj`: trajectory metadata (`rhmc_steps`, `step_size`, `alpha`, and placeholder `jac_logdet`).

When the jacobian flag is enabled, `traj["jac_logdet"]` will host future correction estimates (currently set to `None`).

## KL Estimation Modes

`LossManager.compute_total_loss` now accepts RHMC-specific options:

| Option | Default | Description |
| ---- | ---- | ---- |
| `rhmc_kl_mode` | `mc` | Monte Carlo KL. Use `jac` to subtract supplied Jacobian estimates or `bound` to fall back to the geodesic bound. |
| `rhmc_kl_source` | `z0` | Which sample to anchor log \(q\) computations on. Use `z0` when the sampler returns initial statistics, otherwise fall back to `zk`. |
| `rhmc_kl_jacobian` | `false` | Enable Jacobian subtraction when `mode="jac"` and a trajectory estimate is available. |

The Monte Carlo path reduces to
\[
\mathrm{KL} \approx \mathbb{E}_{q}\left[\log q(z_0 \mid x) - \log p(z_K)\right]
\]
when `rhmc_kl_source="z0"`. Set `RHMC_KL_MODE`, `RHMC_KL_SOURCE`, or `RHMC_KL_JACOBIAN` in the environment to override experiment defaults.

## Hydra Configuration Snippets

Add the new knobs to your model config:

```yaml
model:
  posterior_type: riemannian_rhmc
  rhmc_alpha: 1.0
  rhmc_eps_reg: 1e-4
  rhmc_steps: 2
  rhmc_step_size: 0.01
  rhmc_kl_mode: mc        # {mc, jac, bound}
  rhmc_kl_source: z0      # {z0, zk}
  rhmc_kl_jacobian: false
```

For quick visual diagnostics:

```bash
python -u scripts/visualize_enhanced_kl.py \
  model.rhmc_kl_mode=mc \
  model.rhmc_kl_source=z0 \
  +viz.save=true
```

Setting `model.rhmc_kl_mode=jac model.rhmc_kl_jacobian=true` enables the Jacobian correction once an estimator is plugged in.

## Why the Jacobian Is Off by Default

The RHMC flow is symplectic in the continuous limit, so the log-determinant of the Jacobian is theoretically zero. Empirically, finite step sizes introduce drift that requires bespoke estimators (e.g., Hutchinson trace estimators). Until a stable estimator ships, the flag stays off to avoid injecting noisy gradients.
