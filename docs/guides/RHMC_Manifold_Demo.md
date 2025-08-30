# RHMC Manifold Demo: How the Sampler Was Computed

This guide explains exactly how the RHMC sampler in `tests/test_rhmc_manifold_visual_demo.py` is constructed, how the target density is defined, and how the visuals are generated.

## Overview
- We build a synthetic 2D manifold by defining a position-dependent inverse metric \(G^{-1}(z)\) using RBF interpolation over a ring of centroids.
- We run a Riemannian Hamiltonian Monte Carlo (RHMC) sampler that uses \(G(z)\) in its kinetic energy and adds the standard manifold volume correction via `logdet(G(z))` inside the Hamiltonian.
- We set the target density to a Gaussian ring (annulus) to encourage samples to concentrate on a circle of radius \(r_0\).
- We save three visuals under `outputs/rhmc_manifold_visual_demo/`.

## Synthetic Manifold Metric
1. Create `K` centroids on a ring of radius ~2.2 in 2D.
2. At each centroid, define an SPD anchor matrix for \(G^{-1}\) (anisotropic: larger tangential eigenvalue, smaller radial eigenvalue). This biases motion along the ring.
3. For any point \(z\), compute RBF weights
   \[ w_k(z) = \exp\left(-\frac{\lVert z-c_k \rVert^2}{T^2}\right) \]
   and interpolate
   \[ G^{-1}(z) = \sum_k w_k(z)\, M_k + \lambda I. \]
4. Obtain \(G(z) = (G^{-1}(z))^{-1}\) on demand.

In code (inside the minimal metric model used by the sampler):
```python
G_inv(z) = sum_k w_k(z) * M_k + lambda * I
G(z) = inv(G_inv(z))
```
Where `T` is the interpolation temperature and `lambda` is a small regularizer to keep \(G^{-1}\) well-conditioned.

## Target Density (Ring)
We set a ring-shaped target density by defining
\[ \log \pi(z) = -\frac{1}{2} \frac{(\lVert z \rVert - r_0)^2}{\sigma^2} \]
with gradient
\[ \nabla_z \log \pi(z) = - \frac{\lVert z \rVert - r_0}{\sigma^2} \frac{z}{\lVert z \rVert}. \]
This focuses samples near radius \(r_0\) with radial width controlled by \(\sigma\).

In the test, we use:
- `r0 = 2.2`
- `sigma = 0.25` (tighter ring)

These are injected into the sampler as `sampler.log_pi` and `sampler.grad_func`.

## RHMC Sampler Details
We now default to the RHVAE volume-element sampler (`RHVAEVolumeElementHMCSampler`) in `src/models/samplers/hmc_sampler.py`. It targets π(z) ∝ sqrt(det(G^{-1}(z))) with a tempered Euclidean momentum update, matching the robust behavior observed in ring-metric tests. For reference, the classic `RiemannianHMCSampler` remains available.
\[ H(z, \rho) = -\log \pi(z) + \tfrac{1}{2} \rho^\top G^{-1}(z) \rho + \tfrac{1}{2} \log\det G(z). \]
Key steps:
- Momentum initialization: \(\rho \sim \mathcal{N}(0, G(z))\) via a Cholesky factor of \(G(z)\).
- Generalized leapfrog updates: position integrates using \(G^{-1}(z)\), momenta updated using `grad_func(z)` (i.e., \(-\nabla_z \log \pi(z))`).
- Metropolis-Hastings acceptance using the Hamiltonian difference.

Hyperparameters used in the demo (tight ring):
- `mcmc_steps_nbr = 110`
- `n_lf = 24`
- `eps_lf = 0.018`
- `beta_zero = 1.0`
- `temperature (metric interpolation) = 0.5`
- `regularization lambda = 0.03`

These values balance a tighter annulus with a reasonable acceptance rate.

## Visuals Produced
The test saves three figures to `outputs/rhmc_manifold_visual_demo/`:
- `metric_determinant_contour.png`: Contour of `det(G^{-1}(z))` over a grid, with centroids overlaid.
- `rhmc_samples_scatter.png`: Scatter of RHMC samples.
- `combined_overlay.png`: Samples overlaid on `det(G^{-1})` contours plus centroids.

## How to Reproduce
- Run the test directly (no pytest required):
```bash
python - <<'PY'
from tests.test_rhmc_manifold_visual_demo import test_rhmc_sampler_visual_demo

test_rhmc_sampler_visual_demo()
print('OK')
PY
```
- Outputs will appear in `outputs/rhmc_manifold_visual_demo/`.

## Notes and Extensions
- Tightness: decrease `sigma` or increase `n_lf`/`mcmc_steps_nbr` for an even sharper ring; tune `eps_lf` to keep good acceptance.
- Geometry: the metric anchors control anisotropy; increasing the tangential eigenvalue relative to radial strengthens ring-following behavior.
- Dual RHMC: `DualRiemannianHMCSampler` is available and treats the geometry in a complementary way (using \(G^{-1}\) in the kinetic energy). It requires z to carry gradient during the leapfrog; use it if you prefer that formulation.




