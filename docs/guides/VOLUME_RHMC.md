# Volume-Element RHMC Sampler (Default)

This guide documents the default RHMC sampler used across the pipeline: `RHVAEVolumeElementHMCSampler` in `src/models/samplers/hmc_sampler.py`.

- Target density: π(z) ∝ sqrt(det(G^{-1}(z)))
- Momentum: Euclidean Gaussian with a tempering schedule
- Updates: Euclidean leapfrog on ∇ log sqrt det(G^{-1})

## Usage

- RHVAE training/visualization: the experiment class `src/models/rhvae_experiment.py` now instantiates `RHVAEVolumeElementHMCSampler` by default (`self.rhmc_sampler`). All RHMC visual panels and prior samples use this sampler.
- Standalone testing: `scripts/rhmc_from_checkpoint.py` supports `sampler=volume` and also a synthetic `ring` metric for quick validation.

Example:
```bash
python -u scripts/rhmc_from_checkpoint.py synthetic=ring sampler=volume n_samples=4096 mcmc_steps=200 n_lf=15 eps_lf=0.03 beta_zero=1.0 out_dir=outputs/rhmc_ring_demo device=auto | cat
```

## Key parameters
- `n_lf`: number of leapfrog steps (default 15–50 in our runs)
- `eps_lf`: leapfrog step size (try 0.02–0.05)
- `beta_zero`: tempering parameter (1.0 = no tempering amplification)

## Notes
- We use raw (unnormalized) centroid weights in the metric adapter for samplers and visuals.
- PCA(2)-aligned visualizations subtract the PCA mean before projecting centroids and samples.
- Acceptance rate is printed and stored on the sampler as `last_acceptance_rate`.


