# Riemannian RHMC Posterior Implementation

## Overview

This document describes the implementation of a new posterior sampling method that combines **Riemannian initial sampling** with **Riemannian Hamiltonian Monte Carlo (RHMC) exploration**, inspired by the [pyraug library](https://github.com/clementchadebec/pyraug).

## Mathematical Formulation

### The New Posterior Type: `riemannian_rhmc`

Our implementation follows this two-step process:

1. **Riemannian Initial Sampling**: 
   ```
   z₀ ~ N_Riem(μ_φ(x), α G(μ_φ(x)))
   ```
   where:
   - `μ_φ(x)` is the encoder mean
   - `G(μ_φ(x))` is the metric tensor at the encoder mean
   - `α` is a scaling coefficient (default: 1.0)

2. **RHMC Exploration** (K steps):
   ```
   ρ₀ ~ N(0, G(z₀))                    # Initial momentum
   (z_K, ρ_K) = Φ^K(z₀, ρ₀)           # K leapfrog steps
   return z_K                          # Final position
   ```

### Key Differences from Existing Methods

| Method | Initial Sampling | Exploration | Acceptance/Rejection |
|--------|------------------|-------------|---------------------|
| **Standard VAE** | `z ~ N(μ, σ²I)` | None | No |
| **RHVAE Original** | `z ~ N(μ, σ²I)` | RHMC | Yes (for generation) |
| **Riemannian Metric** | `z ~ N_Riem(μ, αG(μ))` | None | No |
| **Our RHMC** | `z ~ N_Riem(μ, αG(μ))` | RHMC | No (differentiable) |

## Implementation Details

### Core Components

1. **`RiemannianRHMCPosterior`** (`src/rlvae/models/components/riemannian_rhmc_posterior.py`)
   - Main sampling class
   - Handles both initial sampling and RHMC exploration
   - Configurable parameters for stability

2. **Integration in `SamplerManager`**
   - Added to the sampling pipeline
   - Accessible via `posterior_type: "riemannian_rhmc"`

3. **Model Configuration**
   - New model config: `conf/model/riemannian_rhmc_vae.yaml`
   - Extends standard Riemannian VAE with RHMC parameters

### Configuration Parameters

```yaml
posterior:
  type: "riemannian_rhmc"
  rhmc_steps: 3              # Number of RHMC steps
  rhmc_step_size: 0.01       # Leapfrog step size
  rhmc_alpha: 1.0            # Coefficient for G(μ) in initial sampling
  eps_regularization: 1e-6   # Numerical stability
  max_grad_norm: 5.0         # Gradient clipping
  min_step_size: 1e-4        # Minimum step size
```

### Hamiltonian Dynamics

The RHMC exploration uses the Hamiltonian:
```
H(z, ρ) = U(z) + (1/2) ρᵀ G⁻¹(z) ρ
```

where:
- `U(z) = (1/2) zᵀz` is the potential energy (standard Gaussian prior)
- `(1/2) ρᵀ G⁻¹(z) ρ` is the kinetic energy with Riemannian metric

### Leapfrog Integration

Each RHMC step uses the leapfrog scheme:
1. **Half momentum step**: `ρ_{1/2} = ρ₀ - (ε/2) ∇_z U(z₀)`
2. **Position step**: `z₁ = z₀ + ε G⁻¹(z₀) ρ_{1/2}`
3. **Half momentum step**: `ρ₁ = ρ_{1/2} - (ε/2) ∇_z U(z₁)`

## Usage

### Basic Usage

```python
# In experiment configuration
model:
  posterior:
    type: "riemannian_rhmc"
    rhmc_steps: 5
    rhmc_step_size: 0.008
    rhmc_alpha: 0.8
```

### Testing

Run the test experiment:
```bash
python -u run_experiment.py experiment=riemannian_rhmc_test wandb.mode=online
```

Or test the implementation directly:
```bash
python test_rhmc_simple.py
```

## Advantages

1. **Riemannian-Aware from Start**: Unlike RHVAE, the initial sampling already respects the learned geometry
2. **Differentiable**: No acceptance/rejection step, preserving gradients for training
3. **Rich Exploration**: RHMC steps provide better posterior coverage than simple sampling
4. **Configurable**: Tunable parameters for different datasets and stability requirements

## Inspiration from PyRAUG

This implementation draws inspiration from the [pyraug library](https://github.com/clementchadebec/pyraug), particularly:

- **RHVAE Architecture**: The general structure of Riemannian VAEs
- **Hamiltonian Dynamics**: The use of RHMC for posterior exploration
- **Metric Integration**: How to incorporate learned metrics into sampling

However, our implementation differs by:
- Starting with Riemannian initial sampling (not Euclidean)
- Removing acceptance/rejection for differentiability
- Integrating into the modular RlVAE architecture

## Performance Considerations

### Computational Cost
- **Initial Sampling**: ~2x cost of standard sampling (Cholesky decomposition)
- **RHMC Steps**: Linear in `rhmc_steps` (typically 3-5 steps)
- **Total**: ~3-6x cost of standard sampling

### Memory Usage
- Additional storage for momentum variables
- Metric tensor computations at each step
- Gradient computations for leapfrog integration

### Stability
- Gradient clipping prevents explosion
- Regularization ensures positive definite matrices
- Adaptive step sizing (future work)

## Future Improvements

1. **Adaptive Step Sizing**: Automatically adjust step size based on acceptance rates
2. **Tempering**: Use simulated tempering for better mixing
3. **Momentum Persistence**: Reuse momentum across sampling calls
4. **Metric Caching**: Cache metric computations for efficiency
5. **Higher-Order Integrators**: Use more accurate integration schemes

## Experimental Results

### Test Results (Simple Mock Model)
- ✅ All components working correctly
- ✅ Riemannian initial sampling functional
- ✅ RHMC exploration stable
- ✅ Differentiable end-to-end

### Full Pipeline Integration
- 🔄 Currently testing with `ellipse_sequences` dataset
- 🔄 Comparing with existing posterior types
- 🔄 Evaluating on 2D latent space visualization

## References

1. [PyRAUG: Data Augmentation with Variational Autoencoders](https://github.com/clementchadebec/pyraug)
2. [RHVAE: Riemannian Hamiltonian Variational Auto-Encoder](https://arxiv.org/abs/2010.11518)
3. [Hamiltonian Monte Carlo on Riemannian Manifolds](https://arxiv.org/abs/1112.4118)
4. [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770)

---

*Implementation by: Assistant (inspired by pyraug)*  
*Date: October 2025*  
*Status: ✅ Implemented, 🔄 Testing*
