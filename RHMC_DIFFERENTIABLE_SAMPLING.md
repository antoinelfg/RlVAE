# RHMC for Differentiable Sampling: Why No Accept/Reject

## Core Requirement: Gradient Flow for Training

### The Fundamental Constraint

**Our RHMC posterior MUST be differentiable** because we need to backpropagate through the sampling process to train the VAE. This creates a fundamental design constraint:

```python
# During training:
z = posterior.sample(mu, log_var)  # Must be differentiable!
recon = decoder(z)
loss = reconstruction_loss + kl_divergence
loss.backward()  # Gradients flow through z back to encoder
```

### Why Accept/Reject Breaks Backpropagation

Standard HMC/RHMC uses Metropolis-Hastings accept/reject:

```python
# Standard HMC (NOT differentiable):
z_proposal = leapfrog(z0, rho, steps)
accept_prob = min(1, exp(-H(z_proposal) + H(z0)))
if random() < accept_prob:
    z = z_proposal  # ✅ Accept
else:
    z = z0          # ❌ Reject - DISCRETE DECISION!
```

**Problem**: The accept/reject step is a **discrete, non-differentiable operation**:
- Uses `if/else` based on random comparison
- Gradient flow is blocked at the rejection boundary
- Cannot compute ∂z/∂μ or ∂z/∂θ (encoder params)

### Our Solution: Continuous Approximation via Volume Correction

Instead of hard accept/reject, we use a **manifold-aware potential** that provides continuous guidance:

```python
# Our RHMC (differentiable):
z0 = sample_riemannian(mu, G(mu))           # Differentiable
z = leapfrog(z0, rho, steps)                # Differentiable
# No accept/reject - direct use of z
# But: potential ∇U(z) includes volume correction!
```

**Key Innovation**: The potential gradient includes `∇ log det(G(z))`:

```
U(z) = 0.5·||z||² - 0.5·log det(G(z))
         ↑              ↑
    Gaussian prior   Volume correction
                    (attracts to manifold)

∇U(z) = z - 0.5·∇ log det(G(z))
```

The volume correction term acts as **soft guidance** that:
1. Attracts samples to high-density manifold regions (where det(G) is large)
2. Repels samples from low-density regions (where det(G) is small)
3. Maintains differentiability throughout

## Comparison: Accept/Reject vs Volume Correction

### Standard HMC with Accept/Reject
```
Pros:
- ✅ Exact sampling from target distribution
- ✅ Detailed balance guaranteed
- ✅ Asymptotically correct

Cons:
- ❌ Non-differentiable (discrete decision)
- ❌ Cannot backpropagate through sampling
- ❌ Not usable for gradient-based training
- ❌ Rejection rate can be high (wastes computation)
```

### Our RHMC with Volume Correction
```
Pros:
- ✅ Fully differentiable (continuous guidance)
- ✅ Allows backpropagation through sampling
- ✅ Compatible with gradient-based training
- ✅ No wasted samples (no rejections)
- ✅ Manifold-aware (respects geometry)

Cons:
- ⚠️ Approximate sampling (not exact target distribution)
- ⚠️ Requires careful tuning of step size
- ⚠️ Numerical stability needs attention
```

## Theoretical Justification

### Why Volume Correction Works

The volume element in Riemannian geometry is `√det(G)`. The correct density on the manifold is:

```
p(z) ∝ exp(-||z||²/2) · √det(G(z))
```

Taking the log:

```
log p(z) = -||z||²/2 + (1/2)·log det(G(z)) + const
```

The negative log density (potential energy) is:

```
U(z) = ||z||²/2 - (1/2)·log det(G(z))
```

Thus, the gradient used in leapfrog:

```
∇U(z) = z - (1/2)·∇ log det(G(z))
```

This gradient naturally guides samples toward high-density regions **without needing accept/reject**.

### Relationship to Langevin Dynamics

Our approach is similar to **Riemannian Langevin Dynamics**, which also uses continuous guidance:

```
z_{t+1} = z_t - ε·∇U(z_t) + √(2ε)·ξ_t
```

But we use **Hamiltonian dynamics** (leapfrog) instead, which:
- Preserves energy better (less drift)
- Explores more efficiently (uses momentum)
- Still maintains differentiability

## Practical Implications

### 1. No Exact Distributional Guarantees
- Our samples are **approximately** from the target posterior
- Quality depends on:
  - Step size (smaller = more accurate, but slower exploration)
  - Number of leapfrog steps (more = better mixing)
  - Metric quality (better G = better guidance)

### 2. Gradient Flow is Preserved
```python
# Full gradient path:
μ, log_var = encoder(x)           # ✓ differentiable
z0 = μ + L(G(μ))·ε                 # ✓ differentiable (Riemannian sample)
z = leapfrog(z0, ...)              # ✓ differentiable (no accept/reject!)
recon = decoder(z)                 # ✓ differentiable
loss = ... + KL(z, prior)          # ✓ differentiable
loss.backward()                    # ✓ gradients flow all the way back!
```

### 3. Stability Requires Careful Design
Without accept/reject to "fix" bad proposals:
- ✅ Need accurate potential gradient (volume correction implemented)
- ✅ Need safety bounds (momentum/velocity clipping implemented)
- ✅ Need metric regularization (eps·I added to G and G⁻¹)
- ✅ Need monitoring (spatial confinement, density variation)

## Validation Strategy

Since we can't rely on accept/reject for correctness, we validate via:

1. **Spatial Confinement**: Samples stay near encoder means
   - Target: `mean(||z - μ||_G) ≤ 2.0`

2. **Density Preservation**: Samples respect manifold density
   - Target: `std(log det(G⁻¹(z))) ≤ 0.5`

3. **Iterative Stability**: No progressive drift over epochs
   - Target: `divergence < 50%` over training

4. **Training Metrics**: Effective posterior for VAE training
   - Bounded KL divergence
   - Good reconstruction quality
   - Reasonable ELBO

## Alternative Approaches (Future Work)

If we wanted exact sampling WITHOUT losing differentiability:

1. **Gumbel-Softmax for Accept/Reject**
   - Use continuous relaxation of discrete decision
   - Temperature annealing for accuracy
   - More complex, but potentially more accurate

2. **Normalizing Flows**
   - Learn a flow q(z|x) that approximates p(z|x)
   - Fully differentiable
   - No RHMC needed

3. **Amortized Inference**
   - Train a separate inference network
   - Direct mapping x → z
   - No sampling during forward pass

## Conclusion

**We deliberately avoid accept/reject to maintain differentiability for gradient-based training.**

The manifold-aware potential with volume correction:
- ✅ Provides the **guidance** that accept/reject would give
- ✅ Does so in a **continuous, differentiable** manner
- ✅ Enables **backpropagation** through the sampling process
- ✅ Is the **correct solution** for differentiable Riemannian sampling

This is not a limitation—it's a **design choice** that makes RHMC compatible with modern deep learning optimization!

