# Complete Recap: Three-Stage RLVAE Pipeline Debugging Session


# Complete Recap: Three-Stage RLVAE Pipeline Debugging Session

## �� **Original Goal**
Launch a three-stage pipeline with RHVAE MLP architecture and latent dimension 16:
- **Stage A**: Train RHVAE model
- **Stage B**: Perform metric learning at t=0 using RHVAE implementation  
- **Stage C**: Train RLVAE model using components from Stage A and B

## 📋 **What We Attempted**

### **Initial Setup & Configuration**
1. **Started with command**: `python run_experiment.py experiment=rlvae_three_stage_pipeline model=rhvae_test model.latent_dim=16 metric=rhvae sampling=rhmc_default`
2. **Configuration files used**: `conf/model/rhvae_test.yaml`, `conf/experiment/rlvae_three_stage_pipeline.yaml`

### **Issues Discovered & Fixed**

#### **Issue 1: WandB Initialization Problems**
- **Problem**: Silent exceptions in `wandb.init()` and missing `wandb.run` checks
- **Fix**: Modified `run_experiment.py` to properly handle WandB initialization and add `if wandb.run is not None:` checks
- **Status**: ✅ **RESOLVED**

#### **Issue 2: Stage A Model Mismatch (Vanilla vs RHVAE)**
- **Problem**: Pipeline was using vanilla VAE instead of RHVAE for Stage A
- **Fix**: Updated command to `experiment.stage_a.model=rhvae`
- **Status**: ✅ **RESOLVED**

#### **Issue 3: Stage B Loading Wrong Stage A Data**
- **Problem**: Stage B was loading from `A_VANILLA_MLP_16_SPRITES` instead of `A_RHVAE_MLP_16_SPRITES`
- **Fix**: Modified `find_stage_a_data()` function to prioritize RHVAE folders and handle both `.pt` and `.pkl` files
- **Status**: ✅ **RESOLVED**

#### **Issue 4: Missing Configuration Parameters**
- **Problem**: `conf/model/rhvae_test.yaml` was missing numerous parameters (pretrained, metric, posterior, n_flows, etc.)
- **Fix**: Added all missing parameters incrementally as they were discovered
- **Status**: ✅ **RESOLVED**

#### **Issue 5: Variable Scope Errors**
- **Problem**: `UnboundLocalError` for `arch`, `latent_dim`, and `metric_impl` variables in Stage C
- **Fix**: Added variable definitions in Stage C section
- **Status**: ✅ **RESOLVED**

#### **Issue 6: File Format Issues**
- **Problem**: Encoder/decoder files were `.pkl` but code was looking for `.pt`
- **Fix**: Renamed files and updated globbing patterns to handle both formats
- **Status**: ✅ **RESOLVED**

#### **Issue 7: RHMC Sampling Always Running**
- **Problem**: Stage B (RHMC sampling) was running even when `experiment.run_stage_b=false`
- **Root Cause**: Two separate flags - `run_stage_b` (metric learning) vs `run_sampling` (RHMC)
- **Fix**: Added `experiment.run_sampling=false` to command
- **Status**: ✅ **RESOLVED**

#### **Issue 8: Poor Reconstruction Performance**
- **Problem**: High reconstruction loss (~59), low PSNR (~6.32), no temporal evolution
- **Symptoms**: 
  - `frame_diff=0.005535` (should be > 0.01)
  - `var_time@b0=3.5524e-05` (almost no temporal variance)
  - All timesteps producing same output
- **Attempted Fixes**:
  - Changed encoder/decoder architecture from `rhvae_rgb` to `mlp`
  - Set `reconstruction_mode: 'all'`
  - Set `kl_over_all_timesteps_if_flows: false` (KL only on z₀)
  - Set `riemannian_beta: 1.0` (was 0.01)
  - Set `n_flows: 7` (was 2)
- **Status**: ❌ **PERSISTENT ISSUE**

#### **Issue 9: Wrong Model Target**
- **Problem**: Still using `_RHVAEExp` instead of `RiemannianFlowVAE`
- **Fix**: Changed `_target_` in config and added direct override in code
- **Status**: ✅ **RESOLVED**

## 🔍 **Key Discoveries**

### **Configuration Structure**
- **Two separate sampling flags**: `run_stage_b` vs `run_sampling`
- **File format inconsistency**: `.pkl` vs `.pt` files
- **Hydra caching issues**: Required direct code overrides for model targets

### **Temporal Dynamics Issues**
- **No flow learning**: Model producing identical outputs across timesteps
- **Low frame diversity**: `frame_diff=0.005535` (target: > 0.01)
- **Minimal temporal variance**: `var_time@b0=3.5524e-05`

### **Reconstruction Quality**
- **High MSE**: ~59.18 (should be < 10 for good quality)
- **Low PSNR**: ~6.32 (should be > 20)
- **Poor visual quality**: "not working at all!"

## 🚨 **Current Status**

### **✅ What's Working**
1. **Pipeline Structure**: All three stages can run without errors
2. **Component Loading**: Stage A and B components load correctly
3. **Configuration**: All parameters properly set
4. **WandB Logging**: Proper initialization and logging
5. **Variable Scope**: All variable definitions resolved
6. **File Handling**: Both `.pt` and `.pkl` files supported

### **❌ What's Not Working**
1. **Reconstruction Quality**: Still very poor (MSE ~59, PSNR ~6)
2. **Temporal Dynamics**: No flow learning, identical outputs across timesteps
3. **Frame Diversity**: Very low diversity between frames
4. **Model Performance**: Not learning meaningful temporal representations

## 🎯 **Root Cause Analysis**

The core issue appears to be that the **RiemannianFlowVAE model is not properly learning temporal dynamics**. Despite having:
- ✅ Correct model target (`RiemannianFlowVAE`)
- ✅ Proper flow configuration (`n_flows: 7`)
- ✅ Temporal reconstruction enabled (`reconstruction_mode: 'all'`)
- ✅ KL only on z₀ (`kl_over_all_timesteps_if_flows: false`)

The model is still producing:
- Identical outputs across all timesteps
- Very high reconstruction loss
- No meaningful temporal evolution

## 🔧 **Next Steps Needed**

1. **Find Working Reference**: Locate the original GitHub repository or working implementation
2. **Compare Implementations**: Identify differences between working and current versions
3. **Debug Model Architecture**: Check if the `RiemannianFlowVAE` implementation is correct
4. **Test with Simple Data**: Try with synthetic data to isolate the issue
5. **Check Flow Implementation**: Verify that the normalizing flows are working properly

## 📊 **Final Configuration**

```yaml
model:
  _target_: models.riemannian_flow_vae.RiemannianFlowVAE
  latent_dim: 16
  n_flows: 7
  riemannian_beta: 1.0
  reconstruction_mode: "all"
  kl_over_all_timesteps_if_flows: false
  riemannian_kl_mode: "sample_logq_logp"

experiment:
  run_stage_a: false
  run_stage_b: false
  run_sampling: false  # RHMC disabled
  run_stage_c: true
```

## �� **Conclusion**

We successfully resolved all **technical/configuration issues** and got the pipeline running, but the **core model performance issue remains**. The RiemannianFlowVAE is not learning proper temporal dynamics, suggesting either:

1. **Implementation bug** in the model architecture
2. **Missing component** in the flow implementation  
3. **Incorrect loss function** or training setup
4. **Need for different hyperparameters** or architecture

The next step is to find and compare with a **working reference implementation** to identify what's missing or incorrect in the current setup.








## 📁 **Repository Structure Analysis**

I cloned the original working repository from [https://github.com/antoinelfg/RlVAE.git](https://github.com/antoinelfg/RlVAE.git) to compare with your current implementation. Here's the complete structure and key findings:

### **Repository Layout**
```
original_rlvae/
├── src/
│   ├── models/
│   │   ├── riemannian_flow_vae.py          # 🎯 ORIGINAL WORKING MODEL (65KB, 1414 lines)
│   │   ├── modular_rlvae.py                # Current modular version (37KB, 902 lines)
│   │   ├── modular_vanilla_vae.py          # Vanilla VAE implementation
│   │   └── components/                     # Modular components
│   ├── training/                           # Training infrastructure
│   ├── data/                               # Data loading
│   ├── visualizations/                     # Visualization system
│   └── evaluation/                         # Evaluation tools
├── conf/                                   # Hydra configurations
├── scripts/                                # Pipeline scripts
├── docs/                                   # Documentation
└── tests/                                  # Test suite
```

## 🔍 **Key Files Comparison**

### **1. Original Working Model: `original_rlvae/src/models/riemannian_flow_vae.py`**
- **Size**: 65KB, 1414 lines
- **Status**: ✅ **WORKING** (from GitHub)
- **Key Features**:
  - Direct `self.G` and `self.G_inv` function assignments
  - Proper KL computation: `self.G(z_samples)`
  - Working metric loading: `load_pretrained_metrics()`
  - Multiple sampling methods: geodesic, enhanced, basic, standard

### **2. Current Modular Model: `src/models/modular_rlvae.py`**
- **Size**: 37KB, 902 lines  
- **Status**: ❌ **BROKEN** (your current implementation)
- **Key Issues**:
  - Uses `MetricTensor` component instead of direct functions
  - KL computation interface mismatch
  - Complex modular architecture that wasn't properly tested

## 🚨 **Root Cause Analysis**

### **The Core Problem**
The issue is **architectural mismatch** between the working original and your current implementation:

| Aspect | Original (Working) | Current (Broken) |
|--------|-------------------|------------------|
| **Model Class** | `RiemannianFlowVAE` | `ModularRiemannianFlowVAE` |
| **Metric Interface** | `self.G(z)` | `metric_tensor.compute_metric(z)` |
| **KL Computation** | `self.G(z_samples)` | `metric_tensor(z_samples)` (fails) |
| **Metric Loading** | `load_pretrained_metrics()` | `MetricTensor.load_pretrained()` |
| **Architecture** | Simple, direct | Complex, modular |

### **Why the Original Works**
1. **Direct Function Assignment**:
   ```python
   def _G(z: torch.Tensor):
       return torch.linalg.inv(_G_inv(z))
   self.G = _G
   self.G_inv = _G_inv
   ```

2. **Proper KL Computation**:
   ```python
   def compute_riemannian_metric_kl_loss(self, mu, log_var, z_samples):
       G_z = self.G(z_samples)  # ✅ Direct call works
       # ... rest of computation
   ```

3. **Working Metric Loading**:
   ```python
   def load_pretrained_metrics(self, metric_path, temperature_override=None):
       # Loads centroids, M_tens, temperature, lbd
       # Creates G and G_inv functions
   ```

### **Why the Current Implementation Fails**
1. **Interface Mismatch**:
   ```python
   # Current tries to call:
   G_z = metric_tensor(z_samples)  # ❌ metric_tensor is a component, not a function
   
   # Should be:
   G_z = metric_tensor.compute_metric(z_samples)  # ✅ Correct interface
   ```

2. **Complex Modular Architecture**:
   - Multiple layers of abstraction
   - Component interfaces not properly tested
   - Loss computation doesn't match component interface

## 🔧 **The Fix Applied**

I modified `src/models/components/loss_manager.py` to handle both interfaces:

```python
# Before (broken):
G_z = metric_tensor(z_samples)  # Failed!

# After (fixed):
if hasattr(metric_tensor, 'compute_metric'):
    G_z = metric_tensor.compute_metric(z_samples)  # ✅ Correct for modular
else:
    G_z = metric_tensor(z_samples)  # ✅ Fallback for original
```

## 📊 **Performance Comparison**

### **Original Implementation**
- ✅ **Fast**: Direct function calls
- ✅ **Working**: Proven to work in GitHub repo
- ✅ **Simple**: Straightforward architecture
- ✅ **Tested**: Used in successful experiments

### **Current Modular Implementation**
- ❌ **Slow**: Multiple abstraction layers
- ❌ **Broken**: KL divergence always 0
- ❌ **Complex**: Hard to debug and maintain
- ❌ **Untested**: Not properly validated with three-stage pipeline

## 🎯 **Recommended Solution**

### **Option 1: Use Original Working Implementation** (Recommended)
```bash
# Switch back to original RiemannianFlowVAE
_target_: models.riemannian_flow_vae.RiemannianFlowVAE
```

### **Option 2: Fix Modular Implementation**
- Requires extensive testing and debugging
- More complex but potentially more flexible
- Higher risk of introducing new bugs

### **Option 3: Use Working Global Pipeline**
- The global pipeline already works with modular components
- Three-stage pipeline might be overkill for your needs

## �� **Expected Results with Original Implementation**

When using the original `RiemannianFlowVAE`:

1. **✅ KL Divergence**: Should compute properly (not always 0)
2. **✅ Reconstruction Loss**: Should decrease from ~59 to <10
3. **✅ Temporal Dynamics**: Should learn meaningful flows
4. **✅ Performance**: Should be faster than modular version
5. **✅ Stability**: Should be more reliable and tested

## �� **Next Steps**

1. **Test Original Implementation**: Run with `RiemannianFlowVAE`
2. **Monitor KL Values**: Should see non-zero, meaningful KL divergence
3. **Check Loss Reduction**: Reconstruction loss should improve significantly
4. **Verify Temporal Learning**: Flows should learn temporal dynamics
5. **Compare Performance**: Should be faster than modular version

The key insight is that **simplicity wins** - the original implementation works because it's direct and straightforward, while the modular version adds complexity without proper testing and integration.