# Examples Directory

This directory contains various examples, tests, and utilities for the RlVAE framework.

## 📁 Directory Structure

### `visualizations/`
Contains visualization and analysis scripts for:
- Enhanced KL divergence analysis
- Posterior adaptation visualization
- Sampling methods comparison
- RHMC training visualization

### `tests/`
Contains test scripts for:
- Enhanced KL divergence testing
- RHMC training validation
- Model verification
- Identity metric testing

### `debug/`
Contains debugging and monitoring scripts for:
- Gradient warning debugging
- Model forward pass debugging
- Enhanced KL monitoring
- Training log analysis

## 🚀 Quick Start

### Running Visualizations
```bash
# Enhanced KL analysis
cd examples/visualizations
python visualize_enhanced_kl_combined.py

# Posterior adaptation
python visualize_posterior_adaptation.py

# Sampling comparison
python compare_sampling_methods.py
```

### Running Tests
```bash
# Enhanced KL tests
cd examples/tests
python test_enhanced_kl_comprehensive.py

# RHMC training tests
python test_full_rhmc_training.py

# Model verification
python test_final_verification.py
```

### Debugging
```bash
# Debug gradient warnings
cd examples/debug
python debug_gradient_warnings.py

# Debug model forward pass
python debug_model_forward.py

# Monitor enhanced KL
python monitor_enhanced_kl.py
```

## 📝 Notes

- These examples are primarily for development and testing
- For production use, use the main `run_experiment.py` script
- Some scripts may require specific data or model checkpoints
- Check individual script documentation for requirements
