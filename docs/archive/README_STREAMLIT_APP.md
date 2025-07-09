# VAE Research Platform - Streamlit Application

A comprehensive, research-grade Streamlit application for managing, running, and analyzing Variational Autoencoder experiments with interactive latent space exploration.

## 🚀 Features

### 🧪 Experiment Management
- **Interactive Configuration**: Intuitive UI for setting up VAE experiments
- **Real-time Monitoring**: Live training metrics and progress tracking
- **Multiple Model Support**: Modular RlVAE, Hybrid RlVAE, Standard VAE
- **Experiment History**: Track and manage multiple experiment runs

### 🔮 Model Inference
- **Model Loading**: Support for PyTorch checkpoints and trained models
- **Interactive Encoding**: Upload images and encode to latent space
- **Flexible Decoding**: Manual latent vector input and random sampling
- **Real-time Visualization**: Immediate feedback for all operations

### 🌌 Latent Space Exploration
- **2D/ND Grid Visualization**: Interactive latent space grids
- **Latent Interpolation**: Linear, spherical, and geodesic interpolation
- **Dimensionality Reduction**: PCA, UMAP, t-SNE embeddings
- **Manual Control**: Real-time latent dimension manipulation

### 📊 Model Comparison
- **Multi-model Analysis**: Side-by-side model comparisons
- **Performance Metrics**: Comprehensive metric evaluation
- **Latent Space Analysis**: Distribution and quality comparisons
- **Automated Reports**: Generate publication-ready comparison reports

### 🎨 Visualization Gallery
- **Loss Analysis**: ELBO decomposition and convergence analysis
- **Distribution Plots**: Latent space distribution visualization
- **Model Diagnostics**: Architecture and parameter analysis
- **Training Curves**: Interactive training progress visualization

## 📁 Project Structure

```
VAE-Research-Platform/
├── app.py                          # Main Streamlit application
├── app/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   └── sidebar.py              # Sidebar navigation and controls
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── overview.py             # Platform dashboard
│   │   ├── experiment_manager.py   # Experiment configuration and execution
│   │   ├── model_inference.py      # Model loading and inference
│   │   ├── latent_exploration.py   # Interactive latent space exploration
│   │   ├── model_comparison.py     # Multi-model analysis
│   │   └── visualization_gallery.py # Advanced visualization tools
│   └── utils/
│       ├── __init__.py
│       └── session_state.py        # Session state management
├── sample_configs/
│   └── modular_rlvae_example.yaml  # Example configuration
├── requirements.txt                # Dependencies
└── README_STREAMLIT_APP.md         # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/antoinelfg/RlVAE.git
   cd RlVAE
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Additional packages for full functionality**:
   ```bash
   pip install umap-learn scikit-learn psutil
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**:
   The app will automatically open at `http://localhost:8501`

## 🚀 Quick Start Guide

### 1. First Time Setup
- Launch the app with `streamlit run app.py`
- The **Overview** page provides platform introduction and quick stats
- Check the **System Information** in the sidebar to verify GPU availability

### 2. Running Your First Experiment
1. Navigate to **🧪 Experiment Manager**
2. In the **Configure** tab:
   - Set experiment name and description
   - Choose model type (recommend starting with `modular_rlvae`)
   - Configure training parameters (start with defaults)
   - Set data parameters (default `cyclic_sprites` dataset)
3. Switch to **Run & Monitor** tab
4. Click **🚀 Start Experiment**
5. Monitor real-time training progress

### 3. Loading and Exploring Models
1. Go to **🔮 Model Inference**
2. Upload a trained model checkpoint or use experiment results
3. Test encoding with uploaded images or random inputs
4. Experiment with manual latent vector decoding

### 4. Interactive Latent Space Exploration
1. Visit **🌌 Latent Exploration** (requires loaded model)
2. Generate latent grids for spatial visualization
3. Create interpolations between latent points
4. Use manual controls for real-time latent manipulation

### 5. Comparing Multiple Models
1. In **📊 Model Comparison**, load multiple model checkpoints
2. Run performance comparisons with automated metrics
3. Analyze latent space differences
4. Generate comprehensive comparison reports

## ⚙️ Configuration

### Model Configuration
The app supports multiple VAE architectures:

- **Modular RlVAE**: Fully configurable Riemannian VAE
- **Hybrid RlVAE**: Performance-optimized variant
- **Standard RlVAE**: Original implementation
- **Vanilla VAE**: Baseline comparison model

### Key Parameters
- **Latent Dimension**: Size of latent representation (2-128)
- **Number of Flows**: Temporal flow layers (0-10)
- **Beta Parameters**: Regularization weights
- **Architecture**: Encoder/decoder networks (MLP, CNN, ResNet)
- **Sampling Method**: Latent space sampling strategy

### Example Configuration
See `sample_configs/modular_rlvae_example.yaml` for a complete example.

## 🔧 Advanced Features

### Real-time Monitoring
- Live training metrics with interactive plots
- GPU memory usage tracking
- Automatic checkpoint saving
- Training progress notifications

### Interactive Visualizations
- Plotly-based interactive charts
- Hoverable data points with detailed information
- Zoomable and pannable visualizations
- Export capabilities for publication

### Latent Space Analysis
- **2D Grids**: Direct visualization for 2D latent spaces
- **ND Slicing**: 2D slices of higher-dimensional spaces
- **Interpolation**: Multiple interpolation methods
- **Embeddings**: PCA, UMAP, t-SNE dimensionality reduction

### Model Diagnostics
- Parameter distribution analysis
- Gradient flow visualization
- Architecture diagrams
- Convergence analysis

## 📊 Visualization Capabilities

### Loss Analysis
- ELBO decomposition over training
- Per-dimension KL divergence
- Loss landscape visualization
- Convergence metrics

### Distribution Analysis
- Latent space histograms
- Correlation matrices
- Principal component analysis
- Statistical summaries

### Comparison Tools
- Side-by-side model metrics
- Radar charts for multi-metric comparison
- Automated statistical analysis
- Performance ranking

## 🔬 Research Applications

### Longitudinal Data Modeling
- Medical time series analysis
- Financial market dynamics
- Scientific experiment tracking

### Riemannian Geometry Research
- Custom metric tensor learning
- Geodesic path analysis
- Manifold structure exploration

### Model Development
- Architecture comparison
- Hyperparameter optimization
- Ablation studies
- Performance benchmarking

## 🤝 Contributing

### Adding New Models
1. Implement model in `src/models/`
2. Add configuration in `conf/model/`
3. Update model factory in `src/models/modular_rlvae.py`
4. Test with the Streamlit interface

### Adding New Visualizations
1. Create visualization function in appropriate page module
2. Add UI controls for parameters
3. Integrate with session state management
4. Test with different model types

### Extending Analysis Tools
1. Add analysis functions to `app/pages/`
2. Create interactive controls
3. Integrate with caching system
4. Document usage and parameters

## 🐛 Troubleshooting

### Common Issues

**App won't start:**
- Check Python version (3.8+ required)
- Verify all dependencies installed
- Try `pip install --upgrade streamlit`

**GPU not detected:**
- Install CUDA-compatible PyTorch
- Check NVIDIA drivers
- Verify CUDA installation

**Model loading fails:**
- Ensure checkpoint contains full model (not just state_dict)
- Check model architecture compatibility
- Verify file format (.pth, .pt, .ckpt)

**Out of memory errors:**
- Reduce batch size in training config
- Lower image resolution
- Use CPU instead of GPU for large models

### Performance Optimization

**Slow visualizations:**
- Reduce number of samples in analysis
- Lower plot resolution
- Enable caching for repeated computations
- Use smaller latent grids

**Training monitoring:**
- Disable real-time updates for faster training
- Reduce logging frequency
- Use tensorboard integration for detailed logs

## 📚 Documentation

### API Reference
- **Session State**: `app/utils/session_state.py`
- **Model Interface**: See individual model files in `src/models/`
- **Visualization Utils**: Check page modules in `app/pages/`

### Configuration Reference
- Model configs: `conf/model/*.yaml`
- Training configs: `conf/training/*.yaml`
- Data configs: `conf/data/*.yaml`

### Examples
- Sample configurations: `sample_configs/`
- Tutorial notebooks: `docs/tutorials/`
- Research examples: `docs/examples/`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original RlVAE research and implementation
- Streamlit team for the amazing framework
- PyTorch Lightning for training infrastructure
- Plotly for interactive visualizations
- The open-source research community

## 📞 Support

For questions, issues, or feature requests:
- 🐛 [Report Issues](https://github.com/antoinelfg/RlVAE/issues)
- 💡 [Feature Requests](https://github.com/antoinelfg/RlVAE/discussions)
- 📧 Contact maintainers through GitHub

---

**Ready to explore VAE research with an intuitive interface?** 🚀

```bash
streamlit run app.py
```

Start your VAE research journey today!