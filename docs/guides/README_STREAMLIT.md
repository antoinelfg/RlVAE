# 🧠 Enhanced RlVAE Streamlit App

A comprehensive, production-ready Streamlit application for running, monitoring, and analyzing Riemannian Flow VAE experiments with real GPU/CPU integration, advanced visualizations, and complete experiment management.

## 🚀 Features

### 🎯 **Complete Experiment Pipeline**
- **Real GPU/CPU Integration**: Full backend integration with PyTorch Lightning
- **Two-Stage Training**: Support for vanilla VAE + RlVAE pipeline
- **WandB Integration**: Automatic experiment tracking and logging
- **Real-time Monitoring**: Live training progress with system metrics

### 📊 **Advanced Visualizations**
- **Training Progress**: Real-time loss curves, metrics breakdown
- **Latent Space Analysis**: PCA, t-SNE, UMAP visualizations
- **Model Comparisons**: Performance heatmaps, architecture analysis
- **System Monitoring**: GPU utilization, memory usage, device info

### 🔧 **Experiment Management**
- **Configuration System**: Hydra-based configuration management
- **Model Registry**: Save, load, and compare trained models
- **Experiment History**: Timeline view, statistics, export capabilities
- **Hyperparameter Optimization**: Integration with WandB sweeps

### 🌌 **Interactive Exploration**
- **Latent Space Navigation**: Interactive 2D/3D latent space exploration
- **Model Inference**: Real-time encoding/decoding with visual feedback
- **Interpolation Tools**: Smooth transitions between latent points
- **Sample Generation**: Random and controlled sample generation

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for GPU acceleration)
- Git

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/antoinelfg/RlVAE.git
cd RlVAE
```

2. **Install dependencies**
```bash
# Install core requirements
pip install -r requirements.txt

# Install Streamlit-specific requirements
pip install -r requirements_streamlit.txt
```

3. **Run the Streamlit app**
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 📖 Usage Guide

### 🏠 **Overview Page**
- **System Status**: Device information, GPU/CPU utilization
- **Quick Start**: Pre-configured experiment templates
- **Recent Experiments**: Timeline of recent runs
- **Statistics**: Model performance overview

### 🧪 **Experiment Manager**

#### Configuration Tab
1. **Experiment Details**: Name, description, project settings
2. **Model Configuration**: 
   - Model type (Modular RlVAE, Hybrid RlVAE, Vanilla VAE)
   - Architecture (MLP, CNN, ResNet)
   - Latent dimensions, flows, regularization parameters
3. **Training Configuration**:
   - Epochs, batch size, learning rate
   - Optimizer, scheduler, early stopping
   - Precision, gradient clipping
4. **Data Configuration**:
   - Dataset selection, data splits
   - Augmentation, preprocessing
5. **Visualization Configuration**:
   - Plot frequency, interactive elements
   - WandB logging settings

#### Run & Monitor Tab
- **Real-time Progress**: Live training curves, loss breakdown
- **System Metrics**: GPU utilization, memory usage
- **Control Panel**: Start, pause, stop experiments
- **Status Updates**: Current epoch, loss values, learning rate

#### History Tab
- **Experiment Timeline**: Visual timeline of all experiments
- **Performance Statistics**: Loss distributions, model comparisons
- **Export Options**: Save results, configurations, models

### 🔮 **Model Inference**
- **Load Models**: Select from saved model registry
- **Encode Data**: Upload images or use sample data
- **Decode Latents**: Generate reconstructions
- **Batch Processing**: Process multiple samples

### 🌌 **Latent Exploration**
- **2D Visualization**: PCA, t-SNE, UMAP projections
- **Interactive Navigation**: Click to explore latent space
- **Interpolation**: Smooth transitions between points
- **Clustering**: Automatic cluster detection and coloring

### 📊 **Model Comparison**
- **Performance Analysis**: Loss comparisons, metrics heatmaps
- **Architecture Comparison**: Parameter analysis, efficiency metrics
- **Statistical Analysis**: Confidence intervals, significance tests
- **Export Reports**: Generate comparison reports

### 🎨 **Visualization Gallery**
- **Training Curves**: Comprehensive loss analysis
- **Latent Space**: Multi-dimensional visualizations
- **Reconstructions**: Before/after comparisons
- **Flow Analysis**: Normalizing flow diagnostics

## 🔧 Advanced Configuration

### Environment Variables
```bash
# WandB Configuration
export WANDB_PROJECT="rlvae-streamlit"
export WANDB_ENTITY="your-username"

# GPU Configuration
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Streamlit Configuration
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Configuration Files
The app uses Hydra for configuration management. Key configuration files:

- `conf/config.yaml`: Main configuration
- `conf/model/`: Model-specific configurations
- `conf/training/`: Training configurations
- `conf/experiment/`: Experiment types

### Custom Model Integration
To add a new model type:

1. **Create model class** in `src/models/`
2. **Add configuration** in `conf/model/`
3. **Update factory** in `src/models/modular_rlvae.py`
4. **Add tests** in `tests/`

## 🚀 GPU Acceleration

### CUDA Setup
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU usage
nvidia-smi

# Set memory fraction (optional)
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
```

### Performance Optimization
- **Mixed Precision**: Automatic 16-bit training
- **Gradient Accumulation**: For large batch sizes
- **Memory Management**: Automatic cleanup
- **Multi-GPU**: Ready for distributed training

## 📊 Monitoring and Logging

### WandB Integration
- **Automatic Logging**: All metrics, plots, configurations
- **Experiment Tracking**: Run history, hyperparameter sweeps
- **Collaboration**: Share experiments with team
- **Artifact Management**: Model versioning

### System Monitoring
- **GPU Metrics**: Utilization, memory, temperature
- **CPU Metrics**: Usage, memory, processes
- **Training Metrics**: Loss, gradients, learning rate
- **Resource Alerts**: Memory warnings, GPU errors

## 🔍 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Install missing dependencies
pip install -r requirements_streamlit.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

#### GPU Issues
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.version.cuda)"

# Reset GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

#### Memory Issues
- Reduce batch size
- Enable gradient accumulation
- Use mixed precision training
- Monitor memory usage

#### WandB Issues
```bash
# Check WandB login
wandb login

# Test connection
python -c "import wandb; wandb.init(mode='disabled')"
```

### Debug Mode
```bash
# Run with debug logging
streamlit run app.py --logger.level=debug

# Enable development mode
export STREAMLIT_DEVELOPMENT_MODE=true
```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_models.py
pytest tests/test_training.py
pytest tests/test_visualizations.py
```

### Integration Tests
```bash
# Test full pipeline
python -m pytest tests/test_integration.py

# Test Streamlit components
python -m pytest tests/test_streamlit.py
```

## 📈 Performance Benchmarks

### Training Speed
- **GPU (RTX 3080)**: ~2-5 min per epoch (batch size 32)
- **CPU (8 cores)**: ~15-30 min per epoch (batch size 8)
- **Memory Usage**: 4-8 GB GPU memory (depending on model size)

### Model Sizes
- **Modular RlVAE**: ~50-200 MB (depending on architecture)
- **Vanilla VAE**: ~30-150 MB
- **Hybrid RlVAE**: ~40-180 MB

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone https://github.com/antoinelfg/RlVAE.git
cd RlVAE

# Install development dependencies
pip install -r requirements_streamlit.txt
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install
```

### Code Style
```bash
# Format code
black app/ src/ tests/

# Lint code
flake8 app/ src/ tests/

# Type checking
mypy app/ src/
```

### Testing
```bash
# Run tests with coverage
pytest --cov=app --cov=src tests/

# Generate coverage report
coverage html
```

## 📚 Documentation

### API Reference
- **Backend Components**: `app/backend/`
- **Visualization Components**: `app/components/`
- **Page Components**: `app/pages/`

### Examples
- **Basic Usage**: See `examples/basic_usage.py`
- **Advanced Configuration**: See `examples/advanced_config.py`
- **Custom Models**: See `examples/custom_model.py`

## 🆘 Support

### Getting Help
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: Project Wiki
- **Email**: Contact maintainers

### Community
- **Slack**: Join our Slack workspace
- **Discord**: Join our Discord server
- **Twitter**: Follow for updates

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Lightning**: Training framework
- **Streamlit**: Web application framework
- **WandB**: Experiment tracking
- **Plotly**: Interactive visualizations
- **Hydra**: Configuration management

---

**Made with ❤️ by the RlVAE Research Team**