# Installation Guide

## 🚀 Quick Installation

### Option 1: Install from PyPI (Recommended)
```bash
pip install rlvae
```

### Option 2: Install from Source
```bash
# Clone the repository
git clone https://github.com/antoinelfg/RlVAE.git
cd RlVAE

# Install in development mode
pip install -e .

# Verify installation
python test_setup.py
```

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space for data and models
- **OS**: Linux, macOS, or Windows

### Recommended Requirements
- **GPU**: CUDA-compatible GPU with 6GB+ VRAM
- **CUDA**: Version 11.8 or higher
- **Python**: 3.10 for optimal performance

## 🐍 Python Environment Setup

### Using conda (Recommended)
```bash
# Create a new environment
conda create -n rlvae python=3.10
conda activate rlvae

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install RlVAE
pip install rlvae
```

### Using virtualenv
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install RlVAE
pip install rlvae
```

## 🔧 Development Installation

For contributing to the project:

```bash
# Clone and enter directory
git clone https://github.com/antoinelfg/RlVAE.git
cd RlVAE

# Install in development mode with all dependencies
pip install -e ".[dev,docs]"

# Install pre-commit hooks
pre-commit install

# Verify everything works
python tests/test_setup.py
```

## 📊 Data Setup

### Required Data Files
The following data files are needed for training:

```bash
data/
├── raw/
│   ├── Sprites_train.pt
│   └── Sprites_test.pt
├── processed/
│   ├── Sprites_train_cyclic.pt
│   ├── Sprites_test_cyclic.pt
│   ├── Sprites_train_cyclic_metadata.pt
│   └── Sprites_test_cyclic_metadata.pt
└── pretrained/
    ├── encoder.pt
    ├── decoder.pt
    ├── metric.pt
    └── metric_T0.7_scaled.pt
```

### Data Preparation
If you don't have the processed data files:

```bash
# Extract cyclic sequences from raw data
python scripts/extract_cyclic_sequences.py

# Train vanilla VAE and extract components
python scripts/train_and_extract_vanilla_vae.py

# Create temperature-scaled metric
python scripts/create_identity_metric_temp_0_7.py
```

## 🧪 Verification

### Quick Test
```bash
# Verify installation
python tests/test_setup.py

# Quick training test (1 epoch)
python run_clean_training.py --loop_mode open --n_epochs 1 --n_train_samples 10
```

### Expected Output
```
🧪 Running Clean Riemannian Flow VAE Setup Tests
==================================================
🧪 Testing imports...
✅ PyTorch: 2.0.1
✅ Lightning: 2.0.9
✅ RHVAE components available
✅ RiemannianFlowVAE import successful

📊 Testing data files...
✅ All required files found

🧠 Testing model creation...
✅ Model created successfully
✅ Forward pass successful, output shape: torch.Size([4, 6, 3, 64, 64])

🔧 Testing pretrained component loading...
✅ Pretrained components loaded successfully

==================================================
🏁 Test Results:
   1. test_imports: ✅ PASS
   2. test_data_files: ✅ PASS
   3. test_model_creation: ✅ PASS
   4. test_pretrained_loading: ✅ PASS

📊 Overall: 4/4 tests passed
🎉 All tests passed! Setup is ready.
```

## 🚨 Troubleshooting

### Common Issues

#### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CPU-only version if needed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### Missing Data Files
```bash
# Run data preparation scripts
python scripts/extract_cyclic_sequences.py
python scripts/train_and_extract_vanilla_vae.py
```

#### Import Errors
```bash
# Reinstall in development mode
pip uninstall rlvae
pip install -e .
```

#### Permission Issues (Linux/macOS)
```bash
# Fix permissions
chmod +x scripts/*.py
chmod +x run_clean_training.py
```

## 🌐 Alternative Installation Methods

### Docker (Coming Soon)
```bash
# Pull Docker image
docker pull rlvae/rlvae:latest

# Run container
docker run -it --gpus all rlvae/rlvae:latest
```

### Singularity (HPC Clusters)
```bash
# Build Singularity image
singularity build rlvae.sif docker://rlvae/rlvae:latest

# Run with GPU support
singularity run --nv rlvae.sif
```

## 📞 Getting Help

If you encounter issues during installation:

1. **Check the troubleshooting section** above
2. **Search existing issues** on GitHub
3. **Open a new issue** with your error details and system information
4. **Join our discussions** for community support 