# Data Directory

This directory contains the datasets and pretrained models for RlVAE training.

## 📁 Directory Structure

```
data/
├── raw/                     # Original datasets (not in git - see below)
│   ├── Sprites_train.pt     # 3.375 GB - Training sprites dataset
│   └── Sprites_test.pt      # 999 MB - Test sprites dataset
├── processed/               # Processed datasets (not in git - see below)
│   ├── Sprites_train_cyclic.pt          # 1.125 GB - Cyclic training sequences
│   ├── Sprites_test_cyclic.pt           # 333 MB - Cyclic test sequences
│   ├── Sprites_train_cyclic_metadata.pt # Metadata for training sequences
│   └── Sprites_test_cyclic_metadata.pt  # Metadata for test sequences
└── pretrained/              # Pretrained model components (included in git)
    ├── encoder.pt           # Pretrained VAE encoder
    ├── decoder.pt           # Pretrained VAE decoder
    ├── metric.pt            # Original metric tensor
    └── metric_T0.7_scaled.pt # Temperature-scaled metric
```

## 🚨 Large Files Not Included

The `raw/` and `processed/` directories contain large files (up to 3.375 GB) that exceed GitHub's size limits. These files are **not included** in this repository.

## 📥 How to Obtain the Data

### Option 1: Download from Source
*Please contact the repository author for access to the original datasets.*

### Option 2: Generate the Data
If you have the raw sprite datasets, you can generate the processed files:

```bash
# 1. Place your raw datasets in data/raw/
#    - Sprites_train.pt
#    - Sprites_test.pt

# 2. Generate processed cyclic sequences
python scripts/extract_cyclic_sequences.py

# 3. Train vanilla VAE and extract components
python scripts/train_and_extract_vanilla_vae.py

# 4. Create temperature-scaled metric
python scripts/create_identity_metric_temp_0_7.py
```

### Option 3: Use Alternative Datasets
The code can be adapted to work with other sequential image datasets. Modify the data loading in `config.py` to point to your datasets.

## ✅ Verification

After obtaining the data files, verify your setup:

```bash
# Check all files are present
python tests/test_setup.py

# Quick training test
python run_clean_training.py --loop_mode open --n_epochs 1 --n_train_samples 10
```

## 📊 Dataset Information

### Raw Sprites Dataset
- **Format**: PyTorch tensors (.pt files)
- **Content**: Sequential sprite animations
- **Usage**: Base datasets for cyclic sequence extraction

### Processed Cyclic Sequences
- **Format**: PyTorch tensors with temporal structure
- **Content**: Extracted cyclic sequences for training
- **Metadata**: Includes cycle information and transformations

### Pretrained Components
- **encoder.pt**: Vanilla VAE encoder (included in repository)
- **decoder.pt**: Vanilla VAE decoder (included in repository)  
- **metric.pt**: Riemannian metric tensor
- **metric_T0.7_scaled.pt**: Temperature-scaled metric (T=0.7)

## 🔧 Data Configuration

Data paths are configured in `config.py`. You can modify these paths if your data is stored elsewhere:

```python
# In config.py
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PRETRAINED_DIR = DATA_DIR / "pretrained"
```

## 📞 Support

If you need help obtaining or setting up the datasets, please:
1. Open an issue on GitHub
2. Contact the repository maintainer
3. Check the troubleshooting section in `docs/installation.md` 