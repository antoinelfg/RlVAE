# RlVAE Repository Organization Guide

This guide outlines the organization and structure of the RlVAE repository, including how to prepare it for GitHub deployment and manage large files effectively.

## 🏗️ Repository Structure Overview

### Current Organization
```
RlVAE/
├── 🧠 src/                                # Core implementation
│   ├── models/                            # Model architectures
│   ├── training/                          # Training infrastructure
│   ├── visualizations/                    # Visualization system
│   ├── data/                              # Data handling
│   └── utils/                             # Utility functions
├── ⚙️ conf/                               # Hydra configuration system
├── 🛠️ scripts/                           # Automation and utilities
├── 📚 docs/                               # Documentation
├── 🧪 tests/                              # Test suite
├── 📊 experiments/                        # Experiment outputs (LARGE)
├── 📈 outputs/                            # Training outputs (LARGE)
├── 🎯 wandb/                              # WandB logs (LARGE)
└── 📁 data/                               # Data files (LARGE)
```

### Target Organization (GitHub Ready)
```
RlVAE/
├── 🧠 src/                                # Core implementation
├── ⚙️ conf/                               # Configuration system
├── 🛠️ scripts/                           # Automation scripts
├── 📚 docs/                               # Documentation
├── 🧪 tests/                              # Test suite
├── 📁 data/                               # Data structure (no large files)
├── 📋 examples/                           # Example configurations
├── 📖 tutorials/                          # Tutorial notebooks
└── 📄 Configuration files
```

## 🚫 Large File Management Strategy

### Files to Exclude from Git
**Size Threshold**: >10MB files should not be committed to Git

#### 1. Model Checkpoints and Artifacts
```
# Excluded (LARGE)
*.ckpt                    # PyTorch Lightning checkpoints
*.pth                     # PyTorch model files
*.pt                      # PyTorch tensors
final_model.pt            # Final trained models
encoder.pt                # Encoder weights
decoder.pt                # Decoder weights
```

#### 2. Training Outputs
```
# Excluded (LARGE)
outputs/                  # Training outputs directory
experiments/              # Experiment results
lightning_logs/           # Lightning training logs
wandb/                    # WandB experiment logs
backups/                  # Backup model files
metric_snapshots/         # Metric tensor snapshots
```

#### 3. Data Files
```
# Excluded (LARGE)
data/raw/*.pt             # Raw data tensors
data/processed/*.pt       # Processed data tensors
data/*.tar.gz             # Compressed data
data/*.zip                # Compressed data
data/MNIST/               # MNIST dataset
data/fid_cache/           # FID cache files
```

#### 4. Generated Visualizations
```
# Excluded (LARGE)
*.png                     # Generated plots
*.jpg                     # Generated images
*.gif                     # Generated animations
*.svg                     # Generated vector graphics
*.html                    # Generated HTML reports
```

### Files to Include in Git
#### 1. Essential Code and Configuration
```
# Always include
src/                      # Source code
conf/                     # Configuration files
scripts/                  # Scripts and utilities
tests/                    # Test suite
docs/                     # Documentation
requirements.txt           # Dependencies
environment.yml            # Conda environment
```

#### 2. Data Structure (No Large Files)
```
# Include structure, not content
data/
├── raw/
│   ├── .gitkeep          # Keep directory structure
│   └── README.md         # Data description
├── processed/
│   ├── .gitkeep          # Keep directory structure
│   └── README.md         # Processing description
└── README.md             # Overall data guide
```

#### 3. Configuration Templates
```
# Include configuration examples
conf/
├── experiment/            # Experiment configurations
├── model/                # Model configurations
├── training/             # Training configurations
├── data/                 # Data configurations
└── sweep/                # Hyperparameter sweep configs
```

## 🔧 Git Configuration and Setup

### 1. .gitignore Configuration
The `.gitignore` file has been updated to handle large files properly:

```gitignore
# ===== RlVAE SPECIFIC IGNORES =====

# Training outputs and experiments (LARGE FILES)
outputs/
wandb/
lightning_logs/
experiments/
backups/
metric_snapshots/

# Model checkpoints and training artifacts
*.ckpt
*.pth
*.pt
*.pkl
final_model.pt
encoder.pt
decoder.pt

# Data directories (keep structure but ignore content)
data/raw/*
data/processed/*
data/*.tar.gz
data/*.zip
data/*.pt
data/*.pkl
data/MNIST/
data/fid_cache/

# Generated images and visualizations (LARGE FILES)
*.png
*.jpg
*.jpeg
*.gif
*.svg
*.html
```

### 2. Git LFS (Large File Storage) - Optional
For files that must be tracked but are large:

```bash
# Install Git LFS
git lfs install

# Track large file types
git lfs track "*.pt"
git lfs track "*.ckpt"
git lfs track "*.pth"

# Add .gitattributes
git add .gitattributes
```

### 3. Repository Size Management
```bash
# Check repository size
git count-objects -vH

# Clean up large files from history (if needed)
git filter-branch --tree-filter 'rm -rf outputs/ experiments/ wandb/' HEAD

# Force push after cleanup
git push origin --force
```

## 📁 Directory Organization Strategy

### 1. Core Source Code (`src/`)
```
src/
├── models/                # Model architectures
│   ├── __init__.py
│   ├── modular_rlvae.py  # Main RlVAE implementation
│   ├── modular_vanilla_vae.py
│   ├── hybrid_rlvae.py
│   ├── riemannian_flow_vae.py
│   └── components/        # Modular components
├── training/              # Training infrastructure
│   ├── __init__.py
│   ├── lightning_trainer.py
│   └── plugins/           # Training plugins
├── visualizations/        # Visualization system
│   ├── __init__.py
│   ├── manager.py
│   ├── basic.py
│   └── manifold.py
├── data/                  # Data handling
│   ├── __init__.py
│   └── datasets.py
└── utils/                 # Utility functions
    ├── __init__.py
    └── helpers.py
```

### 2. Configuration System (`conf/`)
```
conf/
├── config.yaml            # Main configuration
├── experiment/            # Experiment types
│   ├── rlvae_three_stage_pipeline.yaml
│   ├── global_vanilla_rlvae_pipeline.yaml
│   ├── single_run.yaml
│   └── comparison_study.yaml
├── model/                 # Model configurations
│   ├── mlp_rlvae.yaml
│   ├── cnn_rlvae.yaml
│   ├── resnet_rlvae.yaml
│   └── vanilla_vae.yaml
├── training/              # Training configurations
│   ├── quick.yaml
│   ├── default.yaml
│   └── full_data.yaml
├── data/                  # Data configurations
│   ├── cyclic_sprites.yaml
│   └── mnist.yaml
├── sweep/                 # Hyperparameter sweeps
│   ├── learning_rate_optimization.yaml
│   ├── architecture_optimization.yaml
│   └── comprehensive_hyperparameter_sweep.yaml
└── visualization/         # Visualization levels
    ├── minimal.yaml
    ├── standard.yaml
    └── full.yaml
```

### 3. Scripts and Automation (`scripts/`)
```
scripts/
├── slurm/                 # SLURM cluster scripts
│   ├── run_hyperparameter_sweep.sbatch
│   ├── run_hyperparameter_sweep_short.sbatch
│   └── run_experiment_rlvae.sbatch
├── orchestrate_three_stage.py
├── run_sweep.py
└── train_diverse_metric_vae.py
```

### 4. Documentation (`docs/`)
```
docs/
├── README.md              # Main documentation
├── installation.md        # Installation guide
├── TRAINING_GUIDE.md     # Training workflows
├── RLVAE_THREE_STAGE_PIPELINE.md
├── MODULAR_VISUALIZATION_GUIDE.md
├── HYPERPARAMETER_OPTIMIZATION_GUIDE.md
├── SWEEP_README.md       # Hyperparameter sweeps
├── CONTRIBUTING.md       # Contributing guidelines
└── archive/              # Archived documentation
```

### 5. Test Suite (`tests/`)
```
tests/
├── __init__.py
├── unit/                  # Unit tests
│   ├── test_models.py
│   ├── test_training.py
│   └── test_visualizations.py
├── integration/           # Integration tests
│   ├── test_pipeline.py
│   └── test_end_to_end.py
├── stagec/                # Stage C specific tests
│   └── test_stagec.py
└── conftest.py            # Test configuration
```

## 🚀 GitHub Deployment Strategy

### 1. Repository Preparation
```bash
# Clean up large files
rm -rf outputs/ experiments/ wandb/ lightning_logs/

# Remove large data files (keep structure)
find data/ -name "*.pt" -delete
find data/ -name "*.ckpt" -delete
find data/ -name "*.pth" -delete

# Remove generated visualizations
find . -name "*.png" -delete
find . -name "*.jpg" -delete
find . -name "*.gif" -delete

# Check repository size
du -sh .git/
```

### 2. Initial Commit Strategy
```bash
# Stage essential files
git add src/ conf/ scripts/ docs/ tests/
git add requirements.txt environment.yml
git add .gitignore README.md
git add data/*/README.md data/*/.gitkeep

# Commit core repository
git commit -m "Initial commit: Core RlVAE framework"

# Push to GitHub
git push origin main
```

### 3. Large File Handling
```bash
# Option 1: Git LFS for essential large files
git lfs track "*.pt"
git lfs track "*.ckpt"
git add .gitattributes
git commit -m "Add Git LFS tracking for large files"

# Option 2: External storage with download scripts
# Create download_models.sh script for pretrained models
```

## 📋 Repository Maintenance

### 1. Regular Cleanup
```bash
# Monthly cleanup script
#!/bin/bash
# cleanup_repository.sh

echo "Cleaning up large files..."
rm -rf outputs/* experiments/* wandb/* lightning_logs/*
find . -name "*.png" -delete
find . -name "*.ckpt" -delete

echo "Repository cleaned up!"
```

### 2. Size Monitoring
```bash
# Check repository size
git count-objects -vH

# Check for large files
find . -size +10M -not -path "./.git/*" | head -20

# Monitor directory sizes
du -sh ./* | sort -hr
```

### 3. Documentation Updates
```bash
# Update changelog
echo "$(date): Updated documentation" >> NEW_DEVELOPMENTS_CHANGELOG.md

# Update README if needed
# Update .cursor_context.md if needed
```

## 🔍 Quality Assurance

### 1. Pre-commit Checks
```bash
# Run tests
python -m pytest tests/ -v

# Check configuration
python run_experiment.py --config-name=rlvae_three_stage_pipeline --dry-run

# Validate imports
python -c "import src.models.modular_rlvae; print('Import successful')"
```

### 2. Repository Health
```bash
# Check for broken links
find . -name "*.md" -exec grep -l "\[.*\](" {} \; | xargs -I {} grep -o "\[.*\]([^)]*)" {} | grep -v "http" | grep -v "#"

# Check for missing files
find . -name "*.md" -exec grep -l "\[.*\](" {} \; | xargs -I {} grep -o "\[.*\]([^)]*)" {} | grep -v "http" | grep -v "#" | sed 's/.*(\([^)]*\)).*/\1/' | xargs -I {} test -f {} || echo "Missing: {}"
```

## 📊 Repository Metrics

### Target Metrics
- **Repository Size**: <100MB (excluding Git LFS)
- **Code Coverage**: >80%
- **Documentation Coverage**: 100% of public APIs
- **Test Coverage**: >90% of core functionality

### Current Status
- **Repository Size**: ~200GB (before cleanup)
- **Code Coverage**: ~70%
- **Documentation Coverage**: ~85%
- **Test Coverage**: ~75%

## 🎯 Next Steps

### Immediate Actions
1. **Clean up large files** from repository
2. **Update .gitignore** for proper file handling
3. **Organize documentation** structure
4. **Prepare for GitHub deployment**

### Short-term Goals
1. **Deploy to GitHub** with clean structure
2. **Set up CI/CD** for automated testing
3. **Create release tags** for major versions
4. **Establish contribution guidelines**

### Long-term Vision
1. **Maintain clean repository** structure
2. **Automate large file management**
3. **Create comprehensive tutorials**
4. **Establish community guidelines**

---

**Note**: This organization guide ensures that the RlVAE repository is GitHub-ready while maintaining all essential functionality. The strategy balances code accessibility with repository size management.
