# RlVAE Repository Organization Guide

## 🚫 Large File Management

### Files to Exclude (>10MB)
- `outputs/` - Training outputs
- `experiments/` - Experiment results  
- `wandb/` - WandB logs
- `lightning_logs/` - Training logs
- `*.ckpt`, `*.pth`, `*.pt` - Model files
- `*.png`, `*.jpg`, `*.gif` - Generated images
- `data/MNIST/`, `data/fid_cache/` - Large datasets

### Files to Include
- `src/` - Source code
- `conf/` - Configuration files
- `scripts/` - Automation scripts
- `docs/` - Documentation
- `tests/` - Test suite
- `requirements.txt` - Dependencies

## 🔧 Git Setup

### 1. Clean Repository
```bash
# Remove large files
rm -rf outputs/ experiments/ wandb/ lightning_logs/
find . -name "*.pt" -delete
find . -name "*.png" -delete

# Keep data structure
mkdir -p data/raw data/processed
touch data/raw/.gitkeep data/processed/.gitkeep
```

### 2. Update .gitignore
```gitignore
# Large files
outputs/ wandb/ experiments/ lightning_logs/
*.ckpt *.pth *.pt *.png *.jpg *.gif
data/MNIST/ data/fid_cache/
```

### 3. Initial Commit
```bash
git add src/ conf/ scripts/ docs/ tests/
git add requirements.txt README.md .gitignore
git add data/*/.gitkeep
git commit -m "Initial commit: Core RlVAE framework"
```

## 📁 Target Structure
```
RlVAE/
├── src/                   # Core implementation
├── conf/                  # Configuration system
├── scripts/               # Automation scripts
├── docs/                  # Documentation
├── tests/                 # Test suite
├── data/                  # Data structure (no large files)
└── examples/              # Example configs
```

## 🚀 GitHub Deployment
1. Clean repository of large files
2. Update .gitignore
3. Commit core files
4. Push to GitHub
5. Set up CI/CD for testing
