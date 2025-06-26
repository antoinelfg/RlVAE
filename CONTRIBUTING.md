# Contributing to RlVAE

Thank you for your interest in contributing to the Riemannian Flow VAE project! 🎉

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/antoinelfg/RlVAE.git
   cd RlVAE
   ```
3. **Set up the development environment**:
   ```bash
   pip install -e .
   python test_setup.py
   ```

## 🧪 Development Workflow

### Setting up the Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Verify setup
python tests/test_setup.py
```

### Running Tests

```bash
# Validate setup and imports
python tests/test_setup.py

# Quick training test (1 epoch)
python run_clean_training.py --loop_mode open --n_epochs 1 --n_train_samples 10
```

### Code Style

- **Format with Black**: `black src/ scripts/`
- **Type hints**: Use type hints for function signatures
- **Docstrings**: Use Google-style docstrings
- **Imports**: Use absolute imports from `src/`

## 📝 Contribution Types

### 🐛 Bug Reports
- Use the GitHub issue template
- Include minimal reproduction steps
- Provide system information and logs

### ✨ Feature Requests
- Open an issue first to discuss the feature
- Describe the use case and expected behavior
- Consider backward compatibility

### 🧠 Model Improvements
- New sampling methods
- Alternative posterior types
- Enhanced training strategies
- Performance optimizations

### 📊 Data Processing
- New dataset support
- Preprocessing improvements
- Evaluation metrics

## 🔄 Pull Request Process

1. **Create a feature branch**: `git checkout -b feature/your-feature`
2. **Make your changes** with clear, focused commits
3. **Test your changes**: Run `python tests/test_setup.py`
4. **Update documentation** if needed
5. **Submit a pull request** with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots/plots for visual changes

## 📚 Documentation

- **Code comments**: Explain complex algorithms and mathematical operations
- **README updates**: Keep examples and usage up-to-date
- **Training guides**: Document new training procedures
- **Configuration**: Document new config options

## 🎯 Research Contributions

For research-related contributions:

- **Mathematical notation**: Use consistent notation in comments
- **Algorithm references**: Cite relevant papers in docstrings
- **Experimental validation**: Include results and comparisons
- **Reproducibility**: Provide seed values and configuration

## 💡 Tips for Contributors

- **Start small**: Begin with documentation improvements or small bug fixes
- **Ask questions**: Open an issue if you're unsure about implementation details
- **Stay focused**: Keep PRs focused on a single feature or fix
- **Be patient**: Allow time for review and discussion

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For general questions and ideas
- **Documentation**: Check README and training guides first

---

Happy contributing! 🎉 