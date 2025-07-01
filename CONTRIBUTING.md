# Contributing to RlVAE 🤝

We welcome contributions to the RlVAE project! This guide will help you get started.

## 🚀 Quick Start

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/antoinelfg/RlVAE.git
cd RlVAE

# Install in development mode
pip install -e .

# Validate installation
python tests/test_setup.py
```

## 🧪 Testing

Before submitting contributions, ensure all tests pass:

```bash
# Environment validation
python tests/test_setup.py

# Component validation
python tests/test_modular_components.py

# Integration testing
python tests/test_hybrid_model.py

# All tests at once
make test-all
```

### Quick Development Test
```bash
# Test with modular model (recommended)
python run_experiment.py model=modular_rlvae training=quick visualization=minimal
```

## 🏗️ Development Workflow

### 1. Setting Up Your Development Environment
```bash
# Fork the repository and clone your fork
git clone https://github.com/YOUR_USERNAME/RlVAE.git
cd RlVAE

# Create a development branch
git checkout -b feature/your-feature-name

# Install development dependencies
make install-dev
```

### 2. Making Changes
- **Focus on modular components** in `src/models/components/` and `src/models/samplers/`
- **Follow the modular architecture** patterns
- **Add tests** for new functionality
- **Update documentation** as needed

### 3. Testing Your Changes
```bash
# Quick validation
make test

# Full test suite
make test-all

# Code quality checks
make lint format type-check
```

## 📝 Code Style

We use automated code formatting and linting:

```bash
# Format code
make format

# Check formatting
make format-check

# Run linting
make lint

# Type checking
make type-check
```

## 🧩 Contributing to Modular Components

### Adding New Components
1. **Create component** in appropriate directory (`src/models/components/` or `src/models/samplers/`)
2. **Follow existing patterns** (see `MetricTensor` or `LossManager` for examples)
3. **Add comprehensive tests** in `tests/`
4. **Update configurations** in `conf/model/` if needed

### Component Guidelines
- **Inherit from `nn.Module`** for PyTorch components
- **Include type hints** for all methods
- **Add comprehensive docstrings**
- **Handle device placement** properly
- **Include error handling** and validation

## 📚 Documentation

### Update Documentation When:
- Adding new models or components
- Changing configuration options
- Adding new experiment types
- Modifying installation requirements

### Documentation Files
- `README.md` - Main project overview
- `docs/TRAINING_GUIDE.md` - Training workflows
- `docs/MODULAR_TRAINING_GUIDE.md` - Modular system guide
- Component docstrings and type hints

## 🔬 Research Contributions

### Adding New Research Features
1. **Use modular architecture** - leverage existing components
2. **Follow configuration patterns** - use Hydra configs
3. **Add experiments** - create new experiment configurations
4. **Benchmark performance** - compare against existing methods
5. **Document thoroughly** - explain research contribution

### Preferred Areas for Contributions
- **New sampling strategies** in `src/models/samplers/`
- **Advanced metric learning** in `src/models/components/`
- **Visualization improvements** in `src/visualizations/`
- **Performance optimizations** across all components
- **New model architectures** following modular patterns

## 🚦 Pull Request Process

1. **Ensure tests pass**: `make ci`
2. **Update documentation** as needed
3. **Create descriptive PR title** and description
4. **Reference issues** if applicable
5. **Be responsive** to code review feedback

### PR Checklist
- [ ] All tests pass (`make test-all`)
- [ ] Code follows style guidelines (`make format lint`)
- [ ] Documentation updated if needed
- [ ] New functionality includes tests
- [ ] Modular architecture patterns followed

## 🐛 Bug Reports

### Before Reporting
1. **Search existing issues** for similar problems
2. **Test with latest version**
3. **Validate environment**: `python tests/test_setup.py`

### Bug Report Template
```markdown
**Environment**
- Python version:
- PyTorch version:
- CUDA version (if applicable):
- Operating system:

**Description**
Clear description of the bug

**Steps to Reproduce**
1. 
2. 
3. 

**Expected Behavior**
What you expected to happen

**Actual Behavior**
What actually happened

**Additional Context**
Logs, screenshots, or other relevant information
```

## 💡 Feature Requests

We welcome feature requests! Please:
1. **Check existing issues** for similar requests
2. **Describe the use case** clearly
3. **Explain the benefit** to the research community
4. **Consider modular implementation** approaches

## 🏆 Recognition

Contributors will be acknowledged in:
- Repository contributors list
- Release notes for significant contributions
- Documentation acknowledgments

## 📞 Questions?

- **Create an issue** for questions about contributing
- **Start a discussion** for design questions
- **Reference existing code** for implementation patterns

---

**Thank you for contributing to RlVAE! Your contributions help advance Riemannian geometry research.** 🙏 