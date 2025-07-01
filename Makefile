.PHONY: help clean install install-dev test lint format type-check pre-commit docs
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

install-dev: ## Install the package in development mode with all dependencies
	pip install -e ".[dev,docs]"
	pre-commit install

test: ## Run the test suite
	python tests/test_setup.py

test-modular: ## Test modular components
	python tests/test_modular_components.py

test-integration: ## Test hybrid model integration
	python tests/test_hybrid_model.py

test-all: ## Run all tests
	python tests/test_setup.py && python tests/test_modular_components.py && python tests/test_hybrid_model.py

lint: ## Run linting checks
	flake8 src/ scripts/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ scripts/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

format: ## Format code with black and isort
	black src/ scripts/
	isort src/ scripts/

format-check: ## Check code formatting
	black --check --diff src/ scripts/
	isort --check-only --diff src/ scripts/

type-check: ## Run type checking with mypy
	mypy src/ --ignore-missing-imports

security-check: ## Run security checks with bandit
	bandit -r src/ -f json -o bandit-report.json || true

pre-commit: ## Run all pre-commit hooks
	pre-commit run --all-files

clean: ## Clean up temporary files
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

clean-training: ## Clean up training files (interactive)
	python scripts/cleanup_training_files.py

docs: ## Build documentation
	@echo "Documentation build not yet implemented"

train-quick: ## Quick training with modular model
	python run_experiment.py model=modular_rlvae training=quick visualization=minimal

train-full: ## Full training with modular model
	python run_experiment.py model=modular_rlvae training=full_data visualization=standard

train-comparison: ## Compare all models
	python run_experiment.py experiment=comparison_study

experiment-quick: ## Quick experiment validation
	scripts/run_quick_test.sh

experiment-suite: ## Full weekend experiment suite
	scripts/run_weekend_experiments.sh

setup-data: ## Set up data files
	python scripts/extract_cyclic_sequences.py
	python scripts/train_and_extract_vanilla_vae.py
	python scripts/create_identity_metric_temp_0_7.py

validate: ## Validate the setup
	python tests/test_setup.py && echo "✅ Setup validation complete"

ci: lint format-check type-check test ## Run all CI checks locally 