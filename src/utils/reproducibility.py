"""
Reproducibility utilities for RlVAE experiments.
Provides comprehensive seeding and determinism controls.
"""

import os
import random
import warnings
from typing import Optional

import numpy as np
import torch
import lightning as L


def set_global_seed(
    seed: int = 42,
    deterministic: bool = True,
    benchmark: bool = False,
    warn_only: bool = False
) -> None:
    """
    Set global random seed for full reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        deterministic: Enable CUDA deterministic algorithms (slower but reproducible)
        benchmark: Enable CUDNN benchmarking (faster but less reproducible)
        warn_only: Only warn about non-deterministic operations instead of erroring
    """
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Set Lightning seed
    L.seed_everything(seed, workers=True)
    
    # Configure CUDA determinism
    if torch.cuda.is_available() and deterministic:
        # Force deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = benchmark
        
        # Set CUDA deterministic environment variables
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Use deterministic algorithms (PyTorch 1.8+)
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        elif hasattr(torch, 'set_deterministic'):
            torch.set_deterministic(True)
        
        print(f"🔒 CUDA deterministic mode enabled (seed: {seed})")
        if not benchmark:
            print("⚠️  CUDNN benchmarking disabled for reproducibility (may be slower)")
    
    elif torch.cuda.is_available() and not deterministic:
        # Enable benchmarking for performance
        torch.backends.cudnn.benchmark = True
        print(f"⚡ CUDA benchmark mode enabled (seed: {seed}, not fully deterministic)")
    
    print(f"🎲 Global seed set to: {seed}")


def get_seed_info() -> dict:
    """
    Get current seeding and determinism configuration info.
    
    Returns:
        Dictionary with current seed configuration
    """
    info = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'random_state': random.getstate()[1][0],  # First random number
        'numpy_seed_set': hasattr(np.random, '_bit_generator'),
    }
    
    if torch.cuda.is_available():
        info.update({
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'deterministic_algorithms': getattr(torch, '_is_deterministic_algorithms_enabled', 'unknown'),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'not_set'),
        })
    
    return info


def verify_reproducibility(
    device: torch.device,
    shape: tuple = (10, 10),
    seed: int = 42,
    num_runs: int = 3
) -> bool:
    """
    Verify that random operations are reproducible.
    
    Args:
        device: Device to test on
        shape: Shape of test tensors
        seed: Seed to use for testing
        num_runs: Number of runs to compare
        
    Returns:
        True if all runs produce identical results
    """
    results = []
    
    for run in range(num_runs):
        set_global_seed(seed)
        
        # Test random operations
        torch_rand = torch.randn(shape, device=device)
        numpy_rand = torch.from_numpy(np.random.randn(*shape)).to(device)
        
        results.append({
            'torch': torch_rand.cpu(),
            'numpy': numpy_rand.cpu()
        })
    
    # Check if all runs are identical
    for i in range(1, num_runs):
        torch_identical = torch.allclose(results[0]['torch'], results[i]['torch'])
        numpy_identical = torch.allclose(results[0]['numpy'], results[i]['numpy'])
        
        if not (torch_identical and numpy_identical):
            warnings.warn(f"Reproducibility test failed at run {i}")
            return False
    
    print(f"✅ Reproducibility verified across {num_runs} runs")
    return True


def configure_for_experiment(
    seed: Optional[int] = None,
    experiment_type: str = "research",
    device: Optional[torch.device] = None
) -> int:
    """
    Configure reproducibility settings for different experiment types.
    
    Args:
        seed: Random seed (uses 42 if None)
        experiment_type: "research" (full determinism) or "production" (performance)
        device: Device to verify on
        
    Returns:
        The seed that was set
    """
    if seed is None:
        seed = 42
    
    if experiment_type == "research":
        # Full determinism for research reproducibility
        set_global_seed(seed, deterministic=True, benchmark=False)
    elif experiment_type == "production":
        # Performance-oriented for production
        set_global_seed(seed, deterministic=False, benchmark=True)
    else:
        raise ValueError(f"Unknown experiment_type: {experiment_type}")
    
    # Verify reproducibility if device is provided
    if device is not None:
        verify_reproducibility(device, seed=seed)
    
    # Print configuration info
    info = get_seed_info()
    print(f"🔧 Reproducibility configured for: {experiment_type}")
    print(f"   - CUDA available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"   - Deterministic: {info['cudnn_deterministic']}")
        print(f"   - Benchmark: {info['cudnn_benchmark']}")
    
    return seed
