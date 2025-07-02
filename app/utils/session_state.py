"""
Session State Management for VAE Research Platform
=================================================

Manages Streamlit session state for persistent data across page navigations.
"""

import streamlit as st
from typing import Dict, Any, Optional
import torch
from pathlib import Path


def initialize_session_state():
    """Initialize all session state variables with default values."""
    
    # Navigation state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Overview"
    
    # Experiment management
    if 'experiments' not in st.session_state:
        st.session_state.experiments = {}
    
    if 'current_experiment' not in st.session_state:
        st.session_state.current_experiment = None
    
    if 'experiment_status' not in st.session_state:
        st.session_state.experiment_status = "idle"  # idle, running, completed, error
    
    # Model states
    if 'loaded_models' not in st.session_state:
        st.session_state.loaded_models = {}
    
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
    
    if 'model_config' not in st.session_state:
        st.session_state.model_config = None
    
    # Latent space exploration
    if 'latent_samples' not in st.session_state:
        st.session_state.latent_samples = None
    
    if 'latent_grid_cache' not in st.session_state:
        st.session_state.latent_grid_cache = {}
    
    if 'interpolation_cache' not in st.session_state:
        st.session_state.interpolation_cache = {}
    
    # Visualization settings
    if 'visualization_settings' not in st.session_state:
        st.session_state.visualization_settings = {
            'latent_grid_resolution': 10,
            'interpolation_steps': 20,
            'pca_components': 2,
            'umap_neighbors': 15,
            'plot_theme': 'plotly'
        }
    
    # Training monitoring
    if 'training_metrics' not in st.session_state:
        st.session_state.training_metrics = []
    
    if 'real_time_monitoring' not in st.session_state:
        st.session_state.real_time_monitoring = False
    
    # File management
    if 'output_directory' not in st.session_state:
        st.session_state.output_directory = Path("outputs")
    
    if 'checkpoint_paths' not in st.session_state:
        st.session_state.checkpoint_paths = []
    
    # Comparison studies
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = {}
    
    if 'selected_models_for_comparison' not in st.session_state:
        st.session_state.selected_models_for_comparison = []


def get_session_state() -> Dict[str, Any]:
    """Get all session state variables as a dictionary."""
    return dict(st.session_state)


def reset_session_state():
    """Reset all session state variables to default values."""
    keys_to_keep = ['current_page']  # Keep navigation state
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    initialize_session_state()


def save_experiment_state(experiment_name: str, state: Dict[str, Any]):
    """Save experiment state for later retrieval."""
    if 'experiments' not in st.session_state:
        st.session_state.experiments = {}
    
    st.session_state.experiments[experiment_name] = state


def load_experiment_state(experiment_name: str) -> Optional[Dict[str, Any]]:
    """Load previously saved experiment state."""
    return st.session_state.experiments.get(experiment_name)


def update_training_metrics(new_metrics: Dict[str, float]):
    """Update training metrics for real-time monitoring."""
    if 'training_metrics' not in st.session_state:
        st.session_state.training_metrics = []
    
    st.session_state.training_metrics.append(new_metrics)
    
    # Keep only last 1000 metrics to avoid memory issues
    if len(st.session_state.training_metrics) > 1000:
        st.session_state.training_metrics = st.session_state.training_metrics[-1000:]


def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information."""
    import platform
    import sys
    
    device_info = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'cuda_available': torch.cuda.is_available(),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'torch_version': torch.__version__,
        'platform': platform.platform(),
    }
    
    if torch.cuda.is_available():
        device_info.update({
            'cuda_version': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'gpu_count': torch.cuda.device_count()
        })
    
    return device_info


def cache_computation_result(key: str, result: Any, cache_type: str = 'general'):
    """Cache computation results to avoid recomputation."""
    cache_name = f'{cache_type}_cache'
    
    if cache_name not in st.session_state:
        st.session_state[cache_name] = {}
    
    st.session_state[cache_name][key] = result


def get_cached_result(key: str, cache_type: str = 'general') -> Optional[Any]:
    """Retrieve cached computation result."""
    cache_name = f'{cache_type}_cache'
    
    if cache_name not in st.session_state:
        return None
    
    return st.session_state[cache_name].get(key)


def clear_cache(cache_type: str = 'all'):
    """Clear specific or all cached results."""
    if cache_type == 'all':
        cache_keys = [key for key in st.session_state.keys() if key.endswith('_cache')]
        for key in cache_keys:
            st.session_state[key] = {}
    else:
        cache_name = f'{cache_type}_cache'
        if cache_name in st.session_state:
            st.session_state[cache_name] = {}