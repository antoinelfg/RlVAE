"""
Sidebar Component for VAE Research Platform
==========================================

Provides navigation, system information, and quick actions in the sidebar.
"""

import streamlit as st
from typing import Dict, Any
import torch
import sys
import platform
from pathlib import Path
from app.utils.session_state import get_device_info


def render_sidebar():
    """Render the main sidebar with navigation and system info."""
    
    with st.sidebar:
        # Platform status
        st.markdown("### 🚀 Platform Status")
        
        # Device info summary
        device_info = get_device_info()
        device_color = "🟢" if device_info["cuda_available"] else "🟡"
        st.markdown(f"{device_color} **Device:** {device_info['device'].upper()}")
        
        if device_info["cuda_available"]:
            st.markdown(f"🎯 **GPU:** {device_info['gpu_name'][:20]}...")
            st.markdown(f"💾 **Memory:** {device_info['gpu_memory_gb']:.1f} GB")
        
        # Experiment status
        st.markdown("### 🧪 Experiment Status")
        status = st.session_state.get('experiment_status', 'idle')
        
        if status == 'running':
            st.markdown("🟢 **Status:** Running")
            if st.button("⏹️ Stop Experiment", type="secondary"):
                st.session_state.experiment_status = 'stopped'
                st.rerun()
        elif status == 'completed':
            st.markdown("✅ **Status:** Completed")
        elif status == 'error':
            st.markdown("🔴 **Status:** Error")
        else:
            st.markdown("⚪ **Status:** Idle")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", help="Refresh current page"):
                st.rerun()
        with col2:
            if st.button("🧹 Clear Cache", help="Clear all cached data"):
                clear_all_cache()
                st.success("Cache cleared!")
                st.rerun()
        
        # Model quick info
        if st.session_state.current_model is not None:
            st.markdown("### 🎯 Current Model")
            model = st.session_state.current_model
            if hasattr(model, 'get_model_summary'):
                summary = model.get_model_summary()
                st.markdown(f"**Type:** {summary.get('model_name', 'Unknown')}")
                st.markdown(f"**Latent Dim:** {summary.get('architecture', {}).get('latent_dim', '?')}")
        
        # Settings expander
        with st.expander("⚙️ Settings", expanded=False):
            render_settings_panel()
        
        # Help and documentation
        st.markdown("### 📚 Resources")
        st.markdown("""
        - [📖 Documentation](https://github.com/antoinelfg/RlVAE)
        - [🐛 Report Issues](https://github.com/antoinelfg/RlVAE/issues)
        - [💡 Feature Requests](https://github.com/antoinelfg/RlVAE/discussions)
        """)


def render_settings_panel():
    """Render the settings panel in the sidebar."""
    
    st.subheader("Visualization Settings")
    
    # Grid resolution for latent exploration
    grid_res = st.slider(
        "Latent Grid Resolution",
        min_value=5,
        max_value=20,
        value=st.session_state.visualization_settings.get('latent_grid_resolution', 10),
        help="Resolution for latent space grid visualization"
    )
    st.session_state.visualization_settings['latent_grid_resolution'] = grid_res
    
    # Interpolation steps
    interp_steps = st.slider(
        "Interpolation Steps",
        min_value=10,
        max_value=50,
        value=st.session_state.visualization_settings.get('interpolation_steps', 20),
        help="Number of steps for latent interpolation"
    )
    st.session_state.visualization_settings['interpolation_steps'] = interp_steps
    
    # UMAP neighbors
    umap_neighbors = st.slider(
        "UMAP Neighbors",
        min_value=5,
        max_value=50,
        value=st.session_state.visualization_settings.get('umap_neighbors', 15),
        help="Number of neighbors for UMAP embedding"
    )
    st.session_state.visualization_settings['umap_neighbors'] = umap_neighbors
    
    # Plot theme
    plot_theme = st.selectbox(
        "Plot Theme",
        options=['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn'],
        index=0,
        help="Theme for interactive plots"
    )
    st.session_state.visualization_settings['plot_theme'] = plot_theme
    
    st.subheader("Performance Settings")
    
    # Real-time monitoring
    real_time = st.checkbox(
        "Real-time Monitoring",
        value=st.session_state.get('real_time_monitoring', False),
        help="Enable real-time training metrics updates"
    )
    st.session_state.real_time_monitoring = real_time
    
    # Auto-refresh interval
    if real_time:
        refresh_interval = st.slider(
            "Refresh Interval (seconds)",
            min_value=1,
            max_value=30,
            value=5,
            help="How often to refresh real-time data"
        )
        st.session_state.refresh_interval = refresh_interval


def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device and system information."""
    
    device_info = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'cuda_available': torch.cuda.is_available(),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'torch_version': torch.__version__,
        'platform': platform.platform(),
        'processor': platform.processor() or 'Unknown',
        'architecture': platform.architecture()[0],
        'ram_gb': get_system_memory_gb()
    }
    
    if torch.cuda.is_available():
        try:
            device_info.update({
                'cuda_version': torch.version.cuda,
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'gpu_count': torch.cuda.device_count(),
                'gpu_memory_used_gb': torch.cuda.memory_allocated(0) / 1024**3,
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved(0) / 1024**3
            })
        except Exception as e:
            device_info['gpu_error'] = str(e)
    
    return device_info


def get_system_memory_gb() -> float:
    """Get total system memory in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1024**3
    except ImportError:
        return 0.0


def clear_all_cache():
    """Clear all cached data from session state."""
    cache_keys = [key for key in st.session_state.keys() if 'cache' in key]
    for key in cache_keys:
        if isinstance(st.session_state[key], dict):
            st.session_state[key] = {}
        else:
            del st.session_state[key]


def render_model_status():
    """Render current model status in sidebar."""
    if st.session_state.current_model is not None:
        st.markdown("### 🎯 Current Model")
        
        model = st.session_state.current_model
        
        if hasattr(model, 'get_model_summary'):
            summary = model.get_model_summary()
            
            st.markdown(f"**Model:** {summary.get('model_name', 'Unknown')}")
            
            arch = summary.get('architecture', {})
            st.markdown(f"**Latent Dim:** {arch.get('latent_dim', '?')}")
            st.markdown(f"**Flows:** {arch.get('n_flows', '?')}")
            
            config = summary.get('configuration', {})
            if config.get('uses_riemannian', False):
                st.markdown("**Type:** Riemannian VAE ⚡")
            else:
                st.markdown("**Type:** Standard VAE")
        
        # Model actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Analyze", help="Analyze current model"):
                st.session_state.current_page = "📊 Model Comparison"
                st.rerun()
        
        with col2:
            if st.button("🌌 Explore", help="Explore latent space"):
                st.session_state.current_page = "🌌 Latent Exploration"
                st.rerun()


def render_experiment_controls():
    """Render experiment control buttons in sidebar."""
    st.markdown("### 🧪 Experiment Controls")
    
    status = st.session_state.get('experiment_status', 'idle')
    
    if status == 'idle':
        if st.button("▶️ Start New Experiment", type="primary"):
            st.session_state.current_page = "🧪 Experiment Manager"
            st.rerun()
    
    elif status == 'running':
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏸️ Pause"):
                st.session_state.experiment_status = 'paused'
                st.rerun()
        with col2:
            if st.button("⏹️ Stop"):
                st.session_state.experiment_status = 'stopped'
                st.rerun()
    
    elif status in ['completed', 'stopped', 'error']:
        if st.button("🔄 Start New", type="primary"):
            st.session_state.experiment_status = 'idle'
            st.session_state.current_page = "🧪 Experiment Manager"
            st.rerun()