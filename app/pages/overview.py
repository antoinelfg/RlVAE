"""
Overview Page - VAE Research Platform Dashboard
=============================================

Provides an overview of the platform, recent experiments, and quick access to features.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


def render():
    """Render the overview dashboard page."""
    
    st.title("🏠 Platform Overview")
    st.markdown("Welcome to the VAE Research Platform - your comprehensive tool for Variational Autoencoder research and experimentation.")
    
    # Quick stats row
    render_quick_stats()
    
    # Recent activity and system status
    col1, col2 = st.columns(2)
    
    with col1:
        render_recent_experiments()
    
    with col2:
        render_system_status()
    
    # Feature highlights
    render_feature_highlights()
    
    # Getting started guide
    render_getting_started()


def render_quick_stats():
    """Render quick statistics in metrics format."""
    
    st.markdown("### 📊 Quick Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get experiment count
    experiment_count = len(st.session_state.get('experiments', {}))
    
    # Get model count
    model_count = len(st.session_state.get('loaded_models', {}))
    
    # Get cache size (approximate)
    cache_items = sum(
        len(cache) if isinstance(cache, dict) else 1 
        for key, cache in st.session_state.items() 
        if 'cache' in key
    )
    
    # System status
    device_info = st.session_state.get('device_info', {})
    device_status = "🟢 GPU" if device_info.get('cuda_available', False) else "🟡 CPU"
    
    with col1:
        st.metric(
            label="🧪 Experiments", 
            value=experiment_count,
            help="Total number of experiments run"
        )
    
    with col2:
        st.metric(
            label="🎯 Models", 
            value=model_count,
            help="Currently loaded models"
        )
    
    with col3:
        st.metric(
            label="💾 Cache Items", 
            value=cache_items,
            help="Cached computation results"
        )
    
    with col4:
        st.metric(
            label="🖥️ Device", 
            value=device_status,
            help="Current computation device"
        )


def render_recent_experiments():
    """Render recent experiments panel."""
    
    st.markdown("### 🧪 Recent Experiments")
    
    experiments = st.session_state.get('experiments', {})
    
    if not experiments:
        st.info("No experiments run yet. Start your first experiment in the Experiment Manager!")
        
        if st.button("🚀 Start First Experiment", type="primary"):
            st.session_state.current_page = "🧪 Experiment Manager"
            st.rerun()
    else:
        # Show recent experiments
        for exp_name, exp_data in list(experiments.items())[-5:]:  # Last 5 experiments
            with st.expander(f"📋 {exp_name}", expanded=False):
                if isinstance(exp_data, dict):
                    st.write(f"**Status:** {exp_data.get('status', 'Unknown')}")
                    st.write(f"**Model:** {exp_data.get('model_type', 'Unknown')}")
                    if 'timestamp' in exp_data:
                        st.write(f"**Date:** {exp_data['timestamp']}")
                else:
                    st.write("Experiment data available")
        
        # Quick actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 View All"):
                st.session_state.current_page = "🧪 Experiment Manager"
                st.rerun()
        with col2:
            if st.button("🧹 Clear History"):
                st.session_state.experiments = {}
                st.success("Experiment history cleared!")
                st.rerun()


def render_system_status():
    """Render system status and performance metrics."""
    
    st.markdown("### 🖥️ System Status")
    
    # Get device info
    device_info = st.session_state.get('device_info', {})
    
    if device_info.get('cuda_available', False):
        # GPU Status
        st.markdown("#### 🎯 GPU Information")
        st.write(f"**Device:** {device_info.get('gpu_name', 'Unknown GPU')}")
        
        memory_total = device_info.get('gpu_memory_gb', 0)
        memory_used = device_info.get('gpu_memory_used_gb', 0)
        memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
        
        st.progress(memory_percent / 100, text=f"GPU Memory: {memory_used:.1f}/{memory_total:.1f} GB ({memory_percent:.1f}%)")
        
        # Create memory usage chart
        if memory_total > 0:
            fig = go.Figure(data=[
                go.Bar(
                    x=['GPU Memory'],
                    y=[memory_used],
                    name='Used',
                    marker_color='#ff6b6b'
                ),
                go.Bar(
                    x=['GPU Memory'],
                    y=[memory_total - memory_used],
                    name='Available',
                    marker_color='#51cf66',
                    base=[memory_used]
                )
            ])
            
            fig.update_layout(
                barmode='stack',
                height=200,
                showlegend=True,
                title="GPU Memory Usage",
                yaxis_title="Memory (GB)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # CPU Status
        st.markdown("#### 💻 CPU Information")
        st.info("Using CPU for computation. For better performance, consider using a GPU-enabled environment.")
    
    # System info
    st.markdown("#### 🔧 Environment")
    if device_info:
        st.write(f"**Python:** {device_info.get('python_version', 'Unknown')}")
        st.write(f"**PyTorch:** {device_info.get('torch_version', 'Unknown')}")
        st.write(f"**Platform:** {device_info.get('platform', 'Unknown')}")


def render_feature_highlights():
    """Render feature highlights and capabilities."""
    
    st.markdown("### ✨ Platform Features")
    
    features = [
        {
            "title": "🧪 Experiment Management",
            "description": "Configure, run, and monitor VAE experiments with real-time metrics",
            "action": "Experiment Manager"
        },
        {
            "title": "🔮 Model Inference",
            "description": "Load trained models and perform encoding/decoding operations",
            "action": "Model Inference"
        },
        {
            "title": "🌌 Latent Space Exploration",
            "description": "Interactive visualization and exploration of latent representations",
            "action": "Latent Exploration"
        },
        {
            "title": "📊 Model Comparison",
            "description": "Compare different VAE architectures and their performance",
            "action": "Model Comparison"
        },
        {
            "title": "🎨 Visualization Gallery",
            "description": "Advanced plots and analysis tools for deep insights",
            "action": "Visualization Gallery"
        }
    ]
    
    cols = st.columns(len(features))
    
    for idx, feature in enumerate(features):
        with cols[idx]:
            st.markdown(f"#### {feature['title']}")
            st.write(feature['description'])
            
            if st.button(f"Explore", key=f"feature_{idx}"):
                st.session_state.current_page = f"🧪 {feature['action']}" if "Manager" in feature['action'] else f"🔮 {feature['action']}" if "Inference" in feature['action'] else f"🌌 {feature['action']}" if "Exploration" in feature['action'] else f"📊 {feature['action']}" if "Comparison" in feature['action'] else f"🎨 {feature['action']}"
                st.rerun()


def render_getting_started():
    """Render getting started guide."""
    
    st.markdown("### 🚀 Getting Started")
    
    with st.expander("📚 Quick Start Guide", expanded=False):
        st.markdown("""
        #### 1. 🧪 Run Your First Experiment
        - Navigate to **Experiment Manager**
        - Choose a VAE model configuration
        - Set training parameters
        - Click **Start Experiment**
        
        #### 2. 🔮 Load a Pre-trained Model
        - Go to **Model Inference**
        - Upload or select a checkpoint
        - Test encoding/decoding capabilities
        
        #### 3. 🌌 Explore Latent Space
        - Visit **Latent Exploration**
        - Generate interactive visualizations
        - Experiment with latent interpolations
        
        #### 4. 📊 Compare Models
        - Use **Model Comparison** 
        - Select multiple models
        - Analyze performance metrics
        
        #### 5. 🎨 Advanced Analysis
        - Check **Visualization Gallery**
        - Generate publication-ready plots
        - Export results and figures
        """)
    
    with st.expander("🔧 Configuration Help", expanded=False):
        st.markdown("""
        #### Available VAE Models
        - **Modular RlVAE**: Fully modular Riemannian VAE with configurable components
        - **Hybrid RlVAE**: Performance-optimized version with 2x faster computations  
        - **Standard RlVAE**: Original Riemannian Flow VAE implementation
        - **Vanilla VAE**: Baseline VAE for comparisons
        
        #### Key Parameters
        - **Latent Dimension**: Size of the latent space representation
        - **Number of Flows**: Temporal flow layers for dynamics modeling
        - **Learning Rate**: Optimization learning rate
        - **Beta**: VAE regularization parameter
        - **Riemannian Beta**: Additional regularization for Riemannian geometry
        
        #### Performance Tips
        - Use GPU when available for faster training
        - Start with smaller datasets for quick testing
        - Enable real-time monitoring for training insights
        - Cache results to avoid recomputation
        """)
    
    with st.expander("🤝 Support & Resources", expanded=False):
        st.markdown("""
        #### Documentation
        - [📖 Full Documentation](https://github.com/antoinelfg/RlVAE)
        - [🎓 Training Guide](https://github.com/antoinelfg/RlVAE/blob/main/docs/TRAINING_GUIDE.md)
        - [🔧 API Reference](https://github.com/antoinelfg/RlVAE/blob/main/docs/)
        
        #### Community
        - [🐛 Report Issues](https://github.com/antoinelfg/RlVAE/issues)
        - [💡 Feature Requests](https://github.com/antoinelfg/RlVAE/discussions)
        - [💬 Discussions](https://github.com/antoinelfg/RlVAE/discussions)
        
        #### Citation
        If you use this platform in your research, please cite:
        ```
        @software{rlvae_platform,
          title={VAE Research Platform},
          author={VAE Research Team},
          year={2024},
          url={https://github.com/antoinelfg/RlVAE}
        }
        ```
        """)


def create_sample_metrics_chart():
    """Create a sample metrics chart for demonstration."""
    
    # Generate sample training data
    epochs = list(range(1, 51))
    train_loss = [4.5 - 2.0 * np.exp(-x/10) + 0.1 * np.random.randn() for x in epochs]
    val_loss = [4.3 - 1.8 * np.exp(-x/10) + 0.15 * np.random.randn() for x in epochs]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_loss,
        mode='lines',
        name='Training Loss',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_loss,
        mode='lines',
        name='Validation Loss',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    fig.update_layout(
        title="Sample Training Progress",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=300,
        showlegend=True
    )
    
    return fig