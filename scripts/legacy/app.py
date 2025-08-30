"""
VAE Experiment Management & Analysis Platform
===========================================

A comprehensive Streamlit application for managing, running, and analyzing 
Variational Autoencoder experiments with interactive latent space exploration.

Features:
- Experiment configuration and execution
- Real-time training monitoring
- Interactive latent space visualization
- Model comparison and analysis
- Advanced metrics and diagnostics

Author: VAE Research Team
"""

import streamlit as st
import sys
from pathlib import Path
import torch

# Add src to path for imports
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Configure Streamlit page
st.set_page_config(
    page_title="VAE Research Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/antoinelfg/RlVAE',
        'Report a bug': 'https://github.com/antoinelfg/RlVAE/issues',
        'About': """
        # VAE Research Platform
        
        A comprehensive platform for Variational Autoencoder research with:
        - Multiple VAE architectures (Modular RlVAE, Hybrid RlVAE, Standard VAE)
        - Interactive latent space exploration
        - Real-time experiment monitoring
        - Advanced visualization and analysis tools
        
        Built with ❤️ using Streamlit, PyTorch, and PyTorch Lightning.
        """
    }
)

# Import after configuring Streamlit to avoid warnings
from streamlit_option_menu import option_menu
from app.components.sidebar import render_sidebar, get_device_info
from app.pages import (
    overview, 
    experiment_manager, 
    model_inference, 
    latent_exploration, 
    model_comparison,
    visualization_gallery
)
from app.utils.session_state import initialize_session_state


def main():
    """Main application entry point."""
    
    # Initialize session state
    initialize_session_state()
    
    # Custom CSS for professional styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .status-running {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-stopped {
        color: #dc3545;
        font-weight: bold;
    }
    
    .status-idle {
        color: #6c757d;
        font-weight: bold;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 VAE Research Platform</h1>
        <p>Advanced Variational Autoencoder Experiment Management & Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    render_sidebar()
    
    # Main navigation menu
    with st.container():
        selected_page = option_menu(
            menu_title=None,
            options=[
                "🏠 Overview", 
                "🧪 Experiment Manager", 
                "🔮 Model Inference", 
                "🌌 Latent Exploration", 
                "📊 Model Comparison",
                "🎨 Visualization Gallery"
            ],
            icons=[
                "house", 
                "flask", 
                "cpu", 
                "globe", 
                "bar-chart",
                "palette"
            ],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {
                    "padding": "0!important", 
                    "background-color": "transparent"
                },
                "icon": {"color": "#667eea", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "center",
                    "margin": "0px",
                    "--hover-color": "#eee",
                    "background-color": "transparent",
                    "color": "#333"
                },
                "nav-link-selected": {
                    "background": "linear-gradient(90deg, #667eea 0%, #764ba2 100%)",
                    "color": "white"
                },
            }
        )
    
    # Store selected page in session state
    st.session_state.current_page = selected_page
    
    # Route to appropriate page
    if selected_page == "🏠 Overview":
        overview.render()
    elif selected_page == "🧪 Experiment Manager":
        experiment_manager.render()
    elif selected_page == "🔮 Model Inference":
        model_inference.render()
    elif selected_page == "🌌 Latent Exploration":
        latent_exploration.render()
    elif selected_page == "📊 Model Comparison":
        model_comparison.render()
    elif selected_page == "🎨 Visualization Gallery":
        visualization_gallery.render()
    
    # Footer with device info and status
    with st.expander("🔧 System Information", expanded=False):
        device_info = get_device_info()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🖥️ Device", device_info["device"])
            if device_info["cuda_available"]:
                st.metric("📱 GPU Memory", f"{device_info['gpu_memory_gb']:.1f} GB")
        
        with col2:
            st.metric("🐍 Python", device_info["python_version"])
            st.metric("🔥 PyTorch", device_info["torch_version"])
        
        with col3:
            if device_info["cuda_available"]:
                st.metric("⚡ CUDA", device_info["cuda_version"])
                st.metric("🎯 GPU", device_info["gpu_name"])


if __name__ == "__main__":
    main()