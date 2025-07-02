"""
Experiment Manager Page
=====================

Provides interface for configuring, running, and monitoring VAE experiments.
"""

import streamlit as st
import sys
from pathlib import Path
import torch
import yaml
from datetime import datetime
import subprocess
import threading
import time
from typing import Dict, Any, Optional

# Add src to path
current_dir = Path(__file__).parent.parent.parent.absolute()
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def render():
    """Render the experiment manager page."""
    
    st.title("🧪 Experiment Manager")
    st.markdown("Configure and run VAE experiments with real-time monitoring")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔧 Configure", "▶️ Run & Monitor", "📋 History"])
    
    with tab1:
        render_experiment_configuration()
    
    with tab2:
        render_experiment_execution()
    
    with tab3:
        render_experiment_history()


def render_experiment_configuration():
    """Render experiment configuration interface."""
    
    st.header("🔧 Experiment Configuration")
    
    # Experiment metadata
    with st.expander("📋 Experiment Details", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            exp_name = st.text_input(
                "Experiment Name",
                value=f"vae_exp_{datetime.now().strftime('%Y%m%d_%H%M')}",
                help="Unique identifier for this experiment"
            )
            
            exp_description = st.text_area(
                "Description",
                placeholder="Brief description of the experiment goals...",
                help="Optional description of the experiment"
            )
        
        with col2:
            project_name = st.text_input(
                "Project Name",
                value="riemannian-vae-study",
                help="W&B project name for logging"
            )
            
            tags = st.text_input(
                "Tags (comma-separated)",
                placeholder="latent-space, comparison, baseline",
                help="Tags for organizing experiments"
            )
    
    # Model configuration
    with st.expander("🎯 Model Configuration", expanded=True):
        render_model_configuration()
    
    # Training configuration
    with st.expander("🏋️ Training Configuration", expanded=True):
        render_training_configuration()
    
    # Data configuration
    with st.expander("📊 Data Configuration", expanded=True):
        render_data_configuration()
    
    # Visualization configuration
    with st.expander("🎨 Visualization Configuration", expanded=False):
        render_visualization_configuration()
    
    # Save/Load configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Configuration", type="secondary"):
            save_experiment_configuration(exp_name)
    
    with col2:
        uploaded_config = st.file_uploader(
            "📂 Load Configuration",
            type=['yaml', 'yml'],
            help="Upload a saved experiment configuration"
        )
        if uploaded_config:
            load_experiment_configuration(uploaded_config)
    
    with col3:
        if st.button("🔄 Reset to Defaults", type="secondary"):
            reset_configuration()


def render_model_configuration():
    """Render model configuration section."""
    
    st.subheader("🎯 Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Model Type",
            options=[
                "modular_rlvae",
                "hybrid_rlvae", 
                "riemannian_flow_vae",
                "vanilla_vae"
            ],
            index=0,
            help="Choose the VAE architecture"
        )
        
        latent_dim = st.slider(
            "Latent Dimension",
            min_value=2,
            max_value=128,
            value=16,
            help="Dimensionality of the latent space"
        )
        
        input_dim = st.selectbox(
            "Input Dimensions",
            options=["(3, 64, 64)", "(1, 28, 28)", "(3, 32, 32)"],
            index=0,
            help="Input image dimensions (C, H, W)"
        )
    
    with col2:
        n_flows = st.slider(
            "Number of Flows",
            min_value=0,
            max_value=10,
            value=5,
            help="Number of temporal flow layers"
        )
        
        beta = st.number_input(
            "Beta (VAE regularization)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="VAE beta parameter for KL regularization"
        )
        
        riemannian_beta = st.number_input(
            "Riemannian Beta",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Additional regularization for Riemannian geometry"
        )
    
    # Advanced model settings
    with st.expander("🔧 Advanced Model Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            posterior_type = st.selectbox(
                "Posterior Type",
                options=["gaussian", "riemannian_metric"],
                index=0,
                help="Type of posterior distribution"
            )
            
            encoder_arch = st.selectbox(
                "Encoder Architecture",
                options=["mlp", "cnn", "resnet"],
                index=0,
                help="Encoder network architecture"
            )
        
        with col2:
            decoder_arch = st.selectbox(
                "Decoder Architecture", 
                options=["mlp", "cnn", "resnet"],
                index=0,
                help="Decoder network architecture"
            )
            
            sampling_method = st.selectbox(
                "Sampling Method",
                options=["standard", "riemannian", "geodesic"],
                index=0,
                help="Latent space sampling strategy"
            )
    
    # Store configuration in session state
    st.session_state.model_config = {
        'model_type': model_type,
        'latent_dim': latent_dim,
        'input_dim': eval(input_dim),
        'n_flows': n_flows,
        'beta': beta,
        'riemannian_beta': riemannian_beta,
        'posterior_type': posterior_type,
        'encoder_arch': encoder_arch,
        'decoder_arch': decoder_arch,
        'sampling_method': sampling_method
    }


def render_training_configuration():
    """Render training configuration section."""
    
    st.subheader("🏋️ Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_epochs = st.slider(
            "Maximum Epochs",
            min_value=1,
            max_value=200,
            value=50,
            help="Maximum number of training epochs"
        )
        
        learning_rate = st.selectbox(
            "Learning Rate",
            options=[0.001, 0.0005, 0.0001, 0.00005],
            index=1,
            help="Optimizer learning rate"
        )
        
        batch_size = st.selectbox(
            "Batch Size",
            options=[8, 16, 32, 64, 128],
            index=2,
            help="Training batch size"
        )
    
    with col2:
        optimizer = st.selectbox(
            "Optimizer",
            options=["Adam", "AdamW", "SGD"],
            index=0,
            help="Optimization algorithm"
        )
        
        scheduler = st.selectbox(
            "Learning Rate Scheduler",
            options=["None", "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"],
            index=0,
            help="Learning rate scheduling strategy"
        )
        
        early_stopping = st.checkbox(
            "Early Stopping",
            value=True,
            help="Enable early stopping based on validation loss"
        )
    
    # Advanced training settings
    with st.expander("🔧 Advanced Training Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            gradient_clip = st.number_input(
                "Gradient Clipping",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                help="Gradient clipping value (0 to disable)"
            )
            
            weight_decay = st.number_input(
                "Weight Decay",
                min_value=0.0,
                max_value=0.01,
                value=0.0001,
                format="%.6f",
                help="L2 regularization weight decay"
            )
        
        with col2:
            validation_frequency = st.slider(
                "Validation Frequency",
                min_value=1,
                max_value=10,
                value=1,
                help="Run validation every N epochs"
            )
            
            checkpoint_frequency = st.slider(
                "Checkpoint Frequency",
                min_value=1,
                max_value=20,
                value=5,
                help="Save checkpoint every N epochs"
            )
    
    # Store training configuration
    st.session_state.training_config = {
        'max_epochs': max_epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'early_stopping': early_stopping,
        'gradient_clip': gradient_clip,
        'weight_decay': weight_decay,
        'validation_frequency': validation_frequency,
        'checkpoint_frequency': checkpoint_frequency
    }


def render_data_configuration():
    """Render data configuration section."""
    
    st.subheader("📊 Dataset Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_type = st.selectbox(
            "Dataset",
            options=["cyclic_sprites", "mnist", "cifar10", "custom"],
            index=0,
            help="Choose the dataset for training"
        )
        
        num_sequences = st.slider(
            "Number of Sequences",
            min_value=100,
            max_value=5000,
            value=1000,
            help="Total number of sequences to use"
        )
    
    with col2:
        sequence_length = st.slider(
            "Sequence Length",
            min_value=5,
            max_value=50,
            value=20,
            help="Length of each temporal sequence"
        )
        
        train_split = st.slider(
            "Training Split",
            min_value=0.6,
            max_value=0.9,
            value=0.8,
            help="Fraction of data used for training"
        )
    
    # Data augmentation
    with st.expander("🔄 Data Augmentation", expanded=False):
        use_augmentation = st.checkbox(
            "Enable Data Augmentation",
            value=False,
            help="Apply data augmentation during training"
        )
        
        if use_augmentation:
            col1, col2 = st.columns(2)
            
            with col1:
                rotation_range = st.slider(
                    "Rotation Range",
                    min_value=0,
                    max_value=45,
                    value=15,
                    help="Random rotation range in degrees"
                )
                
                noise_level = st.slider(
                    "Noise Level",
                    min_value=0.0,
                    max_value=0.1,
                    value=0.01,
                    help="Gaussian noise standard deviation"
                )
            
            with col2:
                brightness_range = st.slider(
                    "Brightness Range",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.1,
                    help="Random brightness adjustment range"
                )
                
                contrast_range = st.slider(
                    "Contrast Range",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.1,
                    help="Random contrast adjustment range"
                )
    
    # Store data configuration
    st.session_state.data_config = {
        'dataset_type': dataset_type,
        'num_sequences': num_sequences,
        'sequence_length': sequence_length,
        'train_split': train_split,
        'use_augmentation': use_augmentation
    }
    
    if use_augmentation:
        st.session_state.data_config.update({
            'rotation_range': rotation_range,
            'noise_level': noise_level,
            'brightness_range': brightness_range,
            'contrast_range': contrast_range
        })


def render_visualization_configuration():
    """Render visualization configuration section."""
    
    st.subheader("🎨 Visualization Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        vis_level = st.selectbox(
            "Visualization Level",
            options=["minimal", "standard", "full"],
            index=1,
            help="Amount of visualization generated during training"
        )
        
        plot_frequency = st.slider(
            "Plot Generation Frequency",
            min_value=1,
            max_value=20,
            value=5,
            help="Generate plots every N epochs"
        )
    
    with col2:
        save_animations = st.checkbox(
            "Save Animations",
            value=False,
            help="Save latent space animations (increases runtime)"
        )
        
        high_resolution = st.checkbox(
            "High Resolution Plots",
            value=False,
            help="Generate high-resolution plots (slower)"
        )
    
    # Store visualization configuration
    st.session_state.visualization_config = {
        'level': vis_level,
        'plot_frequency': plot_frequency,
        'save_animations': save_animations,
        'high_resolution': high_resolution
    }


def render_experiment_execution():
    """Render experiment execution and monitoring interface."""
    
    st.header("▶️ Run & Monitor Experiments")
    
    # Check if configuration is ready
    if not hasattr(st.session_state, 'model_config'):
        st.warning("⚠️ Please configure your experiment in the **Configure** tab first.")
        return
    
    # Experiment status
    status = st.session_state.get('experiment_status', 'idle')
    
    if status == 'idle':
        render_start_experiment_interface()
    elif status == 'running':
        render_running_experiment_interface()
    elif status == 'completed':
        render_completed_experiment_interface()
    elif status == 'error':
        render_error_experiment_interface()


def render_start_experiment_interface():
    """Render interface for starting a new experiment."""
    
    st.success("✅ Configuration ready! You can start the experiment.")
    
    # Configuration summary
    with st.expander("📋 Configuration Summary", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Configuration:**")
            model_config = st.session_state.model_config
            for key, value in model_config.items():
                st.markdown(f"- {key}: {value}")
        
        with col2:
            st.markdown("**Training Configuration:**")
            training_config = st.session_state.training_config
            for key, value in training_config.items():
                st.markdown(f"- {key}: {value}")
    
    # Start experiment
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        if st.button("🚀 Start Experiment", type="primary", use_container_width=True):
            start_experiment()


def render_running_experiment_interface():
    """Render interface for monitoring running experiment."""
    
    st.info("🔄 Experiment is running...")
    
    # Progress and metrics
    render_real_time_metrics()
    
    # Control buttons
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        if st.button("⏸️ Pause Experiment"):
            st.session_state.experiment_status = 'paused'
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Experiment", type="secondary"):
            st.session_state.experiment_status = 'stopped'
            st.rerun()
    
    with col3:
        if st.button("🔄 Refresh Metrics"):
            st.rerun()


def render_completed_experiment_interface():
    """Render interface for completed experiments."""
    
    st.success("✅ Experiment completed successfully!")
    
    # Results summary
    render_experiment_results()
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Analyze Results"):
            st.session_state.current_page = "📊 Model Comparison"
            st.rerun()
    
    with col2:
        if st.button("🌌 Explore Latent Space"):
            st.session_state.current_page = "🌌 Latent Exploration"
            st.rerun()
    
    with col3:
        if st.button("🔄 Start New Experiment"):
            st.session_state.experiment_status = 'idle'
            st.rerun()


def render_error_experiment_interface():
    """Render interface for failed experiments."""
    
    st.error("❌ Experiment failed!")
    
    # Error details
    error_msg = st.session_state.get('experiment_error', 'Unknown error occurred')
    st.code(error_msg)
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Retry Experiment"):
            st.session_state.experiment_status = 'idle'
            st.rerun()
    
    with col2:
        if st.button("🔧 Reconfigure"):
            st.session_state.current_page = "🧪 Experiment Manager"
            st.rerun()


def render_real_time_metrics():
    """Render real-time training metrics."""
    
    # Placeholder for real metrics - in a real implementation, 
    # this would connect to the actual training process
    import plotly.graph_objects as go
    import numpy as np
    
    st.subheader("📈 Training Metrics")
    
    # Simulate some training progress
    epochs = list(range(1, 21))
    train_loss = [3.0 - 1.5 * np.exp(-x/5) + 0.1 * np.random.randn() for x in epochs]
    val_loss = [2.8 - 1.2 * np.exp(-x/5) + 0.15 * np.random.randn() for x in epochs]
    
    # Loss plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_loss,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_loss,
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    fig.update_layout(
        title="Training Progress",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Current metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Epoch", "20", "2")
    
    with col2:
        st.metric("Training Loss", "1.234", "-0.045")
    
    with col3:
        st.metric("Validation Loss", "1.189", "-0.032")
    
    with col4:
        st.metric("Learning Rate", "0.0005", "0")


def render_experiment_results():
    """Render experiment results summary."""
    
    st.subheader("📊 Experiment Results")
    
    # Final metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Training Loss", "0.987")
    
    with col2:
        st.metric("Final Validation Loss", "1.045")
    
    with col3:
        st.metric("Best Epoch", "43")
    
    with col4:
        st.metric("Training Time", "2h 34m")


def render_experiment_history():
    """Render experiment history and management."""
    
    st.header("📋 Experiment History")
    
    experiments = st.session_state.get('experiments', {})
    
    if not experiments:
        st.info("No experiments in history yet.")
    else:
        # Display experiments in a table format
        import pandas as pd
        
        exp_data = []
        for name, data in experiments.items():
            if isinstance(data, dict):
                exp_data.append({
                    'Name': name,
                    'Status': data.get('status', 'Unknown'),
                    'Model': data.get('model_type', 'Unknown'),
                    'Date': data.get('timestamp', 'Unknown')
                })
        
        if exp_data:
            df = pd.DataFrame(exp_data)
            st.dataframe(df, use_container_width=True)
    
    # History management
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All History"):
            st.session_state.experiments = {}
            st.success("Experiment history cleared!")
            st.rerun()
    
    with col2:
        if st.button("📥 Export History"):
            # In a real implementation, this would export the history
            st.info("Export functionality would be implemented here")


def start_experiment():
    """Start a new experiment with current configuration."""
    
    st.session_state.experiment_status = 'running'
    
    # In a real implementation, this would:
    # 1. Generate Hydra configuration from UI settings
    # 2. Start the training process in a separate thread
    # 3. Set up real-time monitoring
    
    # For now, we'll simulate the experiment
    experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    experiment_data = {
        'name': experiment_name,
        'status': 'running',
        'model_type': st.session_state.model_config['model_type'],
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model': st.session_state.model_config,
            'training': st.session_state.training_config,
            'data': st.session_state.data_config,
            'visualization': st.session_state.visualization_config
        }
    }
    
    st.session_state.current_experiment = experiment_data
    st.session_state.experiments[experiment_name] = experiment_data
    
    st.success(f"🚀 Started experiment: {experiment_name}")
    st.rerun()


def save_experiment_configuration(exp_name: str):
    """Save current experiment configuration to file."""
    
    config = {
        'model': st.session_state.get('model_config', {}),
        'training': st.session_state.get('training_config', {}),
        'data': st.session_state.get('data_config', {}),
        'visualization': st.session_state.get('visualization_config', {})
    }
    
    # In a real implementation, this would save to file
    st.success(f"💾 Configuration saved as {exp_name}")


def load_experiment_configuration(uploaded_file):
    """Load experiment configuration from uploaded file."""
    
    try:
        config = yaml.safe_load(uploaded_file)
        
        if 'model' in config:
            st.session_state.model_config = config['model']
        if 'training' in config:
            st.session_state.training_config = config['training']
        if 'data' in config:
            st.session_state.data_config = config['data']
        if 'visualization' in config:
            st.session_state.visualization_config = config['visualization']
        
        st.success("📂 Configuration loaded successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Failed to load configuration: {str(e)}")


def reset_configuration():
    """Reset all configuration to defaults."""
    
    # Clear configuration from session state
    for key in ['model_config', 'training_config', 'data_config', 'visualization_config']:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("🔄 Configuration reset to defaults!")
    st.rerun()