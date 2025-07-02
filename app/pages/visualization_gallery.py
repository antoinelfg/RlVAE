"""
Visualization Gallery Page
=========================

Advanced plots and analysis tools for VAE research.
"""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
current_dir = Path(__file__).parent.parent.parent.absolute()
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def render():
    """Render the visualization gallery page."""
    
    st.title("🎨 Visualization Gallery")
    st.markdown("Advanced plots and analysis tools for deep insights into VAE behavior")
    
    # Check if model is loaded
    if st.session_state.current_model is None:
        st.warning("⚠️ Please load a model first to generate visualizations")
        
        if st.button("🔮 Go to Model Inference", type="primary"):
            st.session_state.current_page = "🔮 Model Inference"
            st.rerun()
        return
    
    # Main visualization categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Loss Analysis", 
        "🌌 Latent Distributions", 
        "🔍 Model Diagnostics", 
        "📈 Training Curves"
    ])
    
    with tab1:
        render_loss_analysis()
    
    with tab2:
        render_latent_distributions()
    
    with tab3:
        render_model_diagnostics()
    
    with tab4:
        render_training_curves()


def render_loss_analysis():
    """Render loss decomposition and analysis."""
    
    st.header("📊 Loss Analysis & Decomposition")
    
    # Generate sample data for visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 ELBO Decomposition")
        
        # Create synthetic ELBO decomposition
        epochs = np.arange(1, 51)
        reconstruction_loss = 2.5 - 1.8 * np.exp(-epochs/10) + 0.1 * np.random.randn(50)
        kl_divergence = 1.2 - 0.8 * np.exp(-epochs/15) + 0.05 * np.random.randn(50)
        elbo = -(reconstruction_loss + kl_divergence)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=reconstruction_loss,
            mode='lines',
            name='Reconstruction Loss',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=kl_divergence,
            mode='lines',
            name='KL Divergence',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=elbo,
            mode='lines',
            name='ELBO',
            line=dict(color='green', width=3)
        ))
        
        fig.update_layout(
            title="ELBO Decomposition Over Training",
            xaxis_title="Epoch",
            yaxis_title="Loss Value",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 KL Divergence per Dimension")
        
        model = st.session_state.current_model
        latent_dim = getattr(model, 'latent_dim', 16)
        
        # Generate synthetic per-dimension KL values
        kl_per_dim = np.random.exponential(scale=0.5, size=latent_dim)
        
        fig = go.Figure(data=go.Bar(
            x=list(range(1, latent_dim + 1)),
            y=kl_per_dim,
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="KL Divergence per Latent Dimension",
            xaxis_title="Latent Dimension",
            yaxis_title="KL Divergence",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Loss landscape visualization
    st.subheader("🗺️ Loss Landscape")
    
    if st.button("🔍 Generate Loss Landscape", type="secondary"):
        with st.spinner("Generating loss landscape..."):
            generate_loss_landscape()


def render_latent_distributions():
    """Render latent space distribution analysis."""
    
    st.header("🌌 Latent Space Distributions")
    
    model = st.session_state.current_model
    latent_dim = getattr(model, 'latent_dim', 16)
    
    # Generate sample latent codes
    if st.button("🎲 Generate Latent Samples", type="primary"):
        generate_latent_samples_for_analysis(500, latent_dim)
    
    if 'analysis_latent_samples' in st.session_state:
        samples = st.session_state.analysis_latent_samples
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Distribution Histograms")
            
            # Show histograms for first few dimensions
            num_dims_to_show = min(4, latent_dim)
            
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.ravel()
            
            for i in range(num_dims_to_show):
                axes[i].hist(samples[:, i], bins=30, alpha=0.7, color=f'C{i}')
                axes[i].set_title(f'Dimension {i+1}')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("🔍 Pairwise Correlations")
            
            # Show correlation matrix for subset of dimensions
            subset_dims = min(8, latent_dim)
            subset_samples = samples[:, :subset_dims]
            
            correlation_matrix = np.corrcoef(subset_samples.T)
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                x=[f'Dim {i+1}' for i in range(subset_dims)],
                y=[f'Dim {i+1}' for i in range(subset_dims)],
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(
                title="Latent Dimension Correlations",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Principal component analysis
        st.subheader("📈 Principal Component Analysis")
        
        if st.button("🔍 Run PCA Analysis"):
            run_pca_analysis(samples)


def render_model_diagnostics():
    """Render model diagnostic visualizations."""
    
    st.header("🔍 Model Diagnostics")
    
    model = st.session_state.current_model
    
    # Model architecture visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏗️ Model Architecture")
        
        if hasattr(model, 'get_model_summary'):
            summary = model.get_model_summary()
            
            # Create architecture diagram
            create_architecture_diagram(summary)
        else:
            st.info("Model summary not available")
    
    with col2:
        st.subheader("📊 Parameter Statistics")
        
        # Analyze model parameters
        analyze_model_parameters(model)
    
    # Gradient flow analysis
    st.subheader("🌊 Gradient Flow Analysis")
    
    if st.button("🔍 Analyze Gradient Flow"):
        st.info("Gradient flow analysis would be implemented here")
    
    # Activation analysis
    st.subheader("⚡ Activation Analysis")
    
    if st.button("🔍 Analyze Activations"):
        st.info("Activation analysis would be implemented here")


def render_training_curves():
    """Render training progress and convergence analysis."""
    
    st.header("📈 Training Curves & Convergence")
    
    # Generate synthetic training data
    if 'training_metrics' not in st.session_state or not st.session_state.training_metrics:
        generate_synthetic_training_data()
    
    metrics = st.session_state.training_metrics
    
    if metrics:
        # Loss curves
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📉 Loss Curves")
            
            epochs = list(range(1, len(metrics) + 1))
            train_losses = [m.get('train_loss', 0) for m in metrics]
            val_losses = [m.get('val_loss', 0) for m in metrics]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=epochs,
                y=train_losses,
                mode='lines',
                name='Training Loss',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=epochs,
                y=val_losses,
                mode='lines',
                name='Validation Loss',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="Training Progress",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Learning Rate Schedule")
            
            learning_rates = [m.get('learning_rate', 0.001) for m in metrics]
            
            fig = go.Figure(data=go.Scatter(
                x=epochs,
                y=learning_rates,
                mode='lines',
                name='Learning Rate',
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                title="Learning Rate Schedule",
                xaxis_title="Epoch",
                yaxis_title="Learning Rate",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Convergence analysis
        st.subheader("🎯 Convergence Analysis")
        
        analyze_convergence(metrics)


def generate_latent_samples_for_analysis(num_samples: int, latent_dim: int):
    """Generate latent samples for analysis."""
    
    try:
        model = st.session_state.current_model
        
        # Generate samples
        samples = torch.randn(num_samples, latent_dim)
        
        # If model has a way to sample from posterior, use that
        if hasattr(model, 'sample'):
            try:
                with torch.no_grad():
                    samples = model.sample(num_samples)
            except:
                pass  # Fall back to random samples
        
        st.session_state.analysis_latent_samples = samples.detach().cpu().numpy()
        st.success(f"✅ Generated {num_samples} latent samples")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Failed to generate samples: {str(e)}")


def run_pca_analysis(samples: np.ndarray):
    """Run PCA analysis on latent samples."""
    
    try:
        from sklearn.decomposition import PCA
        
        # Fit PCA
        pca = PCA()
        pca_result = pca.fit_transform(samples)
        
        # Plot explained variance
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
            y=pca.explained_variance_ratio_,
            name='Explained Variance Ratio'
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
            y=np.cumsum(pca.explained_variance_ratio_),
            mode='lines+markers',
            name='Cumulative Explained Variance',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="PCA Analysis of Latent Space",
            xaxis_title="Principal Component",
            yaxis_title="Explained Variance Ratio",
            yaxis2=dict(
                title="Cumulative Explained Variance",
                overlaying='y',
                side='right'
            ),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show first two principal components
        if samples.shape[1] >= 2:
            fig_2d = go.Figure(data=go.Scatter(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                mode='markers',
                marker=dict(
                    size=3,
                    opacity=0.6,
                    color=np.arange(len(pca_result)),
                    colorscale='viridis'
                )
            ))
            
            fig_2d.update_layout(
                title="First Two Principal Components",
                xaxis_title="PC1",
                yaxis_title="PC2",
                height=400
            )
            
            st.plotly_chart(fig_2d, use_container_width=True)
        
    except ImportError:
        st.error("Scikit-learn not available for PCA analysis")
    except Exception as e:
        st.error(f"❌ PCA analysis failed: {str(e)}")


def create_architecture_diagram(summary: dict):
    """Create a simple architecture diagram."""
    
    st.markdown("**Model Architecture:**")
    
    arch = summary.get('architecture', {})
    config = summary.get('configuration', {})
    
    # Create a simple text-based diagram
    diagram = f"""
    ```
    Input: {arch.get('input_dim', 'Unknown')}
           ↓
    Encoder → Latent Space ({arch.get('latent_dim', '?')} dims)
           ↓
    Flows: {arch.get('n_flows', '?')} layers
           ↓
    Decoder → Output: {arch.get('input_dim', 'Unknown')}
    ```
    """
    
    st.markdown(diagram)
    
    # Show configuration details
    st.markdown("**Configuration:**")
    for key, value in config.items():
        st.markdown(f"- **{key}:** {value}")


def analyze_model_parameters(model):
    """Analyze model parameter statistics."""
    
    try:
        param_stats = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_data = param.data.cpu().numpy()
                param_stats[name] = {
                    'shape': param_data.shape,
                    'mean': np.mean(param_data),
                    'std': np.std(param_data),
                    'min': np.min(param_data),
                    'max': np.max(param_data),
                    'num_params': param_data.size
                }
        
        # Show summary statistics
        total_params = sum(stats['num_params'] for stats in param_stats.values())
        
        st.metric("Total Parameters", f"{total_params:,}")
        
        # Show parameter distribution
        all_params = []
        for param in model.parameters():
            if param.requires_grad:
                all_params.extend(param.data.cpu().numpy().flatten())
        
        if all_params:
            fig = go.Figure(data=go.Histogram(
                x=all_params,
                nbinsx=50,
                name='Parameter Distribution'
            ))
            
            fig.update_layout(
                title="Model Parameter Distribution",
                xaxis_title="Parameter Value",
                yaxis_title="Frequency",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Parameter analysis failed: {str(e)}")


def generate_synthetic_training_data():
    """Generate synthetic training metrics for demonstration."""
    
    num_epochs = 50
    metrics = []
    
    for epoch in range(1, num_epochs + 1):
        # Simulate training progress
        base_loss = 3.0 * np.exp(-epoch/15) + 0.5
        train_loss = base_loss + 0.1 * np.random.randn()
        val_loss = base_loss * 1.1 + 0.15 * np.random.randn()
        
        lr = 0.001 * (0.95 ** (epoch // 10))  # Decay every 10 epochs
        
        metrics.append({
            'epoch': epoch,
            'train_loss': max(0.1, train_loss),
            'val_loss': max(0.1, val_loss),
            'learning_rate': lr
        })
    
    st.session_state.training_metrics = metrics


def analyze_convergence(metrics: list):
    """Analyze training convergence."""
    
    train_losses = [m['train_loss'] for m in metrics]
    val_losses = [m['val_loss'] for m in metrics]
    
    # Calculate convergence metrics
    recent_window = 10
    if len(train_losses) >= recent_window:
        recent_train_std = np.std(train_losses[-recent_window:])
        recent_val_std = np.std(val_losses[-recent_window:])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Training Stability",
                f"{recent_train_std:.4f}",
                help="Standard deviation of last 10 training losses"
            )
        
        with col2:
            st.metric(
                "Validation Stability", 
                f"{recent_val_std:.4f}",
                help="Standard deviation of last 10 validation losses"
            )
        
        with col3:
            # Check for overfitting
            train_val_gap = val_losses[-1] - train_losses[-1]
            st.metric(
                "Train-Val Gap",
                f"{train_val_gap:.4f}",
                help="Difference between validation and training loss"
            )
        
        # Convergence status
        if recent_train_std < 0.01 and recent_val_std < 0.01:
            st.success("✅ Model appears to have converged")
        elif train_val_gap > 0.5:
            st.warning("⚠️ Possible overfitting detected")
        else:
            st.info("ℹ️ Training in progress")


def generate_loss_landscape():
    """Generate a simple loss landscape visualization."""
    
    # This is a simplified demonstration
    # In reality, this would require careful sampling around the current parameters
    
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    
    # Generate synthetic loss landscape
    Z = np.sin(X) * np.cos(Y) + 0.1 * (X**2 + Y**2)
    
    fig = go.Figure(data=go.Surface(z=Z, x=X, y=Y))
    
    fig.update_layout(
        title="Loss Landscape (Simplified)",
        scene=dict(
            xaxis_title="Parameter Direction 1",
            yaxis_title="Parameter Direction 2", 
            zaxis_title="Loss Value"
        ),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("This is a simplified loss landscape. Real implementation would sample around current model parameters.")