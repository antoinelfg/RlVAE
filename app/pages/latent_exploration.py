"""
Latent Space Exploration Page
===========================

Interactive visualization and exploration of VAE latent spaces.
"""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from typing import Optional, Tuple, List

# Add src to path
current_dir = Path(__file__).parent.parent.parent.absolute()
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def render():
    """Render the latent space exploration page."""
    
    st.title("🌌 Latent Space Exploration")
    st.markdown("Interactive visualization and exploration of VAE latent representations")
    
    # Check if model is loaded
    if st.session_state.current_model is None:
        st.warning("⚠️ Please load a model first in the **Model Inference** page")
        
        if st.button("🔮 Go to Model Inference", type="primary"):
            st.session_state.current_page = "🔮 Model Inference"
            st.rerun()
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Latent Grid", 
        "🔄 Interpolation", 
        "📊 Embeddings", 
        "🎛️ Manual Control"
    ])
    
    with tab1:
        render_latent_grid()
    
    with tab2:
        render_interpolation_interface()
    
    with tab3:
        render_embedding_analysis()
    
    with tab4:
        render_manual_latent_control()


def render_latent_grid():
    """Render latent space grid visualization."""
    
    st.header("🗺️ Latent Space Grid")
    
    model = st.session_state.current_model
    latent_dim = getattr(model, 'latent_dim', 16)
    
    if latent_dim == 2:
        render_2d_latent_grid()
    elif latent_dim > 2:
        render_nd_latent_grid(latent_dim)
    else:
        st.error("Latent dimension must be at least 2 for grid visualization")


def render_2d_latent_grid():
    """Render 2D latent space grid."""
    
    st.subheader("📍 2D Latent Space")
    
    # Grid parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        grid_size = st.slider(
            "Grid Size",
            min_value=5,
            max_value=15,
            value=st.session_state.visualization_settings.get('latent_grid_resolution', 10),
            help="Number of points along each axis"
        )
    
    with col2:
        z_range = st.slider(
            "Z Range",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            help="Range of latent values to explore"
        )
    
    with col3:
        if st.button("🔄 Generate Grid", type="primary"):
            generate_2d_grid(grid_size, z_range)
    
    # Display grid if available
    if 'latent_2d_grid' in st.session_state:
        display_2d_grid()


def render_nd_latent_grid(latent_dim: int):
    """Render N-dimensional latent space grid by fixing some dimensions."""
    
    st.subheader(f"📊 {latent_dim}D Latent Space (2D Slice)")
    
    # Dimension selection
    col1, col2 = st.columns(2)
    
    with col1:
        dim1 = st.selectbox(
            "X-axis Dimension",
            options=list(range(latent_dim)),
            index=0,
            help="Choose latent dimension for X-axis"
        )
    
    with col2:
        dim2 = st.selectbox(
            "Y-axis Dimension", 
            options=list(range(latent_dim)),
            index=min(1, latent_dim-1),
            help="Choose latent dimension for Y-axis"
        )
    
    # Fixed dimensions
    with st.expander("🔧 Fixed Dimensions", expanded=False):
        fixed_values = {}
        for i in range(latent_dim):
            if i not in [dim1, dim2]:
                value = st.slider(
                    f"Dimension {i}",
                    min_value=-3.0,
                    max_value=3.0,
                    value=0.0,
                    step=0.1,
                    key=f"fixed_dim_{i}"
                )
                fixed_values[i] = value
    
    # Grid parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        grid_size = st.slider(
            "Grid Size",
            min_value=5,
            max_value=15,
            value=st.session_state.visualization_settings.get('latent_grid_resolution', 10),
            help="Number of points along each axis"
        )
    
    with col2:
        z_range = st.slider(
            "Z Range",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            help="Range of latent values to explore"
        )
    
    with col3:
        if st.button("🔄 Generate ND Grid", type="primary"):
            generate_2d_grid(latent_dim, dim1, dim2, fixed_values, grid_size, z_range)
    
    # Display grid if available
    if 'latent_nd_grid' in st.session_state:
        display_2d_grid()


def render_interpolation_interface():
    """Render latent interpolation interface."""
    
    st.header("🔄 Latent Interpolation")
    
    # Interpolation options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Start Point")
        
        start_option = st.radio(
            "Start Point Source",
            options=["Random", "Manual", "Encoded"],
            help="Choose how to define the start point"
        )
        
        start_point = get_interpolation_point(start_option, "start")
    
    with col2:
        st.subheader("🏁 End Point")
        
        end_option = st.radio(
            "End Point Source",
            options=["Random", "Manual", "Encoded"],
            help="Choose how to define the end point"
        )
        
        end_point = get_interpolation_point(end_option, "end")
    
    # Interpolation parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_steps = st.slider(
            "Interpolation Steps",
            min_value=5,
            max_value=50,
            value=st.session_state.visualization_settings.get('interpolation_steps', 20),
            help="Number of interpolation steps"
        )
    
    with col2:
        interpolation_method = st.selectbox(
            "Interpolation Method",
            options=["linear", "spherical", "geodesic"],
            help="Method for interpolating between points"
        )
    
    with col3:
        if st.button("🔄 Generate Interpolation", type="primary"):
            if start_point is not None and end_point is not None:
                generate_interpolation(start_point, end_point, num_steps, interpolation_method)
            else:
                st.error("Please define both start and end points")
    
    # Display interpolation if available
    if 'interpolation_results' in st.session_state:
        display_interpolation_results()


def render_embedding_analysis():
    """Render embedding analysis with dimensionality reduction."""
    
    st.header("📊 Latent Space Embeddings")
    
    # Generate sample points
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎲 Sample Generation")
        
        num_samples = st.slider(
            "Number of Samples",
            min_value=50,
            max_value=1000,
            value=200,
            help="Number of latent samples to generate"
        )
        
        sampling_method = st.selectbox(
            "Sampling Method",
            options=["random_normal", "random_uniform", "spherical"],
            help="Method for generating samples"
        )
        
        if st.button("🎲 Generate Samples", type="primary"):
            generate_embedding_samples(num_samples, sampling_method)
    
    with col2:
        st.subheader("📈 Dimensionality Reduction")
        
        reduction_method = st.selectbox(
            "Reduction Method",
            options=["PCA", "UMAP", "t-SNE"],
            help="Method for reducing dimensionality"
        )
        
        if reduction_method == "UMAP":
            n_neighbors = st.slider(
                "UMAP Neighbors",
                min_value=5,
                max_value=50,
                value=st.session_state.visualization_settings.get('umap_neighbors', 15),
                help="Number of neighbors for UMAP"
            )
        
        if st.button("📈 Apply Reduction", type="secondary"):
            if 'embedding_samples' in st.session_state:
                apply_dimensionality_reduction(reduction_method)
            else:
                st.error("Please generate samples first")
    
    # Display embedding results
    if 'embedding_2d' in st.session_state:
        display_embedding_visualization()


def render_manual_latent_control():
    """Render manual latent space control with real-time updates."""
    
    st.header("🎛️ Manual Latent Control")
    
    model = st.session_state.current_model
    latent_dim = getattr(model, 'latent_dim', 16)
    
    st.markdown("Adjust latent dimensions and see real-time decoded outputs")
    
    # Create sliders for each dimension
    latent_values = []
    
    # Show sliders in columns
    num_cols = 4
    cols = st.columns(num_cols)
    
    for i in range(latent_dim):
        col_idx = i % num_cols
        with cols[col_idx]:
            value = st.slider(
                f"Dim {i+1}",
                min_value=-3.0,
                max_value=3.0,
                value=0.0,
                step=0.1,
                key=f"manual_latent_{i}"
            )
            latent_values.append(value)
    
    # Real-time decoding
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎨 Decode Current State", type="primary"):
            decode_manual_latent(latent_values)
    
    with col2:
        auto_update = st.checkbox(
            "Auto Update",
            value=False,
            help="Automatically decode when sliders change (may be slow)"
        )
    
    # Auto-update functionality
    if auto_update:
        decode_manual_latent(latent_values)
    
    # Display decoded result
    if 'manual_decoded' in st.session_state:
        st.subheader("🎨 Decoded Output")
        st.image(st.session_state.manual_decoded, caption="Real-time Decoded Image", use_container_width=True)


def generate_2d_grid(grid_size: int, z_range: float):
    """Generate 2D latent grid and decode."""
    
    model = st.session_state.current_model
    
    # Create grid
    x = np.linspace(-z_range, z_range, grid_size)
    y = np.linspace(-z_range, z_range, grid_size)
    xx, yy = np.meshgrid(x, y)
    
    # Flatten grid points
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    grid_tensor = torch.FloatTensor(grid_points)
    
    try:
        with torch.no_grad():
            # Decode grid points
            if hasattr(model, 'decode'):
                decoded = model.decode(grid_tensor)
            elif hasattr(model, 'decoder'):
                decoder_output = model.decoder(grid_tensor)
                decoded = decoder_output["reconstruction"] if isinstance(decoder_output, dict) else decoder_output
            else:
                st.error("Model doesn't have recognized decoding method")
                return
        
        # Store results
        st.session_state.latent_2d_grid = {
            'grid_points': grid_points,
            'decoded_images': decoded,
            'grid_size': grid_size,
            'x': x,
            'y': y
        }
        
        st.success(f"✅ Generated {grid_size}x{grid_size} grid successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Failed to generate grid: {str(e)}")


def display_2d_grid():
    """Display 2D latent grid results."""
    
    grid_data = st.session_state.latent_2d_grid
    
    st.subheader("🗺️ 2D Latent Grid Visualization")
    
    # Convert tensors to images
    from app.pages.model_inference import tensor_to_images
    
    try:
        decoded_tensor = grid_data['decoded_images']
        images = tensor_to_images(decoded_tensor)
        
        # Create grid layout
        grid_size = grid_data['grid_size']
        
        # Display as image grid
        for i in range(grid_size):
            cols = st.columns(grid_size)
            for j in range(grid_size):
                img_idx = i * grid_size + j
                if img_idx < len(images):
                    with cols[j]:
                        st.image(images[img_idx], use_container_width=True)
        
        # Interactive plot
        st.subheader("📍 Interactive Grid Navigation")
        
        # Create scatter plot of grid points
        x = grid_data['grid_points'][:, 0]
        y = grid_data['grid_points'][:, 1]
        
        fig = go.Figure(data=go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=8,
                color=np.arange(len(x)),
                colorscale='viridis',
                showscale=True
            ),
            text=[f"Point ({x[i]:.2f}, {y[i]:.2f})" for i in range(len(x))],
            hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Latent Space Grid Points",
            xaxis_title="Latent Dimension 1",
            yaxis_title="Latent Dimension 2",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Failed to display grid: {str(e)}")


def get_interpolation_point(option: str, point_type: str) -> Optional[torch.Tensor]:
    """Get interpolation point based on selected option."""
    
    model = st.session_state.current_model
    latent_dim = getattr(model, 'latent_dim', 16)
    
    if option == "Random":
        point = torch.randn(latent_dim)
        return point
    
    elif option == "Manual":
        st.markdown(f"**{point_type.capitalize()} Point Values:**")
        values = []
        
        # Show first 8 dimensions
        num_show = min(latent_dim, 8)
        cols = st.columns(min(num_show, 4))
        
        for i in range(num_show):
            col_idx = i % len(cols)
            with cols[col_idx]:
                value = st.number_input(
                    f"Dim {i+1}",
                    value=0.0,
                    key=f"interp_{point_type}_{i}"
                )
                values.append(value)
        
        # Fill remaining with zeros
        while len(values) < latent_dim:
            values.append(0.0)
        
        if latent_dim > 8:
            st.info(f"Showing first 8 of {latent_dim} dimensions. Others set to 0.")
        
        return torch.tensor(values)
    
    elif option == "Encoded":
        if 'encoded_latents' in st.session_state and st.session_state.encoded_latents is not None:
            return st.session_state.encoded_latents.squeeze()
        else:
            st.warning("No encoded latents available. Encode an image first.")
            return None
    
    return None


def generate_interpolation(start: torch.Tensor, end: torch.Tensor, steps: int, method: str):
    """Generate interpolation between two points."""
    
    try:
        if method == "linear":
            # Linear interpolation
            alphas = torch.linspace(0, 1, steps)
            interpolated = torch.stack([
                (1 - alpha) * start + alpha * end for alpha in alphas
            ])
        
        elif method == "spherical":
            # Spherical interpolation with numerical stability
            eps = 1e-8  # Small epsilon to prevent numerical issues
            
            # Check for zero norms and handle gracefully
            start_norm_val = torch.norm(start)
            end_norm_val = torch.norm(end)
            
            if start_norm_val < eps or end_norm_val < eps:
                st.warning("⚠️ One of the vectors has zero/near-zero norm. Falling back to linear interpolation.")
                # Fall back to linear interpolation
                alphas = torch.linspace(0, 1, steps)
                interpolated = torch.stack([
                    (1 - alpha) * start + alpha * end for alpha in alphas
                ])
            else:
                # Normalize vectors
                start_norm = start / start_norm_val
                end_norm = end / end_norm_val
                
                # Calculate dot product (works for both 1D and batched inputs)
                dot_product = torch.sum(start_norm * end_norm, dim=-1 if start_norm.dim() > 1 else 0)
                dot_product = torch.clamp(dot_product, -1 + eps, 1 - eps)  # Prevent numerical issues in acos
                
                theta = torch.acos(dot_product)
                sin_theta = torch.sin(theta)
                
                # Check if vectors are parallel/anti-parallel (theta ≈ 0 or π)
                if abs(sin_theta) < eps:
                    st.info("ℹ️ Vectors are parallel/anti-parallel. Using linear interpolation.")
                    # Fall back to linear interpolation
                    alphas = torch.linspace(0, 1, steps)
                    interpolated = torch.stack([
                        (1 - alpha) * start + alpha * end for alpha in alphas
                    ])
                else:
                    # Proper spherical interpolation
                    alphas = torch.linspace(0, 1, steps)
                    interpolated = torch.stack([
                        (torch.sin((1 - alpha) * theta) * start_norm + torch.sin(alpha * theta) * end_norm) / sin_theta
                        for alpha in alphas
                    ])
        
        elif method == "geodesic":
            # Simplified geodesic (would need Riemannian metric for true geodesic)
            st.info("Using linear interpolation as geodesic requires Riemannian metric")
            alphas = torch.linspace(0, 1, steps)
            interpolated = torch.stack([
                (1 - alpha) * start + alpha * end for alpha in alphas
            ])
        
        else:
            # Default to linear
            alphas = torch.linspace(0, 1, steps)
            interpolated = torch.stack([
                (1 - alpha) * start + alpha * end for alpha in alphas
            ])
        
        # Decode interpolated points
        model = st.session_state.current_model
        
        with torch.no_grad():
            if hasattr(model, 'decode'):
                decoded = model.decode(interpolated)
            elif hasattr(model, 'decoder'):
                decoder_output = model.decoder(interpolated)
                decoded = decoder_output["reconstruction"] if isinstance(decoder_output, dict) else decoder_output
            else:
                st.error("Model doesn't have recognized decoding method")
                return
        
        # Store results
        st.session_state.interpolation_results = {
            'latent_points': interpolated,
            'decoded_images': decoded,
            'start_point': start,
            'end_point': end,
            'method': method,
            'steps': steps
        }
        
        st.success(f"✅ Generated {steps}-step {method} interpolation!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Failed to generate interpolation: {str(e)}")


def display_interpolation_results():
    """Display interpolation results."""
    
    results = st.session_state.interpolation_results
    
    st.subheader("🔄 Interpolation Results")
    
    # Convert to images
    from app.pages.model_inference import tensor_to_images
    
    try:
        images = tensor_to_images(results['decoded_images'])
        
        # Display as image sequence
        st.markdown("**Image Sequence:**")
        
        # Create columns for images
        num_cols = min(len(images), 10)
        cols = st.columns(num_cols)
        
        for i, img in enumerate(images):
            col_idx = i % num_cols
            with cols[col_idx]:
                st.image(img, caption=f"Step {i+1}", use_container_width=True)
        
        # Latent space trajectory
        st.subheader("📈 Latent Space Trajectory")
        
        latent_points = results['latent_points'].detach().cpu().numpy()
        
        # Plot first two dimensions
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=latent_points[:, 0],
            y=latent_points[:, 1] if latent_points.shape[1] > 1 else np.zeros(len(latent_points)),
            mode='lines+markers',
            name='Interpolation Path',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ))
        
        # Mark start and end points
        fig.add_trace(go.Scatter(
            x=[latent_points[0, 0]],
            y=[latent_points[0, 1] if latent_points.shape[1] > 1 else 0],
            mode='markers',
            name='Start',
            marker=dict(size=12, color='green', symbol='star')
        ))
        
        fig.add_trace(go.Scatter(
            x=[latent_points[-1, 0]],
            y=[latent_points[-1, 1] if latent_points.shape[1] > 1 else 0],
            mode='markers',
            name='End',
            marker=dict(size=12, color='red', symbol='star')
        ))
        
        fig.update_layout(
            title=f"{results['method'].capitalize()} Interpolation Path",
            xaxis_title="Latent Dimension 1",
            yaxis_title="Latent Dimension 2" if latent_points.shape[1] > 1 else "Fixed at 0",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Interpolation"):
                st.info("Save functionality would be implemented here")
        
        with col2:
            if st.button("🎬 Create Animation"):
                st.info("Animation creation would be implemented here")
        
    except Exception as e:
        st.error(f"❌ Failed to display interpolation: {str(e)}")


def decode_manual_latent(latent_values: List[float]):
    """Decode manually specified latent values."""
    
    try:
        model = st.session_state.current_model
        latent_tensor = torch.tensor(latent_values).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            if hasattr(model, 'decode'):
                decoded = model.decode(latent_tensor)
            elif hasattr(model, 'decoder'):
                decoder_output = model.decoder(latent_tensor)
                decoded = decoder_output["reconstruction"] if isinstance(decoder_output, dict) else decoder_output
            else:
                st.error("Model doesn't have recognized decoding method")
                return
        
        # Convert to image
        from app.pages.model_inference import tensor_to_images
        images = tensor_to_images(decoded)
        
        if images:
            st.session_state.manual_decoded = images[0]
        
    except Exception as e:
        st.error(f"❌ Failed to decode: {str(e)}")


def generate_embedding_samples(num_samples: int, method: str):
    """Generate samples for embedding analysis."""
    
    model = st.session_state.current_model
    latent_dim = getattr(model, 'latent_dim', 16)
    
    try:
        if method == "random_normal":
            samples = torch.randn(num_samples, latent_dim)
        elif method == "random_uniform":
            samples = torch.rand(num_samples, latent_dim) * 4 - 2  # [-2, 2]
        elif method == "spherical":
            samples = torch.randn(num_samples, latent_dim)
            samples = samples / torch.norm(samples, dim=1, keepdim=True)
        else:
            samples = torch.randn(num_samples, latent_dim)
        
        st.session_state.embedding_samples = samples
        st.success(f"✅ Generated {num_samples} samples using {method}")
        
    except Exception as e:
        st.error(f"❌ Failed to generate samples: {str(e)}")


def apply_dimensionality_reduction(method: str):
    """Apply dimensionality reduction to embedding samples."""
    
    try:
        samples = st.session_state.embedding_samples.detach().cpu().numpy()
        
        if method == "PCA":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            embedded = reducer.fit_transform(samples)
            
        elif method == "UMAP":
            try:
                import umap
                n_neighbors = st.session_state.visualization_settings.get('umap_neighbors', 15)
                reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors)
                embedded = reducer.fit_transform(samples)
            except ImportError:
                st.error("UMAP not installed. Please install with: pip install umap-learn")
                return
                
        elif method == "t-SNE":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            embedded = reducer.fit_transform(samples)
        
        else:
            st.error(f"Unknown reduction method: {method}")
            return
        
        st.session_state.embedding_2d = embedded
        st.session_state.embedding_method = method
        st.success(f"✅ Applied {method} reduction successfully")
        
    except Exception as e:
        st.error(f"❌ Failed to apply {method}: {str(e)}")


def display_embedding_visualization():
    """Display embedding visualization."""
    
    embedded = st.session_state.embedding_2d
    method = st.session_state.embedding_method
    
    st.subheader(f"📊 {method} Embedding Visualization")
    
    # Create interactive scatter plot
    fig = go.Figure(data=go.Scatter(
        x=embedded[:, 0],
        y=embedded[:, 1],
        mode='markers',
        marker=dict(
            size=6,
            color=np.arange(len(embedded)),
            colorscale='viridis',
            showscale=True,
            colorbar=dict(title="Sample Index")
        ),
        text=[f"Sample {i}" for i in range(len(embedded))],
        hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"{method} Embedding of Latent Samples",
        xaxis_title=f"{method} Component 1",
        yaxis_title=f"{method} Component 2",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("X Range", f"{embedded[:, 0].max() - embedded[:, 0].min():.2f}")
    
    with col2:
        st.metric("Y Range", f"{embedded[:, 1].max() - embedded[:, 1].min():.2f}")
    
    with col3:
        st.metric("Num Samples", len(embedded))