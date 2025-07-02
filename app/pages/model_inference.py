"""
Model Inference Page
==================

Interface for loading trained VAE models and performing inference operations.
"""

import streamlit as st
import torch
import numpy as np
from pathlib import Path
import sys
from typing import Optional, Dict, Any
import io
from PIL import Image

# Add src to path
current_dir = Path(__file__).parent.parent.parent.absolute()
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def render():
    """Render the model inference page."""
    
    st.title("🔮 Model Inference")
    st.markdown("Load trained VAE models and perform encoding/decoding operations")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📂 Load Model", "🔍 Encode", "🎨 Decode"])
    
    with tab1:
        render_model_loading()
    
    with tab2:
        render_encoding_interface()
    
    with tab3:
        render_decoding_interface()


def render_model_loading():
    """Render model loading interface."""
    
    st.header("📂 Load Trained Model")
    
    # Model loading options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Load from Checkpoint")
        
        # File uploader for checkpoint
        uploaded_checkpoint = st.file_uploader(
            "Upload Model Checkpoint",
            type=['pth', 'pt', 'ckpt'],
            help="Upload a PyTorch model checkpoint file"
        )
        
        if uploaded_checkpoint:
            if st.button("📥 Load Checkpoint", type="primary"):
                load_model_from_checkpoint(uploaded_checkpoint)
    
    with col2:
        st.subheader("📋 Available Models")
        
        # List available checkpoints
        checkpoint_dir = Path("outputs")
        available_checkpoints = []
        
        if checkpoint_dir.exists():
            for path in checkpoint_dir.rglob("*.pth"):
                available_checkpoints.append(str(path.relative_to(checkpoint_dir)))
        
        if available_checkpoints:
            selected_checkpoint = st.selectbox(
                "Select Checkpoint",
                options=available_checkpoints,
                help="Choose from available model checkpoints"
            )
            
            if st.button("📥 Load Selected", type="secondary"):
                load_model_from_path(checkpoint_dir / selected_checkpoint)
        else:
            st.info("No checkpoints found in outputs directory")
    
    # Current model status
    render_current_model_status()
    
    # Model configuration
    if st.session_state.current_model is not None:
        with st.expander("⚙️ Model Configuration", expanded=False):
            render_model_configuration_display()


def render_current_model_status():
    """Display current loaded model status."""
    
    st.subheader("🎯 Current Model")
    
    if st.session_state.current_model is None:
        st.warning("⚠️ No model currently loaded")
        return
    
    model = st.session_state.current_model
    
    # Model info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if hasattr(model, 'get_model_summary'):
            summary = model.get_model_summary()
            model_name = summary.get('model_name', 'Unknown')
            st.metric("Model Type", model_name)
        else:
            st.metric("Model Type", type(model).__name__)
    
    with col2:
        if hasattr(model, 'latent_dim'):
            st.metric("Latent Dimension", model.latent_dim)
        else:
            st.metric("Latent Dimension", "Unknown")
    
    with col3:
        if hasattr(model, 'input_dim'):
            input_shape = str(model.input_dim)
            st.metric("Input Shape", input_shape)
        else:
            st.metric("Input Shape", "Unknown")
    
    # Model actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌌 Explore Latent Space"):
            st.session_state.current_page = "🌌 Latent Exploration"
            st.rerun()
    
    with col2:
        if st.button("📊 Analyze Model"):
            st.session_state.current_page = "📊 Model Comparison"
            st.rerun()
    
    with col3:
        if st.button("🗑️ Unload Model", type="secondary"):
            st.session_state.current_model = None
            st.session_state.model_config = None
            st.success("Model unloaded")
            st.rerun()


def render_model_configuration_display():
    """Display detailed model configuration."""
    
    if st.session_state.current_model is None:
        return
    
    model = st.session_state.current_model
    
    if hasattr(model, 'get_model_summary'):
        summary = model.get_model_summary()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Architecture:**")
            arch = summary.get('architecture', {})
            for key, value in arch.items():
                st.markdown(f"- {key}: {value}")
        
        with col2:
            st.markdown("**Configuration:**")
            config = summary.get('configuration', {})
            for key, value in config.items():
                st.markdown(f"- {key}: {value}")
        
        if 'hyperparameters' in summary:
            st.markdown("**Hyperparameters:**")
            hyperparams = summary['hyperparameters']
            for key, value in hyperparams.items():
                st.markdown(f"- {key}: {value}")


def render_encoding_interface():
    """Render encoding interface for input data."""
    
    st.header("🔍 Encode Input Data")
    
    if st.session_state.current_model is None:
        st.warning("⚠️ Please load a model first in the **Load Model** tab")
        return
    
    # Input options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Upload Image")
        
        uploaded_image = st.file_uploader(
            "Upload Image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to encode"
        )
        
        if uploaded_image:
            # Display uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Encode Image", type="primary"):
                encode_uploaded_image(image)
    
    with col2:
        st.subheader("🎲 Random Sample")
        
        if hasattr(st.session_state.current_model, 'input_dim'):
            input_dim = st.session_state.current_model.input_dim
            
            if st.button("🎲 Generate Random Input", type="secondary"):
                generate_and_encode_random_input(input_dim)
        
        st.subheader("🗂️ Sample from Dataset")
        
        dataset_option = st.selectbox(
            "Dataset",
            options=["cyclic_sprites", "mnist", "cifar10"],
            help="Choose dataset to sample from"
        )
        
        if st.button("📊 Sample from Dataset"):
            sample_and_encode_from_dataset(dataset_option)
    
    # Encoding results
    render_encoding_results()


def render_decoding_interface():
    """Render decoding interface for latent vectors."""
    
    st.header("🎨 Decode Latent Vectors")
    
    if st.session_state.current_model is None:
        st.warning("⚠️ Please load a model first in the **Load Model** tab")
        return
    
    model = st.session_state.current_model
    latent_dim = getattr(model, 'latent_dim', 16)
    
    # Latent vector input options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎛️ Manual Latent Input")
        
        # Create sliders for each latent dimension
        latent_vector = []
        
        # Show first 8 dimensions as sliders
        num_sliders = min(latent_dim, 8)
        
        for i in range(num_sliders):
            value = st.slider(
                f"Latent Dim {i+1}",
                min_value=-3.0,
                max_value=3.0,
                value=0.0,
                step=0.1,
                key=f"latent_slider_{i}"
            )
            latent_vector.append(value)
        
        # Fill remaining dimensions with zeros
        while len(latent_vector) < latent_dim:
            latent_vector.append(0.0)
        
        if latent_dim > 8:
            st.info(f"Showing first 8 of {latent_dim} latent dimensions. Others set to 0.")
        
        if st.button("🎨 Decode Manual Input", type="primary"):
            decode_latent_vector(torch.tensor(latent_vector).unsqueeze(0))
    
    with col2:
        st.subheader("🎲 Random Latent Sampling")
        
        sampling_method = st.selectbox(
            "Sampling Method",
            options=["standard_normal", "uniform", "spherical"],
            help="Method for generating random latent vectors"
        )
        
        num_samples = st.slider(
            "Number of Samples",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of random samples to generate"
        )
        
        if st.button("🎲 Generate Random Samples", type="secondary"):
            generate_and_decode_random_latents(latent_dim, num_samples, sampling_method)
        
        st.subheader("🔄 Use Encoded Latents")
        
        if 'encoded_latents' in st.session_state and st.session_state.encoded_latents is not None:
            if st.button("🔄 Decode Last Encoded"):
                decode_latent_vector(st.session_state.encoded_latents)
        else:
            st.info("No encoded latents available. Encode an image first.")
    
    # Decoding results
    render_decoding_results()


def render_encoding_results():
    """Display encoding results."""
    
    if 'encoded_latents' not in st.session_state or st.session_state.encoded_latents is None:
        return
    
    st.subheader("📊 Encoding Results")
    
    latents = st.session_state.encoded_latents
    
    # Latent statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Latent Norm", f"{torch.norm(latents).item():.3f}")
    
    with col2:
        st.metric("Mean Value", f"{torch.mean(latents).item():.3f}")
    
    with col3:
        st.metric("Std Value", f"{torch.std(latents).item():.3f}")
    
    # Latent vector visualization
    import plotly.graph_objects as go
    
    latent_np = latents.squeeze().detach().cpu().numpy()
    
    fig = go.Figure(data=go.Bar(
        x=list(range(len(latent_np))),
        y=latent_np,
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title="Latent Vector Components",
        xaxis_title="Latent Dimension",
        yaxis_title="Value",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Raw latent values
    with st.expander("🔢 Raw Latent Values", expanded=False):
        st.code(str(latent_np))


def render_decoding_results():
    """Display decoding results."""
    
    if 'decoded_images' not in st.session_state or not st.session_state.decoded_images:
        return
    
    st.subheader("🎨 Decoding Results")
    
    images = st.session_state.decoded_images
    
    # Display decoded images
    if len(images) == 1:
        st.image(images[0], caption="Decoded Image", use_container_width=True)
    else:
        # Multiple images in columns
        cols = st.columns(min(len(images), 5))
        for i, img in enumerate(images):
            with cols[i % len(cols)]:
                st.image(img, caption=f"Sample {i+1}", use_container_width=True)


def load_model_from_checkpoint(uploaded_file):
    """Load model from uploaded checkpoint file."""
    
    try:
        # Save uploaded file temporarily
        temp_path = Path("temp_checkpoint.pth")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Load checkpoint
        checkpoint = torch.load(temp_path, map_location='cpu')
        
        # Extract model from checkpoint (implementation depends on checkpoint format)
        if 'model' in checkpoint:
            model = checkpoint['model']
        elif 'state_dict' in checkpoint:
            # Would need to instantiate model and load state dict
            st.error("State dict loading not implemented yet. Please use full model checkpoints.")
            temp_path.unlink()
            return
        else:
            st.error("Unrecognized checkpoint format")
            temp_path.unlink()
            return
        
        # Store model in session state
        st.session_state.current_model = model
        
        # Clean up
        temp_path.unlink()
        
        st.success("✅ Model loaded successfully from checkpoint!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")


def load_model_from_path(checkpoint_path: Path):
    """Load model from checkpoint path."""
    
    try:
        if not checkpoint_path.exists():
            st.error(f"Checkpoint file not found: {checkpoint_path}")
            return
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract model (implementation depends on format)
        if 'model' in checkpoint:
            model = checkpoint['model']
        else:
            st.error("Model extraction not implemented for this checkpoint format")
            return
        
        # Store model
        st.session_state.current_model = model
        
        st.success(f"✅ Model loaded from {checkpoint_path.name}")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")


def encode_uploaded_image(image: Image.Image):
    """Encode uploaded image using current model."""
    
    try:
        model = st.session_state.current_model
        
        # Preprocess image (implementation depends on model requirements)
        tensor = preprocess_image_for_model(image, model)
        
        # Encode
        with torch.no_grad():
            if hasattr(model, 'encode'):
                latents = model.encode(tensor)
            elif hasattr(model, 'encoder'):
                encoder_output = model.encoder(tensor)
                latents = encoder_output.embedding if hasattr(encoder_output, 'embedding') else encoder_output
            else:
                st.error("Model doesn't have recognized encoding method")
                return
        
        # Store results
        st.session_state.encoded_latents = latents
        
        st.success("✅ Image encoded successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Encoding failed: {str(e)}")


def generate_and_encode_random_input(input_dim):
    """Generate random input and encode it."""
    
    try:
        # Generate random input tensor
        if len(input_dim) == 3:  # Image input
            random_input = torch.randn(1, *input_dim)
        else:
            random_input = torch.randn(1, *input_dim)
        
        model = st.session_state.current_model
        
        # Encode
        with torch.no_grad():
            if hasattr(model, 'encode'):
                latents = model.encode(random_input)
            elif hasattr(model, 'encoder'):
                encoder_output = model.encoder(random_input)
                latents = encoder_output.embedding if hasattr(encoder_output, 'embedding') else encoder_output
            else:
                st.error("Model doesn't have recognized encoding method")
                return
        
        # Store results
        st.session_state.encoded_latents = latents
        
        st.success("✅ Random input generated and encoded!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Failed to generate and encode: {str(e)}")


def sample_and_encode_from_dataset(dataset_name: str):
    """Sample from dataset and encode."""
    
    # This would require implementing dataset loading
    st.info("Dataset sampling not yet implemented. Use image upload or random generation.")


def decode_latent_vector(latent_vector: torch.Tensor):
    """Decode latent vector to image."""
    
    try:
        model = st.session_state.current_model
        
        # Decode
        with torch.no_grad():
            if hasattr(model, 'decode'):
                decoded = model.decode(latent_vector)
            elif hasattr(model, 'decoder'):
                decoder_output = model.decoder(latent_vector)
                decoded = decoder_output["reconstruction"] if isinstance(decoder_output, dict) else decoder_output
            else:
                st.error("Model doesn't have recognized decoding method")
                return
        
        # Convert to images
        images = tensor_to_images(decoded)
        
        # Store results
        st.session_state.decoded_images = images
        
        st.success("✅ Latent vector decoded successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Decoding failed: {str(e)}")


def generate_and_decode_random_latents(latent_dim: int, num_samples: int, method: str):
    """Generate and decode random latent vectors."""
    
    try:
        # Generate random latents
        if method == "standard_normal":
            latents = torch.randn(num_samples, latent_dim)
        elif method == "uniform":
            latents = torch.rand(num_samples, latent_dim) * 2 - 1  # [-1, 1]
        elif method == "spherical":
            latents = torch.randn(num_samples, latent_dim)
            latents = latents / torch.norm(latents, dim=1, keepdim=True)
        else:
            latents = torch.randn(num_samples, latent_dim)
        
        # Decode
        decode_latent_vector(latents)
        
    except Exception as e:
        st.error(f"❌ Failed to generate and decode: {str(e)}")


def preprocess_image_for_model(image: Image.Image, model) -> torch.Tensor:
    """Preprocess image for model input."""
    
    # Get expected input dimensions
    if hasattr(model, 'input_dim'):
        target_size = model.input_dim[-2:]  # Height, width
        channels = model.input_dim[0]
    else:
        # Default assumptions
        target_size = (64, 64)
        channels = 3
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to RGB if needed
    if channels == 3 and image.mode != 'RGB':
        image = image.convert('RGB')
    elif channels == 1 and image.mode != 'L':
        image = image.convert('L')
    
    # Convert to tensor
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) if channels == 1 else transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return tensor


def tensor_to_images(tensor: torch.Tensor) -> list:
    """Convert tensor to list of PIL Images."""
    
    import torchvision.transforms as transforms
    
    # Denormalize (assuming [-1, 1] range)
    tensor = tensor * 0.5 + 0.5
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL Images
    to_pil = transforms.ToPILImage()
    images = []
    
    for i in range(tensor.size(0)):
        img_tensor = tensor[i]
        image = to_pil(img_tensor)
        images.append(image)
    
    return images