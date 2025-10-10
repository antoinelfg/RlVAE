import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from sklearn.decomposition import PCA
import torch

def visualize_latent_space(latents, images, epoch, num_highlight=10):
    """
    Visualize the latent space with PCA and add hover functionality for selected images.
    
    Args:
        latents (np.ndarray): Latent representations of shape [num_samples, latent_dim]
        images (list): List of numpy arrays representing the images
        epoch (int): Current epoch number
        num_highlight (int): Number of points to highlight with hover images
    """
    # Flatten the latent representations to 2D
    latents = latents.reshape(latents.shape[0], -1)
    
    # Perform PCA on the latent representations
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)
    
    # Prepare images for hover
    images = [(img.squeeze(0) if img.shape[0] == 1 else img.mean(axis=0)) for img in images]  # Convert to grayscale if needed
    images = [(img - img.min()) / (img.max() - img.min()) for img in images]  # Normalize
    
    # Select random indices for highlighted points
    num_samples = len(latents_2d)
    if num_samples > num_highlight:
        highlight_indices = np.random.choice(num_samples, num_highlight, replace=False)
    else:
        highlight_indices = np.arange(num_samples)
    
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'PCA1': latents_2d[:, 0],
        'PCA2': latents_2d[:, 1],
        'epoch': [epoch] * len(latents_2d),
        'highlight': [i in highlight_indices for i in range(num_samples)],
        'image': [wandb.Image(img, caption=f"Sample {i}") for i, img in enumerate(images)]
    })
    
    # Create an interactive plotly scatter plot with hover data
    fig = go.Figure()
    
    # Add regular points
    fig.add_trace(go.Scatter(
        x=df[~df['highlight']]['PCA1'],
        y=df[~df['highlight']]['PCA2'],
        mode='markers',
        marker=dict(color='blue', size=8),
        hoverinfo='skip'
    ))
    
    # Add highlighted points with hover images
    fig.add_trace(go.Scatter(
        x=df[df['highlight']]['PCA1'],
        y=df[df['highlight']]['PCA2'],
        mode='markers',
        marker=dict(color='red', size=12),
        customdata=df[df['highlight']]['image'],
        hovertemplate="<br>".join([
            "PCA1: %{x}",
            "PCA2: %{y}",
            "<extra></extra>"
        ])
    ))
    
    # Update layout
    fig.update_layout(
        title=f'PCA of Latent Space (Epoch {epoch})',
        xaxis_title='PCA1',
        yaxis_title='PCA2',
        showlegend=False
    )
    
    return fig

# Example usage:
if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="latent_space_visualization")
    
    # Example data (replace with your actual data)
    num_samples = 100
    latent_dim = 32
    image_size = 64
    
    # Generate random latent vectors and images for demonstration
    latents = np.random.randn(num_samples, latent_dim)
    images = [np.random.rand(image_size, image_size) for _ in range(num_samples)]
    
    # Create visualization
    fig = visualize_latent_space(latents, images, epoch=0)
    
    # Log to wandb
    wandb.log({
        "Latent Space PCA": wandb.Plotly(fig),
        "epoch": 0
    })
    
    wandb.finish() 