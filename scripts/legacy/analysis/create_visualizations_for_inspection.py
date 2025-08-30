#!/usr/bin/env python3
"""
Create Visualizations for Inspection
===================================

Generate and save reconstruction and interpolation visualizations locally
for better inspection of the VAE quality.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings("ignore")

# Add the benchmark_VAE to the path
sys.path.append('benchmark_VAE/src')

from pythae.models import VAE, VAEConfig
import wandb

def create_rgb_encoder_decoder(input_dim, latent_dim):
    """Create encoder and decoder for RGB data."""
    import torch.nn as nn
    from pythae.models.nn import BaseEncoder, BaseDecoder
    
    class RGBEncoder(BaseEncoder):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            
            # Simple CNN for RGB
            self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)  # 64x64 -> 32x32
            self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)  # 32x32 -> 16x16
            self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)  # 16x16 -> 8x8
            self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)  # 8x8 -> 4x4
            
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(256 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, latent_dim)
            self.fc3 = nn.Linear(512, latent_dim)
            
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.relu(self.conv4(x))
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            
            mu = self.fc2(x)
            log_var = self.fc3(x)
            
            # Return in the format expected by pythae
            from pythae.models.base.base_utils import ModelOutput
            return ModelOutput(embedding=mu, log_covariance=log_var)
    
    class RGBDecoder(BaseDecoder):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            
            self.fc1 = nn.Linear(latent_dim, 512)
            self.fc2 = nn.Linear(512, 256 * 4 * 4)
            
            self.unflatten = nn.Unflatten(1, (256, 4, 4))
            self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 4x4 -> 8x8
            self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 8x8 -> 16x16
            self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)    # 16x16 -> 32x32
            self.deconv4 = nn.ConvTranspose2d(32, 3, 4, 2, 1)     # 32x32 -> 64x64
            
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, z):
            x = self.relu(self.fc1(z))
            x = self.relu(self.fc2(x))
            x = self.unflatten(x)
            x = self.relu(self.deconv1(x))
            x = self.relu(self.deconv2(x))
            x = self.relu(self.deconv3(x))
            x = self.sigmoid(self.deconv4(x))
            
            # Return in the format expected by pythae
            from pythae.models.base.base_utils import ModelOutput
            return ModelOutput(reconstruction=x)
    
    return RGBEncoder(input_dim, latent_dim), RGBDecoder(input_dim, latent_dim)

def load_trained_model():
    """Load the trained model from the latest run."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create VAE configuration
    model_config = VAEConfig(
        input_dim=(3, 64, 64),
        latent_dim=16,
        beta=1.0
    )
    
    # Create custom encoder and decoder for RGB
    encoder, decoder = create_rgb_encoder_decoder((3, 64, 64), 16)
    
    # Create VAE model
    model = VAE(
        model_config=model_config,
        encoder=encoder, 
        decoder=decoder
    )
    model.to(device)
    
    # Load the trained weights (you might need to adjust the path)
    try:
        # Try to load from the latest checkpoint
        checkpoint_path = "wandb/latest-run/files/model.pt"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("✅ Loaded trained model from checkpoint")
    except:
        print("⚠️ Could not load checkpoint, using untrained model")
    
    return model, device

def create_detailed_reconstruction_visualization(model, dataset, device):
    """Create detailed reconstruction visualization with better quality."""
    print(f"\n🎨 Creating detailed reconstruction visualization")
    
    model.eval()
    with torch.no_grad():
        # Sample more indices for better inspection
        indices = np.random.choice(len(dataset), 16, replace=False)
        original_images = []
        reconstructed_images = []
        
        for idx in indices:
            # Get original image
            original = dataset.data[idx].unsqueeze(0).to(device)
            original_images.append(original.cpu())
            
            # Get latent representation
            encoder_output = model.encoder(original)
            z = encoder_output.embedding
            
            # Reconstruct
            decoder_output = model.decoder(z)
            reconstructed = decoder_output.reconstruction
            reconstructed_images.append(reconstructed.cpu())
        
        # Create grid of original vs reconstructed
        original_grid = torch.cat(original_images, dim=0)
        recon_grid = torch.cat(reconstructed_images, dim=0)
        
        # Display as 2x16 grid (original on top, reconstructed on bottom)
        combined = torch.cat([original_grid, recon_grid], dim=0)
        
        # Reshape for display (2 rows, 16 columns)
        combined_display = combined.view(2, 16, 3, 64, 64).permute(0, 1, 3, 2, 4).contiguous()
        combined_display = combined_display.view(2 * 64, 16 * 64, 3)
        
        # Create figure with larger size
        fig, ax = plt.subplots(1, 1, figsize=(24, 10))
        ax.imshow(combined_display.cpu().numpy())
        ax.set_title("Original vs Reconstructed Sprites Samples (16 samples)", fontsize=16, fontweight='bold')
        ax.axis('off')
        ax.text(0.02, 0.98, 'Original', transform=ax.transAxes, fontsize=14, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax.text(0.02, 0.48, 'Reconstructed', transform=ax.transAxes, fontsize=14, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save locally
        plt.savefig("detailed_reconstruction_inspection.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Detailed reconstruction saved as 'detailed_reconstruction_inspection.png'")

def create_detailed_interpolation_visualization(model, centroids, device):
    """Create detailed interpolation visualization with better quality."""
    print(f"\n🎨 Creating detailed interpolation visualization")
    
    with torch.no_grad():
        # Pick multiple different centroids for interpolation
        centroid_pairs = [
            (0, -1),  # First and last
            (10, 40),  # Middle pairs
            (5, 25),
            (15, 35),
            (2, 30),  # Additional pairs
            (8, 45)
        ]
        
        all_interpolations = []
        
        for start_idx, end_idx in centroid_pairs:
            z1 = centroids[start_idx].unsqueeze(0).to(device)
            z2 = centroids[end_idx].unsqueeze(0).to(device)
            
            # Create more interpolation points
            n_interp = 12
            alphas = torch.linspace(0, 1, n_interp).to(device)
            interpolated_images = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                decoder_output = model.decoder(z_interp)
                interpolated = decoder_output.reconstruction
                interpolated_images.append(interpolated.cpu())
            
            # Create row for this interpolation
            interp_row = torch.cat(interpolated_images, dim=0)
            all_interpolations.append(interp_row)
        
        # Stack all interpolations
        all_interp = torch.stack(all_interpolations)
        
        # Reshape for display (6 rows, 12 columns)
        interp_display = all_interp.view(6, 12, 3, 64, 64).permute(0, 1, 3, 2, 4).contiguous()
        interp_display = interp_display.view(6 * 64, 12 * 64, 3)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(24, 16))
        ax.imshow(interp_display.cpu().numpy())
        ax.set_title("Latent Space Interpolation Between Different Centroids (6 paths)", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Add labels for each row
        labels = ["Centroid 0 → 49", "Centroid 10 → 40", "Centroid 5 → 25", 
                 "Centroid 15 → 35", "Centroid 2 → 30", "Centroid 8 → 45"]
        for i, label in enumerate(labels):
            ax.text(0.02, 0.95 - i*0.15, label, transform=ax.transAxes, fontsize=12, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save locally
        plt.savefig("detailed_interpolation_inspection.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Detailed interpolation saved as 'detailed_interpolation_inspection.png'")

def analyze_latent_space_distribution(model, dataset, device):
    """Analyze the latent space distribution to understand the 1D manifold issue."""
    print(f"\n🔍 Analyzing latent space distribution")
    
    model.eval()
    with torch.no_grad():
        # Extract latent representations
        latent_data = []
        for i in range(0, len(dataset), 256):
            batch = dataset.data[i:i+256].to(device)
            output = model.encoder(batch)
            latent_data.append(output.embedding)
        
        latent_data = torch.cat(latent_data, dim=0)
        
        # Analyze the distribution
        print(f"   Latent data shape: {latent_data.shape}")
        print(f"   Latent data range: [{latent_data.min():.3f}, {latent_data.max():.3f}]")
        
        # Check variance along each dimension
        variances = torch.var(latent_data, dim=0)
        print(f"   Variance along each dimension:")
        for i, var in enumerate(variances):
            print(f"     Dim {i}: {var:.6f}")
        
        # Find the most important dimensions
        sorted_dims = torch.argsort(variances, descending=True)
        print(f"   Most important dimensions: {sorted_dims[:5].tolist()}")
        print(f"   Least important dimensions: {sorted_dims[-5:].tolist()}")
        
        # Create a scatter plot of the first two dimensions
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        scatter = ax.scatter(latent_data[:, 0].cpu(), latent_data[:, 1].cpu(), 
                           alpha=0.5, s=1)
        ax.set_xlabel("z₁ (first dimension)")
        ax.set_ylabel("z₂ (second dimension)")
        ax.set_title("Latent Space Distribution (First 2 Dimensions)")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("latent_space_distribution_analysis.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Latent space distribution analysis saved as 'latent_space_distribution_analysis.png'")
        
        return latent_data

def main():
    """Main function to create detailed visualizations."""
    print("🔍 Creating Detailed Visualizations for Inspection")
    print("=" * 60)
    
    # Load model and data
    model, device = load_trained_model()
    
    # Load Sprites data
    print("📂 Loading Sprites data...")
    sprites_data = torch.load('data/processed/Sprites_train_cyclic.pt', map_location=device)
    print(f"   Loaded Sprites: {sprites_data.shape}")
    
    # Resize from 28x28 to 64x64 if needed
    if sprites_data.shape[-1] == 28:
        import torch.nn.functional as F
        sprites_data = F.interpolate(sprites_data.view(-1, *sprites_data.shape[2:]), 
                                   size=(64, 64), mode='bilinear', align_corners=False)
        sprites_data = sprites_data.view(sprites_data.shape[0], -1, *sprites_data.shape[1:])
        print(f"   Resized to: {sprites_data.shape}")
    
    # Use subset for analysis
    sprites_subset = sprites_data[:24000//8]  # Account for 8 frames per sprite
    flattened = sprites_subset.view(sprites_subset.shape[0] * sprites_subset.shape[1], *sprites_subset.shape[2:])
    print(f"   Analysis data: {flattened.shape}")
    
    # Create dataset wrapper
    class SpritesDataset:
        def __init__(self, data):
            self.data = data
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = SpritesDataset(flattened)
    
    # Analyze latent space distribution
    latent_data = analyze_latent_space_distribution(model, dataset, device)
    
    # Create detailed visualizations
    create_detailed_reconstruction_visualization(model, dataset, device)
    create_detailed_interpolation_visualization(model, latent_data[:50], device)  # Use first 50 as centroids
    
    print(f"\n✅ Detailed visualizations created for inspection!")

if __name__ == "__main__":
    main() 