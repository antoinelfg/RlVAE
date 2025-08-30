#!/usr/bin/env python3
"""
Train Simple Better VAE
=======================

Train a better VAE on CPU to avoid GPU memory issues.
Use simpler architecture but better training techniques.
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

def create_simple_better_encoder_decoder(input_dim, latent_dim):
    """Create simple but better encoder and decoder."""
    import torch.nn as nn
    from pythae.models.nn import BaseEncoder, BaseDecoder
    
    class SimpleBetterEncoder(BaseEncoder):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            
            # Simple but effective architecture
            self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)  # 64x64 -> 32x32
            self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)  # 32x32 -> 16x16
            self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)  # 16x16 -> 8x8
            
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(128 * 8 * 8, 512)
            self.fc2 = nn.Linear(512, latent_dim)
            self.fc3 = nn.Linear(512, latent_dim)
            
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
            
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            
            mu = self.fc2(x)
            log_var = self.fc3(x)
            
            # Return in the format expected by pythae
            from pythae.models.base.base_utils import ModelOutput
            return ModelOutput(embedding=mu, log_covariance=log_var)
    
    class SimpleBetterDecoder(BaseDecoder):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            
            self.fc1 = nn.Linear(latent_dim, 512)
            self.fc2 = nn.Linear(512, 128 * 8 * 8)
            
            self.unflatten = nn.Unflatten(1, (128, 8, 8))
            self.deconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # 8x8 -> 16x16
            self.deconv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)   # 16x16 -> 32x32
            self.deconv3 = nn.ConvTranspose2d(32, 3, 4, 2, 1)    # 32x32 -> 64x64
            
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, z):
            x = self.relu(self.fc1(z))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.unflatten(x)
            x = self.relu(self.deconv1(x))
            x = self.relu(self.deconv2(x))
            x = self.sigmoid(self.deconv3(x))
            
            # Return in the format expected by pythae
            from pythae.models.base.base_utils import ModelOutput
            return ModelOutput(reconstruction=x)
    
    return SimpleBetterEncoder(input_dim, latent_dim), SimpleBetterDecoder(input_dim, latent_dim)

def train_simple_better_vae():
    """Train a simple but better VAE on CPU."""
    print("🚀 Training Simple Better VAE on CPU")
    print("=" * 50)
    
    device = torch.device('cpu')  # Use CPU to avoid GPU memory issues
    
    # Initialize WandB
    wandb.init(
        project="rlvae_metric_analysis",
        name="simple_better_vae_cpu",
        tags=["simple_vae", "sprites", "cpu_training"],
        config={
            "max_epochs": 100,
            "batch_size": 64,
            "learning_rate": 1e-4,
            "latent_dim": 16,
            "beta": 0.01,  # Very low beta to prevent posterior collapse
            "max_train_samples": 24000
        }
    )
    
    # Load Sprites data
    print("📂 Loading Sprites data...")
    sprites_data = torch.load('data/processed/Sprites_train_cyclic.pt', map_location='cpu')
    print(f"   Loaded Sprites: {sprites_data.shape}")
    
    # Resize from 28x28 to 64x64
    if sprites_data.shape[-1] == 28:
        import torch.nn.functional as F
        sprites_data = F.interpolate(sprites_data.view(-1, *sprites_data.shape[2:]), 
                                   size=(64, 64), mode='bilinear', align_corners=False)
        sprites_data = sprites_data.view(sprites_data.shape[0], -1, *sprites_data.shape[1:])
        print(f"   Resized to: {sprites_data.shape}")
    
    # Use subset for training
    max_train_samples = 24000
    sprites_subset = sprites_data[:max_train_samples//8]  # Account for 8 frames per sprite
    flattened = sprites_subset.view(sprites_subset.shape[0] * sprites_subset.shape[1], *sprites_subset.shape[2:])
    print(f"   Training data: {flattened.shape}")
    
    # Create VAE configuration with very low beta
    model_config = VAEConfig(
        input_dim=(3, 64, 64),
        latent_dim=16,
        beta=0.01  # Very low beta to prevent posterior collapse
    )
    
    # Create simple but better encoder and decoder
    encoder, decoder = create_simple_better_encoder_decoder((3, 64, 64), 16)
    
    # Create VAE model
    model = VAE(
        model_config=model_config,
        encoder=encoder, 
        decoder=decoder
    )
    model.to(device)
    
    print(f"✅ Simple Better VAE model created")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset wrapper
    class SpritesDataset:
        def __init__(self, data):
            self.data = data
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = SpritesDataset(flattened)
    
    # Training loop
    max_epochs = 100
    batch_size = 64
    learning_rate = 1e-4
    
    print(f"\n🎓 Training for {max_epochs} epochs on CPU...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    model.train()
    
    for epoch in range(max_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        n_batches = 0
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset.data[i:i+batch_size]
            if len(batch) < 2:
                continue
                
            optimizer.zero_grad()
            
            # Forward pass
            inputs = {"data": batch}
            output = model(inputs)
            
            # Compute losses
            recon_loss = output.recon_loss
            kl_loss = output.reg_loss  # KL loss is called reg_loss in pythae
            total_loss_batch = output.loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            n_batches += 1
        
        # Update learning rate
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        scheduler.step(avg_loss)
        
        # Log metrics
        avg_recon_loss = total_recon_loss / n_batches if n_batches > 0 else 0
        avg_kl_loss = total_kl_loss / n_batches if n_batches > 0 else 0
        
        wandb.log({
            "epoch": epoch,
            "train/loss": avg_loss,
            "train/recon_loss": avg_recon_loss,
            "train/kl_loss": avg_kl_loss,
            "train/learning_rate": optimizer.param_groups[0]['lr']
        })
        
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1}/{max_epochs}, Loss: {avg_loss:.6f}, Recon: {avg_recon_loss:.6f}, KL: {avg_kl_loss:.6f}")
    
    print(f"✅ Simple better training completed")
    
    # Save the trained model
    torch.save(model.state_dict(), "simple_better_vae.pt")
    print("✅ Simple better model saved as 'simple_better_vae.pt'")
    
    return model, dataset, device

def analyze_simple_better_latent_space(model, dataset, device):
    """Analyze the latent space distribution of the simple better trained model."""
    print(f"\n🔍 Analyzing Simple Better Latent Space Distribution")
    
    model.eval()
    with torch.no_grad():
        # Extract latent representations
        latent_data = []
        for i in range(0, len(dataset), 256):
            batch = dataset.data[i:i+256]
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
        
        # Check if distribution is more structured (less Gaussian)
        from scipy import stats
        for i in range(min(5, len(variances))):
            dim_data = latent_data[:, i].numpy()
            # Test for normality
            _, p_value = stats.normaltest(dim_data)
            print(f"     Dim {i} normality test p-value: {p_value:.6f} ({'NOT normal' if p_value < 0.05 else 'normal'})")
        
        # Create a scatter plot of the first two dimensions
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        scatter = ax.scatter(latent_data[:, 0].numpy(), latent_data[:, 1].numpy(), 
                           alpha=0.5, s=1, c=range(len(latent_data)), cmap='viridis')
        ax.set_xlabel("z₁ (first dimension)")
        ax.set_ylabel("z₂ (second dimension)")
        ax.set_title("Simple Better Latent Space Distribution (First 2 Dimensions)")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("simple_better_latent_space_distribution.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        # Create a heatmap of all dimensions
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        im = ax.imshow(latent_data[:1000].numpy().T, aspect='auto', cmap='viridis')
        ax.set_xlabel("Data Points")
        ax.set_ylabel("Latent Dimensions")
        ax.set_title("Simple Better Latent Space Heatmap (First 1000 points)")
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig("simple_better_latent_space_heatmap.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        # Create a 2D projection using the most important dimensions
        if len(sorted_dims) >= 2:
            dim1, dim2 = sorted_dims[0], sorted_dims[1]
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            scatter = ax.scatter(latent_data[:, dim1].numpy(), latent_data[:, dim2].numpy(), 
                               alpha=0.5, s=1, c=range(len(latent_data)), cmap='viridis')
            ax.set_xlabel(f"z_{dim1} (most important dimension)")
            ax.set_ylabel(f"z_{dim2} (second most important dimension)")
            ax.set_title(f"Simple Better Latent Space: Most Important Dimensions ({dim1}, {dim2})")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("simple_better_latent_space_important_dims.png", dpi=150, bbox_inches='tight')
            plt.show()
        
        print("✅ Simple better latent space analysis saved")
        
        return latent_data

def create_simple_better_reconstruction_visualization(model, dataset, device):
    """Create reconstruction visualization with simple better model."""
    print(f"\n🎨 Creating Simple Better Reconstruction Visualization")
    
    model.eval()
    with torch.no_grad():
        # Sample more indices for better inspection
        indices = np.random.choice(len(dataset), 20, replace=False)
        original_images = []
        reconstructed_images = []
        
        for idx in indices:
            # Get original image
            original = dataset.data[idx].unsqueeze(0)
            original_images.append(original)
            
            # Get latent representation
            encoder_output = model.encoder(original)
            z = encoder_output.embedding
            
            # Reconstruct
            decoder_output = model.decoder(z)
            reconstructed = decoder_output.reconstruction
            reconstructed_images.append(reconstructed)
        
        # Create grid of original vs reconstructed
        original_grid = torch.cat(original_images, dim=0)
        recon_grid = torch.cat(reconstructed_images, dim=0)
        
        # Display as 2x20 grid (original on top, reconstructed on bottom)
        combined = torch.cat([original_grid, recon_grid], dim=0)
        
        # Reshape for display (2 rows, 20 columns)
        combined_display = combined.view(2, 20, 3, 64, 64).permute(0, 1, 3, 2, 4).contiguous()
        combined_display = combined_display.view(2 * 64, 20 * 64, 3)
        
        # Create figure with larger size
        fig, ax = plt.subplots(1, 1, figsize=(30, 12))
        ax.imshow(combined_display.numpy())
        ax.set_title("Simple Better Reconstruction: Original vs Reconstructed Sprites (20 samples)", fontsize=16, fontweight='bold')
        ax.axis('off')
        ax.text(0.02, 0.98, 'Original', transform=ax.transAxes, fontsize=14, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax.text(0.02, 0.48, 'Reconstructed', transform=ax.transAxes, fontsize=14, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save locally
        plt.savefig("simple_better_reconstruction_visualization.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Simple better reconstruction saved as 'simple_better_reconstruction_visualization.png'")

def main():
    """Main function to train simple better VAE and analyze."""
    print("🚀 Training Simple Better VAE on CPU")
    print("=" * 50)
    
    # Step 1: Train simple better VAE
    model, dataset, device = train_simple_better_vae()
    
    # Step 2: Analyze simple better latent space
    latent_data = analyze_simple_better_latent_space(model, dataset, device)
    
    # Step 3: Create simple better reconstruction visualization
    create_simple_better_reconstruction_visualization(model, dataset, device)
    
    print(f"\n✅ Simple better VAE training and analysis completed!")
    wandb.finish()

if __name__ == "__main__":
    main() 