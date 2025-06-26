import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from vanilla_vae import VanillaVAE
import os
from tqdm import tqdm
import wandb
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler

class SpritesDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
        # Reshape data from [batch, seq, h, w, c] to [batch*seq, c, h, w]
        self.data = self.data.permute(0, 1, 4, 2, 3)  # [batch, seq, c, h, w]
        self.data = self.data.reshape(-1, 3, 64, 64)  # [batch*seq, c, h, w]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def main():
    # Initialize wandb
    wandb.init(
        project="vanilla_vae_training",
        name="vanilla_vae_sprites",
        config={
            "architecture": "VanillaVAE",
            "dataset": "Sprites",
            "latent_dim": 16,
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 50
        }
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs('src/datasprites/vanilla_vae_components', exist_ok=True)
    
    print("Loading data...")
    # Load data
    train_dataset = SpritesDataset('src/datasprites/Sprites_train.pt')
    test_dataset = SpritesDataset('src/datasprites/Sprites_test.pt')
    
    print(f"Training data size: {len(train_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = VanillaVAE(
        input_dim=(3, 64, 64),  # Sprites are 64x64 RGB images
        latent_dim=16
    ).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    num_epochs = 50
    print("Starting training...")
    
    # Initialize batch counter
    batch_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kld_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            
            # Forward pass
            output = model(batch)
            loss = output.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += output.reconstruction_loss.item()
            total_kld_loss += output.reg_loss.item()
            
            # Log reconstruction images every 100 batches
            if batch_counter % 100 == 0:
                with torch.no_grad():
                    # Get reconstructions
                    recon_batch = output.recon_x
                    # Create grid of original and reconstructed images
                    comparison = torch.cat([batch[:8], recon_batch[:8]], dim=0)
                    grid = vutils.make_grid(comparison, nrow=8, normalize=True)
                    wandb.log({
                        "reconstructions": wandb.Image(grid),
                        "batch": batch_counter
                    })
            
            batch_counter += 1
        
        # Calculate average losses
        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_kld_loss = total_kld_loss / len(train_loader)
        
        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_loss,
            "train/reconstruction_loss": avg_recon_loss,
            "train/kld_loss": avg_kld_loss
        })
        
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kld_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch)
                val_loss += output.loss.item()
                val_recon_loss += output.reconstruction_loss.item()
                val_kld_loss += output.reg_loss.item()
        
        # Calculate average validation losses
        avg_val_loss = val_loss / len(test_loader)
        avg_val_recon_loss = val_recon_loss / len(test_loader)
        avg_val_kld_loss = val_kld_loss / len(test_loader)
        
        # Log validation metrics
        wandb.log({
            "epoch": epoch + 1,
            "val/loss": avg_val_loss,
            "val/reconstruction_loss": avg_val_recon_loss,
            "val/kld_loss": avg_val_kld_loss
        })
        
        print(f"Validation Loss: {avg_val_loss:.4f}")
    
    print("Training complete. Extracting components...")
    
    # Extract and save components
    torch.save(model.state_dict(), 'src/datasprites/vanilla_vae_components/model_state.pt')
    
    # Extract encoder and decoder
    torch.save(model.encoder.state_dict(), 'src/datasprites/vanilla_vae_components/encoder.pt')
    torch.save(model.decoder.state_dict(), 'src/datasprites/vanilla_vae_components/decoder.pt')
    
    # --- RHVAE-style metric extraction ---
    print("Extracting RHVAE-style metric from training data (timestep 0 only)...")
    model.eval()
    all_mus = []
    all_logvars = []
    with torch.no_grad():
        for batch in tqdm(train_loader):
            batch = batch.to(device)
            # If batch is [B, T, C, H, W], take only timestep 0
            if batch.dim() == 5:
                batch_0 = batch[:, 0]  # [B, C, H, W]
            elif batch.dim() == 4:
                batch_0 = batch  # [B, C, H, W] (no time dimension)
            else:
                raise ValueError(f"Unexpected batch shape: {batch.shape}")
            mu, log_var = model.encode(batch_0)
            all_mus.append(mu.cpu())
            all_logvars.append(log_var.cpu())
    all_mus = torch.cat(all_mus, dim=0)  # [N, latent_dim]
    all_logvars = torch.cat(all_logvars, dim=0)  # [N, latent_dim]

    # --- Use k-medoids for centroid selection ---
    n_centroids = 50
    mus_np = all_mus.numpy()
    scaler = StandardScaler()
    mus_scaled = scaler.fit_transform(mus_np)
    kmedoids = KMedoids(n_clusters=n_centroids, random_state=42, max_iter=1000, init='k-medoids++')
    kmedoids.fit(mus_scaled)
    print(f"✅ Using k-medoids for centroid selection.")
    centroids = torch.from_numpy(mus_np[kmedoids.medoid_indices_])

    # For each centroid, compute a local metric (covariance of nearby points)
    temperature = 0.1
    regularization = 0.01
    latent_dim = all_mus.shape[1]
    M_matrices = []

    print("Computing local metric matrices...")
    for i, c in enumerate(tqdm(centroids)):
        # Compute distances to all points
        dists = torch.norm(all_mus - c, dim=1)
        # Gaussian weights with temperature
        weights = torch.exp(-dists ** 2 / (temperature ** 2))
        weights = weights / (weights.sum() + 1e-8)
        # Weighted mean
        mean = (weights.unsqueeze(1) * all_mus).sum(dim=0)
        # Weighted covariance
        diffs = all_mus - mean.unsqueeze(0)
        weighted_cov = torch.einsum('n,ni,nj->ij', weights, diffs, diffs)
        # Add regularization to ensure positive definiteness
        metric = weighted_cov + regularization * torch.eye(latent_dim)
        # Ensure positive definiteness
        eigenvals = torch.linalg.eigvals(metric).real
        min_eigenval = eigenvals.min().item()
        if min_eigenval < 1e-6:
            metric = metric + (1e-6 - min_eigenval) * torch.eye(latent_dim)
        M_matrices.append(metric)
        if i % 10 == 0:
            print(f"Centroid {i}: min eigenvalue = {min_eigenval:.6f}, "
                  f"condition number = {torch.linalg.cond(metric).item():.2f}")
    M_matrices = torch.stack(M_matrices, dim=0)  # [n_centroids, latent_dim, latent_dim]

    # Print metric statistics
    eigenvals = torch.linalg.eigvals(M_matrices).real
    min_eigenvals = eigenvals.min(dim=-1)[0]
    max_eigenvals = eigenvals.max(dim=-1)[0]
    cond_nums = max_eigenvals / (min_eigenvals + 1e-10)
    dets = torch.linalg.det(M_matrices)
    
    print("\nMetric statistics:")
    print(f"Min eigenvalue: {min_eigenvals.min().item():.6f}")
    print(f"Max eigenvalue: {max_eigenvals.max().item():.6f}")
    print(f"Mean condition number: {cond_nums.mean().item():.2f}")
    print(f"Determinant range: [{dets.min().item():.3e}, {dets.max().item():.3e}]")

    metric_data = {
        'centroids': centroids,
        'M_matrices': M_matrices,
        'temperature': torch.tensor(temperature),
        'regularization': torch.tensor(regularization),
        'latent_dim': latent_dim,
        'n_centroids': n_centroids
    }

    # Save in the expected format
    torch.save(metric_data, 'src/datasprites/vanilla_vae_components/metric.pt')
    print("Saved RHVAE-style metric to src/datasprites/vanilla_vae_components/metric.pt")

    # --- Check reconstructions visually and print summary ---
    print("Checking reconstructions...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader)).to(device)
        output = model(batch)
        recon = output.recon_x.cpu()
        orig = batch.cpu()
        n = min(8, orig.size(0))
        comparison = torch.cat([orig[:n], recon[:n]], dim=0)
        grid = vutils.make_grid(comparison, nrow=n, normalize=True)
        plt.figure(figsize=(12, 4))
        plt.axis('off')
        plt.title('Top: Original, Bottom: Reconstruction')
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.show()
    print(f"Recon MSE: {F.mse_loss(recon, orig).item():.4f}")
    print("Components extracted and saved successfully!")
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    main() 