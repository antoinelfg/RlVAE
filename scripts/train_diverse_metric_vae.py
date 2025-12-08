#!/usr/bin/env python3
"""
Diverse Metric VAE Training Script
=================================

Enhanced vanilla VAE training with optimized parameters for metric diversity.
Supports CNN, ResNet, and MLP (Pythae) architectures with custom latent dimensions.
Uses parameter experimentation results to generate metrics with wider eigenvalue
ranges and better variance for improved RHVAE performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader, ConcatDataset
from pythae.models.vae import VAE
from pythae.models.vae.vae_config import VAEConfig
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modular VAE components
from src.models.modular_vanilla_vae import create_cnn_vanilla_vae, create_resnet_vanilla_vae, create_mlp_vanilla_vae

from tqdm import tqdm
import wandb
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import seaborn as sns

class SpritesDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, normalize=False, timestep_only=None):
        """Load dSprites dataset."""
        self.data = torch.load(data_path)
        
        print(f"🔍 Original data shape: {self.data.shape}")
        
        # Handle different data formats
        if len(self.data.shape) == 5:
            if self.data.shape[-1] == 3:
                # Raw format: [batch, seq, h, w, c] -> [batch, seq, c, h, w]
                self.data = self.data.permute(0, 1, 4, 2, 3)
            # Now: [batch, seq, c, h, w]
            
            if timestep_only is not None:
                # Extract only specified timestep
                self.data = self.data[:, timestep_only]  # [batch, c, h, w]
                print(f"📊 Using only timestep {timestep_only}")
            else:
                # Flatten all sequences: [batch*seq, c, h, w]
                batch_size, seq_len = self.data.shape[:2]
                self.data = self.data.reshape(batch_size * seq_len, *self.data.shape[2:])
                print(f"📊 Flattened {batch_size} sequences × {seq_len} timesteps")
                
        elif len(self.data.shape) == 4:
            # Already in correct format: [batch, c, h, w]
            print("📊 Data already in correct format [batch, c, h, w]")
        else:
            raise ValueError(f"Unexpected data shape: {self.data.shape}")
        
        print(f"✅ Data range: [{self.data.min().item():.1f}, {self.data.max().item():.1f}]")
        print(f"✅ Final data shape: {self.data.shape}")
        
        # Ensure contiguous memory layout
        self.data = self.data.contiguous()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def create_model(architecture: str, input_dim=(3, 64, 64), latent_dim=16):
    """Create model based on architecture choice."""
    
    print(f"Creating {architecture.upper()} model with latent_dim={latent_dim}")
    
    if architecture.lower() == "cnn":
        print("🏗️  Using CNN architecture (modular)")
        model = create_cnn_vanilla_vae(
            input_dim=input_dim,
            latent_dim=latent_dim,
            beta=1.0
        )
        
    elif architecture.lower() == "resnet":
        print("🏗️  Using ResNet architecture (modular)")
        model = create_resnet_vanilla_vae(
            input_dim=input_dim,
            latent_dim=latent_dim,
            beta=1.0
        )
        
    elif architecture.lower() in ["mlp", "pythae"]:
        print("🏗️  Using MLP architecture (Pythae VAE)")
        vae_config = VAEConfig(
            input_dim=input_dim,
            latent_dim=latent_dim
        )
        model = VAE(model_config=vae_config)
        
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from: cnn, resnet, mlp, pythae")
    
    return model

def extract_diverse_metric(
    model,
    architecture,
    latent_dim,
    temperature=0.5,
    regularization=0.01,
    num_centroids=50,
    save_dir="data/pretrained",
    input_dim=(3, 64, 64),
    data_path=None,
    timestep_only: int = 0,
    standardize_latents: bool = False,
    centroid_method: str = "kmedoids",  # kmedoids|kmeans|fps|balanced
    neighbor_mode: str = "global",      # global|knn
    knn_k: int = 300,
    coarse_k: int = 8,
    normalize_M: str = "none",          # none|trace|det
    target_mean_eig: float = 1.0,
):
    """
    Extract metric with enhanced diversity parameters.
    Args:
        temperature: Controls metric locality (higher = more diverse eigenvalues)
        regularization: Base regularization (lower = more diverse)
        latent_dim: Dimension of latent space
        input_dim: Input dimension tuple (C, H, W)
        data_path: Path to the dataset to use for metric extraction
    """
    print(f"Extracting DIVERSE metric with T={temperature}, λ={regularization}")
    print(f"Architecture: {architecture}, Latent dim: {latent_dim}, Input dim: {input_dim}")
    
    device = next(model.parameters()).device
    model.eval()
    
    # Use the provided data_path or default to sprites
    if data_path is None:
        data_path = 'data/processed/Sprites_train_cyclic.pt'
    # allow caller to select timestep; default is 0 for Stage B
    train_t0_dataset = SpritesDataset(data_path, normalize=False, timestep_only=timestep_only)
    # If the data shape does not match input_dim, reshape
    if hasattr(train_t0_dataset, 'data') and train_t0_dataset.data.shape[1:] != input_dim:
        print(f"[extract_diverse_metric] Reshaping data from {train_t0_dataset.data.shape[1:]} to {input_dim}")
        train_t0_dataset.data = train_t0_dataset.data.reshape(-1, *input_dim)
    loader = DataLoader(train_t0_dataset, batch_size=256, shuffle=False)
    
    all_mus = []
    all_logvars = []
    
    print("Extracting latent representations from timestep=0...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding timestep=0 data"):
            batch = batch.to(device)
            
            # Handle different model types
            if architecture.lower() in ["mlp", "pythae"]:
                # Pythae VAE expects dict input
                inputs = {"data": batch}
                # Use encoder directly
                encoder_output = model.encoder(batch)
                mu = encoder_output.embedding
                log_var = encoder_output.log_covariance
            else:
                # Modular VAE (CNN, ResNet)
                mu, log_var = model.encode(batch)
            
            all_mus.append(mu.cpu())
            all_logvars.append(log_var.cpu())
    
    all_mus = torch.cat(all_mus, dim=0)  # [N, latent_dim]
    all_logvars = torch.cat(all_logvars, dim=0)  # [N, latent_dim]
    
    print(f"Collected {len(all_mus)} latent representations from timestep=0")
    print(f"Latent representation shape: {all_mus.shape}")
    
    # Optional standardization (improves metric locality and conditioning)
    scaler_mean = None
    scaler_scale = None
    if standardize_latents:
        print("[extract_diverse_metric] Standardizing latent means before clustering and covariance computation")
        scaler = StandardScaler()
        all_mus_np = scaler.fit_transform(all_mus.detach().cpu().numpy())
        all_mus = torch.tensor(all_mus_np, device=device, dtype=all_mus.dtype)
        scaler_mean = torch.tensor(scaler.mean_, device=device, dtype=all_mus.dtype)
        scaler_scale = torch.tensor(scaler.scale_, device=device, dtype=all_mus.dtype)

    # Choose centroid selection method
    warnings.filterwarnings("ignore", category=UserWarning)
    if centroid_method == "balanced":
        print(f"Running balanced centroid selection: k-means({coarse_k}) -> FPS per cluster to {num_centroids} total...")
        all_np = all_mus.detach().cpu().numpy()
        km = KMeans(n_clusters=coarse_k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(all_np)
        # allocate counts proportionally to cluster sizes
        counts = []
        for c in range(coarse_k):
            size = (labels == c).sum()
            counts.append(size)
        counts = np.array(counts)
        props = counts / counts.sum()
        alloc = np.maximum(1, np.floor(props * num_centroids).astype(int))
        # fix rounding
        while alloc.sum() < num_centroids:
            alloc[np.argmax(props - alloc / num_centroids)] += 1
        # select per-cluster FPS
        chosen_indices = []
        for c in range(coarse_k):
            idx_c = np.where(labels == c)[0]
            if len(idx_c) == 0:
                continue
            need = int(alloc[c])
            # FPS within the cluster
            indices_tensor = torch.tensor(idx_c, device=all_mus.device)
            subset = all_mus[indices_tensor]
            # seed with random
            start = torch.randint(0, subset.shape[0], (1,), device=all_mus.device).item()
            sel = [start]
            d = torch.cdist(subset[start:start+1], subset).squeeze(0)
            for _ in range(1, min(need, subset.shape[0])):
                far = torch.argmax(d).item()
                sel.append(far)
                d = torch.minimum(d, torch.cdist(subset[far:far+1], subset).squeeze(0))
            chosen_indices.extend(indices_tensor[torch.tensor(sel, device=all_mus.device)].tolist())
        centroids_idx = torch.tensor(chosen_indices, device='cpu')
        centroids_mu = all_mus[centroids_idx.to(all_mus.device)]
    elif centroid_method == "kmeans":
        print(f"Running k-means++ to find {num_centroids} centers...")
        km = KMeans(n_clusters=num_centroids, init="k-means++", n_init=10, random_state=42)
        km.fit(all_mus.detach().cpu().numpy())
        centers = torch.tensor(km.cluster_centers_, device=device, dtype=all_mus.dtype)
        centroids_mu = centers
        centroids_idx = None
    elif centroid_method == "fps":
        print(f"Running farthest-point sampling to select {num_centroids} centroids...")
        with torch.no_grad():
            N = all_mus.shape[0]
            chosen = []
            # start from a random point
            start_idx = torch.randint(0, N, (1,), device=device).item()
            chosen.append(start_idx)
            # maintain min distances to chosen set
            dists = torch.cdist(all_mus[start_idx:start_idx+1].to(device), all_mus.to(device)).squeeze(0)  # [N]
            for _ in range(1, num_centroids):
                far_idx = torch.argmax(dists).item()
                chosen.append(far_idx)
                new_d = torch.cdist(all_mus[far_idx:far_idx+1].to(device), all_mus.to(device)).squeeze(0)
                dists = torch.minimum(dists, new_d)
        centroids_idx = torch.tensor(chosen, device='cpu')
        centroids_mu = all_mus[centroids_idx.to(all_mus.device)]
    else:
        print(f"Running k-medoids to find {num_centroids} centroids...")
        kmedoids = KMedoids(n_clusters=num_centroids, random_state=42).fit(all_mus.detach().cpu().numpy())
        medoids = torch.tensor(kmedoids.cluster_centers_, device=device, dtype=all_mus.dtype)
        centroids_idx = torch.tensor(kmedoids.medoid_indices_, device='cpu')
        # Index on matching device to avoid device mismatch
        centroids_mu = all_mus[centroids_idx.to(all_mus.device)]
    # centroids_logvar kept for API compatibility (unused here)
    centroids_logvar = all_logvars[:centroids_mu.shape[0]]
    
    print(f"✅ Selected {len(centroids_mu)} centroids via k-medoids")
    print(f"✅ Using DIVERSE parameters: T={temperature}, λ={regularization}")
    
    M_matrices = []
    
    print("Computing diverse local metric matrices...")
    # Ensure all tensors involved in distance computations share the same device
    # Move centroid vectors to the device of all_mus for consistent arithmetic
    centroids_mu = centroids_mu.to(all_mus.device)
    for i, c in enumerate(tqdm(centroids_mu)):
        if neighbor_mode == "knn":
            # Use k nearest neighbors for local covariance
            # All on same device
            dists = torch.norm(all_mus - c, dim=1)
            k = min(knn_k, all_mus.shape[0])
            knn_vals, knn_idx = torch.topk(-dists, k=k)  # negative distances -> largest are closest? wrong
            # Correct: topk with smallest distances
            knn_idx = torch.topk(dists, k=k, largest=False).indices
            local = all_mus[knn_idx]
            mean = local.mean(dim=0)
            diffs = local - mean.unsqueeze(0)
            weighted_cov = torch.einsum('ni,nj->ij', diffs, diffs) / max(1, local.shape[0]-1)
        else:
            # Global weighted covariance
            # All on same device
            dists = torch.norm(all_mus - c, dim=1)
            weights = torch.exp(-dists ** 2 / (temperature ** 2))
            weights = weights / (weights.sum() + 1e-8)
            mean = (weights.unsqueeze(1) * all_mus).sum(dim=0)
            diffs = all_mus - mean.unsqueeze(0)
            weighted_cov = torch.einsum('n,ni,nj->ij', weights, diffs, diffs)

        # Regularize
        metric = weighted_cov + regularization * torch.eye(latent_dim, device=all_mus.device, dtype=all_mus.dtype)
        
        # Ensure positive definiteness
        eigenvals = torch.linalg.eigvals(metric).real
        min_eigenval = eigenvals.min().item()
        if min_eigenval < 1e-6:
            metric = metric + (1e-6 - min_eigenval) * torch.eye(latent_dim)

        # Optional normalization of scale so average eigenvalue is controlled
        if normalize_M and normalize_M.lower() != "none":
            if normalize_M.lower() == "trace":
                mean_eig = torch.trace(metric) / float(latent_dim)
                metric = metric / (mean_eig + 1e-8) * float(target_mean_eig)
            elif normalize_M.lower() == "det":
                det_val = torch.linalg.det(metric)
                # scale so det ≈ target_mean_eig^latent_dim
                desired_det = float(target_mean_eig) ** float(latent_dim)
                scale = (det_val.abs() + 1e-12) ** (1.0 / float(latent_dim))
                metric = metric / (scale + 1e-8) * (desired_det ** (1.0 / float(latent_dim)))
        
        M_matrices.append(metric)
        
        if i % 10 == 0:
            print(f"Centroid {i}: min eigenvalue = {min_eigenval:.6f}, "
                  f"condition number = {torch.linalg.cond(metric).item():.2f}")
    
    M_matrices = torch.stack(M_matrices, dim=0)  # [n_centroids, latent_dim, latent_dim]
    
    # Print enhanced metric statistics
    eigenvals = torch.linalg.eigvals(M_matrices).real
    min_eigenvals = eigenvals.min(dim=-1)[0]
    max_eigenvals = eigenvals.max(dim=-1)[0]
    cond_nums = max_eigenvals / (min_eigenvals + 1e-10)
    dets = torch.linalg.det(M_matrices)
    eigenval_ratio = max_eigenvals.max() / min_eigenvals.min()
    eigenval_std = eigenvals.std()
    
    print(f"\n🎯 DIVERSE Metric Statistics ({architecture.upper()}, latent_dim={latent_dim}):")
    print(f"Eigenvalue ratio: {eigenval_ratio.item():.2f}")
    print(f"Eigenvalue std: {eigenval_std.item():.6f}")
    print(f"Min eigenvalue: {min_eigenvals.min().item():.6f}")
    print(f"Max eigenvalue: {max_eigenvals.max().item():.6f}")
    print(f"Mean condition number: {cond_nums.mean().item():.2f}")
    print(f"Determinant range: [{dets.min().item():.3e}, {dets.max().item():.3e}]")
    
    # Save metric data (include 2D projections for stage summary viz if latent_dim==2)
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Aggregate scale statistics
    with torch.no_grad():
        traces = torch.stack([torch.trace(M) for M in M_matrices])
        mean_trace = traces.mean().item()
        mean_eig_overall = (traces / float(latent_dim)).mean().item()
        dets_all = torch.stack([torch.linalg.det(M) for M in M_matrices])
        mean_det = dets_all.mean().item()

    metric_data = {
        'centroids': centroids_mu,
        'M_matrices': M_matrices,
        'temperature': torch.tensor(temperature),
        'regularization': torch.tensor(regularization),
        'latent_dim': latent_dim,
        'n_centroids': num_centroids,
        'centroids_idx': centroids_idx,
        'architecture': architecture,
        'timestamp': timestamp,
        'extraction_method': f'diverse_k_medoids_{architecture}_T{temperature}_reg{regularization}',
        'diversity_stats': {
            'eigenvalue_ratio': eigenval_ratio.item(),
            'eigenvalue_std': eigenval_std.item(),
            'condition_number_mean': cond_nums.mean().item(),
        },
        'standardize_latents': bool(standardize_latents),
        'scale_stats': {
            'mean_trace': mean_trace,
            'mean_mean_eig': mean_eig_overall,
            'mean_det': mean_det,
            'normalize_M': normalize_M,
            'target_mean_eig': float(target_mean_eig),
        },
    }
    # Cache a light-weight subsample of z for visualization overlays
    try:
        sample_idx = torch.linspace(0, all_mus.shape[0]-1, steps=min(4000, all_mus.shape[0])).long()
        metric_data['z_sample'] = all_mus[sample_idx].cpu()
    except Exception:
        pass
    if standardize_latents:
        metric_data['scaler_mean'] = scaler_mean
        metric_data['scaler_scale'] = scaler_scale
    
    metric_path = save_path / f"metric_diverse_{architecture}_ld{latent_dim}_{timestamp}.pt"
    torch.save(metric_data, metric_path)
    
    print(f"✅ Saved DIVERSE metric to {metric_path}")
    print(f"   Architecture: {architecture.upper()}")
    print(f"   Latent dim: {latent_dim}")
    print(f"   Eigenvalue ratio: {eigenval_ratio.item():.2f}x")
    print(f"   Eigenvalue std: {eigenval_std.item():.6f}")
    
    return str(metric_path)

def save_model_components(model, architecture, latent_dim, save_dir="data/pretrained"):
    """Save model components with architecture and latent dim info."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full model
    model_path = save_path / f"vae_diverse_{architecture}_ld{latent_dim}_{timestamp}.pt"
    torch.save(model.state_dict(), model_path)
    
    # Save encoder and decoder
    encoder_path = save_path / f"encoder_diverse_{architecture}_ld{latent_dim}_{timestamp}.pt"
    decoder_path = save_path / f"decoder_diverse_{architecture}_ld{latent_dim}_{timestamp}.pt"
    
    if architecture.lower() in ["mlp", "pythae"]:
        # Pythae VAE
        torch.save(model.encoder.state_dict(), encoder_path)
        torch.save(model.decoder.state_dict(), decoder_path)
    else:
        # Modular VAE
        torch.save(model.encoder.state_dict(), encoder_path)
        torch.save(model.decoder.state_dict(), decoder_path)
    
    component_paths = {
        'model': str(model_path),
        'encoder': str(encoder_path),
        'decoder': str(decoder_path)
    }
    
    print(f"✅ Saved model components:")
    for component, path in component_paths.items():
        print(f"   {component}: {Path(path).name}")
    
    return component_paths

def main():
    parser = argparse.ArgumentParser(description="Train Vanilla VAE with Diverse Metrics")
    parser.add_argument("--architecture", "-a", 
                       choices=["cnn", "resnet", "mlp", "pythae"], 
                       default="cnn",
                       help="Architecture: cnn, resnet, mlp/pythae (default: cnn)")
    parser.add_argument("--latent-dim", "-ld", type=int, default=16,
                       help="Latent space dimension (default: 16)")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--temperature", "-T", type=float, default=0.5, 
                       help="Metric temperature (higher=more diverse)")
    parser.add_argument("--regularization", "-R", type=float, default=0.01,
                       help="Metric regularization (lower=more diverse)")
    parser.add_argument("--preset", choices=["max_diversity", "balanced", "conservative"],
                       help="Use preset parameter combinations")
    parser.add_argument("--wandb-group", type=str, default=None,
                       help="Wandb group for experiment organization")
    
    args = parser.parse_args()
    
    # Apply presets
    if args.preset == "max_diversity":
        args.temperature = 2.0
        args.regularization = 0.001
        print("🎯 Using MAX DIVERSITY preset: T=2.0, λ=0.001")
    elif args.preset == "balanced":
        args.temperature = 0.5
        args.regularization = 0.01
        print("⚖️ Using BALANCED preset: T=0.5, λ=0.01") 
    elif args.preset == "conservative":
        args.temperature = 0.1
        args.regularization = 0.01
        print("🛡️ Using CONSERVATIVE preset: T=0.1, λ=0.01")
    
    print(f"\n{'='*80}")
    print(f"🧠 TRAINING DIVERSE METRIC VAE")
    print(f"   Architecture: {args.architecture.upper()}")
    print(f"   Latent Dimension: {args.latent_dim}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Regularization: {args.regularization}")
    print(f"{'='*80}")
    
    # Initialize wandb with optional grouping
    wandb_config = {
        "architecture": args.architecture,
        "latent_dim": args.latent_dim,
        "temperature": args.temperature,
        "regularization": args.regularization,
        "preset": args.preset,
        "epochs": args.epochs,
        "stage": "vae_training"
    }
    
    wandb_kwargs = {
        "project": "diverse_metric_vae",
        "name": f"vae_{args.architecture}_ld{args.latent_dim}_T{args.temperature}_R{args.regularization}",
        "config": wandb_config,
        "tags": ["vae", "pipeline", args.architecture]
    }
    
    if args.wandb_group:
        wandb_kwargs["group"] = args.wandb_group
    
    wandb.init(**wandb_kwargs)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_dataset = SpritesDataset('data/processed/Sprites_train_cyclic.pt', normalize=False)
    test_dataset = SpritesDataset('data/processed/Sprites_test_cyclic.pt', normalize=False)
    full_dataset = ConcatDataset([train_dataset, test_dataset])
    
    train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = create_model(args.architecture, input_dim=(3, 64, 64), latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"Training for {args.epochs} epochs...")
    
    # Initialize batch counter for reconstruction logging
    batch_counter = 0
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kld_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch = batch.to(device)
            
            # Handle different model types
            if args.architecture.lower() in ["mlp", "pythae"]:
                # Pythae VAE
                inputs = {"data": batch}
                output = model(inputs)
                loss = output.loss
                # Pythae VAE uses different attribute names
                recon_loss = output.recon_loss.item()
                kld_loss = output.reg_loss.item()
                # Get reconstructions for logging
                recon_batch = output.recon_x
            else:
                # Modular VAE
                output = model(batch)
                loss = output.loss
                recon_loss = output.reconstruction_loss.item()
                kld_loss = output.reg_loss.item()
                # Get reconstructions for logging
                recon_batch = output.recon_x
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss
            total_kld_loss += kld_loss
            
            # Log reconstruction images every 100 batches
            if batch_counter % 100 == 0:
                with torch.no_grad():
                    # Get reconstructions and clamp to [0,1] for proper visualization
                    recon_display = recon_batch.clamp(0, 1)
                    # Create grid of original and reconstructed images
                    comparison = torch.cat([batch[:8], recon_display[:8]], dim=0)
                    grid = vutils.make_grid(comparison, nrow=8, normalize=False)
                    wandb.log({
                        "reconstructions": wandb.Image(grid),
                        "batch": batch_counter,
                        "epoch": epoch + 1
                    })
            
            batch_counter += 1
        
        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_kld_loss = total_kld_loss / len(train_loader)
        
        # Validation loop
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kld_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                # Handle different model types for validation
                if args.architecture.lower() in ["mlp", "pythae"]:
                    inputs = {"data": batch}
                    output = model(inputs)
                    # Pythae VAE attribute names
                    recon_loss_val = output.recon_loss.item()
                    kld_loss_val = output.reg_loss.item()
                else:
                    output = model(batch)
                    # Modular VAE attribute names
                    recon_loss_val = output.reconstruction_loss.item()
                    kld_loss_val = output.reg_loss.item()
                
                # Skip NaN validation losses
                if not (torch.isnan(output.loss) or torch.isinf(output.loss)):
                    val_loss += output.loss.item()
                    val_recon_loss += recon_loss_val
                    val_kld_loss += kld_loss_val
                    val_batches += 1
        
        # Calculate average validation losses
        if val_batches > 0:
            avg_val_loss = val_loss / val_batches
            avg_val_recon_loss = val_recon_loss / val_batches
            avg_val_kld_loss = val_kld_loss / val_batches
        else:
            avg_val_loss = avg_val_recon_loss = avg_val_kld_loss = 0
        
        # Log all metrics to wandb
        wandb.log({
            "epoch": epoch + 1, 
            "train/loss": avg_loss,
            "train/reconstruction_loss": avg_recon_loss,
            "train/kld_loss": avg_kld_loss,
            "val/loss": avg_val_loss,
            "val/reconstruction_loss": avg_val_recon_loss,
            "val/kld_loss": avg_val_kld_loss
        })
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train - Loss: {avg_loss:.4f} (Recon: {avg_recon_loss:.4f} + KL: {avg_kld_loss:.4f})")
        print(f"  Val   - Loss: {avg_val_loss:.4f} (Recon: {avg_val_recon_loss:.4f} + KL: {avg_val_kld_loss:.4f})")
    
    print("Training complete. Saving components...")
    
    # Save model components
    component_paths = save_model_components(model, args.architecture, args.latent_dim)
    
    # Check final reconstructions visually and print summary
    print("Checking final reconstructions...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader)).to(device)
        
        # Handle different model types for final reconstruction
        if args.architecture.lower() in ["mlp", "pythae"]:
            inputs = {"data": batch}
            output = model(inputs)
            recon = output.recon_x.cpu().clamp(0, 1)
        else:
            output = model(batch)
            recon = output.recon_x.cpu().clamp(0, 1)
        
        orig = batch.cpu()
        n = min(8, orig.size(0))
        comparison = torch.cat([orig[:n], recon[:n]], dim=0)
        grid = vutils.make_grid(comparison, nrow=n, normalize=False)
        
        # Log final reconstruction comparison
        wandb.log({
            "final_reconstruction_comparison": wandb.Image(grid),
            "final_reconstruction_mse": F.mse_loss(recon, orig).item()
        })
        
        print(f"Final Reconstruction MSE: {F.mse_loss(recon, orig).item():.6f}")
    
    print("Extracting DIVERSE metric...")
    
    # Extract diverse metric
    metric_path = extract_diverse_metric(
        model, args.architecture, args.latent_dim,
        temperature=args.temperature, 
        regularization=args.regularization,
        input_dim=(3, 64, 64) # Pass input_dim here
    )
    
    component_paths['metric'] = metric_path
    
    # Log final results
    wandb.log({
        "model_path": component_paths['model'],
        "encoder_path": component_paths['encoder'],
        "decoder_path": component_paths['decoder'],
        "metric_path": metric_path
    })

    # --- METRIC ANALYSIS & VISUALIZATION ---
    plt.style.use('default')
    sns.set_palette("husl")
    try:
        metric_data = torch.load(metric_path, map_location='cpu', weights_only=False)
        M_matrices = metric_data['M_matrices']
        centroids = metric_data['centroids']
        eigenvals = torch.linalg.eigvals(M_matrices).real
        min_eigenvals = eigenvals.min(dim=-1)[0]
        max_eigenvals = eigenvals.max(dim=-1)[0]
        mean_eigenvals = eigenvals.mean(dim=-1)
        cond_nums = max_eigenvals / (min_eigenvals + 1e-12)
        determinants = torch.linalg.det(M_matrices)
        eigenval_spread = max_eigenvals - min_eigenvals
        # 1. Eigenvalue distributions
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Metric Eigenvalue & Condition Analysis', fontsize=16, fontweight='bold')
        axes[0,0].hist(min_eigenvals.numpy(), bins=40, color='red', edgecolor='black')
        axes[0,0].set_title('Min Eigenvalue')
        axes[0,1].hist(max_eigenvals.numpy(), bins=40, color='blue', edgecolor='black')
        axes[0,1].set_title('Max Eigenvalue')
        axes[0,2].hist(mean_eigenvals.numpy(), bins=40, color='green', edgecolor='black')
        axes[0,2].set_title('Mean Eigenvalue')
        axes[1,0].hist(cond_nums.numpy(), bins=40, color='orange', edgecolor='black')
        axes[1,0].set_title('Condition Number')
        log_dets = torch.log10(torch.abs(determinants) + 1e-50)
        axes[1,1].hist(log_dets.numpy(), bins=40, color='purple', edgecolor='black')
        axes[1,1].set_title('Log₁₀|Determinant|')
        axes[1,2].hist(eigenval_spread.numpy(), bins=40, color='cyan', edgecolor='black')
        axes[1,2].set_title('Eigenvalue Spread')
        plt.tight_layout()
        wandb.log({"metric_analysis/eigenvalue_distributions": wandb.Image(fig)})
        plt.close(fig)
        # 2. Heatmaps of a few metric matrices
        n_heatmaps = min(6, len(M_matrices))
        fig, axes = plt.subplots(1, n_heatmaps, figsize=(4*n_heatmaps, 4))
        for i in range(n_heatmaps):
            im = axes[i].imshow(M_matrices[i].numpy(), cmap='RdYlBu_r', aspect='auto')
            axes[i].set_title(f'Matrix {i}')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        plt.tight_layout()
        wandb.log({"metric_analysis/metric_matrix_heatmaps": wandb.Image(fig)})
        plt.close(fig)
        # 3. Centroid statistics
        centroid_norms = torch.norm(centroids, dim=1)
        pairwise_dists = torch.cdist(centroids, centroids)
        triu_mask = torch.triu(torch.ones_like(pairwise_dists, dtype=bool), diagonal=1)
        pairwise_vals = pairwise_dists[triu_mask]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(centroid_norms.numpy(), bins=30, color='red', edgecolor='black')
        axes[0].set_title('Centroid Norms')
        axes[1].hist(pairwise_vals.numpy(), bins=40, color='blue', edgecolor='black')
        axes[1].set_title('Pairwise Centroid Distances')
        plt.tight_layout()
        wandb.log({"metric_analysis/centroid_stats": wandb.Image(fig)})
        plt.close(fig)
    except Exception as e:
        print(f"⚠️  Could not analyze metric for graphs: {e}")

    wandb.finish()
    print(f"\n✅ Diverse metric training complete!")
    print(f"   Architecture: {args.architecture.upper()}")
    print(f"   Latent dim: {args.latent_dim}")
    print(f"   Components saved: {[Path(p).name for p in component_paths.values()]}")

if __name__ == "__main__":
    main()
