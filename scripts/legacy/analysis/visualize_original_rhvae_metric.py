#!/usr/bin/env python3
"""
Visualize Original RHVAE Metric
===============================

Create comprehensive visualizations for the original RHVAE implementation.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
import os

class HVAE(nn.Module):
    """Base HVAE class for the RHVAE implementation."""
    def __init__(self, args):
        super().__init__()
        self.input_dim = args.input_dim
        self.n_channels = args.n_channels
        self.latent_dim = args.latent_dim
        self.device = args.device
        self.n_lf = args.n_lf
        self.eps_lf = args.eps_lf
        self.beta_zero = args.beta_zero
        self.beta_zero_sqrt = np.sqrt(args.beta_zero)
        
        # Placeholder for VAE components
        self.encoder = nn.Linear(self.input_dim * self.n_channels, self.latent_dim * 2)
        self.decoder = nn.Linear(self.latent_dim, self.input_dim * self.n_channels)
        
        # Normal distribution
        self.normal = torch.distributions.Normal(0, 1)
    
    def vae_forward(self, x):
        """VAE forward pass."""
        # Encode
        encoded = self.encoder(x.reshape(-1, self.input_dim * self.n_channels))
        mu, log_var = encoded[:, :self.latent_dim], encoded[:, self.latent_dim:]
        
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z0 = mu + eps * std
        
        return self.decoder(z0), z0, eps, mu, log_var
    
    def decode(self, z):
        """Decode latent to reconstruction."""
        return self.decoder(z)
    
    def hamiltonian(self, recon_x, x, z, rho, G_inv, G_log_det):
        """Compute Hamiltonian."""
        # Simplified Hamiltonian
        recon_loss = torch.nn.functional.mse_loss(recon_x, x.reshape(recon_x.shape))
        kinetic = 0.5 * torch.sum(rho * rho)
        return recon_loss + kinetic
    
    def log_p_xz(self, recon_x, x, z):
        """Log probability p(x|z)."""
        return -torch.nn.functional.mse_loss(recon_x, x.reshape(recon_x.shape), reduction='sum')
    
    def _tempering(self, k, n_lf):
        """Tempering function."""
        return self.beta_zero_sqrt

class RHVAE(HVAE):
    def __init__(self, args):
        super().__init__(args)
        # defines the Neural net to compute the metric

        # first layer
        self.metric_fc1 = nn.Linear(self.input_dim*self.n_channels, args.metric_fc)

        # diagonal
        self.metric_fc21 = nn.Linear(args.metric_fc, self.latent_dim)
        # remaining coefficients
        k = int(self.latent_dim * (self.latent_dim - 1) / 2)
        self.metric_fc22 = nn.Linear(args.metric_fc, k)

        self.T = nn.Parameter(torch.Tensor([args.temperature]), requires_grad=False)
        self.lbd = nn.Parameter(
            torch.Tensor([args.regularization]), requires_grad=False
        )

        # this is used to store the matrices and centroids throughout trainning for
        # further use in metric update (L is the cholesky decomposition of M)
        self.M = []
        self.centroids = []

        # define a starting metric (gamma_i = 0 & L = I_d)
        def G(z):
            return (
                torch.eye(self.latent_dim, device=self.device).unsqueeze(0)
                * torch.exp(-torch.norm(z.unsqueeze(1), dim=-1) ** 2)
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

        self.G = G

    def metric_forward(self, x):
        """
        This function returns the outputs of the metric neural network

        Outputs:
        --------

        L (Tensor): The L matrix as used in the metric definition
        M (Tensor): L L^T
        """

        h1 = torch.relu(self.metric_fc1(x.reshape(-1, self.input_dim*self.n_channels)))
        h21, h22 = self.metric_fc21(h1), self.metric_fc22(h1)

        L = torch.zeros((x.shape[0], self.latent_dim, self.latent_dim)).to(self.device)
        indices = torch.tril_indices(
            row=self.latent_dim, col=self.latent_dim, offset=-1
        )

        # get non-diagonal coefficients
        L[:, indices[0], indices[1]] = h22

        # add diagonal coefficients
        L = L + torch.diag_embed(h21.exp())

        return L, L @ torch.transpose(L, 1, 2)

    def update_metric(self):
        """
        As soon as the model has seen all the data points (i.e. at the end of 1 loop)
        we update the final metric function using \mu(x_i) as centroids
        """
        # convert to 1 big tensor
        self.M_tens = torch.cat(self.M)
        self.centroids_tens = torch.cat(self.centroids)

        # define new metric
        def G(z):
            return torch.inverse(
                (
                    self.M_tens.unsqueeze(0)
                    * torch.exp(
                        -torch.norm(
                            self.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                        )
                        ** 2
                        / (self.T ** 2)
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                ).sum(dim=1)
                + self.lbd * torch.eye(self.latent_dim).to(self.device)
            )

        def G_inv(z):
            return (
                self.M_tens.unsqueeze(0)
                * torch.exp(
                    -torch.norm(
                        self.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                    )
                    ** 2
                    / (self.T ** 2)
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

        self.G = G
        self.G_inv = G_inv
        self.M = []
        self.centroids = []

    def forward(self, x):
        """
        The RHVAE model
        """

        recon_x, z0, eps0, mu, log_var = self.vae_forward(x)

        z = z0

        if self.training:

            # update the metric using batch data points
            L, M = self.metric_forward(x)

            # store LL^T and mu(x_i) to update final metric
            self.M.append(M.clone().detach())
            self.centroids.append(mu.clone().detach())

            G_inv = (
                M.unsqueeze(0)
                * torch.exp(
                    -torch.norm(mu.unsqueeze(0) - z.unsqueeze(1), dim=-1) ** 2
                    / (self.T ** 2)
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

        else:
            G = self.G(z)
            G_inv = self.G_inv(z)
            L = torch.cholesky(G)

        G_log_det = -torch.logdet(G_inv)

        gamma = torch.randn_like(z0, device=self.device)
        rho = gamma / self.beta_zero_sqrt
        beta_sqrt_old = self.beta_zero_sqrt

        # sample \rho from N(0, G)
        rho = (L @ rho.unsqueeze(-1)).squeeze(-1)

        recon_x = self.decode(z)

        for k in range(self.n_lf):

            # perform leapfrog steps

            # step 1
            rho_ = self.leap_step_1(recon_x, x, z, rho, G_inv, G_log_det)

            # step 2
            z = self.leap_step_2(recon_x, x, z, rho_, G_inv, G_log_det)

            recon_x = self.decode(z)

            if self.training:
                G_inv = (
                    M.unsqueeze(0)
                    * torch.exp(
                        -torch.norm(mu.unsqueeze(0) - z.unsqueeze(1), dim=-1) ** 2
                        / (self.T ** 2)
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

            else:
                # compute metric value on new z using final metric
                G = self.G(z)
                G_inv = self.G_inv(z)

            G_log_det = -torch.logdet(G_inv)

            # step 3
            rho__ = self.leap_step_3(recon_x, x, z, rho_, G_inv, G_log_det)

            # tempering
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt

        return recon_x, z, z0, rho, eps0, gamma, mu, log_var, G_inv, G_log_det

    def leap_step_1(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves first equation of generalized leapfrog integrator
        using fixed point iterations
        """

        def f_(rho_):
            H = self.hamiltonian(recon_x, x, z, rho_, G_inv, G_log_det)
            gz = grad(H, z, retain_graph=True)[0]
            return rho - 0.5 * self.eps_lf * gz

        rho_ = rho.clone()
        for _ in range(steps):
            rho_ = f_(rho_)
        return rho_

    def leap_step_2(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves second equation of generalized leapfrog integrator
        using fixed point iterations
        """
        H0 = self.hamiltonian(recon_x, x, z, rho, G_inv, G_log_det)
        grho_0 = grad(H0, rho)[0]

        def f_(z_):
            H = self.hamiltonian(recon_x, x, z_, rho, G_inv, G_log_det)
            grho = grad(H, rho, retain_graph=True)[0]
            return z + 0.5 * self.eps_lf * (grho_0 + grho)

        z_ = z.clone()
        for _ in range(steps):
            z_ = f_(z_)
        return z_

    def leap_step_3(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves third equation of generalized leapfrog integrator
        using fixed point iterations
        """
        H = self.hamiltonian(recon_x, x, z, rho, G_inv, G_log_det)
        gz = grad(H, z, create_graph=True)[0]
        return rho - 0.5 * self.eps_lf * gz

def create_rhvae_visualizations():
    """Create comprehensive visualizations for the original RHVAE."""
    print("🎨 CREATING COMPREHENSIVE RHVAE VISUALIZATIONS")
    print("=" * 55)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create args object
    class Args:
        def __init__(self):
            self.input_dim = 64 * 64  # 64x64 images
            self.n_channels = 3  # RGB
            self.latent_dim = 16
            self.device = device
            self.n_lf = 5
            self.eps_lf = 1e-3
            self.beta_zero = 1.0
            self.metric_fc = 64  # Hidden dimension for metric network
            self.temperature = 1.0
            self.regularization = 1e-4
    
    args = Args()
    
    # Load real Sprites data
    print("📂 Loading real Sprites data...")
    sprites_data = torch.load('data/processed/Sprites_train_cyclic.pt', map_location=device)
    
    # Resize from 28x28 to 64x64
    if sprites_data.shape[-1] == 28:
        import torch.nn.functional as F
        sprites_data = F.interpolate(sprites_data.view(-1, *sprites_data.shape[2:]), 
                                   size=(64, 64), mode='bilinear', align_corners=False)
        sprites_data = sprites_data.view(sprites_data.shape[0], -1, *sprites_data.shape[1:])
    
    # Use subset for training
    sprites_subset = sprites_data[:200]
    flattened = sprites_subset.view(sprites_subset.shape[0] * sprites_subset.shape[1], *sprites_subset.shape[2:])
    print(f"   Data shape: {flattened.shape}")
    
    # Create RHVAE model
    model = RHVAE(args)
    model.to(device)
    
    print(f"✅ Model created")
    print(f"   Metric parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train the model and collect data
    print(f"\n🎓 TRAINING MODEL AND COLLECTING DATA:")
    print("-" * 40)
    
    batch_size = 32
    n_batches = len(flattened) // batch_size
    
    # Collect training data
    all_z = []
    all_mu = []
    all_L = []
    all_M = []
    all_G_inv = []
    all_det_G_inv = []
    
    model.train()
    
    for batch_idx in range(min(10, n_batches)):  # Process 10 batches
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_data = flattened[start_idx:end_idx]
        
        # Forward pass
        output = model(batch_data)
        recon_x, z, z0, rho, eps0, gamma, mu, log_var, G_inv, G_log_det = output
        
        # Get L and M matrices
        L, M = model.metric_forward(batch_data)
        
        # Store data
        all_z.append(z.detach().cpu())
        all_mu.append(mu.detach().cpu())
        all_L.append(L.detach().cpu())
        all_M.append(M.detach().cpu())
        all_G_inv.append(G_inv.detach().cpu())
        all_det_G_inv.append(torch.exp(-G_log_det).detach().cpu())
        
        print(f"   Batch {batch_idx + 1}: z shape {z.shape}, det(G⁻¹) range [{torch.exp(-G_log_det).min():.1e}, {torch.exp(-G_log_det).max():.1e}]")
    
    # Update metric
    model.update_metric()
    print(f"   ✅ Metric updated")
    
    # Concatenate all data
    all_z = torch.cat(all_z, dim=0)
    all_mu = torch.cat(all_mu, dim=0)
    all_L = torch.cat(all_L, dim=0)
    all_M = torch.cat(all_M, dim=0)
    all_G_inv = torch.cat(all_G_inv, dim=0)
    all_det_G_inv = torch.cat(all_det_G_inv, dim=0)
    
    print(f"   Total samples: {len(all_z)}")
    print(f"   z range: [{all_z.min():.3f}, {all_z.max():.3f}]")
    print(f"   det(G⁻¹) range: [{all_det_G_inv.min():.1e}, {all_det_G_inv.max():.1e}]")
    
    # Create output directory
    output_dir = "original_rhvae_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🎨 CREATING VISUALIZATIONS:")
    print("-" * 30)
    
    # 1. LATENT SPACE VISUALIZATION (2D projection)
    print("   1. Creating latent space visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Latent points (first 2 dimensions)
    ax1 = axes[0, 0]
    z_2d = all_z[:, :2].numpy()
    ax1.scatter(z_2d[:, 0], z_2d[:, 1], c='blue', s=20, alpha=0.6, label='Latent Points')
    ax1.set_xlabel('z₁ (first dimension)')
    ax1.set_ylabel('z₂ (second dimension)')
    ax1.set_title('1. Latent Space (2D projection)\nOriginal RHVAE', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Encoder means
    ax2 = axes[0, 1]
    mu_2d = all_mu[:, :2].numpy()
    ax2.scatter(mu_2d[:, 0], mu_2d[:, 1], c='red', s=20, alpha=0.6, label='Encoder Means')
    ax2.set_xlabel('μ₁ (first dimension)')
    ax2.set_ylabel('μ₂ (second dimension)')
    ax2.set_title('2. Encoder Means (2D projection)\nCentroids for metric', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Determinant distribution
    ax3 = axes[1, 0]
    ax3.hist(all_det_G_inv.numpy(), bins=30, alpha=0.7, color='green', edgecolor='black')
    ax3.set_xlabel('det(G⁻¹)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('3. Determinant Distribution\nTraining mode', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Determinant vs position
    ax4 = axes[1, 1]
    scatter = ax4.scatter(z_2d[:, 0], z_2d[:, 1], c=all_det_G_inv.numpy(), 
                          s=30, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ax=ax4, label='det(G⁻¹)')
    ax4.set_xlabel('z₁ (first dimension)')
    ax4.set_ylabel('z₂ (second dimension)')
    ax4.set_title('4. Determinant Heatmap\nColored by det(G⁻¹)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_latent_space_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. METRIC MATRICES ANALYSIS
    print("   2. Creating metric matrices analysis...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: L matrix determinants
    ax1 = axes[0, 0]
    det_L = torch.det(all_L).numpy()
    ax1.hist(det_L, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('det(L)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('1. L Matrix Determinants\nLower triangular matrices', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: M matrix determinants
    ax2 = axes[0, 1]
    det_M = torch.det(all_M).numpy()
    ax2.hist(det_M, bins=20, alpha=0.7, color='red', edgecolor='black')
    ax2.set_xlabel('det(M)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('2. M Matrix Determinants\nM = L L^T', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sample L matrix
    ax3 = axes[0, 2]
    sample_L = all_L[0].numpy()
    im = ax3.imshow(sample_L, cmap='viridis', alpha=0.8)
    ax3.set_title(f'3. Sample L Matrix\ndet = {det_L[0]:.2f}', fontweight='bold')
    plt.colorbar(im, ax=ax3)
    
    # Plot 4: Sample M matrix
    ax4 = axes[1, 0]
    sample_M = all_M[0].numpy()
    im = ax4.imshow(sample_M, cmap='viridis', alpha=0.8)
    ax4.set_title(f'4. Sample M Matrix\ndet = {det_M[0]:.2f}', fontweight='bold')
    plt.colorbar(im, ax=ax4)
    
    # Plot 5: Sample G⁻¹ matrix
    ax5 = axes[1, 1]
    sample_G_inv = all_G_inv[0].numpy()
    im = ax5.imshow(sample_G_inv, cmap='viridis', alpha=0.8)
    ax5.set_title(f'5. Sample G⁻¹ Matrix\ndet = {all_det_G_inv[0]:.1e}', fontweight='bold')
    plt.colorbar(im, ax=ax5)
    
    # Plot 6: Correlation analysis
    ax6 = axes[1, 2]
    ax6.scatter(det_L, det_M, alpha=0.6, s=20)
    ax6.set_xlabel('det(L)')
    ax6.set_ylabel('det(M)')
    ax6.set_title('6. L vs M Determinants\nCorrelation analysis', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_metric_matrices_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. METRIC FORMULA VISUALIZATION
    print("   3. Creating metric formula visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Weight computation
    ax1 = axes[0, 0]
    # Compute weights for a test point
    test_point = torch.zeros(16, device=device)
    test_point[:2] = torch.tensor([0.0, 0.0], device=device)
    
    # Compute distances to centroids
    distances = torch.norm(test_point.unsqueeze(0) - all_mu.to(device), dim=1)
    weights = torch.exp(-distances ** 2 / (args.temperature ** 2))
    weights = weights / weights.sum()
    
    # Plot weight distribution
    centroid_indices = np.arange(len(all_mu))
    ax1.bar(centroid_indices[:20], weights[:20].cpu().numpy(), alpha=0.7)
    ax1.set_xlabel('Centroid Index')
    ax1.set_ylabel('Weight')
    ax1.set_title('1. Weight Distribution\nw(z) = exp(-||z - c||² / T²)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distance vs weight
    ax2 = axes[0, 1]
    ax2.scatter(distances.cpu().numpy(), weights.cpu().numpy(), alpha=0.6, s=20)
    ax2.set_xlabel('Distance to Centroid')
    ax2.set_ylabel('Weight')
    ax2.set_title('2. Distance vs Weight\nGaussian kernel', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Metric construction
    ax3 = axes[1, 0]
    # Show the formula
    formula_text = r"$G^{-1}(z) = \sum_i w_i(z) M_i + \lambda I$" + "\n" + \
                   r"$w_i(z) = \exp(-\frac{||z - c_i||^2}{T^2})$" + "\n" + \
                   r"$M_i = L_i L_i^T$"
    ax3.text(0.1, 0.5, formula_text, transform=ax3.transAxes, fontsize=14, 
             verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax3.set_title('3. Parametric Metric Formula\nOriginal RHVAE', fontweight='bold')
    ax3.axis('off')
    
    # Plot 4: Eigenvalue analysis
    ax4 = axes[1, 1]
    # Compute eigenvalues of G⁻¹ matrices
    eigenvals = []
    for G_inv in all_G_inv[:50]:  # Use first 50 for speed
        eigenvals.extend(torch.linalg.eigvals(G_inv).real.numpy())
    
    ax4.hist(eigenvals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax4.set_xlabel('Eigenvalues of G⁻¹')
    ax4.set_ylabel('Frequency')
    ax4.set_title('4. Eigenvalue Distribution\nPositive definiteness check', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_metric_formula_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. COMPARISON WITH UPDATED METRIC
    print("   4. Creating comparison with updated metric...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Test updated metric
    model.eval()
    z_test = torch.randn(100, args.latent_dim, device=device)
    
    # Get updated metric values
    G_inv_updated = model.G_inv(z_test)
    det_G_inv_updated = torch.det(G_inv_updated)
    
    # Plot 1: Training vs Updated determinants
    ax1 = axes[0, 0]
    ax1.hist(all_det_G_inv.numpy(), bins=20, alpha=0.7, label='Training', color='blue')
    ax1.hist(det_G_inv_updated.cpu().numpy(), bins=20, alpha=0.7, label='Updated', color='red')
    ax1.set_xlabel('det(G⁻¹)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('1. Training vs Updated\nDeterminant Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Updated metric heatmap
    ax2 = axes[0, 1]
    z_test_2d = z_test[:, :2].cpu().numpy()
    scatter = ax2.scatter(z_test_2d[:, 0], z_test_2d[:, 1], 
                          c=det_G_inv_updated.cpu().numpy(), s=30, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ax=ax2, label='det(G⁻¹)')
    ax2.set_xlabel('z₁ (first dimension)')
    ax2.set_ylabel('z₂ (second dimension)')
    ax2.set_title('2. Updated Metric Heatmap\nAfter metric update', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sample updated G⁻¹ matrix
    ax3 = axes[1, 0]
    sample_G_inv_updated = G_inv_updated[0].cpu().numpy()
    im = ax3.imshow(sample_G_inv_updated, cmap='viridis', alpha=0.8)
    ax3.set_title(f'3. Updated G⁻¹ Matrix\ndet = {det_G_inv_updated[0]:.1e}', fontweight='bold')
    plt.colorbar(im, ax=ax3)
    
    # Plot 4: Eigenvalue comparison
    ax4 = axes[1, 1]
    eigenvals_training = []
    for G_inv in all_G_inv[:20]:
        eigenvals_training.extend(torch.linalg.eigvals(G_inv).real.numpy())
    
    eigenvals_updated = []
    for G_inv in G_inv_updated[:20]:
        eigenvals_updated.extend(torch.linalg.eigvals(G_inv).real.cpu().numpy())
    
    ax4.hist(eigenvals_training, bins=20, alpha=0.7, label='Training', color='blue')
    ax4.hist(eigenvals_updated, bins=20, alpha=0.7, label='Updated', color='red')
    ax4.set_xlabel('Eigenvalues')
    ax4.set_ylabel('Frequency')
    ax4.set_title('4. Eigenvalue Comparison\nPositive definiteness', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_comparison_updated_metric.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. SUMMARY STATISTICS
    print("   5. Creating summary statistics...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Key statistics
    ax1 = axes[0]
    stats = {
        'Samples': len(all_z),
        'Latent Dim': args.latent_dim,
        'Mean det(L)': det_L.mean(),
        'Mean det(M)': det_M.mean(),
        'Mean det(G⁻¹)': all_det_G_inv.mean().item(),
        'Temperature': args.temperature,
        'Regularization': args.regularization
    }
    
    y_pos = np.arange(len(stats))
    ax1.barh(y_pos, [1]*len(stats), color='lightblue', alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(list(stats.keys()))
    ax1.set_xlim(0, 1.2)
    ax1.set_title('Key Statistics', fontweight='bold')
    
    # Add value labels
    for i, (key, value) in enumerate(stats.items()):
        ax1.text(1.05, i, f'{value:.3f}', va='center', fontweight='bold')
    
    # Plot 2: Determinant ranges
    ax2 = axes[1]
    ranges = {
        'det(L)': [det_L.min(), det_L.max()],
        'det(M)': [det_M.min(), det_M.max()],
        'det(G⁻¹)': [all_det_G_inv.min().item(), all_det_G_inv.max().item()]
    }
    
    x_pos = np.arange(len(ranges))
    widths = [ranges[key][1] - ranges[key][0] for key in ranges.keys()]
    ax2.bar(x_pos, widths, alpha=0.7, color=['blue', 'red', 'green'])
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(list(ranges.keys()))
    ax2.set_ylabel('Range (max - min)')
    ax2.set_title('Determinant Ranges', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_summary_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ All visualizations saved to {output_dir}/")
    print(f"📊 Generated {len(os.listdir(output_dir))} comprehensive analysis graphs")
    
    return output_dir

if __name__ == "__main__":
    output_folder = create_rhvae_visualizations() 