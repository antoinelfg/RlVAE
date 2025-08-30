#!/usr/bin/env python3
"""
Enhanced KL Visualization with Real Data
========================================
Creates comprehensive visualizations of the enhanced KL mechanism using real data
from the test results, showing metric evolution, posterior adaptation, and KL dynamics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import wandb
from pathlib import Path
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from original_rlvae.src.models.riemannian_flow_vae import RiemannianFlowVAE
from original_rlvae.src.data.cyclic_dataset import CyclicSpritesDataset
from torch.utils.data import DataLoader

def create_enhanced_kl_visualization():
    """Create comprehensive visualization of enhanced KL mechanism with real data."""
    
    print("🎨 Creating Enhanced KL Visualization with Real Data")
    print("=" * 60)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Load real data
    print("\n1️⃣ Loading real data...")
    train_dataset = CyclicSpritesDataset(
        data_path="data/processed/Sprites_train_cyclic.pt",
        subset_size=1000  # Use subset for visualization
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(f"✅ Loaded {len(train_dataset)} samples")
    
    # Create model with enhanced KL
    print("\n2️⃣ Creating model with enhanced KL...")
    model = RiemannianFlowVAE(
        latent_dim=16,
        input_dim=(3, 64, 64),
        n_flows=4,
        flow_type="planar",
        flow_hidden_dims=[64, 64],
        beta=1.0,
        riemannian_beta=1.0,
        posterior_type="riemannian_metric",
        riemannian_kl_mode="sample_logq_logp",
        temperature=0.1,
        lbd=0.01,
        n_centroids=50,
        adaptive_kl_enabled=True,
        adaptive_kl_ramp_up_steps=20,
        adaptive_kl_alignment_weight=0.15,
        update_metric_during_training=True,
        metric_update_frequency=5,
        metric_update_alpha=0.01,
        metric_update_temperature=0.1,
        metric_update_regularization=0.01,
        device=device
    )
    
    # Load pretrained components
    print("\n3️⃣ Loading pretrained components...")
    model.load_pretrained_metrics("data/pretrained/metric_diverse_mlp_ld16_20250820_112010.pt")
    
    # Load encoder/decoder weights
    encoder_state = torch.load("data/pretrained/encoder_diverse_mlp_ld16_20250820_112008.pt", map_location=device)
    decoder_state = torch.load("data/pretrained/decoder_diverse_mlp_ld16_20250820_112008.pt", map_location=device)
    
    model.encoder.load_state_dict(encoder_state)
    model.decoder.load_state_dict(decoder_state)
    
    # Move model to device
    model = model.to(device)
    print("✅ Loaded all pretrained components and moved to device")
    
    # Initialize tracking variables
    metric_evolution = []
    kl_losses = []
    beta_values = []
    centroid_changes = []
    metric_changes = []
    posterior_samples = []
    
    print("\n4️⃣ Running enhanced KL simulation...")
    
    # Simulate training steps with metric updates
    for step in range(50):  # 50 steps to show evolution
        # Get batch of real data
        try:
            batch = next(iter(train_loader))
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
        except StopIteration:
            # Recreate iterator if exhausted
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            batch = next(iter(train_loader))
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
        
        x = x.to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
            
        # Track KL loss
        kl_loss = output.kld_loss.item() if hasattr(output, 'kld_loss') else 0.0
        kl_losses.append(kl_loss)
        
        # Track beta evolution
        beta_values.append(model.riemannian_beta)
        
        # Store current metric state
        current_centroids = model.centroids_tens.clone().detach()
        current_metric = model.M_tens.clone().detach()
        
        # Perform metric update every 5 steps
        if step % 5 == 0 and step > 0:
            # Simulate metric update
            model._metric_update_counter = 4  # Trigger update
            model._adapt_kl_loss_for_metric_update()
            
            # Calculate changes
            new_centroids = model.centroids_tens.clone().detach()
            new_metric = model.M_tens.clone().detach()
            
            centroid_change = torch.norm(new_centroids - current_centroids).item()
            metric_change = torch.norm(new_metric - current_metric).item()
            
            centroid_changes.append(centroid_change)
            metric_changes.append(metric_change)
            
            print(f"   Step {step}: β={model.riemannian_beta:.3f}, KL={kl_loss:.3f}, "
                  f"Δcentroids={centroid_change:.3f}, Δmetric={metric_change:.3f}")
        else:
            centroid_changes.append(0.0)
            metric_changes.append(0.0)
        
        # Store metric evolution
        metric_evolution.append({
            'step': step,
            'centroids': current_centroids.cpu().numpy(),
            'metric': current_metric.cpu().numpy(),
            'beta': model.riemannian_beta,
            'kl_loss': kl_loss
        })
        
        # Sample from posterior occasionally
        if step % 10 == 0:
            with torch.no_grad():
                # Get real mu and log_var from encoder
                encoder_out = model.encoder(x[:, 0])  # Use first frame
                if hasattr(encoder_out, 'reparameterization'):
                    mu = encoder_out.reparameterization.mu
                    log_var = encoder_out.reparameterization.log_var
                elif hasattr(encoder_out, 'mu') and hasattr(encoder_out, 'log_var'):
                    mu = encoder_out.mu
                    log_var = encoder_out.log_var
                else:
                    # Extract from ModelOutput
                    mu = encoder_out['mu'] if 'mu' in encoder_out else torch.randn(32, 16, device=device)
                    log_var = encoder_out['log_var'] if 'log_var' in encoder_out else torch.zeros(32, 16, device=device)
                
                # Sample from metric-aware posterior
                sample = model.sample_metric_aware_posterior(mu, log_var)
                posterior_samples.append(sample.cpu().numpy())
    
    print(f"✅ Completed {len(metric_evolution)} steps")
    
    # Create comprehensive visualization
    print("\n5️⃣ Creating visualizations...")
    
    # Set up the figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. KL Loss Evolution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(kl_losses, 'b-', linewidth=2, alpha=0.8)
    ax1.set_title('KL Loss Evolution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('KL Loss')
    ax1.grid(True, alpha=0.3)
    
    # 2. Beta Evolution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(beta_values, 'r-', linewidth=2, alpha=0.8)
    ax2.set_title('Adaptive Beta Evolution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Riemannian Beta')
    ax2.grid(True, alpha=0.3)
    
    # 3. Metric Changes
    ax3 = fig.add_subplot(gs[0, 2])
    steps = list(range(len(metric_changes)))
    ax3.bar(steps, metric_changes, alpha=0.7, color='green')
    ax3.set_title('Metric Update Changes', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Update Step')
    ax3.set_ylabel('Metric Change Magnitude')
    ax3.grid(True, alpha=0.3)
    
    # 4. Centroid Evolution (2D projection)
    ax4 = fig.add_subplot(gs[1, :])
    for i, step_data in enumerate(metric_evolution[::5]):  # Every 5th step
        centroids = step_data['centroids']
        # PCA to 2D for visualization
        if centroids.shape[0] > 1:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            centroids_2d = pca.fit_transform(centroids)
            alpha = 0.3 + 0.7 * (i / len(metric_evolution[::5]))
            ax4.scatter(centroids_2d[:, 0], centroids_2d[:, 1], 
                       alpha=alpha, s=20, c=f'C{i}', label=f'Step {step_data["step"]}')
    ax4.set_title('Centroid Evolution (PCA 2D)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('PCA Component 1')
    ax4.set_ylabel('PCA Component 2')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 5. Metric Determinant Evolution
    ax5 = fig.add_subplot(gs[2, 0])
    det_evolution = []
    for step_data in metric_evolution:
        metric = step_data['metric']
        det = np.linalg.det(metric)
        det_evolution.append(det)
    ax5.plot(det_evolution, 'purple', linewidth=2, alpha=0.8)
    ax5.set_title('Metric Determinant Evolution', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Training Step')
    ax5.set_ylabel('Det(G)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Posterior Sample Distribution
    ax6 = fig.add_subplot(gs[2, 1])
    if posterior_samples:
        all_samples = np.concatenate(posterior_samples, axis=0)
        # PCA to 2D
        pca = PCA(n_components=2)
        samples_2d = pca.fit_transform(all_samples)
        ax6.scatter(samples_2d[:, 0], samples_2d[:, 1], alpha=0.6, s=10)
        ax6.set_title('Posterior Samples (PCA 2D)', fontsize=14, fontweight='bold')
        ax6.set_xlabel('PCA Component 1')
        ax6.set_ylabel('PCA Component 2')
    
    # 7. Metric Eigenvalue Distribution
    ax7 = fig.add_subplot(gs[2, 2])
    all_eigenvals = []
    for step_data in metric_evolution[::5]:
        metric = step_data['metric']
        eigenvals = np.linalg.eigvals(metric)
        all_eigenvals.extend(eigenvals.real)
    # Convert to numpy array and flatten
    all_eigenvals = np.array(all_eigenvals).flatten()
    ax7.hist(all_eigenvals, bins=30, alpha=0.7, color='orange')
    ax7.set_title('Metric Eigenvalue Distribution', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Eigenvalue')
    ax7.set_ylabel('Frequency')
    
    # 8. Enhanced KL Components
    ax8 = fig.add_subplot(gs[3, :])
    # Create a comprehensive view of all components
    x_vals = list(range(len(metric_evolution)))
    
    # Normalize for better visualization
    kl_norm = np.array(kl_losses) / max(kl_losses) if max(kl_losses) > 0 else np.array(kl_losses)
    beta_norm = np.array(beta_values) / max(beta_values) if max(beta_values) > 0 else np.array(beta_values)
    metric_norm = np.array(metric_changes) / max(metric_changes) if max(metric_changes) > 0 else np.array(metric_changes)
    
    ax8.plot(x_vals, kl_norm, 'b-', linewidth=2, label='KL Loss (normalized)', alpha=0.8)
    ax8.plot(x_vals, beta_norm, 'r-', linewidth=2, label='Beta (normalized)', alpha=0.8)
    ax8.bar(x_vals, metric_norm, alpha=0.5, color='green', label='Metric Changes (normalized)')
    
    ax8.set_title('Enhanced KL Mechanism Components', fontsize=16, fontweight='bold')
    ax8.set_xlabel('Training Step')
    ax8.set_ylabel('Normalized Values')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Add text summary
    fig.suptitle('Enhanced Riemannian KL Mechanism: Real Data Analysis', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Add summary statistics
    summary_text = f"""
    Summary Statistics:
    • Total Steps: {len(metric_evolution)}
    • Metric Updates: {len([x for x in metric_changes if x > 0])}
    • Final Beta: {beta_values[-1]:.3f}
    • Final KL Loss: {kl_losses[-1]:.3f}
    • Max Metric Change: {max(metric_changes):.3f}
    • Max Centroid Change: {max(centroid_changes):.3f}
    """
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Save the visualization
    output_path = "enhanced_kl_real_data_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}")
    
    # Log to WandB if available
    try:
        wandb.init(project="enhanced-kl-visualization", name="real-data-analysis")
        wandb.log({
            "enhanced_kl_visualization": wandb.Image(output_path),
            "final_kl_loss": kl_losses[-1],
            "final_beta": beta_values[-1],
            "total_metric_updates": len([x for x in metric_changes if x > 0]),
            "max_metric_change": max(metric_changes)
        })
        print("✅ Logged to WandB")
    except Exception as e:
        print(f"⚠️ WandB logging failed: {e}")
    
    plt.show()
    
    return {
        'kl_losses': kl_losses,
        'beta_values': beta_values,
        'metric_changes': metric_changes,
        'centroid_changes': centroid_changes,
        'metric_evolution': metric_evolution
    }

if __name__ == "__main__":
    # Create the visualization
    results = create_enhanced_kl_visualization()
    
    print("\n🎉 Enhanced KL Visualization Complete!")
    print("=" * 60)
    print("📊 Key Insights:")
    print(f"   • KL Loss Range: {min(results['kl_losses']):.3f} - {max(results['kl_losses']):.3f}")
    print(f"   • Beta Evolution: {min(results['beta_values']):.3f} → {max(results['beta_values']):.3f}")
    print(f"   • Metric Updates: {len([x for x in results['metric_changes'] if x > 0])}")
    print(f"   • Max Metric Change: {max(results['metric_changes']):.3f}")
    print(f"   • Max Centroid Change: {max(results['centroid_changes']):.3f}")
