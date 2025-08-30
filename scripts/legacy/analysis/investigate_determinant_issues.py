#!/usr/bin/env python3
"""
Investigate Determinant Issues
=============================

Investigate why the determinant values are so small and test different approaches
to improve the metric computation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def investigate_metric_matrices():
    """Investigate the metric matrices computed by retrieveG."""
    print("🔍 Investigating Metric Matrix Issues")
    print("=" * 50)
    
    # Load the trained model and data
    from train_vanilla_vae_real_data import train_vanilla_vae, manual_retrieveG
    
    # Train a fresh model (or load existing)
    model, dataset, device = train_vanilla_vae()
    
    # Get the metric matrices
    G_sampl, centroids, latent_data = manual_retrieveG(
        model, dataset, device, num_centroids=50, T_multiplier=1.0
    )
    
    print(f"\n📊 Metric Matrix Analysis:")
    print(f"   Latent data shape: {latent_data.shape}")
    print(f"   Centroids shape: {centroids.shape}")
    
    # Test the G_sampl function
    test_points = torch.randn(10, 16, device=device)
    G_matrices = G_sampl(test_points)
    
    print(f"   G matrices shape: {G_matrices.shape}")
    print(f"   G matrices range: [{G_matrices.min():.6f}, {G_matrices.max():.6f}]")
    
    # Analyze determinants
    dets = torch.linalg.det(G_matrices)
    print(f"   G determinants range: [{dets.min():.6e}, {dets.max():.6e}]")
    print(f"   G determinants mean: {dets.mean():.6e}")
    print(f"   G determinants std: {dets.std():.6e}")
    
    # Analyze eigenvalues
    eigenvals = torch.linalg.eigvals(G_matrices).real
    print(f"   Eigenvalues range: [{eigenvals.min():.6f}, {eigenvals.max():.6f}]")
    print(f"   Eigenvalues mean: {eigenvals.mean():.6f}")
    
    # Check for numerical issues
    print(f"\n🔧 Numerical Analysis:")
    print(f"   G matrices has NaN: {torch.isnan(G_matrices).any()}")
    print(f"   G matrices has Inf: {torch.isinf(G_matrices).any()}")
    print(f"   G matrices is symmetric: {torch.allclose(G_matrices, G_matrices.transpose(-2, -1))}")
    
    # Test different regularization values
    print(f"\n🧪 Testing Different Regularization Values:")
    reg_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    for reg in reg_values:
        # Create a simple test metric with different regularization
        test_G = torch.eye(16, device=device).unsqueeze(0).repeat(10, 1, 1) + reg
        test_det = torch.linalg.det(test_G)
        print(f"   Regularization {reg:.3f}: det(G) = {test_det.mean():.6e}")
    
    return G_matrices, dets, eigenvals

def test_improved_metric_computation():
    """Test improved metric computation with better numerical stability."""
    print(f"\n🚀 Testing Improved Metric Computation")
    print("=" * 50)
    
    # Load the trained model and data
    from train_vanilla_vae_real_data import train_vanilla_vae
    
    model, dataset, device = train_vanilla_vae()
    
    # Set model to eval mode
    model.eval()
    
    # Get latent representations
    with torch.no_grad():
        latent_data = []
        for i in range(0, len(dataset), 256):
            batch = dataset.data[i:i+256].to(device)
            output = model.encoder(batch)
            latent_data.append(output.embedding)
        
        latent_data = torch.cat(latent_data, dim=0)
        print(f"   Extracted latent data: {latent_data.shape}")
        
        # Compute centroids using k-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)
        kmeans.fit(latent_data.cpu().numpy())
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
        
        # Compute temperature
        T_is = []
        for i in range(len(centroids)-1):
            mask = torch.tensor([k for k in range(len(centroids)) if k != i])
            dist = torch.norm(centroids[i].unsqueeze(0) - centroids[mask], dim=-1)
            T_i = torch.min(dist, dim=0)[0]
            T_is.append(T_i.item())
        
        T = np.max(T_is) * 1.0
        print(f"   Computed temperature: {T:.3f}")
        
        # Create improved metric matrices with better numerical stability
        metric_matrices = []
        for i, centroid in enumerate(centroids):
            distances = torch.norm(latent_data - centroid, dim=1)
            closest_indices = torch.argsort(distances)[:100]
            cluster_points = latent_data[closest_indices]
            
            if len(cluster_points) > 1:
                # Compute covariance with better numerical stability
                mean_point = cluster_points.mean(dim=0, keepdim=True)
                centered_points = cluster_points - mean_point
                
                # Use more stable covariance computation
                cov_matrix = torch.matmul(centered_points.T, centered_points) / (len(centered_points) - 1)
                
                # Add stronger regularization for numerical stability
                cov_matrix += torch.eye(cov_matrix.shape[0], device=device) * 0.1
                
                try:
                    # Use more stable inverse computation
                    metric_matrix = torch.linalg.inv(cov_matrix)
                    
                    # Ensure positive definiteness
                    eigenvals = torch.linalg.eigvals(metric_matrix).real
                    if torch.any(eigenvals <= 0):
                        # Add regularization to make it positive definite
                        metric_matrix += torch.eye(metric_matrix.shape[0], device=device) * 0.1
                        
                except:
                    # Fallback to identity
                    metric_matrix = torch.eye(cluster_points.shape[1], device=device)
            else:
                metric_matrix = torch.eye(latent_data.shape[1], device=device)
            
            metric_matrices.append(metric_matrix)
        
        metric_matrices = torch.stack(metric_matrices)
        
        # Test the improved metric
        test_points = torch.randn(10, 16, device=device)
        
        # Create improved G function
        def improved_G_sampl(z):
            batch_size = z.shape[0]
            G = torch.zeros(batch_size, 16, 16, device=device)
            
            for i in range(batch_size):
                z_i = z[i:i+1]
                
                # Compute distances to centroids
                distances = torch.norm(z_i.unsqueeze(1) - centroids.unsqueeze(0), dim=2)
                
                # Compute weights with better numerical stability
                weights = torch.exp(-distances**2 / (T**2))
                weights = weights / (weights.sum() + 1e-8)  # Avoid division by zero
                
                # Interpolate metric matrices
                G_i = torch.zeros(16, 16, device=device)
                for j in range(len(centroids)):
                    G_i += weights[0, j] * metric_matrices[j]
                
                # Add stronger regularization
                G_i += torch.eye(16, device=device) * 0.1
                
                G[i] = G_i
            
            return G
        
        # Test the improved metric
        G_matrices = improved_G_sampl(test_points)
        dets = torch.linalg.det(G_matrices)
        eigenvals = torch.linalg.eigvals(G_matrices).real
        
        print(f"\n✅ Improved Metric Results:")
        print(f"   G matrices range: [{G_matrices.min():.6f}, {G_matrices.max():.6f}]")
        print(f"   G determinants range: [{dets.min():.6e}, {dets.max():.6e}]")
        print(f"   G determinants mean: {dets.mean():.6e}")
        print(f"   Eigenvalues range: [{eigenvals.min():.6f}, {eigenvals.max():.6f}]")
        
        return improved_G_sampl, centroids, latent_data

def create_comparison_visualization():
    """Create a visualization comparing different approaches."""
    print(f"\n🎨 Creating Comparison Visualization")
    
    # Test both approaches
    G_matrices_orig, dets_orig, eigenvals_orig = investigate_metric_matrices()
    improved_G_sampl, centroids, latent_data = test_improved_metric_computation()
    
    # Test improved metric
    device = next(improved_G_sampl.parameters()).device if hasattr(improved_G_sampl, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_points = torch.randn(10, 16, device=device)
    G_matrices_improved = improved_G_sampl(test_points)
    dets_improved = torch.linalg.det(G_matrices_improved)
    eigenvals_improved = torch.linalg.eigvals(G_matrices_improved).real
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Metric Computation Comparison", fontsize=16, fontweight='bold')
    
    # Plot 1: Determinant comparison
    ax1 = axes[0, 0]
    ax1.hist(dets_orig.cpu().numpy(), bins=20, alpha=0.7, label='Original', color='blue')
    ax1.hist(dets_improved.cpu().numpy(), bins=20, alpha=0.7, label='Improved', color='red')
    ax1.set_xlabel('det(G)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Determinant Distribution')
    ax1.legend()
    ax1.set_yscale('log')
    
    # Plot 2: Eigenvalue comparison
    ax2 = axes[0, 1]
    ax2.hist(eigenvals_orig.cpu().numpy().flatten(), bins=20, alpha=0.7, label='Original', color='blue')
    ax2.hist(eigenvals_improved.cpu().numpy().flatten(), bins=20, alpha=0.7, label='Improved', color='red')
    ax2.set_xlabel('Eigenvalues')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Eigenvalue Distribution')
    ax2.legend()
    
    # Plot 3: Matrix values comparison
    ax3 = axes[1, 0]
    ax3.hist(G_matrices_orig.cpu().numpy().flatten(), bins=20, alpha=0.7, label='Original', color='blue')
    ax3.hist(G_matrices_improved.cpu().numpy().flatten(), bins=20, alpha=0.7, label='Improved', color='red')
    ax3.set_xlabel('G matrix values')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Matrix Values Distribution')
    ax3.legend()
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    Original Approach:
    - det(G) range: [{dets_orig.min():.3e}, {dets_orig.max():.3e}]
    - det(G) mean: {dets_orig.mean():.3e}
    - Eigenvalues range: [{eigenvals_orig.min():.3f}, {eigenvals_orig.max():.3f}]
    
    Improved Approach:
    - det(G) range: [{dets_improved.min():.3e}, {dets_improved.max():.3e}]
    - det(G) mean: {dets_improved.mean():.3e}
    - Eigenvalues range: [{eigenvals_improved.min():.3f}, {eigenvals_improved.max():.3f}]
    
    Improvement Factor:
    - det(G) mean: {dets_improved.mean() / dets_orig.mean():.1f}x
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("metric_comparison_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Comparison visualization saved as 'metric_comparison_analysis.png'")

if __name__ == "__main__":
    create_comparison_visualization() 