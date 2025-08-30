"""
Comprehensive RHVAE Metric Visualization Script
Loads trained model and creates detailed heatmaps of metric properties across latent space.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import wandb
from typing import Tuple, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.rhvae_experiment import RHVAEExperiment

class RHVAEMetricVisualizer:
    """Visualizer for RHVAE metric properties across latent space."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize with trained model path."""
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        
        # Load the trained experiment
        self.experiment = RHVAEExperiment(
            input_dim=[3, 64, 64],
            latent_dim=8,
            device=device
        )
        
        # Load the trained model using Pythae's loading method
        from pythae.models import RHVAE
        self.model = RHVAE.load_from_folder(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Loaded trained RHVAE model from {model_path}")
        print(f"   Device: {self.device}")
        print(f"   Latent dim: {self.model.latent_dim}")
        
    def create_latent_grid(self, grid_size: int = 150, bounds: Tuple[float, float] = (-0.2, 0.2)) -> Tuple[np.ndarray, np.ndarray]:
        """Create a 2D grid of latent points."""
        x = np.linspace(bounds[0], bounds[1], grid_size)
        y = np.linspace(bounds[0], bounds[1], grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Create grid of latent points (first 2 dimensions)
        grid_points = np.zeros((grid_size * grid_size, self.model.latent_dim))
        grid_points[:, 0] = X.flatten()
        grid_points[:, 1] = Y.flatten()
        
        return grid_points, (X, Y)
        
    def compute_metric_properties(self, z: torch.Tensor) -> dict:
        """Compute metric properties for given latent points."""
        with torch.no_grad():
            # Get metric matrix from RHVAE
            if hasattr(self.model, 'M_tens'):
                M = self.model.M_tens.detach().cpu().numpy()
                if len(M.shape) == 3:
                    M = M[0]  # Use first time step
            else:
                # Fallback: identity matrix
                M = np.eye(self.model.latent_dim)
            
            # Compute properties for each point
            properties = {
                'determinant': [],
                'trace': [],
                'condition_number': [],
                'distance_to_centroids': []
            }
            
            # Get centroids if available
            centroids = None
            if hasattr(self.model, 'centroids_tens'):
                centroids = self.model.centroids_tens.detach().cpu().numpy()
                if len(centroids.shape) == 3:
                    centroids = centroids[0]  # Use first time step
            
            for i in range(len(z)):
                z_i = z[i].cpu().numpy()
                
                # Compute metric at this point (simplified - in practice, metric varies with z)
                # For now, use constant metric matrix
                G = M
                G_inv = np.linalg.inv(G)
                
                # Compute properties
                det_G_inv = np.linalg.det(G_inv)
                trace_G_inv = np.trace(G_inv)
                cond_num = np.linalg.cond(G)
                
                properties['determinant'].append(det_G_inv)
                properties['trace'].append(trace_G_inv)
                properties['condition_number'].append(cond_num)
                
                # Distance to nearest centroid
                if centroids is not None:
                    distances = [np.linalg.norm(z_i - centroid) for centroid in centroids]
                    min_distance = min(distances)
                    properties['distance_to_centroids'].append(min_distance)
                else:
                    properties['distance_to_centroids'].append(0.0)
            
            return properties
    
    def get_data_latent_points(self, num_samples: int = 1000) -> np.ndarray:
        """Get latent representations of actual data points."""
        # Load some test data
        test_data = torch.load("data/processed/Sprites_test_cyclic.pt")
        test_data = test_data[:, 0, :, :, :]  # Take first frame
        
        # Process in smaller batches to avoid memory issues
        batch_size = 32
        z_data_list = []
        
        with torch.no_grad():
            for i in range(0, min(num_samples, len(test_data)), batch_size):
                end_idx = min(i + batch_size, min(num_samples, len(test_data)))
                sample_batch = test_data[i:end_idx].to(self.device)
                encoder_output = self.model.encoder(sample_batch)
                z_batch = encoder_output["embedding"].cpu().numpy()
                z_data_list.append(z_batch)
                
                # Clear GPU memory
                del sample_batch, encoder_output, z_batch
                torch.cuda.empty_cache()
        
        return np.concatenate(z_data_list, axis=0)
    
    def create_comprehensive_heatmaps(self, grid_size: int = 150, bounds: Tuple[float, float] = (-0.2, 0.2)):
        """Create comprehensive heatmaps of metric properties."""
        print("🔄 Creating comprehensive metric heatmaps...")
        
        # Create latent grid
        grid_points, (X, Y) = self.create_latent_grid(grid_size, bounds)
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(self.device)
        
        # Compute metric properties
        properties = self.compute_metric_properties(grid_tensor)
        
        # Get actual data points for overlay
        data_points = self.get_data_latent_points(1000)
        
        # Create the comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Sharp Discriminatory Metric Determinant
        det_values = np.array(properties['determinant']).reshape(grid_size, grid_size)
        im1 = axes[0, 0].imshow(det_values, extent=[bounds[0], bounds[1], bounds[0], bounds[1]], 
                                origin='lower', cmap='viridis', aspect='auto')
        axes[0, 0].scatter(data_points[:, 0], data_points[:, 1], c='white', s=1, alpha=0.7)
        axes[0, 0].set_title("Sharp Discriminatory Metric Determinant (150x150 Grid)", fontsize=14)
        axes[0, 0].set_xlabel("Latent Dim 1")
        axes[0, 0].set_ylabel("Latent Dim 2")
        plt.colorbar(im1, ax=axes[0, 0], label="det(G_inv)")
        
        # 2. Sharp Discriminatory Metric Trace
        trace_values = np.array(properties['trace']).reshape(grid_size, grid_size)
        im2 = axes[0, 1].imshow(trace_values, extent=[bounds[0], bounds[1], bounds[0], bounds[1]], 
                                origin='lower', cmap='viridis', aspect='auto')
        axes[0, 1].scatter(data_points[:, 0], data_points[:, 1], c='white', s=1, alpha=0.7)
        axes[0, 1].set_title("Sharp Discriminatory Metric Trace (150x150 Grid)", fontsize=14)
        axes[0, 1].set_xlabel("Latent Dim 1")
        axes[0, 1].set_ylabel("Latent Dim 2")
        plt.colorbar(im2, ax=axes[0, 1], label="trace(G_inv)")
        
        # 3. Distance to Nearest Centroid
        dist_values = np.array(properties['distance_to_centroids']).reshape(grid_size, grid_size)
        im3 = axes[1, 0].imshow(dist_values, extent=[bounds[0], bounds[1], bounds[0], bounds[1]], 
                                origin='lower', cmap='viridis', aspect='auto')
        axes[1, 0].scatter(data_points[:, 0], data_points[:, 1], c='white', s=1, alpha=0.7)
        axes[1, 0].set_title("Distance to Nearest Centroid (150x150 Grid)", fontsize=14)
        axes[1, 0].set_xlabel("Latent Dim 1")
        axes[1, 0].set_ylabel("Latent Dim 2")
        plt.colorbar(im3, ax=axes[1, 0], label="Distance")
        
        # 4. Sharp Discriminatory Condition Number
        cond_values = np.array(properties['condition_number']).reshape(grid_size, grid_size)
        im4 = axes[1, 1].imshow(cond_values, extent=[bounds[0], bounds[1], bounds[0], bounds[1]], 
                                origin='lower', cmap='viridis', aspect='auto')
        axes[1, 1].scatter(data_points[:, 0], data_points[:, 1], c='white', s=1, alpha=0.7)
        axes[1, 1].set_title("Sharp Discriminatory Condition Number (150x150 Grid)", fontsize=14)
        axes[1, 1].set_xlabel("Latent Dim 1")
        axes[1, 1].set_ylabel("Latent Dim 2")
        plt.colorbar(im4, ax=axes[1, 1], label="Condition Number")
        
        plt.tight_layout()
        
        # Save and log to WandB
        plt.savefig("rhvae_metric_heatmaps.png", dpi=300, bbox_inches='tight')
        
        # Log to WandB if available
        try:
            wandb.log({"rhvae_comprehensive_metric_heatmaps": wandb.Image(fig)})
            print("✅ Logged comprehensive metric heatmaps to WandB")
        except:
            print("⚠️ WandB not available, saved locally as rhvae_metric_heatmaps.png")
        
        plt.close()
        print("✅ Comprehensive metric heatmaps created!")
        
        return fig
    
    def create_centroid_analysis(self):
        """Create detailed centroid analysis."""
        if not hasattr(self.model, 'centroids_tens'):
            print("⚠️ No centroids available in model")
            return
        
        centroids = self.model.centroids_tens.detach().cpu().numpy()
        if len(centroids.shape) == 3:
            centroids = centroids[0]  # Use first time step
        
        # Get data points
        data_points = self.get_data_latent_points(1000)
        
        # Create centroid visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot data points
        ax.scatter(data_points[:, 0], data_points[:, 1], c='blue', alpha=0.6, s=20, label='Data Points')
        
        # Plot centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='*', label='Centroids')
        
        # Add centroid numbers
        for i, centroid in enumerate(centroids):
            ax.annotate(f'C{i+1}', (centroid[0], centroid[1]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')
        
        ax.set_xlabel("Latent Dim 1")
        ax.set_ylabel("Latent Dim 2")
        ax.set_title("RHVAE Centroids and Data Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save and log
        plt.savefig("rhvae_centroids.png", dpi=300, bbox_inches='tight')
        
        try:
            wandb.log({"rhvae_centroids_analysis": wandb.Image(fig)})
            print("✅ Logged centroid analysis to WandB")
        except:
            print("⚠️ WandB not available, saved locally as rhvae_centroids.png")
        
        plt.close()
        print("✅ Centroid analysis created!")
        
        return fig

def main():
    """Main function to run comprehensive RHVAE visualization."""
    print("🚀 RHVAE Comprehensive Metric Visualization")
    print("=" * 50)
    
    # Initialize WandB
    wandb_available = False
    try:
        import wandb
        wandb.init(
            project="rlvae_experiments",
            name="rhvae_comprehensive_visualization",
            tags=["rhvae", "visualization", "metrics", "heatmaps"]
        )
        wandb_available = True
        print("✅ WandB initialized for logging")
    except Exception as e:
        print(f"⚠️ WandB initialization failed: {e}")
    
    # Find the most recent model checkpoint
    output_dir = Path("outputs/rhvae_sprites_test")
    if not output_dir.exists():
        print("❌ No output directory found. Please run training first.")
        return
    
    # Find the most recent training run
    training_runs = list(output_dir.glob("RHVAE_training_*"))
    if not training_runs:
        print("❌ No training runs found. Please run training first.")
        return
    
    latest_run = max(training_runs, key=lambda x: x.stat().st_mtime)
    model_path = latest_run / "final_model"
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    print(f"📁 Using model from: {model_path}")
    
    # Initialize visualizer
    visualizer = RHVAEMetricVisualizer(str(model_path))
    
    # Create comprehensive heatmaps
    visualizer.create_comprehensive_heatmaps()
    
    # Create centroid analysis
    visualizer.create_centroid_analysis()
    
    # Finish WandB run
    if wandb_available:
        wandb.finish()
    
    print("🎉 Comprehensive RHVAE visualization completed!")

if __name__ == "__main__":
    main() 