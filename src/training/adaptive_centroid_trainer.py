#!/usr/bin/env python3
"""
Adaptive Centroid Trainer
==========================

Extends the modular training system with periodic centroid updates during Stage 2 training.
This implements your brilliant idea of updating centroids every N epochs to maintain 
manifold alignment throughout training.

Features:
- Periodic centroid recomputation based on current model latent distribution
- Manifold evolution tracking and visualization
- Seamless integration with existing pipeline and visualization system
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from omegaconf import DictConfig
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
import wandb
from tqdm import tqdm

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.training.train_with_modular_visualizations import CleanCyclicLoopTrainer
from src.visualizations.manager import VisualizationManager, VisualizationConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveCentroidTrainer(CleanCyclicLoopTrainer):
    """
    Enhanced trainer with adaptive centroid updates.
    
    Extends the base modular trainer to include:
    - Periodic centroid recomputation every N epochs
    - Manifold evolution tracking
    - Enhanced visualizations showing manifold changes
    """
    
    def __init__(self, config, project_name="adaptive-centroid-rlvae", run_name=None, 
                 centroid_update_frequency: int = 2, n_samples_for_centroids: int = 500, 
                 freeze_mode: bool = False, kl_controlled_mode: bool = True):
        """
        Initialize adaptive centroid trainer.
        
        Args:
            config: Training configuration
            project_name: WandB project name
            run_name: WandB run name
            centroid_update_frequency: Update centroids every N epochs
            n_samples_for_centroids: Number of samples to use for centroid computation
            freeze_mode: If True, analyze centroid evolution but don't update the model
            kl_controlled_mode: If True, perform real updates with KL divergence control
        """
        # Initialize base trainer
        super().__init__(config, project_name, run_name)
        
        # Adaptive centroid parameters
        self.centroid_update_frequency = centroid_update_frequency
        self.n_samples_for_centroids = n_samples_for_centroids
        self.freeze_mode = freeze_mode
        self.kl_controlled_mode = kl_controlled_mode
        
        # KL Control Parameters - More realistic thresholds for Riemannian VAE
        self.kl_stability_threshold = 50.0  # Maximum allowed KL before intervention (Riemannian KL can be higher)
        self.kl_growth_threshold = 3.0      # Maximum allowed KL growth rate (more tolerant)
        self.max_rollback_attempts = 3      # Maximum rollback attempts per update
        self.adaptive_alpha_min = 0.005     # Minimum interpolation rate (more conservative)
        self.adaptive_alpha_max = 0.2       # Maximum interpolation rate (less aggressive)
        self.current_alpha = 0.05           # Current interpolation rate (start conservative)
        
        # KL Monitoring History
        self.kl_history = []
        self.pre_update_kl = None
        self.stability_metrics = {
            'successful_updates': 0,
            'rollbacks': 0,
            'alpha_reductions': 0,
            'kl_stabilizations': 0
        }
        
        # Model State Management for Rollbacks
        self.pre_update_model_state = None
        self.pre_update_optimizer_state = None
        
        # Track centroid evolution
        self.centroid_history = []
        self.metric_history = []
        self.manifold_evolution_metrics = {
            'centroid_shifts': [],
            'metric_changes': [],
            'update_epochs': [],
            'latent_variance_evolution': [],
            'coverage_evolution': []
        }
        
        # Store original centroids for comparison (support multiple model types)
        self.centroids_accessor = None
        self.metric_accessor = None
        
        # Check for modular metric (ModularRiemannianFlowVAE)
        if hasattr(self.model, 'modular_metric') and hasattr(self.model.modular_metric, 'centroids'):
            self.original_centroids = self.model.modular_metric.centroids.clone().detach()
            self.original_metric_matrices = self.model.modular_metric.metric_matrices.clone().detach()
            self.centroids_accessor = 'modular_metric'
            logger.info(f"🎯 Found modular metric centroids: {self.original_centroids.shape}")
            
        # Check for direct metric tensors (RiemannianFlowVAE)
        elif hasattr(self.model, 'centroids_tens') and hasattr(self.model, 'M_tens'):
            self.original_centroids = self.model.centroids_tens.clone().detach()
            self.original_metric_matrices = self.model.M_tens.clone().detach()
            self.centroids_accessor = 'direct_tensors'
            logger.info(f"🎯 Found direct tensor centroids: {self.original_centroids.shape}")
            
        else:
            logger.warning("⚠️ Model doesn't have accessible centroids - adaptive updates disabled")
            logger.warning("   Expected: model.modular_metric.centroids OR model.centroids_tens")
            self.centroid_update_frequency = float('inf')  # Disable updates
        
        logger.info(f"🔄 Adaptive centroid trainer initialized")
        logger.info(f"📊 Update frequency: every {centroid_update_frequency} epochs")
        logger.info(f"🎯 Samples per update: {n_samples_for_centroids}")
    
    def should_update_centroids(self, epoch: int) -> bool:
        """Determine if centroids should be updated at this epoch."""
        # Don't update at epoch 0 to establish baseline
        return epoch > 0 and epoch % self.centroid_update_frequency == 0
    
    def extract_current_latent_distribution(self, data_loader, n_samples: int = None) -> np.ndarray:
        """Extract current latent distribution from the model."""
        if n_samples is None:
            n_samples = self.n_samples_for_centroids
            
        logger.info(f"📊 Extracting current latent distribution ({n_samples} samples)")
        
        self.model.eval()
        latent_representations = []
        
        extracted = 0
        with torch.no_grad():
            for batch in data_loader:
                if extracted >= n_samples:
                    break
                
                try:
                    batch = batch.to(self.device)
                    
                    # Handle sequence data - take first frame
                    if len(batch.shape) == 5:  # [B, seq_len, c, h, w]
                        x = batch[:, 0]  # First frame of all sequences in batch
                    elif len(batch.shape) == 4:  # [seq_len, c, h, w]
                        x = batch[0:1]  # First frame, add batch dim
                    else:
                        x = batch
                    
                    # Extract latent representation
                    encoder_out = self.model.encoder(x)
                    mu = encoder_out.embedding
                    
                    # Add all samples in batch
                    for i in range(mu.shape[0]):
                        if extracted < n_samples:
                            latent_representations.append(mu[i:i+1].cpu().numpy())
                            extracted += 1
                        else:
                            break
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process batch: {e}")
                    continue
        
        if latent_representations:
            latent_array = np.vstack(latent_representations)
            logger.info(f"✅ Successfully extracted {len(latent_array)} latent representations")
        else:
            raise RuntimeError("Failed to extract any latent representations")
        
        self.model.train()  # Return to training mode
        return latent_array
    
    def compute_new_centroids_and_metrics(self, latent_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Compute new centroids and metric matrices based on current latent distribution."""
        logger.info("🧠 Computing new centroids from current latent distribution...")
        
        # Get current number of centroids based on accessor type
        if self.centroids_accessor == 'modular_metric':
            n_centroids = len(self.model.modular_metric.centroids)
        elif self.centroids_accessor == 'direct_tensors':
            n_centroids = len(self.model.centroids_tens)
        else:
            raise RuntimeError("No valid centroids accessor found")
        
        # Ensure we have enough samples for clustering
        if len(latent_data) < n_centroids:
            logger.warning(f"   ⚠️  Only {len(latent_data)} samples for {n_centroids} centroids - using available samples")
            # Use all samples as centroids and pad with noise
            if len(latent_data) > 0:
                new_centroids = latent_data.copy()
                # Pad with small noise around existing samples if needed
                while len(new_centroids) < n_centroids:
                    # Add noisy copies of existing points
                    idx = len(new_centroids) % len(latent_data)
                    noisy_copy = latent_data[idx] + np.random.normal(0, 0.01, latent_data.shape[1])
                    new_centroids = np.vstack([new_centroids, noisy_copy])
                cluster_labels = np.arange(len(latent_data))  # Each sample is its own cluster
            else:
                logger.error(f"   ❌ No latent samples available")
                raise RuntimeError("No latent samples for centroid computation")
        else:
            # Normal K-means clustering
            kmeans = KMeans(n_clusters=n_centroids, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(latent_data)
            new_centroids = kmeans.cluster_centers_
        
        # Compute new metric matrices based on cluster statistics
        new_metric_matrices = []
        
        for i in range(n_centroids):
            cluster_points = latent_data[cluster_labels == i]
            
            if len(cluster_points) > 1:
                # Use covariance of cluster points to define local metric
                cov_matrix = np.cov(cluster_points.T)
                
                # Add regularization to ensure positive definite
                cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
                
                # The metric tensor is the inverse of covariance (precision matrix)
                try:
                    metric_matrix = np.linalg.inv(cov_matrix)
                except np.linalg.LinAlgError:
                    # Fallback to identity if inversion fails
                    logger.warning(f"⚠️ Singular covariance for cluster {i}, using identity")
                    metric_matrix = np.eye(cov_matrix.shape[0])
            else:
                # Single point cluster - use identity
                metric_matrix = np.eye(latent_data.shape[1])
            
            new_metric_matrices.append(metric_matrix)
        
        new_metric_matrices = np.array(new_metric_matrices)
        
        # Compute manifold evolution metrics
        evolution_metrics = self._compute_manifold_evolution_metrics(
            latent_data, new_centroids, new_metric_matrices, cluster_labels
        )
        
        logger.info(f"✅ Computed new centroids and metrics")
        return new_centroids, new_metric_matrices, evolution_metrics
    
    def _compute_manifold_evolution_metrics(self, latent_data: np.ndarray, centroids: np.ndarray, 
                                          metrics: np.ndarray, labels: np.ndarray) -> Dict:
        """Compute various metrics tracking manifold evolution."""
        
        # Compute centroid shift from previous update
        if len(self.centroid_history) > 0:
            prev_centroids = self.centroid_history[-1]
            centroid_shift = np.mean(np.linalg.norm(centroids - prev_centroids, axis=1))
        else:
            centroid_shift = np.mean(np.linalg.norm(centroids - self.original_centroids.cpu().numpy(), axis=1))
        
        # Compute metric change
        if len(self.metric_history) > 0:
            prev_metrics = self.metric_history[-1]
            metric_change = np.mean([np.linalg.norm(metrics[i] - prev_metrics[i]) for i in range(len(metrics))])
        else:
            orig_metrics = self.original_metric_matrices.cpu().numpy()
            metric_change = np.mean([np.linalg.norm(metrics[i] - orig_metrics[i]) for i in range(len(metrics))])
        
        # Compute latent space variance
        latent_variance = np.var(latent_data, axis=0).mean()
        
        # Compute manifold coverage (how well centroids cover the latent space)
        distances_to_nearest_centroid = []
        for point in latent_data:
            dists = np.linalg.norm(centroids - point, axis=1)
            distances_to_nearest_centroid.append(np.min(dists))
        coverage_metric = np.mean(distances_to_nearest_centroid)
        
        return {
            'centroid_shift': centroid_shift,
            'metric_change': metric_change,
            'latent_variance': latent_variance,
            'coverage_metric': coverage_metric,
            'cluster_sizes': [np.sum(labels == i) for i in range(len(centroids))],
            'inertia': np.sum([np.min(np.linalg.norm(centroids - point, axis=1))**2 for point in latent_data])
        }
    
    def update_model_centroids(self, new_centroids: np.ndarray, new_metric_matrices: np.ndarray, 
                              evolution_metrics: Dict, epoch: int) -> None:
        """Update the model's centroids and metric matrices."""
        logger.info(f"🔄 Updating model centroids at epoch {epoch}")
        
        # Convert to tensors and update model
        new_centroids_tensor = torch.tensor(new_centroids, dtype=torch.float32, device=self.device)
        new_matrices_tensor = torch.tensor(new_metric_matrices, dtype=torch.float32, device=self.device)
        
        # Store in history before updating
        self.centroid_history.append(new_centroids.copy())
        self.metric_history.append(new_metric_matrices.copy())
        
        # Store evolution metrics
        self.manifold_evolution_metrics['centroid_shifts'].append(evolution_metrics['centroid_shift'])
        self.manifold_evolution_metrics['metric_changes'].append(evolution_metrics['metric_change'])
        self.manifold_evolution_metrics['update_epochs'].append(epoch)
        self.manifold_evolution_metrics['latent_variance_evolution'].append(evolution_metrics['latent_variance'])
        self.manifold_evolution_metrics['coverage_evolution'].append(evolution_metrics['coverage_metric'])
        
        # 🌊 GRADUAL CENTROID UPDATE for numerical stability
        alpha = 0.3  # Interpolation factor (30% new, 70% old)
        
        # Update model parameters based on accessor type with gradual transition
        if self.centroids_accessor == 'modular_metric':
            current_centroids = self.model.modular_metric.centroids.data
            current_matrices = self.model.modular_metric.metric_matrices.data
            
            # Gradual interpolation
            self.model.modular_metric.centroids.data = alpha * new_centroids_tensor + (1 - alpha) * current_centroids
            self.model.modular_metric.metric_matrices.data = alpha * new_matrices_tensor + (1 - alpha) * current_matrices
            logger.info(f"✅ Gradually updated modular metric centroids (α={alpha})")
            
        elif self.centroids_accessor == 'direct_tensors':
            current_centroids = self.model.centroids_tens.data
            current_matrices = self.model.M_tens.data
            
            # Gradual interpolation  
            self.model.centroids_tens.data = alpha * new_centroids_tensor + (1 - alpha) * current_centroids
            self.model.M_tens.data = alpha * new_matrices_tensor + (1 - alpha) * current_matrices
            logger.info(f"✅ Gradually updated direct tensor centroids (α={alpha})")
            
            # Also update the metric functions if they exist
            if hasattr(self.model, 'G_inv') and hasattr(self.model, 'G'):
                # Force recreation of metric functions with new centroids
                self._recreate_metric_functions()
                
        else:
            logger.error("❌ No valid centroids accessor found - cannot update")
            return
        
        # Log to WandB
        wandb.log({
            'adaptive_centroids/centroid_shift': evolution_metrics['centroid_shift'],
            'adaptive_centroids/metric_change': evolution_metrics['metric_change'],
            'adaptive_centroids/latent_variance': evolution_metrics['latent_variance'],
            'adaptive_centroids/coverage_metric': evolution_metrics['coverage_metric'],
            'adaptive_centroids/n_updates': len(self.centroid_history),
            'epoch': epoch
        })
        
        # ✨ UPDATE RHVAE SAMPLERS WITH NEW CENTROIDS ✨
        self._update_rhvae_samplers(new_centroids_tensor, new_matrices_tensor)
        
        # ✨ CREATE EVOLVED MANIFOLD SAMPLING VISUALIZATION ✨
        if hasattr(self, 'visualization_manager') and self.visualization_manager is not None:
            if hasattr(self.visualization_manager, 'manifold_evolution') and self.visualization_manager.manifold_evolution is not None:
                self.visualization_manager.manifold_evolution.create_manifold_0_sampling_visualization(epoch)
                logger.info(f"📊 Created evolved manifold sampling visualization for epoch {epoch}")
        
        # Enhanced stability checks after update
        if evolution_metrics['centroid_shift'] > 10.0:
            logger.warning(f"⚠️ Large centroid shift detected: {evolution_metrics['centroid_shift']:.4f}")
            logger.warning("⚠️ Consider reducing learning rate or centroid update frequency")
            
        if evolution_metrics['metric_change'] > 500000:  # 500k threshold
            logger.warning(f"⚠️ Extreme metric change detected: {evolution_metrics['metric_change']:.1f}")
            logger.warning("⚠️ Using gradual updates with α=0.3 to maintain stability")
            
        logger.info(f"✅ Model centroids updated at epoch {epoch}")
        logger.info(f"📊 Centroid shift: {evolution_metrics['centroid_shift']:.4f}")
        logger.info(f"📊 Metric change: {evolution_metrics['metric_change']:.4f}")
    
    def smooth_metric_refresh(self, new_centroids: np.ndarray, new_metric_matrices: np.ndarray, 
                            evolution_metrics: Dict, epoch: int) -> None:
        """
        Smooth metric refresh that mimics the initial loading process for stability.
        This approach recreates the metric system cleanly like load_pretrained_metrics().
        """
        logger.info(f"🌊 Performing smooth metric refresh at epoch {epoch}")
        
        # Convert to tensors
        new_centroids_tensor = torch.tensor(new_centroids, dtype=torch.float32, device=self.device)
        new_matrices_tensor = torch.tensor(new_metric_matrices, dtype=torch.float32, device=self.device)
        
        # Store evolution metrics
        self.centroid_history.append(new_centroids.copy())
        self.metric_history.append(new_metric_matrices.copy())
        self.manifold_evolution_metrics['centroid_shifts'].append(evolution_metrics['centroid_shift'])
        self.manifold_evolution_metrics['metric_changes'].append(evolution_metrics['metric_change'])
        self.manifold_evolution_metrics['update_epochs'].append(epoch)
        self.manifold_evolution_metrics['latent_variance_evolution'].append(evolution_metrics['latent_variance'])
        self.manifold_evolution_metrics['coverage_evolution'].append(evolution_metrics['coverage_metric'])
        
        # 🎯 ULTRA-CONSERVATIVE INTERPOLATION with validation
        alpha = 0.02  # Ultra-conservative: 2% new, 98% old
        
        if self.centroids_accessor == 'direct_tensors':
            # Get current state
            current_centroids = self.model.centroids_tens.data.clone()
            current_matrices = self.model.M_tens.data.clone()
            
            # Gradual interpolation
            interpolated_centroids = alpha * new_centroids_tensor + (1 - alpha) * current_centroids
            interpolated_matrices = alpha * new_matrices_tensor + (1 - alpha) * current_matrices
            
            # 🛡️ VALIDATION STEP: Test metric stability before committing
            validation_passed = self._validate_metric_stability(interpolated_centroids, interpolated_matrices)
            
            if validation_passed:
                # 🌊 CLEAN METRIC REFRESH (like initial loading)
                self.model.centroids_tens = interpolated_centroids.detach().requires_grad_(False)
                self.model.M_tens = interpolated_matrices.detach().requires_grad_(False)
                logger.info(f"✅ Metric validation passed - applying refresh (α={alpha})")
            else:
                # 🚫 ROLLBACK: Keep current metric
                interpolated_centroids = current_centroids
                interpolated_matrices = current_matrices
                logger.warning(f"⚠️ Metric validation failed - keeping current metric")
            
            # Recreate metric functions cleanly (exactly like load_pretrained_metrics)
            def _G_inv(z: torch.Tensor):
                # Ensure float32 and proper device
                z = z.to(dtype=torch.float32, device=self.device)
                centroids = self.model.centroids_tens.to(dtype=torch.float32, device=self.device)
                M_tens = self.model.M_tens.to(dtype=torch.float32, device=self.device)
                
                diff = centroids.unsqueeze(0) - z.unsqueeze(1)  # (B, K, D)
                weights = torch.exp(-torch.norm(diff, dim=-1) ** 2 / (self.model.temperature ** 2))
                weighted_M = M_tens.unsqueeze(0) * weights.unsqueeze(-1).unsqueeze(-1)
                G_inv = weighted_M.sum(dim=1) + self.model.lbd * torch.eye(self.model.latent_dim, device=z.device, dtype=torch.float32)
                return G_inv

            def _G(z: torch.Tensor):
                return torch.linalg.inv(_G_inv(z))

            # Replace metric functions cleanly
            self.model.G = _G
            self.model.G_inv = _G_inv
            
            logger.info(f"✅ Smooth metric refresh complete (α={alpha})")
            
        # Update RHVAE samplers with new metric
        self._update_rhvae_samplers_refresh(interpolated_centroids, interpolated_matrices)
        
        # ✨ CREATE EVOLVED MANIFOLD SAMPLING VISUALIZATION ✨
        if hasattr(self, 'visualization_manager') and self.visualization_manager is not None:
            if hasattr(self.visualization_manager, 'manifold_evolution') and self.visualization_manager.manifold_evolution is not None:
                self.visualization_manager.manifold_evolution.create_manifold_0_sampling_visualization(epoch)
                logger.info(f"📊 Created evolved manifold sampling visualization for epoch {epoch}")
        
        # Enhanced stability checks
        if evolution_metrics['centroid_shift'] > 10.0:
            logger.warning(f"⚠️ Large centroid shift detected: {evolution_metrics['centroid_shift']:.4f}")
            
        if evolution_metrics['metric_change'] > 500000:
            logger.warning(f"⚠️ Extreme metric change detected: {evolution_metrics['metric_change']:.1f}")
            logger.warning(f"⚠️ Using conservative refresh with α={alpha} to maintain stability")
            
        logger.info(f"✅ Smooth metric refresh completed at epoch {epoch}")
        logger.info(f"📊 Centroid shift: {evolution_metrics['centroid_shift']:.4f}")
        logger.info(f"📊 Metric change: {evolution_metrics['metric_change']:.4f}")
    
    def _update_rhvae_samplers_refresh(self, new_centroids_tensor: torch.Tensor, new_matrices_tensor: torch.Tensor):
        """Update RHVAE samplers with refreshed metric (clean version)."""
        
        # Update visualization manager's RHVAE samplers if they exist
        if hasattr(self, 'visualization_manager') and self.visualization_manager is not None:
            vis_manager = self.visualization_manager
            
            if hasattr(vis_manager, 'manifold_evolution') and vis_manager.manifold_evolution is not None:
                manifold_vis = vis_manager.manifold_evolution
                
                if hasattr(manifold_vis, 'rhvae_sampler') and manifold_vis.rhvae_sampler is not None:
                    sampler = manifold_vis.rhvae_sampler
                    if hasattr(sampler, 'model'):
                        # Clean update (detached tensors)
                        if hasattr(sampler.model, 'centroids_tens'):
                            sampler.model.centroids_tens = new_centroids_tensor.detach().requires_grad_(False)
                            logger.info("✅ Refreshed RHVAE sampler centroids")
                            
                        if hasattr(sampler.model, 'M_tens'):
                            sampler.model.M_tens = new_matrices_tensor.detach().requires_grad_(False)
                            logger.info("✅ Refreshed RHVAE sampler metric matrices")
                            
                        # Recreate metric functions cleanly
                        if hasattr(sampler.model, 'G') and hasattr(sampler.model, 'G_inv'):
                            self._recreate_sampler_metric_functions_clean(sampler.model)
                            logger.info("✅ Refreshed RHVAE sampler metric functions")
        
        logger.info("🎯 RHVAE samplers refreshed with evolved manifold structure!")
    
    def _recreate_sampler_metric_functions_clean(self, sampler_model):
        """Recreate metric functions for sampler model with clean approach (like initial loading)."""
        def _G_inv(z: torch.Tensor):
            z = z.to(dtype=torch.float32, device=self.device)
            centroids = sampler_model.centroids_tens.to(dtype=torch.float32, device=self.device)
            M_tens = sampler_model.M_tens.to(dtype=torch.float32, device=self.device)
            
            diff = centroids.unsqueeze(0) - z.unsqueeze(1)
            weights = torch.exp(-torch.norm(diff, dim=-1) ** 2 / (sampler_model.temperature ** 2))
            weighted_M = M_tens.unsqueeze(0) * weights.unsqueeze(-1).unsqueeze(-1)
            G_inv = weighted_M.sum(dim=1) + sampler_model.lbd * torch.eye(sampler_model.latent_dim, device=z.device, dtype=torch.float32)
            return G_inv
        
        def _G(z: torch.Tensor):
            return torch.linalg.inv(_G_inv(z))
        
        sampler_model.G = _G
        sampler_model.G_inv = _G_inv
    
    def _recreate_metric_functions(self):
        """Recreate metric functions after centroid updates for direct tensor models with stability checks."""
        # Add numerical stability to metric function recreation
        def _G_inv(z: torch.Tensor):
            # Ensure all tensors are float32 to avoid complex number issues
            z = z.to(dtype=torch.float32)
            centroids_real = self.model.centroids_tens.to(dtype=torch.float32)
            M_tens_real = self.model.M_tens.to(dtype=torch.float32)
            
            diff = centroids_real.unsqueeze(0) - z.unsqueeze(1)  # (B, K, D)
            
            # Clamp temperature to prevent division by zero or extreme values
            temp_safe = torch.clamp(self.model.temperature, min=1e-4, max=10.0)
            
            # Use stable exponential with clamping
            dist_sq = torch.norm(diff, dim=-1) ** 2
            exp_arg = -dist_sq / (temp_safe ** 2)
            exp_arg = torch.clamp(exp_arg, min=-50, max=50)  # Prevent overflow/underflow
            weights = torch.exp(exp_arg)
            
            # Normalize weights to prevent extreme values
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            
            weighted_M = M_tens_real.unsqueeze(0) * weights.unsqueeze(-1).unsqueeze(-1)
            
            # Add larger regularization for stability
            reg_term = torch.clamp(self.model.lbd, min=1e-4, max=1.0) * torch.eye(self.model.latent_dim, device=z.device, dtype=torch.float32)
            G_inv = weighted_M.sum(dim=1) + reg_term
            
            # Simplified positive definiteness check without eigenvals to avoid complex issues
            try:
                # Test if matrix is positive definite by trying Cholesky decomposition
                torch.linalg.cholesky(G_inv + 1e-6 * torch.eye(self.model.latent_dim, device=z.device, dtype=torch.float32))
            except:
                # If not positive definite, add more regularization
                additional_reg = 1e-4 * torch.eye(self.model.latent_dim, device=z.device, dtype=torch.float32)
                G_inv = G_inv + additional_reg
                
            return G_inv.to(dtype=torch.float32)

        def _G(z: torch.Tensor):
            G_inv_val = _G_inv(z)
            try:
                # Use SVD for more stable inversion
                U, S, Vh = torch.linalg.svd(G_inv_val)
                S_inv = 1.0 / torch.clamp(S, min=1e-8)  # Prevent division by zero
                G = U @ torch.diag_embed(S_inv) @ Vh
                return G
            except:
                # Fallback to regularized pseudo-inverse
                return torch.linalg.pinv(G_inv_val + 1e-6 * torch.eye(G_inv_val.shape[-1], device=z.device))

        self.model.G = _G
        self.model.G_inv = _G_inv
        logger.info("🔄 Recreated metric functions with numerical stability safeguards")
    
    def _update_rhvae_samplers(self, new_centroids_tensor: torch.Tensor, new_matrices_tensor: torch.Tensor):
        """Update any RHVAE samplers with the new centroids and metric matrices."""
        
        # Update visualization manager's RHVAE samplers if they exist
        if hasattr(self, 'visualization_manager') and self.visualization_manager is not None:
            vis_manager = self.visualization_manager
            
            # Check if manifold evolution visualizer has RHVAE samplers
            if hasattr(vis_manager, 'manifold_evolution') and vis_manager.manifold_evolution is not None:
                manifold_vis = vis_manager.manifold_evolution
                
                # Update RHVAE sampler if it exists
                if hasattr(manifold_vis, 'rhvae_sampler') and manifold_vis.rhvae_sampler is not None:
                    sampler = manifold_vis.rhvae_sampler
                    if hasattr(sampler, 'model'):
                        # Update the sampler's model centroids and metric matrices
                        if hasattr(sampler.model, 'centroids_tens'):
                            sampler.model.centroids_tens.data = new_centroids_tensor.clone()
                            logger.info("✅ Updated RHVAE sampler centroids")
                            
                        if hasattr(sampler.model, 'M_tens'):
                            sampler.model.M_tens.data = new_matrices_tensor.clone()
                            logger.info("✅ Updated RHVAE sampler metric matrices")
                            
                        # Recreate metric functions for the sampler model
                        if hasattr(sampler.model, 'G') and hasattr(sampler.model, 'G_inv'):
                            self._recreate_sampler_metric_functions(sampler.model)
                            logger.info("✅ Updated RHVAE sampler metric functions")
        
        # Also check if the model itself has any attached samplers
        if hasattr(self.model, '_rhvae_samplers'):
            for sampler in self.model._rhvae_samplers:
                if hasattr(sampler, 'model'):
                    if hasattr(sampler.model, 'centroids_tens'):
                        sampler.model.centroids_tens.data = new_centroids_tensor.clone()
                    if hasattr(sampler.model, 'M_tens'):
                        sampler.model.M_tens.data = new_matrices_tensor.clone()
                    if hasattr(sampler.model, 'G') and hasattr(sampler.model, 'G_inv'):
                        self._recreate_sampler_metric_functions(sampler.model)
            logger.info(f"✅ Updated {len(self.model._rhvae_samplers)} attached RHVAE samplers")
        
        logger.info("🎯 RHVAE samplers now use the evolved manifold structure!")
    
    def _recreate_sampler_metric_functions(self, sampler_model):
        """Recreate metric functions for a sampler model with numerical stability."""
        def _G_inv(z: torch.Tensor):
            # Ensure all tensors are float32
            z = z.to(dtype=torch.float32)
            centroids_real = sampler_model.centroids_tens.to(dtype=torch.float32)
            M_tens_real = sampler_model.M_tens.to(dtype=torch.float32)
            
            diff = centroids_real.unsqueeze(0) - z.unsqueeze(1)  # (B, K, D)
            temp_safe = torch.clamp(sampler_model.temperature, min=1e-4, max=10.0)
            weights = torch.exp(-torch.norm(diff, dim=-1) ** 2 / (temp_safe ** 2))
            weighted_M = M_tens_real.unsqueeze(0) * weights.unsqueeze(-1).unsqueeze(-1)
            G_inv = weighted_M.sum(dim=1) + sampler_model.lbd * torch.eye(sampler_model.latent_dim, device=z.device, dtype=torch.float32)
            
            # Simple positive definiteness check
            try:
                torch.linalg.cholesky(G_inv + 1e-6 * torch.eye(sampler_model.latent_dim, device=z.device, dtype=torch.float32))
            except:
                G_inv = G_inv + 1e-4 * torch.eye(G_inv.size(-1), device=G_inv.device, dtype=torch.float32)
            
            return G_inv.to(dtype=torch.float32)
        
        def _G(z: torch.Tensor):
            G_inv_val = _G_inv(z)
            try:
                return torch.linalg.inv(G_inv_val)
            except:
                # Fallback to pseudo-inverse
                return torch.linalg.pinv(G_inv_val + 1e-6 * torch.eye(G_inv_val.size(-1), device=G_inv_val.device, dtype=torch.float32))
        
        sampler_model.G = _G
        sampler_model.G_inv = _G_inv
    
    def _validate_metric_stability(self, test_centroids: torch.Tensor, test_matrices: torch.Tensor) -> bool:
        """
        Validate that proposed metric changes won't cause numerical instability.
        Tests the metric on a small sample to ensure stable computation.
        """
        try:
            # Create test metric functions
            def test_G_inv(z: torch.Tensor):
                z = z.to(dtype=torch.float32, device=self.device)
                centroids = test_centroids.to(dtype=torch.float32, device=self.device)
                M_tens = test_matrices.to(dtype=torch.float32, device=self.device)
                
                diff = centroids.unsqueeze(0) - z.unsqueeze(1)
                weights = torch.exp(-torch.norm(diff, dim=-1) ** 2 / (self.model.temperature ** 2))
                weighted_M = M_tens.unsqueeze(0) * weights.unsqueeze(-1).unsqueeze(-1)
                G_inv = weighted_M.sum(dim=1) + self.model.lbd * torch.eye(self.model.latent_dim, device=z.device, dtype=torch.float32)
                return G_inv
            
            # Test on small sample of latent points
            test_batch_size = 4
            test_z = torch.randn(test_batch_size, self.model.latent_dim, device=self.device, dtype=torch.float32)
            
            # Test metric computation
            with torch.no_grad():
                G_inv = test_G_inv(test_z)
                
                # Check for NaN/Inf
                if torch.isnan(G_inv).any() or torch.isinf(G_inv).any():
                    logger.warning("⚠️ Validation failed: NaN/Inf in metric tensor")
                    return False
                
                # Check eigenvalues for positive definiteness
                eigenvals = torch.linalg.eigvals(G_inv)
                min_eigenval = torch.real(eigenvals).min()
                max_eigenval = torch.real(eigenvals).max()
                
                if min_eigenval <= 1e-8:
                    logger.warning(f"⚠️ Validation failed: negative/zero eigenvalue {min_eigenval:.2e}")
                    return False
                
                # Check condition number
                condition_number = max_eigenval / min_eigenval
                if condition_number > 1e6:
                    logger.warning(f"⚠️ Validation failed: poor conditioning {condition_number:.2e}")
                    return False
                
                # Test matrix inversion
                try:
                    G = torch.linalg.inv(G_inv)
                    if torch.isnan(G).any() or torch.isinf(G).any():
                        logger.warning("⚠️ Validation failed: NaN/Inf in inverse metric")
                        return False
                except:
                    logger.warning("⚠️ Validation failed: matrix inversion error")
                    return False
                
                logger.info(f"✅ Metric validation passed: eigenvals [{min_eigenval:.3e}, {max_eigenval:.3e}], cond={condition_number:.1e}")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️ Validation failed with exception: {e}")
            return False
    
    def perform_adaptive_centroid_update(self, data_loader, epoch: int) -> None:
        """Perform the complete adaptive centroid update process with KL divergence control."""
        was_training = False  # Initialize to avoid UnboundLocalError
        try:
            # 🧊 FREEZE MODE: Only analyze, don't update model
            if self.freeze_mode:
                logger.info(f"🧊 FREEZE MODE - Epoch {epoch}: Analyzing manifold evolution without updating")
                self._perform_freeze_mode_analysis(data_loader, epoch)
                return
            
            # 🎯 KL-CONTROLLED MODE: Real updates with stability monitoring
            if self.kl_controlled_mode:
                logger.info(f"🎯 KL-CONTROLLED MODE - Epoch {epoch}: Real updates with stability control")
                self._perform_kl_controlled_update(data_loader, epoch)
                return
            
            # 🛡️ LEGACY MODE: Standard update (kept for backward compatibility)
            was_training = self.model.training
            self.model.eval()
            
            # Extract current latent distribution
            latent_data = self.extract_current_latent_distribution(data_loader)
            
            # Compute new centroids and metrics
            new_centroids, new_metric_matrices, evolution_metrics = self.compute_new_centroids_and_metrics(latent_data)
            
            # 🌊 SMOOTH METRIC REFRESH (mimics initial loading process)
            self.smooth_metric_refresh(new_centroids, new_metric_matrices, evolution_metrics, epoch)
            
            # Create manifold evolution visualization
            self._create_manifold_evolution_visualization(epoch)
            
        except Exception as e:
            logger.error(f"❌ Adaptive centroid update failed at epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 🛡️ RESTORE TRAINING MODE
            if was_training:
                self.model.train()
    
    def _create_manifold_evolution_visualization(self, epoch: int) -> None:
        """Create visualization showing manifold evolution."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            if len(self.centroid_history) < 2:
                return  # Need at least 2 updates for evolution plot
            
            # Create manifold evolution plot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "Centroid Evolution Over Training",
                    "Manifold Metrics Evolution", 
                    "Latent Space Coverage",
                    "Current vs Original Centroids"
                ],
                specs=[
                    [{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "scatter"}, {"type": "scatter"}]
                ]
            )
            
            # 1. Centroid positions over time
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            # Show first 10 centroids for clarity
            for update_idx, centroids in enumerate(self.centroid_history):
                epoch_num = self.manifold_evolution_metrics['update_epochs'][update_idx]
                
                fig.add_trace(
                    go.Scatter(
                        x=centroids[:10, 0], y=centroids[:10, 1],
                        mode='markers',
                        marker=dict(size=8, color=colors[update_idx % len(colors)], opacity=0.7),
                        name=f'Epoch {epoch_num}',
                        hovertemplate=f"Epoch {epoch_num}<br>Z1: %{{x:.3f}}<br>Z2: %{{y:.3f}}<extra></extra>"
                    ), row=1, col=1
                )
            
            # 2. Evolution metrics
            epochs = self.manifold_evolution_metrics['update_epochs']
            
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=self.manifold_evolution_metrics['centroid_shifts'],
                    mode='lines+markers',
                    name='Centroid Shift',
                    line=dict(color='blue'),
                    hovertemplate="Epoch: %{x}<br>Shift: %{y:.4f}<extra></extra>"
                ), row=1, col=2
            )
            
            # 3. Coverage evolution
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=self.manifold_evolution_metrics['coverage_evolution'],
                    mode='lines+markers',
                    name='Coverage Metric',
                    line=dict(color='green'),
                    hovertemplate="Epoch: %{x}<br>Coverage: %{y:.4f}<extra></extra>"
                ), row=2, col=1
            )
            
            # 4. Current vs Original comparison
            current_centroids = self.centroid_history[-1]
            original_centroids = self.original_centroids.cpu().numpy()
            
            fig.add_trace(
                go.Scatter(
                    x=original_centroids[:10, 0], y=original_centroids[:10, 1],
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='circle'),
                    name='Original',
                    hovertemplate="Original<br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<extra></extra>"
                ), row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=current_centroids[:10, 0], y=current_centroids[:10, 1],
                    mode='markers',
                    marker=dict(size=10, color='green', symbol='diamond'),
                    name='🚀 Current',
                    hovertemplate="Current<br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<extra></extra>"
                ), row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                title=dict(
                    text=f"🚀 Manifold Evolution - Epoch {epoch}<br><sub>Living manifold adaptation during training</sub>",
                    x=0.5,
                    font=dict(size=16)
                ),
                showlegend=True
            )
            
            # Log to WandB
            wandb.log({f"manifold_evolution/epoch_{epoch}": wandb.Html(fig.to_html())})
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create manifold evolution visualization: {e}")
    
    def train_epoch(self, train_loader, epoch):
        """Override train_epoch to include adaptive centroid updates."""
        # Check if we should update centroids BEFORE training
        if self.should_update_centroids(epoch):
            logger.info(f"🔄 Performing adaptive centroid update at epoch {epoch}")
            self.perform_adaptive_centroid_update(train_loader, epoch)
        
        # Run normal training epoch
        return super().train_epoch(train_loader, epoch)
    
    def train(self, n_epochs=30):
        """Override train method to add adaptive centroid summary."""
        logger.info(f"🚀 Starting adaptive centroid training for {n_epochs} epochs")
        
        # Call parent train method
        result = super().train(n_epochs)
        
        # Create final summary
        self._create_final_adaptive_summary()
        
        return result
    
    def _create_final_adaptive_summary(self) -> None:
        """Create final summary of adaptive centroid training."""
        if not self.centroid_history:
            return
        
        n_updates = len(self.centroid_history)
        total_shift = sum(self.manifold_evolution_metrics['centroid_shifts'])
        final_coverage = self.manifold_evolution_metrics['coverage_evolution'][-1]
        
        summary_metrics = {
            'adaptive_centroids/total_updates': n_updates,
            'adaptive_centroids/total_centroid_shift': total_shift,
            'adaptive_centroids/final_coverage': final_coverage,
            'adaptive_centroids/update_frequency': self.centroid_update_frequency
        }
        
        wandb.log(summary_metrics)
        
        logger.info("🎉 Adaptive centroid training completed!")
        logger.info(f"📊 Total centroid updates: {n_updates}")
        logger.info(f"📊 Total centroid movement: {total_shift:.4f}")
        logger.info(f"📊 Final manifold coverage: {final_coverage:.4f}")

    def _perform_freeze_mode_analysis(self, data_loader, epoch: int) -> None:
        """
        Perform freeze mode analysis: extract latent distributions, compute what new 
        centroids would be, visualize evolution, but DON'T update the model.
        """
        try:
            logger.info(f"🧊 FREEZE MODE ANALYSIS - Epoch {epoch}")
            logger.info("   📊 Extracting latent distribution...")
            
            # Extract current latent distribution (same as normal mode)
            latent_data = self.extract_current_latent_distribution(data_loader)
            
            # Compute what new centroids WOULD be (but don't apply them)
            logger.info("   🎯 Computing what new centroids would be...")
            new_centroids, new_metric_matrices, evolution_metrics = self.compute_new_centroids_and_metrics(latent_data)
            
            # Store the analysis data for visualization (but don't update model)
            logger.info("   📈 Storing evolution data for analysis...")
            
            # Convert to tensors for consistent storage
            new_centroids_tensor = torch.tensor(new_centroids, dtype=torch.float32, device=self.device)
            
            # Store what WOULD happen (for tracking evolution)
            self.centroid_history.append(new_centroids_tensor.cpu().numpy())
            self.manifold_evolution_metrics['update_epochs'].append(epoch)
            self.manifold_evolution_metrics['centroid_shifts'].append(evolution_metrics['centroid_shift'])
            self.manifold_evolution_metrics['latent_variance_evolution'].append(evolution_metrics['latent_variance'])
            self.manifold_evolution_metrics['coverage_evolution'].append(evolution_metrics['coverage_metric'])
            
            # Log freeze mode metrics to WandB
            freeze_metrics = {
                f'freeze_analysis/epoch': epoch,
                f'freeze_analysis/would_be_centroid_shift': evolution_metrics['centroid_shift'],
                f'freeze_analysis/would_be_coverage': evolution_metrics['coverage_metric'],
                f'freeze_analysis/current_latent_variance': evolution_metrics['latent_variance'],
                f'freeze_analysis/n_centroids_analyzed': len(new_centroids)
            }
            
            wandb.log(freeze_metrics)
            
            # Create visualization of what WOULD happen
            logger.info("   🎨 Creating freeze mode visualization...")
            self._create_manifold_evolution_visualization(epoch)
            
            logger.info(f"   ✅ FREEZE MODE: Analysis complete")
            logger.info(f"   📊 Centroid shift would be: {evolution_metrics['centroid_shift']:.4f}")
            logger.info(f"   📊 Coverage would be: {evolution_metrics['coverage_metric']:.4f}")
            logger.info(f"   🧊 Model unchanged - original metric tensors preserved")
            
        except Exception as e:
            logger.error(f"❌ Freeze mode analysis failed at epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()

    def _perform_kl_controlled_update(self, data_loader, epoch: int) -> None:
        """
        Perform KL-controlled adaptive centroid updates.
        Real metric updates with automatic stability monitoring and rollback.
        """
        try:
            logger.info(f"🎯 KL-CONTROLLED UPDATE - Epoch {epoch}")
            logger.info(f"   📊 Current adaptive alpha: {self.current_alpha:.3f}")
            
            # Step 1: Measure baseline KL divergence
            baseline_kl = self._measure_current_kl_divergence(data_loader)
            self.pre_update_kl = baseline_kl
            logger.info(f"   📊 Baseline KL divergence: {baseline_kl:.4f}")
            
            # Step 2: Save model state for potential rollback
            self._save_model_state()
            logger.info(f"   💾 Model state saved for rollback protection")
            
            # Step 3: Extract latent distribution and compute new centroids
            logger.info(f"   📊 Extracting latent distribution...")
            latent_data = self.extract_current_latent_distribution(data_loader)
            
            logger.info(f"   🎯 Computing new centroids and metrics...")
            new_centroids, new_metric_matrices, evolution_metrics = self.compute_new_centroids_and_metrics(latent_data)
            
            # Step 4: Attempt controlled update with monitoring
            update_successful = False
            rollback_attempt = 0
            
            while not update_successful and rollback_attempt < self.max_rollback_attempts:
                logger.info(f"   🔄 Update attempt {rollback_attempt + 1}/{self.max_rollback_attempts}")
                logger.info(f"   📊 Using alpha = {self.current_alpha:.3f}")
                
                # Apply gradual update with current alpha
                self._apply_controlled_metric_update(new_centroids, new_metric_matrices, 
                                                   evolution_metrics, epoch, self.current_alpha)
                
                # Step 5: Measure post-update KL divergence
                post_update_kl = self._measure_current_kl_divergence(data_loader)
                kl_growth = post_update_kl / baseline_kl if baseline_kl > 0 else float('inf')
                
                logger.info(f"   📊 Post-update KL: {post_update_kl:.4f} (growth: {kl_growth:.2f}x)")
                
                # Step 6: Stability check
                if self._is_kl_stable(baseline_kl, post_update_kl, kl_growth):
                    # Update successful!
                    update_successful = True
                    self.stability_metrics['successful_updates'] += 1
                    self.kl_history.append(post_update_kl)
                    
                    logger.info(f"   ✅ KL-controlled update SUCCESSFUL!")
                    logger.info(f"   📊 KL change: {baseline_kl:.4f} → {post_update_kl:.4f}")
                    logger.info(f"   🎯 Centroid shift: {evolution_metrics['centroid_shift']:.4f}")
                    
                    # Gradually increase alpha for next update (successful updates = more confidence)
                    self.current_alpha = min(self.adaptive_alpha_max, self.current_alpha * 1.1)
                    
                else:
                    # Update caused instability - rollback and retry
                    rollback_attempt += 1
                    self.stability_metrics['rollbacks'] += 1
                    
                    logger.warning(f"   ⚠️  KL instability detected - rolling back")
                    logger.warning(f"   📊 KL grew from {baseline_kl:.4f} to {post_update_kl:.4f}")
                    
                    # Rollback to previous state
                    self._rollback_model_state()
                    
                    if rollback_attempt < self.max_rollback_attempts:
                        # Reduce alpha and try again
                        self.current_alpha = max(self.adaptive_alpha_min, self.current_alpha * 0.5)
                        self.stability_metrics['alpha_reductions'] += 1
                        logger.info(f"   🔧 Reduced alpha to {self.current_alpha:.3f} for retry")
                    else:
                        logger.error(f"   ❌ Max rollback attempts reached - skipping update")
            
            # Step 7: Create visualizations and log metrics
            if update_successful:
                self._create_manifold_evolution_visualization(epoch)
                self._log_kl_control_metrics(epoch, baseline_kl, post_update_kl, kl_growth)
            else:
                logger.warning(f"   ⚠️  Update skipped due to persistent KL instability")
                self._log_failed_update_metrics(epoch, baseline_kl)
                
        except Exception as e:
            logger.error(f"❌ KL-controlled update failed at epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            # Ensure rollback on any error
            if hasattr(self, 'pre_update_model_state') and self.pre_update_model_state is not None:
                self._rollback_model_state()

    def _measure_current_kl_divergence(self, data_loader) -> float:
        """
        Measure current KL divergence on a sample of data.
        
        CRITICAL: This must provide GENUINE KL measurements, not fallback values,
        for the stability control system to work properly.
        """
        self.model.eval()
        total_kl = 0.0
        n_samples = 0
        max_batches = 3  # Reduced for efficiency but enough for stable measurement
        
        logger.info(f"   📊 Measuring KL divergence (sampling {max_batches} batches)")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break
                    
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                
                x = x.to(self.device)
                
                # FIXED: Ensure proper input format for the model
                try:
                    # The model expects [batch_size, n_obs, *input_dim] format
                    if len(x.shape) == 5:  # [B, T, C, H, W] - sequence data (correct format)
                        model_input = x
                        batch_size = x.shape[0]
                    elif len(x.shape) == 4:  # [B, C, H, W] - single images
                        # Add temporal dimension: [B, C, H, W] -> [B, 1, C, H, W]
                        model_input = x.unsqueeze(1)
                        batch_size = x.shape[0]
                    else:
                        logger.warning(f"   ⚠️  Unexpected input shape: {x.shape}, skipping")
                        continue
                    
                    # Forward pass through the model with proper input format
                    output = self.model(model_input)
                    
                    # Extract KL divergence from model output
                    kl_value = None
                    if hasattr(output, 'kld_loss') and output.kld_loss is not None:
                        kl_value = output.kld_loss
                    elif hasattr(output, 'kl') and output.kl is not None:
                        kl_value = output.kl
                    elif isinstance(output, dict) and 'kl_divergence' in output:
                        kl_value = output['kl_divergence']
                    elif isinstance(output, dict) and 'kl' in output:
                        kl_value = output['kl']
                    elif hasattr(output, 'z') and hasattr(output.z, 'kl'):
                        kl_value = output.z.kl
                    
                    if kl_value is not None:
                        # Ensure we get a scalar value
                        if torch.is_tensor(kl_value):
                            if kl_value.dim() > 0:
                                kl_scalar = kl_value.mean().item()
                            else:
                                kl_scalar = kl_value.item()
                        else:
                            kl_scalar = float(kl_value)
                        
                        # Sanity check for finite values
                        if torch.isfinite(torch.tensor(kl_scalar)):
                            total_kl += kl_scalar * batch_size  # Weight by batch size
                            n_samples += batch_size
                            logger.debug(f"   📊 Batch {batch_idx}: KL = {kl_scalar:.4f} (batch_size = {batch_size})")
                        else:
                            logger.warning(f"   ⚠️  Non-finite KL value: {kl_scalar}")
                    else:
                        logger.warning(f"   ⚠️  No KL found in model output")
                        
                except Exception as e:
                    logger.warning(f"   ⚠️  Error in KL measurement batch {batch_idx}: {e}")
                    logger.debug(f"   Input shape: {x.shape}, Model input shape: {model_input.shape if 'model_input' in locals() else 'undefined'}")
                    continue
        
        if n_samples == 0:
            logger.error(f"   ❌ NO VALID KL MEASUREMENTS OBTAINED!")
            logger.error(f"   This indicates a fundamental issue with the model or data format.")
            logger.error(f"   KL control cannot work without genuine KL measurements.")
            raise RuntimeError("KL measurement failed - cannot proceed with stability control")
            
        avg_kl = total_kl / n_samples
        logger.info(f"   📊 Measured KL: {avg_kl:.6f} (from {n_samples} samples)")
        
        return avg_kl

    def _save_model_state(self):
        """Save current model state for potential rollback."""
        import copy
        self.pre_update_model_state = copy.deepcopy(self.model.state_dict())
        if hasattr(self, 'optimizer'):
            self.pre_update_optimizer_state = copy.deepcopy(self.optimizer.state_dict())

    def _rollback_model_state(self):
        """Rollback model to pre-update state."""
        if self.pre_update_model_state is not None:
            self.model.load_state_dict(self.pre_update_model_state)
            logger.info(f"   🔄 Model state rolled back")
        
        if hasattr(self, 'optimizer') and self.pre_update_optimizer_state is not None:
            self.optimizer.load_state_dict(self.pre_update_optimizer_state)
            logger.info(f"   🔄 Optimizer state rolled back")

    def _apply_controlled_metric_update(self, new_centroids, new_metric_matrices, 
                                      evolution_metrics, epoch, alpha):
        """Apply metric update with specified interpolation rate."""
        # Convert to tensors
        new_centroids_tensor = torch.tensor(new_centroids, dtype=torch.float32, device=self.device)
        new_matrices_tensor = torch.tensor(new_metric_matrices, dtype=torch.float32, device=self.device)
        
        # Apply gradual interpolation update
        if self.centroids_accessor == 'modular_metric':
            # Gradual interpolation with specified alpha
            old_centroids = self.model.modular_metric.centroids.data
            old_matrices = self.model.modular_metric.metric_matrices.data
            
            updated_centroids = alpha * new_centroids_tensor + (1 - alpha) * old_centroids
            updated_matrices = alpha * new_matrices_tensor + (1 - alpha) * old_matrices
            
            # Update the model
            self.model.modular_metric.centroids.data.copy_(updated_centroids)
            self.model.modular_metric.metric_matrices.data.copy_(updated_matrices)
            
        elif self.centroids_accessor == 'direct_tensors':
            # Update direct tensors
            old_centroids = self.model.centroids_tens.data
            old_matrices = self.model.M_tens.data
            
            updated_centroids = alpha * new_centroids_tensor + (1 - alpha) * old_centroids
            updated_matrices = alpha * new_matrices_tensor + (1 - alpha) * old_matrices
            
            self.model.centroids_tens.data.copy_(updated_centroids)
            self.model.M_tens.data.copy_(updated_matrices)
            
            # Recreate metric functions
            self._recreate_metric_functions()
        
        # Update RHVAE samplers if available
        self._update_rhvae_samplers(updated_centroids, updated_matrices)

    def _is_kl_stable(self, baseline_kl, post_kl, growth_rate) -> bool:
        """Check if KL divergence remains stable after update."""
        # Check absolute KL value
        if post_kl > self.kl_stability_threshold:
            logger.warning(f"   ⚠️  KL above threshold: {post_kl:.4f} > {self.kl_stability_threshold}")
            return False
        
        # Check growth rate
        if growth_rate > self.kl_growth_threshold:
            logger.warning(f"   ⚠️  KL growth too high: {growth_rate:.2f}x > {self.kl_growth_threshold}x")
            return False
        
        # Check for NaN or infinite values
        if not (torch.isfinite(torch.tensor(post_kl)) and torch.isfinite(torch.tensor(baseline_kl))):
            logger.warning(f"   ⚠️  Non-finite KL values detected")
            return False
        
        return True

    def _log_kl_control_metrics(self, epoch, baseline_kl, post_kl, growth_rate):
        """Log KL control metrics to WandB."""
        kl_metrics = {
            f'kl_control/epoch': epoch,
            f'kl_control/baseline_kl': baseline_kl,
            f'kl_control/post_update_kl': post_kl,
            f'kl_control/kl_growth_rate': growth_rate,
            f'kl_control/current_alpha': self.current_alpha,
            f'kl_control/successful_updates': self.stability_metrics['successful_updates'],
            f'kl_control/total_rollbacks': self.stability_metrics['rollbacks'],
            f'kl_control/alpha_reductions': self.stability_metrics['alpha_reductions']
        }
        
        wandb.log(kl_metrics)

    def _log_failed_update_metrics(self, epoch, baseline_kl):
        """Log metrics for failed updates."""
        fail_metrics = {
            f'kl_control/epoch': epoch,
            f'kl_control/baseline_kl': baseline_kl,
            f'kl_control/update_failed': 1,
            f'kl_control/current_alpha': self.current_alpha,
            f'kl_control/total_rollbacks': self.stability_metrics['rollbacks']
        }
        
        wandb.log(fail_metrics)


def create_adaptive_config(base_config, centroid_update_frequency: int = 2) -> DictConfig:
    """Create configuration for adaptive centroid training."""
    config = base_config.copy()
    
    # Add adaptive centroid parameters
    config.adaptive_centroids = DictConfig({
        'enabled': True,
        'update_frequency': centroid_update_frequency,
        'n_samples_for_centroids': 500,
        'visualize_evolution': True
    })
    
    return config 