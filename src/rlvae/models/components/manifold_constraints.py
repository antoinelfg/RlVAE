"""
Manifold Constraints for RHMC Posterior
=======================================

Implements geometric constraints to ensure RHMC samples stay on the learned manifold.
Provides projection methods and elastic recall mechanisms.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import warnings


class ManifoldConstraints(nn.Module):
    """
    Geometric constraints for keeping RHMC samples on the learned manifold.
    
    Key features:
    - Projection to high-density regions
    - Elastic recall from low-density areas
    - Adaptive constraint strength based on manifold density
    """
    
    def __init__(self, 
                 projection_strength: float = 0.5,
                 density_threshold: float = 0.1,
                 elastic_strength: float = 0.3,
                 max_projection_distance: float = 1.0):
        super().__init__()
        
        self.projection_strength = projection_strength
        self.density_threshold = density_threshold
        self.elastic_strength = elastic_strength
        self.max_projection_distance = max_projection_distance
        
    def apply_manifold_constraints(self, 
                                 z: torch.Tensor, 
                                 model: nn.Module,
                                 mu_reference: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply manifold constraints to keep samples on the learned manifold.
        
        Args:
            z: Current latent samples [B, D]
            model: Model with centroids_tens and G/G_inv methods
            mu_reference: Optional reference encoder means [B, D]
            
        Returns:
            z_constrained: Constrained samples [B, D]
        """
        if not (hasattr(model, 'centroids_tens') and hasattr(model, 'G_inv')):
            warnings.warn("Model missing centroids_tens or G_inv, skipping constraints")
            return z
        
        z_constrained = z.clone()
        batch_size = z.shape[0]
        
        try:
            # 1. Compute manifold density at current positions
            density_scores = self._compute_manifold_density(z, model)
            
            # 2. Identify samples in low-density regions
            low_density_mask = density_scores < self.density_threshold
            
            if low_density_mask.any():
                # 3. Project low-density samples to nearest high-density regions
                z_constrained[low_density_mask] = self._project_to_manifold(
                    z[low_density_mask], 
                    model,
                    mu_reference[low_density_mask] if mu_reference is not None else None
                )
            
            # 4. Apply elastic recall toward reference points
            if mu_reference is not None:
                z_constrained = self._apply_elastic_recall(z_constrained, mu_reference, model)
            
            return z_constrained
            
        except Exception as e:
            warnings.warn(f"Manifold constraints failed: {e}, returning original samples")
            return z
    
    def _compute_manifold_density(self, z: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """
        Compute manifold density scores based on log det(G^-1(z)) and centroid proximity.
        
        Args:
            z: Latent samples [B, D]
            model: Model with metric computation
            
        Returns:
            density_scores: Density scores [B], higher = more on-manifold
        """
        # Component 1: Metric determinant (geometric density)
        try:
            G_inv = model.G_inv(z)
            log_det_G_inv = torch.logdet(G_inv)
            # Normalize to [0, 1] range approximately
            geometric_density = torch.sigmoid(log_det_G_inv - log_det_G_inv.mean())
        except:
            geometric_density = torch.ones(z.shape[0], device=z.device) * 0.5
        
        # Component 2: Proximity to centroids
        centroids = model.centroids_tens
        distances = torch.cdist(z, centroids)  # [B, K]
        min_distances = torch.min(distances, dim=-1)[0]  # [B]
        
        # Convert distance to proximity score
        proximity_density = torch.exp(-min_distances / 2.0)  # Exponential decay
        
        # Combine both components
        density_scores = 0.6 * geometric_density + 0.4 * proximity_density
        
        return density_scores
    
    def _project_to_manifold(self, 
                           z_low_density: torch.Tensor, 
                           model: nn.Module,
                           mu_reference: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Project low-density samples to nearest high-density manifold regions.
        
        Args:
            z_low_density: Samples in low-density regions [B_low, D]
            model: Model with centroids
            mu_reference: Optional reference points [B_low, D]
            
        Returns:
            z_projected: Projected samples [B_low, D]
        """
        centroids = model.centroids_tens
        
        if mu_reference is not None:
            # Project toward reference points with centroid guidance
            target_points = mu_reference
        else:
            # Project toward nearest centroids
            distances = torch.cdist(z_low_density, centroids)
            nearest_indices = torch.argmin(distances, dim=-1)
            target_points = centroids[nearest_indices]
        
        # Compute projection direction
        projection_direction = target_points - z_low_density
        projection_distance = torch.norm(projection_direction, dim=-1, keepdim=True)
        
        # Limit projection distance
        max_distance = self.max_projection_distance
        projection_distance = torch.clamp(projection_distance, max=max_distance)
        
        # Normalize and scale projection
        projection_direction_normalized = projection_direction / (torch.norm(projection_direction, dim=-1, keepdim=True) + 1e-12)
        projection_vector = projection_direction_normalized * projection_distance * self.projection_strength
        
        z_projected = z_low_density + projection_vector
        
        return z_projected
    
    def _apply_elastic_recall(self, 
                            z: torch.Tensor, 
                            mu_reference: torch.Tensor, 
                            model: nn.Module) -> torch.Tensor:
        """
        Apply elastic recall force toward reference points (encoder means).
        
        Args:
            z: Current samples [B, D]
            mu_reference: Reference encoder means [B, D]
            model: Model for metric computation
            
        Returns:
            z_recalled: Samples with elastic recall applied [B, D]
        """
        # Compute recall direction
        recall_direction = mu_reference - z
        recall_distance = torch.norm(recall_direction, dim=-1, keepdim=True)
        
        # Adaptive recall strength based on distance
        # Stronger recall for samples far from reference
        adaptive_strength = self.elastic_strength * torch.tanh(recall_distance / 2.0)
        
        # Apply metric-aware scaling
        try:
            G_inv = model.G_inv(z)
            # Use metric to weight recall in different directions
            recall_direction_weighted = torch.einsum('bij,bj->bi', G_inv, recall_direction)
            recall_vector = recall_direction_weighted * adaptive_strength
        except:
            # Fallback to isotropic recall
            recall_vector = recall_direction * adaptive_strength
        
        z_recalled = z + recall_vector
        
        return z_recalled
    
    def compute_constraint_metrics(self, 
                                 z_before: torch.Tensor, 
                                 z_after: torch.Tensor, 
                                 model: nn.Module) -> Dict[str, float]:
        """
        Compute metrics to evaluate constraint effectiveness.
        
        Args:
            z_before: Samples before constraints [B, D]
            z_after: Samples after constraints [B, D]
            model: Model for density computation
            
        Returns:
            metrics: Dictionary of constraint metrics
        """
        metrics = {}
        
        # 1. Density improvement
        density_before = self._compute_manifold_density(z_before, model)
        density_after = self._compute_manifold_density(z_after, model)
        
        metrics['density_improvement'] = (density_after.mean() - density_before.mean()).item()
        metrics['density_before'] = density_before.mean().item()
        metrics['density_after'] = density_after.mean().item()
        
        # 2. Movement magnitude
        movement = torch.norm(z_after - z_before, dim=-1)
        metrics['avg_movement'] = movement.mean().item()
        metrics['max_movement'] = movement.max().item()
        
        # 3. Constraint activation rate
        low_density_before = (density_before < self.density_threshold).float().mean().item()
        low_density_after = (density_after < self.density_threshold).float().mean().item()
        
        metrics['low_density_before'] = low_density_before
        metrics['low_density_after'] = low_density_after
        metrics['constraint_effectiveness'] = low_density_before - low_density_after
        
        return metrics


class AdaptiveRHMCConstraints(ManifoldConstraints):
    """
    Adaptive version of manifold constraints that adjusts parameters based on training progress.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Adaptive parameters
        self.initial_projection_strength = self.projection_strength
        self.initial_elastic_strength = self.elastic_strength
        
        # Training progress tracking
        self.step_count = 0
        self.density_history = []
        
    def update_adaptive_parameters(self, current_epoch: int, total_epochs: int):
        """Update constraint parameters based on training progress."""
        
        # Reduce constraint strength as training progresses
        progress = current_epoch / max(total_epochs, 1)
        
        # Stronger constraints early in training, weaker later
        self.projection_strength = self.initial_projection_strength * (1.0 - 0.5 * progress)
        self.elastic_strength = self.initial_elastic_strength * (1.0 - 0.3 * progress)
        
        # Adjust density threshold based on recent history
        if len(self.density_history) > 10:
            recent_density = sum(self.density_history[-10:]) / 10
            self.density_threshold = max(0.05, min(0.2, recent_density * 0.8))
    
    def apply_manifold_constraints(self, z, model, mu_reference=None):
        """Apply constraints with adaptive parameter updates."""
        
        # Track density for adaptation
        if hasattr(model, 'G_inv'):
            try:
                current_density = self._compute_manifold_density(z, model).mean().item()
                self.density_history.append(current_density)
                
                # Keep history bounded
                if len(self.density_history) > 50:
                    self.density_history = self.density_history[-50:]
            except:
                pass
        
        self.step_count += 1
        
        return super().apply_manifold_constraints(z, model, mu_reference)


