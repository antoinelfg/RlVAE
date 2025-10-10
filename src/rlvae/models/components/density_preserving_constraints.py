"""
Density-Preserving Constraints for RHMC Posterior
================================================

Advanced constraints that specifically target density variation reduction.
Ensures samples stay in regions of consistent Riemannian density.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import warnings
import math


class DensityPreservingConstraints(nn.Module):
    """
    Advanced constraints that preserve Riemannian density consistency.
    
    Key features:
    - Density-aware projection to maintain log det(G^-1) consistency
    - Adaptive regularization based on density gradients
    - Volume-preserving transformations
    """
    
    def __init__(self, 
                 target_density_std: float = 0.05,  # Target std for log det(G^-1)
                 density_regularization: float = 0.8,  # Strength of density regularization
                 volume_preservation_weight: float = 0.3,  # Weight for volume preservation
                 adaptive_threshold: float = 0.15,  # Threshold for adaptive intervention
                 max_density_correction: float = 0.5):  # Maximum density correction per step
        super().__init__()
        
        self.target_density_std = target_density_std
        self.density_regularization = density_regularization
        self.volume_preservation_weight = volume_preservation_weight
        self.adaptive_threshold = adaptive_threshold
        self.max_density_correction = max_density_correction
        
        # Tracking for adaptive behavior
        self.density_history = []
        self.reference_density = None
        
    def apply_density_preserving_constraints(self, 
                                           z: torch.Tensor, 
                                           model: nn.Module,
                                           mu_reference: Optional[torch.Tensor] = None,
                                           step_info: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Apply density-preserving constraints to reduce log det(G^-1) variation.
        
        Args:
            z: Current latent samples [B, D]
            model: Model with G_inv method
            mu_reference: Optional reference encoder means [B, D]
            step_info: Optional step information for adaptive behavior
            
        Returns:
            z_constrained: Density-preserving constrained samples [B, D]
        """
        if not hasattr(model, 'G_inv'):
            warnings.warn("Model missing G_inv method, skipping density constraints")
            return z
        
        try:
            # 1. Compute current density distribution
            current_densities = self._compute_log_densities(z, model)
            
            # 2. Establish reference density if not set
            if self.reference_density is None:
                if mu_reference is not None:
                    ref_densities = self._compute_log_densities(mu_reference, model)
                    self.reference_density = ref_densities.mean().item()
                else:
                    self.reference_density = current_densities.mean().item()
            
            # 3. Compute density variation
            density_std = torch.std(current_densities).item()
            
            # 4. Apply constraints if variation exceeds threshold
            if density_std > self.adaptive_threshold:
                z_constrained = self._apply_density_regularization(
                    z, current_densities, model, mu_reference
                )
            else:
                z_constrained = z
            
            # 5. Track density evolution
            self.density_history.append(density_std)
            if len(self.density_history) > 100:
                self.density_history = self.density_history[-100:]
            
            return z_constrained
            
        except Exception as e:
            warnings.warn(f"Density-preserving constraints failed: {e}")
            return z
    
    def _compute_log_densities(self, z: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """
        Compute log det(G^-1(z)) for density measurement.
        
        Args:
            z: Latent samples [B, D]
            model: Model with G_inv method
            
        Returns:
            log_densities: Log densities [B]
        """
        try:
            G_inv = model.G_inv(z)
            log_det_G_inv = torch.logdet(G_inv)
            return log_det_G_inv
        except Exception as e:
            # Fallback: use distance-based density approximation
            warnings.warn(f"G_inv computation failed: {e}, using distance approximation")
            if hasattr(model, 'centroids_tens'):
                distances = torch.cdist(z, model.centroids_tens)
                min_distances = torch.min(distances, dim=-1)[0]
                return -min_distances  # Negative distance as density proxy
            else:
                return torch.zeros(z.shape[0], device=z.device)
    
    def _apply_density_regularization(self, 
                                    z: torch.Tensor, 
                                    current_densities: torch.Tensor, 
                                    model: nn.Module,
                                    mu_reference: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply simplified density regularization to reduce variation.
        
        Strategy:
        1. Compute target density (median for robustness)
        2. Move outlier samples toward regions with target density
        3. Use centroid-based attraction for efficiency
        """
        batch_size = z.shape[0]
        z_corrected = z.clone()
        
        # Use median as target density (more robust than mean)
        target_density = torch.median(current_densities).item()
        
        # Identify samples that deviate significantly from target density
        density_deviations = torch.abs(current_densities - target_density)
        correction_mask = density_deviations > self.adaptive_threshold
        
        if not correction_mask.any():
            return z_corrected
        
        # Apply corrections to outlier samples
        z_outliers = z[correction_mask]
        densities_outliers = current_densities[correction_mask]
        
        # Simplified approach: Move toward nearest centroid with good density
        if hasattr(model, 'centroids_tens'):
            z_corrected_outliers = self._centroid_based_density_correction(
                z_outliers, densities_outliers, target_density, model, mu_reference
            )
        else:
            # Fallback: Move toward reference points
            if mu_reference is not None:
                mu_outliers = mu_reference[correction_mask]
                interpolation_factor = 0.3 * self.density_regularization
                z_corrected_outliers = (1 - interpolation_factor) * z_outliers + interpolation_factor * mu_outliers
            else:
                z_corrected_outliers = z_outliers
        
        z_corrected[correction_mask] = z_corrected_outliers
        
        return z_corrected
    
    def _centroid_based_density_correction(self, 
                                         z: torch.Tensor, 
                                         current_densities: torch.Tensor, 
                                         target_density: float, 
                                         model: nn.Module,
                                         mu_reference: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Move samples toward centroids that have densities closer to target.
        """
        centroids = model.centroids_tens
        
        # Compute density at each centroid
        try:
            centroid_densities = self._compute_log_densities(centroids, model)
        except:
            # Fallback: assume centroids have good density
            centroid_densities = torch.full((len(centroids),), target_density, device=z.device)
        
        # Find best centroid for each sample
        z_corrected = z.clone()
        
        for i, (sample, current_density) in enumerate(zip(z, current_densities)):
            # Find centroid with density closest to target
            density_distances = torch.abs(centroid_densities - target_density)
            best_centroid_idx = torch.argmin(density_distances)
            best_centroid = centroids[best_centroid_idx]
            
            # Move toward best centroid
            direction = best_centroid - sample
            
            # Adaptive step size based on density error
            density_error = abs(current_density - target_density)
            step_size = min(0.2, density_error * 0.1) * self.density_regularization
            
            correction = direction * step_size
            
            # Limit correction magnitude
            correction_magnitude = torch.norm(correction)
            if correction_magnitude > self.max_density_correction:
                correction = correction * (self.max_density_correction / correction_magnitude)
            
            z_corrected[i] = sample + correction
        
        return z_corrected
    
    def _gradient_based_density_correction(self, 
                                         z: torch.Tensor, 
                                         current_densities: torch.Tensor, 
                                         target_density: float, 
                                         model: nn.Module) -> torch.Tensor:
        """
        Use finite differences to approximate density gradient and correct samples.
        """
        z_corrected = z.clone()
        
        try:
            # Use finite differences to approximate gradient of log det(G^-1)
            eps = 1e-4
            batch_size, latent_dim = z.shape
            
            density_gradients = torch.zeros_like(z)
            
            for dim in range(latent_dim):
                # Forward difference
                z_plus = z.clone()
                z_plus[:, dim] += eps
                
                z_minus = z.clone()
                z_minus[:, dim] -= eps
                
                # Compute densities
                try:
                    G_inv_plus = model.G_inv(z_plus)
                    G_inv_minus = model.G_inv(z_minus)
                    
                    log_det_plus = torch.logdet(G_inv_plus)
                    log_det_minus = torch.logdet(G_inv_minus)
                    
                    # Finite difference gradient
                    density_gradients[:, dim] = (log_det_plus - log_det_minus) / (2 * eps)
                    
                except:
                    # If G_inv fails, use distance-based approximation
                    if hasattr(model, 'centroids_tens'):
                        dist_plus = torch.cdist(z_plus, model.centroids_tens).min(dim=-1)[0]
                        dist_minus = torch.cdist(z_minus, model.centroids_tens).min(dim=-1)[0]
                        density_gradients[:, dim] = -(dist_plus - dist_minus) / (2 * eps)
            
            # Normalize gradients
            grad_norm = torch.norm(density_gradients, dim=-1, keepdim=True)
            normalized_grad = density_gradients / (grad_norm + 1e-8)
            
            # Determine correction direction and magnitude
            density_errors = current_densities - target_density
            correction_magnitudes = torch.abs(density_errors).unsqueeze(-1) * 0.05  # Smaller step
            
            # Apply correction in opposite direction of density error
            corrections = -torch.sign(density_errors).unsqueeze(-1) * normalized_grad * correction_magnitudes
            z_corrected = z + corrections
            
        except Exception as e:
            warnings.warn(f"Gradient-based density correction failed: {e}")
            z_corrected = z
        
        return z_corrected
    
    def _reference_based_correction(self, 
                                  z: torch.Tensor, 
                                  mu_reference: torch.Tensor, 
                                  current_densities: torch.Tensor, 
                                  target_density: float, 
                                  model: nn.Module) -> torch.Tensor:
        """
        Use reference points (encoder means) to guide density correction.
        """
        # Compute reference densities
        try:
            ref_densities = self._compute_log_densities(mu_reference, model)
            
            # Find references with densities close to target
            density_distances = torch.abs(ref_densities - target_density)
            
            # Weight corrections based on reference quality
            ref_weights = torch.exp(-density_distances / 0.1)  # Exponential weighting
            
            # Interpolate toward good references
            interpolation_factors = ref_weights * 0.3  # Conservative interpolation
            
            z_corrected = (1 - interpolation_factors.unsqueeze(-1)) * z + \
                         interpolation_factors.unsqueeze(-1) * mu_reference
            
            return z_corrected
            
        except Exception as e:
            warnings.warn(f"Reference-based correction failed: {e}")
            return z
    
    def compute_density_metrics(self, 
                              z_before: torch.Tensor, 
                              z_after: torch.Tensor, 
                              model: nn.Module) -> Dict[str, float]:
        """
        Compute metrics to evaluate density preservation effectiveness.
        """
        metrics = {}
        
        try:
            # Density variations before and after
            densities_before = self._compute_log_densities(z_before, model)
            densities_after = self._compute_log_densities(z_after, model)
            
            std_before = torch.std(densities_before).item()
            std_after = torch.std(densities_after).item()
            
            metrics['density_std_before'] = std_before
            metrics['density_std_after'] = std_after
            metrics['density_improvement'] = std_before - std_after
            metrics['density_improvement_pct'] = (std_before - std_after) / std_before * 100 if std_before > 0 else 0
            
            # Mean density preservation
            mean_before = torch.mean(densities_before).item()
            mean_after = torch.mean(densities_after).item()
            
            metrics['mean_density_shift'] = abs(mean_after - mean_before)
            
            # Success criteria
            metrics['target_achieved'] = std_after < self.target_density_std
            
        except Exception as e:
            warnings.warn(f"Density metrics computation failed: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def get_adaptive_status(self) -> Dict[str, Any]:
        """Get current adaptive status and recommendations."""
        status = {
            'reference_density': self.reference_density,
            'recent_density_std': self.density_history[-10:] if len(self.density_history) >= 10 else self.density_history,
            'avg_recent_std': sum(self.density_history[-10:]) / len(self.density_history[-10:]) if len(self.density_history) >= 10 else None
        }
        
        # Recommendations
        if status['avg_recent_std'] is not None:
            if status['avg_recent_std'] > self.adaptive_threshold * 2:
                status['recommendation'] = "Increase density_regularization strength"
            elif status['avg_recent_std'] < self.target_density_std:
                status['recommendation'] = "Density variation under control, consider reducing regularization"
            else:
                status['recommendation'] = "Current settings appropriate"
        
        return status


class VolumePreservingRHMCConstraints(DensityPreservingConstraints):
    """
    Extension that adds explicit volume preservation to density constraints.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.volume_history = []
    
    def apply_density_preserving_constraints(self, z, model, mu_reference=None, step_info=None):
        """Apply constraints with volume preservation tracking."""
        
        # Compute initial volume (determinant of metric)
        try:
            G_initial = model.G(z) if hasattr(model, 'G') else None
            initial_volume = torch.det(G_initial).mean().item() if G_initial is not None else None
        except:
            initial_volume = None
        
        # Apply density constraints
        z_constrained = super().apply_density_preserving_constraints(z, model, mu_reference, step_info)
        
        # Track volume preservation
        if initial_volume is not None:
            try:
                G_final = model.G(z_constrained)
                final_volume = torch.det(G_final).mean().item()
                volume_change = abs(final_volume - initial_volume) / initial_volume
                
                self.volume_history.append(volume_change)
                if len(self.volume_history) > 50:
                    self.volume_history = self.volume_history[-50:]
                
                # Apply volume correction if change is too large
                if volume_change > 0.1:  # 10% volume change threshold
                    z_constrained = self._apply_volume_correction(z, z_constrained, model)
                    
            except Exception as e:
                warnings.warn(f"Volume tracking failed: {e}")
        
        return z_constrained
    
    def _apply_volume_correction(self, z_original, z_constrained, model):
        """Apply correction to preserve volume when constraints are too aggressive."""
        
        # Simple approach: reduce the magnitude of corrections
        correction = z_constrained - z_original
        correction_reduced = correction * 0.5  # Reduce correction by half
        
        return z_original + correction_reduced
