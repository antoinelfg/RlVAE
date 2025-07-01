"""
Working Riemannian Sampler
=========================

Enhanced training sampling with multiple Riemannian methods.
"""

import torch
from typing import Dict, Any
from .base_sampler import BaseRiemannianSampler


class WorkingRiemannianSampler(BaseRiemannianSampler):
    """Working Riemannian sampler based on successful test_rhvae_sampling.py approach."""
    
    def __init__(self, model):
        super().__init__(model)
        
    def sample_riemannian_latents(self, mu: torch.Tensor, log_var: torch.Tensor, 
                                 method: str = 'enhanced') -> torch.Tensor:
        """
        Sample latent codes using improved Riemannian approach.
        
        Methods:
        - 'geodesic': Geodesic-aware training sampling (BEST FOR MANIFOLD)
        - 'enhanced': Original enhanced training sampling
        - 'basic': Basic metric-aware sampling
        - 'standard': Standard reparameterization
        """
        if method == 'geodesic':
            return self.sample_geodesic_riemannian_latents(mu, log_var)
        elif method == 'enhanced':
            return self.sample_enhanced_riemannian_latents(mu, log_var)
        elif method == 'basic':
            return self.sample_basic_riemannian_latents(mu, log_var)
        else:
            # Standard reparameterization
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(0.5 * log_var)
    
    def sample_enhanced_riemannian_latents(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        🏆 ENHANCED: Geodesic-inspired training sampling.
        
        This method uses insights from the geodesic approach but adapted for training:
        1. Find nearest centroids to mu
        2. Use metric-aware transformation with centroid influence
        3. Preserve gradients for training
        """
        # Standard reparameterization as base
        eps = torch.randn_like(mu)
        z_standard = mu + eps * torch.exp(0.5 * log_var)
        
        # Apply enhanced Riemannian transformation if metric available
        if self.validate_metric_availability():
            try:
                # 1. Find nearest centroids to the posterior means
                centroids = self.model.centroids_tens  # [K, D]
                mu_expanded = mu.unsqueeze(1)  # [B, 1, D]
                centroids_expanded = centroids.unsqueeze(0)  # [1, K, D]
                distances = torch.norm(mu_expanded - centroids_expanded, dim=-1)  # [B, K]
                
                # 2. Use top-2 nearest centroids for "geodesic-like" influence
                _, top2_indices = torch.topk(distances, k=2, dim=-1, largest=False)  # [B, 2]
                
                # 3. Compute interpolation weights based on distances
                top2_distances = torch.gather(distances, 1, top2_indices)  # [B, 2]
                # Use inverse distance weighting (closer = higher weight)
                weights = 1.0 / (top2_distances + 1e-8)
                weights = weights / weights.sum(dim=-1, keepdim=True)  # [B, 2]
                
                # 4. Create "virtual centroid" by weighted interpolation
                centroid1 = self.model.centroids_tens[top2_indices[:, 0]]  # [B, D]
                centroid2 = self.model.centroids_tens[top2_indices[:, 1]]  # [B, D]
                virtual_centroid = weights[:, 0:1] * centroid1 + weights[:, 1:2] * centroid2  # [B, D]
                
                # 5. Compute metric at virtual centroid (geodesic-inspired)
                G_inv_virtual = self.model.G_inv(virtual_centroid)  # [B, D, D]
                
                # 6. Apply metric transformation with centroid influence
                try:
                    # Cholesky decomposition for stability
                    L = torch.linalg.cholesky(G_inv_virtual + 1e-6 * torch.eye(G_inv_virtual.shape[-1], device=G_inv_virtual.device))
                    eps_transformed = torch.einsum('bij,bj->bi', L, eps)
                except:
                    # Fallback to eigendecomposition
                    eigenvals, eigenvecs = torch.linalg.eigh(G_inv_virtual)
                    eigenvals = torch.clamp(eigenvals, min=1e-6)
                    sqrt_G_inv = eigenvecs @ torch.diag_embed(torch.sqrt(eigenvals)) @ eigenvecs.transpose(-2, -1)
                    eps_transformed = torch.einsum('bij,bj->bi', sqrt_G_inv, eps)
                
                # 7. Gentle mixing with standard sampling (preserve gradients)
                centroid_influence = 0.15  # Small influence to maintain training stability
                z_enhanced = mu + eps_transformed * torch.exp(0.5 * log_var) * centroid_influence + \
                            eps * torch.exp(0.5 * log_var) * (1.0 - centroid_influence)
                
                return z_enhanced
                
            except Exception as e:
                print(f"⚠️ Enhanced Riemannian sampling failed: {e}, using basic method")
                return self.sample_basic_riemannian_latents(mu, log_var)
        
        return z_standard
    
    def sample_geodesic_riemannian_latents(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        🚀 GEODESIC: Geodesic-aware training sampling.
        
        This method uses proper geodesic concepts:
        1. Find nearest centroid pairs to posterior mean
        2. Sample along geodesic-like path between centroids  
        3. Apply metric-aware perturbation perpendicular to geodesic
        4. Preserve gradients for training
        """
        # Standard reparameterization as fallback
        eps = torch.randn_like(mu)
        z_standard = mu + eps * torch.exp(0.5 * log_var)
        
        # Apply geodesic Riemannian transformation if metric available
        if self.validate_metric_availability():
            try:
                batch_size = mu.shape[0]
                
                # 1. Find two nearest centroids to the posterior mean
                centroids = self.model.centroids_tens  # [K, D]
                mu_expanded = mu.unsqueeze(1)  # [B, 1, D]
                centroids_expanded = centroids.unsqueeze(0)  # [1, K, D]
                distances = torch.norm(mu_expanded - centroids_expanded, dim=-1)  # [B, K]
                
                # Get 2 nearest centroids for each batch element
                _, nearest_indices = torch.topk(distances, k=2, dim=-1, largest=False)  # [B, 2]
                
                # 2. Create geodesic path between nearest centroids
                centroid_1 = self.model.centroids_tens[nearest_indices[:, 0]]  # [B, D]
                centroid_2 = self.model.centroids_tens[nearest_indices[:, 1]]  # [B, D]
                
                # Sample random position along geodesic (simplified as linear interpolation)
                t_geodesic = torch.rand(batch_size, 1, device=mu.device)  # [B, 1]
                z_geodesic = (1 - t_geodesic) * centroid_1 + t_geodesic * centroid_2  # [B, D]
                
                # 3. Create geodesic direction vector
                geodesic_direction = centroid_2 - centroid_1  # [B, D]
                geodesic_direction = geodesic_direction / (torch.norm(geodesic_direction, dim=-1, keepdim=True) + 1e-8)
                
                # 4. Project posterior mean perturbation perpendicular to geodesic
                mu_to_geodesic = mu - z_geodesic  # [B, D]
                parallel_component = torch.sum(mu_to_geodesic * geodesic_direction, dim=-1, keepdim=True) * geodesic_direction
                perpendicular_component = mu_to_geodesic - parallel_component  # [B, D]
                
                # 5. Apply metric transformation at geodesic point
                G_inv_geodesic = self.model.G_inv(z_geodesic)  # [B, D, D]
                
                # Use proper metric (inverse of G_inv)
                try:
                    G_geodesic = torch.linalg.inv(G_inv_geodesic)  # [B, D, D]
                    L = torch.linalg.cholesky(G_geodesic + 1e-6 * torch.eye(G_geodesic.shape[-1], device=G_geodesic.device))
                    eps_perpendicular = torch.einsum('bij,bj->bi', L, eps)
                except:
                    # Fallback to eigendecomposition
                    G_geodesic = torch.linalg.inv(G_inv_geodesic)
                    eigenvals, eigenvecs = torch.linalg.eigh(G_geodesic)
                    eigenvals = torch.clamp(eigenvals, min=1e-6)
                    sqrt_G = eigenvecs @ torch.diag_embed(torch.sqrt(eigenvals)) @ eigenvecs.transpose(-2, -1)
                    eps_perpendicular = torch.einsum('bij,bj->bi', sqrt_G, eps)
                
                # 6. Combine: geodesic position + perpendicular metric noise + parallel posterior displacement
                geodesic_scale = 0.3  # Influence of geodesic structure
                z_geodesic_final = (
                    z_geodesic +  # Start from geodesic
                    geodesic_scale * eps_perpendicular * torch.exp(0.5 * log_var) +  # Perpendicular metric noise
                    (1.0 - geodesic_scale) * (mu - z_geodesic) +  # Bias toward posterior mean
                    0.1 * parallel_component  # Small parallel component
                )
                
                return z_geodesic_final
                
            except Exception as e:
                print(f"⚠️ Geodesic Riemannian sampling failed: {e}, using standard method")
                return z_standard
        
        return z_standard
    
    def sample_basic_riemannian_latents(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Basic Riemannian training sampling (original method).
        """
        # Standard reparameterization (ALWAYS preserve gradients)
        eps = torch.randn_like(mu)
        z_samples = mu + eps * torch.exp(0.5 * log_var)
        
        # Apply Riemannian refinement if metric available
        if self.validate_metric_availability():
            try:
                # Compute metric tensor at the current sample points
                G_inv_z = self.model.G_inv(z_samples)  # [batch_size, latent_dim, latent_dim]
                
                # Metric-aware noise correction using Cholesky decomposition for stability
                try:
                    # Try Cholesky decomposition first (more stable)
                    L = torch.linalg.cholesky(G_inv_z + 1e-6 * torch.eye(G_inv_z.shape[-1], device=G_inv_z.device))
                    eps_transformed = torch.einsum('bij,bj->bi', L, eps)
                except:
                    # Fallback to eigendecomposition if Cholesky fails
                    eigenvals, eigenvecs = torch.linalg.eigh(G_inv_z)
                    eigenvals = torch.clamp(eigenvals, min=1e-6)  # Numerical stability
                    sqrt_G_inv = eigenvecs @ torch.diag_embed(torch.sqrt(eigenvals)) @ eigenvecs.transpose(-2, -1)
                    eps_transformed = torch.einsum('bij,bj->bi', sqrt_G_inv, eps)
                
                # Apply the metric-aware correction with small scale
                correction_scale = 0.1  # Small scale to avoid disrupting gradients
                z_corrected = mu + eps_transformed * torch.exp(0.5 * log_var) * correction_scale + \
                             eps * torch.exp(0.5 * log_var) * (1.0 - correction_scale)
                
                return z_corrected
                
            except Exception as e:
                print(f"⚠️ Basic Riemannian sampling failed: {e}, using standard method")
                return z_samples
        
        return z_samples
    
    def sample_prior(self, num_samples: int, method: str = 'geodesic') -> torch.Tensor:
        """
        Sample from the Riemannian prior using various methods.
        
        Args:
            num_samples: Number of samples to generate
            method: Prior sampling method ('geodesic', 'centroid_aware', 'weighted_mixture', 'basic')
            
        Returns:
            Prior samples [num_samples, latent_dim]
        """
        if method == 'geodesic':
            return self.sample_geodesic_prior(num_samples)
        elif method == 'centroid_aware':
            return self.sample_centroid_aware_prior(num_samples)
        elif method == 'weighted_mixture':
            return self.sample_weighted_mixture_prior(num_samples)
        else:
            return self.sample_basic_prior(num_samples)
    
    def sample_geodesic_prior(self, num_samples: int) -> torch.Tensor:
        """
        Sample from prior using geodesic-aware approach.
        """
        if not self.validate_metric_availability():
            # Fallback to standard Gaussian
            return torch.randn(num_samples, self.model.latent_dim, device=self.device)
        
        try:
            # Sample random centroids and interpolate along geodesics
            centroids = self.model.centroids_tens  # [K, D]
            n_centroids = centroids.shape[0]
            
            # Sample random pairs of centroids
            idx1 = torch.randint(0, n_centroids, (num_samples,), device=centroids.device)
            idx2 = torch.randint(0, n_centroids, (num_samples,), device=centroids.device)
            
            centroid1 = centroids[idx1]  # [num_samples, D]
            centroid2 = centroids[idx2]  # [num_samples, D]
            
            # Sample along geodesic between centroids
            t = torch.rand(num_samples, 1, device=centroids.device)  # [num_samples, 1]
            z_geodesic = (1 - t) * centroid1 + t * centroid2  # [num_samples, D]
            
            # Add small metric-aware noise
            eps = torch.randn_like(z_geodesic)
            G_inv = self.model.G_inv(z_geodesic)  # [num_samples, D, D]
            
            try:
                L = torch.linalg.cholesky(G_inv + 1e-6 * torch.eye(G_inv.shape[-1], device=G_inv.device))
                eps_transformed = torch.einsum('bij,bj->bi', L, eps)
            except:
                # Fallback to eigendecomposition
                eigenvals, eigenvecs = torch.linalg.eigh(G_inv)
                eigenvals = torch.clamp(eigenvals, min=1e-6)
                sqrt_G_inv = eigenvecs @ torch.diag_embed(torch.sqrt(eigenvals)) @ eigenvecs.transpose(-2, -1)
                eps_transformed = torch.einsum('bij,bj->bi', sqrt_G_inv, eps)
            
            # Combine geodesic position with metric noise
            noise_scale = 0.1
            z_final = z_geodesic + noise_scale * eps_transformed
            
            return z_final
            
        except Exception as e:
            print(f"⚠️ Geodesic prior sampling failed: {e}, using basic method")
            return self.sample_basic_prior(num_samples)
    
    def sample_centroid_aware_prior(self, num_samples: int) -> torch.Tensor:
        """
        Sample from prior using centroid-aware approach.
        """
        if not self.validate_metric_availability():
            # Fallback to standard Gaussian
            return torch.randn(num_samples, self.model.latent_dim, device=self.device)
        
        try:
            # Sample centroids with probability proportional to their "importance"
            centroids = self.model.centroids_tens  # [K, D]
            n_centroids = centroids.shape[0]
            
            # Simple uniform sampling of centroids
            centroid_indices = torch.randint(0, n_centroids, (num_samples,), device=self.device)
            z_centroids = centroids[centroid_indices]  # [num_samples, D]
            
            # Add small Gaussian noise around centroids
            noise_std = 0.1
            z_noise = torch.randn_like(z_centroids) * noise_std
            z_final = z_centroids + z_noise
            
            return z_final
            
        except Exception as e:
            print(f"⚠️ Centroid-aware prior sampling failed: {e}, using basic method")
            return self.sample_basic_prior(num_samples)
    
    def sample_weighted_mixture_prior(self, num_samples: int) -> torch.Tensor:
        """
        Sample from prior using weighted mixture of centroids.
        """
        if not self.validate_metric_availability():
            # Fallback to standard Gaussian
            return torch.randn(num_samples, self.model.latent_dim, device=self.device)
        
        try:
            # Sample from mixture of Gaussians centered at centroids
            centroids = self.model.centroids_tens  # [K, D]
            n_centroids = centroids.shape[0]
            
            # Sample mixture components
            component_indices = torch.randint(0, n_centroids, (num_samples,), device=self.device)
            
            # Sample from each component
            z_samples = torch.zeros(num_samples, self.model.latent_dim, device=self.device)
            
            for i in range(n_centroids):
                mask = (component_indices == i)
                if mask.sum() > 0:
                    # Sample from Gaussian around this centroid
                    centroid = centroids[i]  # [D]
                    noise = torch.randn(mask.sum(), self.model.latent_dim, device=self.device) * 0.1
                    z_samples[mask] = centroid + noise
            
            return z_samples
            
        except Exception as e:
            print(f"⚠️ Weighted mixture prior sampling failed: {e}, using basic method")
            return self.sample_basic_prior(num_samples)
    
    def sample_basic_prior(self, num_samples: int) -> torch.Tensor:
        """
        Sample from basic prior (standard Gaussian).
        """
        return torch.randn(num_samples, self.model.latent_dim, device=self.device)
    
    def get_sampling_methods(self) -> Dict[str, str]:
        """Override to provide sampler-specific methods."""
        return {
            'enhanced': 'Enhanced Riemannian sampling with centroid influence',
            'geodesic': 'Geodesic-aware sampling along manifold paths',
            'basic': 'Basic metric-aware sampling',
            'standard': 'Standard reparameterization (no Riemannian)'
        } 