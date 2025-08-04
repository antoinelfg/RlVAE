"""
FID (Fréchet Inception Distance) Score Module
============================================

This module implements FID score computation for evaluating generative model quality.
FID measures the distance between distributions of real and generated images using
features from a pre-trained Inception network.

Key Features:
- Pre-trained Inception-v3 feature extraction
- Efficient batch processing
- Memory-optimized computation for large datasets
- Support for different image formats and resolutions
- Integration with PyTorch tensors and PIL images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np
from scipy.linalg import sqrtm
from typing import Union, Tuple, List, Optional
import warnings
from pathlib import Path
import pickle


class InceptionFeatureExtractor(nn.Module):
    """
    Inception-v3 feature extractor for FID computation.
    
    Extracts features from the final average pooling layer (before classification).
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained Inception-v3
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, 
                                    transform_input=False)
        self.inception.eval()
        
        # Remove the final classification layers
        # We want features from the final average pooling layer (2048-dimensional)
        self.inception.fc = nn.Identity()
        self.inception.AuxLogits = None  # Remove auxiliary classifier
        
        # Move to device
        self.inception = self.inception.to(self.device)
        
        # Standard ImageNet preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),  # Inception-v3 input size
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract Inception features from input images.
        
        Args:
            x: Input images [batch_size, 3, H, W] in range [0, 1]
            
        Returns:
            features: Inception features [batch_size, 2048]
        """
        # Ensure input is in correct format
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"Expected input shape [B, 3, H, W], got {x.shape}")
        
        # Preprocess
        x = self.preprocess(x)
        
        # Extract features
        with torch.no_grad():
            features = self.inception(x)
            
        return features


class FIDScorer:
    """
    FID (Fréchet Inception Distance) Score computation.
    
    FID measures the distance between two multivariate Gaussians fitted to
    Inception features of real and generated images:
    
    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^(1/2))
    """
    
    def __init__(self, device: Optional[torch.device] = None, cache_dir: Optional[str] = None):
        """
        Initialize FID scorer.
        
        Args:
            device: Device for computation
            cache_dir: Directory to cache real image statistics
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize feature extractor
        print("🚀 Loading Inception-v3 for FID computation...")
        self.feature_extractor = InceptionFeatureExtractor(device=self.device)
        
        # Cache for real image statistics
        self._real_stats_cache = {}
        
    def compute_features(self, images: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        """
        Compute Inception features for a batch of images.
        
        Args:
            images: Input images [N, 3, H, W] in range [0, 1]
            batch_size: Batch size for processing
            
        Returns:
            features: Inception features [N, 2048]
        """
        if len(images) == 0:
            return torch.empty(0, 2048, device=self.device)
        
        # Ensure images are on correct device
        images = images.to(self.device)
        
        # Process in batches to avoid memory issues
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            features = self.feature_extractor(batch)
            all_features.append(features.cpu())
        
        return torch.cat(all_features, dim=0)
    
    def compute_statistics(self, features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and covariance from features.
        
        Args:
            features: Inception features [N, 2048]
            
        Returns:
            mu: Mean vector [2048]
            sigma: Covariance matrix [2048, 2048]
        """
        features_np = features.cpu().numpy()
        
        mu = np.mean(features_np, axis=0)
        sigma = np.cov(features_np, rowvar=False)
        
        return mu, sigma
    
    def compute_fid(self, mu1: np.ndarray, sigma1: np.ndarray, 
                   mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
        """
        Compute FID score between two distributions.
        
        Args:
            mu1, sigma1: Mean and covariance of first distribution
            mu2, sigma2: Mean and covariance of second distribution
            eps: Small value for numerical stability
            
        Returns:
            fid_score: FID score (lower is better)
        """
        # Ensure inputs are float64 for numerical stability
        mu1, mu2 = mu1.astype(np.float64), mu2.astype(np.float64)
        sigma1, sigma2 = sigma1.astype(np.float64), sigma2.astype(np.float64)
        
        # Compute mean difference
        mu_diff = mu1 - mu2
        
        # Compute sqrt of product of covariances
        # Add small regularization for numerical stability
        sigma1_reg = sigma1 + eps * np.eye(sigma1.shape[0])
        sigma2_reg = sigma2 + eps * np.eye(sigma2.shape[0])
        
        try:
            # Compute matrix square root: sqrt(sigma1 @ sigma2)
            product = sigma1_reg @ sigma2_reg
            sqrt_product = sqrtm(product)
            
            # Ensure result is real (numerical errors can introduce tiny imaginary parts)
            if np.iscomplexobj(sqrt_product):
                if np.allclose(sqrt_product.imag, 0, atol=1e-3):
                    sqrt_product = sqrt_product.real
                else:
                    warnings.warn("Matrix square root has large imaginary component. "
                                "This may indicate numerical instability.")
                    sqrt_product = sqrt_product.real
            
        except Exception as e:
            warnings.warn(f"Matrix square root computation failed: {e}. Using eigendecomposition fallback.")
            # Fallback: use eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh(product)
            eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
            sqrt_product = eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.T
        
        # Compute FID
        fid_score = (np.dot(mu_diff, mu_diff) + 
                    np.trace(sigma1_reg) + np.trace(sigma2_reg) - 
                    2 * np.trace(sqrt_product))
        
        return float(fid_score)
    
    def evaluate_generation(self, real_images: torch.Tensor, 
                           generated_images: torch.Tensor,
                           batch_size: int = 32) -> dict:
        """
        Evaluate generation quality using FID score.
        
        Args:
            real_images: Real images [N, 3, H, W] in range [0, 1]
            generated_images: Generated images [M, 3, H, W] in range [0, 1]
            batch_size: Batch size for feature computation
            
        Returns:
            results: Dictionary containing FID score and additional metrics
        """
        print(f"🎯 Computing FID score for {len(real_images)} real and {len(generated_images)} generated images...")
        
        # Compute features
        print("   📊 Extracting features from real images...")
        real_features = self.compute_features(real_images, batch_size=batch_size)
        
        print("   📊 Extracting features from generated images...")
        gen_features = self.compute_features(generated_images, batch_size=batch_size)
        
        # Compute statistics
        print("   📈 Computing distribution statistics...")
        real_mu, real_sigma = self.compute_statistics(real_features)
        gen_mu, gen_sigma = self.compute_statistics(gen_features)
        
        # Compute FID
        print("   🧮 Computing FID score...")
        fid_score = self.compute_fid(real_mu, real_sigma, gen_mu, gen_sigma)
        
        # Additional diagnostics
        feature_distance = np.linalg.norm(real_mu - gen_mu)
        real_var = np.trace(real_sigma) / len(real_sigma)
        gen_var = np.trace(gen_sigma) / len(gen_sigma)
        
        results = {
            'fid_score': fid_score,
            'feature_distance': feature_distance,
            'real_feature_variance': real_var,
            'generated_feature_variance': gen_var,
            'n_real_images': len(real_images),
            'n_generated_images': len(generated_images),
        }
        
        print(f"   ✅ FID Score: {fid_score:.3f}")
        
        return results
    
    def cache_real_statistics(self, real_images: torch.Tensor, 
                             cache_key: str, batch_size: int = 32) -> None:
        """
        Cache statistics for real images to avoid recomputation.
        
        Args:
            real_images: Real images [N, 3, H, W] in range [0, 1]
            cache_key: Key for caching (e.g., dataset name)
            batch_size: Batch size for feature computation
        """
        print(f"🗄️ Caching real image statistics with key: {cache_key}")
        
        # Compute features and statistics
        real_features = self.compute_features(real_images, batch_size=batch_size)
        real_mu, real_sigma = self.compute_statistics(real_features)
        
        # Store in memory cache
        self._real_stats_cache[cache_key] = {
            'mu': real_mu,
            'sigma': real_sigma,
            'n_images': len(real_images)
        }
        
        # Store in disk cache if directory provided
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True)
            cache_file = self.cache_dir / f"real_stats_{cache_key}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(self._real_stats_cache[cache_key], f)
            
            print(f"   💾 Saved to disk: {cache_file}")
    
    def load_real_statistics(self, cache_key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load cached real image statistics.
        
        Args:
            cache_key: Key for cached statistics
            
        Returns:
            (mu, sigma) if found, None otherwise
        """
        # Check memory cache first
        if cache_key in self._real_stats_cache:
            stats = self._real_stats_cache[cache_key]
            return stats['mu'], stats['sigma']
        
        # Check disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"real_stats_{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        stats = pickle.load(f)
                    
                    # Store in memory for future use
                    self._real_stats_cache[cache_key] = stats
                    
                    print(f"📁 Loaded cached statistics from: {cache_file}")
                    return stats['mu'], stats['sigma']
                    
                except Exception as e:
                    warnings.warn(f"Failed to load cached statistics: {e}")
        
        return None
    
    def evaluate_with_cached_real(self, generated_images: torch.Tensor,
                                 real_cache_key: str, batch_size: int = 32) -> Optional[dict]:
        """
        Evaluate generation using cached real image statistics.
        
        Args:
            generated_images: Generated images [M, 3, H, W] in range [0, 1]
            real_cache_key: Key for cached real statistics
            batch_size: Batch size for feature computation
            
        Returns:
            results: Dictionary with FID score, or None if cache not found
        """
        # Load cached real statistics
        real_stats = self.load_real_statistics(real_cache_key)
        if real_stats is None:
            print(f"❌ No cached statistics found for key: {real_cache_key}")
            return None
        
        real_mu, real_sigma = real_stats
        
        print(f"🎯 Computing FID score using cached real statistics ({real_cache_key})...")
        
        # Compute generated features and statistics
        print("   📊 Extracting features from generated images...")
        gen_features = self.compute_features(generated_images, batch_size=batch_size)
        gen_mu, gen_sigma = self.compute_statistics(gen_features)
        
        # Compute FID
        print("   🧮 Computing FID score...")
        fid_score = self.compute_fid(real_mu, real_sigma, gen_mu, gen_sigma)
        
        # Additional diagnostics
        feature_distance = np.linalg.norm(real_mu - gen_mu)
        real_var = np.trace(real_sigma) / len(real_sigma)
        gen_var = np.trace(gen_sigma) / len(gen_sigma)
        
        results = {
            'fid_score': fid_score,
            'feature_distance': feature_distance,
            'real_feature_variance': real_var,
            'generated_feature_variance': gen_var,
            'n_generated_images': len(generated_images),
            'real_cache_key': real_cache_key,
        }
        
        print(f"   ✅ FID Score: {fid_score:.3f}")
        
        return results


def create_fid_scorer(device: Optional[torch.device] = None, 
                     cache_dir: str = "data/fid_cache") -> FIDScorer:
    """
    Factory function to create FID scorer with default settings.
    
    Args:
        device: Device for computation
        cache_dir: Directory for caching real image statistics
        
    Returns:
        FID scorer instance
    """
    return FIDScorer(device=device, cache_dir=cache_dir) 