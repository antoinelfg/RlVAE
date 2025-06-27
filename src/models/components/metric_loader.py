"""
Metric Loader Module

This module provides utilities for loading and validating pretrained metric tensors
from various file formats and sources.

Key Features:
- Multiple file format support (.pt, .pth, .pkl)
- Comprehensive data validation
- Flexible key mapping for different file structures
- Error handling and recovery
- Device management
"""

import torch
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import warnings


class MetricLoader:
    """
    Utility class for loading pretrained Riemannian metric tensors.
    
    Handles various file formats and data structures commonly used
    for storing metric tensor components (centroids, matrices, parameters).
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_from_file(
        self, 
        path: Union[str, Path],
        temperature_override: Optional[float] = None,
        regularization_override: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Load metric data from file.
        
        Args:
            path: Path to metric file
            temperature_override: Override temperature parameter
            regularization_override: Override regularization parameter
            
        Returns:
            Dictionary containing:
            - 'centroids': [n_centroids, latent_dim]
            - 'metric_matrices': [n_centroids, latent_dim, latent_dim]  
            - 'temperature': scalar
            - 'regularization': scalar
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Metric file not found: {path}")
        
        print(f"🔧 Loading metric data from: {path}")
        
        # Load raw data
        try:
            raw_data = torch.load(path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load metric file: {e}")
        
        # Extract and validate components
        centroids = self._extract_centroids(raw_data)
        metric_matrices = self._extract_metric_matrices(raw_data, centroids.shape)
        temperature = self._extract_temperature(raw_data, temperature_override)
        regularization = self._extract_regularization(raw_data, regularization_override)
        
        # Validate consistency
        self._validate_data_consistency(centroids, metric_matrices)
        
        result = {
            'centroids': centroids,
            'metric_matrices': metric_matrices,
            'temperature': temperature,
            'regularization': regularization,
        }
        
        print(f"✅ Loaded metric: {len(centroids)} centroids, {centroids.shape[1]}D, T={temperature:.3f}, λ={regularization:.3f}")
        
        return result
    
    def _extract_centroids(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract centroids from raw data."""
        # Try different possible keys
        possible_keys = ['centroids', 'metric_centroids', 'centers', 'mu']
        
        for key in possible_keys:
            if key in data:
                centroids = data[key]
                if isinstance(centroids, torch.Tensor):
                    return centroids.to(self.device)
                else:
                    return torch.tensor(centroids, device=self.device)
        
        raise ValueError(f"No centroids found. Expected one of: {possible_keys}")
    
    def _extract_metric_matrices(
        self, 
        data: Dict[str, Any], 
        centroid_shape: torch.Size
    ) -> torch.Tensor:
        """Extract metric matrices from raw data."""
        n_centroids, latent_dim = centroid_shape
        
        # Try different possible keys and formats
        if 'M_matrices' in data:
            matrices = data['M_matrices']
        elif 'metric_vars' in data:
            matrices = data['metric_vars']
        elif 'M_i_flat' in data:
            # Convert from flattened diagonal format
            flat = data['M_i_flat']
            if isinstance(flat, torch.Tensor):
                flat = flat.to(self.device)
            else:
                flat = torch.tensor(flat, device=self.device)
            matrices = torch.diag_embed(flat)  # [n_centroids, latent_dim, latent_dim]
        elif 'M_tens' in data:
            matrices = data['M_tens']
        else:
            # Try to construct identity matrices as fallback
            warnings.warn("No metric matrices found, using identity matrices")
            matrices = torch.eye(latent_dim, device=self.device).unsqueeze(0).repeat(n_centroids, 1, 1)
        
        # Convert to tensor if needed
        if not isinstance(matrices, torch.Tensor):
            matrices = torch.tensor(matrices, device=self.device)
        else:
            matrices = matrices.to(self.device)
        
        # Validate shape
        expected_shape = (n_centroids, latent_dim, latent_dim)
        if matrices.shape != expected_shape:
            raise ValueError(f"Metric matrices shape {matrices.shape} != expected {expected_shape}")
        
        return matrices
    
    def _extract_temperature(
        self, 
        data: Dict[str, Any], 
        override: Optional[float]
    ) -> float:
        """Extract temperature parameter."""
        if override is not None:
            return float(override)
        
        # Try different possible keys
        possible_keys = ['temperature', 'temp', 'T', 'beta']
        
        for key in possible_keys:
            if key in data:
                temp = data[key]
                if isinstance(temp, torch.Tensor):
                    return float(temp.item())
                else:
                    return float(temp)
        
        # Default value
        default_temp = 0.1
        warnings.warn(f"No temperature found, using default: {default_temp}")
        return default_temp
    
    def _extract_regularization(
        self, 
        data: Dict[str, Any], 
        override: Optional[float]
    ) -> float:
        """Extract regularization parameter."""
        if override is not None:
            return float(override)
        
        # Try different possible keys
        possible_keys = ['regularization', 'reg', 'lambda', 'lbd']
        
        for key in possible_keys:
            if key in data:
                reg = data[key]
                if isinstance(reg, torch.Tensor):
                    return float(reg.item())
                else:
                    return float(reg)
        
        # Default value
        default_reg = 0.01
        warnings.warn(f"No regularization found, using default: {default_reg}")
        return default_reg
    
    def _validate_data_consistency(
        self, 
        centroids: torch.Tensor, 
        metric_matrices: torch.Tensor
    ) -> None:
        """Validate that loaded data is consistent."""
        n_centroids, latent_dim = centroids.shape
        
        # Check metric matrices shape
        if metric_matrices.shape != (n_centroids, latent_dim, latent_dim):
            raise ValueError(f"Inconsistent shapes: centroids {centroids.shape}, matrices {metric_matrices.shape}")
        
        # Check for NaN or inf values
        if torch.isnan(centroids).any() or torch.isinf(centroids).any():
            raise ValueError("Centroids contain NaN or inf values")
        
        if torch.isnan(metric_matrices).any() or torch.isinf(metric_matrices).any():
            raise ValueError("Metric matrices contain NaN or inf values")
        
        # Check that matrices are positive definite (or at least positive semidefinite)
        for i in range(n_centroids):
            eigenvals = torch.linalg.eigvals(metric_matrices[i]).real
            if (eigenvals < -1e-6).any():  # Allow small numerical errors
                warnings.warn(f"Metric matrix {i} is not positive semidefinite (min eigenval: {eigenvals.min():.3e})")
    
    def save_to_file(
        self,
        path: Union[str, Path],
        centroids: torch.Tensor,
        metric_matrices: torch.Tensor,
        temperature: float,
        regularization: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save metric data to file.
        
        Args:
            path: Output file path
            centroids: Centroid positions [n_centroids, latent_dim]
            metric_matrices: Metric matrices [n_centroids, latent_dim, latent_dim]
            temperature: Temperature parameter
            regularization: Regularization parameter
            metadata: Additional metadata to save
        """
        path = Path(path)
        
        data = {
            'centroids': centroids.cpu(),
            'metric_matrices': metric_matrices.cpu(),
            'temperature': temperature,
            'regularization': regularization,
        }
        
        if metadata:
            data['metadata'] = metadata
        
        torch.save(data, path)
        print(f"✅ Saved metric data to: {path}")
    
    def convert_old_format(
        self,
        old_path: Union[str, Path],
        new_path: Union[str, Path],
        temperature_override: Optional[float] = None,
        regularization_override: Optional[float] = None
    ) -> None:
        """
        Convert old metric file format to new standardized format.
        
        Args:
            old_path: Path to old format file
            new_path: Path for new format file
            temperature_override: Override temperature
            regularization_override: Override regularization
        """
        # Load using flexible loader
        data = self.load_from_file(old_path, temperature_override, regularization_override)
        
        # Save in standardized format
        self.save_to_file(
            new_path,
            data['centroids'],
            data['metric_matrices'],
            data['temperature'],
            data['regularization'],
            metadata={'converted_from': str(old_path)}
        )
        
        print(f"✅ Converted {old_path} → {new_path}")
    
    def validate_metric_file(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate a metric file and return diagnostic information.
        
        Args:
            path: Path to metric file
            
        Returns:
            Validation report dictionary
        """
        try:
            data = self.load_from_file(path)
            
            centroids = data['centroids']
            matrices = data['metric_matrices']
            
            # Compute diagnostics
            n_centroids, latent_dim = centroids.shape
            
            # Matrix properties
            eigenvals = []
            condition_numbers = []
            determinants = []
            
            for i in range(n_centroids):
                evals = torch.linalg.eigvals(matrices[i]).real
                eigenvals.append(evals)
                condition_numbers.append((evals.max() / (evals.min() + 1e-8)).item())
                determinants.append(torch.linalg.det(matrices[i]).item())
            
            eigenvals = torch.stack(eigenvals)
            
            report = {
                'valid': True,
                'n_centroids': n_centroids,
                'latent_dim': latent_dim,
                'temperature': data['temperature'],
                'regularization': data['regularization'],
                'eigenvalue_range': (eigenvals.min().item(), eigenvals.max().item()),
                'condition_number_range': (min(condition_numbers), max(condition_numbers)),
                'determinant_range': (min(determinants), max(determinants)),
                'has_negative_eigenvals': (eigenvals < -1e-6).any().item(),
                'mean_condition_number': sum(condition_numbers) / len(condition_numbers),
            }
            
        except Exception as e:
            report = {
                'valid': False,
                'error': str(e),
                'error_type': type(e).__name__,
            }
        
        return report 