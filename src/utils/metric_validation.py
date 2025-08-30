"""
Metric Tensor Validation and Sanity Check Utilities

This module provides comprehensive validation for Riemannian metric tensors
to ensure they are symmetric positive definite (SPD) and numerically stable.
"""

import torch
import torch.nn as nn
import warnings
from typing import Tuple, Dict, Optional, Any
import numpy as np


class MetricValidationError(Exception):
    """Custom exception for metric validation failures."""
    pass


def validate_spd_matrix(
    G: torch.Tensor, 
    name: str = "G", 
    eps_tol: float = 1e-6,
    check_symmetry: bool = True,
    check_positive_definite: bool = True,
    warn_only: bool = False
) -> Dict[str, Any]:
    """
    Validate that a tensor represents valid SPD matrices.
    
    Args:
        G: Metric tensor [batch_size, dim, dim]
        name: Name for error messages
        eps_tol: Tolerance for numerical checks
        check_symmetry: Whether to check symmetry
        check_positive_definite: Whether to check positive definiteness
        warn_only: If True, issues warnings instead of raising errors
        
    Returns:
        Dict with validation results and diagnostics
    """
    batch_size, dim1, dim2 = G.shape
    
    if dim1 != dim2:
        msg = f"{name} matrices must be square, got shape {G.shape}"
        if warn_only:
            warnings.warn(msg)
        else:
            raise MetricValidationError(msg)
    
    dim = dim1
    diagnostics = {
        'shape': G.shape,
        'batch_size': batch_size,
        'dim': dim,
        'is_finite': torch.isfinite(G).all().item(),
        'has_nan': torch.isnan(G).any().item(),
        'has_inf': torch.isinf(G).any().item(),
    }
    
    # Check for NaN/Inf values
    if not diagnostics['is_finite']:
        msg = f"{name} contains non-finite values (NaN: {diagnostics['has_nan']}, Inf: {diagnostics['has_inf']})"
        if warn_only:
            warnings.warn(msg)
        else:
            raise MetricValidationError(msg)
    
    # Check symmetry
    if check_symmetry:
        G_T = G.transpose(-1, -2)
        sym_error = torch.norm(G - G_T, dim=(-2, -1)).max().item()
        diagnostics['symmetry_error'] = sym_error
        diagnostics['is_symmetric'] = sym_error < eps_tol
        
        if not diagnostics['is_symmetric']:
            msg = f"{name} matrices are not symmetric (max error: {sym_error:.2e})"
            if warn_only:
                warnings.warn(msg)
            else:
                raise MetricValidationError(msg)
    
    # Check positive definiteness via eigenvalues
    if check_positive_definite:
        try:
            eigenvals = torch.linalg.eigvals(G).real  # [batch_size, dim]
            min_eigenval = eigenvals.min().item()
            max_eigenval = eigenvals.max().item()
            
            diagnostics['eigenvals_min'] = min_eigenval
            diagnostics['eigenvals_max'] = max_eigenval
            diagnostics['condition_number'] = max_eigenval / max(min_eigenval, eps_tol)
            diagnostics['is_positive_definite'] = min_eigenval > eps_tol
            diagnostics['eigenvals_range'] = (min_eigenval, max_eigenval)
            
            if not diagnostics['is_positive_definite']:
                msg = f"{name} matrices are not positive definite (min eigenval: {min_eigenval:.2e})"
                if warn_only:
                    warnings.warn(msg)
                else:
                    raise MetricValidationError(msg)
                    
        except Exception as e:
            msg = f"Failed to compute eigenvalues for {name}: {e}"
            if warn_only:
                warnings.warn(msg)
                diagnostics['eigenval_computation_failed'] = True
            else:
                raise MetricValidationError(msg)
    
    return diagnostics


def validate_metric_inverse_consistency(
    G: torch.Tensor, 
    G_inv: torch.Tensor, 
    eps_tol: float = 1e-4,
    warn_only: bool = False
) -> Dict[str, Any]:
    """
    Validate that G and G_inv are proper inverses.
    
    Args:
        G: Metric tensor [batch_size, dim, dim]
        G_inv: Inverse metric tensor [batch_size, dim, dim]
        eps_tol: Tolerance for inverse check
        warn_only: If True, issues warnings instead of raising errors
        
    Returns:
        Dict with validation results
    """
    batch_size, dim = G.shape[0], G.shape[1]
    identity = torch.eye(dim, device=G.device, dtype=G.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    
    # Check G @ G_inv = I
    product1 = torch.bmm(G, G_inv)
    error1 = torch.norm(product1 - identity, dim=(-2, -1)).max().item()
    
    # Check G_inv @ G = I  
    product2 = torch.bmm(G_inv, G)
    error2 = torch.norm(product2 - identity, dim=(-2, -1)).max().item()
    
    max_error = max(error1, error2)
    is_valid_inverse = max_error < eps_tol
    
    diagnostics = {
        'forward_error': error1,
        'backward_error': error2,
        'max_error': max_error,
        'is_valid_inverse': is_valid_inverse,
        'tolerance': eps_tol
    }
    
    if not is_valid_inverse:
        msg = f"G and G_inv are not proper inverses (max error: {max_error:.2e}, tolerance: {eps_tol:.2e})"
        if warn_only:
            warnings.warn(msg)
        else:
            raise MetricValidationError(msg)
    
    return diagnostics


def compute_metric_diagnostics(G: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute comprehensive diagnostics for metric tensors.
    
    Args:
        G: Metric tensor [batch_size, dim, dim]
        
    Returns:
        Dict with diagnostic tensors
    """
    # Eigenvalue analysis
    eigenvals = torch.linalg.eigvals(G).real  # [batch_size, dim]
    
    # Determinant and trace
    det_G = torch.linalg.det(G)  # [batch_size]
    trace_G = torch.diagonal(G, dim1=-2, dim2=-1).sum(dim=-1)  # [batch_size]
    
    # Condition number
    eigenvals_clamped = torch.clamp(eigenvals, min=1e-12)
    condition_number = eigenvals_clamped.max(dim=1)[0] / eigenvals_clamped.min(dim=1)[0]
    
    # Frobenius norm
    frobenius_norm = torch.norm(G, dim=(-2, -1))
    
    return {
        'eigenvals_min': eigenvals.min(dim=1)[0],
        'eigenvals_max': eigenvals.max(dim=1)[0],
        'eigenvals_mean': eigenvals.mean(dim=1),
        'det': det_G,
        'trace': trace_G,
        'condition_number': condition_number,
        'frobenius_norm': frobenius_norm,
        'log_det': torch.logdet(G)
    }


def add_eps_regularization(
    G: torch.Tensor, 
    eps: float = 1e-6,
    method: str = "diagonal"
) -> torch.Tensor:
    """
    Add regularization to ensure positive definiteness.
    
    Args:
        G: Input metric tensor [batch_size, dim, dim]
        eps: Regularization strength
        method: Regularization method ("diagonal", "scaled_identity", "eigenval_clamp")
        
    Returns:
        Regularized metric tensor
    """
    batch_size, dim = G.shape[0], G.shape[1]
    
    if method == "diagonal":
        # Add eps to diagonal
        eye = torch.eye(dim, device=G.device, dtype=G.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        return G + eps * eye
        
    elif method == "scaled_identity":
        # Add eps * trace(G) / dim to diagonal
        trace_G = torch.diagonal(G, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
        scale = eps * trace_G / dim
        eye = torch.eye(dim, device=G.device, dtype=G.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        return G + scale * eye
        
    elif method == "eigenval_clamp":
        # Clamp eigenvalues to be at least eps
        eigenvals, eigenvecs = torch.linalg.eigh(G)
        eigenvals_clamped = torch.clamp(eigenvals, min=eps)
        return torch.bmm(torch.bmm(eigenvecs, torch.diag_embed(eigenvals_clamped)), eigenvecs.transpose(-2, -1))
        
    else:
        raise ValueError(f"Unknown regularization method: {method}")


def safe_cholesky(
    G: torch.Tensor, 
    eps: float = 1e-6,
    max_attempts: int = 3
) -> Tuple[torch.Tensor, bool]:
    """
    Compute Cholesky decomposition with fallback regularization.
    
    Args:
        G: Metric tensor [batch_size, dim, dim]
        eps: Initial regularization strength
        max_attempts: Maximum regularization attempts
        
    Returns:
        Tuple of (Cholesky factor, success_flag)
    """
    for attempt in range(max_attempts):
        try:
            # Try standard Cholesky
            L = torch.linalg.cholesky(G)
            return L, True
            
        except torch.linalg.LinAlgError:
            if attempt < max_attempts - 1:
                # Add regularization and try again
                current_eps = eps * (10 ** attempt)
                G = add_eps_regularization(G, current_eps, method="diagonal")
                warnings.warn(f"Cholesky failed, adding regularization eps={current_eps:.2e} (attempt {attempt+1})")
            else:
                warnings.warn("Cholesky decomposition failed after all attempts")
                return torch.zeros_like(G), False


class MetricValidator:
    """
    Comprehensive metric tensor validator with configurable tolerance.
    """
    
    def __init__(
        self,
        eps_chol: float = 1e-6,
        eps_spd: float = 1e-6,
        eps_inverse: float = 1e-4,
        warn_only: bool = False
    ):
        self.eps_chol = eps_chol
        self.eps_spd = eps_spd
        self.eps_inverse = eps_inverse
        self.warn_only = warn_only
        
        # Statistics tracking
        self.validation_count = 0
        self.failure_count = 0
        self.last_diagnostics = None
    
    def validate_metric_tensor(
        self, 
        G: torch.Tensor, 
        G_inv: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of metric tensor.
        
        Args:
            G: Metric tensor [batch_size, dim, dim]
            G_inv: Optional inverse metric tensor
            
        Returns:
            Dict with validation results and diagnostics
        """
        self.validation_count += 1
        
        try:
            # Validate G is SPD
            spd_results = validate_spd_matrix(
                G, name="G", eps_tol=self.eps_spd, warn_only=self.warn_only
            )
            
            results = {
                'spd_validation': spd_results,
                'diagnostics': compute_metric_diagnostics(G)
            }
            
            # Validate inverse consistency if G_inv provided
            if G_inv is not None:
                # Validate G_inv is SPD
                spd_inv_results = validate_spd_matrix(
                    G_inv, name="G_inv", eps_tol=self.eps_spd, warn_only=self.warn_only
                )
                
                # Check inverse consistency
                inverse_results = validate_metric_inverse_consistency(
                    G, G_inv, eps_tol=self.eps_inverse, warn_only=self.warn_only
                )
                
                results['spd_inv_validation'] = spd_inv_results
                results['inverse_validation'] = inverse_results
            
            # Test Cholesky decomposition
            L, chol_success = safe_cholesky(G, eps=self.eps_chol)
            results['cholesky'] = {'success': chol_success, 'factor': L}
            
            self.last_diagnostics = results
            return results
            
        except Exception as e:
            self.failure_count += 1
            if not self.warn_only:
                raise
            warnings.warn(f"Metric validation failed: {e}")
            return {'error': str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            'total_validations': self.validation_count,
            'total_failures': self.failure_count,
            'failure_rate': self.failure_count / max(self.validation_count, 1),
            'last_diagnostics': self.last_diagnostics
        }
