#!/usr/bin/env python3
"""
Rescale Stage B metric atoms to increase eigenvalue scale while preserving/amplifying anisotropy.

This script addresses the root cause of negative KL divergence:
- Stage B produces G⁻¹ matrices with very small determinants (mean ~0.009)
- Small eigenvalues lead to very negative log|Σ_μ| in log q
- Interpolation further reduces anisotropy ratios

Strategy:
1. Scale ALL eigenvalues by a factor (increases det, reduces negative log|Σ|)
2. Optional: amplify anisotropy by scaling eigenvalues differently
3. Preserve the overall structure and geometric relationships
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from typing import Tuple, Dict


def analyze_atoms(G_inv_matrices: torch.Tensor) -> Dict[str, np.ndarray]:
    """Analyze eigenvalue statistics of G⁻¹ atoms."""
    K, D, _ = G_inv_matrices.shape
    
    eigenvalues = []
    ratios = []
    dets = []
    traces = []
    
    for i in range(K):
        G_inv = G_inv_matrices[i]
        eigvals = torch.linalg.eigvalsh(G_inv).numpy()
        eigenvalues.append(eigvals)
        
        ratio = eigvals.max() / eigvals.min()
        ratios.append(ratio)
        
        det = torch.det(G_inv).item()
        dets.append(det)
        
        trace = torch.trace(G_inv).item()
        traces.append(trace)
    
    eigenvalues = np.array(eigenvalues)
    
    return {
        'eigenvalues': eigenvalues,
        'ratios': np.array(ratios),
        'dets': np.array(dets),
        'traces': np.array(traces),
    }


def print_statistics(stats: Dict[str, np.ndarray], prefix: str = ""):
    """Print comprehensive statistics."""
    eigvals = stats['eigenvalues']
    ratios = stats['ratios']
    dets = stats['dets']
    traces = stats['traces']
    
    print(f"\n{prefix}Statistics:")
    print("=" * 70)
    print(f"Number of atoms: {len(ratios)}")
    print(f"Latent dimension: {eigvals.shape[1]}")
    print()
    
    print("Eigenvalues:")
    print(f"  Min:    {eigvals.min():.6f}")
    print(f"  Max:    {eigvals.max():.6f}")
    print(f"  Mean:   {eigvals.mean():.6f}")
    print(f"  Median: {np.median(eigvals):.6f}")
    print()
    
    print("Anisotropy Ratios (max_eig / min_eig):")
    print(f"  Min:    {ratios.min():.4f}")
    print(f"  Max:    {ratios.max():.4f}")
    print(f"  Mean:   {ratios.mean():.4f}")
    print(f"  Median: {np.median(ratios):.4f}")
    print()
    
    print("Determinants:")
    print(f"  Min:    {dets.min():.6f}")
    print(f"  Max:    {dets.max():.6f}")
    print(f"  Mean:   {dets.mean():.6f}")
    print(f"  Median: {np.median(dets):.6f}")
    print()
    
    print("Traces:")
    print(f"  Min:    {traces.min():.6f}")
    print(f"  Max:    {traces.max():.6f}")
    print(f"  Mean:   {traces.mean():.6f}")
    print(f"  Median: {np.median(traces):.6f}")
    print()
    
    # Distribution of anisotropy ratios
    print("Distribution of Anisotropy Ratios:")
    bins = [1.0, 1.1, 1.5, 2.0, 5.0, 10.0, 100.0, float('inf')]
    labels = ['[1.0-1.1)', '[1.1-1.5)', '[1.5-2.0)', '[2.0-5.0)', '[5.0-10)', '[10-100)', '[100+]']
    for i in range(len(bins)-1):
        count = np.sum((ratios >= bins[i]) & (ratios < bins[i+1]))
        pct = 100.0 * count / len(ratios)
        print(f"  {labels[i]:12s}: {count:4d} ({pct:5.1f}%)")


def rescale_isotropic(G_inv_matrices: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """
    Rescale all eigenvalues uniformly (isotropic scaling).
    
    This preserves anisotropy ratios exactly.
    """
    return G_inv_matrices * scale_factor


def rescale_anisotropic(
    G_inv_matrices: torch.Tensor,
    scale_factor: float,
    anisotropy_amplification: float = 1.0
) -> torch.Tensor:
    """
    Rescale eigenvalues with optional anisotropy amplification.
    
    Args:
        G_inv_matrices: [K, D, D] precision matrices
        scale_factor: Global scaling factor (increases all eigenvalues)
        anisotropy_amplification: > 1.0 amplifies anisotropy, < 1.0 reduces it
    
    Returns:
        Rescaled matrices with preserved SPD property
    """
    K, D, _ = G_inv_matrices.shape
    G_rescaled = torch.zeros_like(G_inv_matrices)
    
    for i in range(K):
        G_inv = G_inv_matrices[i]
        
        # Eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(G_inv)
        
        # Apply global scaling
        eigvals_scaled = eigvals * scale_factor
        
        # Amplify anisotropy
        if anisotropy_amplification != 1.0 and D == 2:
            # For 2D: move eigenvalues away from geometric mean
            geom_mean = torch.exp(torch.log(eigvals_scaled).mean())
            
            # Scale eigenvalues relative to geometric mean
            eigvals_scaled = geom_mean * ((eigvals_scaled / geom_mean) ** anisotropy_amplification)
        
        # Reconstruct matrix
        G_rescaled[i] = eigvecs @ torch.diag(eigvals_scaled) @ eigvecs.T
        
        # Ensure symmetry
        G_rescaled[i] = 0.5 * (G_rescaled[i] + G_rescaled[i].T)
    
    return G_rescaled


def main():
    parser = argparse.ArgumentParser(
        description="Rescale Stage B metric atoms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/stages/B_RHVAE_MLP_2_SPRITES/metric.pt",
        help="Path to input metric.pt file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/stages/B_RHVAE_MLP_2_SPRITES/metric_rescaled.pt",
        help="Path to output rescaled metric file"
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=10.0,
        help="Global scaling factor for eigenvalues (increases determinant by scale^D)"
    )
    parser.add_argument(
        "--anisotropy-amplification",
        type=float,
        default=1.0,
        help="Anisotropy amplification factor (>1.0 amplifies, <1.0 reduces, 1.0 preserves)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['isotropic', 'anisotropic'],
        default='isotropic',
        help="Rescaling mode"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze only, don't save"
    )
    
    args = parser.parse_args()
    
    # Load metric
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return 1
    
    print(f"📂 Loading metric from: {input_path}")
    data = torch.load(input_path, map_location='cpu')
    
    centroids = data['centroids']
    G_inv_orig = data['M_matrices']
    
    print(f"✅ Loaded {len(centroids)} centroids")
    print(f"   Latent dimension: {centroids.shape[1]}")
    print(f"   G⁻¹ matrices shape: {G_inv_orig.shape}")
    
    # Analyze original
    stats_orig = analyze_atoms(G_inv_orig)
    print_statistics(stats_orig, prefix="📊 ORIGINAL")
    
    # Apply rescaling
    print(f"\n🔧 Applying rescaling...")
    print(f"   Mode:                     {args.mode}")
    print(f"   Scale factor:             {args.scale_factor}")
    if args.mode == 'anisotropic':
        print(f"   Anisotropy amplification: {args.anisotropy_amplification}")
    
    if args.mode == 'isotropic':
        G_inv_rescaled = rescale_isotropic(G_inv_orig, args.scale_factor)
    else:
        G_inv_rescaled = rescale_anisotropic(
            G_inv_orig,
            args.scale_factor,
            args.anisotropy_amplification
        )
    
    # Analyze rescaled
    stats_rescaled = analyze_atoms(G_inv_rescaled)
    print_statistics(stats_rescaled, prefix="📊 RESCALED")
    
    # Compare
    print("\n📈 COMPARISON:")
    print("=" * 70)
    print(f"Eigenvalue scale increase:     {stats_rescaled['eigenvalues'].mean() / stats_orig['eigenvalues'].mean():.2f}x")
    print(f"Determinant scale increase:    {stats_rescaled['dets'].mean() / stats_orig['dets'].mean():.2f}x")
    print(f"Trace scale increase:          {stats_rescaled['traces'].mean() / stats_orig['traces'].mean():.2f}x")
    print(f"Mean anisotropy ratio change:  {stats_orig['ratios'].mean():.3f} → {stats_rescaled['ratios'].mean():.3f}")
    print()
    
    # Expected impact on Sigma_mu and log q
    D = centroids.shape[1]
    print("\n🎯 EXPECTED IMPACT (with rhmc_alpha=0.1, eps_reg=1e-3):")
    print("=" * 70)
    
    # Original Sigma_mu eigenvalues (approximation)
    alpha = 0.1
    eps_reg = 1e-3
    sigma_eig_orig_min = alpha * stats_orig['eigenvalues'].mean() + eps_reg
    sigma_eig_orig_max = alpha * stats_orig['eigenvalues'].mean() + eps_reg
    
    sigma_eig_new_min = alpha * stats_rescaled['eigenvalues'].min() + eps_reg
    sigma_eig_new_max = alpha * stats_rescaled['eigenvalues'].max() + eps_reg
    
    log_det_sigma_orig = D * np.log(sigma_eig_orig_min)  # Approximation for isotropic
    log_det_sigma_new = np.log(sigma_eig_new_min) + np.log(sigma_eig_new_max)  # For 2D
    
    print(f"Original Σ_μ eigenvalues (approx): [{sigma_eig_orig_min:.6f}, {sigma_eig_orig_max:.6f}]")
    print(f"Rescaled Σ_μ eigenvalues (approx): [{sigma_eig_new_min:.6f}, {sigma_eig_new_max:.6f}]")
    print()
    print(f"Original log|Σ_μ| (approx):        {log_det_sigma_orig:.4f}")
    print(f"Rescaled log|Σ_μ| (approx):        {log_det_sigma_new:.4f}")
    print(f"Δ log|Σ_μ|:                        {log_det_sigma_new - log_det_sigma_orig:+.4f}")
    print()
    print(f"Volume term in log q:              -0.5 * log|Σ_μ|")
    print(f"Original volume term (approx):     {-0.5 * log_det_sigma_orig:.4f}")
    print(f"Rescaled volume term (approx):     {-0.5 * log_det_sigma_new:.4f}")
    print(f"Δ volume term (LESS NEGATIVE):     {-0.5 * (log_det_sigma_new - log_det_sigma_orig):+.4f}")
    print()
    print("💡 This should make log q LESS NEGATIVE, improving KL!")
    
    # Save
    if not args.dry_run:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data_rescaled = {
            'centroids': centroids,
            'M_matrices': G_inv_rescaled,
            'rescaling_info': {
                'scale_factor': args.scale_factor,
                'anisotropy_amplification': args.anisotropy_amplification,
                'mode': args.mode,
                'original_stats': {
                    'eigenvalue_mean': float(stats_orig['eigenvalues'].mean()),
                    'det_mean': float(stats_orig['dets'].mean()),
                    'ratio_mean': float(stats_orig['ratios'].mean()),
                },
                'rescaled_stats': {
                    'eigenvalue_mean': float(stats_rescaled['eigenvalues'].mean()),
                    'det_mean': float(stats_rescaled['dets'].mean()),
                    'ratio_mean': float(stats_rescaled['ratios'].mean()),
                },
            }
        }
        
        torch.save(data_rescaled, output_path)
        print(f"\n💾 Saved rescaled metric to: {output_path}")
        print(f"   To use it, update your config:")
        print(f"   metric:")
        print(f"     path: {output_path}")
    else:
        print(f"\n🔍 DRY RUN - No file saved")
        print(f"   Remove --dry-run to save to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())

