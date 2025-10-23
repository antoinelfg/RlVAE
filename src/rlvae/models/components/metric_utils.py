"""
Shared utilities for RLVAE metric computations.

This module centralises weight computation and determinant helpers so that
MetricTensor, visualization stubs, and samplers operate on the same definitions.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def normalize_metric_atoms(
    atoms: torch.Tensor,
    mode: str = "none",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Normalize metric atoms according to the requested mode.

    Args:
        atoms: [K, D, D] SPD matrices representing precision atoms.
        mode:  'none' (identity), 'trace', or 'fro' (Frobenius norm).
        eps:   Numerical floor to avoid division-by-zero.

    Returns:
        Normalized atoms with the same shape/dtype/device.
    """
    mode = (mode or "none").lower()
    if mode == "none":
        return atoms

    if mode == "trace":
        traces = torch.einsum("kii->k", atoms).unsqueeze(-1).unsqueeze(-1)
        scales = (traces.abs() + eps)
    elif mode in {"fro", "frobenius"}:
        frob = torch.linalg.norm(atoms.reshape(atoms.shape[0], -1), dim=-1)
        scales = frob.unsqueeze(-1).unsqueeze(-1) + eps
    else:
        raise ValueError(f"Unknown metric normalization mode '{mode}'.")

    return atoms / scales


def compute_metric_weights(
    z: torch.Tensor,
    centroids: torch.Tensor,
    metric_atoms: torch.Tensor,
    temperature: torch.Tensor | float,
    *,
    kernel: str = "mahalanobis_normed",
    normalize: bool = False,
    topk: Optional[int] = None,
    stabilize: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute mixture weights for inverse-metric atoms.

    Args:
        z:          [B, D] latent locations.
        centroids:  [K, D] centroid anchors.
        metric_atoms: [K, D, D] matrices used for Mahalanobis kernels. These
                      should already be normalised according to the chosen mode.
        temperature: Scalar or tensor controlling decay (interpreted as σ).
        kernel:     'isotropic', 'mahalanobis', or 'mahalanobis_normed'.
        normalize:  If True, divide weights by their row-wise sum.
        topk:       Optional integer; if set, only the closest top-k centroids
                    influence each sample.
        stabilize:  If True, subtract the row-wise minimum distance before
                    exponentiation for numerical stability.
        eps:        Numeric stability constant.

    Returns:
        Weights tensor [B, K] (non-negative).
    """
    if z.numel() == 0 or centroids.numel() == 0:
        raise ValueError("compute_metric_weights requires non-empty z and centroids.")

    device = z.device
    dtype = z.dtype
    kernel = (kernel or "mahalanobis_normed").lower()

    temp = torch.as_tensor(temperature, device=device, dtype=dtype).clamp_min(eps)
    temp2 = temp ** 2

    diff = z.unsqueeze(1) - centroids.unsqueeze(0)  # [B, K, D]

    if kernel == "isotropic":
        dist_sq = torch.sum(diff * diff, dim=-1) / (temp2 + eps)
    else:
        mats = metric_atoms.to(device=device, dtype=dtype)
        tmp = torch.einsum("bkd,kde->bke", diff, mats)
        dist_sq = torch.sum(tmp * diff, dim=-1) / (temp2 + eps)

    if topk is not None and topk > 0 and topk < dist_sq.shape[1]:
        vals, idx = torch.topk(dist_sq, k=topk, dim=1, largest=False)
        pruned = torch.full_like(dist_sq, float("inf"))
        pruned.scatter_(1, idx, vals)
        dist_sq = pruned

    if stabilize:
        finite_min = torch.nan_to_num(dist_sq, nan=float("inf"), posinf=float("inf"), neginf=0.0)
        row_min = torch.min(finite_min, dim=-1, keepdim=True)[0]
        row_min = torch.where(torch.isfinite(row_min), row_min, torch.zeros_like(row_min))
        dist_sq = dist_sq - row_min

    weights = torch.exp(-dist_sq)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize:
        weights_sum = weights.sum(dim=1, keepdim=True) + eps
        weights = weights / weights_sum

    return weights


def half_logdet_volume(matrix: torch.Tensor, representation: str, *, jitter: float = 1e-6) -> torch.Tensor:
    """
    Compute ±½ log|det| for either metric or inverse-metric tensors.

    Args:
        matrix: SPD matrix batch [B, D, D].
        representation: 'g' if ``matrix`` is a metric, 'ginv' if precision.
        jitter: Diagonal jitter for the Cholesky factorisation.

    Returns:
        Tensor [B] containing +½ log|det G^{-1}| when representation='ginv',
        and -½ log|det G| when representation='g'.
    """
    representation = representation.lower()
    if representation not in {"g", "ginv"}:
        raise ValueError(f"Unknown representation '{representation}' (expected 'g' or 'ginv').")

    chol = None
    orig_dtype = matrix.dtype
    work_matrix = matrix
    if matrix.dtype in (torch.float16, torch.bfloat16):
        work_matrix = matrix.float()
    try:
        chol = torch.linalg.cholesky(work_matrix)
    except RuntimeError:
        d = matrix.shape[-1]
        eye = torch.eye(d, device=matrix.device, dtype=work_matrix.dtype).unsqueeze(0)
        work_matrix = work_matrix + jitter * eye
        chol = torch.linalg.cholesky(work_matrix)

    diag = torch.diagonal(chol, dim1=-2, dim2=-1).abs() + 1e-18
    logdet = 2.0 * torch.log(diag).sum(dim=-1)
    half = 0.5 * logdet
    half = half if representation == "ginv" else -half
    return half.to(orig_dtype)
