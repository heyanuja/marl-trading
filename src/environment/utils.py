from __future__ import annotations

import numpy as np


def make_constant_correlation(num_assets: int, corr: float) -> np.ndarray:
    """Create an NxN correlation matrix with constant off-diagonal correlation."""
    if num_assets <= 0:
        raise ValueError(f"num_assets must be positive, got {num_assets}")
    if not (-1.0 < corr < 1.0):
        raise ValueError(f"corr must be in (-1, 1), got {corr}")

    mat = np.full((num_assets, num_assets), float(corr), dtype=np.float64)
    np.fill_diagonal(mat, 1.0)
    return mat


def safe_cholesky(matrix: np.ndarray, *, jitter: float = 1e-8) -> np.ndarray:
    """Cholesky factorization with diagonal jitter for numerical stability."""
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {mat.shape}")

    try:
        return np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        mat = mat.copy()
        mat.flat[:: mat.shape[0] + 1] += jitter
        return np.linalg.cholesky(mat)
