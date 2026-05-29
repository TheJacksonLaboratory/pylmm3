"""Kinship (realized relationship) matrix estimation from SNP genotypes."""

import numpy as np


def calculateKinship(
    W: np.ndarray,
    center: bool = False,
) -> np.ndarray:
    """Compute the realized relationship matrix (RRM/GRM) from a raw genotype matrix.

    Each SNP column is imputed (missing → column mean), standardized to zero
    mean and unit variance, and invariant SNPs are dropped. The result is
    divided by the number of valid (non-invariant) SNPs retained, not the
    total column count of W.

    When `center=False` (the CLI default), the returned K will differ
    numerically from files produced by the original pylmm if any SNPs were
    monomorphic — see docs/kinship_denominator_fix.md.

    Args:
        W:
            Raw (un-normalized) SNP matrix of shape `(n_samples, n_snps)`.
            Missing values should be encoded as `np.nan`.
        center:
            If `True`, apply EMMA-style trace normalization so that
            `trace(K) == n - 1`. This absorbs the per-SNP denominator
            difference, making the result identical to the original pylmm
            output regardless of how many SNPs were filtered.

    Returns:
        Realized relationship matrix of shape `(n_samples, n_samples)`.

    Raises:
        ValueError: If every SNP column is invariant (zero variance after
            imputation), leaving no valid SNPs to build K from.
    """
    n = W.shape[0]
    W = W.astype(np.float64, copy=True)

    col_means = np.nanmean(W, axis=0)
    col_vars  = np.nanvar(W, axis=0)

    nan_rows, nan_cols = np.where(np.isnan(W))
    W[nan_rows, nan_cols] = col_means[nan_cols]

    keep = col_vars > 0
    if not keep.any():
        raise ValueError("No valid (non-invariant) SNPs found")

    W = (W[:, keep] - col_means[keep]) / np.sqrt(col_vars[keep])
    K = (W @ W.T) / W.shape[1]

    if center:
        S = np.trace(K) - n * K.mean()
        return (n - 1) * K / S

    return K
