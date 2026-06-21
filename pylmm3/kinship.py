"""Kinship (realized relationship) matrix estimation from SNP genotypes."""

import numpy as np


class NoVariantSNPsError(ValueError):
    """Raised when every SNP column is invariant, leaving no SNP to build K from.

    Subclasses ``ValueError`` so existing ``except ValueError`` callers keep
    working, while callers that can treat a monomorphic cohort as a skippable
    (rather than fatal) condition can catch this type precisely.
    """


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
    monomorphic.

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
        NoVariantSNPsError: If every SNP column is invariant (zero variance
            after imputation), leaving no valid SNPs to build K from. This is a
            ``ValueError`` subclass.
    """
    n = W.shape[0]
    W = W.astype(np.float64, copy=True)

    col_means = np.nanmean(W, axis=0)
    col_vars  = np.nanvar(W, axis=0)

    nan_rows, nan_cols = np.where(np.isnan(W))
    W[nan_rows, nan_cols] = col_means[nan_cols]

    keep = col_vars > 0
    if not keep.any():
        raise NoVariantSNPsError("No valid (non-invariant) SNPs found")

    W = (W[:, keep] - col_means[keep]) / np.sqrt(col_vars[keep])
    K = (W @ W.T) / W.shape[1]

    if center:
        S = np.trace(K) - n * K.mean()
        return (n - 1) * K / S

    return K


def calculateKinshipBlocks(
    blocks,
    n_samples: int,
    center: bool = False,
) -> np.ndarray:
    """Compute the RRM/GRM by accumulating SNP column-blocks, bounding memory.

    Equivalent to ``calculateKinship`` on the horizontally-concatenated blocks,
    but holds only the running ``n_samples × n_samples`` accumulator plus one
    block at a time rather than the full ``n_samples × n_snps`` matrix. The
    result is mathematically identical because per-SNP standardization is
    column-separable and the Gram matrix is a sum of per-SNP outer products;
    the divisor is the total number of valid SNPs retained across all blocks,
    and ``center`` normalization is applied once to the finished matrix.

    Args:
        blocks:
            Iterable of raw (un-normalized) SNP column-blocks, each of shape
            ``(n_samples, block_width)`` with missing values encoded as
            ``np.nan``. Blocks are not mutated.
        n_samples:
            Number of individuals (the dimension of the returned matrix).
        center:
            If ``True``, apply EMMA-style trace normalization so that
            ``trace(K) == n_samples - 1`` (matches ``calculateKinship``).

    Returns:
        Realized relationship matrix of shape ``(n_samples, n_samples)``.

    Raises:
        NoVariantSNPsError: If every SNP across all blocks is invariant,
            leaving no valid SNPs to build K from.
    """
    K = np.zeros((n_samples, n_samples), dtype=np.float64)
    kept_total = 0

    for block in blocks:
        B = np.array(block, dtype=np.float64)  # copy: never impute into caller's data

        col_means = np.nanmean(B, axis=0)
        col_vars  = np.nanvar(B, axis=0)

        nan_rows, nan_cols = np.where(np.isnan(B))
        B[nan_rows, nan_cols] = col_means[nan_cols]

        keep = col_vars > 0
        if not keep.any():
            continue

        Bs = (B[:, keep] - col_means[keep]) / np.sqrt(col_vars[keep])
        K += Bs @ Bs.T
        kept_total += int(keep.sum())

    if kept_total == 0:
        raise NoVariantSNPsError("No valid (non-invariant) SNPs found")

    K /= kept_total

    if center:
        S = np.trace(K) - n_samples * K.mean()
        return (n_samples - 1) * K / S

    return K

