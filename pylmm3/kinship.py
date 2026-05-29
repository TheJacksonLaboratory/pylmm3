import numpy as np


def calculateKinship(W: np.ndarray, center: bool = False) -> np.ndarray:
    """
    Compute the realized relationship matrix (RRM/GRM) from a raw genotype
    matrix W of shape (n_samples, n_snps).

    Each SNP column is imputed (missing → column mean), standardized to zero
    mean and unit variance, and invariant SNPs are dropped. K is divided by
    the number of valid SNPs used, not total SNPs.

    Arguments:
        W: raw (un-normalized) SNP matrix of shape (n_samples, n_snps)
        center: apply EMMA-style trace normalization so that tr(K) = n - 1.

    Returns:
        realized relationship matrix of shape (n_samples, n_samples)
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
