"""Unit tests for kinship matrix estimation (pylmm3.kinship)."""

import numpy as np
import pytest

from pylmm3.kinship import calculateKinship, NoVariantSNPsError


def test_shape_and_symmetry(genotypes):
    """K is (n_samples, n_samples) and symmetric."""
    K = calculateKinship(genotypes)
    n = genotypes.shape[0]
    assert K.shape == (n, n)
    assert np.allclose(K, K.T)


def test_centering_normalizes_trace(genotypes):
    """With center=True the EMMA normalization gives trace(K) == n - 1."""
    n = genotypes.shape[0]
    K = calculateKinship(genotypes, center=True)
    assert np.trace(K) == pytest.approx(n - 1)


def test_invariant_snps_dropped():
    """Monomorphic columns are excluded; only varying SNPs build K."""
    # Two informative SNPs, one constant column that must be ignored.
    W = np.array([
        [0.0, 1.0, 0.5],
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.5],
        [1.0, 0.0, 0.5],
    ])
    K_with_const = calculateKinship(W)
    K_without = calculateKinship(W[:, :2])
    assert np.allclose(K_with_const, K_without)


def test_all_invariant_raises():
    """If every SNP is monomorphic, NoVariantSNPsError (a ValueError) is raised."""
    assert issubclass(NoVariantSNPsError, ValueError)
    with pytest.raises(NoVariantSNPsError):
        calculateKinship(np.ones((5, 4)))


def test_missing_values_imputed():
    """NaN genotypes are imputed with the column mean and do not propagate."""
    W = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [np.nan, 1.0],
        [1.0, 0.0],
    ])
    K = calculateKinship(W)
    assert np.isfinite(K).all()
