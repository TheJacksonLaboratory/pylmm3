"""Unit tests for kinship matrix estimation (pylmm3.kinship)."""

import numpy as np
import pytest

from pylmm3.kinship import (
    calculateKinship,
    calculate_kinship_blocked,
    NoVariantSNPsError,
)


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


# --- blocked / streamed kinship accumulation (#1) -----------------------

def _column_blocks(W, block_size):
    """Split W into consecutive column-blocks of width block_size."""
    return [W[:, j:j + block_size] for j in range(0, W.shape[1], block_size)]


@pytest.mark.parametrize("block_size", [1, 7, 40, 1000])
def test_blocked_matches_full_matrix(genotypes, block_size):
    """K accumulated block-by-block equals K from the full matrix (any block size)."""
    n = genotypes.shape[0]
    K_blocked = calculate_kinship_blocked(_column_blocks(genotypes, block_size), n_samples=n)
    assert np.allclose(K_blocked, calculateKinship(genotypes))


@pytest.mark.parametrize("block_size", [1, 2, 3])
def test_blocked_matches_full_with_invariant_and_missing(block_size):
    """Per-column drop/impute is block-separable: invariant cols and NaNs agree."""
    W = np.array([
        [0.0, 1.0, 0.5, np.nan],
        [1.0, 0.0, 0.5, 0.0],
        [0.5, 1.0, 0.5, 1.0],
        [1.0, np.nan, 0.5, 0.5],
    ])  # column 2 is invariant (all 0.5) and must be dropped in both paths
    K_blocked = calculate_kinship_blocked(_column_blocks(W, block_size), n_samples=W.shape[0])
    assert np.allclose(K_blocked, calculateKinship(W))


def test_blocked_does_not_mutate_input_blocks():
    """Accumulation must not impute NaNs in the caller's block arrays."""
    W = np.array([[0.0, 1.0], [1.0, 0.0], [np.nan, 1.0], [1.0, 0.0]])
    blocks = _column_blocks(W, 1)
    calculate_kinship_blocked(blocks, n_samples=W.shape[0])
    assert np.isnan(blocks[0][2, 0])  # original NaN still present


def test_blocked_all_invariant_raises():
    """Every column invariant across all blocks → NoVariantSNPsError."""
    blocks = _column_blocks(np.ones((5, 4)), 2)
    with pytest.raises(NoVariantSNPsError):
        calculate_kinship_blocked(blocks, n_samples=5)


@pytest.mark.parametrize("block_size", [1, 7, 40])
def test_blocked_center_matches_full(genotypes, block_size):
    """center=True trace normalization is applied to the accumulated K too."""
    n = genotypes.shape[0]
    K_blocked = calculate_kinship_blocked(
        _column_blocks(genotypes, block_size), n_samples=n, center=True
    )
    assert np.allclose(K_blocked, calculateKinship(genotypes, center=True))
