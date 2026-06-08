"""Unit tests for the LMM solver (pylmm3.lmm)."""

import numpy as np
import pytest
from scipy import stats

from pylmm3.lmm import LMM


def test_init_defaults_intercept_and_unfit_state():
    """With no X0, X0 defaults to an intercept; opt* are None before fit()."""
    Y = np.random.default_rng(0).standard_normal(20)
    lmm = LMM(Y, np.eye(20))
    assert lmm.N == 20
    assert lmm.X0.shape == (20, 1)
    assert np.allclose(lmm.X0, 1.0)
    assert lmm.optH is None
    assert lmm.optBeta is None
    assert lmm.optSigma is None
    assert lmm.LLs is None


def test_init_removes_missing_phenotypes(kinship, phenotype):
    """NaN phenotypes are dropped, and K/X0 are subset to match."""
    Y = phenotype.copy()
    Y[3] = np.nan
    Y[7] = np.nan
    lmm = LMM(Y, kinship)
    assert lmm.N == len(phenotype) - 2
    assert lmm.K.shape == (lmm.N, lmm.N)
    assert lmm.Y.shape == (lmm.N, 1)
    # nonmissing mask reflects the dropped individuals
    assert lmm.nonmissing.sum() == lmm.N
    assert not lmm.nonmissing[3]
    assert not lmm.nonmissing[7]


def test_init_computes_eigendecomposition_when_absent():
    """When Kva/Kve are not supplied, they are computed from K.

    Uses a well-conditioned SPD K (min eigenvalue ≫ 1e-6) so no clamping
    occurs and the decomposition reconstructs K exactly.
    """
    rng = np.random.default_rng(7)
    n = 30
    A = rng.standard_normal((n, n))
    K = A @ A.T / n + np.eye(n)
    lmm = LMM(rng.standard_normal(n), K)
    assert lmm.Kva.shape == (n,)
    assert lmm.Kve.shape == (n, n)
    assert np.allclose(lmm.Kve @ lmm.Kve.T, np.eye(n))  # orthonormal
    # K ≈ Kve · diag(Kva) · Kve.T
    reconstructed = lmm.Kve @ np.diag(lmm.Kva) @ lmm.Kve.T
    assert np.allclose(reconstructed, lmm.K, atol=1e-8)


def test_eigenvalues_clamped_to_floor():
    """Near-zero / negative eigenvalues are clamped to 1e-6."""
    # A rank-deficient K has zero eigenvalues that must be clamped.
    Y = np.random.default_rng(1).standard_normal(10)
    K = np.ones((10, 10))  # rank 1 → nine zero eigenvalues
    lmm = LMM(Y, K)
    assert (lmm.Kva >= 1e-6).all()


def test_supplied_kva_not_mutated_by_clamping():
    """A caller-supplied Kva is not mutated when near-zero eigenvalues clamp.

    Regression for the in-place mutation bug (commit aa2a2e2): __init__ now
    stores ``np.array(Kva, copy=True)``, so the floor-clamp writes to the copy
    and leaves the caller's array intact. With the old ``self.Kva = Kva`` the
    clamp would rewrite the caller's array in place.
    """
    from scipy import linalg

    rng = np.random.default_rng(7)
    n = 12
    A = rng.standard_normal((n, n))
    K = A @ A.T / n + np.eye(n)
    Kva, Kve = linalg.eigh(K)
    Kva[0] = 1e-9                       # force one eigenvalue below the 1e-6 floor
    Kva_before = Kva.copy()

    lmm = LMM(rng.standard_normal(n), K, Kva=Kva, Kve=Kve)

    assert lmm.Kva[0] == 1e-6                          # clamp applied to the copy
    np.testing.assert_array_equal(Kva, Kva_before)     # caller's array untouched


def test_getmax_scans_penultimate_grid_point(monkeypatch):
    """getMax detects a local maximum at the second-to-last grid index.

    Regression for the off-by-one in the bracket-detection loop (commit
    aa2a2e2): the old ``range(1, n - 2)`` never examined index ``n - 2``, so a
    peak at the penultimate grid point was missed and getMax fell through to an
    endpoint. The loop now runs ``range(1, n - 1)``. We hand getMax a grid whose
    only interior maximum sits at ``n - 2`` and stub LL_brent to a parabola
    minimized at 0.85 (inside the resulting bracket): the fixed code refines to
    0.85, the buggy code would return the endpoint H[n-1] = 0.9.
    """
    lmm = LMM(np.random.default_rng(0).standard_normal(10), np.eye(10))
    # Strictly increasing then a single down-step, so the sole interior local
    # max is at index 8 (= n - 2 for n = 10).
    lmm.LLs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8], dtype=float)
    H = np.arange(10) / 10.0                       # H[8] = 0.8; bracket (0.7, 0.9)
    monkeypatch.setattr(lmm, "LL_brent", lambda h, X, REML: (h - 0.85) ** 2)

    hmax = lmm.getMax(H, lmm.X0t, REML=False)
    assert hmax == pytest.approx(0.85, abs=1e-4)


def test_fit_returns_valid_heritability(kinship, phenotype):
    """fit() returns hmax in [0, 1] and populates the opt* attributes."""
    lmm = LMM(phenotype, kinship)
    hmax, beta, sigma, LL = lmm.fit()
    assert 0.0 <= hmax <= 1.0
    assert lmm.optH == hmax
    assert lmm.optBeta is not None
    assert lmm.optSigma is not None
    assert np.isfinite(LL)
    assert lmm.LLs.shape == (100,)  # default ngrids


def test_reml_loglik_finite_with_many_covariates(rng):
    """REML log-likelihood stays finite for large q.

    The old code used linalg.det(), which overflows to inf for many
    covariates, making log(inf) - log(inf) = nan. slogdet computes the
    log-determinant without ever materializing the determinant, so the
    REML term stays finite. Here det(X.T·X) genuinely overflows, so a
    finite REML LL proves the slogdet path is in use.
    """
    from scipy import linalg

    n, q = 400, 150
    K = np.eye(n)
    Y = rng.standard_normal(n)
    X0 = rng.standard_normal((n, q))
    lmm = LMM(Y, K, X0=X0)
    # Precondition: the naive determinant overflows.
    assert np.isinf(linalg.det(lmm.X0t.T @ lmm.X0t))
    LL = lmm.LL(0.5, REML=True)[0]
    assert np.isfinite(LL)


def test_tstat_matches_scipy():
    """t = beta / sqrt(var·sigma); p = 2·sf(|t|, N-q)."""
    lmm = LMM(np.random.default_rng(2).standard_normal(25), np.eye(25))
    beta = np.array([0.5])
    var = np.array([0.04])
    sigma = np.array([1.0])
    q = 2
    ts, ps = lmm.tstat(beta, var, sigma, q)

    expected_ts = 0.5 / np.sqrt(0.04 * 1.0)
    expected_ps = 2.0 * stats.t.sf(abs(expected_ts), lmm.N - q)
    assert isinstance(ts, float) and isinstance(ps, float)
    assert ts == pytest.approx(expected_ts)
    assert ps == pytest.approx(expected_ps)


def test_tstat_log_pvalue():
    """log=True returns log of the linear-scale p-value."""
    lmm = LMM(np.random.default_rng(3).standard_normal(25), np.eye(25))
    beta, var, sigma, q = np.array([0.5]), np.array([0.04]), np.array([1.0]), 2
    _, ps = lmm.tstat(beta, var, sigma, q)
    _, logp = lmm.tstat(beta, var, sigma, q, log=True)
    assert logp == pytest.approx(np.log(ps))


def test_tstat_rejects_nonscalar():
    """A non-length-1 effect estimate raises ValueError."""
    lmm = LMM(np.random.default_rng(4).standard_normal(20), np.eye(20))
    with pytest.raises(ValueError):
        lmm.tstat(np.array([1.0, 2.0]), np.array([0.1, 0.2]), np.array([1.0]), 2)


def test_association_recovers_signal(kinship, phenotype, genotypes):
    """The causal SNP (column 0) yields a small p-value; a null SNP does not."""
    lmm = LMM(phenotype, kinship)
    lmm.fit(REML=True)

    causal = genotypes[:, 0].reshape(-1, 1)
    ts, ps = lmm.association(causal, REML=False)
    assert isinstance(ts, float) and isinstance(ps, float)
    assert ps < 0.05

    # returnBeta yields four values
    ts2, ps2, beta, betavar = lmm.association(causal, REML=False, returnBeta=True)
    assert ts2 == pytest.approx(ts)
    assert np.isfinite(beta) and betavar >= 0


def test_meanandvar_posterior_moments(kinship, phenotype):
    """meanAndVar returns finite moments with mean in the grid range."""
    lmm = LMM(phenotype, kinship)
    lmm.fit()
    mean, var = lmm.meanAndVar()
    assert 0.0 <= mean <= 1.0
    assert var >= 0.0
