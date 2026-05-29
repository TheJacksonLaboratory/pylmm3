import logging
import time

import numpy as np

from pylmm3.lmm import LMM

logger = logging.getLogger(__name__)

_GWAS_DTYPE = np.dtype([
    ("SNP_ID",  "U50"),
    ("BETA",    np.float64),
    ("BETA_SD", np.float64),
    ("F_STAT",  np.float64),
    ("P_VALUE", np.float64),
])


def runGWAS(
    Y,
    K,
    snp_iter,
    X0=None,
    Kva=None,
    Kve=None,
    refit=False,
    REML=False,
    normalizeGenotype=True,
):
    """
    Run a GWAS scan using the linear mixed model.

    Parameters
    ----------
    Y : array-like, shape (N,)
        Phenotype vector. NaN entries are removed from Y, K, and X0 before
        fitting; snp_iter must still yield full-length N vectors.
    K : ndarray, shape (N, N)
        Pre-computed kinship matrix.
    snp_iter : iterable of (ndarray, str)
        Yields (snp_vector, snp_id) pairs. snp_vector must be length N.
    X0 : ndarray, shape (N, q), optional
        Covariate matrix. Defaults to a column of ones (intercept only).
    Kva, Kve : ndarray, optional
        Pre-computed eigenvalues/eigenvectors of K. Ignored when Y has
        missing values (eigendecomposition is recomputed after subsetting).
    refit : bool
        Re-estimate variance components at each SNP.
    REML : bool
        Use REML for the per-SNP association test (default False). The null
        model is always fit with REML=True, matching original pylmm behavior.
    normalizeGenotype : bool
        When False, SNPs with per-SNP missing individuals are standardized
        to zero mean / unit variance before testing.

    Returns
    -------
    numpy structured array with fields SNP_ID, BETA, BETA_SD, F_STAT, P_VALUE.
    Monomorphic or all-missing SNPs carry NaN in the numeric fields.
    Convert to DataFrame: ``pd.DataFrame(result)``.
    """
    Y = np.asarray(Y, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    # Remove individuals with missing phenotype
    keep = ~np.isnan(Y)
    if not keep.all():
        logger.info("Removing %d individuals with missing phenotype", (~keep).sum())
        Y  = Y[keep]
        K  = K[keep, :][:, keep]
        if X0 is not None:
            X0 = X0[keep, :]
        Kva = np.array([])
        Kve = np.array([])

    if X0 is None:
        X0 = np.ones((len(Y), 1))
    if Kva is None or (hasattr(Kva, '__len__') and len(Kva) == 0):
        Kva = np.array([])
    if Kve is None or (hasattr(Kve, '__len__') and len(Kve) == 0):
        Kve = np.array([])

    n = len(Y)

    t0 = time.perf_counter()
    L = LMM(Y, K, Kva, Kve, X0)
    if not refit:
        L.fit(REML=True)  # null model always uses REML, matching original pylmm behavior
        logger.info("Null fit: h=%.3f  sigma=%.3f  (%.3fs)", L.optH, L.optSigma, time.perf_counter() - t0)
    else:
        logger.info("LMM ready (%.3fs) — variance components will be refit per SNP", time.perf_counter() - t0)

    snp_ids = []
    betas    = []
    beta_sds = []
    f_stats  = []
    p_values = []

    t_scan = time.perf_counter()
    for count, (snp, snp_id) in enumerate(snp_iter, 1):
        if count % 1000 == 0:
            elapsed = time.perf_counter() - t_scan
            logger.info("At SNP %d  (%.0f SNPs/s)", count, count / elapsed)

        x = np.asarray(snp, dtype=np.float64)[keep].reshape((n, 1))
        v = np.isnan(x).reshape(-1)

        if v.any():
            keeps = ~v
            xs = x[keeps, :]
            if keeps.sum() <= 1 or xs.var() <= 1e-6:
                snp_ids.append(snp_id)
                betas.append(np.nan)
                beta_sds.append(np.nan)
                f_stats.append(np.nan)
                p_values.append(np.nan)
                continue

            if not normalizeGenotype:
                xs = (xs - xs.mean()) / np.sqrt(xs.var())

            Ys  = Y[keeps]
            X0s = X0[keeps, :]
            Ks  = K[keeps, :][:, keeps]
            Ls  = LMM(Ys, Ks, X0=X0s)
            if refit:
                Ls.fit(X=xs, REML=REML)
            else:
                Ls.fit(REML=REML)
            ts, ps, beta, betaVar = Ls.association(xs, REML=REML, returnBeta=True)
        else:
            if x.var() == 0:
                snp_ids.append(snp_id)
                betas.append(np.nan)
                beta_sds.append(np.nan)
                f_stats.append(np.nan)
                p_values.append(np.nan)
                continue

            if refit:
                L.fit(X=x, REML=REML)
            ts, ps, beta, betaVar = L.association(x, REML=REML, returnBeta=True)

        snp_ids.append(snp_id)
        betas.append(float(beta))
        beta_sds.append(float(np.sqrt(betaVar).sum()))
        f_stats.append(float(ts))
        p_values.append(float(ps))

    logger.info("Scanned %d SNPs in %.3fs", len(snp_ids), time.perf_counter() - t_scan)

    result = np.empty(len(snp_ids), dtype=_GWAS_DTYPE)
    result["SNP_ID"]  = snp_ids
    result["BETA"]    = betas
    result["BETA_SD"] = beta_sds
    result["F_STAT"]  = f_stats
    result["P_VALUE"] = p_values
    return result
