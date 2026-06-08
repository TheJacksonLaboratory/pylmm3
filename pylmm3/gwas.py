"""Reference GWAS scan implementation — per-SNP linear mixed model loop.

Numerically validated against the original pylmm output (max relative error
< 3e-9). Prefer `gwas_fast.runGWAS` for production runs; this module is
the correctness reference and the fallback for missing-genotype SNPs.
"""

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
    Y: np.ndarray,
    K: np.ndarray,
    snp_iter,
    X0: np.ndarray | None = None,
    Kva: np.ndarray | None = None,
    Kve: np.ndarray | None = None,
    refit: bool = False,
    REML: bool = False,
    normalizeGenotype: bool = True,
) -> np.ndarray:
    """Run a GWAS scan using the linear mixed model (reference implementation).

    Fits a null LMM once (REML=True), then tests each SNP in a per-SNP loop
    by calling `LMM.association`. SNPs with missing genotypes spawn a fresh
    sub-LMM on the reduced sample set. Monomorphic and all-missing SNPs are
    emitted with NaN statistics without raising an error.

    For large cohorts prefer `gwas_fast.runGWAS`, which is algebraically
    identical but 50–200× faster via batched matrix operations.

    Args:
        Y:
            Phenotype vector of shape `(N,)`. `NaN` entries are removed from
            Y, K, and X0 before fitting; `snp_iter` must still yield full-
            length N vectors since subsetting is applied internally.
        K:
            Pre-computed kinship matrix of shape `(N, N)`.
        snp_iter:
            Iterable of `(snp_vector, snp_id)` pairs. Each `snp_vector` must
            be length N and may contain `np.nan` for missing genotypes.
        X0:
            Covariate matrix of shape `(N, q)`. Defaults to a column of ones
            (intercept only).
        Kva:
            Pre-computed eigenvalues of K, shape `(N,)`. Ignored (and
            recomputed) when Y has missing values, since K is subset before
            the eigendecomposition.
        Kve:
            Pre-computed eigenvectors of K, shape `(N, N)`. Same caveat as
            `Kva`.
        refit:
            If `True`, re-estimate variance components (h, σ²) at every SNP
            instead of using the null-fit optH. Much slower; use only when
            the assumption of a fixed genetic architecture across SNPs is
            unacceptable.
        REML:
            Whether to use the REML-corrected log-likelihood for per-SNP
            association tests. The null model is always fit with `REML=True`
            regardless of this flag, matching original pylmm behavior.
        normalizeGenotype:
            When `False`, SNPs with per-SNP missing individuals are
            standardized to zero mean / unit variance before testing.
            Has no effect on fully-observed SNPs (already normalized by the
            `plink` reader).

    Returns:
        A numpy structured array with fields `SNP_ID`, `BETA`, `BETA_SD`,
        `F_STAT`, and `P_VALUE`. Monomorphic or all-missing SNPs carry `NaN`
        in the numeric fields. Convert to a DataFrame with
        `pd.DataFrame(result)`.
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
    logger.info("LMM ready in %.3fs (eigendecomposition included unless precomputed)", time.perf_counter() - t0)
    if not refit:
        t_fit = time.perf_counter()
        L.fit(REML=True)  # null model always uses REML, matching original pylmm behavior
        logger.info("Null fit: h=%.3f  sigma=%.3f  (%.3fs)", L.optH, L.optSigma, time.perf_counter() - t_fit)
    else:
        logger.info("Variance components will be refit per SNP (no null fit)")

    snp_ids  = []
    betas    = []
    beta_sds = []
    f_stats  = []
    p_values = []
    n_skipped = 0

    t_scan = time.perf_counter()
    for count, (snp, snp_id) in enumerate(snp_iter, 1):
        if count % 10000 == 0:
            elapsed = time.perf_counter() - t_scan
            logger.debug("At SNP %d  (%.0f SNPs/s)", count, count / elapsed)

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
                n_skipped += 1
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
                n_skipped += 1
                continue

            if refit:
                L.fit(X=x, REML=REML)
            ts, ps, beta, betaVar = L.association(x, REML=REML, returnBeta=True)

        snp_ids.append(snp_id)
        betas.append(float(beta))
        beta_sds.append(float(np.sqrt(betaVar).sum()))
        f_stats.append(float(ts))
        p_values.append(float(ps))

    total = len(snp_ids)
    logger.info(
        "Scanned %d SNPs in %.3fs — skipped %d (%.1f%%) due to missing genotypes or low variance",
        total, time.perf_counter() - t_scan, n_skipped, 100.0 * n_skipped / total if total else 0.0,
    )

    result = np.empty(len(snp_ids), dtype=_GWAS_DTYPE)
    result["SNP_ID"]  = snp_ids
    result["BETA"]    = betas
    result["BETA_SD"] = beta_sds
    result["F_STAT"]  = f_stats
    result["P_VALUE"] = p_values
    return result
