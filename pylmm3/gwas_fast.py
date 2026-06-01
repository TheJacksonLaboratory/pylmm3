"""Vectorized GWAS scan — batched matrix implementation of the LMM association test.

Instead of calling Kve.T @ x per SNP (gemv), B SNPs are batched into a matrix G
and Kve.T @ G is computed once (dgemm). The per-SNP information matrix is then
decomposed via the Schur complement of the fixed covariate block A, which is
constant across SNPs at fixed h=optH and inverted only once. The batch formulas
are algebraically identical to the per-SNP loop in gwas.py; see
docs/gwas_fast_design.md for the full derivation and validation results.
"""

import logging
import time

import numpy as np
from scipy import linalg, stats

from pylmm3.lmm import LMM
from pylmm3.gwas import _GWAS_DTYPE

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
    batch_size: int = 2000,
) -> np.ndarray:
    """Vectorized drop-in replacement for `pylmm3.gwas.runGWAS`.

    Produces numerically identical results (max relative error < 3e-9 vs
    the reference implementation). SNPs with missing genotypes or when
    `refit=True` fall back to the per-SNP loop from `gwas.py` unchanged,
    flushing any buffered batch first to preserve output ordering.

    The fast path batches fully-observed, non-monomorphic SNPs into blocks
    of `batch_size` columns and processes each block with a single dgemm
    call (`Kve.T @ G`). Per-batch scalar quantities are derived from the
    Schur complement of the fixed covariate block A (inverted once at the
    null-fit h=optH). See `docs/gwas_fast_design.md` for the full derivation.

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
            recomputed) when Y has missing values.
        Kve:
            Pre-computed eigenvectors of K, shape `(N, N)`. Same caveat as
            `Kva`.
        refit:
            If `True`, re-estimate variance components at every SNP. Forces
            the per-SNP fallback path for all SNPs, negating the speedup.
        REML:
            Whether to use REML for per-SNP association tests. The null model
            is always fit with `REML=True` regardless of this flag.
        normalizeGenotype:
            When `False`, SNPs with per-SNP missing individuals are
            standardized to zero mean / unit variance before testing.
        batch_size:
            Number of SNPs to accumulate into each matrix batch. Memory per
            batch is approximately `n * batch_size * 8` bytes (≈ 16 MB for
            n=1000, batch_size=2000). Reduce if memory is constrained.

    Returns:
        A numpy structured array with fields `SNP_ID`, `BETA`, `BETA_SD`,
        `F_STAT`, and `P_VALUE`. Monomorphic or all-missing SNPs carry `NaN`
        in the numeric fields. Convert to a DataFrame with
        `pd.DataFrame(result)`.
    """
    Y = np.asarray(Y, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

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
        L.fit(REML=True)
        logger.info("Null fit: h=%.3f  sigma=%.3f  (%.3fs)",
                    L.optH, L.optSigma, time.perf_counter() - t0)

        # All quantities below are fixed at h = optH and reused every batch.
        # Full derivation in docs/gwas_fast_design.md §3.
        h        = L.optH
        S        = 1.0 / (h * L.Kva + (1.0 - h))   # (n,) precision weights
        X0t      = L.X0t                             # (n, q) rotated covariates
        Yt       = L.Yt.ravel()                      # (n,)  rotated phenotype
        SX0t     = S[:, None] * X0t                  # (n, q)
        A        = X0t.T @ SX0t                      # (q, q) covariate info matrix
        A_inv    = linalg.inv(A)                     # (q, q) inverted once
        SYt      = S * Yt                            # (n,)
        c0       = X0t.T @ SYt                       # (q,)  constant
        YtSYt    = float(Yt @ SYt)                   # scalar constant
        A_inv_c0 = A_inv @ c0                        # (q,)
        dof      = n - (L.q + 1)
    else:
        logger.info("LMM ready (%.3fs) — variance components will be refit per SNP",
                    time.perf_counter() - t0)

    snp_ids  = []
    betas    = []
    beta_sds = []
    f_stats  = []
    p_values = []
    n_skipped = 0

    buf_snps = []   # (n,) genotype vectors queued for the next batch
    buf_ids  = []

    def flush_batch() -> None:
        """Process all buffered SNPs as a single vectorized batch.

        Rotates the buffer matrix G = stack(buf_snps) into the eigenbasis
        with one dgemm, then computes per-SNP effect sizes and t-statistics
        via the Schur complement formula derived in docs/gwas_fast_design.md.
        Results are appended to the outer snp_ids/betas/... lists and the
        buffer is cleared.

        SNPs where D_vec <= 0 (collinear with covariates in the rotated basis)
        receive NaN statistics rather than a division error.
        """
        if not buf_snps:
            return
        B  = len(buf_snps)
        G  = np.column_stack(buf_snps)               # (n, B)
        Gt = L.Kve.T @ G                             # (n, B) — one dgemm for all B SNPs

        SGt    = S[:, None] * Gt                     # (n, B)
        b_mat  = SX0t.T @ Gt                         # (q, B) covariate–SNP cross-products
        Ab_mat = A_inv @ b_mat                       # (q, B)
        d_vec  = (Gt * SGt).sum(0)                   # (B,) SNP self-dot-products
        D_vec  = d_vec - (b_mat * Ab_mat).sum(0)     # (B,) Schur complement; 1/D = XX_i⁻¹[q,q]
        c1     = SYt @ Gt                            # (B,) phenotype–SNP dot products
        adj    = A_inv_c0 @ b_mat                    # (B,) covariate adjustment

        with np.errstate(divide='ignore', invalid='ignore'):
            bs     = (c1 - adj) / D_vec              # (B,) SNP effect sizes
            beta0k = A_inv_c0[:, None] - Ab_mat * bs # (q, B) covariate effects
            Q      = YtSYt - (c0 @ beta0k) - bs * c1 # (B,) residual SS
            sigma  = Q / dof                          # (B,)
            var    = sigma / D_vec                    # (B,) sampling variance of β_snp
            sd     = np.sqrt(np.maximum(var, 0.0))   # (B,)
            ts     = bs / sd                          # (B,)

        ps  = 2.0 * stats.t.sf(np.abs(ts), dof)

        # D_vec <= 0 means the SNP is collinear with covariates in the rotated basis.
        bad = (D_vec <= 0) | ~np.isfinite(D_vec)
        bs[bad] = sd[bad] = ts[bad] = ps[bad] = np.nan

        snp_ids.extend(buf_ids)
        betas.extend(bs.tolist())
        beta_sds.extend(sd.tolist())
        f_stats.extend(ts.tolist())
        p_values.extend(ps.tolist())

        buf_snps.clear()
        buf_ids.clear()

    t_scan = time.perf_counter()
    for count, (snp, snp_id) in enumerate(snp_iter, 1):
        if count % 10000 == 0:
            elapsed = time.perf_counter() - t_scan
            logger.debug("At SNP %d  (%.0f SNPs/s)", count, count / elapsed)

        x = np.asarray(snp, dtype=np.float64)[keep]
        v = np.isnan(x)

        if refit or v.any():
            # Fallback path: identical to gwas.py.  Flush first to preserve order.
            flush_batch()

            if v.any():
                keeps = ~v
                xs = x[keeps].reshape(-1, 1)
                if keeps.sum() <= 1 or xs.var() <= 1e-6:
                    snp_ids.append(snp_id)
                    betas.append(np.nan); beta_sds.append(np.nan)
                    f_stats.append(np.nan); p_values.append(np.nan)
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
                    betas.append(np.nan); beta_sds.append(np.nan)
                    f_stats.append(np.nan); p_values.append(np.nan)
                    n_skipped += 1
                    continue
                if refit:
                    L.fit(X=x.reshape(-1, 1), REML=REML)
                ts, ps, beta, betaVar = L.association(
                    x.reshape(-1, 1), REML=REML, returnBeta=True)

            snp_ids.append(snp_id)
            betas.append(float(beta))
            beta_sds.append(float(np.sqrt(betaVar).sum()))
            f_stats.append(float(ts))
            p_values.append(float(ps))
            continue

        # Fast vectorized path: refit=False, no missing genotypes.
        if x.var() == 0:
            # Monomorphic SNP.  Flush before writing directly to output lists;
            # without this, buffered SNPs would appear after this one in the
            # output even though they came before it in the input.
            flush_batch()
            snp_ids.append(snp_id)
            betas.append(np.nan); beta_sds.append(np.nan)
            f_stats.append(np.nan); p_values.append(np.nan)
            n_skipped += 1
            continue

        buf_snps.append(x)
        buf_ids.append(snp_id)
        if len(buf_snps) == batch_size:
            flush_batch()

    flush_batch()  # remaining partial batch

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
