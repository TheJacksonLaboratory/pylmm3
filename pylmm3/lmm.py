
# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2015  Nicholas A. Furlotte (nick.furlotte@gmail.com)

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Linear mixed model (LMM) solver for GWAS — simplified EMMA/fastLMM implementation.

Optimizes the model Y = X·β + u + ε where u ~ N(0, h·σ²·K) and ε ~ N(0, (1-h)·σ²·I).
The three model parameters are heritability (h), covariate coefficients (β), and total
phenotypic variance (σ²).
"""

import sys
import time
import numpy as np
from scipy import linalg
from scipy import optimize
from scipy import stats


class LMM:
    """Linear mixed model solver (EMMA/fastLMM style).

    Fits the model Y = X·β + u + ε by maximum likelihood, where the genetic
    random effect u ~ N(0, h·σ²·K) captures population structure. Heritability
    h is the fraction of total variance explained by K.

    All inputs are expected to be numpy arrays. The constructor removes
    individuals with missing phenotype before any computation.

    Attributes:
        K: Kinship matrix subset to non-missing individuals, shape `(N, N)`.
        Kva: Eigenvalues of K (clamped to ≥ 1e-6), shape `(N,)`.
        Kve: Eigenvectors of K, shape `(N, N)`.
        N: Number of non-missing individuals.
        Y: Phenotype column vector, shape `(N, 1)`.
        X0: Covariate matrix, shape `(N, q)`.
        nonmissing: Boolean mask of length n (original) marking kept individuals.
        optH: Optimal heritability from the most recent `fit()` call.
        optSigma: Optimal total variance from the most recent `fit()` call.
        optBeta: Optimal covariate coefficients from the most recent `fit()` call.
        LLs: Grid of log-likelihoods computed during `fit()`.
    """

    def __init__(
        self,
        Y: np.ndarray,
        K: np.ndarray,
        Kva: np.ndarray | None = None,
        Kve: np.ndarray | None = None,
        X0: np.ndarray | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the LMM, removing missing phenotypes and computing eigens if needed.

        Args:
            Y:
                Phenotype vector of shape `(n,)` or `(n, 1)`. Both 1-D and
                column-vector layouts are accepted; internally flattened. `NaN`
                entries are dropped, and corresponding rows/columns of K and X0
                are removed before fitting.
            K:
                Pre-computed kinship matrix of shape `(n, n)`.
            Kva:
                Eigenvalues of K from `linalg.eigh(K)`, shape `(n,)`. If `None`
                or empty, the eigendecomposition is computed here (expensive
                for large n).
            Kve:
                Eigenvectors of K from `linalg.eigh(K)`, shape `(n, n)`. Must be
                supplied together with Kva; ignored when Kva is absent.
            X0:
                Covariate matrix of shape `(n, q)`. Defaults to a column of
                ones (intercept only).
            verbose:
                If `True`, write progress messages to stderr.
        """

        if X0 is None:
            X0 = np.ones(len(Y)).reshape(len(Y), 1)
        self.verbose = verbose

        x = ~np.isnan(Y)
        x = x.reshape(-1,)
        if x.sum() != len(Y): 
            if self.verbose:
                sys.stderr.write(
                    "Removing %d missing values from Y\n" %
                    ((~x).sum()))
            Y = Y[x]
            K = K[x, :][:, x]
            X0 = X0[x, :]
            Kva = []
            Kve = []
        self.nonmissing = x

        if Kva is None or len(Kva) == 0 or Kve is None or len(Kve) == 0:
            if self.verbose:
                sys.stderr.write(
                    "Obtaining eigendecomposition for %dx%d matrix\n" %
                    (K.shape[0], K.shape[1]))
            begin = time.time()
            Kva, Kve = linalg.eigh(K)
            end = time.time()
            if self.verbose:
                sys.stderr.write("Total time: %0.3f\n" % (end - begin))

        self.K = K
        self.Kva = Kva
        self.Kve = Kve
        self.N = self.K.shape[0]
        self.Y = Y.reshape((self.N, 1))
        self.X0 = X0

        n_clamped = int((self.Kva < 1e-6).sum())
        if n_clamped:
            if self.verbose:
                sys.stderr.write("Clamping %d near-zero eigenvalues to 1e-6\n" % n_clamped)
            self.Kva[self.Kva < 1e-6] = 1e-6

        self.transform()

    def transform(self) -> None:
        """Rotate Y and X0 into the eigenbasis of K.

        Left-multiplies Y and X0 by Kve.T. Because K = Kve·diag(Kva)·Kve.T,
        this rotation diagonalizes the LMM covariance: after rotation,
        Cov(Yt)_ii = σ²·(h·Kva_i + (1-h)), enabling elementwise precision
        weights in all subsequent likelihood computations.

        Also pre-allocates X0t_stack — X0t with one trailing column of ones
        appended — used as a mutable buffer by `LL()` and `fit()`. When a
        SNP vector is tested, column q of the buffer is overwritten in-place
        with the rotated SNP, avoiding a new allocation per SNP.
        """

        self.Yt = self.Kve.T @ self.Y
        self.X0t = self.Kve.T @ self.X0
        self.X0t_stack = np.hstack([self.X0t, np.ones((self.N, 1))])
        self.q = self.X0t.shape[1]

    def getMLSoln(
        self,
        h: float,
        X: np.ndarray,
    ) -> tuple:
        """Compute the ML estimates of β and σ² at fixed heritability h.

        At fixed h the LMM log-likelihood is maximized analytically. The
        covariance in the rotated basis is diagonal, so WLS with precision
        weights S_i = 1/(h·λ_i + (1-h)) gives the GLS estimator for β.

        Args:
            h:
                Heritability in [0, 1]. Determines the precision weights
                S_i = 1/(h·λ_i + (1-h)) applied to all weighted products.
            X:
                Design matrix in the *rotated* eigenbasis, shape `(N, q+k)`.
                Must already have been multiplied by Kve.T; typically X0t or
                X0t_stack with a SNP column filled in.

        Returns:
            A 5-tuple `(beta, sigma, Q, XX_i, XX)` where:

            - **beta** — GLS coefficient vector, shape `(q+k, 1)`.
            - **sigma** — ML estimate of σ² (scalar array). Computed as
              Q / (N − (q+k)).
            - **Q** — weighted residual SS: (Yt − X·β).T·diag(S)·(Yt − X·β).
            - **XX_i** — inverse of the precision-weighted information matrix
              X.T·diag(S)·X, shape `(q+k, q+k)`. Its diagonal gives the
              sampling variances of β (up to σ²).
            - **XX** — the precision-weighted information matrix before
              inversion, shape `(q+k, q+k)`.
        """

        S = 1.0 / (h * self.Kva + (1.0 - h))
        Xt = X.T * S
        XX = Xt @ X
        XX_i = linalg.inv(XX)
        beta = XX_i @ Xt @ self.Yt
        Yt = self.Yt - X @ beta
        Q = np.dot(Yt.T * S, Yt)
        sigma = Q * 1.0 / (float(self.N) - float(X.shape[1]))
        return beta, sigma, Q, XX_i, XX

    def LL_brent(
        self,
        h: float,
        X: np.ndarray | None = None,
        REML: bool = False,
    ) -> float:
        """Negated log-likelihood for use as a Brent minimization objective.

        Returns the negative of `LL(h)` so that `scipy.optimize.brent`
        (which minimizes) finds the ML maximum. Returns a large constant
        (1e6) when h < 0 because Brent's method is not strictly bounded by its
        bracket and can stray negative, where the LL formula is undefined.

        Args:
            h:
                Heritability candidate. Values below 0 return 1e6 without
                evaluating the likelihood.
            X:
                Pre-rotated design matrix passed through to `LL()`. When
                `None`, `LL()` falls back to X0t.
            REML:
                Whether to evaluate the REML-corrected log-likelihood.

        Returns:
            Negative log-likelihood (float), or 1e6 if h < 0.
        """
        if h < 0:
            return 1e6
        return -self.LL(h, X, stack=False, REML=REML)[0]

    def LL(
        self,
        h: float,
        X: np.ndarray | None = None,
        stack: bool = True,
        REML: bool = False,
    ) -> tuple:
        """Compute the profile log-likelihood of the LMM at fixed heritability h.

        With β and σ² profiled out analytically (via `getMLSoln`), the
        ML log-likelihood reduces to:

            LL = -½ [n·log(2π) + Σlog(h·λᵢ + (1-h)) + n + n·log(Q/n)]

        where Σlog(h·λᵢ + (1-h)) is the log-determinant of the scaled
        covariance Σ/σ², and Q/n is the profiled σ².

        When `REML=True`, the standard REML correction is added:

            +½ [q·log(2π·σ²) + log|X.T·X| − log|X.T·Σ⁻¹·X|]

        **Known issue (BUG-B):** `linalg.det()` overflows for matrices
        larger than ~500×500. The REML path will produce `inf` on full
        cohorts. Fix: replace with `np.linalg.slogdet`.

        Args:
            h:
                Heritability in [0, 1].
            X:
                Covariate/SNP design vector or matrix. Interpretation depends
                on `stack`:

                - `None` → use X0t (null model with covariates only).
                - array + `stack=True` → rotate X by Kve.T and write into
                  column q of X0t_stack (modifies the buffer in-place).
                - array + `stack=False` → use X directly; caller is
                  responsible for rotation into the eigenbasis.
            stack:
                If `True` and X is not `None`, rotate and stack X onto X0t
                before computing. Set `False` when X is already in the rotated
                eigenbasis (e.g. inside `fit()`).
            REML:
                If `True`, add the REML correction term to the ML
                log-likelihood. Default `False` for association tests;
                the null model fit always passes `True` explicitly.

        Returns:
            A 4-tuple `(LL, beta, sigma, XX_i)`:

            - **LL** — scalar log-likelihood value.
            - **beta** — ML coefficient vector from `getMLSoln`.
            - **sigma** — ML total variance estimate from `getMLSoln`.
            - **XX_i** — inverse precision-weighted information matrix from
              `getMLSoln`; its diagonal gives sampling variances of β.
        """

        if X is None:
            X = self.X0t
        elif stack:
            self.X0t_stack[:, self.q] = (self.Kve.T @ X)[:, 0]
            X = self.X0t_stack

        n = float(self.N)
        q = float(X.shape[1])
        beta, sigma, Q, XX_i, XX = self.getMLSoln(h, X)
        LL = n * np.log(2 * np.pi) + np.log(h * self.Kva + \
                        (1.0 - h)).sum() + n + n * np.log(1.0 / n * Q)
        LL = -0.5 * LL

        if REML:
            # linalg.det() overflows to inf when the number of covariates q ≥ ~100
            # (e.g. principal-component covariates), making log(inf) - log(inf) = nan.
            # np.linalg.slogdet() computes the log-determinant via LU without ever
            # materialising the determinant itself, so it is numerically safe for any q.
            _, logdet_XTX = np.linalg.slogdet(X.T @ X)
            _, logdet_XX  = np.linalg.slogdet(XX)
            LL_REML_part = q * np.log(2.0 * np.pi * sigma) + logdet_XTX - logdet_XX
            LL = LL + 0.5 * LL_REML_part

        LL = LL.sum()
        return LL, beta, sigma, XX_i

    def getMax(
        self,
        H: np.ndarray,
        X: np.ndarray | None = None,
        REML: bool = False,
    ) -> float:
        """Find the MLE of h by scanning the LL grid and refining with Brent's method.

        Identifies local maxima in `self.LLs` (points strictly higher than
        both neighbors), brackets each with adjacent grid points, and calls
        `scipy.optimize.brent` via `LL_brent` to find the precise optimum
        within each bracket. When no interior maximum exists, falls back to
        whichever grid endpoint has the higher LL. If multiple optima are
        found, the first is returned with a verbose warning.

        Args:
            H:
                Heritability grid array, shape `(ngrids,)`. Must be the same
                grid used to compute `self.LLs` so that indices correspond.
            X:
                Pre-rotated design matrix in the eigenbasis, passed through to
                `LL_brent`. Typically X0t or X0t_stack.
            REML:
                Whether to use the REML-corrected likelihood during Brent
                refinement. Must match the criterion used to build `self.LLs`.

        Returns:
            Optimal heritability estimate (float) in approximately [0, 1].
        """
        n = len(self.LLs)
        HOpt = []
        for i in range(1, n - 2):
            if self.LLs[i - 1] < self.LLs[i] and self.LLs[i] > self.LLs[i + 1]:
                HOpt.append(optimize.brent(self.LL_brent, args=(
                    X, REML), brack=(H[i - 1], H[i + 1])))
                if np.isnan(HOpt[-1]):
                    HOpt[-1] = H[i - 1]
                # if np.isnan(HOpt[-1]): HOpt[-1] = self.LLs[i-1]
                # if np.isnan(HOpt[-1][0]): HOpt[-1][0] = [self.LLs[i-1]]

        if len(HOpt) > 1:
            if self.verbose:
                sys.stderr.write(
                    "NOTE: Found multiple optima.  Returning first...\n")
            return HOpt[0]
        elif len(HOpt) == 1:
            return HOpt[0]
        elif self.LLs[0] > self.LLs[n - 1]:
            return H[0]
        else:
            return H[n - 1]

    def fit(
        self,
        X: np.ndarray | None = None,
        ngrids: int = 100,
        REML: bool = True,
    ) -> tuple:
        """Find the MLE of heritability and store all associated ML quantities.

        Evaluates the log-likelihood on a uniform grid of `ngrids` points in
        [0, 1), then calls `getMax` to bracket and refine the optimum with
        Brent's method. Stores results in instance attributes for use by
        `association`.

        Typically called once on the null model (no SNP in X) before the GWAS
        scan. Callers that need to refit variance components per SNP pass X
        explicitly.

        Args:
            X:
                Optional SNP or additional covariate vector/matrix in the
                *unrotated* space, shape `(N, 1)`. If provided, it is rotated
                by Kve.T and appended to X0t before fitting. Pass `None`
                (default) for the null model with covariates only.
            ngrids:
                Number of equally spaced heritability values to evaluate
                before refining with Brent. Higher values reduce the chance
                of missing a narrow peak but cost proportionally more LL
                evaluations.
            REML:
                Whether to maximize the REML-corrected log-likelihood.
                The null model should always use `REML=True` (default) to
                match the original pylmm behavior. Per-SNP association tests
                typically use `REML=False`.

        Returns:
            A 4-tuple `(hmax, beta, sigma, LL)` where `hmax` is the optimal
            heritability, `beta` and `sigma` are the ML coefficient and variance
            estimates at `hmax`, and `LL` is the maximized log-likelihood.
            The same values are also stored in `self.optH`, `self.optBeta`,
            `self.optSigma`, and `self.optLL`.
        """

        if X is None:
            X = self.X0t
        else:
            self.X0t_stack[:, self.q] = (self.Kve.T @ X)[:, 0]
            X = self.X0t_stack

        H = np.array(range(ngrids)) / float(ngrids)
        L = np.array([self.LL(h, X, stack=False, REML=REML)[0] for h in H])
        self.LLs = L

        hmax = self.getMax(H, X, REML)
        L, beta, sigma, betaSTDERR = self.LL(hmax, X, stack=False, REML=REML)

        self.H = H
        self.optH = hmax
        self.optLL = L
        self.optBeta = beta
        self.optSigma = sigma.sum()

        return hmax, beta, sigma, L

    def association(
        self,
        X: np.ndarray,
        h: float | None = None,
        stack: bool = True,
        REML: bool = True,
        returnBeta: bool = False,
    ) -> tuple:
        """Compute the association statistic for a single SNP.

        Appends the SNP vector X to the covariate matrix, evaluates the LL at
        the fixed heritability h (defaulting to the null-fit optH), and
        extracts the t-statistic and p-value for the SNP's effect size from
        the last row/column of the information matrix inverse.

        Args:
            X:
                SNP genotype vector or column matrix in the *unrotated* space,
                shape `(N, 1)`. Rotated internally when `stack=True`.
            h:
                Heritability to use for the association test. When `None`
                (default), `self.optH` from the null fit is used. Passing an
                explicit value re-evaluates the LL at that h without touching
                stored optH.
            stack:
                If `True` (default), rotate X by Kve.T and write it into
                column q of X0t_stack before computing. Set `False` only
                when X is already in the rotated eigenbasis.
            REML:
                Whether to evaluate the REML-corrected log-likelihood for the
                association test. The null model is always fit with `REML=True`;
                per-SNP tests default to `False`.
            returnBeta:
                If `True`, also return the SNP effect size and its sampling
                variance (var·σ²) in addition to the test statistics.

        Returns:
            `(ts, ps)` by default, or `(ts, ps, beta, betaVar)` when
            `returnBeta=True`. `ts` is the t-statistic and `ps` is the
            two-sided p-value for the SNP effect.
        """
        if stack:
            self.X0t_stack[:, self.q] = (self.Kve.T @ X)[:, 0]
            X = self.X0t_stack

        if h is None:
            h = self.optH

        L, beta, sigma, betaVAR = self.LL(h, X, stack=False, REML=REML)
        q = len(beta)
        ts, ps = self.tstat(beta[q - 1], betaVAR[q - 1, q - 1], sigma, q)

        if returnBeta:
            return ts, ps, beta[q - 1].sum(), betaVAR[q - 1,
                                                      q - 1].sum() * sigma
        return ts, ps

    def tstat(
        self,
        beta: np.ndarray,
        var: np.ndarray,
        sigma: np.ndarray,
        q: int,
        log: bool = False,
    ) -> tuple[float, float]:
        """Compute a t-statistic and two-sided p-value for a single effect estimate.

        The test statistic is t = β / SE(β), where SE(β) = √(var·σ²). Here
        `var` is the diagonal element of XX_i corresponding to the SNP
        coefficient, and `sigma` is the ML total variance estimate, so var·σ²
        is the sampling variance of β. Although this is formally an F-test,
        with a single SNP hypothesis it reduces to a two-sided t-test with
        N − q degrees of freedom.

        Uses `stats.t.sf` (survival function) rather than the CDF complement
        for better numerical precision at extreme p-values.

        Args:
            beta:
                ML estimate of the SNP effect size. Expected to be a
                length-1 array (scalar wrapped in a numpy array).
            var:
                Diagonal element of XX_i for the SNP coefficient. Combined
                with `sigma` gives the sampling variance: var·sigma.
            sigma:
                ML total variance estimate σ² from `getMLSoln`.
            q:
                Number of columns in the full design matrix (covariates + SNP).
                Used as the df adjustment: df = N − q.
            log:
                If `True`, return the natural log of the p-value using
                `stats.t.logsf` instead of the raw p-value. Useful for very
                small p-values where the linear-scale result would underflow.

        Returns:
            `(ts, ps)` — the t-statistic and p-value (or log p-value when
            `log=True`) as Python floats extracted via `.sum()`.

        Raises:
            Exception: If `ts` or `ps` have length != 1, indicating an
                unexpected shape in the intermediate arrays.
        """

        ts = beta / np.sqrt(var * sigma)
        if log:
            ps = np.log(2.0) + stats.t.logsf(np.abs(ts), self.N - q)
        else:
            ps = 2.0 * (stats.t.sf(np.abs(ts), self.N - q))
        if not len(ts) == 1 or not len(ps) == 1:
            raise Exception("Something bad happened :(")
        return ts.sum(), ps.sum()

    def plotFit(
        self,
        color: str = 'b-',
        title: str = '',
    ) -> None:
        """Plot an approximate posterior distribution over heritability.

        Converts the grid log-likelihoods to normalized probabilities via
        exp(LL − max(LL)) / Z. With a flat prior on h, this is proportional
        to the true posterior over heritability. Use this after `fit()` to
        visually diagnose whether the likelihood surface has one clean peak or
        multiple local optima, and to assess uncertainty in the heritability
        estimate.

        Requires matplotlib; imported lazily so it is not a hard dependency.

        Args:
            color:
                matplotlib line style string passed directly to `pl.plot`.
                Defaults to `'b-'` (solid blue).
            title:
                Plot title string passed to `pl.title`.
        """
        import matplotlib.pyplot as pl

        mx = self.LLs.max()
        p = np.exp(self.LLs - mx)
        p = p / p.sum()

        pl.plot(self.H, p, color)
        pl.xlabel("Heritability")
        pl.ylabel("Probability of data")
        pl.title(title)

    def meanAndVar(self) -> tuple[float, float]:
        """Compute the mean and variance of h under the approximate posterior.

        Uses the same normalized probability weights as `plotFit`: treats
        exp(LL − max(LL)) / Z as a discrete distribution over the heritability
        grid and computes its first two central moments.

        Returns:
            `(mean, variance)` of heritability h under the grid-based
            approximate posterior.
        """

        mx = self.LLs.max()
        p = np.exp(self.LLs - mx)
        p = p / p.sum()

        mn = (self.H * p).sum()
        vx = ((self.H - mn)**2 * p).sum()

        return mn, vx
