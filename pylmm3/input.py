
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

"""PLINK file reader supporting BED (binary), TPED, and EMMA genotype formats.

Provides the `plink` class, which is an iterator that streams one SNP at a time
from a PLINK dataset. Each iteration yields a `(genotype_array, snp_id)` pair.
Optionally reads a kinship matrix and phenotype file from companion files.

Iteration is driven by a generator returned from `__iter__`: genotype file
handles live in the generator frame under a `with` block, so they close
deterministically when iteration finishes, breaks, or raises. Re-iterating or
nesting iteration over the same `plink` object is safe — each pass gets its own
handles and never leaks descriptors.
"""

import logging
import os
from collections.abc import Iterator

import numpy as np

logger = logging.getLogger(__name__)

# quick lookup table for decoding PLINK .bed genotypes
_BED_LOOKUP = np.array([0.0, np.nan, 0.5, 1.0], dtype=np.float64)


class plink:
    """Iterator over SNPs in a PLINK dataset (BED, TPED, or EMMA format).

    Each call to `next()` yields a `(genotype_array, snp_id)` pair where
    `genotype_array` is a float64 numpy array of length N (the number of
    individuals) with values in {0.0, 0.5, 1.0, np.nan}, and `snp_id` is the
    RS identifier string from the `.bim` / `.map` file.

    The class reads phenotypes and optionally a kinship matrix from companion
    files on construction, then streams SNP data lazily on iteration.

    Attributes:
        fbase: Base path prefix for all PLINK files (no extension).
        type: Format indicator — `'b'` for BED, `'t'` for TPED, `'emma'` for EMMA.
        indivs: List of `(fam_id, indiv_id)` tuples in dataset order.
        N: Number of individuals in the dataset.
        kFile: Path to the kinship file (`.kin`), or `None` if not found.
        K: Kinship matrix as a numpy array, or `None` if not loaded.
        phenos: Phenotype matrix of shape `(N, num_traits)`, or `None`.
        normGenotype: Whether to normalize each SNP to mean 0 / variance 1.
        numSNPs: Total number of SNPs in the dataset, or -1 for EMMA (which
            has no index); set by `getSNPIterator`.
    """

    def __init__(
            self,
            fbase: str,
            kFile: str | None = None,
            phenoFile: str | None = None,
            type: str = 'b',
            normGenotype: bool = True,
            readKFile: bool = False,
            fastLMM_kinship: bool = False,
    ) -> None:
        """Initialize the PLINK reader and load companion files.

        Args:
            fbase:
                Base path for the PLINK fileset without extension (e.g.
                `'/data/study'`). The reader will append `.bed`/`.bim`/`.fam`
                (BED mode), `.tped`/`.tfam`/`.map` (TPED mode), or use
                `fbase` directly as the SNP file (EMMA mode).
            kFile:
                Explicit path to a kinship matrix file. If `None`, the reader
                looks for `fbase + '.kin'`. If that is also absent, `self.K`
                is set to `None`.
            phenoFile:
                Path to a whitespace-delimited phenotype file with columns
                `fam_id indiv_id pheno1 [pheno2 ...]`. If `None`, defaults to
                `fbase + '.phenos'`.
            type:
                Genotype file format: `'b'` for binary BED, `'t'` for TPED,
                `'emma'` for plain-text EMMA format (one SNP per line, values
                space-separated).
            normGenotype:
                If `True` (default), each yielded genotype array is normalized
                to zero mean and unit variance after imputing missing values
                with the column mean.
            readKFile:
                If `True`, load the kinship matrix into `self.K` on
                construction. If `False` (default), the file path is recorded
                but the matrix is not read.
            fastLMM_kinship:
                If `True`, parse the kinship file in fastLMM tab-delimited
                format (header row of IDs, row-leading ID column). If `False`
                (default), parse as a plain whitespace-delimited matrix
                without ID columns.
        """

        self.fbase = fbase
        self.type = type
        if not type == 'emma':
            self.indivs = self.getIndivs(self.fbase, type)
        else:
            # Just read a line from the SNP file and see how many individuals
            # we have
            f = open(fbase, 'r')
            self.indivs = range(len(f.readline().strip().split()))
            f.close()
        self.kFile = kFile
        self.phenos = None
        self.normGenotype = normGenotype
        self.phenoFile = phenoFile
        # Originally I was using the fastLMM style that has indiv IDs embedded.
        # NOW I want to use this module to just read SNPs so I'm allowing
        # the programmer to turn off the kinship reading.
        self.readKFile = readKFile
        self.fastLMM_kinship_style = fastLMM_kinship

        if self.kFile:
            self.K = self.readKinship(self.kFile)
        elif os.path.isfile("%s.kin" % fbase):
            self.kFile = "%s.kin" % fbase
            if self.readKFile:
                self.K = self.readKinship(self.kFile)
        else:
            self.kFile = None
            self.K = None

        self.getPhenos(self.phenoFile)

    def getSNPIterator(self) -> "Iterator[tuple[np.ndarray, str]]":
        """Set `self.numSNPs` and return a fresh SNP iterator.

        Retained for callers (e.g. `load_snp_matrix`) that read `self.numSNPs`
        after calling this. Iteration itself is driven by a generator returned
        from `__iter__`, so file handles are scoped to the generator frame and
        closed deterministically rather than living on the instance.

        Returns:
            A fresh iterator over `(genotype_array, snp_id)` pairs.
        """
        self.numSNPs = self._count_snps()
        return iter(self)

    def _count_snps(self) -> int:
        """Count SNPs without opening the genotype stream.

        Reads the line count of the `.bim` index (falling back to `.map` for
        TPED). EMMA files have no index, so -1 is returned and iteration stops
        at end-of-file instead.

        Returns:
            Number of SNPs, or -1 for EMMA format.
        """
        if self.type == 'emma':
            return -1
        index = self.fbase + '.bim'
        if self.type == 't' and not os.path.isfile(index):
            index = self.fbase + '.map'
        with open(index, 'r') as f:
            return sum(1 for _ in f)

    def _iter_emma(self) -> "Iterator[tuple[np.ndarray, str]]":
        """Yield `(genotype_array, snp_id)` pairs from an EMMA-format file.

        Each line is one SNP of space-separated dosage values; non-numeric
        tokens become `np.nan`. SNP ids are synthesized as `SNP_<n>` (1-based)
        since EMMA files carry no rs identifiers. The handle is scoped to this
        generator and closed on exhaustion, `break`, or exception.
        """
        with open(self.fbase, 'r') as fhandle:
            for i, line in enumerate(fhandle, 1):
                G = []
                for x in line.strip().split():
                    try:
                        G.append(float(x))
                    except ValueError:
                        G.append(np.nan)
                G = np.array(G)
                if self.normGenotype:
                    G = self.normalizeGenotype(G)
                yield G, "SNP_%d" % i

    def _iter_tped(self) -> "Iterator[tuple[np.ndarray, str]]":
        """Yield `(genotype_array, snp_id)` pairs from a TPED fileset.

        Genotypes are decoded from `.tped` (stripping the four leading header
        fields); SNP ids are taken from column 2 of the `.bim` index (falling
        back to `.map`). Both handles are scoped to this generator. Iteration
        stops when either file is exhausted.
        """
        index = self.fbase + '.bim'
        if not os.path.isfile(index):
            index = self.fbase + '.map'
        with open(index, 'r') as idx, open(self.fbase + '.tped', 'r') as tped:
            for idx_line, geno_line in zip(idx, tped):
                G = self.getGenos_tped(geno_line.strip().split()[4:])
                if self.normGenotype:
                    G = self.normalizeGenotype(G)
                yield G, idx_line.strip().split()[1]

    def _iter_bed(self) -> "Iterator[tuple[np.ndarray, str]]":
        """Yield `(genotype_array, snp_id)` pairs from a BED fileset.

        Validates the two-byte PLINK magic number (`0x6c 0x1b`) and the
        SNP-major mode byte (`0x01`), then reads one packed record per `.bim`
        line, pairing it with the rs id in column 2 of that line. Both handles
        are scoped to this generator.

        Raises:
            ValueError: If the magic number is invalid or the file is not in
                SNP-major mode.
        """
        bytes_per_snp = self.N // 4 + (self.N % 4 and 1 or 0)
        logger.debug("BED bytes per SNP: %d", bytes_per_snp)
        with open(self.fbase + '.bim', 'r') as bim, open(self.fbase + '.bed', 'rb') as bed:
            if bed.read(2) != b'\x6c\x1b':
                raise ValueError(f"Invalid PLINK BED magic number in {self.fbase}.bed")
            if bed.read(1) != b'\x01':
                raise ValueError(f"BED file is not in SNP-major mode: {self.fbase}.bed")
            for line in bim:
                raw = bed.read(bytes_per_snp)
                G = self.formatBinaryGenotypes(raw, self.normGenotype)
                yield G, line.strip().split()[1]

    def __iter__(self) -> "Iterator[tuple[np.ndarray, str]]":
        """Return a fresh generator over `(genotype_array, snp_id)` pairs.

        Each call opens its own genotype file handles inside the generator
        frame, so re-iterating or nesting iteration over the same `plink`
        object is safe and never leaks descriptors.
        """
        if self.type == 'b':
            return self._iter_bed()
        elif self.type == 't':
            return self._iter_tped()
        elif self.type == 'emma':
            return self._iter_emma()
        logger.error("Unknown SNP type %r — expected 'b', 't', or 'emma'", self.type)
        return iter(())

    def getGenos_tped(
        self,
        X: list[str],
    ) -> np.ndarray:
        """Decode one SNP row from a PLINK TPED file into a dosage array.

        The TPED format encodes diploid genotypes as pairs of numeric allele
        strings. After stripping the four header fields (chr, rsid, cM, bp),
        each individual contributes exactly two adjacent tokens, so
        `len(X) == 2 * N`.

        PLINK numeric allele encoding:

        | Allele pair  | Dosage   | Meaning                              |
        |--------------|----------|--------------------------------------|
        | `'0' '0'`    | `np.nan` | Missing genotype                     |
        | `'1' '1'`    | `0.0`    | Homozygous reference                 |
        | `'2' '2'`    | `1.0`    | Homozygous alternate                 |
        | `'1' '2'`    | `0.5`    | Heterozygous (either phase)          |
        | `'0' x`      | `0.5`    | One allele known, one missing — het  |
        | other equal  | `np.nan` | Nucleotide-encoded or unrecognized   |

        The catch-all else branch emits `np.nan` for any equal-allele pair
        that is not `'0'`, `'1'`, or `'2'` (e.g. `'A'/'A'` from nucleotide-
        encoded TPED files). Without it, such input raises `UnboundLocalError`.

        Args:
            X:
                Flat list of allele strings for one SNP, stripped of the four
                leading header fields. Length must be `2 * N`.

        Returns:
            Float64 array of dosage values, shape `(N,)`.
        """
        G = []
        for a, b in zip(X[::2], X[1::2]):
            if a == b == '0':
                G.append(np.nan)
            elif a == b == '1':
                G.append(0.0)
            elif a == b == '2':
                G.append(1.0)
            elif a != b:
                G.append(0.5)
            else:
                logger.warning("Unrecognized allele pair: %r %r", a, b)
                G.append(np.nan)
        return np.array(G)
    

    def formatBinaryGenotypes(
        self,
        raw: bytes,
        norm: bool = True,
    ) -> np.ndarray:
        """Decode raw bytes from a PLINK BED record into a float64 genotype array.

        Each byte packs four 2-bit genotype codes. The codes are mapped via
        `_BED_LOOKUP` (a 4-element table indexed by the 2-bit value):

        | 2-bit code | Dosage   | Meaning                    |
        |------------|----------|----------------------------|
        | `0b00`     | `1.0`    | Homozygous alternate (AA)  |
        | `0b01`     | `np.nan` | Missing                    |
        | `0b10`     | `0.5`    | Heterozygous (Aa)          |
        | `0b11`     | `0.0`    | Homozygous reference (aa)  |

        The output is trimmed to the first `self.N` elements to discard
        padding bits in the last byte.

        Args:
            raw:
                Bytes object for a single SNP record, length
                `ceil(N / 4)` bytes.
            norm:
                If `True` (default), normalize the decoded array to zero mean
                and unit variance via `normalizeGenotype` before returning.

        Returns:
            Float64 array of dosage values, shape `(N,)`.
        """
        arr = np.frombuffer(raw, dtype=np.uint8)
        out = np.empty(len(arr) * 4, dtype=np.float64)
        for shift in (0, 2, 4, 6):
            out[shift // 2::4] = _BED_LOOKUP[(arr >> shift) & 0x3]
        G = out[:self.N]
        if norm:
            G = self.normalizeGenotype(G)
        return G


    def normalizeGenotype(
        self,
        G: np.ndarray,
    ) -> np.ndarray:
        """Normalize a genotype array to zero mean and unit variance.

        Missing values (`np.nan`) are excluded from the mean and variance
        calculation, then imputed with the column mean before scaling.
        If all values are missing, the original array is returned unchanged.
        If the variance is zero (monomorphic SNP), the array is centered but
        not scaled (s is set to 1.0 to avoid division by zero).

        Args:
            G:
                Float64 genotype array of shape `(N,)`, possibly containing
                `np.nan` for missing genotypes.

        Returns:
            Normalized float64 array of shape `(N,)` with missing values
            imputed as the mean.
        """
        x = ~np.isnan(G)

        # All genotypes missing: keep original length & NaNs
        if not x.any():
            return G

        m = G[x].mean()
        v = G[x].var()
        s = 1.0 if v == 0 else np.sqrt(v)
        G[np.isnan(G)] = m
        G = (G - m) / s

        return G
    

    def getPhenos(
        self,
        phenoFile: str | None = None,
    ) -> np.ndarray | None:
        """Load the phenotype file and store results in `self.phenos`.

        The file is whitespace-delimited with columns
        `fam_id indiv_id pheno1 [pheno2 ...]`. Values of `'NA'` or `'-9'`
        are converted to `np.nan`. For BED/TPED datasets, rows are reordered
        to match `self.indivs`; individuals in the phenotype file but absent
        from the genotype dataset are silently dropped.

        Args:
            phenoFile:
                Path to the phenotype file. If `None`, defaults to
                `self.fbase + '.phenos'`. If that file does not exist,
                the method returns without modifying `self.phenos`.

        Returns:
            Phenotype matrix of shape `(N, num_traits)`, or `None` if the
            file was not found. Also stored in `self.phenos`.
        """
        if not phenoFile:
            self.phenoFile = phenoFile = self.fbase + ".phenos"
        if not os.path.isfile(phenoFile):
            # sys.stderr.write("Could not find phenotype file: %s\n" % (phenoFile))
            return
        f = open(phenoFile, 'r')
        keys = []
        P = []
        for line in f:
            v = line.strip().split()
            keys.append((v[0], v[1]))
            P.append([(x.strip() == 'NA' or x.strip() == '-9')
                     and np.nan or float(x) for x in v[2:]])
        f.close()
        P = np.array(P)

        # reorder to match self.indivs
        if not self.type == 'emma':
            D = {}
            L = []
            for i in range(len(keys)):
                D[keys[i]] = i
            for i in range(len(self.indivs)):
                if self.indivs[i] not in D:
                    continue
                L.append(D[self.indivs[i]])
            P = P[L, :]

        self.phenos = P
        return P

    def getIndivs(
        self,
        base: str,
        type: str = 'b',
    ) -> list[tuple[str, str]]:
        """Read individual IDs from the FAM or TFAM file.

        Parses the first two columns (`fam_id`, `indiv_id`) of the family
        file and stores them as a list of 2-tuples in `self.indivs`. Also
        sets `self.N` to the number of individuals found and writes a
        count message to stderr.

        Args:
            base:
                Base path for the PLINK fileset (same as `self.fbase`).
            type:
                Format indicator: `'t'` reads from `base + '.tfam'`, all
                other values read from `base + '.fam'`.

        Returns:
            List of `(fam_id, indiv_id)` tuples in file order.
        """
        if type == 't':
            famFile = "%s.tfam" % base
        else:
            famFile = "%s.fam" % base

        keys = []
        i = 0
        f = open(famFile, 'r')
        for line in f:
            v = line.strip().split()
            famId = v[0]
            indivId = v[1]
            k = (famId.strip(), indivId.strip())
            keys.append(k)
            i += 1
        f.close()

        self.N = len(keys)
        logger.debug("Read %d individuals from %s", self.N, famFile)

        return keys

    def readKinship(
        self,
        kFile: str,
    ) -> np.ndarray | None:
        """Load a kinship matrix from a file, reordering to match `self.indivs`.

        Supports two formats controlled by `self.fastLMM_kinship_style`:

        - **Plain** (default): whitespace-delimited square matrix, no header
          or ID columns. Rows assumed to be in the same order as `self.indivs`.
        - **fastLMM**: tab-delimited with a header row of `'fam_id indiv_id'`
          keys and a leading ID column on each data row. Rows/columns are
          reordered to match `self.indivs`; individuals absent from the
          kinship file are dropped and recorded in `self.indivs_removed`.

        Args:
            kFile:
                Path to the kinship matrix file (`.kin`).

        Returns:
            Kinship matrix as a numpy array of shape `(M, M)` where M is the
            number of individuals present in both datasets, or `None` if
            `self.indivs` is empty.
        """
        if self.indivs is None or len(self.indivs) == 0:
            logger.warning("No individuals loaded — cannot read kinship from %s", kFile)
            return

        logger.debug("Reading kinship matrix from %s", kFile)

        f = open(kFile, 'r')
        # read indivs
        if self.fastLMM_kinship_style:
            v = f.readline().strip().split("\t")[1:]
            keys = [tuple(y.split()) for y in v]
            D = {}
            for i in range(len(keys)):
                D[keys[i]] = i

        # read matrix
        K = []
        if self.fastLMM_kinship_style:
            for line in f:
                K.append([float(x) for x in line.strip().split("\t")[1:]])
        else:
            for line in f:
                K.append([float(x) for x in line.strip().split()])
        f.close()
        K = np.array(K)

        if self.fastLMM_kinship_style:
            # reorder to match self.indivs
            L = []
            KK = []
            X = []
            for i in range(len(self.indivs)):
                if self.indivs[i] not in D:
                    X.append(self.indivs[i])
                else:
                    KK.append(self.indivs[i])
                    L.append(D[self.indivs[i]])
            # np.ix_ allocates only the (M×M) result; K[L,:][:,L] allocates
            # an intermediate (M×N) copy first.  Tradeoff: ~2x more memory vs
            # ~4x faster for two-step, but this runs once per analysis so
            # the memory saving matters more than the speed difference.
            K = K[np.ix_(L, L)]
            self.indivs = KK
            self.indivs_removed = X
            if self.indivs_removed:
                logger.warning("Removed %d individuals not found in kinship file", len(self.indivs_removed))

        return K

    def getCovariatesEMMA(
        self,
        emmaFile: str,
    ) -> np.ndarray:
        """Load a covariate matrix in EMMA format.

        Reads a whitespace-delimited file where each row is one covariate
        and each column is one individual (transposed relative to the usual
        convention). Values of `'NA'` are converted to `np.nan`. The result
        is transposed before returning so that the output is `(N, num_covariates)`.

        Args:
            emmaFile:
                Path to the EMMA-format covariate file.

        Returns:
            Covariate matrix of shape `(N, num_covariates)`.
        """
        f = open(emmaFile, 'r')
        P = []
        for line in f:
            v = [x == 'NA' and np.nan or float(x)
                 for x in line.strip().split()]
            P.append(v)
        f.close()
        P = np.array(P).T
        return P

    def getCovariates(self, covFile=None):
        if not covFile:
            logger.debug("No covariate file provided")
            return
        if not os.path.isfile(covFile):
            logger.warning("Covariate file not found: %s", covFile)
            return
        f = open(covFile, 'r')
        keys = []
        P = []
        for line in f:
            v = line.strip().split()
            keys.append((v[0], v[1]))
            P.append([x == 'NA' and np.nan or float(x) for x in v[2:]])
        f.close()
        P = np.array(P)

        # reorder to match self.indivs
        D = {}
        L = []
        for i in range(len(keys)):
            D[keys[i]] = i
        for i in range(len(self.indivs)):
            if self.indivs[i] not in D:
                continue
            L.append(D[self.indivs[i]])
        P = P[L, :]

        return P


def load_snp_matrix(plink_data: "plink", num_snps: int | None = None) -> np.ndarray:
    """Load un-normalized SNPs into an (n_samples, n_snps) float64 array.

    Calls getSNPIterator() once to obtain numSNPs for pre-allocation, then
    iterates via the plink object to fill the matrix. Modifies
    plink_data.normGenotype in place (sets to False).

    Args:
        plink_data: An initialized plink object (BED, TPED, or EMMA format).
        num_snps:   Override numSNPs (only meaningful for EMMA format).

    Returns:
        float64 array of shape (n_samples, n_snps).
    """
    plink_data.normGenotype = False
    plink_data.getSNPIterator()
    if num_snps is not None:
        plink_data.numSNPs = num_snps

    n = len(plink_data.indivs)
    W = np.empty((n, plink_data.numSNPs), dtype=np.float64)
    j = 0
    for snp, _ in plink_data:
        # For EMMA, numSNPs is a caller-supplied count; stop once the
        # pre-allocated columns are full rather than overrunning W.
        if j >= plink_data.numSNPs:
            break
        W[:, j] = snp
        j += 1
    return W[:, :j]
