
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

Known issues (see docs/CURRENT_ISSUES.md):
    - CR-A: Dead branch in `normalizeGenotype` (`if s == 0` is unreachable).
    - CR-B: `except BaseException` in `__next__` should be `except Exception`.
    - File handles are stored as instance variables and closed in `__del__`,
      which is fragile. Prefer using the iterator in a `for` loop that exhausts
      it, or wrap the caller in a `with` block when possible.
"""

import logging
import os
import sys

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
        numSNPs: Total number of SNPs in the dataset (set by `getSNPIterator`).
        have_read: Running count of SNPs yielded so far.
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

        self.fhandle = None
        self.snpFileHandle = None

    def __del__(self) -> None:
        """Close open file handles when the object is garbage-collected."""
        if self.fhandle:
            self.fhandle.close()
        if self.snpFileHandle:
            self.snpFileHandle.close()

    def getSNPIterator(self) -> "plink":
        """Open the genotype file and return self as a ready iterator.

        Dispatches to the format-specific initializer based on `self.type`.
        After this call, `self.numSNPs` and `self.have_read` are set and
        iteration can begin via `next()` or a `for` loop.

        Returns:
            `self`, ready for iteration.
        """
        if self.type == 'b':
            return self.getSNPIterator_bed()
        elif self.type == 't':
            return self.getSNPIterator_tped()
        elif self.type == 'emma':
            return self.getSNPIterator_emma()
        else:
            sys.stderr.write("Please set type to either b or t\n")
            return

    def getSNPIterator_emma(self) -> "plink":
        """Open the EMMA-format SNP file and initialize iteration state.

        Returns:
            `self`, ready for iteration. `self.numSNPs` is set to -1 because
            EMMA files have no header with a SNP count; iteration stops when
            `readline()` returns an empty string.
        """
        self.have_read = 0
        self.numSNPs = -1
        file = self.fbase
        self.fhandle = open(file, 'r')

        return self

    def getSNPIterator_tped(self) -> "plink":
        """Open the TPED and BIM/MAP files and initialize iteration state.

        Counts SNPs by reading the `.bim` file (falling back to `.map` if
        `.bim` is absent), opens the `.tped` file for streaming, and resets
        the `have_read` counter.

        Returns:
            `self`, ready for iteration.
        """
        # get the number of snps
        file = self.fbase + '.bim'
        if not os.path.isfile(file):
            file = self.fbase + '.map'
        i = 0
        f = open(file, 'r')
        for line in f:
            i += 1
        f.close()
        self.numSNPs = i
        self.have_read = 0
        self.snpFileHandle = open(file, 'r')

        file = self.fbase + '.tped'
        self.fhandle = open(file, 'r')

        return self

    def getSNPIterator_bed(self) -> "plink":
        """Open the BED and BIM files, validate the magic bytes, and initialize iteration.

        Counts SNPs by reading the `.bim` file, opens the `.bed` file in
        binary mode, validates the two-byte PLINK magic number (`0x6c 0x1b`)
        and the SNP-major mode byte (`0x01`), then positions the file cursor
        at the first SNP record.

        Returns:
            `self`, ready for iteration.

        Raises:
            StopIteration: If the magic number is invalid or the file is not
                in SNP-major mode.
        """
        # get the number of snps
        file = self.fbase + '.bim'
        i = 0
        f = open(file, 'r')
        for line in f:
            i += 1
        f.close()
        self.numSNPs = i
        self.have_read = 0
        self.snpFileHandle = open(file, 'r')

        self.BytestoRead = self.N // 4 + (self.N % 4 and 1 or 0)
        logger.debug(f"[PLINK] Bytes to read: {self.BytestoRead}")
        self._formatStr = 'c' * self.BytestoRead

        file = self.fbase + '.bed'
        self.fhandle = open(file, 'rb')

        magicNumber = self.fhandle.read(2)
        order = self.fhandle.read(1)
        if magicNumber != b'\x6c\x1b':
            sys.stderr.write("Invalid PLINK BED magic number\n")
            raise StopIteration
        if order != b'\x01':
            sys.stderr.write("BED file is not in SNP-major mode\n")
            raise StopIteration

        return self

    def __iter__(self) -> "plink":
        """Return self as the iterator (calls `getSNPIterator` to open files)."""
        return self.getSNPIterator()

    def __next__(self) -> tuple[np.ndarray, str]:
        """Yield the next SNP as a `(genotype_array, snp_id)` pair.

        Dispatches to the format-specific read logic for BED, TPED, or EMMA.
        For EMMA format, non-numeric tokens are silently converted to `np.nan`
        (known issue CR-B: the `except BaseException` should be `except Exception`).

        Returns:
            A 2-tuple `(G, snp_id)` where `G` is a float64 array of length N
            with values in {0.0, 0.5, 1.0, np.nan}, and `snp_id` is the RS
            identifier string from the `.bim` / `.map` file, or `'SNP_<n>'`
            for EMMA format.

        Raises:
            StopIteration: When all SNPs have been yielded.
        """
        if self.have_read == self.numSNPs:
            raise StopIteration
        self.have_read += 1
        
        if self.type == 'b':
            X = self.fhandle.read(self.BytestoRead)
            # use the new formatBinaryGenotypes function that uses numpy for decoding
            res = self.formatBinaryGenotypes(X, self.normGenotype), self.snpFileHandle.readline().strip().split()[1]
            return res
        
        elif self.type == 't':
            X = self.fhandle.readline()
            XX = X.strip().split()
            chrm, rsid, pos1, pos2 = tuple(XX[:4])
            XX = XX[4:]
            G = self.getGenos_tped(XX)
            if self.normGenotype:
                G = self.normalizeGenotype(G)
            return G, self.snpFileHandle.readline().strip().split()[1]

        elif self.type == 'emma':
            X = self.fhandle.readline()
            if X == '':
                raise StopIteration
            XX = X.strip().split()
            G = []
            for x in XX:
                try:
                    G.append(float(x))
                except BaseException:
                    G.append(np.nan)
            G = np.array(G)
            if self.normGenotype:
                G = self.normalizeGenotype(G)
            return G, "SNP_%d" % self.have_read

        else:
            sys.stderr.write("Do not understand type %s\n" % (self.type))

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

        Known issue (CR-A): the `if s == 0:` branch is unreachable — s is
        either 1.0 (when var == 0) or sqrt(positive). See CURRENT_ISSUES.md.

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
        if G[x].var() == 0:
            s = 1.0
        else:
            s = np.sqrt(G[x].var())
        G[np.isnan(G)] = m
        if s == 0:
            G = G - m
        else:
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
        sys.stderr.write("Read %d individuals from %s\n" % (self.N, famFile))

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
            sys.stderr.write(
                "Did not read any individuals so can't load kinship\n")
            return

        sys.stderr.write("Reading kinship matrix from %s\n" % (kFile))

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
            K = K[L, :][:, L]
            self.indivs = KK
            self.indivs_removed = X
            if len(self.indivs_removed):
                sys.stderr.write(
                    "Removed %d individuals that did not appear in Kinship\n" %
                    (len(
                        self.indivs_removed)))

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

    def getCovariates(
        self,
        covFile: str | None = None,
    ) -> np.ndarray | None:
        """Load a covariate matrix and reorder rows to match `self.indivs`.

        The file is whitespace-delimited with columns
        `fam_id indiv_id cov1 [cov2 ...]`. Values of `'NA'` are converted to
        `np.nan`. Rows are reordered to match `self.indivs`; individuals in
        the covariate file but absent from the genotype dataset are dropped.

        Args:
            covFile:
                Path to the covariate file. If `None` or the file does not
                exist, a message is written to stderr and `None` is returned.

        Returns:
            Covariate matrix of shape `(N, num_covariates)`, or `None` if
            the file was not found.
        """
        if not os.path.isfile(covFile):
            sys.stderr.write("Could not find covariate file: %s\n" % (covFile))
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
