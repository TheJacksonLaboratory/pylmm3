"""Shared fixtures for the pylmm3 test suite.

All data is generated deterministically (fixed seed) so tests are
self-contained — no external `.local/` fixture files are required.
"""

import numpy as np
import pytest

from pylmm3 import input as pin
from pylmm3.kinship import calculateKinship


@pytest.fixture
def rng():
    """A seeded RNG so every test sees identical synthetic data."""
    return np.random.default_rng(12345)


@pytest.fixture
def genotypes(rng):
    """Raw dosage matrix W of shape (n_samples, n_snps), values in {0, 0.5, 1}."""
    n, m = 50, 40
    return rng.integers(0, 3, size=(n, m)).astype(np.float64) / 2.0


@pytest.fixture
def kinship(genotypes):
    """Realized relationship matrix built from `genotypes`."""
    return calculateKinship(genotypes)


@pytest.fixture
def phenotype(genotypes, rng):
    """Phenotype with a real additive effect from the first SNP plus noise.

    The signal lets association tests assert that the causal SNP is
    recovered with a small p-value.
    """
    n = genotypes.shape[0]
    return genotypes[:, 0] * 2.0 + rng.standard_normal(n) * 0.5


@pytest.fixture
def snp_iter_factory(genotypes):
    """Return a callable producing a fresh (snp_vector, snp_id) iterator.

    A factory (rather than a single iterator) is needed because the
    reference and fast GWAS paths each consume the iterator once.
    """
    def _make():
        for j in range(genotypes.shape[1]):
            yield genotypes[:, j].copy(), f"snp{j}"
    return _make


@pytest.fixture
def bare_plink():
    """A `plink` instance with no open files, for testing pure helper methods.

    Built via `__new__` to skip file I/O in __init__; the pure decoding
    helpers (normalizeGenotype, getGenos_tped, formatBinaryGenotypes) only
    read `self.N`, which tests set as needed.
    """
    return pin.plink.__new__(pin.plink)


@pytest.fixture
def emma_file(tmp_path):
    """Write a 3-SNP x 4-individual EMMA genotype file; return its path."""
    path = tmp_path / "geno.emma"
    path.write_text("0 0.5 1 NA\n1 1 0 0\n0.5 0.5 0.5 0.5\n")
    return str(path)


@pytest.fixture
def tped_fileset(tmp_path):
    """Write a TPED fileset (.tped/.tfam/.map), 2 indivs x 2 SNPs; return base path."""
    base = tmp_path / "study"
    (tmp_path / "study.tfam").write_text("fam1 i1 0 0 0 -9\nfam1 i2 0 0 0 -9\n")
    (tmp_path / "study.map").write_text("1 rs1 0 100\n1 rs2 0 200\n")
    (tmp_path / "study.tped").write_text(
        "1 rs1 0 100 1 1 2 2\n"
        "1 rs2 0 200 1 2 0 0\n"
    )
    return str(base)


@pytest.fixture
def bed_fileset(tmp_path):
    """Write a BED fileset (.bed/.bim/.fam), 4 indivs x 2 SNPs; return base path."""
    base = tmp_path / "study"
    (tmp_path / "study.fam").write_text(
        "".join(f"fam i{i} 0 0 0 -9\n" for i in range(4))
    )
    (tmp_path / "study.bim").write_text("1 rs1 0 100 A G\n1 rs2 0 200 A G\n")
    with open(tmp_path / "study.bed", "wb") as f:
        f.write(bytes([0x6c, 0x1b, 0x01]))  # magic + SNP-major mode
        f.write(bytes([0b11100100]))         # rs1
        f.write(bytes([0b00011011]))         # rs2
    return str(base)
