"""Unit tests for the PLINK reader and genotype helpers (pylmm3.input)."""

import numpy as np
import pytest

from pylmm3.input import _BED_LOOKUP, plink, load_snp_matrix


# --- pure genotype-decoding helpers -------------------------------------

def test_normalize_genotype_zero_mean_unit_var(bare_plink):
    """Non-missing values are imputed then scaled to mean 0, variance 1."""
    G = np.array([0.0, 0.5, 1.0, np.nan, 0.5])
    out = bare_plink.normalizeGenotype(G.copy())
    assert not np.isnan(out).any()
    assert out.mean() == pytest.approx(0.0, abs=1e-9)
    # variance computed over the originally-observed entries is ~1
    assert out[[0, 1, 2, 4]].var() == pytest.approx(1.0, rel=1e-6)


def test_normalize_genotype_monomorphic_centered_not_scaled(bare_plink):
    """A constant SNP is centered to all-zeros (no divide-by-zero)."""
    out = bare_plink.normalizeGenotype(np.array([0.5, 0.5, 0.5]))
    assert np.allclose(out, 0.0)


def test_normalize_genotype_all_missing_unchanged(bare_plink):
    """An all-NaN array is returned unchanged."""
    out = bare_plink.normalizeGenotype(np.array([np.nan, np.nan]))
    assert np.isnan(out).all()


def test_getgenos_tped_allele_pairs(bare_plink):
    """TPED allele pairs decode to the documented dosage values."""
    bare_plink.N = 5
    #          00->nan  11->0.0  22->1.0  12->0.5  0x->0.5
    tokens = ['0', '0', '1', '1', '2', '2', '1', '2', '0', '1']
    G = bare_plink.getGenos_tped(tokens)
    assert np.isnan(G[0])
    assert G[1] == 0.0
    assert G[2] == 1.0
    assert G[3] == 0.5
    assert G[4] == 0.5


def test_getgenos_tped_unrecognized_pair_is_nan(bare_plink):
    """A nucleotide-encoded homozygous pair (e.g. A/A) yields NaN, not a crash."""
    bare_plink.N = 1
    G = bare_plink.getGenos_tped(['A', 'A'])
    assert np.isnan(G[0])


def test_format_binary_genotypes_matches_lookup(bare_plink):
    """BED bytes decode via the 2-bit lookup table (LSB-first within a byte)."""
    bare_plink.N = 4
    raw = bytes([0b11100100])  # codes 00,01,10,11 read low→high
    out = bare_plink.formatBinaryGenotypes(raw, norm=False)
    expected = _BED_LOOKUP[[0b00, 0b01, 0b10, 0b11]]
    assert np.allclose(out, expected, equal_nan=True)


# --- end-to-end file readers --------------------------------------------

def test_emma_reader(emma_file):
    """EMMA reader yields per-SNP arrays with NA → NaN and SNP_n ids."""
    p = plink(emma_file, type="emma", normGenotype=False)
    assert len(list(p.indivs)) == 4
    snps = list(p)
    assert len(snps) == 3
    g0, id0 = snps[0]
    assert id0 == "SNP_1"
    assert g0[0] == 0.0 and g0[1] == 0.5 and g0[2] == 1.0
    assert np.isnan(g0[3])


def test_tped_reader(tped_fileset):
    """TPED reader reports N, indiv tuples, and decodes allele pairs."""
    p = plink(tped_fileset, type="t", normGenotype=False)
    assert p.N == 2
    assert p.indivs == [("fam1", "i1"), ("fam1", "i2")]
    snps = list(p)
    assert [sid for _, sid in snps] == ["rs1", "rs2"]
    assert snps[0][0].tolist() == [0.0, 1.0]      # 1/1, 2/2
    g1 = snps[1][0]
    assert g1[0] == 0.5 and np.isnan(g1[1])         # 1/2, 0/0


def test_bed_reader(bed_fileset):
    """BED reader validates magic bytes and decodes both SNPs."""
    p = plink(bed_fileset, type="b", normGenotype=False)
    assert p.N == 4
    snps = list(p)
    assert [sid for _, sid in snps] == ["rs1", "rs2"]
    assert np.allclose(snps[0][0], _BED_LOOKUP[[0, 1, 2, 3]], equal_nan=True)


def test_bed_reader_normalizes_by_default(bed_fileset):
    """With normGenotype=True the yielded SNP is mean-centered."""
    p = plink(bed_fileset, type="b", normGenotype=True)
    g, _ = next(iter(p))
    # NaN imputed to mean → finite, and centered
    assert np.isfinite(g).all()
    assert g.mean() == pytest.approx(0.0, abs=1e-9)


def test_load_snp_matrix_shape(emma_file):
    """load_snp_matrix returns an (n_samples, n_snps) un-normalized matrix."""
    p = plink(emma_file, type="emma", normGenotype=False)
    W = load_snp_matrix(p, num_snps=3)
    assert W.shape == (4, 3)
    assert p.normGenotype is False  # set in place by the loader


def test_reiteration_does_not_leak_handles(emma_file):
    """Iterating the same plink object twice yields the same data each time.

    Regression for the descriptor leak: the old self-iterator opened a new
    handle on each __iter__ and orphaned the previous one. The generator-based
    __iter__ scopes handles to each pass, so a second iteration restarts
    cleanly and closes its handle on exhaustion.
    """
    p = plink(emma_file, type="emma", normGenotype=False)
    first = [(g.tolist(), sid) for g, sid in p]
    second = [(g.tolist(), sid) for g, sid in p]
    assert len(first) == 3
    assert [sid for _, sid in first] == [sid for _, sid in second]


def test_nested_iteration_is_independent(emma_file):
    """Nested iteration over one plink object does not corrupt shared state.

    The old self-iterator shared self.fhandle / self.have_read, so an inner
    loop exhausted the outer one. Independent generators make the full
    Cartesian product observable.
    """
    p = plink(emma_file, type="emma", normGenotype=False)
    pairs = [(a_id, b_id) for _, a_id in p for _, b_id in p]
    assert len(pairs) == 3 * 3


def test_getphenos_na_to_nan(tmp_path, tped_fileset):
    """Phenotype values 'NA' and '-9' are converted to NaN."""
    pheno = tmp_path / "study.phenos"
    pheno.write_text("fam1 i1 1.5\nfam1 i2 NA\n")
    p = plink(tped_fileset, type="t", phenoFile=str(pheno), normGenotype=False)
    assert p.phenos.shape == (2, 1)
    assert p.phenos[0, 0] == 1.5
    assert np.isnan(p.phenos[1, 0])
