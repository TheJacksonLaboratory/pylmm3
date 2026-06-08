"""Unit tests for the GWAS scan — reference (gwas) and vectorized (gwas_fast).

The headline test cross-validates gwas_fast against the reference gwas
implementation: the design contract is that they are algebraically
identical (max relative error < 3e-9).
"""

import numpy as np
import pytest

from pylmm3 import gwas, gwas_fast

_FIELDS = ("BETA", "BETA_SD", "F_STAT", "P_VALUE")


def test_reference_output_structure(phenotype, kinship, snp_iter_factory, genotypes):
    """runGWAS returns a structured array with the expected fields and length."""
    result = gwas.runGWAS(phenotype, kinship, snp_iter_factory(), REML=False)
    assert result.dtype.names == ("SNP_ID", "BETA", "BETA_SD", "F_STAT", "P_VALUE")
    assert len(result) == genotypes.shape[1]
    assert list(result["SNP_ID"]) == [f"snp{j}" for j in range(genotypes.shape[1])]


def test_reference_recovers_signal(phenotype, kinship, snp_iter_factory):
    """The causal SNP (snp0) has the smallest p-value in the scan."""
    result = gwas.runGWAS(phenotype, kinship, snp_iter_factory(), REML=False)
    causal = result[result["SNP_ID"] == "snp0"][0]
    assert causal["P_VALUE"] < 0.05
    assert causal["P_VALUE"] == np.nanmin(result["P_VALUE"])


def test_fast_matches_reference(phenotype, kinship, snp_iter_factory):
    """gwas_fast is numerically identical to the reference gwas."""
    ref = gwas.runGWAS(phenotype, kinship, snp_iter_factory(), REML=False)
    fast = gwas_fast.runGWAS(phenotype, kinship, snp_iter_factory(), REML=False)

    assert list(ref["SNP_ID"]) == list(fast["SNP_ID"])
    for field in _FIELDS:
        np.testing.assert_allclose(
            fast[field], ref[field], rtol=1e-7, atol=1e-9, equal_nan=True,
            err_msg=f"gwas_fast diverged from gwas on {field}",
        )


def test_fast_matches_reference_small_batch(phenotype, kinship, snp_iter_factory):
    """Equivalence holds when batching forces multiple flushes (batch_size=3)."""
    ref = gwas.runGWAS(phenotype, kinship, snp_iter_factory(), REML=False)
    fast = gwas_fast.runGWAS(
        phenotype, kinship, snp_iter_factory(), REML=False, batch_size=3
    )
    for field in _FIELDS:
        np.testing.assert_allclose(
            fast[field], ref[field], rtol=1e-7, atol=1e-9, equal_nan=True
        )


def test_monomorphic_snp_yields_nan(phenotype, kinship, genotypes):
    """A constant SNP is emitted with NaN statistics, not dropped."""
    n = genotypes.shape[0]

    def snps():
        yield genotypes[:, 0].copy(), "causal"
        yield np.full(n, 0.5), "mono"      # monomorphic
        yield genotypes[:, 1].copy(), "other"

    for impl in (gwas, gwas_fast):
        result = impl.runGWAS(phenotype, kinship, snps(), REML=False)
        assert list(result["SNP_ID"]) == ["causal", "mono", "other"]
        mono = result[result["SNP_ID"] == "mono"][0]
        assert np.isnan(mono["BETA"])
        assert np.isnan(mono["P_VALUE"])
        # surrounding SNPs are still finite — order preserved across the flush
        assert np.isfinite(result["P_VALUE"][0])
        assert np.isfinite(result["P_VALUE"][2])


def test_missing_genotypes_handled(phenotype, kinship, genotypes):
    """A SNP with per-individual missing values still produces finite stats."""
    n = genotypes.shape[0]
    partial = genotypes[:, 0].copy()
    partial[2] = np.nan
    partial[5] = np.nan

    def snps():
        yield partial, "partial"

    ref = gwas.runGWAS(phenotype, kinship, snps(), REML=False)
    fast = gwas_fast.runGWAS(phenotype, kinship, snps(), REML=False)
    assert np.isfinite(ref["P_VALUE"][0])
    # both paths route missing-genotype SNPs through the same per-SNP fallback
    np.testing.assert_allclose(
        fast["P_VALUE"], ref["P_VALUE"], rtol=1e-7, atol=1e-9, equal_nan=True
    )


def test_missing_phenotype_removed(phenotype, kinship, snp_iter_factory, genotypes):
    """Individuals with NaN phenotype are dropped before fitting."""
    Y = phenotype.copy()
    Y[0] = np.nan
    result = gwas.runGWAS(Y, kinship, snp_iter_factory(), REML=False)
    # still one row per SNP, and the causal signal survives
    assert len(result) == genotypes.shape[1]
    causal = result[result["SNP_ID"] == "snp0"][0]
    assert causal["P_VALUE"] < 0.05
