#!/usr/bin/python

from argparse import ArgumentParser
import gzip
import logging
import os
import sys
import time

import numpy as np

from pylmm3 import input
from pylmm3.log import configure as configure_logging
from pylmm3.gwas_fast import runGWAS as runGWAS_fast
from pylmm3.gwas import runGWAS as runGWAS_original

logger = logging.getLogger(__name__)


def _write_results(out, results):
    out.write("\t".join(["SNP_ID", "BETA", "BETA_SD", "F_STAT", "P_VALUE"]) + "\n")
    for row in results:
        out.write("\t".join([str(x) for x in [
            row["SNP_ID"], row["BETA"], row["BETA_SD"], row["F_STAT"], row["P_VALUE"]
        ]]) + "\n")


def main():
    parser = ArgumentParser(
        usage="%(prog)s [options] --kfile kinshipFile --[tfile | bfile] plinkFileBase outfile",
        description=(
            "Basic genome-wide association (GWAS) using a linear mixed model. "
            "Provide a phenotype and genotype file plus a pre-computed kinship matrix "
            "(see pylmmKinship). Outputs a tab-separated result file with per-SNP statistics."
        )
    )

    basicGroup       = parser.add_argument_group("Basic Options")
    advancedGroup    = parser.add_argument_group("Advanced Options")
    experimentalGroup = parser.add_argument_group("Experimental Options")

    basicGroup.add_argument("--tfile", dest="tfile",
                            help="The base for a PLINK tped file")
    basicGroup.add_argument("--bfile", dest="bfile",
                            help="The base for a PLINK binary bed file")
    basicGroup.add_argument(
        "--phenofile", dest="phenoFile", default=None,
        help="Phenotype file in plink format. Defaults to <plinkFileBase>.phenos.")
    basicGroup.add_argument(
        "--emmaSNP", dest="emmaFile", default=None,
        help="EMMA-format genotype file (individuals on columns, SNPs on rows).")
    basicGroup.add_argument(
        "--emmaPHENO", dest="emmaPheno", default=None,
        help="EMMA-format phenotype file (one phenotype per row).")
    basicGroup.add_argument(
        "--emmaCOV", dest="emmaCov", default=None,
        help="EMMA-format covariate file (one covariate per row).")
    basicGroup.add_argument(
        "--kfile", dest="kfile",
        help="Pre-computed kinship matrix (nxn plain text, from pylmmKinship).")
    basicGroup.add_argument(
        "--covfile", dest="covfile",
        help="Covariate file in plink format.")
    basicGroup.add_argument(
        "-p", type=int, dest="pheno", default=0,
        help="0-indexed phenotype column to test (counting from column 3 of phenofile).")

    advancedGroup.add_argument(
        "--removeMissingGenotypes", action="store_false", dest="normalizeGenotype", default=True,
        help="Drop individuals with missing genotypes instead of imputing with MAF.")
    advancedGroup.add_argument(
        "--refit", action="store_true", dest="refit", default=False,
        help="Re-estimate variance components at each SNP.")
    advancedGroup.add_argument(
        "--REML", action="store_true", dest="REML", default=False,
        help="Use REML for the per-SNP association test (null model always uses REML).")
    advancedGroup.add_argument(
        "--eigen", dest="eigenfile",
        help="Base path for pre-computed eigendecomposition (<base>.kva and <base>.kve).")
    advancedGroup.add_argument(
        "--noMean", dest="noMean", default=False, action="store_true",
        help="Suppress automatic global mean covariate when --covfile is provided.")
    advancedGroup.add_argument(
        "--log-level", dest="log_level", default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity (default: WARNING, or PYLMM3_LOG_LEVEL env var).")
    advancedGroup.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose", default=False,
        help="Shorthand for --log-level INFO.")

    experimentalGroup.add_argument(
        "--kfile2", dest="kfile2",
        help="Second kinship matrix for confounding correction (not implemented).")
    experimentalGroup.add_argument(
        "--orig", action="store_true", dest="ORIG", default=False,
        help="Use the original GWAS implementation (default is fast implementation).")

    parser.add_argument("outfile", help="Output path for GWAS results.")

    options = parser.parse_args()
    outFile = options.outfile

    if options.log_level:
        configure_logging(getattr(logging, options.log_level))
    elif options.verbose:
        configure_logging(logging.INFO)
    else:
        configure_logging()

    t_total = time.perf_counter()

    if not options.tfile and not options.bfile and not options.emmaFile:
        parser.error(
            "You must provide at least one PLINK input file base (--tfile or --bfile) or an EMMA formatted file (--emmaSNP).")
    if not options.kfile:
        parser.error("Please provide a pre-computed kinship file")
    if options.kfile2:
        parser.error("--kfile2 is not implemented.")

    # Read PLINK input
    logger.info("Reading SNP input...")
    if options.bfile:
        plink_data = input.plink(
            options.bfile, type='b',
            phenoFile=options.phenoFile,
            normGenotype=options.normalizeGenotype)
    elif options.tfile:
        plink_data = input.plink(
            options.tfile, type='t',
            phenoFile=options.phenoFile,
            normGenotype=options.normalizeGenotype)
    elif options.emmaFile:
        plink_data = input.plink(
            options.emmaFile, type='emma',
            phenoFile=options.phenoFile,
            normGenotype=options.normalizeGenotype)
    else:
        parser.error("You must provide at least one PLINK input file base")

    if not os.path.isfile(
        options.phenoFile or plink_data.fbase + '.phenos') and not os.path.isfile(
            options.emmaPheno or ''):
        parser.error(
            "No .pheno file exists for %s. Provide one with --phenofile or --emmaPHENO." %
            (options.phenoFile or plink_data.fbase + '.phenos'))

    # Read EMMA phenotype file if provided
    if options.emmaPheno:
        P = []
        with open(options.emmaPheno, 'r') as f:
            for line in f:
                v = line.strip().split()
                p = []
                for x in v:
                    try:
                        p.append(float(x))
                    except Exception:
                        p.append(np.nan)
                P.append(p)
        plink_data.phenos = np.array(P).T

    # Read covariates
    if options.covfile:
        logger.info("Reading covariate file...")
        P = plink_data.getCovariates(options.covfile)
        X0 = P if options.noMean else np.hstack([np.ones((plink_data.phenos.shape[0], 1)), P])
    elif options.emmaCov:
        logger.info("Reading covariate file...")
        P = plink_data.getCovariatesEMMA(options.emmaCov)
        X0 = P if options.noMean else np.hstack([np.ones((plink_data.phenos.shape[0], 1)), P])
    else:
        X0 = np.ones((plink_data.phenos.shape[0], 1))

    if np.isnan(X0).sum():
        parser.error("The covariate file contains missing values. Remove or impute missing individuals before running.")

    # Read kinship
    logger.info("Reading kinship...")
    t0 = time.perf_counter()
    open_func = gzip.open if options.kfile.endswith('.gz') else open
    with open_func(options.kfile, 'rt') as f:
        # np.fromstring(str, sep=' ') was deprecated in NumPy 1.14 and removed
        # in NumPy 2.0. np.loadtxt() reads the same space-separated text format
        # and returns the (N, N) matrix directly (reshape below is a no-op).
        K = np.loadtxt(f)
    K = K.reshape((len(plink_data.indivs), len(plink_data.indivs)))
    logger.info("Read %dx%d kinship in %.3fs", K.shape[0], K.shape[1], time.perf_counter() - t0)

    # Load pre-computed eigendecomposition if provided
    Kva, Kve = None, None
    if options.eigenfile:
        logger.info("Loading pre-computed eigendecomposition...")
        Kva = np.loadtxt(options.eigenfile + ".kva")
        Kve = np.loadtxt(options.eigenfile + ".kve")

    # Phenotype vector — NaN removal is handled inside run_gwas
    Y = plink_data.phenos[:, options.pheno]

    # Run GWAS
    if options.ORIG:
        logger.info("Starting GWAS (original) scan...")
        results = runGWAS_original(
            Y, K, plink_data,
            X0=X0,
            Kva=Kva,
            Kve=Kve,
            refit=options.refit,
            REML=options.REML,
            normalizeGenotype=options.normalizeGenotype,
        )
    else:
        logger.info("Starting GWAS (fast) scan...")
        results = runGWAS_fast(
            Y, K, plink_data,
            X0=X0,
            Kva=Kva,
            Kve=Kve,
            refit=options.refit,
            REML=options.REML,
            normalizeGenotype=options.normalizeGenotype,
        )

    logger.info("Total: %.3fs", time.perf_counter() - t_total)

    with open(outFile, 'w') as out:
        _write_results(out, results)


if __name__ == "__main__":
    main()
