#!/usr/bin/python

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

from argparse import ArgumentParser
from pylmm3 import input
from pylmm3.lmm import LMM
from scipy import linalg
import numpy as np
import os
import time
import sys

import logging

logger = logging.getLogger(__name__)

def printOutHead(out):
    out.write("\t".join(["SNP_ID", "BETA", "BETA_SD", "F_STAT", "P_VALUE"]) + "\n")


def outputResult(out, id, beta, betaSD, ts, ps):
    out.write("\t".join([str(x) for x in [id, beta, betaSD, ts, ps]]) + "\n")


def main():
    parser = ArgumentParser(
        usage="%(prog)s [options] --kfile kinshipFile --[tfile | bfile] plinkFileBase outfile",
        description=(
            "Basic genome-wide association (GWAS) using a linear mixed model. "
            "Provide a phenotype and genotype file plus a pre-computed kinship matrix "
            "(see pylmmKinship). Outputs a tab-separated result file with per-SNP statistics. "
            "Input files are standard PLINK format; phenotype file accepts NA or -9 for missing values."
        )
    )

    basicGroup = parser.add_argument_group("Basic Options")
    advancedGroup = parser.add_argument_group("Advanced Options")
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
        help="Drop individuals with missing genotypes instead of imputing with MAF. "
             "Triggers eigendecomposition recompute per SNP with missing values.")
    advancedGroup.add_argument(
        "--refit", action="store_true", dest="refit", default=False,
        help="Re-estimate variance components at each SNP (slower; more accurate for large-effect SNPs).")
    advancedGroup.add_argument(
        "--REML", action="store_true", dest="REML", default=False,
        help="Use restricted maximum-likelihood (default is maximum-likelihood).")
    advancedGroup.add_argument(
        "--eigen", dest="eigenfile",
        help="Base path for pre-computed eigendecomposition (<base>.Kva and <base>.Kve).")
    advancedGroup.add_argument(
        "--noMean", dest="noMean", default=False, action="store_true",
        help="Suppress automatic global mean covariate when --covfile is provided.")
    advancedGroup.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose", default=False,
        help="Print extra info to stderr.")

    experimentalGroup.add_argument(
        "--kfile2", dest="kfile2",
        help="Second kinship matrix for confounding correction (not implemented).")

    parser.add_argument("outfile", help="Output path for GWAS results.")

    options = parser.parse_args()
    outFile = options.outfile
    t_total = time.perf_counter()

    if not options.tfile and not options.bfile and not options.emmaFile:
        # if not options.pfile and not options.tfile and not options.bfile:
        parser.error(
            "You must provide at least one PLINK input file base (--tfile or --bfile) or an EMMA formatted file (--emmaSNP).")
    if not options.kfile:
        parser.error("Please provide a pre-computed kinship file")
    if options.kfile2:
        parser.error("--kfile2 is not implemented.")

    # READING PLINK input
    if options.verbose:
        sys.stderr.write("Reading SNP input...\n")
    if options.bfile:
        IN = input.plink(
            options.bfile,
            type='b',
            phenoFile=options.phenoFile,
            normGenotype=options.normalizeGenotype)
    elif options.tfile:
        IN = input.plink(
            options.tfile,
            type='t',
            phenoFile=options.phenoFile,
            normGenotype=options.normalizeGenotype)
    # elif options.pfile: IN = input.plink(options.pfile,type='p',
    # phenoFile=options.phenoFile,normGenotype=options.normalizeGenotype)
    elif options.emmaFile:
        IN = input.plink(
            options.emmaFile,
            type='emma',
            phenoFile=options.phenoFile,
            normGenotype=options.normalizeGenotype)
    else:
        parser.error("You must provide at least one PLINK input file base")

    if not os.path.isfile(
        options.phenoFile or IN.fbase +
        '.phenos') and not os.path.isfile(
            options.emmaPheno):
        parser.error(
            "No .pheno file exist for %s.  Please provide a phenotype file using the --phenofile or --emmaPHENO argument." %
            (options.phenoFile or IN.fbase + '.phenos'))

    # Read the emma phenotype file if provided.
    # Format should be rows are phenotypes and columns are individuals.
    if options.emmaPheno:
        f = open(options.emmaPheno, 'r')
        P = []
        for line in f:
            v = line.strip().split()
            p = []
            for x in v:
                try:
                    p.append(float(x))
                except BaseException:
                    p.append(np.nan)
            P.append(p)
        f.close()
        IN.phenos = np.array(P).T

    # READING Covariate File
    if options.covfile:
        if options.verbose:
            sys.stderr.write("Reading covariate file...\n")
        P = IN.getCovariates(options.covfile)
        if options.noMean:
            X0 = P
        else:
            X0 = np.hstack([np.ones((IN.phenos.shape[0], 1)), P])
    elif options.emmaCov:
        if options.verbose:
            sys.stderr.write("Reading covariate file...\n")
        P = IN.getCovariatesEMMA(options.emmaCov)
        if options.noMean:
            X0 = P
        else:
            X0 = np.hstack([np.ones((IN.phenos.shape[0], 1)), P])
    else:
        X0 = np.ones((IN.phenos.shape[0], 1))

    if np.isnan(X0).sum():
        parser.error("The covariate file %s contains missing values. At this time we are not dealing with this case.  Either remove those individuals with missing values or replace them in some way.")

    # READING Kinship
    if options.verbose:
        sys.stderr.write("Reading kinship...\n")
    t0 = time.perf_counter()
    K = np.loadtxt(options.kfile)
    if options.verbose:
        sys.stderr.write("Read %dx%d kinship in %.3fs\n" %
                         (K.shape[0], K.shape[1], time.perf_counter() - t0))

    # PROCESS the phenotype data -- Remove missing phenotype values
    # Keep will now index into the "full" data to select what we keep (either
    # everything or a subset of non missing data
    Y = IN.phenos[:, options.pheno]
    v = np.isnan(Y)
    keep = ~v
    if v.sum():
        if options.verbose:
            sys.stderr.write(
                "Cleaning the phenotype vector by removing %d individuals...\n" %
                (v.sum()))
        Y = Y[keep]
        X0 = X0[keep, :]
        K = K[keep, :][:, keep]
        Kva = []
        Kve = []

    # Only load the decomposition if we did not remove individuals.
    # Otherwise it would not be correct and we would have to compute it again.
    if not v.sum() and options.eigenfile:
        if options.verbose:
            sys.stderr.write("Loading pre-computed eigendecomposition...\n")
        Kva = np.load(options.eigenfile + ".Kva")
        Kve = np.load(options.eigenfile + ".Kve")
    else:
        Kva = []
        Kve = []

    # CREATE LMM object for association (includes eigendecomposition if not pre-loaded)
    n = K.shape[0]
    logger.debug(f"n: {n}")
    t0 = time.perf_counter()
    L = LMM(Y, K, Kva, Kve, X0, verbose=options.verbose)
    if options.verbose:
        sys.stderr.write("LMM setup (eigendecomposition) in %.3fs\n" % (time.perf_counter() - t0))

    # Fit the null model -- if refit is true we will refit for each SNP, so no
    # reason to run here
    if not options.refit:
        if options.verbose:
            sys.stderr.write("Fitting null model\n")
        t0 = time.perf_counter()
        L.fit()
        if options.verbose:
            sys.stderr.write("Null fit: heritability=%.3f, sigma=%.3f (%.3fs)\n" %
                             (L.optH, L.optSigma, time.perf_counter() - t0))

    # Buffers for pvalues and t-stats
    PS = []
    TS = []
    count = 0

    t_scan = time.perf_counter()
    with open(outFile, 'w') as out:
        printOutHead(out)

        for snp, id in IN:
            count += 1

            if options.verbose and count % 1000 == 0:
                elapsed = time.perf_counter() - t_scan
                sys.stderr.write("At SNP %d  (%.0f SNPs/s)\n" % (count, count / elapsed))

            x = snp[keep].reshape((n, 1))
            v = np.isnan(x).reshape((-1,))

            # Check SNPs for missing values
            if v.sum():
                keeps = ~v
                xs = x[keeps, :]
                if keeps.sum() <= 1 or xs.var() <= 1e-6:
                    PS.append(np.nan)
                    TS.append(np.nan)
                    outputResult(out, id, np.nan, np.nan, np.nan, np.nan)
                    continue

                # Its ok to center the genotype -  I used options.normalizeGenotype to
                # force the removal of missing genotypes as opposed to replacing them
                # with MAF.
                if not options.normalizeGenotype:
                    xs = (xs - xs.mean()) / np.sqrt(xs.var())
                Ys = Y[keeps]
                X0s = X0[keeps, :]
                Ks = K[keeps, :][:, keeps]
                Ls = LMM(Ys, Ks, X0=X0s, verbose=options.verbose)
                if options.refit:
                    Ls.fit(X=xs, REML=options.REML)
                else:
                    Ls.fit(REML=options.REML)
                ts, ps, beta, betaVar = Ls.association(
                    xs, REML=options.REML, returnBeta=True)
            else:
                if x.var() == 0:
                    PS.append(np.nan)
                    TS.append(np.nan)
                    outputResult(out, id, np.nan, np.nan, np.nan, np.nan)
                    continue

                if options.refit:
                    L.fit(X=x, REML=options.REML)
                ts, ps, beta, betaVar = L.association(
                    x, REML=options.REML, returnBeta=True)

            outputResult(out, id, beta, np.sqrt(betaVar).sum(), ts, ps)
            PS.append(ps)
            TS.append(ts)

    if options.verbose:
        elapsed = time.perf_counter() - t_scan
        sys.stderr.write("Scanned %d SNPs in %.3fs (%.0f SNPs/s)\n" %
                         (count, elapsed, count / elapsed))
        sys.stderr.write("Total: %.3fs\n" % (time.perf_counter() - t_total))


if __name__ == "__main__":
    main()