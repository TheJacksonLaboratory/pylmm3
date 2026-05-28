#!/usr/bin/python

import logging
import sys
import time

import numpy as np
from argparse import ArgumentParser
from scipy.linalg import eigh

from pylmm3 import input
from pylmm3.kinship import calculateKinship

logger = logging.getLogger(__name__)


def _load_snp_matrix(plink_data, num_snps: int = None) -> np.ndarray:
    """
    Load raw (un-normalized) SNPs from a plink input object into an
    (n_samples, n_snps) float64 array, ready to pass to calculateKinship.
    """
    plink_data.normGenotype = False
    plink_data.getSNPIterator()
    if num_snps is not None:
        plink_data.numSNPs = num_snps

    n = len(plink_data.indivs)
    W = np.empty((n, plink_data.numSNPs), dtype=np.float64)
    j = 0
    for snp, _ in plink_data:
        W[:, j] = snp
        j += 1
    return W[:, :j]


def main():
    parser = ArgumentParser(
        usage="%(prog)s [options] --[tfile | bfile] plinkFileBase outfile",
        description="Compute a kinship (realized relationship) matrix from PLINK genotype files."
    )

    basicGroup = parser.add_argument_group("Basic Options")

    basicGroup.add_argument("--tfile", dest="tfile",
                            help="The base for a PLINK tped file")
    basicGroup.add_argument("--bfile", dest="bfile",
                            help="The base for a PLINK binary ped file")
    basicGroup.add_argument(
        "--emmaSNP", dest="emmaFile", default=None,
        help="EMMA-format genotype file (individuals on rows, SNPs on columns).")
    basicGroup.add_argument(
        "--emmaNumSNPs", dest="numSNPs", type=int, default=0,
        help="Number of SNPs in the EMMA file (required with --emmaSNP).")
    basicGroup.add_argument(
        "-e", "--efile", dest="saveEig",
        help="Save eigendecomposition to <file>.kva and <file>.kve.")
    basicGroup.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose", default=False,
        help="Print extra info to stderr.")

    parser.add_argument("outfile", help="Output path for the kinship matrix.")

    options = parser.parse_args()
    outFile = options.outfile

    if options.verbose:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")

    t_total = time.perf_counter()

    if not options.tfile and not options.bfile and not options.emmaFile:
        parser.error(
            "You must provide at least one PLINK input file base (--tfile or --bfile) or an emma formatted file (--emmaSNP).")

    logger.info("Reading PLINK input...")
    if options.bfile:
        plink_data = input.plink(options.bfile, type='b')
    elif options.tfile:
        plink_data = input.plink(options.tfile, type='t')
    elif options.emmaFile:
        if not options.numSNPs:
            parser.error(
                "You must provide the number of SNPs when specifying an emma formatted file.")
        plink_data = input.plink(options.emmaFile, type='emma')
    else:
        parser.error(
            "You must provide at least one PLINK input file base (--tfile or --bfile) or an emma formatted file (--emmaSNP).")

    num_snps = options.numSNPs if options.emmaFile else None
    logger.info("Loading SNPs...")
    t0 = time.perf_counter()
    W = _load_snp_matrix(plink_data, num_snps=num_snps)
    logger.info("Loaded %d SNPs x %d individuals in %.3fs",
                W.shape[1], W.shape[0], time.perf_counter() - t0)

    logger.info("Computing kinship matrix...")
    t0 = time.perf_counter()
    K = calculateKinship(W)
    logger.info("Computed %dx%d kinship in %.3fs",
                K.shape[0], K.shape[1], time.perf_counter() - t0)

    logger.info("Saving kinship to %s", outFile)
    t0 = time.perf_counter()
    np.savetxt(outFile, K)
    logger.info("Saved in %.3fs", time.perf_counter() - t0)

    if options.saveEig:
        logger.info("Computing eigendecomposition...")
        t0 = time.perf_counter()
        Kva, Kve = eigh(K)
        logger.info("Eigendecomposition in %.3fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        np.savetxt(outFile + ".kva", Kva)
        np.savetxt(outFile + ".kve", Kve)
        logger.info("Saved eigendecomposition to %s.[kva|kve] in %.3fs",
                    outFile, time.perf_counter() - t0)

    logger.info("Total: %.3fs", time.perf_counter() - t_total)


if __name__ == "__main__":
    main()
