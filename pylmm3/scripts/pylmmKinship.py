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

import os
import sys
import time

from pylmm3 import input
from pylmm3.lmm import calculateKinship

from scipy.linalg import eigh
import numpy as np

from argparse import ArgumentParser


def _load_snp_matrix(plink_data, num_snps: int = None) -> np.ndarray:
    """
    Load raw (un-normalized) SNPs from a plink input object into an
    (n_samples, n_snps) float64 array, ready to pass to calculateKinship.

    Arguments:
        plink_data: a plink input object, already initialized with the 
        appropriate file base and type.

        num_snps: required for emma format, which cannot infer SNP count 
        from the file header; ignored for bed/tped formats.

    Returns:
        raw (un-normalized) SNP matrix of shape (n_samples, n_snps)
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
    t_total = time.perf_counter()


    if not options.tfile and not options.bfile and not options.emmaFile:
        parser.error(
            "You must provide at least one PLINK input file base (--tfile or --bfile) or an emma formatted file (--emmaSNP).")

    if options.verbose:
        sys.stderr.write("Reading PLINK input...\n")
    if options.bfile:
        plink_data = input.plink(options.bfile, type='b')
    elif options.tfile:
        plink_data = input.plink(options.tfile, type='t')
    # elif options.pfile: plink_data = input.plink(options.pfile, type='p')
    elif options.emmaFile:
        if not options.numSNPs:
            parser.error(
                "You must provide the number of SNPs when specifying an emma formatted file.")
        plink_data = input.plink(options.emmaFile, type='emma')
    else:
        parser.error(
            "You must provide at least one PLINK input file base (--tfile or --bfile) or an emma formatted file (--emmaSNP).")

    num_snps = options.numSNPs if options.emmaFile else None
    if options.verbose:
        sys.stderr.write("Loading SNPs...\n")
    t0 = time.perf_counter()
    W = _load_snp_matrix(plink_data, num_snps=num_snps)
    if options.verbose:
        sys.stderr.write("Loaded %d SNPs x %d individuals in %.3fs\n" %
                         (W.shape[1], W.shape[0], time.perf_counter() - t0))

    if options.verbose:
        sys.stderr.write("Computing kinship matrix...\n")
    t0 = time.perf_counter()
    K = calculateKinship(W)
    if options.verbose:
        sys.stderr.write("Computed %dx%d kinship in %.3fs\n" %
                         (K.shape[0], K.shape[1], time.perf_counter() - t0))

    if options.verbose:
        sys.stderr.write("Saving kinship to %s\n" % outFile)
    t0 = time.perf_counter()
    np.savetxt(outFile, K)
    if options.verbose:
        sys.stderr.write("Saved in %.3fs\n" % (time.perf_counter() - t0))

    if options.saveEig:
        if options.verbose:
            sys.stderr.write("Computing eigendecomposition\n")
        t0 = time.perf_counter()
        Kva, Kve = eigh(K)
        if options.verbose:
            sys.stderr.write("Eigendecomposition in %.3fs\n" % (time.perf_counter() - t0))

        t0 = time.perf_counter()
        np.savetxt(outFile + ".kva", Kva)
        np.savetxt(outFile + ".kve", Kve)
        if options.verbose:
            sys.stderr.write("Saved eigendecomposition to %s.[kva|kve] in %.3fs\n" %
                             (outFile, time.perf_counter() - t0))

    if options.verbose:
        sys.stderr.write("Total: %.3fs\n" % (time.perf_counter() - t_total))


if __name__ == "__main__":
    main()
