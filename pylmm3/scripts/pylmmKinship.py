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

from pylmm3 import input
from pylmm3.lmm import calculateKinship

from scipy.linalg import eigh
import numpy as np

from optparse import OptionParser, OptionGroup


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
    usage = """usage: %prog [options] --[tfile | bfile] plinkFileBase outfile"""

    parser = OptionParser(usage=usage)

    basicGroup = OptionGroup(parser, "Basic Options")
    # advancedGroup = OptionGroup(parser, "Advanced Options")

    # basicGroup.add_option("--pfile", dest="pfile",
    #                  help="The base for a PLINK ped file")
    basicGroup.add_option("--tfile", dest="tfile",
                        help="The base for a PLINK tped file")
    basicGroup.add_option("--bfile", dest="bfile",
                        help="The base for a PLINK binary ped file")
    basicGroup.add_option(
        "--emmaSNP",
        dest="emmaFile",
        default=None,
        help="For backwards compatibility with emma, we allow for \"EMMA\" file formats.  This is just a text file with individuals on the rows and snps on the columns.")
    basicGroup.add_option(
        "--emmaNumSNPs",
        dest="numSNPs",
        type="int",
        default=0,
        help="When providing the emmaSNP file you need to specify how many snps are in the file")

    basicGroup.add_option(
        "-e",
        "--efile",
        dest="saveEig",
        help="Save eigendecomposition to this file.")

    basicGroup.add_option("-v", "--verbose",
                        action="store_true", dest="verbose", default=False,
                        help="Print extra info")

    parser.add_option_group(basicGroup)
    # parser.add_option_group(advancedGroup)

    (options, args) = parser.parse_args()
    if len(args) != 1:
        parser.print_help()
        sys.exit()

    outFile = args[0]


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
    W = _load_snp_matrix(plink_data, num_snps=num_snps)
    if options.verbose:
        sys.stderr.write("Computing kinship matrix...\n")
    K = calculateKinship(W)
    if options.verbose:
        sys.stderr.write("Saving Kinship file to %s\n" % outFile)
    np.savetxt(outFile, K)

    if options.saveEig:
        if options.verbose:
            sys.stderr.write("Obtaining Eigendecomposition\n")
        Kva, Kve = eigh(K)
        if options.verbose:
            sys.stderr.write(
                "Saving eigendecomposition to %s.[kva | kve]\n" %
                outFile)
        np.savetxt(outFile + ".kva", Kva)
        np.savetxt(outFile + ".kve", Kve)


if __name__ == "__main__":
    main()
