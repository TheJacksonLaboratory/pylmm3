from importlib.metadata import version, PackageNotFoundError

from pylmm3.lmm import LMM
from pylmm3.kinship import calculateKinship, NoVariantSNPsError
from pylmm3.gwas_fast import runGWAS
from pylmm3.input import load_snp_matrix

def get_version() -> str:
    """Return the installed version of pylmm3."""
    try:
        return version("pylmm3")
    except PackageNotFoundError:
        # Returns this if the package isn't installed in the environment yet
        return "unknown"

__version__ = get_version()

__all__ = [
    "LMM",
    "calculateKinship",
    "NoVariantSNPsError",
    "runGWAS",
    "load_snp_matrix",
    "get_version",
    "__version__"
]
