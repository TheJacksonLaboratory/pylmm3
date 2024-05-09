from distutils.core import setup

setup(
    name="pyLMM",
    url="https://github.com/nick-sebasco/pylmm3",
    description="pyLMM is a lightweight linear mixed model solver for use in GWAS.",
    packages=["pylmm"],
    scripts=["scripts/pylmmGWAS.py", "scripts/pylmmKinship.py"],
)
