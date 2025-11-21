
## pylmm3 - A lightweight linear mixed-model solver (Python 3)

pylmm is a fast and lightweight linear mixed-model (LMM) solver for use in genome-wide association studies (GWAS).  This repository refactors the code to work with Python 3.

pylmm has a standalone program for running association studies called pylmmGWAS.  It can also be used as a python module to build your own custom programs.  If you want to create your own code, look at example.py for some basic usage patterns.  If you want to run basic GWAS analysis, the command below, which uses example data might be a helpful guide.

### An Example Command (GWAS):

```
python -m pylmm3.scripts.pylmmGWAS -v --bfile data/snps.132k.clean.noX --kfile data/snps.132k.clean.noX.pylmm.kin --phenofile data/snps.132k.clean.noX.fake.phenos out.foo
```

The GWAS program pylmmGWAS.py reads PLINK formated input files (BED or TPED only).  There is also an option to use "EMMA" formatted files.  We included this in order to make it easier for people using EMMA currently to try pylmm.  

### An Example Command (Kinship):

```
python -m pylmm3.scripts.pylmmKinship --bfile data/snps.132k.clean.noX kinship_matrix_output.txt
```

The kinship matrix file can be calculated using pylmmKinship.py which also takes PLINK or EMMA files as input.  The kinship matrix output is just a plain text file and follows the same format as that used by EMMA, so that you can use pre-computed kinship matrices from EMMA as well, or any other program for that matter.

## Installation

### Prerequisites

Ensure you have Python 3 installed on your system. pylmm is compatible with Python 3 and requires numpy and scipy.  Additionally, pylmm3 is available on <a href="https://test.pypi.org/project/pylmm3/">Pypi</a>. 

### Steps

1. **Clone the Repository**:
   Start by cloning the pylmm repository to your local machine using git:

    ```bash
    git clone git@bitbucket.org:jacksonlaboratory/pylmm3.git
    cd pylmm3
    ```

2.  **Install Poetry**
If you haven't already, install Poetry, a dependency management tool for Python:

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

2. **Create a Virtual Environment**:
   Create a Python virtual environment and install the dependencies using Poetry:

    ```bash
    poetry install
    ```

4. **Activate the virtual environment**

    ```bash
    poetry shell
    ```

pylmm is offered under the GNU Affero GPL (https://www.gnu.org/licenses/why-affero-gpl.html).




