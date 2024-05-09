
## pylmm - A lightweight linear mixed-model solver

pylmm is a fast and lightweight linear mixed-model (LMM) solver for use in genome-wide association studies (GWAS).

pylmm has a standalone program for running association studies called pylmmGWAS.  It can also be used as a python module to build your own custom programs.  If you want to create your own code, look at example.py for some basic usage patterns.  If you want to run basic GWAS analysis, the command below, which uses example data might be a helpful guide.

### An Example Command:

```
python pylmmGWAS.py -v --bfile data/snps.132k.clean.noX --kfile data/snps.132k.clean.noX.pylmm.kin --phenofile data/snps.132k.clean.noX.fake.phenos out.foo
```

The GWAS program pylmmGWAS.py reads PLINK formated input files (BED or TPED only).  There is also an option to use "EMMA" formatted files.  We included this in order to make it easier for people using EMMA currently to try pylmm.  The kinship matrix file can be calculated using pylmmKinship.py which also takes PLINK or EMMA files as input.  The kinship matrix output is just a plain text file and follows the same format as that used by EMMA, so that you can use pre-computed kinship matrices from EMMA as well, or any other program for that matter.

## Installation

### Prerequisites

Ensure you have Python 3 installed on your system. pylmm is compatible with Python 3 and requires numpy and scipy. 

### Steps

1. **Clone the Repository**:
   Start by cloning the pylmm repository to your local machine using git:

   ```bash
   git clone https://github.com/nickFurlotte/pylmm
   cd pylmm
   ```

2. **Create a Virtual Environment**:
   Create a Python virtual environment named pylmm-env for the pylmm project. This will help manage dependencies and isolate them from your global Python environment.

   ```bash
   python -m venv pylmm-env
   ```

   Activate the virtual environment:

   ```bash
   source pylmm-env/bin/activate
   ```

3. **Install the Requirements**:
   Install the required Python packages using pip. It is recommended to upgrade pip to the latest version before installing dependencies:
   ```bash
   pip install --upgrade pip; pip install -r requirements.txt
   ```
4. **Install pylmm**:
   ```bash
   pip install .
   ```



pylmm is offered under the GNU Affero GPL (https://www.gnu.org/licenses/why-affero-gpl.html).




