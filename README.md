# pylmm3

**A fast, lightweight linear mixed-model solver for genome-wide association studies.**

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0-green)

pylmm3 is a Python 3 implementation of the EMMA/fastLMM linear mixed model (LMM)
framework. It corrects for population stratification and cryptic relatedness in GWAS
by modeling pairwise genetic similarity (kinship) as a random effect, then tests
each SNP for an additional fixed effect on top of that background.

Two command-line tools and a clean Python API:

- **`pylmmKinship`** — compute the realized relationship matrix (GRM) from PLINK genotype files
- **`pylmmGWAS`** — run the genome-wide association scan given a kinship matrix and phenotype file

---

## Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Running the Tools](#running-the-tools)
  - [Option 1 — `uv run` (no activation)](#option-1--uv-run-no-activation)
  - [Option 2 — Activate the virtual environment](#option-2--activate-the-virtual-environment)
- [Two-Stage Workflow](#two-stage-workflow)
- [Python API](#python-api)
- [CLI Reference](#cli-reference)
- [Logging](#logging)
- [Output Format](#output-format)
- [Testing](#testing)
- [How It Works](#how-it-works)
- [Known Limitations](#known-limitations)
- [Authors](#authors)
- [License](#license)

---

## Quick Start

```bash
git clone git@bitbucket.org:jacksonlaboratory/pylmm3.git
cd pylmm3
uv sync

# Stage 1 — build the kinship matrix
uv run pylmmKinship --bfile /path/to/study study.kin

# Stage 2 — run the GWAS
uv run pylmmGWAS --bfile /path/to/study --kfile study.kin --phenofile study.phenos results.tsv
```

---

## Installation

Choose the path that fits your situation. All three install [numpy](https://numpy.org)
and [scipy](https://scipy.org) automatically.

**Install [uv](https://docs.astral.sh/uv/) first** (if you don't have it):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

### Tier 1 — Just run the CLI tools (no clone, no project setup)

`uv tool install` puts `pylmmGWAS` and `pylmmKinship` on your PATH in an isolated
environment. This is the right path for anyone who just wants to run GWAS jobs.

```bash
uv tool install git+https://bitbucket.org/jacksonlaboratory/pylmm3.git

# Commands are now available globally:
pylmmKinship --bfile /data/study study.kin
pylmmGWAS    --bfile /data/study --kfile study.kin --phenofile study.phenos out.tsv
```

---

### Tier 2 — Use pylmm3 as a library inside your own project

```bash
# uv project
uv add git+https://bitbucket.org/jacksonlaboratory/pylmm3.git

# pip / any other tool
pip install git+https://bitbucket.org/jacksonlaboratory/pylmm3.git
```

Then import:

```python
from pylmm3 import LMM, calculateKinship, runGWAS
```

---

### Tier 3 — Local development (clone and edit source)

```bash
git clone git@bitbucket.org:jacksonlaboratory/pylmm3.git
cd pylmm3
uv sync          # creates .venv, installs all dependencies + pylmm3 in editable mode
```

`uv sync` reads `pyproject.toml` and `uv.lock` for a fully reproducible environment.
See [Running the Tools](#running-the-tools) below for how to invoke the CLIs after sync.

---

## Running the Tools

After `uv sync` there are two equally valid ways to run pylmm3.

### Option 1 — `uv run` (no activation)

`uv run` transparently invokes commands inside `.venv` without requiring you to
activate anything first. Works from any shell, any directory — no PATH changes needed.

```bash
# CLI entry points (registered in pyproject.toml)
uv run pylmmKinship --bfile /data/study study.kin
uv run pylmmGWAS   --bfile /data/study --kfile study.kin --phenofile study.phenos out.tsv

# Module invocation (identical result)
uv run python -m pylmm3.scripts.pylmmKinship --bfile /data/study study.kin
uv run python -m pylmm3.scripts.pylmmGWAS   --bfile /data/study --kfile study.kin out.tsv

# Interactive Python with pylmm3 available
uv run python
```

### Option 2 — Activate the virtual environment

Activate once per shell session; then call the tools directly.

```bash
source .venv/bin/activate

pylmmKinship --bfile /data/study study.kin
pylmmGWAS    --bfile /data/study --kfile study.kin --phenofile study.phenos out.tsv

python -c "from pylmm3 import LMM, calculateKinship, runGWAS; print('ready')"

deactivate   # when done
```

---

## Two-Stage Workflow

The kinship matrix is the bridge between the two tools. Once computed it can be
**reused** across multiple GWAS runs on the same cohort — testing different
phenotypes or covariate configurations without re-building kinship.

```
Genotype files (.bed / .bim / .fam)
Phenotype file  (.phenos)
         │
         ▼
  ┌─────────────────┐
  │  pylmmKinship   │  ← reads genotypes, normalizes SNPs, builds K = W·Wᵀ / m
  └─────────────────┘
         │
     study.kin              (reusable across phenotypes)
     study.kin.kva/.kve     (optional; skip O(n³) eigen on next run)
         │
         ▼
  ┌─────────────────┐
  │   pylmmGWAS     │  ← fits null LMM once, then tests each SNP
  └─────────────────┘
         │
         ▼
  results.tsv
  SNP_ID  BETA  BETA_SD  F_STAT  P_VALUE
```

### Typical run — with logging and saved eigendecomposition

```bash
# Build kinship, save eigenvectors for reuse (INFO shows milestones + timing)
uv run pylmmKinship \
  --bfile /data/study \
  --efile study.eigen \
  --log-level INFO \
  study.kin

# Run GWAS — load saved eigens, skip the O(n³) decomposition at startup
uv run pylmmGWAS \
  --bfile     /data/study \
  --kfile     study.kin \
  --eigen     study.eigen \
  --phenofile study.phenos \
  --log-level INFO \
  results.tsv
```

---

## Python API

The public API exports four symbols:

```python
from pylmm3 import LMM, calculateKinship, runGWAS, load_snp_matrix
```

### Build a kinship matrix

```python
import numpy as np
from pylmm3 import input as plink_input, calculateKinship

# Load genotypes from a PLINK binary fileset (.bed/.bim/.fam)
reader = plink_input.plink("study", type='b')
W = plink_input.load_snp_matrix(reader)   # (n_individuals, n_snps), np.nan for missing

K = calculateKinship(W)                   # (n, n) realized relationship matrix
np.savetxt("study.kin", K)                # plain-text format read by pylmmGWAS
```

### Run a GWAS scan

```python
from pylmm3 import runGWAS          # vectorized fast path (gwas_fast)
from pylmm3 import input as plink_input
import numpy as np

# Load data
reader = plink_input.plink("study", type='b', phenoFile="study.phenos")
Y = reader.phenos[:, 0]              # first phenotype column
K = np.loadtxt("study.kin")

# Fit null LMM once, scan all SNPs
results = runGWAS(Y, K, reader)      # numpy structured array

# Access results
import pandas as pd
df = pd.DataFrame(results)
print(df.sort_values("P_VALUE").head(10))
```

> **Reference vs fast path:** `from pylmm3 import runGWAS` is the vectorized
> implementation (`gwas_fast.py`), which is the same default used by the CLI.
> The reference per-SNP loop is available as `from pylmm3.gwas import runGWAS`
> and produces numerically identical results (max relative error < 3×10⁻⁹).

### Use the `LMM` class directly

```python
from pylmm3 import LMM
from pylmm3 import input as plink_input
import numpy as np

# Load genotypes and phenotypes via the PLINK reader
reader = plink_input.plink("study", type='b', phenoFile="study.phenos")
Y = reader.phenos[:, 0]              # first phenotype column

# Kinship is plain text written by pylmmKinship / np.savetxt
K = np.loadtxt("study.kin")

# Initialize — eigendecomposition computed here if Kva/Kve not provided
model = LMM(Y, K)

# Fit the null model
model.fit(REML=True)
print(f"Heritability: {model.optH:.3f}  σ²: {model.optSigma:.4f}")

# Test a single SNP (genotype vector, length N, values in {0.0, 0.5, 1.0, nan})
# Apply model.nonmissing mask — LMM removes individuals with missing phenotype,
# so the SNP vector must be subset to the same individuals before calling association().
snp, snp_id = next(iter(reader))
ts, ps = model.association(snp[model.nonmissing].reshape(-1, 1))
print(f"{snp_id}: t = {ts:.4f}  p = {ps:.2e}")
```

> **Note:** The `verbose` parameter on `LMM()` is accepted for backward compatibility but
> is ignored. Use `PYLMM3_LOG_LEVEL=DEBUG` or `--log-level DEBUG` to see internal detail.
> See [Logging](#logging) below.

---

## CLI Reference

### `pylmmKinship`

```
uv run pylmmKinship [options] --[bfile | tfile | emmaSNP] <base> <outfile>
```

| Flag | Required | Description |
|------|----------|-------------|
| `--bfile <base>` | one of three | Base path for PLINK binary files (`.bed` / `.bim` / `.fam`) |
| `--tfile <base>` | one of three | Base path for PLINK text files (`.tped` / `.tfam`) |
| `--emmaSNP <file>` | one of three | EMMA-format genotype file |
| `--emmaNumSNPs <n>` | with `--emmaSNP` | Number of SNPs in the EMMA file |
| `-e`, `--efile <base>` | no | Save eigendecomposition to `<base>.kva` and `<base>.kve` |
| `--log-level LEVEL` | no | Log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `WARNING`) |
| `-v`, `--verbose` | no | Shorthand for `--log-level INFO` |
| `<outfile>` | **yes** | Output path for the kinship matrix |

### `pylmmGWAS`

```
uv run pylmmGWAS [options] --kfile <kin> --[bfile | tfile | emmaSNP] <base> <outfile>
```

**Basic options**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--bfile <base>` | one of three | — | PLINK binary fileset base path |
| `--tfile <base>` | one of three | — | PLINK text fileset base path |
| `--emmaSNP <file>` | one of three | — | EMMA-format genotype file |
| `--emmaPHENO <file>` | no | — | EMMA-format phenotype file |
| `--emmaCOV <file>` | no | — | EMMA-format covariate file |
| `--kfile <file>` | **yes** | — | Pre-computed kinship matrix (`.kin` or `.kin.gz`) |
| `--phenofile <file>` | no | `<bfile>.phenos` | PLINK phenotype file |
| `-p <int>` | no | `0` | 0-indexed phenotype column (counting from column 3) |
| `--covfile <file>` | no | — | PLINK covariate file |
| `<outfile>` | **yes** | — | Output path for GWAS results |

**Advanced options**

| Flag | Default | Description |
|------|---------|-------------|
| `--eigen <base>` | — | Load pre-computed eigendecomposition (`<base>.kva` / `<base>.kve`); skips O(n³) decomp at startup |
| `--REML` | off | Use REML for per-SNP association tests (null model always uses REML regardless) |
| `--refit` | off | Re-estimate variance components at every SNP; more accurate, much slower |
| `--removeMissingGenotypes` | off | Drop individuals with missing genotypes per SNP instead of imputing with the mean |
| `--noMean` | off | Suppress automatic intercept when `--covfile` is provided |
| `--orig` | off | Use the reference per-SNP loop instead of the default vectorized scan |
| `--log-level LEVEL` | `WARNING` | Log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `-v`, `--verbose` | off | Shorthand for `--log-level INFO` |

---

## Logging

pylmm3 uses Python's standard `logging` module throughout. All loggers are named
`pylmm3.<module>` (e.g. `pylmm3.gwas_fast`, `pylmm3.lmm`) and propagate to the
root logger — pylmm3 never installs its own handler in library code.

### Log levels

| Level | Default? | What you see |
|-------|----------|--------------|
| `ERROR` | always | Unrecoverable failures — bad BED magic, unknown file type |
| `WARNING` | always | Dropped individuals, missing kinship entries, multiple optima found during heritability optimization |
| `INFO` | off | Pipeline milestones and timing — SNP load, kinship compute, null fit, total elapsed |
| `DEBUG` | off | Internal detail — BED bytes per SNP, eigendecomposition timing, per-SNP scan progress ticks |

The default level is **WARNING** (nearly silent). Production runs produce output only
when something is wrong.

### Controlling the level

**Via environment variable** — set before the process starts; no code changes needed:

```bash
PYLMM3_LOG_LEVEL=INFO  uv run pylmmGWAS --bfile study --kfile study.kin results.tsv
PYLMM3_LOG_LEVEL=DEBUG uv run pylmmGWAS --bfile study --kfile study.kin results.tsv
```

**Via CLI flag** — overrides the env var for that invocation:

```bash
uv run pylmmGWAS --log-level INFO  ... results.tsv   # milestones + timing
uv run pylmmGWAS --log-level DEBUG ... results.tsv   # full internal trace
uv run pylmmGWAS --verbose         ... results.tsv   # shorthand for INFO
```

**Priority:** `--log-level` flag > `PYLMM3_LOG_LEVEL` env var > `WARNING` default.

### Log format

```
[INFO   ] 2026-05-31 14:23:01.234  pylmm3.gwas_fast  Null fit: h=0.412  sigma=1.834  (2.341s)
[WARNING] 2026-05-31 14:23:01.235  pylmm3.lmm        Found 2 optima for h — returning first (h=0.4119)
[DEBUG  ] 2026-05-31 14:23:01.236  pylmm3.input      BED bytes per SNP: 83
```

All output goes to **stderr**. The `[LEVEL  ]` field is always 9 characters wide so
columns align across log lines.

### Using pylmm3 as a library

pylmm3 follows the standard library logging contract: it never calls
`logging.basicConfig()` or installs handlers. The calling application (your script,
a Temporal worker, etc.) is responsible for configuring the root handler. pylmm3
loggers propagate up normally.

When `PYLMM3_LOG_LEVEL` is set and no root handler exists yet, `configure()` in
`pylmm3.log` installs a minimal fallback handler so logs are not silently swallowed
in un-configured environments. This fallback is a no-op if a handler is already
present.

```python
# Your application configures logging once at startup — pylmm3 just propagates
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

from pylmm3 import runGWAS   # pylmm3 loggers now flow through your handler
```

### Why `%s`-style formatting in logger calls

All logger calls in pylmm3 use `%`-style arguments, not f-strings:

```python
# Correct — lazy: string is never formatted if DEBUG is disabled
logger.debug("BED bytes per SNP: %d", self.BytestoRead)

# Wrong — eager: f-string is always evaluated, even when DEBUG is off
logger.debug(f"BED bytes per SNP: {self.BytestoRead}")
```

Python's `logging` module defers the `%` substitution to `Formatter.format()`, which
is only called when a handler is actually going to emit the record. With f-strings the
interpolation happens at the call site before the level check — wasted work on every
disabled log call. In a tight loop over 250,000 SNPs, even trivial per-call overhead
adds up. The `%s` pattern is also the style recommended in the Python logging docs and
enforced by `pylint W1203`.

---

## Output Format

### Kinship matrix (`<outfile>`)

Space-delimited n×n matrix written by `numpy.savetxt`. No header. Row and column
order matches the FAM/TFAM file exactly. Values near 0 indicate unrelated pairs;
values near 1 indicate identical individuals.

### Eigendecomposition (`<base>.kva`, `<base>.kve`)

Written when `-e <base>` is passed to `pylmmKinship`. Plain-text files:
- `<base>.kva` — eigenvalues of K, length n
- `<base>.kve` — eigenvectors of K, n × n

Pass to `pylmmGWAS --eigen <base>` to skip the decomposition on subsequent runs.

### GWAS results (`<outfile>`)

Tab-separated. One header row, then one row per SNP. Monomorphic or all-missing
SNPs are written as `nan` across all value columns rather than being silently dropped.

| Column | Type | Description |
|--------|------|-------------|
| `SNP_ID` | string | RS identifier from the `.bim` file |
| `BETA` | float | Effect size estimate — phenotype change per unit dosage |
| `BETA_SD` | float | Standard error of the effect estimate |
| `F_STAT` | float | t-statistic for the SNP association test *(column name is a historical artifact — this is a t-statistic)* |
| `P_VALUE` | float | Two-tailed p-value from the t-distribution with n − q degrees of freedom |

---

## Testing

The test suite lives in [`tests/`](tests/) and uses [pytest](https://docs.pytest.org).
It is fully self-contained — synthetic data is generated from a fixed random seed
and temporary PLINK filesets are written per test, so **no external fixture files
are required**.

```bash
uv run pytest -q          # quiet — one line of summary
uv run pytest -v          # verbose — one line per test
```

Run a subset while developing:

```bash
uv run pytest tests/test_gwas.py                                  # one file
uv run pytest tests/test_lmm.py::test_tstat_matches_scipy         # one test
uv run pytest -k gwas                                             # by name pattern
uv run pytest -x                                                  # stop at first failure
```

### What the suite covers

| File | Module under test | Focus |
|------|-------------------|-------|
| `tests/test_lmm.py` | `pylmm3.lmm` | Missing-phenotype removal, eigendecomposition + clamping, `fit()`, single-SNP `association()`, `tstat` vs `scipy.stats`, and the REML log-likelihood staying finite when `det(XᵀX)` overflows (slogdet regression guard) |
| `tests/test_kinship.py` | `pylmm3.kinship` | Shape/symmetry, `center=True` → `trace(K) = n − 1`, invariant-SNP dropping, NaN imputation |
| `tests/test_input.py` | `pylmm3.input` | Genotype normalization, TPED/BED decoding, end-to-end EMMA/TPED/BED readers, phenotype `NA`/`-9` → `NaN` |
| `tests/test_gwas.py` | `pylmm3.gwas`, `pylmm3.gwas_fast` | Output structure, signal recovery, monomorphic/missing handling, and a cross-validation that the vectorized `gwas_fast` is numerically identical to the reference `gwas` |

Shared fixtures (seeded RNG, synthetic genotypes/kinship/phenotype, and temporary
PLINK filesets) live in [`tests/conftest.py`](tests/conftest.py).

---

## How It Works

### Kinship matrix

Each SNP is imputed (missing → column mean), standardized to zero mean and unit
variance, and invariant SNPs are dropped. The realized relationship matrix is:

```
K = W · Wᵀ / m
```

where W is the (n × m) matrix of standardized genotypes and m is the number of
valid SNPs retained. With `center=True`, K is further normalized so that
`trace(K) = n − 1` (EMMA-style).

### LMM and heritability estimation

The model is `Y = X·β + u + ε` where u ~ N(0, h·σ²·K) captures population
structure and ε ~ N(0, (1−h)·σ²·I) is the residual. Fitting requires optimizing
over the single free parameter h (heritability).

The key computational trick: K is decomposed once as K = V·Λ·Vᵀ via
`scipy.linalg.eigh`. Rotating Y and X into this eigenbasis diagonalizes the
covariance, reducing the O(n³) log-likelihood evaluation to O(n) per heritability
value. The profile likelihood is then maximized over a 100-point grid, with
[Brent's method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brent.html)
applied to refine each local maximum.

### Vectorized GWAS scan

The default scan (`gwas_fast.py`) batches fully-observed SNPs into blocks of 2000
and processes each block with a single BLAS `dgemm` call (`Kve.T @ G`), then
applies the Schur complement of the fixed covariate block — which is constant
across SNPs at the null-fit h — to compute per-SNP effect sizes and t-statistics
entirely in NumPy. This is algebraically identical to the reference per-SNP loop
in `gwas.py` (validated: max relative error < 3×10⁻⁹ across 231,164 SNPs) and
runs 50–200× faster in practice.

SNPs with missing genotypes or when `--refit` is active automatically fall back
to the per-SNP path.

---

## Known Limitations

| Limitation | Detail |
|------------|--------|
| **Memory** | K is an n×n float64 matrix where n is the number of **individuals** in the cohort. At n = 10,000 that is ~800 MB; at n = 100,000 it is ~80 GB. |
| **Single-threaded** | No parallelism across SNPs.  |
| **REML with many covariates** | `--REML` on the per-SNP path uses `linalg.det()`, which overflows when the covariate count q ≥ ~100. The null model uses `slogdet` and is safe. In practice GWAS runs use q = 2 (intercept + genotype) and are unaffected. |
| **`--removeMissingGenotypes` cost** | Dropping missing individuals triggers an O(n³) eigendecomposition recompute per affected SNP. Avoid this flag on cohorts with high missing-genotype rates. |
| **Covariate missing values** | The covariate file does not support missing values. Impute externally before passing to `pylmmGWAS`. |
| **`--kfile2`** | Accepted by the parser but immediately exits with an error — the two-kinship confounding path is not implemented. |

---

## Authors

pylmm3 is developed and maintained at [The Jackson Laboratory](https://www.jax.org):

| Name | Email |
|------|-------|
| **Matt Vincent** | [matt.vincent@jax.org](mailto:matt.vincent@jax.org) |
| **Nick Sebasco** | [nick.sebasco@jax.org](mailto:nick.sebasco@jax.org) |

pylmm3 is a Python 3 port of the original
[pylmm](https://github.com/nickFurlotte/pylmm) by
[Nicholas A. Furlotte](mailto:nick.furlotte@gmail.com).

---

## License

Copyright © 2015 Nicholas A. Furlotte  
Copyright © 2024–2026 The Jackson Laboratory

pylmm3 is free software licensed under the
[GNU Affero General Public License v3.0 or later](https://www.gnu.org/licenses/agpl-3.0.en.html).

> This program is distributed in the hope that it will be useful,
> but WITHOUT ANY WARRANTY; without even the implied warranty of
> MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
