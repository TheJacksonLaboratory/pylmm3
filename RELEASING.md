# Releasing pylmm3

pylmm3 is a public, AGPL-licensed package published to **public PyPI**
(https://pypi.org/project/pylmm3/). Releases are automated by
`.github/workflows/release.yml`.

## The model: a version bump is a release

Edit `version` in `pyproject.toml`, push to `main`, and the workflow:

1. runs the test suite,
2. `uv build`s the wheel + sdist,
3. publishes to PyPI (keyless, via Trusted Publishing),
4. tags the commit `v<version>` and pushes the tag.

Pushing to `main` **without** bumping the version is a no-op: the tag already
exists, so the workflow's guard skips the publish. Every published version maps to
exactly one git tag.

## One-time setup (PyPI project owner does this once)

### Trusted Publishing (recommended, keyless)

On https://pypi.org/manage/project/pylmm3/settings/ -> "Publishing", add a GitHub
Actions trusted publisher:

- Owner / repository: `TheJacksonLaboratory/pylmm3`
- Workflow name: `release.yml`
- Environment: leave blank (or set one and add `environment:` to the job for extra
  protection)

That's it. No secret is stored in GitHub; `release.yml` already requests the
`id-token: write` permission that OIDC needs.

### API-token fallback (only if Trusted Publishing isn't set up yet)

1. PyPI -> Account settings -> API tokens -> create a token scoped to `pylmm3`.
2. GitHub repo -> Settings -> Secrets and variables -> Actions -> add secret
   `PYPI_API_TOKEN` with that value.
3. In `release.yml`, replace the publish step with:

   ```yaml
   - name: Publish to PyPI (API token)
     if: steps.guard.outputs.release == 'true'
     env:
       UV_PUBLISH_TOKEN: ${{ secrets.PYPI_API_TOKEN }}
     run: uv publish
   ```

Prefer Trusted Publishing and delete the token once it's working — a stored token is
a long-lived credential someone has to rotate.

### Ownership

The project currently traces to the account that published 0.1.0/0.1.1 (Nick
Sebasco, June 2024). Make sure a JAX-controlled account is an **Owner** of the PyPI
project so publishing doesn't depend on one personal account.

## Cutting a release

1. Bump `version` in `pyproject.toml`:
   - Real release: plain semver, e.g. `0.2.3`.
   - Pre-release (PEP 440): `0.2.3a1`, `0.2.3b1`, `0.2.3rc1`, or `0.2.3.dev1`.
2. Commit and push to `main`.
3. Watch the Actions run: Test -> Build -> Publish -> Tag `v0.2.3`.

## Testing an unreleased pylmm3 against gwas-analysis-pylmm

Three options, fastest first:

- **Local editable checkout (no publish needed).** In `gwas-analysis-pylmm`'s
  `pyproject.toml`, temporarily point the source at your checkout:
  ```toml
  [tool.uv.sources]
  pylmm3 = { path = "../pylmm3", editable = true }
  ```
  Best inner loop while iterating on both at once.

- **Pre-release on PyPI.** Publish `0.2.3a1`. `pip`/`uv` ignore pre-releases unless
  asked, so gwas opts in with an explicit pin (`pylmm3==0.2.3a1`) or
  `uv pip install --prerelease=allow` / `pip install --pre`.

- **TestPyPI** (keeps real PyPI clean). Publish to https://test.pypi.org, then in gwas
  add it as an extra index for the test only:
  ```
  uv pip install --index-url https://pypi.org/simple \
                 --extra-index-url https://test.pypi.org/simple pylmm3
  ```

## How gwas-analysis-pylmm consumes pylmm3

Once a `0.2.x` (>= what gwas pins) is live on PyPI, gwas resolves pylmm3 as an ordinary
public dependency — no private index. See that repo's
`docs/artifact-registry-research.md` for the exact config change (and the sequencing:
make it only after pylmm3 is on PyPI, or gwas's `uv sync` will fail to resolve it).
