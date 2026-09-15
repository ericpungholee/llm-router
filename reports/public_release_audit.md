# Public-release audit

Reviewed the repository state immediately before its public-release cleanup on
2026-09-14. The full local suite passes **246 tests**, with **zero failures,
errors, skips, or network attempts**. A clean temporary checkout and freshly
installed development environment pass **219 tests**, with **27 explicit skips**
for unavailable research artifacts and the optional encoder. Both environments
use Python 3.9.6. [Machine-readable checks](public_release_audit.json).

## Changes

- Reorganized the README around the current offline router, results, limits, and
  reproducible setup. Moved the five-model paid-pilot documentation to an archival
  guide and separated the versioned experiment history.
- Made the default dependency install sufficient for the offline ML suite;
  optional encoder packages remain separate. Verified installation from scratch.
- Added pinned Ruff tooling and GitHub Actions checks for Python 3.9 and 3.11.
  Formatted active source and tests, sorted imports, removed unused names, and
  replaced ambiguous or unnecessarily compressed code.
- Fixed null required-field validation and non-string successful response fields.
  Local inference now rejects null, blank, non-string, or duplicate prompt IDs
  and duplicate columns. Malformed JSONL and missing files produce CLI errors.
  Nine regression tests cover these cases without needing model artifacts.
- Removed the unused binary RouterBench importer and its dependency file. It
  downloaded an unpinned remote pickle and was unrelated to the current dataset.
- Broadened ignore rules for local credentials, generated artifacts, and caches.
  Added third-party attribution and documented the trust requirements of Joblib
  loading and the historical subprocess grader. Original code remains unlicensed
  at the owner's request.

## Verification

Python syntax and JSON/JSONL parsing passed across the source tree. Local
Markdown file links resolve. Ruff lint, formatting, whitespace, and package
consistency checks pass. A pattern scan of the working files and all **233
reachable historical blobs** found no matching credential patterns. No generated
data, model weights, local environment files, or files over 5 MiB are included.
This pattern scan is not a guarantee that every possible secret is detectable.

The formatter preserves the exact source bytes pinned by the historical pilot;
those files still receive correctness lint and tests. Frozen benchmark samples,
configs, pilot snapshots, experiment specifications, and result reports retain
their hashes. Training and evaluation algorithms are unchanged; AST comparisons
were used to separate formatting/import changes from the reviewed validation,
CLI, unused-variable, and readability edits.

The V4 report was regenerated into a temporary directory. Its results JSON is
byte-identical to the published JSON. The Markdown differs only by the separately
appended historical 237-test footer. Frozen local inference also returns the
expected policy and label-free probabilities. The full artifact suite verifies
all saved experiment versions against their source outcomes and predictions.

No dataset preparation, model campaign, or paid evaluation was run during this
cleanup. GitHub CI uses no provider credentials and does not download datasets or
model weights; the hosted workflow provides its own result after the push.

```bash
python -m pip install -r requirements-dev.txt
python tests/run_offline_suite.py
ruff check .
ruff format --check .
python -m pip check
git diff --check
```
