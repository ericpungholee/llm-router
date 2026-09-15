# Development

Install `requirements-dev.txt` in a virtual environment. The default dependency
set supports TF-IDF, handcrafted features, and the offline tests. The optional
encoder dependencies are in `experiments/requirements-router-v2.txt`.

Run from the repository root:

```bash
python tests/run_offline_suite.py
ruff check .
ruff format --check .
git diff --check
```

Use `ruff format .` for formatting and review `ruff check --fix .` changes before
committing. The Ruff configuration targets Python 3.9 compatibility. CI checks
Python 3.9 and 3.11 without downloading research datasets or model weights.
Artifact-dependent tests skip when the corresponding files are absent; the
complete local suite validates the saved experiments as well. Test calls to
provider adapters use mocks; actual HTTP, DNS, and socket access are blocked.

## Preserve research evidence

Do not edit frozen protocols, model-pool configs, source hashes, split manifests,
result reports, or pilot snapshots to make a new result pass. Create a separately
versioned experiment when changing the research question or selection rule.
Large generated artifacts belong under the ignored `data/` and `artifacts/`
directories; the tracked reports retain compact, reviewable results.

Some historical provider and grader source files have their exact bytes pinned
by `reports/snapshots/hard_pilot_4096_final.provenance.json`. Their explicit Ruff
format exclusions preserve those bytes. Correctness lint and tests still cover
them. Functional changes to those files require acknowledging a different pilot
configuration, not rewriting the old provenance. The formatter excludes the
entire provider-client directory for consistency with the archived harness.

Local routing loads trusted Joblib artifacts. Never load a model bundle from an
untrusted contributor just to inspect it. The CLI's network guard prevents
ordinary Python network calls; it is not an operating-system security sandbox.
Likewise, the historical generated-code grader requires a separate isolated
environment for untrusted programs. No code grading occurs during the offline
ML experiments, which consume saved success labels.

Keep credentials and local environment files out of Git. `.env.example` contains
names and placeholders only. The historical provider commands are documented
separately in [provider_harness.md](provider_harness.md).
