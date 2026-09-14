# Multi-provider ML-powered LLM Router

This repository trains and evaluates an offline learned LLM router from frozen
benchmark outcomes. The latest [fixed-reference experiment](reports/router_v4_results.md)
achieves **72.57% standard success versus GPT-5's 69.12%, at 12.57% lower recorded
cost**. It fixes GPT-5 as the quality reference and selects a comparative TF-IDF
routing margin using ordinary validation plus source domains excluded from
auxiliary training. All four declared standard criteria pass, including the
dataset quality guard. C=1; the standard margin is 0.10.

OOD quality is **86.35% versus GPT-5's 86.45%**, within the stated 0.5-point
noninferiority margin, with only **1.13% savings**. Useful cost-saving transfer to
code remains unproven. Standard savings are concentrated in SimpleQA; excluding
it leaves 0.28% savings. All results remain exploratory on previously inspected
tests, not fresh external confirmation. No provider calls or new data are used.

Reproduce v4 locally using the existing processed artifacts:

```bash
router_run_dir=$(mktemp -d /tmp/router-v4-replay.XXXXXX)
.venv/bin/python experiments/fixed_reference_router.py --output-dir "$router_run_dir" --reports-dir "$router_run_dir/reports"
.venv/bin/python tests/run_offline_suite.py
```

The [v4 protocol](reports/router_v4_experiment_spec.md) was fixed before fitting.
Models, predictions, source-domain memberships, validation bounds and bootstrap
samples are retained under the ignored `artifacts/router_v4/` directory.

The earlier [v2 experiment](reports/router_v2_results.md) achieved 73.10% standard
success with 22.07% savings, but its stricter dataset guard failed by one code
success. Its frozen embedding controls did not beat TF-IDF. V4 adds a stricter
source-domain validation requirement; all earlier reports remain unchanged.

The subsequent [fixed TF-IDF robustness check](reports/router_v3_results.md)
uses five grouped folds to evaluate every prompt once. Standard quality is
**71.94% versus GPT-5's 67.80%, at 21.20% lower recorded cost**, with positive
descriptive micro/macro intervals. Removing SimpleQA leaves only 1.34% savings.
The OOD check fails to provide routing value: all five non-code validation sets
select Qwen as the reference, so the rule routes every code prompt to Qwen
(64.17% quality versus fixed GPT-5's 86.45%). Source-domain reference selection
does not reliably protect unseen-domain quality. These reused-benchmark checks
are not independent confirmation and do not replace the saved v2 router.

Reproduce the fixed check locally, without downloads or new tuning:

```bash
router_check_dir=$(mktemp -d /tmp/router-v3-replay.XXXXXX)
.venv/bin/python experiments/crossfit_tfidf_router.py --output-dir "$router_check_dir" --reports-dir "$router_check_dir/reports"
```

The [v3 protocol](reports/router_v3_robustness_spec.md) fixes C=1 and the v2
comparative margins before fitting. Models, partitions, predictions, decisions
and bootstrap samples are kept under the ignored `artifacts/router_v3/`.

Route new prompt text locally, without calling a provider:

```bash
.venv/bin/python experiments/route_local.py --artifact-dir artifacts/router_v4/standard --prompt 'What is the capital of France?'
```

Reproduce the v2 campaign from a fresh artifact directory:

```bash
.venv/bin/python -m pip install -r experiments/requirements-router-v2.txt
.venv/bin/python -m routing_ml.embeddings --download
.venv/bin/python -m routing_ml.embeddings
.venv/bin/python experiments/embedding_logreg_router.py
.venv/bin/python tests/run_offline_suite.py
```

The download step fetches only pinned public encoder files; encoding, fitting,
selection and routing run locally. The [v2 protocol](reports/router_v2_experiment_spec.md)
and source snapshot are frozen before fitting. Completed runs under
`artifacts/router_v2/` are protected from overwrite. Regenerate their report with
`experiments/embedding_logreg_router.py --report-only`.

To replay fitting and evaluation now, reusing the verified local embedding cache
and preserving the completed artifacts:

```bash
router_replay_dir=$(mktemp -d /tmp/router-v2-replay.XXXXXX)
ln -s "$PWD/artifacts/router_v2/embeddings" "$router_replay_dir/embeddings"
.venv/bin/python experiments/embedding_logreg_router.py --output-dir "$router_replay_dir" --reports-dir "$router_replay_dir/reports"
```

The earlier [v1 results](reports/router_v1_results.md) remain unchanged: its
validation-selected standard policy saved 82.70% but lost 1.99 quality points
and failed noninferiority. V2 separates representation controls from a new
comparative routing/uncertainty protocol.

Reproduce training and evaluation using the existing processed artifacts:

```bash
.venv/bin/python -m pip install -r experiments/requirements-router-v1.txt
.venv/bin/python experiments/tfidf_logreg_router.py
.venv/bin/python tests/run_offline_suite.py
```

Training runs locally with networking blocked. It does not call model providers
or regenerate processed data. Reproducible models and probability/evaluation
tables are saved under the git-ignored `artifacts/router_v1/`; compact reports and
figures are under `reports/`. The command completes standard TF-IDF first, checks
a full deterministic training replay, then fits OOD independently and runs the
handcrafted-feature ablation in both regimes.

## Primary ML dataset: LLMRouterBench

The primary training source is now the official, hash-pinned LLMRouterBench
release: **8,706 complete prompts × 8 models = 69,648 outcomes**, covering math,
code, scientific reasoning, and knowledge QA. The provider harness and hard
pilot below remain external evaluation infrastructure.

Reproduce preparation from a fresh checkout; no API keys or provider calls:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r experiments/requirements-llmrouterbench.txt
.venv/bin/python scripts/prepare_llmrouterbench.py --download
.venv/bin/python tests/run_offline_suite.py
```

The first run downloads the public 1.28 GB JSON archive, verifies its SHA-256,
and writes canonical Parquet to `data/processed/llmrouterbench/`. Later runs
work offline. Raw and processed data are ignored by git; small frozen configs,
audit reports, and reproduction metadata are tracked.

Read the [source/schema audit](reports/llmrouterbench_data_audit.md),
[model coverage, splits, and baselines](reports/llmrouterbench_preparation.md),
and [frozen ML experiment specification](reports/ml_experiment_spec.md).
Standard train/validation/test sizes are 6,094/1,307/1,305. The independent OOD
regime holds out all 1,055 LiveCodeBench prompts. Labels, costs, and splits are
validated; unknown grades are never converted to failures.

The first experiment fits eight per-model TF-IDF + logistic-regression success
predictors, followed by validation-selected cost-aware decisions.
`routing_data.loading.load_split()` returns aligned prompts,
prompt-visible features, success targets, and evaluation costs separately.

## Closed 4096-token hard pilot

The completed 4096-token hard pilot is frozen separately from the earlier
historical snapshot. Reproduce its accounting, common-subset comparisons,
exhaustion diagnostics, and preregistered treatment plan entirely offline:

```bash
python3 reports/reproduce_4096_pilot.py --output-dir /tmp/hard-pilot-4096-report
python3 tests/run_offline_suite.py
```

See [the completed pilot report](reports/hard_pilot_4096_analysis.md) and
[provenance](reports/snapshots/hard_pilot_4096_final.provenance.json).
There are 45/75 grades and only five fully graded five-model prompts; this
experiment does not justify learned-router training or evaluation.
`hard_pilot_treatment.py` selects only the 20 final `output_limit_exhausted`
pairs, excludes Grok, preserves prompt/configuration/grading inputs, and allows
one attempt per pair with zero retries in a fresh output. The default action
is offline preflight; `--confirm` is a separate paid boundary.
Both candidate paid treatments currently fail strict preflight because two
new Qwen completions exceed their recorded request cap. Registry-contract
estimates ($1.656349 at 8192, $3.216105 at 16384) are conditional, not verified
worst-case billing bounds. The report documents the exact commands, the
8192 feasibility recommendation, and success criteria. No treatment has run.

## Earlier historical snapshot

The earlier stopped run's normalized, offline-regraded CSV remains at
`reports/snapshots/hard_pilot_final.csv`. Its provenance records stability,
before/after hashes, a regrade audit, and credential inspection. That historical
state has 24/75 grades and no five-model complete prompts; the newly closed
4096 experiment above contains the later completed paid resume.

Reproduce the historical reports with Python 3.9+ without `data/`, API keys,
or provider calls:

```bash
python3 pilot_analysis.py reports/snapshots/hard_pilot_final.csv
python3 reports/reproduce_hard_pilot.py
python3 -m unittest discover -s tests -v
```

The report script verifies the frozen input hash and regenerates the Markdown
and JSON reports. Keep the frozen CSV immutable; use a separate working CSV for
any later authorized resume. The report lists all incomplete pairs and the exact
preflight/resume commands; further calls are not part of this offline analysis.

## Enabled hosted models

| Creator | Model | Provider | Exact API identifier |
| --- | --- | --- | --- |
| OpenAI | GPT-5.6 Sol | OpenAI | `gpt-5.6-sol` |
| Anthropic | Claude Opus 5 | Anthropic | `claude-opus-5` |
| xAI | Grok 4.6 | xAI | `grok-4.6` |
| DeepSeek | DeepSeek V4.1 Flash | DeepSeek | `deepseek-flash` |
| Qwen / Alibaba | Qwen3.8 2.4T-A95B | OpenRouter | `qwen/qwen3.8-2.4t-a95b` |

`model_registry.py` is the central source of model metadata. Before any live
call, `providers.py` checks every provider/identifier pair against the exact IDs
implemented by the five adapters under `provider_clients/`. An adapter never
substitutes a fallback model. A provider rejection is saved and printed with the
provider's error message. Failure scope is part of the shared `ProviderError`
classification; evaluation does not reinterpret HTTP statuses.

DeepSeek uses peak, uncached pricing ($0.30/M input and $1.20/M output) for
conservative spend checks. Actual off-peak rates may be lower. OpenRouter's
provider-reported cost is retained when available.

## Setup

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Populate the five API keys in `.env`; never commit that file. No mode is selected
by default, so invoking `generate_dataset.py` without an explicit mode cannot
make a live call.

## Fully offline dry run

```bash
python generate_dataset.py --dry-run
```

Dry-run uses deterministic local responses and never accesses a provider. It
evaluates the local 150-prompt candidate pool and writes synthetic metrics to
`data/results/dry_run_results.csv`.

## Hard candidate pool

The active manifest contains 150 held-out candidates: 50 domain-balanced
MMLU-Pro test questions, 50 exact-gradeable level-5 MATH-500 test problems, and
50 recent medium/hard LiveCodeBench competition problems from the
`release_v6/test6` window. Coding records are genuine AtCoder stdin problems
dated February-April 2025; functional-interface records and simple handwritten
fixtures are excluded.

Selection uses source metadata and deterministic heuristics only, never model
outcomes. `benchmarks/hard_pilot.json` identifies a future 15-prompt pilot with
five prompts per benchmark. Inspect it without making provider calls:

```bash
python3 benchmark_pool_summary.py
```

## Five-call smoke test

```bash
python generate_dataset.py --smoke-test --max-spend-usd 0.25
```

The smoke test uses one arithmetic multiple-choice prompt, five enabled models,
sequential dispatch, a 128-token output bound, no retries, and at most five HTTP
inference attempts. It prints the exact model before each attempt, prints
accumulated spend after each success, and atomically rewrites
`data/results/smoke_test_results.csv` after every outcome. Provider failures are
recorded and do not stop the other providers.

Live outputs are resumed automatically when the output file already exists, so
paid terminal prompt/model pairs are never called twice. Use a different
`--output` path only when an intentionally separate run is desired.

## Hard pilot and resume

The fixed hard-pilot definition selects five prompts from each benchmark. The
run evaluates 15 prompts across five models for 75 unique prompt/model
pairs, writing to `data/results/hard_pilot_results.csv` so it cannot overwrite
the completed nine-prompt pilot.

```bash
python3 generate_dataset.py --hard-pilot --resume --exclude-xai \
  --max-spend-usd 2.00 --output data/results/hard_pilot_results.csv
```

This command prints preflight and exits before dispatch because `--confirm` is
absent. xAI is currently blocked by spend preflight and the runtime guard: its
historical usage exceeds the documented shared answer/reasoning limit, so its
pre-dispatch cost bound remains unresolved. `--exclude-xai` defers dispatch only;
the registry and 75-pair experiment matrix retain all five models, every paid
response, and all historical recorded spend. See the
[xAI accounting audit](reports/xai_cost_safety.md) for official semantics and the
remaining historical billing uncertainty.

Pass `--confirm` only after reviewing preflight and reconciling historical
billing. Calls remain sequential,
with 75 unique pairs, up to two retries per pair for transient failures, at most
225 provider attempts in the worst case, and a maximum cap of $2.00.

The current ledger omits a previously reported Grok dispatch interrupted before
checkpointing. Obtain its charge or a verified cumulative pilot total from an
existing billing record before paid resume; do not estimate it from 4,096 tokens.
If verified prior spend exceeds the recorded ledger, reduce `--max-spend-usd`
by that difference so the actual cumulative ceiling remains $2.00. Historical
paid rows need no guessed cost edits. Once reconciled, authorize the other four
providers by adding `--confirm` with the reconciled cap and retaining
`--exclude-xai`. Finishing all remaining pairs within the cap is not guaranteed.

Preflight reports remaining pairs, up to three provider attempts per pending
pair, previously recorded spend, conservative maximum additional/total spend,
and the configured cap. Paid terminal pairs are excluded from the pending retry
budget. The cap includes previous runs' recorded costs; resume does not reset it.
If the conservative total exceeds $2, preflight warns that completion is not
guaranteed. The spend guard checks every attempt, including retries, and stops
before dispatch if its bound could exceed the cap.

Failure and resume rules:

* Retryable errors (network/timeouts, ordinary 429s, 5xx) use the configured retry
  policy. Exhausted failures affect only that pair, never later prompts/models.
* Empty/invalid provider responses and unrecognized rejections stay local.
  Invalid parameters stay local because they may be prompt-specific. Explicit
  `configuration_error` and invalid-model errors block the exact provider/model.
* Authentication errors block that provider for the current run. Explicit empty
  account balance/billing exhaustion blocks that provider; zero model/resource
  quota and request affordability errors block only the affected model. No
  provider error stops unrelated providers or the whole run.
* Resume preserves every paid response (`success`, `parsing_failure`, or
  `grading_failure`), including incorrect answers. Reparse/regrade saved raw
  responses offline rather than calling those pairs again.
* Pair failures and skipped rows are retried on resume. Auth/quota/billing blocks
  are rechecked each run, allowing credentials/account state to be repaired.
  Invalid-model/configuration blocks persist only for matching provider, exact
  model ID, temperature, reasoning settings, and output limit. Changing the
  relevant request configuration releases the block; old CSVs use their saved
  settings. Completed paid responses remain terminal even after settings change.
  Corrected registry IDs/providers move pending unpaid rows for the same canonical
  model to the new identity, retaining prior spend/attempts. Paid rows retain their
  original identity; a CSV with paid models outside the new plan is rejected.
* Explicit `output_limit_exhausted` outcomes block only the same pair under its
  saved configuration. Historical blank responses without that evidence remain
  retryable. xAI spend exclusion is separate from these provider retry states.

Transient failures and invalid provider responses reserve their maximum possible
cost because inference may have been billed without returned usage. Reserves and
attempt counts are checkpointed before retrying and retained when a failed pair
is replaced on resume. Historical CSVs remain readable; their attempt counts are
inferred from status/retry telemetry where necessary. No historical costs or raw
responses are rewritten merely by preflight or analysis.

xAI successful responses use `usage.cost_in_usd_ticks / 10_000_000_000` as their
provider-reported billed cost when available. Safe diagnostics retain integer
ticks and token telemetry, without reasoning content. A successful answer with
output usage above the requested limit is retained and flagged. Missing ticks
leave the existing local estimate as an estimate; historical bills are never
fabricated. Reported billing arrives after dispatch and does not establish a
pre-dispatch bound.

The previous 45-row pilot remains at `data/results/pilot_results.csv`. Regrade
that historical artifact entirely offline with:

```bash
python3 generate_dataset.py --regrade-pilot
```

This reparses the preserved raw responses and atomically updates only their
grading fields. It does not load API keys or call a provider.

The default live cap is $1.00. `--max-spend-usd` cannot exceed
`GLOBAL_MAX_SPEND_USD`. Immediately before every attempt, the remaining cap is
checked against the prompt plus that mode's maximum output tokens. A run stops
before dispatch if the bound could exceed its cap.

## Parsing, grading, and result records

Raw provider text and parsed answers are separate fields. Multiple choice is
normalized to one option, math uses deterministic boxed/final-answer extraction,
and code is extracted from a Python fence (or a code-shaped raw response) before
tests. Parsing failures have null `score` and `correct` values and a
`parsing_failure` status; they are never labeled as wrong answers.

The code grader allows an explicit competitive-programming standard-library
import set, rejects other imports, dangerous built-ins, and dunder access, then
runs the accepted restricted Python subset with `-I -S` and per-test timeouts.
Rejected or malformed programs are grading failures rather than wrong answers.

Each row contains prompt and benchmark metadata; creator, canonical model, exact
API ID, provider, and model type; raw and parsed outputs; score and correctness;
token counts, cost, and latency; generation/reasoning settings and timestamp;
provider request/response metadata; retry count; and structured status/error
fields. Writes use a temporary file plus atomic replacement for crash-resistant
incremental persistence.

## Tests and read-only summary

```bash
python -m unittest discover -s tests -v
python experiment_summary.py
python3 pilot_analysis.py data/results/hard_pilot_results.csv
```

Tests mock every provider adapter and cover mode cardinality, spend caps and
failed-attempt accounting, resume behavior, parsing failures, provider isolation,
exact model IDs, and retry ceilings. Neither command makes provider calls.

`pilot_analysis.py` emits deterministic JSON: per-model calls, responses, graded
responses, correct responses, accuracy over graded responses, distinct failure
counts, skips, missing pairs, recorded cost, and average latency of returned
responses. Attempt/response counts and recorded costs are cumulative across
resume; failure category counts describe the current pair outcomes. Spend stops
are reported separately from provider failures. It also reports benchmark/model
accuracy, cheaper-model wins, prompts solved only by higher-cost models, oracle
accuracy, best-single-model and cheapest-model baselines, and cheapest-correct
oracle cost. The default plan is the fixed 15 x 5 matrix, including wholly absent
prompts/models. Use `--observed-plan` for historical/synthetic CSVs; that option
cannot detect wholly absent prompts/models.

Policy comparisons share the fully graded prompt subset, excluding all missing,
skipped, provider, parsing, and grading failures. Model cost order uses average
observed response cost on that subset; higher cost is a proxy, not proof of
strength. Oracle selection uses each prompt's cheapest correct response cost
(excluding failed-attempt reserves), with stable provider/model tie breaks.
Oracle cost covers solved prompts only; unsolved prompts are listed explicitly.
Legacy responses with retries cannot provide exact separate response costs, so
cost policy analysis is withheld for those comparisons. Completeness is reported
as graded pairs / 75; only an entirely graded matrix is labeled complete/valid.

Reparse/regrade the hard pilot without paying for another response:

```bash
python3 generate_dataset.py --regrade-results --output data/results/hard_pilot_results.csv
```

This preserves provider failures/skips and all paid raw outputs and telemetry.
Persistent parsing/grading failures still leave the matrix incomplete; inspect
their saved responses and grader diagnostics rather than replacing null grades
with incorrect answers.

Before ML work, inspect completeness and failure categories first, then compare
benchmark accuracies, prompt-level disagreements, oracle lift over the best
single model, and accuracy/cost relative to the cheapest baseline. These are
descriptive pilot results, not a statistical guarantee of generalization.

## Scope boundary

Router v1 covers TF-IDF and the 16-feature logistic-regression ablation. V2 adds
a frozen local encoder comparison and conservative comparative routing, with
offline inference of model choices. Neural routers, gradient boosting, paid
evaluation, provider integration for serving, and a frontend remain future work.
Historical RouterBench work remains under `experiments/` and
`data/legacy/routerbench/` only.
