# Multi-provider ML-powered LLM Router

This repository builds deterministic evaluation data for a future learned LLM
router. It does **not** train or serve the router yet.

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
provider's error message; an invalid model remains disabled on resume until its
registry identifier changes or the failed row is deliberately removed after a
manual correction.

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

## Hard pilot (not run yet)

The fixed hard-pilot definition selects five prompts from each benchmark. A
future run will evaluate 15 prompts across five models for 75 prompt/model
pairs, writing to `data/results/hard_pilot_results.csv` so it cannot overwrite
the completed nine-prompt pilot.

```bash
python3 generate_dataset.py --pilot --max-spend-usd 1.00 --resume
```

Do not run it until its source summary and spend estimate have been reviewed.
Calls remain sequential, and the existing retry and spend controls apply.

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
```

Tests mock every provider adapter and cover mode cardinality, spend caps and
failed-attempt accounting, resume behavior, parsing failures, provider isolation,
exact model IDs, and retry ceilings. Neither command makes provider calls.

## Scope boundary

No ML router training, embedding generation, routing policy, web service,
frontend, or full benchmark sweep is part of this phase. Historical RouterBench
work remains under `experiments/` and `data/legacy/routerbench/` only.
