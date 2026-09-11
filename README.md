# Multi-provider LLM Router

This project will build an ML router that selects a hosted LLM for each prompt
using expected quality, cost, and latency. Fresh evaluations of current models
will become the router's training data.

The intended model pool has one model reached through each provider:

- OpenAI API
- Anthropic API
- Google Gemini API
- DeepSeek API
- A remotely hosted open-weight model through Groq or a similar provider

No local GPU or local model serving is required. Exact model names will be
chosen before real data collection begins.

## Current phase

The repository currently contains only dataset-generation and evaluation
infrastructure. It does **not** train a router, create embeddings, issue real API
requests, run large benchmark sweeps, serve models, or include an API, UI, or
database.

The dry-run pipeline:

1. Loads three tiny JSONL benchmark examples.
2. Simulates responses from five model/provider slots.
3. Applies deterministic grading.
4. Normalizes quality, token, synthetic cost, and synthetic latency data.
5. Validates every required field and writes a CSV.

Dry-run cost and latency values are synthetic pipeline-test data, not provider
pricing or performance claims.

## Setup and dry run

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python generate_dataset.py --dry-run
```

This makes no API calls and writes:

```text
data/results/dry_run_results.csv
```

Live mode is deliberately disabled until exact models, prices, prompts, and
evaluation procedures are approved.

## Normalized evaluation schema

Each row represents one model evaluated on one prompt:

```text
prompt_id
prompt
task_type
benchmark_name
reference_answer
model_creator
model_name
inference_provider
model_type
score
correct
input_tokens
output_tokens
estimated_cost_usd
latency_ms
```

`model_type` is either `closed` or `open_weight`. The open-weight model is still
called through a hosted inference API.

## Benchmark input

Small benchmark inputs use JSON Lines, as shown in `benchmarks/sample.jsonl`.
Required fields are `prompt_id`, `prompt`, `task_type`, `benchmark_name`, and
`reference_answer`.

Supported task shapes are:

- `multiple_choice`, with a `choices` object
- `short_answer_math`
- `code`, with optional `tests` reserved for a later sandboxed test runner

Multiple choice and math are deterministically graded. The mock code path uses
exact matching only; it does not execute untrusted code. Real code evaluation is
intentionally deferred.

## Providers and secrets

Copy `.env.example` to `.env` when real integrations are implemented:

```bash
cp .env.example .env
```

Never commit `.env`. The shared provider interface is in `providers.py`; live
implementations are explicit stubs and validate the relevant environment key
before refusing the unimplemented call.

## Historical RouterBench experiment

The old RouterBench preparation work is retained only as a reproducible
historical baseline:

```text
experiments/legacy_routerbench.py
experiments/requirements-routerbench.txt
data/legacy/routerbench/          # generated files, ignored by Git
```

It is not an input to the new primary dataset pipeline.
