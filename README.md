# Multi-provider ML-powered LLM Router

The final project will predict which hosted LLM should handle a prompt using
expected quality, cost, and latency. Fresh, per-prompt evaluations will become
the router's training data. No local GPU or local model serving is required.

This repository is currently at **Phase 2: experiment design and live-evaluation
readiness**. Live evaluation has not started. Provider implementations remain
stubs; any future live run is guarded by the spend controls below.

## Six-model comparison

All model metadata is centralized in `model_registry.py`.

| Creator | Canonical model | Inference provider | Type | API identifier |
| --- | --- | --- | --- | --- |
| OpenAI | GPT-5.6 Sol | OpenAI | closed | `gpt-5.6-sol` |
| Anthropic | Claude Opus 5 | Anthropic | closed | `claude-opus-5` |
| Google | Gemini 3.1 Pro | Google | closed | `gemini-3.1-pro-preview` |
| xAI | Grok 4.6 | xAI | closed | `grok-4.6` |
| DeepSeek | DeepSeek V4.1 Flash | DeepSeek | open_weight | `deepseek-flash` |
| Qwen / Alibaba | Qwen3.8 2.4T-A95B | OpenRouter | open_weight | `qwen/qwen3.8-2.4t-a95b` |

Creator and inference provider are deliberately separate. Qwen / Alibaba is
the model creator; OpenRouter is the inference provider for this evaluation.

The registry also holds enabled status, temperature handling, reasoning/thinking
settings, maximum output tokens, token prices, and uncertainty notes. The newly
changed DeepSeek V4.1 pricing remains explicitly unresolved and prevents that
entry from becoming live-ready.

Verified documentation used for the initial registry:

- [OpenAI GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
- [Anthropic Claude Opus 5](https://platform.claude.com/docs/en/models/opus-5/whats-new-opus-5)
- [Google Gemini 3](https://ai.google.dev/gemini-api/docs/gemini-3)
- [xAI Grok 4.6](https://docs.x.ai/developers/models/grok-4.6)
- [DeepSeek V4.1 Flash](https://deepseek.com/en/news/deepseek-v4-1-flash/)
- [Qwen3.8 2.4T-A95B on OpenRouter](https://openrouter.ai/qwen/qwen3.8-2.4t-a95b)

## Benchmark design

`benchmarks/manifest.json` declares enabled benchmarks, task types, source
repositories, pinned source revisions, loader names, and local sample files.
Phase 2 contains one verified source-shaped fixture from each benchmark:

- MMLU-Pro: hard multiple-choice reasoning and knowledge
- MATH-500: math problems with known answers
- LiveCodeBench: modern code generation with executable stdin/stdout tests

Source adapters live in `benchmark_loaders.py`; no full benchmark is downloaded.
Deterministic graders live separately in `graders.py`:

- Multiple choice requires exactly one normalized valid option.
- Math uses conservative numeric/fraction comparison, then normalized exact text.
- Code goes through a test-runner interface. Only committed mock code is executed
  by the local dry-run runner; live model code will require a real sandbox.

We begin with deterministic grading because it is reproducible, inexpensive, and
does not introduce another LLM's preferences into the labels. Subjective writing,
summarization, and LLM-judge tasks are intentionally excluded.

## Setup

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy the environment template when preparing provider credentials:

```bash
cp .env.example .env
```

Never commit `.env`.

`GLOBAL_MAX_SPEND_USD` is a hard upper bound for every live run. Each run uses
a `$1.00` spend cap by default; `--spend-cap` may select a different cap only
when it remains at or below the global maximum.

## Read-only experiment summary

```bash
python experiment_summary.py
```

This prints the six enabled models and providers, per-benchmark prompt counts,
expected model calls and result rows, a live-run confirmation preview, missing
API keys, unresolved API identifiers, and unresolved prices. It never calls a
model API.

## Dry run

```bash
python generate_dataset.py --dry-run
```

The dry run simulates the full 3-prompt × 6-model matrix, grades it, validates
every row, and writes `data/results/dry_run_results.csv`. Mock token counts,
latencies, and any zero cost caused by unresolved pricing are synthetic test data.

Each completed row records prompt and benchmark metadata, raw model output,
creator, canonical model name, API identifier, inference provider, model type,
temperature, reasoning/thinking setting, max output tokens, UTC timestamp, token
counts, latency, estimated cost, score, and correctness.

## Live-run spend protection

Live evaluation is opt-in with `python generate_dataset.py --live`. Before the
first provider call, the CLI prints the planned call count, enabled model names,
the conservative maximum estimated cost, the configured per-run cap, and the
global maximum. The run starts only with `--confirm`.

The default per-run cap is `$1.00`; it cannot exceed `GLOBAL_MAX_SPEND_USD`.
Every call is checked against the remaining cap before dispatch, then its
reported token cost is added immediately after completion. A cap violation
stops the run and no automatic retry is attempted. Unresolved model pricing
also stops live preflight because an unbounded call cannot be made safely.

Completed rows are written after each call. A stopped run can be continued with
`--resume`; completed prompt/model pairs are skipped and their recorded spend
continues to count against the same cap.

## Scope boundary

This phase does not train LightGBM or any other router, create embeddings, add
routing logic, issue model API calls, run large benchmark sweeps, serve models,
or include FastAPI, a frontend, deployment, or a database.

The old RouterBench work remains under `experiments/` and
`data/legacy/routerbench/` as a historical baseline only.
