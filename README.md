# Learned LLM Router

A reproducible, offline experiment in choosing an LLM from prompt text. Eight
TF-IDF + logistic-regression predictors estimate each candidate's probability of
success; a validation-selected policy trades inference cost against quality.
The router returns a model choice and probabilities without calling an LLM.

## Results

The latest [fixed-reference experiment](reports/router_v4_results.md) improves
standard-test quality while saving cost against GPT-5:

| Test | Router success | GPT-5 success | Cost savings | Quality difference, 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Standard, 1,305 prompts | 72.57% | 69.12% | 12.57% | +3.45 pp [+1.91, +4.91] |
| Unseen code domain, 1,055 prompts | 86.35% | 86.45% | 1.13% | −0.09 pp [−0.38, +0.19] |

**Useful savings on unseen code remain unproven.** Standard gains concentrate in
SimpleQA; removing it leaves 0.28% savings. These are exploratory results on
reused benchmark outcomes, with historical recorded costs, not current API
prices or independent confirmation. Earlier failed experiments remain in the
[experiment history](docs/experiments.md).

![Standard cost and quality, with per-dataset uncertainty](reports/figures/router_v4_standard_cost_quality.png)

## Quick start

Use Python 3.11 for development. The original numerical runs used Python 3.9.6;
retain that environment when comparing exact saved coefficients.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python tests/run_offline_suite.py
ruff check .
ruff format --check .
```

No API keys, dataset downloads, or model downloads are needed for these checks.
A fresh clone runs synthetic and historical-snapshot tests; checks requiring the
large generated research artifacts skip explicitly. With those artifacts
present, the same command also verifies their predictions, selections, hashes,
bootstrap intervals, and reproducibility. The runner blocks network access.

## Reproduce and use the router

Raw data and fitted models are intentionally excluded from Git. On a fresh
clone, first reconstruct the **existing frozen dataset** using its pinned public
archive (a 1.28 GB download):

```bash
python scripts/prepare_llmrouterbench.py --download
```

Skip that command if `data/processed/llmrouterbench/` already exists. Train and
evaluate the latest experiment locally:

```bash
python experiments/fixed_reference_router.py
python experiments/route_local.py \
  --artifact-dir artifacts/router_v4/standard \
  --prompt 'What is the capital of France?'
```

Training is offline. It refuses to overwrite an existing campaign. For a replay,
use a fresh `--output-dir` and `--reports-dir`, as shown in the
[reproduction guide](docs/experiments.md). Local inference accepts a single
prompt or `--input requests.jsonl`, with one object per line:

```json
{"prompt_id": "request_1", "prompt": "Explain binary search."}
```

Output includes the selected model, eight success probabilities, policy ID, and
training-estimated cost. This is model selection, not response generation.
Always pass the artifact directory explicitly; the CLI retains its older v2
default for compatibility with the frozen v2 report. Load only artifacts you
trust: Joblib files execute Python during loading, and hashes verify integrity,
not authenticity.

## Experiment design

The frozen matrix contains **8,706 prompts × 8 models = 69,648 outcomes**, across
AIME, LiveMathBench, GPQA, LiveCodeBench, MMLU-Pro, and SimpleQA. Candidate order
and identities are pinned in [the model pool](configs/router_model_pool.json).
These historical candidates are separate from the five hosted models in the
archived provider harness.

- Training-only vocabularies, feature scaling, cross-validation, and cost estimates.
- Validation-only policy selection; standard and OOD systems trained independently.
- Static and oracle comparisons, micro/macro metrics, per-dataset diagnostics,
  calibration, Pareto frontiers, and 2,000-replicate paired group bootstraps.
- Saved source hashes, predictions, decisions, environments, and immutable reports.

V1 tunes shared regularization over `{0.1, 1, 10}`. V4 fixes `C=1` and adds
source-domain validation against a fixed GPT-5 quality reference. See the
[frozen v4 protocol](reports/router_v4_experiment_spec.md) for the exact selection
rule and interpretation limits.

## Repository map

| Path | Purpose |
| --- | --- |
| `routing_ml/` | Training, policy selection, metrics, uncertainty, local inference |
| `experiments/` | Versioned experiments and reproduction entry points |
| `routing_data/`, `configs/` | Frozen data preparation, splits, source/model contracts |
| `tests/` | Offline unit, regression, and artifact checks |
| `reports/` | Protocols, results, figures, and historical pilot snapshots |
| `data/`, `artifacts/` | Large reproducible local outputs, ignored by Git |

See [development notes](docs/development.md) for checks and provenance rules,
[data provenance](reports/llmrouterbench_data_audit.md) for source details, and
[third-party notices](THIRD_PARTY_NOTICES.md) for attribution. The separate
[historical provider harness](docs/provider_harness.md) is archived context;
it is not required to train or use this router.
