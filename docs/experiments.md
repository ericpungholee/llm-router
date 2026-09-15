# Experiment history and reproduction

All commands below run from the repository root after the README setup. They
use `data/processed/llmrouterbench/`; do not regenerate an existing processed
matrix. Every version preserves its own selection protocol and artifacts.

| Version | Question | Finding | Protocol / results |
| --- | --- | --- | --- |
| V1 | Can TF-IDF success prediction support absolute threshold/utility routing? | Standard policy saves 82.70% but loses 1.99 quality points; noninferiority fails. Includes the 16-feature control. | [Protocol](../reports/ml_experiment_spec.md) / [results](../reports/router_v1_results.md) |
| V2 | Do a frozen encoder and comparative routing help? | TF-IDF wins; embeddings do not. Standard savings reach 22.07%, but the stricter dataset guard fails. | [Protocol](../reports/router_v2_experiment_spec.md) / [results](../reports/router_v2_results.md) |
| V3 | Does the fixed TF-IDF rule survive grouped cross-fitting? | Gains depend on SimpleQA. OOD source validation chooses Qwen as reference and fails to protect code quality. | [Protocol](../reports/router_v3_robustness_spec.md) / [results](../reports/router_v3_results.md) |
| V4 | Does a fixed GPT-5 reference plus source-domain validation address that failure? | Standard criteria pass; useful OOD savings remain unproven. | [Protocol](../reports/router_v4_experiment_spec.md) / [results](../reports/router_v4_results.md) |

These are successive exploratory experiments on the same benchmark, not four
independent confirmations. Test-set findings informed subsequent research
questions. Each version freezes its own policy before that version's test
scoring, but this does not remove researcher exposure to earlier test results.

## Latest experiment: V4

The default command writes `artifacts/router_v4/` and `reports/` and refuses an
existing nonempty artifact directory. Preserve completed results by replaying to
fresh directories:

```bash
router_run_dir=$(mktemp -d /tmp/router-v4-replay.XXXXXX)
python experiments/fixed_reference_router.py \
  --output-dir "$router_run_dir" --reports-dir "$router_run_dir/reports"
python experiments/route_local.py \
  --artifact-dir "$router_run_dir/standard" --prompt 'Explain binary search.'
```

To regenerate reports from existing artifacts without fitting or scoring again
(published test-run footers record historical checks separately):

```bash
python experiments/fixed_reference_router.py --report-only \
  --output-dir artifacts/router_v4 --reports-dir /tmp/router-v4-report
```

Models, predictions, source-domain partitions, cost estimates, selected policies,
routing decisions, and bootstrap samples stay under the artifact directory.
Artifacts also preserve the source code and environment used by their run.

## Earlier experiments

These commands require fresh output directories too. Their default report paths
contain the tracked historical reports; use `--reports-dir` for independent
replays rather than replacing the published evidence.

```bash
router_v1_run=$(mktemp -d /tmp/router-v1-replay.XXXXXX)
python experiments/tfidf_logreg_router.py \
  --output-dir "$router_v1_run" --reports-dir "$router_v1_run/reports"

router_v3_run=$(mktemp -d /tmp/router-v3-replay.XXXXXX)
python experiments/crossfit_tfidf_router.py \
  --output-dir "$router_v3_run" --reports-dir "$router_v3_run/reports"
```

V2 is optional and requires the larger local encoder environment. Downloading
the pinned public encoder is explicit; embedding computation and training run
locally and do not call an inference API:

```bash
python -m pip install -r experiments/requirements-router-v2.txt
python -m routing_ml.embeddings --download
python -m routing_ml.embeddings
router_v2_run=$(mktemp -d /tmp/router-v2-replay.XXXXXX)
ln -s "$PWD/artifacts/router_v2/embeddings" "$router_v2_run/embeddings"
python experiments/embedding_logreg_router.py \
  --output-dir "$router_v2_run" --reports-dir "$router_v2_run/reports"
```

For a fresh data reconstruction, use the README's pinned download command. The
[source audit](../reports/llmrouterbench_data_audit.md) and
[preparation report](../reports/llmrouterbench_preparation.md) document exclusions,
complete-case filtering, split counts, and source hashes. The unused older
binary RouterBench importer has been removed; no current experiment depends on
its downloaded pickle.
