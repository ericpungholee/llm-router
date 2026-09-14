# First ML routing experiment specification

Status: data preparation complete; **no training performed**. Freeze these choices before fitting models. The target is the probability that a particular benchmark model/configuration produces a successful answer to a visible request, under the collection's generation budget. It is not a claim about today's API versions or unbounded reasoning.

## Data and experimental units

Use `data/processed/llmrouterbench/`, pinned by `provenance.json`. Official dataset revision: `0e5af1b84bf73437a01a1849c0f1d2468baa93fc`; source commit: `c77cb0506949d8f959e97967d2fefca0e8ff1b05`. Selection is frozen in `configs/router_tasks.json` and `configs/router_model_pool.json`. See the [raw audit](llmrouterbench_data_audit.md) and [observed preparation results](llmrouterbench_preparation.md).

There are 8,706 independent prompt rows and 69,648 outcomes across eight candidate models. A sample is a whole prompt vector, never an individual model/outcome row. Positive observed costs, positive integral token counts, and available binary labels are required for all eight models. Fifty-four prompts were excluded (0.6164%); no label imputation. Raw scores, nulls, resource anomalies, and exclusion masks remain in audit artifacts.

Tasks: AIME (56), LiveMathBench (117), GPQA (198), LiveCodeBench (1,055), MMLU-Pro (2,979), SimpleQA (4,301). The distribution is dominated by knowledge QA. Report both prompt-weighted micro results and equal-dataset macro results, plus each dataset separately; do not imply balanced-domain performance. GPQA represents scientific reasoning. No selected task supports a claim about instruction following or tool use. ArenaHard's 0/0.5/1 preference rewards are preserved but excluded; this experiment introduces no continuous-score threshold.

`raw_score` is the original benchmark float. `success_label` and `success` are identical nullable binary aliases before complete-case filtering and non-null 0/1 in final outcomes. Request failures/empty outputs without positive completion usage have unknown success even when upstream stored zero. Positive usage with an empty final answer retains the observed grade. Existing judge errors and collection-budget effects remain label noise; do not regrade using paid calls. SimpleQA's score is a standardized judge label (A=correct), not independently verified truth.

## Split protocol

Use the persisted `splits.parquet` assignments, seed 3407. Assignment uses SHA-256 order within dataset strata, with largest-remainder allocation of whole duplicate groups. Group connected components share normalized (Unicode NFKC, case-folded, whitespace-collapsed) original questions or rendered request text. Split generation does not read labels, model outputs, costs, or response metadata. No selected duplicate groups remain after source selection. Exact/group leakage is checked; semantic paraphrase detection and unknown pretraining contamination are not guaranteed.

| Regime | Train | Validation | Test | Interpretation |
| --- | --- | --- | --- | --- |
| Standard | 6,094 | 1,307 | 1,305 | Approximately 70/15/15 in each dataset, rounded to whole groups |
| OOD | 6,503 | 1,148 | 1,055 | Entire LiveCodeBench code-generation domain is test; remaining datasets split 85/15 |

Run each regime independently: fresh text vocabulary, scaling, classifier parameters, calibration, estimated costs, and baseline model selection. Standard training includes code that appears in OOD test, so **never reuse a standard-trained model or its preprocessing in OOD**. An in-domain prompt connected to a held-out test prompt would be marked `excluded_overlap`, absent from training and validation. None occur in this frozen subset.

Select hyperparameters by group-stratified cross-validation inside training only. Use validation for routing-policy choice and best-single selection. No test outcomes determine a winning model, hyperparameter, threshold, cost estimate, or checkpoint. The raw audit and requested descriptive test references have already been inspected; this is a frozen exploratory benchmark split, not a previously unseen external evaluation. Do not revise the pool/subset after seeing learned test performance. Preserve the old hard pilot/provider harness for later external evaluation; its results are not added to training.

## Input contract and feature restrictions

`routing_data.loading.load_split(directory, regime, split)` returns aligned prompt metadata, numeric features X, success matrix Y, and realized cost matrix C with candidate columns in frozen pool order. Only `prompts.prompt` enters text models; only the 16 allowlisted columns enter numeric models. Y and C are targets/evaluation data, never input features.

Numeric features: character length, regex word count, `ceil(character_length/4)` token estimate, line count, code-fence delimiter count, code indicator, math-notation indicator, answer-choice line count, digit count, question-mark count, ASCII punctuation count, comma/period/colon/semicolon counts, and bracket count. Implementation: `routing_data/features.py`, version `prompt-text-v1`. Counts are cheap heuristics; token estimate is not a provider tokenizer and has language-dependent error. A code-fence count counts delimiters, not paired fenced blocks. Regex flags are not learned task classifiers.

Dataset/task/source IDs and split/hash identifiers are excluded from default predictors. A task-category heuristic may use task labels only in an explicitly labeled **known-domain / privileged-metadata** ablation: benchmark task names are not automatically available to a deployed router. It must fall back to training-global estimates for unseen domains. The visible prompt can itself reveal formatting/domain; that is allowed. Never input predictions, raw outputs, ground truth, success labels, actual model selected by another router, response tokens, latency, actual prompt-token usage, or grading-derived metadata. No embeddings are generated in this phase.

## First training experiment to run next

Fit **eight independent TF-IDF + logistic-regression classifiers**, one for each candidate's binary success label, sharing a train-fitted TF-IDF vocabulary. Begin with the standard regime, then repeat from scratch for OOD.

- TF-IDF: word unigrams/bigrams, lowercase, `min_df=2`, `max_features=50000`, `sublinear_tf=True`, L2 document normalization; use only the stored rendered prompt. Pin the future scikit-learn version in the training change.
- Logistic regression: L2, `solver=liblinear`, no class weighting, seed 3407, `max_iter=2000`. Start `C=1`; compare `{0.1, 1, 10}` using five training-only stratified group folds and mean per-model log loss. Fit a fresh vocabulary in every fold. Choose one shared C by mean log loss across candidates; break ties toward smaller C. If a fold/model has one label class, use its training prevalence as a constant predictor and record this fallback.
- Refit the selected setting on the training split. Do not refit on train+validation after policy selection in this first experiment. Use raw logistic probabilities initially; report calibration rather than adding a calibration fit by default.
- Use the training-only fixed mean observed cost per model for decisions. Scan the frozen threshold and utility grids below on validation; freeze selected policies and evaluate once on test. Save predicted probabilities separately from features.
- Primary decision: does a validation-selected policy reduce mean test cost while preserving best-single quality within 0.5 percentage points? Report the success difference and paired uncertainty, even when the condition fails. Also report the full frozen grid and OOD degradation.

Follow-up experiments (not implemented): prompt-visible handcrafted features + logistic regression with train-fitted scaling; gradient-boosted trees on those features; a known-task heuristic; frozen local embedding + logistic regression; embedding + small MLP. Choose an embedding model/revision and local inference method in a future task, with no paid embedding requirement. Keep comparable splits, labels, costs, selection budgets, and seed lists across approaches. Additional seeds may vary optimization, not silently reshuffle the benchmark test.

## Decision rules

Let p_m(x) be the predicted success probability for model m and c_hat_m the mean recorded cost estimated using only the current regime's training prompts. Initially c_hat is a fixed per-model vector. Realized test costs are unknown at decision time and are used only after routing to score the chosen action. A future prompt-dependent cost predictor must itself use training data and prompt-visible features only.

Threshold policy: choose the model with smallest c_hat_m among those with p_m(x) >= tau. If none qualify, choose maximum p_m(x), then smallest c_hat_m, then lexicographic model ID. Within the qualifying set break cost ties by larger predicted success, then model ID. Grid: tau in {0.00, 0.05, ..., 1.00}.

Utility policy: choose argmax_m [p_m(x) - lambda * normalized_cost_m], with normalized_cost_m = c_hat_m / max_j(c_hat_j). Break ties by lower c_hat_m and then model ID. Grid: lambda in {0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}. The normalization is derived from training only and is reused on validation/test.

Increasing tau generally demands more confident models; increasing lambda puts more weight on cost. Sweeping these yields candidate cost/quality operating points. Quality and cost need not be monotone in tau for imperfect predictions and the fallback; compute the actual nondominated frontier. No oracle labels or realized test costs enter either rule. These are one-call routing decisions, with no escalation, retries, or cascades.

## Reference policies

Always-cheapest uses the single model with lowest training mean observed cost, lexicographic tie-breaking. Both regimes select `qwen3-235b-a22b-2507`. The cheapest fixed model on the test set is also reported descriptively and happens to agree. Per-prompt minimum realized cost is separately labeled `hindsight_cheapest_cost`; it is an ex-post cost reference, not an implementable always-cheapest router.

Best-single maximizes mean validation success, breaking ties by training mean cost then model ID. Both regimes select `gpt-5`. Never select it using test performance. Report each candidate's micro success, mean/median cost, coverage, and per-task performance for each split in `static_model_summary.parquet`. These statistics are descriptive, not trained predictions.

Oracle success for prompt i is o_i = max_m y_im. Cheapest-success oracle chooses the lowest **realized** cost among successful models, ties by model ID. If no model succeeds, the cheapest-success identity/cost remains null and o_i=0. For a total-cost reference, charge the actual cost of the training-cheapest fallback model on such prompts. Thus every evaluated prompt incurs one model cost and oracle accuracy equals mean(o_i). Also expose conditional cost among solvable prompts separately when needed. Never report the oracle as achievable router performance.

Observed standard test references: cheapest 65.2874% at $0.00043555/prompt; validation-selected best-single 69.1188% at $0.01963688; oracle success 86.6667% with total-policy cost $0.00307040. Exact values are in the generated JSON. The 17.5479 percentage-point success gap and 84.3641% oracle cost reduction are ex-post headroom only. OOD best-single is 86.4455% versus oracle 91.1848%, a 4.7393 percentage-point gap.

## Metrics and model selection

For test prompts i=1,...,N, selected model r_i, stored binary label y_im, and realized cost c_im:

| Metric | Definition and reporting rule |
| --- | --- |
| Success rate / routing quality Q | (1/N) sum_i y_i,r_i. This is the routing outcome metric. Per-model classifier accuracy is secondary. |
| Mean cost C | (1/N) sum_i c_i,r_i, USD per prompt. Include failed answers' charged cost. No router overhead is invented; report measured local routing compute separately when training is implemented. |
| Macro success / cost | Compute Q and C within each dataset, then average datasets equally. Also report every dataset independently. |
| Cost savings at best-single quality | For the frozen set R of grid policies, Savings@Q_B = 1 - min_{r in R: Q_r >= Q_B - epsilon} C_r / C_B, epsilon=0.005 absolute. Report exact-quality epsilon=0 sensitivity. If infeasible, say infeasible. This test-envelope value is descriptive; it does not choose deployment policy. |
| Deployable quality-preserving policy | Choose minimum-cost feasible policy on validation using Q_r,val >= Q_B,val - 0.005, ties by greater quality then stable policy ID. Include best-single as a feasible fallback. Freeze it; report its test savings and test quality difference without choosing a replacement from test. |
| Quality gain at fixed cost | At each preset budget b, Gain@b = max_{r in R: C_r <= b} Q_r - max_{m: C_m <= b} Q_m. If either set is empty, report infeasible. Budgets are {0.25,0.5,0.75,1.0} times best-single training mean cost. Test-envelope values are descriptive; additionally report the policy chosen under that budget on validation and any test budget violation. |
| Pareto frontier | Points (C_r,Q_r) not dominated by another with C no larger and Q no smaller, with at least one strict improvement. Plot frozen policies and all static models, distinguishing validation-selected points. Do not optimize the grid after seeing test. |
| Oracle gap recovered | (Q_router - Q_B)/(Q_oracle - Q_B). Negative is allowed; report undefined if denominator <= 0. Use the same prompts and current regime's validation-selected best-single. Success gap and oracle cost headroom are separate quantities. |
| OOD performance | Compute all metrics on the entire held-out code domain from the independent OOD fit. Report standard versus OOD code performance with their different training distributions and denominators. No OOD-specific tuning. |
| Calibration | For each model and for the selected action, report Brier mean((p-y)^2), clipped log loss with p in [1e-6,1-1e-6], and ECE=sum_b(n_b/N)*abs(mean_b(p)-mean_b(y)) in 10 fixed equal-width bins. Assign p=1 to last bin; omit empty bins. Report all-model macro Brier/log loss and a reliability plot. |

Use 2,000 paired bootstrap replicates, seed 3407, resampling whole prompt groups within dataset strata for standard evaluation and whole prompts/groups within held-out code for OOD. Recompute router-minus-baseline success differences and cost savings using the same sampled prompts. Report percentile 95% intervals. Training fits and policy selection remain fixed during bootstrap; these intervals measure evaluation-sample uncertainty, not model-selection or training-seed uncertainty. Treat noninferiority within epsilon as supported only when the lower interval bound for Q_router - Q_B is >= -epsilon. This may require more data for small tasks; do not declare preservation from an unsupported point estimate alone.

For future calibration, use training out-of-fold predictions or a separate calibration portion of validation whose labels do not also tune routing policies. Do not train a calibrator on test. Report classifier AUC/log loss as diagnostics, but judge the project on cost/quality decisions, frontier, oracle gap, and domain generalization.

## Reproduction and artifact use

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r experiments/requirements-llmrouterbench.txt
.venv/bin/python scripts/prepare_llmrouterbench.py --download
.venv/bin/python tests/run_offline_suite.py
```

The first preprocessing command downloads only the public 1.28 GB archive if absent and verifies its size/SHA-256 before reading it. Subsequent runs work offline. No extraction is necessary; stream JSON payloads one file at a time. All raw/processed data stays ignored by git. The canonical artifacts are Parquet; the 80-row CSV sample is for inspection only. Preserve `provenance.json`, config copies, code hashes, and persisted splits alongside results. The reference environment uses CPython 3.9.6 and pinned pandas 2.2.3, NumPy 2.0.2, and PyArrow 19.0.1; byte reproducibility is checked in that environment. For another platform/Python, compare logical tables/splits if Parquet encoder bytes differ.

`outcomes.parquet`, `prompts.parquet`, `splits.parquet`, and `prompt_features.parquet` are training inputs. `baseline_choices.parquet`, `oracle_targets.parquet`, and `static_model_summary.parquet` are evaluation references. `audit_raw_outcomes.parquet` preserves every archive score, including continuous rewards; `audit_selected_outcomes.parquet` preserves masks, raw costs, and invalid tokens before filtering. `missingness_by_prompt.parquet`, model/dataset/task/model-dataset coverage tables, and `audit_source_files.parquet` document exclusions and every source JSON hash. `matrix_metadata.json` and `reports/llmrouterbench_processed_summary.json` contain exact counts/statistics. No further infrastructure is required to start the specified logistic-regression experiment.
