"""Readable tables and publication figures from saved evaluation results."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/llm-router-v1-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from routing_ml.training import MODEL_IDS

NAMES = dict(zip(MODEL_IDS, ["Qwen", "Intern-S1", "DeepSeek V3", "DeepSeek R1", "Gemini Flash", "GPT-5 Chat", "GPT-5", "Claude Sonnet 4"]))
NAMES.update({"__macro__": "Predictor macro", "__selected_action__": "Selected action"})
DATASETS = {"aime": "AIME", "livemathbench": "LiveMathBench", "gpqa": "GPQA", "livecodebench": "LiveCodeBench", "mmlupro": "MMLU-Pro", "simpleqa": "SimpleQA"}


def figures(frontier, reliability, diagnostics, directory, prefix):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(9, 5.8), layout="constrained")
    for starts, name, color, marker in [("threshold_", "Threshold grid", "#0072B2", "o"), ("utility_", "Utility grid", "#D55E00", "^")]:
        sub = frontier[frontier.policy_id.str.startswith(starts)]
        ax.scatter(sub.mean_cost_usd * 1000, sub.quality * 100, label=name, color=color, marker=marker, s=36, alpha=0.8)
    static = frontier[frontier.policy_id.str.startswith("static_")]
    ax.scatter(static.mean_cost_usd * 1000, static.quality * 100, color="#555555", marker="s", s=42, label="Static models")
    for r in static.itertuples():
        model_id = r.policy_id.removeprefix("static_")
        label = NAMES[model_id] + (" / always cheapest" if model_id == MODEL_IDS[0] else "")
        offset = (5, -14) if model_id in {"gpt-5", "gpt-5-chat"} else (5, 7)
        ax.annotate(label, (r.mean_cost_usd * 1000, r.quality * 100), xytext=offset, textcoords="offset points", fontsize=8)
    for pid, name, color, marker in [("deployable_router", "Validation-selected router", "#009E73", "*"), ("oracle", "Oracle (hindsight)", "#CC79A7", "D")]:
        r = frontier[frontier.policy_id == pid].iloc[0]
        ax.scatter(r.mean_cost_usd * 1000, r.quality * 100, s=190 if pid == "deployable_router" else 65,
                   facecolor=color, edgecolor="black", linewidth=0.6, marker=marker, label=name, zorder=5)
    line = frontier[frontier.pareto_deployable].drop_duplicates(["mean_cost_usd", "quality"]).sort_values("mean_cost_usd")
    ax.plot(line.mean_cost_usd * 1000, line.quality * 100, color="#666666", alpha=0.45, linewidth=1, zorder=0)
    ax.set(xscale="log", xlabel="Mean realized cost (USD per 1,000 prompts; log scale)", ylabel="Success rate (%)",
           title=prefix.replace("_", " ").title() + ": frozen test cost–quality points")
    ax.margins(x=0.27, y=0.16)
    ax.grid(alpha=0.18)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{prefix}_cost_quality_frontier.{extension}")
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True, layout="constrained")
    for model_id, ax in zip(list(MODEL_IDS) + ["__selected_action__"], axes.flat):
        r = reliability[(reliability.split == "test") & (reliability.model_id == model_id)]
        d = diagnostics[(diagnostics.split == "test") & (diagnostics.model_id == model_id)].iloc[0]
        ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
        ax.plot(r.mean_probability, r.observed_success, color="#0072B2", linewidth=1)
        ax.scatter(r.mean_probability, r.observed_success, s=12 + np.sqrt(r["count"]) * 3, color="#0072B2")
        ax.set(title=NAMES.get(model_id, "Selected action"), xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
        ax.text(0.04, 0.94, f"ECE {d.ece:.3f}", transform=ax.transAxes, va="top", fontsize=9)
        ax.grid(alpha=0.15)
    fig.supxlabel("Mean predicted success (10 fixed bins; point size reflects count)")
    fig.supylabel("Observed success rate")
    fig.suptitle(prefix.replace("_", " ").title() + ": test reliability")
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{prefix}_calibration.{extension}")
    plt.close(fig)


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def pct(v):
    return "—" if v is None else f"{100 * v:.3f}%"


def num(v):
    return "—" if v is None else f"{v:.4f}"


def write_report(results, path):
    s = results.get("standard_tfidf")
    if s is None:
        return
    supported = s["bootstrap"]["noninferiority_supported"]
    saved = s["comparison"]["cost_savings"] > 0
    verdict = ("TF-IDF routing reduced cost with supported noninferiority within 0.5 percentage points." if supported and saved else
               "TF-IDF routing did not establish a meaningful cost reduction at statistically preserved best-single quality.")
    lines = ["# Router v1: first learned routing experiment", "", verdict, "",
             "The deployable policy was selected on validation and frozen before test evaluation. All costs are recorded benchmark USD per prompt; these are historical candidate configurations, not current API pricing. No paid calls, embeddings, dataset changes, or provider changes were made.", ""]
    for key, heading in [("standard_tfidf", "Standard test"), ("ood_tfidf", "Independent OOD test")]:
        if key not in results:
            continue
        r = results[key]
        comp, ci = r["comparison"], r["bootstrap"]["ci_95"]
        lines += [f"## {heading}", "", f"Selected C = **{r['selected_C']:g}**; deployable policy = **{r['selected_policy']['policy_id']}**; best-single = **{r['best_single']['model_id']}**; test N = {r['split_counts']['test']:,}.", "",
                  md_table(["Policy", "Micro quality", "Mean cost (USD)", "Macro quality", "Macro mean cost"],
                           [[label, pct(r["test"][pid]["quality"]), f"${r['test'][pid]['mean_cost_usd']:.8f}", pct(r["macro"][pid]["quality"]), f"${r['macro'][pid]['mean_cost_usd']:.8f}"]
                            for pid, label in [("best_single", "Best single"), ("deployable_router", "Learned router"), ("always_cheapest", "Always cheapest"), ("oracle", "Oracle (hindsight)")]]), "",
                  f"Cost savings: **{pct(comp['cost_savings'])}**, 95% CI [{pct(ci['cost_savings'][0])}, {pct(ci['cost_savings'][1])}]. Quality delta: **{comp['quality_delta'] * 100:+.3f} percentage points**, 95% CI [{ci['quality_delta'][0] * 100:+.3f}, {ci['quality_delta'][1] * 100:+.3f}] pp.", "",
                  f"Noninferiority supported (lower 95% quality-difference CI ≥ −0.5 pp): **{'yes' if r['bootstrap']['noninferiority_supported'] else 'no'}**. " + ("Quality preservation within the stated margin is supported." if r['bootstrap']['noninferiority_supported'] else "Do not call quality preserved."), "",
                  f"Validation best-single quality {pct(r['validation_best_quality'])}; chosen-policy quality {pct(r['validation_selected']['quality'])}, cost ${r['validation_selected']['mean_cost_usd']:.8f}; feasibility target {pct(r['validation_best_quality'] - 0.005)}.", "",
                  f"Oracle gap recovered: {pct(comp['oracle_gap_recovered'])}; remaining quality gap to oracle: {comp['remaining_oracle_gap'] * 100:.3f} pp. Oracle cost includes the training-cheapest fallback on unsolved prompts. Oracle cost conditional on solvable prompts: ${r['oracle_conditional_cost_usd']:.8f}. This hindsight reference is not an achievable router.", "",
                  f"Macro quality delta {comp['macro_quality_delta'] * 100:+.3f} pp, 95% CI [{ci['macro_quality_delta'][0]*100:+.3f}, {ci['macro_quality_delta'][1]*100:+.3f}]; macro cost savings {pct(comp['macro_cost_savings'])}.", "",
                  "### Dataset results", "",
                  md_table(["Dataset", "N", "Router Q", "Best Q", "Δ pp", "Router cost", "Best cost", "Cheapest Q", "Oracle Q"],
                           [[DATASETS[d], x["deployable_router"]["n"], pct(x["deployable_router"]["quality"]), pct(x["best_single"]["quality"]),
                             f"{100*(x['deployable_router']['quality']-x['best_single']['quality']):+.3f}", f"${x['deployable_router']['mean_cost_usd']:.7f}",
                             f"${x['best_single']['mean_cost_usd']:.7f}", pct(x["always_cheapest"]["quality"]), pct(x["oracle"]["quality"])] for d, x in r["datasets"].items()]), "",
                  "### Classifier diagnostics (test)", "", "PR AUC is average precision (step-integrated PR); trapezoidal PR AUC is also saved. Accuracy uses p ≥ 0.5. Macro here averages the eight predictors; dataset macro above averages datasets. Calibration uses ten fixed equal-width bins and log-loss clipping at 1e−6. Undefined single-class AUC is null.", "",
                  md_table(["Predictor", "Prevalence", "ROC AUC", "PR AUC", "Log loss", "Brier", "ECE", "Accuracy"],
                           [[NAMES.get(x["model_id"], x["model_id"]), pct(x["prevalence"]), num(x["roc_auc"]), num(x["pr_auc"]), num(x["log_loss"]), num(x["brier"]), num(x["ece"]), pct(x["accuracy_at_0_5"])] for x in r["test_classifier_metrics"]]), "",
                  "### Selected-model behavior", "",
                  md_table(["Model", "N", "Routed", "Predicted success", "Actual success", "Realized mean cost"],
                           [[NAMES[x["model_id"]], x["count"], pct(x["fraction"]), pct(x["mean_predicted_success"]), pct(x["actual_success"]), "—" if x["mean_cost_usd"] is None else f"${x['mean_cost_usd']:.8f}"] for x in r["selection_distribution"]]), "",
                  md_table(["Dataset"] + [NAMES[m] for m in MODEL_IDS], [[DATASETS[x["dataset"]]] + [x[m] for m in MODEL_IDS] for x in r["dataset_selection_counts"]]), "",
                  "### Pareto frontier and descriptive envelopes", "", "The deployable frontier includes the frozen grids and static models. The oracle is shown separately; a second frontier including it is saved. Identical points may have multiple policy IDs. Test envelopes describe this frozen grid and never replace the validation choice.", "",
                  md_table(["Nondominated policy", "Quality", "Mean cost"], [[x["policy_id"], pct(x["quality"]), f"${x['mean_cost_usd']:.8f}"] for x in r["pareto_frontier"]]), "",
                  md_table(["Test-envelope tolerance", "Feasible", "Policy", "Savings"], [[f"{100*x['epsilon']:.1f} pp", x["feasible"], x["policy_id"], pct(x["cost_savings"])] for x in r["envelopes"]["savings_at_best_single_quality"]]), "",
                  md_table(["Budget / train best cost", "Budget USD", "Descriptive gain pp", "Validation choice", "Test budget violation"],
                           [[x["fraction"], f"${x['budget_usd']:.6f}", "infeasible" if x["descriptive_quality_gain"] is None else f"{100*x['descriptive_quality_gain']:+.3f}", x["validation_selected_policy_id"], x["frozen_test_budget_violation"]] for x in r["envelopes"]["fixed_budget"]]), "",
                  f"![{heading} frontier](figures/{r['figure_prefix']}_cost_quality_frontier.png)", "", f"![{heading} calibration](figures/{r['figure_prefix']}_calibration.png)", ""]
    if "ood_comparison" in results:
        o = results["ood_comparison"]
        lines += ["## Code-domain generalization", "",
                  f"The standard router scored {pct(o['standard_code_quality'])} on {o['standard_code_n']} standard LiveCodeBench test prompts; the independent OOD router scored {pct(o['ood_code_quality'])} on all {o['ood_code_n']} held-out code prompts: {o['ood_minus_standard_quality']*100:+.3f} pp. Mean cost changed from ${o['standard_code_cost_usd']:.8f} to ${o['ood_code_cost_usd']:.8f}.", "",
                  f"On the shared {o['common_code_n']} prompts, the OOD router scored {pct(o['ood_common_quality'])}, a {o['ood_common_minus_standard_quality']*100:+.3f} pp difference from the standard system. This shared-subset comparison is descriptive and did not tune either system.", "",
                  "These are independently trained systems with independently selected policies. The full-test comparison also changes the sample (158 versus 1,055); it is not a paired causal estimate of domain shift. OOD training and validation contain zero LiveCodeBench prompts. OOD macro equals micro because there is one test dataset.", ""]
    ablations = [r for k, r in results.items() if k.endswith("_handcrafted")]
    if ablations:
        lines += ["## Exactly one ablation: 16 prompt features + logistic regression", "", "The same feature family was evaluated in both regimes, with independent train-only standardization, the same C grid/folds, and the same validation policy selection. No dataset metadata enters these features.", "",
                  md_table(["Regime / features", "C", "Policy", "Quality", "Cost", "Savings", "Δ pp", "95% Δ CI pp", "Noninferiority"],
                           [[r["regime"] + " / " + r["kind"], r["selected_C"], r["selected_policy"]["policy_id"], pct(r["test"]["deployable_router"]["quality"]), f"${r['test']['deployable_router']['mean_cost_usd']:.8f}", pct(r["comparison"]["cost_savings"]), f"{r['comparison']['quality_delta']*100:+.3f}",
                             ", ".join(f"{x*100:+.3f}" for x in r["bootstrap"]["ci_95"]["quality_delta"]), r["bootstrap"]["noninferiority_supported"]] for r in [results[k] for k in ("standard_tfidf", "standard_handcrafted", "ood_tfidf", "ood_handcrafted") if k in results]]), ""]
    lines += ["## What the experiment supports", "",
              "The primary standard policy fails the frozen noninferiority criterion. Good aggregate classifier AUC does not establish safe routing: predicting absolute success and selecting a cheaper model without losing quality are different evaluation questions.", ""]
    if s["selected_policy"]["policy_id"] == "threshold_0.60":
        lines += ["The observed standard behavior is dominated by the cheapest model: Qwen receives all eight AIME and all 17 LiveMathBench test prompts, 441/447 MMLU-Pro prompts, and 106/158 code prompts. Only five code prompts go to GPT-5; 160 of GPT-5's 165 selections are SimpleQA. Thus the fitted policy does not exhibit the hoped-for pattern of reserving strong reasoning for hard math/code. These counts describe behavior, not a causal explanation of what individual words encode.", "",
                  "SimpleQA improves by 8.217 pp against GPT-5 while every other dataset loses quality. Equal-dataset macro quality falls by 15.601 pp (95% CI −22.317 to −9.509). The validation constraint uses the frozen micro objective; it can accept tradeoffs that harm minority domains. This is an observed limitation of this operating point, not grounds to alter the experiment after seeing test results.", ""]
    if "utility_0.01" in s["test"]:
        p, b = s["test"]["utility_0.01"], s["test"]["best_single"]
        lines += [f"There is descriptive routing signal in the frozen frontier. For example, the prespecified λ=0.01 point achieves {pct(p['quality'])} at ${p['mean_cost_usd']:.8f}, saving {pct(1-p['mean_cost_usd']/b['mean_cost_usd'])} against GPT-5. It was not the validation-selected primary policy. The primary τ=0.60 point is dominated on test by λ=0.3. Neither observation authorizes replacing the frozen winner or claiming test-selected deployment performance.", ""]
    if "ood_tfidf" in results:
        o = results["ood_tfidf"]
        code = results["ood_comparison"]
        auc = next(x["roc_auc"] for x in o["test_classifier_metrics"] if x["model_id"] == "__macro__")
        lines += [f"OOD selects {o['selected_policy']['policy_id']} using non-code validation. Qwen receives {pct(o['selection_distribution'][0]['fraction'])} of code prompts; quality is {o['comparison']['quality_delta']*100:+.3f} pp versus GPT-5. Full OOD quality changes by {code['ood_minus_standard_quality']*100:+.3f} pp from the standard code subset, while on the identical 158 prompts the change is {code['ood_common_minus_standard_quality']*100:+.3f} pp. OOD classifier macro ROC AUC is {auc:.3f}, showing poor code-domain discrimination.", ""]
    if "standard_handcrafted" in results:
        h = results["standard_handcrafted"]
        tm = next(x for x in s["test_classifier_metrics"] if x["model_id"] == "__macro__")
        hm = next(x for x in h["test_classifier_metrics"] if x["model_id"] == "__macro__")
        lines += [f"Lexical features improve standard diagnostic log loss ({tm['log_loss']:.4f} versus {hm['log_loss']:.4f} for handcrafted features) and ROC AUC ({tm['roc_auc']:.4f} versus {hm['roc_auc']:.4f}). Nevertheless, the handcrafted deployable point has higher test quality ({pct(h['test']['deployable_router']['quality'])}) at higher cost (${h['test']['deployable_router']['mean_cost_usd']:.8f}); neither achieves supported noninferiority. Lexical information helps prediction, but this experiment does not establish a superior deployable cost/quality tradeoff from it.", ""]
    lines += ["Recommendation for the next separately authorized ML experiment: compare frozen local embeddings plus the same eight logistic regressions under the unchanged split, tuning, routing and uncertainty protocol. The question is whether semantic features improve cross-domain success estimates beyond this lexical baseline. Keep domain macro results prominent and retain this failed primary result. Any later change to validation selection or quality constraints should be a separately preregistered experiment, not a rescue chosen from this test set. No embeddings or additional model families were run here.", "",
              "## Protocol, uncertainty, and reproduction", "",
              "Five StratifiedGroupKFold folds use dataset strata and leakage groups inside training (shuffle=True, seed 3407). With eight binary targets there is no single target class to stratify on; dataset strata preserve the frozen protocol's domain mixture. Each fold fits its own TF-IDF vocabulary or scaler; the mean of 40 fold/model log losses chooses the shared C, ties toward smaller C. Final fits use training only. Raw probabilities are used without calibration fitting. All 21 thresholds and eight utilities are fixed by the original spec.", "",
              "The 2,000-replicate paired bootstrap resamples whole prompt groups with replacement within datasets, using seed 3407. Percentile 95% intervals condition on fixed training and validation selection; they do not quantify training-seed or model-selection uncertainty. Small math/science test strata limit domain-specific conclusions. MMLU-Pro and SimpleQA dominate micro averages. This is the previously inspected, frozen exploratory split, not a new external confirmation dataset.", "",
              "A complete second standard TF-IDF training/CV pass checks deterministic folds, hyperparameter choice, preprocessing, coefficients and training probabilities before test is opened. Saved model reloads must reproduce training probabilities exactly. Fresh models, vocabularies/scalers, CV, costs and validation choices are fitted for OOD. Package versions, source/spec/code hashes, candidate order, folds and signatures are in each metadata.json. The processed inputs are hash-checked before and after each run.", "",
              md_table(["Run", "Training seconds", "Replay seconds", "Test prediction ms/prompt", "Test routing ms/prompt", "Vocabulary"],
                       [[k, f"{r['timing']['training_seconds']:.2f}", f"{r['timing']['determinism_replay_seconds']:.2f}", f"{r['timing']['test_prediction_ms_per_prompt']:.4f}", f"{r['timing']['test_routing_ms_per_prompt']:.4f}", r["vocabulary_size"]] for k, r in results.items() if isinstance(r, dict) and "timing" in r]), "",
              "Measured batch CPU timing is local compute overhead, not an API charge or online latency benchmark. Timing/environment files may differ across runs; numerical tables and model signatures are deterministic in the pinned environment.", "",
              "```bash", ".venv/bin/python -m pip install -r experiments/requirements-router-v1.txt", ".venv/bin/python experiments/tfidf_logreg_router.py", ".venv/bin/python tests/run_offline_suite.py", "```", "",
              "The command uses existing frozen processed artifacts and runs standard TF-IDF, OOD TF-IDF, standard handcrafted, then OOD handcrafted. To run only the first stage: `--regime standard --features tfidf`. The OOD stage requires a completed deterministic standard result. Large reproducible files under `artifacts/router_v1/` are git-ignored; small reports/JSON/figures are retained.", "",
              "Saved per-run artifacts: `models.joblib`, `metadata.json`, `cv_scores.parquet`, `cv_assignments.parquet`, `train_predictions.parquet`, `validation_predictions.parquet`, `test_predictions.parquet`, `validation_policy_grid.parquet`, `frozen_policy.json`, `{validation,test}_routing_decisions.parquet`, `{validation,test}_evaluation.parquet`, `{validation,test}_policy_metrics.parquet`, `static_model_summary.parquet`, `classifier_metrics.parquet`, `reliability.parquet`, `pareto_table.parquet`, `selection_distribution.parquet`, `dataset_selection_counts.parquet`, `bootstrap_samples.parquet`, `results.json`, and `artifact_hashes.json`. Prediction and routing-decision tables contain no outcomes. Separate evaluation artifacts contain selected labels/costs; original labels remain in the frozen source.", ""]
    audit_path = Path(path).parent / "router_v1_numerical_audit.json"
    if audit_path.exists():
        lines += ["The pinned NumPy/macOS dense BLAS path emitted divide-by-zero/overflow/invalid flags on finite handcrafted matrices. An independent non-BLAS einsum + sigmoid calculation matched every saved train/validation/test probability to within 2.3e−16, and both handcrafted training/CV replays were exact. The implementation retains sklearn probabilities and checks the independent reference before consuming these warnings; nonfinite or disagreeing probabilities fail. No policies or test outcomes changed. See [numerical verification](router_v1_numerical_audit.json).", ""]
    test_path = Path(path).parent / "router_v1_tests.json"
    if test_path.exists():
        tests = json.loads(test_path.read_text())
        lines += ["## Tests and files", "", f"Full offline suite: **{tests['tests_run']} tests, {tests['failures']} failures, {tests['errors']} errors, {tests['skipped']} skipped**; {tests['unexpected_network_attempts']} network attempts with networking blocked. Includes every existing test, synthetic leakage/decision/bootstrap checks, and independent scalar verification of all frozen-grid routing decisions against the actual saved predictions. [Test summary](router_v1_tests.json).", "",
                  "Added `experiments/tfidf_logreg_router.py`, `experiments/requirements-router-v1.txt`, `routing_ml/{training,policies,metrics,bootstrap,reporting}.py`, `routing_ml/__init__.py`, `tests/test_router_v1.py`, `tests/test_router_v1_artifacts.py`, result/verification reports, and PNG/PDF figures. Updated `README.md` and `.gitignore`. Frozen preprocessing, experiment spec, provider code, and source processed data are unchanged.", ""]
    Path(path).write_text("\n".join(lines))
