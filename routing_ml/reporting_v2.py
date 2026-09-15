"""Report the frozen v2 campaign, without choosing a winner from test results."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from routing_ml.reporting import DATASETS, NAMES, md_table, num, pct
from routing_ml.semantic_training import FAMILIES
from routing_ml.training import MODEL_IDS


def plot_results(output, reports, regime):
    import matplotlib.pyplot as plt

    table = pd.read_parquet(output / "pareto_table.parquet")
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    colors = dict(zip(FAMILIES, ["#777777", "#E69F00", "#0072B2", "#56B4E9"]))
    for family in FAMILIES:
        rows = table[table.policy_id.str.startswith(family + "__")]
        ax.scatter(
            rows.mean_cost_usd * 1000,
            rows.quality * 100,
            color=colors[family],
            s=14,
            alpha=0.35,
            label=family.replace("_", " + "),
        )
    static = table[table.policy_id.str.startswith("static_")]
    ax.scatter(
        static.mean_cost_usd * 1000,
        static.quality * 100,
        marker="s",
        color="black",
        s=24,
        label="Static models",
    )
    for row in static.itertuples():
        name = row.policy_id.removeprefix("static_")
        ax.annotate(
            NAMES[name],
            (row.mean_cost_usd * 1000, row.quality * 100),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=7,
        )
    for pid, label, marker, color in [
        ("primary", "Frozen primary", "*", "#009E73"),
        ("oracle", "Oracle (hindsight)", "D", "#CC79A7"),
    ]:
        row = table[table.policy_id == pid].iloc[0]
        ax.scatter(
            row.mean_cost_usd * 1000,
            row.quality * 100,
            marker=marker,
            color=color,
            edgecolors="black",
            linewidths=0.6,
            s=180 if pid == "primary" else 60,
            label=label,
            zorder=6,
        )
    frontier = (
        table[table.pareto_deployable]
        .drop_duplicates(["mean_cost_usd", "quality"])
        .sort_values("mean_cost_usd")
    )
    ax.plot(
        frontier.mean_cost_usd * 1000,
        frontier.quality * 100,
        color="#555555",
        alpha=0.5,
        linewidth=1,
    )
    ax.set(
        xscale="log",
        xlabel="Mean realized cost (USD per 1,000 prompts; log scale)",
        ylabel="Success rate (%)",
        title=f"Router v2 — {regime.upper() if regime == 'ood' else 'standard'} exploratory test",
    )
    ax.margins(x=0.25, y=0.17)
    ax.grid(alpha=0.15)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(reports / "figures" / f"router_v2_{regime}_cost_quality.{ext}", dpi=300)
    plt.close(fig)
    rel = pd.read_parquet(output / "reliability.parquet")
    rel = rel[(rel.family == "embedding") & (rel.split == "test")]
    fig, axes = plt.subplots(
        2, 4, figsize=(11, 5.5), sharex=True, sharey=True, layout="constrained"
    )
    for model_id, ax in zip(MODEL_IDS, axes.flat):
        rows = rel[rel.model_id == model_id]
        ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
        ax.plot(rows.mean_probability, rows.observed_success, color="#0072B2", linewidth=1)
        ax.scatter(
            rows.mean_probability,
            rows.observed_success,
            s=12 + 3 * np.sqrt(rows["count"]),
            color="#0072B2",
        )
        ax.set(
            title=NAMES[model_id], xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1]
        )
        ax.grid(alpha=0.15)
    fig.supxlabel("Mean predicted success (10 fixed bins)")
    fig.supylabel("Observed success")
    fig.suptitle(
        f"Router v2 — {regime.upper() if regime == 'ood' else 'standard'} embedding-head reliability"
    )
    for ext in ("png", "pdf"):
        fig.savefig(reports / "figures" / f"router_v2_{regime}_calibration.{ext}", dpi=300)
    plt.close(fig)


def generate_report(root_output, reports):
    root_output, reports = Path(root_output), Path(reports)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "figures").mkdir(exist_ok=True)
    runs = {
        r: json.loads((root_output / r / "results.json").read_text())
        for r in ("standard", "ood")
        if (root_output / r / "results.json").exists()
    }
    if not runs:
        return
    standard = runs["standard"]
    primary = standard["comparison"]["primary"]
    verdict = (
        "The frozen v2 standard router meets all four preregistered exploratory criteria: at least 10% savings, micro and macro noninferiority bounds, and the per-dataset point guard."
        if primary["promising_exploratory"]
        else "The frozen v2 standard router does not meet all four preregistered exploratory criteria. The failed conditions are reported below; no test policy replaces the validation choice."
    )
    lines = [
        "# Router v2: local semantic features and conservative routing",
        "",
        verdict,
        "",
        f"Standard primary: **{pct(primary['quality'])} success**, **${primary['cost_usd']:.8f}/prompt**, **{pct(primary['cost_savings'])} savings**, quality change **{100 * primary['quality_delta']:+.3f} pp** versus validation-selected best-single.",
        "",
        "These are **exploratory results on already-inspected benchmark tests**, not fresh external confirmation. The v2 protocol was written before supervised v2 fitting; representation, C, policy and novelty choices use training/validation only. All v1 results remain unchanged. [Frozen v2 protocol](router_v2_experiment_spec.md).",
        "",
        "The immutable local encoder is [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. It produces 384-dimensional vectors. Long prompts are encoded in complete 510-content-token chunks and combined by a token-weighted mean. No prompt is truncated; no response, label, cost or dataset identifier enters encoder input. The frozen encoder/cache is shared across regimes, while every learned head, scaler, cost estimate and novelty reference set is independent.",
        "",
    ]
    for regime, r in runs.items():
        p = r["comparison"]["primary"]
        ci = p["bootstrap"]["ci_95"]
        b = r["baselines"]
        lines += [
            f"## {'Standard' if regime == 'standard' else 'Independent OOD'} test",
            "",
            f"N={r['split_counts']['test']:,}; frozen candidate **`{r['primary']['candidate_id']}`**. Best-single **{r['best_single']['model_id']}**, selected from validation.",
            "",
            md_table(
                ["Policy", "Success", "Mean cost USD"],
                [["Frozen primary", pct(p["quality"]), f"${p['cost_usd']:.8f}"]]
                + [
                    [name, pct(b[pid]["quality"]), f"${b[pid]['mean_cost_usd']:.8f}"]
                    for pid, name in [
                        ("best_single", "Best single"),
                        ("always_cheapest", "Always cheapest"),
                        ("oracle", "Oracle (hindsight)"),
                    ]
                ],
            ),
            "",
            f"Quality difference {100 * p['quality_delta']:+.3f} pp; 95% paired CI [{ci['quality_delta'][0] * 100:+.3f}, {ci['quality_delta'][1] * 100:+.3f}] pp. Cost savings {pct(p['cost_savings'])}; 95% CI [{pct(ci['cost_savings'][0])}, {pct(ci['cost_savings'][1])}].",
            "",
            f"Equal-dataset macro quality {pct(p['macro_quality'])}; macro difference {100 * p['macro_quality_delta']:+.3f} pp, CI [{ci['macro_quality_delta'][0] * 100:+.3f}, {ci['macro_quality_delta'][1] * 100:+.3f}] pp. Oracle gap recovered {pct(p['oracle_gap_recovered'])}; remaining gap {100 * p['remaining_oracle_gap']:.3f} pp. Oracle uses hindsight success and realized costs, with a charged training-cheapest fallback when unsolved; it is not attainable deployment performance.",
            "",
            md_table(
                ["Preregistered condition", "Pass"],
                [[k.replace("_", " "), v] for k, v in p["conditions"].items()],
            ),
            "",
            f"Primary is on the descriptive test Pareto frontier: **{r['primary_on_pareto_frontier']}**. Novelty gate rejection rate: **{pct(r['gate_rejection_fraction'])}**. Rejected prompts use best-single; zero savings from a static fallback do not count as routing value.",
            "",
            "### Validation selection audit",
            "",
            f"Joint max-t critical value {r['selection_metadata']['critical_value']:.4f}; {r['selection_metadata']['feasible_count']} feasible deployment IDs out of {r['selection_metadata']['candidate_count']}. The bootstrap includes {r['selection_metadata']['anchor_candidate_count']} candidate/anchor definitions before action deduplication, all eight static comparators, and both micro/macro metrics.",
            "",
            md_table(
                ["Selected policy on validation", "Value"],
                [
                    [key, r["validation_primary"][key]]
                    for key in [
                        "quality",
                        "mean_cost_usd",
                        "micro_delta",
                        "macro_delta",
                        "micro_lower",
                        "macro_lower",
                        "worst_dataset_delta",
                    ]
                ],
            ),
            "",
            "### Dataset outcomes",
            "",
            md_table(
                ["Dataset", "N", "Primary", "Best single", "Δ pp", "Primary cost", "Best cost"],
                [
                    [
                        DATASETS[d],
                        x["primary"]["n"],
                        pct(x["primary"]["quality"]),
                        pct(x["best_single"]["quality"]),
                        f"{100 * (x['primary']['quality'] - x['best_single']['quality']):+.3f}",
                        f"${x['primary']['mean_cost_usd']:.8f}",
                        f"${x['best_single']['mean_cost_usd']:.8f}",
                    ]
                    for d, x in r["datasets"].items()
                ],
            ),
            "",
            "### Representation controls",
            "",
            "Each control uses the original v1 minimum-cost validation selector over its 29 ungated policies. These isolate feature changes; none can replace the primary using test performance.",
            "",
            md_table(
                [
                    "Features",
                    "C",
                    "Control policy",
                    "Quality",
                    "Cost USD",
                    "Savings",
                    "Δ pp",
                    "95% Δ CI pp",
                ],
                [
                    [
                        f,
                        r["families"][f]["selected_C"],
                        r["controls"][f]["policy_id"],
                        pct(r["comparison"]["control_" + f]["quality"]),
                        f"${r['comparison']['control_' + f]['cost_usd']:.8f}",
                        pct(r["comparison"]["control_" + f]["cost_savings"]),
                        f"{100 * r['comparison']['control_' + f]['quality_delta']:+.3f}",
                        ", ".join(
                            f"{100 * x:+.3f}"
                            for x in r["comparison"]["control_" + f]["bootstrap"]["ci_95"][
                                "quality_delta"
                            ]
                        ),
                    ]
                    for f in FAMILIES
                ],
            ),
            "",
            "### Classifier diagnostics",
            "",
            "Macro below means an equal average over eight predictors, not dataset macro. PR AUC is average precision; accuracy uses p ≥ 0.5. Full per-model/train/validation/test and reliability tables are saved separately.",
            "",
            md_table(
                ["Features", "ROC AUC", "PR AUC", "Log loss", "Brier", "ECE", "Accuracy"],
                [
                    [f]
                    + [
                        num(
                            next(
                                x
                                for x in r["test_classifier_metrics"]
                                if x["family"] == f and x["model_id"] == "__macro__"
                            )[metric]
                        )
                        for metric in [
                            "roc_auc",
                            "pr_auc",
                            "log_loss",
                            "brier",
                            "ece",
                            "accuracy_at_0_5",
                        ]
                    ]
                    for f in FAMILIES
                ],
            ),
            "",
            "### Selected-model behavior",
            "",
            md_table(
                ["Model", "Count", "Routed", "Mean prediction", "Actual success", "Cost USD"],
                [
                    [
                        NAMES[x["model_id"]],
                        x["count"],
                        pct(x["fraction"]),
                        pct(x["mean_predicted_success"]),
                        pct(x["actual_success"]),
                        "—" if x["mean_cost_usd"] is None else f"${x['mean_cost_usd']:.8f}",
                    ]
                    for x in r["selection_distribution"]
                ],
            ),
            "",
            md_table(
                ["Dataset"] + [NAMES[m] for m in MODEL_IDS],
                [
                    [DATASETS[x["dataset"]]] + [x[m] for m in MODEL_IDS]
                    for x in r["dataset_selection_counts"]
                ],
            ),
            "",
            f"Prediction values above use the `{r['primary_prediction_reference_family']}` heads; if primary is static fallback these probabilities are diagnostic references, not decision inputs.",
            "",
            f"![{regime} cost quality](figures/router_v2_{regime}_cost_quality.png)",
            "",
            f"![{regime} embedding calibration](figures/router_v2_{regime}_calibration.png)",
            "",
        ]
        plot_results(root_output / regime, reports, regime)
    if "ood" in runs:
        a = pd.read_parquet(root_output / "standard/test_evaluation.parquet")
        b = pd.read_parquet(root_output / "ood/test_evaluation.parquet")
        a = a[(a.policy_id == "primary") & (a.dataset == "livecodebench")]
        b = b[b.policy_id == "primary"]
        common = b[b.prompt_id.isin(a.prompt_id)]
        comparison = dict(
            standard_code_n=len(a),
            standard_code_quality=float(a.success.mean()),
            ood_code_n=len(b),
            ood_code_quality=float(b.success.mean()),
            ood_minus_standard_pp=float(100 * (b.success.mean() - a.success.mean())),
            common_n=len(common),
            ood_common_quality=float(common.success.mean()),
            common_delta_pp=float(100 * (common.success.mean() - a.success.mean())),
        )
        lines += [
            "## Code-domain generalization",
            "",
            f"Standard primary: {pct(comparison['standard_code_quality'])} on {len(a)} code prompts. Independent OOD primary: {pct(comparison['ood_code_quality'])} on {len(b)} prompts ({comparison['ood_minus_standard_pp']:+.3f} pp). On the shared {len(common)} prompts, OOD quality is {pct(comparison['ood_common_quality'])}, change {comparison['common_delta_pp']:+.3f} pp. The full-test comparison changes the sample and training distribution; it is not a causal estimate. OOD training contains no code prompts, and OOD macro equals micro because its test has one dataset.",
            "",
        ]
    else:
        comparison = None
    lines += [
        "## What changed and what remains unresolved",
        "",
        "The useful advance came from the decision rule, not the new encoder. Validation selected TF-IDF in both regimes. Standard TF-IDF has better held-out diagnostic log loss than either semantic family, and the original-selector embedding controls still lose quality. The fixed BGE encoder therefore does not justify replacing the lexical baseline in this experiment.",
        "",
        "The standard comparative policy sends 65.59% of prompts to GPT-5 and 31.11% to Qwen, compared with 12.64% and 81.00% in v1. It keeps GPT-5 for every AIME and LiveMathBench prompt, 155/158 code prompts, and most science/knowledge prompts. Most savings and quality gains come from SimpleQA. The macro gain is positive because other domains largely retain best-single performance, not because gains are distributed evenly across tasks.",
        "",
        "The stricter standard dataset guard fails by one code success: 141/158 versus GPT-5's 142/158, a −0.633 pp difference. All other datasets match or exceed their best-single point quality. This failure remains part of the primary result; the margin was not relaxed and no alternative test policy was substituted. Micro and macro paired intervals nevertheless support improvement under the original core cost/quality criterion.",
        "",
    ]
    if "ood" in runs:
        lines += [
            "OOD routes 1,030/1,055 prompts to GPT-5 and 25 to Qwen. Its quality-difference interval stays within the original 0.5 pp noninferiority margin, but 1.13% savings fall well below the preregistered 10% threshold for meaningful routing value. Thus the earlier catastrophic code downgrade is avoided, while substantial cost-saving generalization remains unproven. Neither frozen primary selected a novelty gate: this improvement is due to comparative probabilities and conservative validation selection, not demonstrated novelty detection.",
            "",
        ]
    elapsed = standard["embedding_cache_metadata"]["inference_seconds"]
    lines += [
        f"The frozen encoder processed all 8,706 prompts locally in {elapsed:.2f} measured inference seconds ({elapsed * 1000 / 8706:.2f} ms per prompt on average, excluding model loading). Neither selected primary needs that encoder at runtime. These are batch CPU timings, not provider latency or an invented dollar charge.",
        "",
        "The next proper step is independent confirmation of a frozen policy, followed by a separately specified investigation of domain-level risk and prompt-dependent cost. Repeatedly selecting policies from these already-inspected test curves would not provide that confirmation. The current campaign provides meaningful exploratory standard cost/quality evidence and honest OOD limits; it does not meet every stricter campaign criterion.",
        "",
        "## Local inference",
        "",
        "The saved primary is usable on new text through a label-free, offline command. It returns all eight predicted success probabilities, the selected benchmark model ID, and the training-derived estimated cost; it does not call that model.",
        "",
        "```bash",
        ".venv/bin/python experiments/route_local.py --prompt 'What is the capital of France?'",
        ".venv/bin/python experiments/route_local.py --artifact-dir artifacts/router_v2/ood --input requests.jsonl",
        "```",
        "",
        "JSONL inputs contain `prompt_id` and `prompt`. The command verifies frozen model/selection hashes, ignores response and dataset metadata, and blocks networking. Tests reproduce every saved standard/OOD primary decision and probability using only the original prompt text.",
        "",
        "## Limits, provenance, and reproduction",
        "",
        "The protocol addresses two distinct v1 failures: absolute probability thresholds can downgrade models with higher expected success, and a micro-only point constraint can hide minority-domain losses. Comparative policies require predicted advantage over best-single; joint validation bounds and the per-dataset point guard reject uncertain tradeoffs. The feature controls distinguish representation improvements from selection changes. A novelty threshold can only provide an empirical deferral heuristic; it does not prove safety under arbitrary distribution shift.",
        "",
        "Selection uses a common 2,000-replicate whole-group bootstrap within dataset strata, with a maximum studentized centered error over policy/anchor/comparator/metric combinations. These approximate simultaneous bounds account for the predefined validation search, but are not distribution-free finite-sample guarantees. Small-domain and zero-variance empirical comparisons can still be optimistic. Test intervals are ordinary paired percentile intervals for the frozen primary and controls, conditional on training and selection. Earlier test inspection and unknown encoder pretraining contamination prevent calling this external confirmation.",
        "",
        "All fitted heads, scalers, cost estimates and novelty references are independent across regimes. The semantic training/CV path is replayed exactly in each regime; saved model reloads reproduce training predictions. Encoder file hashes match the pinned Hugging Face Git/LFS objects; embedding caches bind IDs and full prompt text. The campaign stores protocol/source snapshots before supervised training, source data hashes, package versions, fitted models, CV assignments/scores, predictions, validation bounds, frozen selection hashes, test action matrices/outcomes, calibration, frontier and bootstrap samples. The original processed files and v1 outputs are unchanged.",
        "",
        "```bash",
        ".venv/bin/python -m pip install -r experiments/requirements-router-v2.txt",
        ".venv/bin/python -m routing_ml.embeddings --download",
        ".venv/bin/python -m routing_ml.embeddings",
        ".venv/bin/python experiments/embedding_logreg_router.py",
        ".venv/bin/python tests/run_offline_suite.py",
        "```",
        "",
        "Only the explicit download step accesses public model files; inference and fitting run with networking blocked. Existing completed runs are protected from overwrite. Use `--report-only` to regenerate reports from saved outcomes. To replay fitting and evaluation using the verified cache while preserving completed runs:",
        "",
        "```bash",
        "router_replay_dir=$(mktemp -d /tmp/router-v2-replay.XXXXXX)",
        'ln -s "$PWD/artifacts/router_v2/embeddings" "$router_replay_dir/embeddings"',
        '.venv/bin/python experiments/embedding_logreg_router.py --output-dir "$router_replay_dir" --reports-dir "$router_replay_dir/reports"',
        "```",
        "",
    ]
    domain_path = reports / "router_v2_domain_diagnostic.json"
    if domain_path.exists():
        diagnostic = json.loads(domain_path.read_text())
        d = diagnostic["regimes"]["standard"]
        rows = [x for x in d["metrics"] if x["dataset"] == "__micro__"]
        ci = d["bootstrap"]["ci_95"]["quality_delta"]
        lines += [
            "## Post-hoc diagnostic: privileged domain-static reference",
            "",
            "This additional diagnostic selects the highest-quality static model separately within each validation dataset, breaking ties by training mean cost and model ID. It is explicitly privileged: it requires the dataset name at inference. Unseen domains fall back to global validation best-single. It was not a v2 selection candidate and did not modify either primary. [Diagnostic artifact](router_v2_domain_diagnostic.json).",
            "",
            md_table(
                ["Standard reference", "Success", "Mean cost USD"],
                [[x["policy_id"], pct(x["quality"]), f"${x['mean_cost_usd']:.8f}"] for x in rows],
            ),
            "",
            f"The learned primary improves quality by {100 * d['primary_minus_domain_quality']:+.3f} pp, with an exploratory paired 95% CI [{ci[0] * 100:+.3f}, {ci[1] * 100:+.3f}] pp, but costs {-100 * d['primary_cost_savings_vs_domain']:.3f}% more. It does not dominate this privileged reference. The reference selects Qwen on AIME and DeepSeek V3 on LiveMathBench after ties in their small validation strata; those choices lose quality on test. The learned primary's stronger-model fallback avoids these losses. This comparison exposes both domain-composition effects and the instability of small-domain static selection; it is not evidence of balanced improvements across all tasks.",
            "",
            "Reproduce this diagnostic with `.venv/bin/python experiments/router_domain_diagnostic.py`. Its bootstrap is conditional on fixed validation choices and is secondary, not an additional confirmatory hypothesis test.",
            "",
        ]
    test_path = reports / "router_v2_tests.json"
    if test_path.exists():
        tests = json.loads(test_path.read_text())
        lines += [
            f"Full offline suite: **{tests['tests_run']} tests; {tests['failures']} failures, {tests['errors']} errors, {tests['skipped']} skipped; {tests['unexpected_network_attempts']} network attempts**. [Test summary](router_v2_tests.json).",
            "",
        ]
    (reports / "router_v2_results.md").write_text("\n".join(lines))
    (reports / "router_v2_results.json").write_text(
        json.dumps(dict(runs=runs, code_comparison=comparison), indent=2, sort_keys=True) + "\n"
    )
