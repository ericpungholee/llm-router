"""Reports for the fixed-reference experiment; never chooses a test policy."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from routing_ml.reporting import md_table

DOMAIN_NAMES = dict(
    aime="AIME",
    gpqa="GPQA",
    livecodebench="LiveCodeBench",
    livemathbench="LiveMathBench",
    mmlupro="MMLU-Pro",
    simpleqa="SimpleQA",
)


def pct(x):
    return f"{100 * x:.3f}%"


def pp(x):
    return f"{100 * x:+.3f}"


def ci(values, percent=False):
    f = pct if percent else pp
    return f"[{f(values[0])}, {f(values[1])}]"


def figures(output, reports, report):
    directory = reports / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
            "savefig.dpi": 250,
        }
    )
    for regime in ("standard", "ood"):
        r = report["regimes"][regime]
        table = pd.read_parquet(output / regime / "pareto_table.parquet").set_index("policy_id")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        ax = axes[0]
        static = table[table.index.str.startswith("static_")]
        grid = table[table.index.str.startswith("advantage_")]
        ax.scatter(
            static.mean_cost_usd,
            static.quality * 100,
            marker="s",
            color=".65",
            s=28,
            label="Eight static models",
        )
        ax.scatter(
            grid.mean_cost_usd, grid.quality * 100, color="#2166ac", s=34, label="Fixed margin grid"
        )
        for pid, label, color, marker in (
            ("fixed_gpt5", "Fixed GPT-5", "black", "D"),
            ("primary", "Frozen primary", "#b2182b", "*"),
            ("oracle", "Hindsight oracle", "#238b45", "P"),
        ):
            row = table.loc[pid]
            ax.scatter(
                row.mean_cost_usd,
                row.quality * 100,
                color=color,
                marker=marker,
                s=135 if pid == "primary" else 60,
                label=label,
                zorder=5,
            )
        ax.set_xscale("log")
        ax.set(
            xlabel="Mean recorded cost (USD/prompt; log scale)",
            ylabel="Success (%)",
            title=regime.upper() + " test cost / quality",
        )
        ax.legend(fontsize=8, loc="best")
        d = r["per_dataset"]
        y = np.arange(len(d))
        quality = np.array([x["quality_delta"] * 100 for x in d])
        intervals = np.array([x["bootstrap"]["ci_95"]["quality_delta"] for x in d]) * 100
        axes[1].errorbar(
            quality,
            y,
            xerr=np.maximum(0, np.vstack([quality - intervals[:, 0], intervals[:, 1] - quality])),
            fmt="o",
            color="#2166ac",
            capsize=3,
        )
        axes[1].set_yticks(y, [f"{DOMAIN_NAMES[x['dataset']]} (n={x['n']})" for x in d])
        axes[1].invert_yaxis()
        axes[1].axvline(0, color=".7", lw=1)
        axes[1].axvline(-0.5, color="#b2182b", lw=1, ls="--")
        axes[1].set(
            xlabel="Primary − fixed GPT-5 quality (pp)", title="Paired descriptive 95% intervals"
        )
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"router_v4_{regime}_cost_quality.{extension}")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for ax, regime in zip(axes, ("standard", "ood")):
        rel = pd.read_parquet(output / regime / "test_reliability.parquet")
        rel = rel[rel.model_id == "__selected_action__"]
        ax.plot([0, 1], [0, 1], "--", color=".6", lw=1)
        ax.plot(rel.mean_probability, rel.observed_success, "o-", color="#2166ac")
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Predicted selected-model success",
            ylabel="Observed success",
            title=regime.upper(),
        )
    fig.suptitle("Selected-action calibration (diagnostic)")
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"router_v4_calibration.{extension}")
    plt.close(fig)


def generate_report(output, reports):
    output, reports = Path(output), Path(reports)
    result = json.loads((output / "results.json").read_text())
    behavior = {}
    for regime in ("standard", "ood"):
        selected = result["regimes"][regime]["frozen_selection"]["primary"]["candidate_id"]
        actions = pd.read_parquet(output / regime / "source_holdout_actions.parquet").set_index(
            "prompt_id"
        )[selected]
        evidence = pd.read_parquet(output / regime / "source_holdout_validation_evaluation.parquet")
        routed = evidence[evidence.policy_id == selected].set_index("prompt_id")
        reference = (
            evidence[evidence.policy_id == "fixed_gpt5"].set_index("prompt_id").loc[routed.index]
        )
        records = []
        for domain in ["__micro__"] + sorted(routed.dataset.unique()):
            rows = routed if domain == "__micro__" else routed[routed.dataset == domain]
            records.append(
                dict(
                    dataset=domain,
                    n=len(rows),
                    gpt5_fraction=float(actions.loc[rows.index].eq(6).mean()),
                    cost_savings=float(
                        1 - rows.cost_usd.mean() / reference.loc[rows.index].cost_usd.mean()
                    ),
                )
            )
        behavior[regime] = records
    result["source_validation_behavior"] = behavior
    reports.mkdir(parents=True, exist_ok=True)
    figures(output, reports, result)
    lines = ["# Router v4: fixed GPT-5 reference and source-domain validation", ""]
    for regime in ("standard", "ood"):
        r = result["regimes"][regime]
        primary = r["comparisons"]["primary"]
        passed = primary["promising_exploratory"]
        lines += [
            f"**{regime.upper()}:** frozen primary `{r['frozen_selection']['primary']['candidate_id']}` achieves "
            f"**{pct(primary['quality'])} quality**, **{pct(primary['cost_savings'])} cost savings**, and "
            f"**{pp(primary['quality_delta'])} pp** versus fixed GPT-5. "
            f"Meets all four declared meaningful-routing criteria: **{passed}**.",
            "",
        ]
    lines += [
        "These are exploratory results on already-inspected benchmark outcomes. Fixing the reference prevents "
        "source validation from weakening the quality target, but does not establish independent confirmation. "
        "The source-domain checks affect validation selection only. No test point replaces either frozen primary.",
        "",
        "[Protocol written before fitting](router_v4_experiment_spec.md). V1, v2, v3 and the processed data remain unchanged. "
        "The reference is the benchmark model `gpt-5`; no GPT-5.6 Sol outcomes are present.",
        "",
    ]
    standard = result["regimes"]["standard"]
    lines += [
        "The standard result clears the previous observed code-quality loss by selecting a more conservative "
        "margin from validation evidence. Its 0.10 margin retains GPT-5 for all standard test math, science "
        "and code prompts. All 45 additional successes occur in SimpleQA; removing SimpleQA leaves "
        f"{pct(standard['composition_checks'][0]['cost_savings'])} savings. This supports a QA-focused cost "
        "reduction, not broad domain-independent savings.",
        "",
        "OOD still fails the declared 10% savings criterion. Its 0.05 margin gives exactly the same code "
        "decisions as v2's OOD primary: 1,030 GPT-5 and 25 Qwen. The stable reference removes v3's "
        "source-validation reference failure, but this campaign does not improve cost-saving transfer to code.",
        "",
    ]
    for regime in ("standard", "ood"):
        r = result["regimes"][regime]
        frozen = r["frozen_selection"]
        p = r["comparisons"]["primary"]
        intervals = p["bootstrap"]["ci_95"]
        candidates = pd.read_parquet(output / regime / "validation_policy_grid.parquet")
        bounds = pd.read_parquet(output / regime / "validation_quality_bounds.parquet")
        worst = bounds.loc[bounds.groupby("policy_id").joint_lower.idxmin()].sort_values(
            "policy_id"
        )
        lines += [
            f"## {regime.upper()} test",
            "",
            f"C=1; primary `{frozen['primary']['candidate_id']}`. Fixed reference: GPT-5. "
            f"Validation-selected best-single (secondary reference): `{frozen['validation_best_single']['model_id']}`.",
            "",
            md_table(
                ["Policy", "Quality", "Cost USD/prompt", "Dataset macro quality"],
                [
                    [
                        pid,
                        pct(r["micro"][pid]["quality"]),
                        f"${r['micro'][pid]['mean_cost_usd']:.8f}",
                        pct(r["macro"][pid]["quality"]),
                    ]
                    for pid in (
                        "primary",
                        "fixed_gpt5",
                        "validation_best_single",
                        "always_cheapest",
                        "oracle",
                    )
                ],
            ),
            "",
            f"Quality delta **{pp(p['quality_delta'])} pp**, paired 95% CI **{ci(intervals['quality_delta'])} pp**. "
            f"Cost savings **{pct(p['cost_savings'])}**, CI **{ci(intervals['cost_savings'], True)}**.",
            "",
            f"Equal-dataset macro quality delta {pp(p['macro_quality_delta'])} pp, CI {ci(intervals['macro_quality_delta'])} pp. "
            f"Macro cost savings {pct(1 - r['macro']['primary']['mean_cost_usd'] / r['macro']['fixed_gpt5']['mean_cost_usd'])}, "
            f"CI {ci(intervals['macro_cost_savings'], True)}.",
            "",
            md_table(
                ["Declared criterion", "Pass"], [[k, str(v)] for k, v in p["conditions"].items()]
            ),
            "",
            f"Additional check: every dataset's individual test interval lower bound is at least −0.5 pp: "
            f"**{p['all_dataset_interval_margins_pass']}**. The declared dataset criterion above uses point differences; "
            "these two conditions are reported separately.",
            "",
            "### Frozen controls",
            "",
            md_table(
                [
                    "Policy",
                    "Validation choice",
                    "Quality",
                    "Cost USD",
                    "Savings",
                    "Δ pp",
                    "95% Δ CI pp",
                ],
                [
                    [
                        name,
                        frozen["primary"]["candidate_id"]
                        if name == "primary"
                        else frozen["controls"][
                            "ordinary_only" if name == "ordinary_only_control" else "fixed_margin"
                        ],
                        pct(row["quality"]),
                        f"${row['cost_usd']:.8f}",
                        pct(row["cost_savings"]),
                        pp(row["quality_delta"]),
                        ci(row["bootstrap"]["ci_95"]["quality_delta"]),
                    ]
                    for name, row in r["comparisons"].items()
                ],
            ),
            "",
            "### Validation selection audit",
            "",
            f"Joint bootstrap critical value: {frozen['selection_metadata']['critical_value']:.4f}; "
            f"ordinary-validation-only critical value: {frozen['selection_metadata']['ordinary_only_critical_value']:.4f}. "
            f"{frozen['selection_metadata']['feasible_count']}/6 candidates pass all ordinary/source-domain bounds.",
            "",
            md_table(
                [
                    "Candidate",
                    "Validation quality",
                    "Cost USD",
                    "Worst joint lower pp",
                    "Joint feasible",
                    "Ordinary-only feasible",
                ],
                [
                    [
                        row.policy_id,
                        pct(row.quality),
                        f"${row.mean_cost_usd:.8f}",
                        pp(row.worst_joint_lower),
                        str(row.feasible),
                        str(row.ordinary_only_feasible),
                    ]
                    for row in candidates.itertuples()
                ],
            ),
            "",
            md_table(
                [
                    "Candidate",
                    "Most limiting block",
                    "Metric/domain",
                    "Observed Δ pp",
                    "Joint lower pp",
                ],
                [
                    [
                        row.policy_id,
                        row.block,
                        row.metric,
                        pp(row.quality_delta),
                        pp(row.joint_lower),
                    ]
                    for row in worst.itertuples()
                ],
            ),
            "",
            "The selected primary's behavior on domains excluded from auxiliary training is shown below. "
            "Near-total fallback explains zero observed quality-difference variance; these checks do not "
            "demonstrate cost-saving specialization on unseen domains.",
            "",
            md_table(
                ["Source holdout", "N", "Routed to GPT-5", "Cost savings"],
                [
                    [
                        DOMAIN_NAMES.get(row["dataset"], "All source holdouts"),
                        row["n"],
                        pct(row["gpt5_fraction"]),
                        pct(row["cost_savings"]),
                    ]
                    for row in behavior[regime]
                ],
            ),
            "",
            "### Dataset outcomes",
            "",
            md_table(
                ["Dataset", "N", "Primary quality", "Δ vs GPT-5 pp", "95% Δ CI pp", "Savings"],
                [
                    [
                        DOMAIN_NAMES[d["dataset"]],
                        d["n"],
                        pct(d["quality"]),
                        pp(d["quality_delta"]),
                        ci(d["bootstrap"]["ci_95"]["quality_delta"]),
                        pct(d["cost_savings"]),
                    ]
                    for d in r["per_dataset"]
                ],
            ),
            "",
            md_table(
                ["Excluded from micro evaluation", "N remaining", "Δ pp", "Savings"],
                [
                    [
                        ", ".join(DOMAIN_NAMES[d] for d in row["excluded"]),
                        row["n"],
                        pp(row["quality_delta"]),
                        pct(row["cost_savings"]),
                    ]
                    for row in r["composition_checks"]
                ],
            ),
            "",
            "### Selected-model behavior",
            "",
            md_table(
                ["Model", "N", "Share", "Mean prediction", "Actual success", "Mean cost USD"],
                [
                    [
                        row["model_id"],
                        row["count"],
                        pct(row["fraction"]),
                        "—"
                        if row["mean_predicted_success"] is None
                        else pct(row["mean_predicted_success"]),
                        "—" if row["actual_success"] is None else pct(row["actual_success"]),
                        "—" if row["mean_cost_usd"] is None else f"${row['mean_cost_usd']:.8f}",
                    ]
                    for row in r["distribution"]
                ],
            ),
            "",
            "### Predictor diagnostics",
            "",
            md_table(
                [
                    "Predictor",
                    "Prevalence",
                    "ROC AUC",
                    "PR AUC",
                    "Log loss",
                    "Brier",
                    "ECE",
                    "Accuracy",
                ],
                [
                    [row["model_id"]]
                    + [
                        "—" if row[k] is None else f"{row[k]:.4f}"
                        for k in (
                            "prevalence",
                            "roc_auc",
                            "pr_auc",
                            "log_loss",
                            "brier",
                            "ece",
                            "accuracy_at_0_5",
                        )
                    ]
                    for row in r["classifier_metrics"]
                ],
            ),
            "",
        ]
        recovered = (
            "undefined" if p["oracle_gap_recovered"] is None else pct(p["oracle_gap_recovered"])
        )
        lines += [
            f"Primary is on the descriptive test Pareto frontier among this fixed grid and static references: "
            f"**{r['primary_on_pareto']}**. Oracle gap recovered: {recovered}; remaining gap: {100 * p['remaining_oracle_gap']:.3f} pp. "
            "The oracle uses realized outcomes/costs and is not achievable performance.",
            "",
            f"![{regime} cost quality and domain differences](figures/router_v4_{regime}_cost_quality.png)",
            "",
        ]
    m = result["matched_code"]
    lines += [
        "## Matched code comparison",
        "",
        f"On the shared {m['shared_n']} code test prompts, standard quality is {pct(m['standard_quality'])} and "
        f"OOD quality is {pct(m['ood_shared_quality'])}. OOD − standard is {pp(m['quality_delta_shared_ood_minus_standard'])} pp "
        f"with paired CI {ci(m['bootstrap']['ci_95']['quality_delta'])} pp. Full OOD quality over 1,055 prompts is "
        f"{pct(m['ood_full_quality'])}. Training domains differ; the full standard and OOD code test populations also differ.",
        "",
        "![Selected-action calibration](figures/router_v4_calibration.png)",
        "",
        "## What this experiment establishes",
        "",
        "The quality target is stable: every routing rule falls back to the same predefined GPT-5 reference. "
        "The ordinary-only and fixed-margin controls distinguish this reference choice from the effect of "
        "requiring validation evidence on source domains excluded from auxiliary training. A static fallback "
        "has zero routing savings and cannot count as a successful learned router.",
        "",
        "Selection bounds cover all six policies, ordinary validation, held-out source validation, micro/macro "
        "quality, and each dataset. These are approximate conditional bootstrap bounds. Auxiliary training sets "
        "overlap; their shared fitting uncertainty is not estimated. Zero observed variance gives zero empirical "
        "width, not a guarantee about unseen failures. Test intervals also condition on fixed fits/selection. "
        "The benchmark has already influenced earlier research choices, so passing numerical checks is exploratory.",
        "",
        "Source-domain validation can reject a useful tradeoff when the model cannot generalize that tradeoff "
        "to a withheld source domain. This is an intended test of transfer evidence, not permission to remove a "
        "failed domain or weaken a constraint after seeing outcomes. The per-domain and composition tables show "
        "whether savings are distributed across tasks or concentrated in knowledge QA.",
        "",
        "Further tuning on these same tests cannot supply independent confirmation. The next claim about "
        "deployment should use unseen complete outcomes, with the reference and evaluation criteria fixed first. "
        "No new data acquisition, paid evaluation, provider work, or additional model search occurs here.",
        "",
        "## Reproduction and local inference",
        "",
        "The final and auxiliary fits are independent, training-only TF-IDF/logistic models with C=1. "
        "Final fits replay exactly in each regime; saved models reproduce predictions. Training costs, memberships, "
        "vocabularies, candidate ordering, validation grids/bounds, policy choices, all predictions/actions, "
        "separate outcomes, bootstrap samples, diagnostics, source snapshots and hashes are retained under "
        "`artifacts/router_v4/`. Original processed-data and v1/v2/v3 report hashes remain unchanged.",
        "",
        "Run into an unused directory to preserve completed experiments:",
        "",
        "```bash",
        "router_run_dir=$(mktemp -d /tmp/router-v4-replay.XXXXXX)",
        '.venv/bin/python experiments/fixed_reference_router.py --output-dir "$router_run_dir" --reports-dir "$router_run_dir/reports"',
        ".venv/bin/python tests/run_offline_suite.py",
        "```",
        "",
        "Route new prompt text through the frozen primary locally:",
        "",
        "```bash",
        ".venv/bin/python experiments/route_local.py --artifact-dir artifacts/router_v4/standard --prompt 'What is the capital of France?'",
        ".venv/bin/python experiments/route_local.py --artifact-dir artifacts/router_v4/ood --prompt 'Write a function that finds the median of two sorted arrays.'",
        "```",
        "",
        "These commands choose a benchmark model; they do not call it. Regenerate reports using "
        "`.venv/bin/python experiments/fixed_reference_router.py --report-only`.",
        "",
    ]
    tests = reports / "router_v4_tests.json"
    if tests.exists():
        t = json.loads(tests.read_text())
        lines += [
            f"Full offline suite: **{t['tests_run']} tests, {t['failures']} failures, {t['errors']} errors, "
            f"{t['skipped']} skipped; {t['unexpected_network_attempts']} network attempts**. "
            "[Test result](router_v4_tests.json).",
            "",
        ]
    (reports / "router_v4_results.md").write_text("\n".join(lines))
    (reports / "router_v4_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
