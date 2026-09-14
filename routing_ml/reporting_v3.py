"""Compact reports for the fixed cross-fitting check; no policy selection."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from routing_ml.reporting import md_table

NAMES = dict(aime="AIME", gpqa="GPQA", livecodebench="LiveCodeBench", livemathbench="LiveMathBench",
             mmlupro="MMLU-Pro", simpleqa="SimpleQA")


def pc(value):
    return f"{100 * value:.3f}%"


def pp(value):
    return f"{100 * value:+.3f}"


def interval(values, quality=True):
    formatter = pp if quality else pc
    return f"[{formatter(values[0])}, {formatter(values[1])}]"


def figures(report, output, directory):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 130, "savefig.dpi": 250})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    standard = report["regimes"]["standard"]
    domains = standard["per_dataset"]
    x = np.arange(len(domains))
    q = np.array([d["quality_delta"] * 100 for d in domains])
    ci = np.array([d["bootstrap"]["ci_95"]["quality_delta"] for d in domains]) * 100
    axes[0].errorbar(q, x, xerr=np.maximum(0, np.vstack([q - ci[:, 0], ci[:, 1] - q])), fmt="o", color="#2166ac", capsize=3)
    axes[0].set_yticks(x, [f"{NAMES[d['dataset']]} (n={d['n']})" for d in domains])
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="0.6", lw=1)
    axes[0].axvline(-.5, color="#b2182b", ls="--", lw=1, label="−0.5 pp margin")
    axes[0].set_xlabel("Router − fold best-single quality (pp)")
    axes[0].set_title("Standard: all held-out prompts")
    axes[0].legend(fontsize=8, loc="best")
    for regime, color, marker in (("standard", "#2166ac", "o"), ("ood", "#b2182b", "s")):
        r = report["regimes"][regime]
        axes[1].scatter([f["cost_savings"] * 100 for f in r["folds"]], [f["quality_delta"] * 100 for f in r["folds"]],
                        color=color, marker=marker, alpha=.65, s=38, label=f"{regime.upper()} folds")
        axes[1].scatter(100 * r["primary"]["cost_savings"], 100 * r["primary"]["quality_delta"],
                        color=color, marker="*", s=160, edgecolor="black", linewidth=.5, label=f"{regime.upper()} pooled")
    axes[1].axhline(-.5, color="0.6", ls="--", lw=1)
    axes[1].axvline(10, color="0.6", ls="--", lw=1)
    axes[1].set_xlabel("Cost savings versus fold best-single (%)")
    axes[1].set_ylabel("Quality difference (pp)")
    axes[1].set_title("Fixed rule across five folds")
    axes[1].legend(fontsize=8)
    fig.savefig(directory / "router_v3_robustness.png")
    fig.savefig(directory / "router_v3_robustness.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for ax, regime in zip(axes, ("standard", "ood")):
        reliability = pd.read_parquet(output / regime / "reliability.parquet")
        r = reliability[reliability.model_id == "__selected_action__"]
        ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1)
        ax.plot(r.mean_probability, r.observed_success, "o-", color="#2166ac")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted selected-model success", ylabel="Observed success",
               title=regime.upper() + " held-out selection calibration")
    fig.savefig(directory / "router_v3_calibration.png")
    fig.savefig(directory / "router_v3_calibration.pdf")
    plt.close(fig)


def generate_report(output, reports):
    output, reports = Path(output), Path(reports)
    report = json.loads((output / "results.json").read_text())
    reports.mkdir(parents=True, exist_ok=True)
    figure_dir = reports / "figures"
    figure_dir.mkdir(exist_ok=True)
    figures(report, output, figure_dir)
    lines = ["# Router v3: fixed TF-IDF robustness check", "",
             "This is a robustness check on previously inspected data, **not independent confirmation**. "
             "Every prompt has one held-out decision; no C, margin, feature, seed, or policy search was performed.", ""]
    for regime in ("standard", "ood"):
        r = report["regimes"][regime]
        p = r["primary"]
        lines += [f"**{regime.upper()}:** {pc(r['micro']['router']['quality'])} quality, "
                  f"{pc(p['cost_savings'])} cost savings, {pp(p['quality_delta'])} pp versus fold-validation best-single. "
                  f"All four descriptive checks pass: **{r['all_descriptive_checks_pass']}**.", ""]
    standard = report["regimes"]["standard"]
    ood = report["regimes"]["ood"]
    lines += ["**The standard gain survives this check; OOD routing does not.** "
              "All five non-code validation sets select Qwen as best-single. Because it is also training-cheapest, "
              "the comparative rule has no cheaper eligible model and sends every OOD prompt to Qwen. "
              f"Its {pc(ood['micro']['router']['quality'])} code quality is far below fixed GPT-5's "
              f"{pc(ood['micro']['static_gpt-5']['quality'])}. Equality with the selected Qwen reference "
              "is a trivial equality, not evidence of safe transfer or useful learned routing.", "",
              "Standard gains remain concentrated in SimpleQA. Removing SimpleQA leaves "
              f"{pc(standard['composition_checks'][0]['cost_savings'])} savings and "
              f"{pp(standard['composition_checks'][0]['quality_delta'])} pp quality change. "
              "The result does not establish substantial cost reductions across domains.", ""]
    mmlu = next(d for d in standard['per_dataset'] if d['dataset'] == 'mmlupro')
    lines += [f"MMLU-Pro quality decreases by {pp(mmlu['quality_delta'])} pp, with descriptive CI "
              f"{interval(mmlu['bootstrap']['ci_95']['quality_delta'])} pp. This meets the frozen dataset "
              "point guard but its interval extends below −0.5 pp. Passing the aggregate checks does not "
              "establish noninferiority independently in every domain.", ""]
    lines += ["The comparison evaluates the v2 selected decision rules with new, separate grouped partitions. "
              "It keeps the original v1/v2 results and canonical splits unchanged. The ten new systems are evaluation "
              "fits, and do not replace the saved v2 router.", "",
              "[Protocol frozen before v3 fitting](router_v3_robustness_spec.md).", ""]
    for regime in ("standard", "ood"):
        r = report["regimes"][regime]
        p = r["primary"]
        ci = p["bootstrap"]["ci_95"]
        lines += [f"## {regime.upper()} pooled held-out result", "",
                  f"N={r['n']:,}; C={r['fixed_C']:g}; fixed comparative margin={r['margin']:.2f}. "
                  "Best-single is selected independently from each fold's validation set; its pooled result is "
                  "a cross-fitted selection procedure, not one globally selected model.", "",
                  md_table(["Policy", "Micro quality", "Mean cost USD", "Dataset macro quality"],
                           [[pid, pc(r['micro'][pid]['quality']), f"${r['micro'][pid]['mean_cost_usd']:.8f}", pc(r['macro'][pid]['quality'])]
                            for pid in ("router", "best_single", "static_gpt-5", "always_cheapest", "privileged_domain_static", "oracle")]), "",
                  f"Quality delta **{pp(p['quality_delta'])} pp**, descriptive 95% CI **{interval(ci['quality_delta'])} pp**. "
                  f"Cost savings **{pc(p['cost_savings'])}**, CI **{interval(ci['cost_savings'], False)}**.", "",
                  f"Equal-dataset macro quality delta {pp(r['macro']['router']['quality'] - r['macro']['best_single']['quality'])} pp, "
                  f"CI {interval(ci['macro_quality_delta'])} pp. Macro cost savings "
                  f"{pc(1-r['macro']['router']['mean_cost_usd']/r['macro']['best_single']['mean_cost_usd'])}, "
                  f"CI {interval(ci['macro_cost_savings'], False)}.", "",
                  md_table(["Descriptive condition", "Pass"], [[k, str(v)] for k, v in r['descriptive_checks'].items()]), "",
                  f"Folds meeting the −0.5 pp point margin: {r['fold_checks']['quality_margin_passes']}/5. "
                  f"Folds saving at least 10%: {r['fold_checks']['savings_10_percent_passes']}/5. "
                  "Folds overlap in training and are not independent replicates.", "",
                  "### Fold stability", "",
                  md_table(["Fold", "Fit / val / held out", "Validation best", "Router quality", "Δ pp", "Savings"],
                           [[f['fold'], ' / '.join(str(f['split_counts'][k]) for k in ('train','validation','test')), f['best_single'],
                             pc(f['router']['quality']), pp(f['quality_delta']), pc(f['cost_savings'])] for f in r['folds']]), "",
                  "### Dataset outcomes", "",
                  md_table(["Dataset", "N", "Router", "Best", "Δ pp", "95% Δ CI pp", "Savings"],
                           [[NAMES[d['dataset']], d['n'], pc(d['router']['quality']), pc(d['best_single']['quality']), pp(d['quality_delta']),
                             interval(d['bootstrap']['ci_95']['quality_delta']), pc(d['cost_savings'])] for d in r['per_dataset']]), "",
                  "### Composition checks", "",
                  md_table(["Excluded datasets", "N remaining", "Quality Δ pp", "Savings"],
                           [[', '.join(NAMES[d] for d in row['excluded']), row['n'], pp(row['quality_delta']), pc(row['cost_savings'])]
                            for row in r['composition_checks']]), "",
                  "### Selected models", "",
                  md_table(["Model", "N", "Share", "Predicted success", "Actual success", "Mean cost USD"],
                           [[row['model_id'], row['count'], pc(row['fraction']),
                             '—' if row['mean_predicted_success'] is None else pc(row['mean_predicted_success']),
                             '—' if row['actual_success'] is None else pc(row['actual_success']),
                             '—' if row['mean_cost_usd'] is None else f"${row['mean_cost_usd']:.8f}"] for row in r['distribution']]), "",
                  "### Predictor diagnostics", "",
                  md_table(["Predictor", "Prevalence", "ROC AUC", "PR AUC", "Log loss", "Brier", "ECE", "Accuracy"],
                           [[row['model_id'], f"{row['prevalence']:.4f}"] + [f"{row[k]:.4f}" if row[k] is not None else '—'
                             for k in ('roc_auc','pr_auc','log_loss','brier','ece','accuracy_at_0_5')] for row in r['classifier_metrics']]), ""]
        dc = r['privileged_reference_comparison']
        recovered = 'undefined' if r['oracle_gap_recovered'] is None else pc(r['oracle_gap_recovered'])
        lines += [f"The router's descriptive Pareto membership against the eight static models and fixed references is "
                  f"**{r['router_on_pareto']}**. This restricted comparison is not the broader v2 policy grid. "
                  f"Oracle gap recovered: {recovered}; remaining quality gap: {100*r['remaining_oracle_gap']:.3f} pp. "
                  "The oracle uses hindsight outcomes and realized costs and is not achievable routing performance.", "",
                  f"Against the privileged domain-static reference, router quality changes by {pp(dc['quality_delta'])} pp "
                  f"(CI {interval(dc['bootstrap']['ci_95']['quality_delta'])} pp), with {pc(dc['cost_savings'])} cost savings. "
                  "Negative savings means the router costs more. Domain-static uses dataset names and remains a secondary reference.", ""]
    m = report['matched_code']
    lines += ["## Matched code generalization", "",
              f"The same {m['n']:,} code prompts receive one standard and one OOD held-out prediction. "
              f"Standard quality is {pc(m['standard_quality'])}; OOD is {pc(m['ood_quality'])}. "
              f"OOD − standard is {pp(m['quality_delta_ood_minus_standard'])} pp, "
              f"descriptive CI {interval(m['bootstrap']['ci_95']['quality_delta'])} pp. "
              f"OOD cost savings relative to standard: {pc(m['cost_savings_ood_vs_standard'])}.", "",
              "Both systems use matched outer partitions, but their training domains and fixed routing margins differ. "
              "This is a paired comparison of two fixed algorithms, not an isolated causal effect of code training.", "",
              "The OOD failure exposes reference-selection risk: conservative downgrading relative to a weak "
              "source-validation reference cannot protect quality on an unseen domain. V2's particular non-code "
              "validation split selected GPT-5, producing 86.351% code quality with 1.129% savings. Here all five "
              "validation choices select Qwen. Neither result is replaced, and no post-hoc switch to GPT-5 is made.", "",
              "![Fold and dataset robustness](figures/router_v3_robustness.png)", "",
              "![Selected-action calibration](figures/router_v3_calibration.png)", "",
              "## Limits and next step", "",
              "These new partitions expand evaluation coverage to all 56 AIME, 117 LiveMathBench, 198 GPQA, "
              "1,055 LiveCodeBench, 2,979 MMLU-Pro and 4,301 SimpleQA prompts. They do not add independent evidence "
              "after earlier benchmark-guided research. The original frozen standard/OOD results remain the v2 results.", "",
              "Each standard fit uses approximately 64% of the benchmark, with 16% validation and 20% evaluation. "
              "OOD removes all code from those fitting and validation partitions and evaluates only held-out code. "
              "These training sets are smaller than the canonical v2 fits. Fold ranges expose sensitivity without "
              "treating overlapping fits as independent experiments.", "",
              "All intervals use 2,000 seed-3407 paired whole-group resamples within dataset strata, conditional on "
              "the fixed fitted models and decisions. They omit refitting uncertainty, cross-fold dependence from "
              "shared training observations, and prior adaptation to this benchmark. Passing the numerical margin "
              "therefore provides descriptive support only. Identical actions yield zero empirical variance, which "
              "does not prove a population bound on unseen failures. No bootstrap retraining or post-result policy selection occurs.", "",
              "The next confirmation must use unseen complete prompt-model outcomes with the router, comparator, "
              "cost accounting, and acceptance criteria frozen beforehand. Within this benchmark, inspect the "
              "domain and composition results rather than searching more margins or seeds until every check passes.", "",
              "Before an OOD deployment claim, specify a separate experiment for reference selection under domain "
              "shift, using only source-domain validation for decisions and an explicit deployment-quality reference. "
              "Preserve an untouched final domain for confirmation. This audit does not implement that next experiment.", "",
              "## Artifacts, checks, and reproduction", "",
              "Every fold saves its model, training-only vocabulary/costs, validation reference choices, label-free "
              "predictions/actions, and separate outcomes. Pooled predictions, evaluation rows, all static points, "
              "calibration data, model-by-dataset counts and 2,000-replicate samples are retained. The campaign "
              "hashes original processed files, prior reports, source snapshots and all output artifacts. Full first-fold "
              "training replay in each regime and saved model reloads match exactly. Networking is blocked.", "",
              "From an unused output directory:", "", "```bash",
              ".venv/bin/python experiments/crossfit_tfidf_router.py",
              ".venv/bin/python tests/run_offline_suite.py", "```", "",
              "To preserve completed artifacts when replaying:", "", "```bash",
              "router_check_dir=$(mktemp -d /tmp/router-v3-replay.XXXXXX)",
              '.venv/bin/python experiments/crossfit_tfidf_router.py --output-dir "$router_check_dir" --reports-dir "$router_check_dir/reports"',
              "```", "",
              "Regenerate this report with `.venv/bin/python experiments/crossfit_tfidf_router.py --report-only`. "
              "The original processed artifacts and v1/v2 reports are unchanged.", ""]
    test_path = reports / "router_v3_tests.json"
    if test_path.exists():
        tests = json.loads(test_path.read_text())
        lines += [f"Full offline suite: **{tests['tests_run']} tests; {tests['failures']} failures, "
                  f"{tests['errors']} errors, {tests['skipped']} skipped; {tests['unexpected_network_attempts']} network attempts**. "
                  "[Test summary](router_v3_tests.json).", ""]
    (reports / "router_v3_results.md").write_text("\n".join(lines))
    (reports / "router_v3_results.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
