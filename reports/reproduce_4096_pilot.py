"""Rebuild the completed 4096-token report using only the tracked snapshot."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hard_pilot_experiment import SNAPSHOT, SOURCE, build_report, freeze
from offline_guard import offline_only


def render(r):
    lines = ["# Completed 4096-token hard pilot", ""]

    def add(s=""):
        lines.append(s)

    def table(headers, rows):
        add("| " + " | ".join(headers) + " |")
        add("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            add("| " + " | ".join(str(x) if x is not None else "N/A" for x in row) + " |")
        add()

    def short(label):
        return {
            "anthropic": "Claude",
            "deepseek": "DeepSeek",
            "openai": "GPT-5.6 Sol",
            "openrouter": "Qwen",
            "xai": "Grok",
        }[label.split("/")[0]]

    p, m, a = r["provenance"], r["overall"], r["analysis"]
    c, e = a["routing_comparison"], r["output_exhaustion"]
    add(
        "**Decision: unsuitable for training or evaluating a learned router.** Systematic budget exhaustion removes gradeable answers by provider and task. This is nonrandom missingness, which can bias complete-case model comparisons toward easy-to-finish prompts. Five common prompts cannot establish routing generalization."
    )
    add()
    add(
        f"Immutable input: [`{SNAPSHOT}`](snapshots/hard_pilot_4096_final.csv), frozen at {p['timestamp_utc']} from commit `{p['git_commit']}`. SHA-256 `{p['frozen_sha256']}` matches the source `{SOURCE}` byte for byte. The original was not modified or regraded. [Provenance](snapshots/hard_pilot_4096_final.provenance.json) pins request adapters, formatting, grading, benchmark inputs, registry IDs/prices/settings, and ledger totals. Freeze and reproduction block network/provider access. No provider calls occurred."
    )
    add()
    add("## Final accounting and completeness")
    add()
    table(
        ["Metric", "Value"],
        [
            ["Rows / expected pairs", f"{m['recorded_pairs']} / {m['expected_pairs']}"],
            ["Attempted provider calls (recorded/inferred history)", m["attempted_provider_calls"]],
            ["Successful normalized provider responses", m["successful_provider_responses"]],
            ["Successfully graded pairs", m["successfully_graded_pairs"]],
            ["Correct / graded", f"{m['correct_responses']} / {m['successfully_graded_pairs']}"],
            ["Accuracy on graded pairs", f"{m['accuracy_over_graded']:.2%}"],
            ["Provider failures, excluding spend stops", m["provider_failures"]],
            ["Parsing / grading failures", f"{m['parsing_failures']} / {m['grading_failures']}"],
            [
                "Skipped / spend stops / unrecorded",
                f"{m['skipped_pairs']} / {m['spend_limit_stops']} / {m['unrecorded_pairs']}",
            ],
            ["Output-limit-exhausted final pairs", e["count"]],
            ["Fully graded five-model prompts", c["comparison_prompt_count"]],
            ["Recorded spend USD", r["recorded_spend_usd"]],
        ],
    )
    add(
        f"The existing response counter excludes {p['http_200_exhausted_responses']} saved HTTP-200 exhaustion outcomes; those are provider-returned responses but failed normalized completions. The 83 attempted calls include earlier attempts/resumes, with some history inferred from legacy fields. Exhausted/unknown billed failures carry maximum-cost reserves. Recorded spend is a ledger estimate, not a reconciled invoice; see the historical billing limitations in [xAI cost safety](xai_cost_safety.md). All remaining failure/skipped states remain missing grades, never incorrect answers."
    )
    add()
    headers = [
        "Group",
        "Graded / expected",
        "Correct / graded",
        "Provider fail",
        "Parse / grade fail",
        "Skipped",
        "Spend stop",
        "Spend USD",
    ]

    def metric_row(label, v):
        return [
            label,
            f"{v['successfully_graded_pairs']} / {v['expected_pairs']}",
            f"{v['correct_responses']} / {v['successfully_graded_pairs']}",
            v["provider_failures"],
            f"{v['parsing_failures']} / {v['grading_failures']}",
            v["skipped_pairs"],
            v["spend_limit_stops"],
            f"{v['total_recorded_cost_usd']:.7f}",
        ]

    table(headers, [metric_row(short(k), v) for k, v in a["by_model"].items()])
    table(headers, [metric_row(k, v) for k, v in r["by_benchmark"].items()])
    add(
        "Per-model accuracies above use different successful subsets. They do not rank performance over the full 15-prompt plan. All cells are recorded; completeness measures grades. Grok's existing results are retained, but Grok was excluded from the latest paid resume and is excluded from treatment."
    )
    add()
    add("## Five-model common subset")
    add()
    add(
        f"**Only {c['comparison_prompt_count']} fully graded prompts exist.** Correctness vectors (1 correct, 0 incorrect):"
    )
    add()
    labels = list(r["static_baselines"])
    table(
        ["Prompt"] + [short(k) for k in labels],
        [
            [v["prompt_id"]] + [int(v["correctness"][k]) for k in labels]
            for v in r["common_vectors"]
        ],
    )
    table(
        ["Static model", "Correct / n", "Accuracy", "Response cost USD"],
        [
            [
                short(k),
                f"{v['correct']} / {v['n']}",
                f"{v['accuracy']:.0%}",
                v["total_response_cost_usd"],
            ]
            for k, v in r["static_baselines"].items()
        ],
    )
    add(
        f"Five-model oracle: {c['oracle_accuracy']:.0%}; cheapest-correct oracle cost: **${c['oracle_cheapest_correct_cost_usd']:.7f}** across all five (no unsolved prompts). Best static accuracy is 100%, tied among Claude, DeepSeek, and Qwen; the deterministic provider/ID tie-break selects Claude (${c['always_best_single_model']['total_response_cost_usd']:.7f}). DeepSeek is also 100% and is the cheapest static (${c['always_cheapest_model']['total_response_cost_usd']:.7f}); the oracle saves nothing and adds no accuracy relative to static DeepSeek. Costs use response costs, excluding preceding failure reserves. {r['common_disagreement_count']}/5 prompts show correctness disagreement."
    )
    add()
    add(
        "**No prompt-visible routing baseline can be meaningfully evaluated here.** A benchmark rule could be computed on these same five selected prompts, but there is no held-out split, no common code prompt, and no oracle improvement over static DeepSeek. The generic analysis flag for cost-order inversions is descriptive and does not establish actionable routing signal. Freeze this as a feasibility experiment; do not train ML."
    )
    add()
    add("## Pairwise common-prompt comparisons")
    add()
    add(
        "Each comparison uses its own jointly graded overlap; denominators differ and selected subsets remain biased. Wins count correctness disagreements, not answer-text differences."
    )
    add()
    table(
        [
            "Pair A / B",
            "n",
            "A accuracy",
            "B accuracy",
            "A-only / B-only correct",
            "Both correct / wrong",
        ],
        [
            [
                " / ".join(short(x) for x in v["models"]),
                v["n"],
                f"{v['accuracy_a']:.1%}",
                f"{v['accuracy_b']:.1%}",
                f"{v['a_only_correct']} / {v['b_only_correct']}",
                f"{v['both_correct']} / {v['both_wrong']}",
            ]
            for v in [p["all"] for p in r["pairwise"]]
        ],
    )
    add(
        "Correctness disagreement matrix: disagree / jointly graded n. The JSON also includes each pair's prompt IDs, per-benchmark overlaps, wins, and rates."
    )
    add()
    table(
        ["Model"] + [short(k) for k in labels],
        [
            [short(k)]
            + [
                f"{r['disagreement_matrix'][k][j]['disagreement']} / {r['disagreement_matrix'][k][j]['n']}"
                for j in labels
            ]
            for k in labels
        ],
    )
    add("## Output exhaustion")
    add()
    table(["Model", "Exhausted"], [[short(k), v] for k, v in e["by_model"].items()])
    table(
        ["Benchmark", "Task type", "Exhausted"],
        [
            [b, t, n]
            for b, t, n in [
                ("livecodebench", "code", e["by_benchmark"].get("livecodebench", 0)),
                ("mmlu_pro", "multiple_choice", e["by_benchmark"].get("mmlu_pro", 0)),
                ("math_500", "math", e["by_benchmark"].get("math_500", 0)),
            ]
        ],
    )
    add(
        "All final exhausted pairs are listed below. Counts/tags only; no reasoning contents. Code accounts for 14/20 (70%); DeepSeek and Qwen contribute 14/20. Claude's four exhausted pairs and GPT's two are all code. DeepSeek also exhausts two multiple-choice prompts and one math prompt; Qwen exhausts two multiple-choice and one math prompt. This is concentrated in LiveCodeBench, with provider-specific missingness elsewhere. The missing `abc400_g` cells and Grok skips are separate causes, not inferred exhaustion."
    )
    add()
    table(
        [
            "Prompt",
            "Model",
            "Benchmark / task",
            "Requested",
            "Output",
            "Reasoning",
            "Stop",
            "Visible",
        ],
        [
            [
                v["prompt_id"],
                short(v["model"]),
                v["benchmark"] + " / " + v["task_type"],
                v["requested_max_output_tokens"],
                v["output_tokens"],
                v["reasoning_tokens"],
                v["stop_reason"]
                + (" / " + v["incomplete_reason"] if v["incomplete_reason"] else ""),
                v["visible_text_present"],
            ]
            for v in e["pairs"]
        ],
    )
    add(
        "All 20 report output_tokens=4096 and no visible text. Reasoning diagnostics are provider telemetry: Qwen reports 2161, 3812, and 4116 on three pairs; 4116 exceeds its inclusive output count and is internally inconsistent. Do not add reasoning counts again to output usage or infer hidden contents. Budget-exhausted runs only establish a lower bound on required completion budget; they do not identify the amount needed to finish."
    )
    add()
    add(
        "The pilot uniformly records a requested 4096 cap, but the two Qwen successes above that cap mean a uniformly enforced 4096 billed-output treatment is not established. Preserve those successful observations as recorded; this additional contract inconsistency also limits cross-provider budget comparisons."
    )
    add()
    add("## Treatment recommendation and strict cost preflight")
    add()
    add(
        "Recommend **8192** as the smallest useful feasibility treatment, not a promise of sufficiency. It doubles the observed exhausted ceiling; GPT already finishes two other code tasks within 4096, and Qwen has completed two answers with reported usage 5559/6298, below 8192. Neither these different prompts nor right-censored failures prove 8192 sufficient for hard code. 16384 doubles the incremental capped output cost without an observed completion threshold requiring it. Consider a separately preregistered 16384 experiment only after inspecting 8192 recovery; never auto-escalate or retry."
    )
    add()
    add(
        "OpenAI documents that reasoning can consume the entire response budget before visible output; its broad starting guidance reserves 25,000 tokens, so neither candidate is universally guaranteed. This study deliberately tests a smaller budget at unchanged effort. [OpenAI reasoning guidance](https://developers.openai.com/api/docs/guides/reasoning#allocating-space-for-reasoning). Claude's hard request cap includes thinking and answer text, and high effort can exhaust it. [Claude steering and cost](https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost). DeepSeek exposes thinking separately and documents the completion token ceiling. [DeepSeek thinking](https://api-docs.deepseek.com/guides/thinking_mode/), [completion reference](https://api-docs.deepseek.com/api/create-chat-completion/). OpenRouter bills reasoning as output even when excluded from the returned text. [OpenRouter reasoning](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens). Public docs were inspected without provider API calls."
    )
    add()
    add(
        "Select exactly the 20 explicitly exhausted final pairs (GPT 2, Claude 4, DeepSeek 7, Qwen 7), preserving prompt bytes, model IDs, temperature/default, reasoning settings, request adapters, benchmark inputs and graders. Only max output tokens changes. One client provider attempt per pair, zero retries for any error, no Grok, fresh output and exclusive execution sidecar. Existing outputs cannot be overwritten or resumed. OpenRouter upstream choice/failover remains its existing behavior; one client HTTP attempt cannot prove one upstream backend attempt, and unpinned upstream variation remains a scientific limitation."
    )
    add()
    table(
        [
            "Budget",
            "Calls / maximum client attempts",
            "Registry-contract maximum USD",
            "Verified worst case USD",
        ],
        [
            [
                b,
                f"{v['planned_calls']} / {v['maximum_provider_attempts']}",
                v["registry_contract_maximum_usd"],
                v["verified_worst_case_spend_usd"],
            ]
            for b, v in r["treatments"].items()
        ],
    )
    table(
        ["Model", "Calls", "8192 contract USD", "16384 contract USD"],
        [
            [
                short(k),
                v["calls"],
                v["registry_contract_maximum_usd"],
                r["treatments"]["16384"]["by_model"][k]["registry_contract_maximum_usd"],
            ]
            for k, v in r["treatments"]["8192"]["by_model"].items()
        ],
    )
    add(
        "Formula per actual selected pair: `(UTF-8 prompt byte length × registry input price + budget × registry output price) / 1,000,000`, summed with Decimal and rounded upward to six decimal places. UTF-8 bytes conservatively bound user-text tokenization; existing prices are uncached standard/peak rates. No reasoning double-count, no retries, no xAI. Costs are incremental to the frozen ledger. Registry-contract totals assume the documented inclusive output cap applies to billed usage and existing input-overhead assumptions."
    )
    add()
    add(
        "**Strict preflight is currently blocked for both candidates.** Qwen's new saved telemetry contradicts the output ceiling:"
    )
    add()
    table(
        ["Pair", "Requested", "Output", "Reasoning"],
        [
            [
                v["prompt_id"],
                v["requested_max_output_tokens"],
                v["output_tokens"],
                v["reasoning_tokens"],
            ]
            for v in r["cost_bound_issues"]
        ],
    )
    add(
        "OpenRouter documents a generation ceiling and bills reasoning as output; the current data do not establish whether backend enforcement, request mapping, or usage interpretation caused this discrepancy. [OpenRouter parameters](https://openrouter.ai/docs/api_reference/parameters). A historical maximum, an arbitrary multiplier, the model/context limit, or a local USD cap cannot repair an unverified billing bound. Thus $1.656349/$3.216105 are conservative **contract-based estimates**, not verified worst-case spend. No finite verified maximum for the complete 20-call treatment can be certified from the available data. Reconcile exact request forwarding and billed inclusive usage using existing offline request/billing exports or provider confirmation before enabling paid execution. Resolve bound assumptions and pin the resulting evidence in code/provenance; there is deliberately no override flag. The analysis is complete despite this execution gate."
    )
    add()
    add("## Preregistered success criteria")
    add()
    add(
        "- At least 16/20 previously exhausted pairs (80%) return visible, parseable, successfully graded answers, counting correct and incorrect grades as successful recovery. Keep visible-answer rate and gradeability rate separate; report parsing/grading failures explicitly."
    )
    add(
        "- At least 70% gradeability recovery within each provider (GPT 2/2, Claude 3/4, DeepSeek 5/7, Qwen 5/7). Report residual exhaustion, network failures, and any budget-usage inconsistency by provider and benchmark; small counts cannot prove absence of clustering."
    )
    add(
        "- At least ten complete four-provider prompts in a separate read-only recovery view, including at least two code prompts. Report five-model completeness separately; Grok is still deferred. This is feasibility coverage, not enough training data."
    )
    add(
        "- Only justify scaling to a larger stratified evaluation after recovery/coverage criteria and strict cost checks pass and no unresolved cap anomaly remains. Establish held-out splits and a uniform-budget collection policy before learned-router evaluation. Failed criteria trigger diagnosis and a newly preregistered treatment, not automatic spend or retries."
    )
    add()
    counts = {k: len(v) for k, v in r["maximum_completeness_after_all_selected_recover"].items()}
    add(
        f"Even perfect recovery yields at most {counts['five_model']} five-model and {counts['four_non_xai_models']} four-provider complete prompts: untouched Grok failures/skips and non-exhaustion stops limit coverage. Recovery views must label reused 4096 outcomes and new higher-budget outcomes separately. A selectively repaired matrix is not a uniform 8192 evaluation and must not be silently merged into either experiment. A larger dataset needs an outcome-independent budget policy."
    )
    add()
    add("## Reproduction and exact commands")
    add()
    add(
        "Offline reproduction from a fresh checkout requires no ignored `data/`, credentials, or network:"
    )
    add()
    add(
        "```bash\npython3 reports/reproduce_4096_pilot.py --output-dir /tmp/hard-pilot-4096-report\n```"
    )
    add()
    add("Exact 8192 preflight (currently exits blocked, writes no treatment file):")
    add()
    command = "python3 hard_pilot_treatment.py --max-output-tokens 8192 --max-spend-usd 1.66 --output data/results/hard_pilot_exhaustion_8192.csv"
    add("```bash\n" + command + "\n```")
    add()
    add("Exact paid command, **not executed and currently blocked before dispatch**:")
    add()
    add("```bash\n" + command + " --confirm\n```")
    add()
    add(
        "The 16384 alternative uses `--max-output-tokens 16384 --max-spend-usd 3.22 --output data/results/hard_pilot_exhaustion_16384.csv`; it is also blocked. No paid treatment has run. The test suite exercises only mocked paid paths and blocks real HTTP/socket access."
    )
    add()
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze", action="store_true", help="Exclusive creation; never refreeze/overwrite"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    with offline_only():
        if args.freeze:
            freeze()
        report = build_report()
        content = {
            "hard_pilot_4096_analysis.json": json.dumps(report, indent=2, sort_keys=True) + "\n",
            "hard_pilot_4096_analysis.md": render(report),
        }
        # A reproduction destination cannot replace any pinned experimental input.
        protected = {
            (ROOT / SOURCE).resolve(),
            (ROOT / SNAPSHOT).resolve(),
            (ROOT / SNAPSHOT.with_suffix(".provenance.json")).resolve(),
        }
        if any(
            (args.output_dir / name).resolve() in protected
            or (args.output_dir / name)
            .resolve()
            .is_relative_to((ROOT / "reports/snapshots").resolve())
            for name in content
        ):
            raise ValueError("Report output cannot overwrite frozen inputs")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, data in content.items():
            (args.output_dir / name).write_text(data, encoding="utf-8")
        print(f"Offline report rebuilt: {args.output_dir}")


if __name__ == "__main__":
    main()
