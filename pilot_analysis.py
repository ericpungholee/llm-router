"""Deterministic, offline accounting and routing-signal analysis of pilot CSVs."""

import argparse
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from statistics import mean

from dataset_schema import validate_results


RESPONSE_STATUSES = {"success", "parsing_failure", "grading_failure"}


def model_key(row):
    return str(row["inference_provider"]), str(row["api_model_identifier"])


def model_label(key):
    return "/".join(key)


def response_cost(row):
    """Inference cost alone, excluding reserves for preceding failed attempts."""
    if row.get("response_cost_usd") is not None:
        return Decimal(str(row["response_cost_usd"]))
    # Legacy rows lack a separate response cost. Their total is safe only when
    # no retries preceded the saved response.
    if int(row.get("retry_count", 0)):
        return None
    return Decimal(str(row["estimated_cost_usd"]))


def _metrics(rows, expected_pairs):
    from generate_dataset import historical_attempts, historical_responses

    counts = Counter(str(r["status"]) for r in rows)
    graded = counts["success"]
    correct = sum(r["status"] == "success" and r["correct"] is True for r in rows)
    latencies = [float(r["latency_ms"]) for r in rows if r["status"] in RESPONSE_STATUSES]
    return {
        "expected_pairs": expected_pairs,
        "recorded_pairs": len(rows),
        "successfully_graded_pairs": graded,
        "attempted_provider_calls": sum(historical_attempts(r) for r in rows),
        "successful_provider_responses": sum(historical_responses(r) for r in rows),
        "failed_provider_attempts": sum(historical_attempts(r) - historical_responses(r) for r in rows),
        "successfully_graded_responses": graded,
        "correct_responses": correct,
        "accuracy_over_graded": correct / graded if graded else None,
        "provider_failures": sum(r["status"] == "provider_error" and r["error_type"] != "spend_limit_error" for r in rows),
        "parsing_failures": counts["parsing_failure"],
        "grading_failures": counts["grading_failure"],
        "skipped_pairs": counts["skipped_model"],
        "unrecorded_pairs": expected_pairs - len(rows),
        "spend_limit_stops": sum(r["error_type"] == "spend_limit_error" for r in rows),
        "total_recorded_cost_usd": float(sum((Decimal(str(r["estimated_cost_usd"])) for r in rows), Decimal(0))),
        "average_response_latency_ms": mean(latencies) if latencies else None,
    }


def analyze_results(rows, expected_prompt_ids=None, expected_models=None, expected_prompt_benchmarks=None):
    """Compare policies on the SAME fully graded prompts, never impute failure.

    Cost order is observed average response cost on that common subset; higher
    cost is only a proxy for strength. Oracle cost sums the cheapest correct
    response on solved prompts. Unsolved prompts have no selected response and
    are reported separately, rather than assigned a fictitious free solution.
    """
    validate_results(rows)
    prompts = sorted(set(expected_prompt_ids if expected_prompt_ids is not None else (r["prompt_id"] for r in rows)))
    models = sorted(set(expected_models if expected_models is not None else (model_key(r) for r in rows)))
    matrix = {(str(r["prompt_id"]), model_key(r)): r for r in rows}
    planned = {(p, m) for p in prompts for m in models}
    if set(matrix) - planned:
        raise ValueError("CSV contains pairs outside the selected comparison plan")
    membership = dict(expected_prompt_benchmarks) if expected_prompt_benchmarks is not None else {
        str(r["prompt_id"]): str(r["benchmark"]) for r in rows
    }
    if set(membership) != set(prompts):
        raise ValueError("Expected prompt -> benchmark mapping is required for every planned prompt")
    if any(membership[str(r["prompt_id"])] != r["benchmark"] for r in rows):
        raise ValueError("CSV benchmark membership differs from the comparison plan")
    graded = sum(r["status"] == "success" for r in rows)
    expected = len(planned)
    common = [p for p in prompts if all((p, m) in matrix and matrix[p, m]["status"] == "success" for m in models)]
    by_model = {
        model_label(m): _metrics([r for r in rows if model_key(r) == m], len(prompts))
        for m in models
    }
    benchmarks = sorted(set(membership.values()))
    by_benchmark = {
        b: {
            model_label(m): _metrics(
                [r for r in rows if model_key(r) == m and r["benchmark"] == b],
                sum(benchmark == b for benchmark in membership.values()),
            ) for m in models
        } for b in benchmarks
    }
    comparison = {
        "comparison_prompt_ids": common,
        "comparison_prompt_count": len(common),
        "excluded_incomplete_prompt_ids": [p for p in prompts if p not in common],
        "accuracy_denominator": "prompts successfully graded by every planned model",
        "cost_rule": "observed response cost, excluding failed-attempt reserves; ties broken by provider/model ID",
        "cost_order_is_strength_proxy": True,
        "oracle_accuracy": None,
        "always_best_single_model": None,
        "always_cheapest_model": None,
        "oracle_cheapest_correct_cost_usd": None,
        "oracle_unsolved_prompt_ids": [],
        "cheaper_solved_expensive_missed": [],
        "prompts_requiring_higher_cost_models": [],
        "evidence_of_routing_signal": None,
    }
    if common:
        accuracies = {m: sum(matrix[p, m]["correct"] is True for p in common) / len(common) for m in models}
        solved = [p for p in common if any(matrix[p, m]["correct"] is True for m in models)]
        unsolved = [p for p in common if p not in solved]
        best = min(models, key=lambda m: (-accuracies[m], m))
        comparison.update(
            oracle_accuracy=len(solved) / len(common),
            always_best_single_model={"model": model_label(best), "accuracy": accuracies[best]},
            oracle_unsolved_prompt_ids=unsolved,
            evidence_of_routing_signal=len(solved) / len(common) > accuracies[best],
        )
        costs = {(p, m): response_cost(matrix[p, m]) for p in common for m in models}
        if all(c is not None for c in costs.values()):
            average = {m: sum(costs[p, m] for p in common) / len(common) for m in models}
            order = sorted(models, key=lambda m: (average[m], m))
            cheapest = order[0]
            comparison["cost_order"] = [model_label(m) for m in order]
            comparison["always_cheapest_model"] = {
                "model": model_label(cheapest), "accuracy": accuracies[cheapest],
                "total_response_cost_usd": float(sum(costs[p, cheapest] for p in common)),
            }
            comparison["always_best_single_model"]["total_response_cost_usd"] = float(sum(costs[p, best] for p in common))
            selections = [
                (p, min((m for m in models if matrix[p, m]["correct"] is True), key=lambda m: (costs[p, m], m)))
                for p in solved
            ]
            comparison["oracle_cheapest_correct_cost_usd"] = float(sum((costs[p, m] for p, m in selections), Decimal(0)))
            comparison["oracle_selections"] = [{"prompt_id": p, "model": model_label(m), "cost_usd": float(costs[p, m])} for p, m in selections]
            for p in common:
                winners = [m for m in order if matrix[p, m]["correct"] is True]
                losers = [m for m in order if matrix[p, m]["correct"] is False]
                inversions = [(w, l) for w in winners for l in losers if average[w] < average[l]]
                if inversions:
                    comparison["cheaper_solved_expensive_missed"].append({
                        "prompt_id": p, "benchmark": matrix[p, models[0]]["benchmark"],
                        "comparisons": [{"cheaper_correct": model_label(w), "more_expensive_incorrect": model_label(l)} for w, l in inversions],
                    })
                if winners and average[winners[0]] > average[cheapest]:
                    comparison["prompts_requiring_higher_cost_models"].append({
                        "prompt_id": p, "benchmark": matrix[p, models[0]]["benchmark"],
                        "correct_models": [model_label(m) for m in winners],
                        "cheaper_incorrect_models": [model_label(m) for m in losers if average[m] < average[winners[0]]],
                    })
            comparison["evidence_of_routing_signal"] = bool(
                comparison["evidence_of_routing_signal"] or comparison["cheaper_solved_expensive_missed"]
                or comparison["prompts_requiring_higher_cost_models"]
            )
        else:
            comparison["cost_unavailable_reason"] = "Legacy retry rows do not separate response cost from failed-attempt reserves"
    return {
        "matrix_completeness": {
            "successfully_graded_pairs": graded, "expected_pairs": expected,
            "description": f"{graded} / {expected} prompt-model pairs successfully graded",
            "fraction": graded / expected if expected else None,
            "recorded_pairs": len(rows), "fully_graded_prompts": len(common),
            "unrecorded_pairs": expected - len(rows),
            "planned_prompts": len(prompts), "valid_complete_comparison": graded == expected and expected > 0,
        },
        "by_model": by_model, "by_benchmark_and_model": by_benchmark,
        "routing_comparison": comparison,
        "interpretation": "Descriptive pilot evidence only. Incomplete prompts are excluded from policy comparisons; higher cost does not prove greater capability. Oracle cost covers solved prompts only.",
    }


def main():
    from benchmark_loaders import load_enabled_benchmarks
    from generate_dataset import read_results, select_pilot_prompts
    from model_registry import enabled_models

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?", default=Path("data/results/hard_pilot_results.csv"))
    parser.add_argument("--observed-plan", action="store_true", help="Infer prompts/models from CSV for historical or synthetic runs (cannot detect wholly absent prompts/models).")
    args = parser.parse_args()
    rows = read_results(args.csv)
    if args.observed_plan:
        report = analyze_results(rows)
    else:
        _, records = load_enabled_benchmarks(Path("benchmarks/manifest.json"))
        pilot = select_pilot_prompts(records)
        report = analyze_results(rows, [r.prompt_id for r in pilot],
                                 [(m.inference_provider, m.api_model_identifier) for m in enabled_models()],
                                 {r.prompt_id: r.benchmark_name for r in pilot})
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
