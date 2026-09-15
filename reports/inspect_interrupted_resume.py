"""Read-only comparison of working results with the frozen 24/75 pilot."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_dataset import historical_attempts, read_results, result_key


def inspect(working, frozen):
    working_bytes, frozen_bytes = working.read_bytes(), frozen.read_bytes()
    rows, baseline = read_results(working), read_results(frozen)
    previous = {result_key(r): r for r in baseline}
    if {result_key(r) for r in rows} != set(previous):
        raise ValueError("Working and frozen pair matrices differ")
    # Compare every original CSV field, including text, without type coercion.
    with frozen.open(newline="") as handle:
        originals = {result_key(r): r for r in csv.DictReader(handle) if r["status"] == "success"}
    with working.open(newline="") as handle:
        current = {result_key(r): r for r in csv.DictReader(handle)}
    preserved = all(
        all(current[k].get(field) == value for field, value in r.items())
        for k, r in originals.items()
    )
    if not preserved:
        raise ValueError("A frozen successful row was modified")
    graded = [r for r in rows if r["status"] == "success"]
    added = [r for r in graded if previous[result_key(r)]["status"] != "success"]

    def cost(rs):
        return sum((Decimal(str(r["estimated_cost_usd"])) for r in rs), Decimal(0))

    report = {
        "working_sha256": hashlib.sha256(working_bytes).hexdigest(),
        "frozen_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "frozen_successes_preserved": preserved,
        "frozen_graded_pairs": len(originals),
        "current_graded_pairs": len(graded),
        "additional_successful_grades": len(added),
        "additional_correct_answers": sum(r["correct"] is True for r in added),
        "pending_pairs": len(rows) - len(graded),
        "expected_pairs": len(baseline),
        "statuses": dict(Counter(r["status"] for r in rows)),
        "recorded_spend_usd": str(cost(rows)),
        "additional_recorded_spend_usd": str(cost(rows) - cost(baseline)),
        "recorded_provider_attempts": sum(historical_attempts(r) for r in rows),
        "spend_caveat": "Recorded ledger includes conservative failed-attempt reserves, but excludes the historical uncheckpointed Grok interruption. Actual billed spend is unknown.",
        "new_grades": [
            {k: r[k] for k in ("model_name", "prompt_id", "correct", "output_tokens")}
            for r in added
        ],
        "by_model": [
            {
                "model": m,
                "graded": sum(r["status"] == "success" for r in rows if r["model_name"] == m),
                "pending": sum(r["status"] != "success" for r in rows if r["model_name"] == m),
                "recorded_spend_usd": str(cost([r for r in rows if r["model_name"] == m])),
            }
            for m in dict.fromkeys(r["model_name"] for r in rows)
        ],
        "failures": [
            {
                k: r.get(k)
                for k in (
                    "model_name",
                    "prompt_id",
                    "error_type",
                    "error_message",
                    "stop_reason",
                    "input_tokens",
                    "output_tokens",
                    "provider_attempts",
                    "latency_ms",
                    "provider_diagnostics",
                )
            }
            for r in rows
            if r["status"] == "provider_error"
        ],
        "output_limit_anomalies": [
            {k: r[k] for k in ("model_name", "prompt_id", "max_output_tokens", "output_tokens")}
            for r in rows
            if r["output_tokens"] > r["max_output_tokens"]
        ],
    }
    if working.read_bytes() != working_bytes or frozen.read_bytes() != frozen_bytes:
        raise ValueError("Results changed during offline inspection")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--working", type=Path, default=ROOT / "data/results/hard_pilot_results.csv"
    )
    parser.add_argument(
        "--frozen", type=Path, default=ROOT / "reports/snapshots/hard_pilot_final.csv"
    )
    args = parser.parse_args()
    print(json.dumps(inspect(args.working, args.frozen), indent=2))
