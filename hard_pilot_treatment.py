"""Preflight a separate exhaustion-only treatment; paid dispatch needs --confirm."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from generate_dataset import generate_results, load_dotenv, write_results
from hard_pilot_experiment import (
    ROOT,
    frozen_inputs,
    grading_records,
    treatment_plan,
    validate_treatment_output,
)
from offline_guard import offline_only
from providers import missing_api_keys
from spend_control import SpendPreflightError


def prepare(budget, output, spend_cap, root=ROOT):
    with offline_only():
        output = validate_treatment_output(output, root)
        rows, models, _, provenance = frozen_inputs(root, verify_configuration=True)
        report, plan, calls = treatment_plan(rows, models, budget, spend_cap)
        records = grading_records(root, [r for r, _, _, _ in calls])
        report.update(
            parent_sha256=provenance["frozen_sha256"],
            output=str(output),
            generation_configuration=[asdict(m) for _, m, _, _ in calls],
        )
        return report, plan, calls, records, output


def execute(prepared, root=ROOT):
    report, plan, calls, records, output = prepared
    # Recheck every invariant, including files, at the actual dispatch boundary.
    fresh = prepare(report["max_output_tokens"], output, report["spend_cap_usd"], root)
    if fresh[0] != report:
        raise SpendPreflightError("Treatment plan changed after preflight")
    if not report["strict_preflight_passed"]:
        raise SpendPreflightError(
            "Paid treatment blocked: strict spend preflight failed. "
            "Resolve observed Qwen output/cap discrepancies offline; registry-contract "
            "estimates are not verified worst-case monetary bounds. No provider calls made."
        )
    load_dotenv()
    missing = missing_api_keys(tuple(m for _, m, _, _ in calls))
    if missing:
        raise SpendPreflightError("Missing treatment credentials: " + ", ".join(missing))
    output.parent.mkdir(parents=True, exist_ok=True)
    # A crash cannot authorize a second attempt: this sidecar is never resumed.
    sidecar = output.with_suffix(output.suffix + ".experiment.json")
    with sidecar.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps({**report, "execution_started": True}, indent=2, sort_keys=True) + "\n"
        )
    with output.open("x", encoding="utf-8"):
        pass
    results = []
    for row, model, prompt, budget in calls:
        results = generate_results(
            [records[row["prompt_id"]]],
            dry_run=False,
            models=[model],
            run_plan=plan,
            completed_results=results,
            max_output_tokens=budget,
            max_retries=0,
            max_api_attempts=1,
            exclude_xai=True,
            on_result=lambda saved: write_results(saved, output),
            print_before_call=True,
        )
        # A runtime spend stop cannot be bypassed by starting another pair.
        if any(r["error_type"] == "spend_limit_error" for r in results):
            break
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-output-tokens", type=int, required=True, choices=(8192, 16384))
    parser.add_argument("--max-spend-usd", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    prepared = prepare(args.max_output_tokens, args.output, args.max_spend_usd)
    print(json.dumps(prepared[0], indent=2, sort_keys=True), flush=True)
    if args.confirm:
        execute(prepared)
    else:
        print("Offline preflight only. No provider calls or treatment files written.")
        if not prepared[0]["strict_preflight_passed"]:
            raise SystemExit("Strict preflight BLOCKED: unresolved cost bound or insufficient cap.")


if __name__ == "__main__":
    main()
