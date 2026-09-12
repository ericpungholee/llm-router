"""Generate normalized evaluation data with guarded live-run support."""

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from benchmark_loaders import BenchmarkRecord, load_enabled_benchmarks
from dataset_schema import RESULT_FIELDS, validate_results
from graders import (
    grade_code,
    grade_math,
    grade_multiple_choice,
    run_trusted_dry_run_tests,
)
from model_registry import ModelConfig, enabled_models, validate_registry
from providers import call_model, estimate_cost_usd
from spend_control import (
    RunPlan,
    SpendLimitError,
    SpendTracker,
    build_run_plan,
)


def render_prompt(record: BenchmarkRecord) -> str:
    if record.task_type == "multiple_choice":
        if not record.choices:
            raise ValueError(f"{record.prompt_id} is missing choices")
        choices = "\n".join(
            f"{label}. {text}" for label, text in record.choices.items()
        )
        return f"{record.prompt}\n{choices}\nAnswer with only the option letter."
    return record.prompt


def incorrect_mock_answer(record: BenchmarkRecord) -> str:
    if record.task_type == "multiple_choice":
        if not record.choices:
            raise ValueError(f"{record.prompt_id} is missing choices")
        return next(
            option
            for option in record.choices
            if option != record.reference_answer
        )
    if record.task_type == "math":
        return "0"
    return "print(0)"


def mock_answer(record: BenchmarkRecord, model: ModelConfig) -> str:
    identity = f"{record.prompt_id}:{model.key}"
    simulated_correct = hashlib.sha256(identity.encode("utf-8")).digest()[0] % 3 != 0
    if simulated_correct:
        if record.task_type == "code":
            if record.mock_solution is None:
                raise ValueError(f"{record.prompt_id} needs a mock solution")
            return record.mock_solution
        return record.reference_answer
    return incorrect_mock_answer(record)


def grade(record: BenchmarkRecord, output: str) -> bool:
    if record.task_type == "multiple_choice":
        if not record.choices:
            raise ValueError(f"{record.prompt_id} is missing choices")
        return grade_multiple_choice(
            output,
            record.reference_answer,
            tuple(record.choices),
        )
    if record.task_type == "math":
        return grade_math(output, record.reference_answer)
    if record.task_type == "code":
        return grade_code(output, record.tests, run_trusted_dry_run_tests)
    raise ValueError(f"Unsupported task type: {record.task_type}")


def planned_calls(
    benchmark: Sequence[BenchmarkRecord],
    models: Optional[Sequence[ModelConfig]] = None,
) -> List[Tuple[ModelConfig, str]]:
    selected_models = tuple(models) if models is not None else enabled_models()
    return [
        (model, render_prompt(record))
        for record in benchmark
        for model in selected_models
    ]


def result_key(row: Dict[str, object]) -> Tuple[str, str]:
    return str(row["prompt_id"]), str(row["api_model_id"])


def generate_results(
    benchmark: List[BenchmarkRecord],
    *,
    dry_run: bool,
    run_plan: Optional[RunPlan] = None,
    completed_results: Optional[Sequence[Dict[str, object]]] = None,
    initial_spend_usd: object = "0",
    on_result: Optional[Callable[[List[Dict[str, object]]], None]] = None,
) -> List[Dict[str, object]]:
    models = enabled_models()
    timestamp = datetime.now(timezone.utc).isoformat()
    results = list(completed_results or [])
    completed_keys = {result_key(row) for row in results}
    tracker = (
        SpendTracker(run_plan, initial_spend_usd=initial_spend_usd)
        if run_plan is not None
        else None
    )
    for record in benchmark:
        prompt = render_prompt(record)
        for model in models:
            key = (record.prompt_id, model.api_model_identifier or "UNRESOLVED")
            if key in completed_keys:
                continue
            if tracker is not None:
                tracker.assert_can_call(model, prompt)
            response = call_model(
                model,
                prompt,
                dry_run=dry_run,
                mock_response=mock_answer(record, model) if dry_run else None,
            )
            actual_cost = tracker.record_call(model, response) if tracker else None
            correct = grade(record, response.text)
            estimated_cost = estimate_cost_usd(
                model,
                response.input_tokens,
                response.output_tokens,
            )
            results.append(
                {
                    "prompt_id": record.prompt_id,
                    "prompt": prompt,
                    "task_type": record.task_type,
                    "benchmark_name": record.benchmark_name,
                    "reference_answer": record.reference_answer,
                    "model_creator": model.creator,
                    "model_name": model.canonical_model_name,
                    "api_model_id": model.api_model_identifier or "UNRESOLVED",
                    "inference_provider": model.inference_provider,
                    "model_type": model.model_type,
                    "temperature": model.generation.temperature_record,
                    "reasoning_setting": model.generation.reasoning_setting,
                    "max_output_tokens": model.generation.max_output_tokens,
                    "timestamp_utc": timestamp,
                    "model_output": response.text,
                    "score": 1.0 if correct else 0.0,
                    "correct": correct,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    # Unknown real prices stay unresolved in the registry. Zero is
                    # used only to keep mock rows numeric and is never used live.
                    "estimated_cost_usd": (
                        float(actual_cost)
                        if actual_cost is not None
                        else estimated_cost if estimated_cost is not None else 0.0
                    ),
                    "latency_ms": response.latency_ms,
                }
            )
            completed_keys.add(key)
            if on_result is not None:
                on_result(results)
    expected_task_types = {
        record.benchmark_name: record.task_type for record in benchmark
    }
    validate_results(results, expected_task_types)
    return results


def write_results(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use deterministic mock responses; makes no API calls.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live provider calls after spend preflight and confirmation.",
    )
    parser.add_argument(
        "--spend-cap",
        type=str,
        default=None,
        help="Per-run spend cap in USD; defaults to $1.00 and cannot exceed the global limit.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Explicitly authorize the displayed live-run plan.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from completed rows already present in --output.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/dry_run_results.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run and args.live:
        raise SystemExit("Choose either --dry-run or --live, not both.")
    if not args.dry_run and not args.live:
        raise SystemExit(
            "Choose --dry-run for zero-cost simulation or --live for guarded live calls."
        )

    load_dotenv()
    validate_registry()
    _, benchmark = load_enabled_benchmarks(args.manifest)
    existing_results: List[Dict[str, object]] = []
    if args.resume and args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                typed = dict(row)
                for field in ("max_output_tokens", "input_tokens", "output_tokens"):
                    typed[field] = int(str(typed[field]))
                for field in ("score", "estimated_cost_usd", "latency_ms"):
                    typed[field] = float(str(typed[field]))
                typed["correct"] = str(typed["correct"]).lower() == "true"
                existing_results.append(typed)
        validate_results(existing_results)

    # The run matrix is small and deterministic; use the output rows to skip
    # completed prompt/model pairs while preserving the original call order.
    completed_keys = {result_key(row) for row in existing_results}
    remaining_calls = []
    for record in benchmark:
        prompt = render_prompt(record)
        for model in enabled_models():
            if (record.prompt_id, model.api_model_identifier or "UNRESOLVED") not in completed_keys:
                remaining_calls.append((model, prompt))
    try:
        run_plan = build_run_plan(remaining_calls, spend_cap=args.spend_cap) if args.live else None
    except SpendLimitError as error:
        raise SystemExit(str(error)) from error
    if run_plan is not None:
        print(run_plan.confirmation_text())
        if not args.confirm:
            raise SystemExit("Live run not started: pass --confirm after reviewing the plan.")
    try:
        results = generate_results(
            benchmark,
            dry_run=args.dry_run,
            run_plan=run_plan,
            completed_results=existing_results,
            initial_spend_usd=sum(
                (Decimal(str(row["estimated_cost_usd"])) for row in existing_results),
                Decimal("0"),
            ),
            on_result=lambda rows: write_results(rows, args.output),
        )
    except SpendLimitError as error:
        raise SystemExit(str(error)) from error
    write_results(results, args.output)

    print(f"Loaded {len(benchmark)} benchmark prompts from {args.manifest}")
    print(f"{'Simulated' if args.dry_run else 'Completed'} {len(enabled_models())} hosted model/provider combinations")
    print(f"Validated {len(results)} rows with {len(RESULT_FIELDS)} required fields")
    print(f"Wrote normalized results to {args.output}")
    if args.dry_run:
        print("Dry run complete: no API calls were made and estimated metrics are synthetic")
    else:
        print("Live run complete: spend was tracked after every call")


if __name__ == "__main__":
    main()
