"""Generate normalized multi-provider benchmark evaluation data."""

import argparse
import csv
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Mapping

from dotenv import load_dotenv

from dataset_schema import RESULT_FIELDS, validate_results
from providers import MODEL_SPECS, ModelSpec, call_model


BENCHMARK_FIELDS = (
    "prompt_id",
    "prompt",
    "task_type",
    "benchmark_name",
    "reference_answer",
)


def load_benchmark(path: Path) -> List[Dict[str, object]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = [field for field in BENCHMARK_FIELDS if field not in record]
            if missing:
                raise ValueError(f"{path}:{line_number} is missing {missing}")
            if record["task_type"] not in {
                "multiple_choice",
                "short_answer_math",
                "code",
            }:
                raise ValueError(
                    f"{path}:{line_number} has unsupported task_type "
                    f"{record['task_type']}"
                )
            records.append(record)

    prompt_ids = [str(record["prompt_id"]) for record in records]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("prompt_id values must be unique")
    if not records:
        raise ValueError(f"No benchmark records found in {path}")
    return records


def render_prompt(record: Mapping[str, object]) -> str:
    prompt = str(record["prompt"])
    if record["task_type"] == "multiple_choice":
        choices = record.get("choices")
        if not isinstance(choices, dict) or not choices:
            raise ValueError(f"{record['prompt_id']} requires a non-empty choices object")
        rendered_choices = "\n".join(
            f"{label}. {choice}" for label, choice in choices.items()
        )
        return f"{prompt}\n{rendered_choices}\nAnswer with only the choice letter."
    return prompt


def _normalize_answer(value: str) -> str:
    return " ".join(value.strip().lower().split())


def grade_response(task_type: str, response: str, reference_answer: str) -> bool:
    if task_type == "short_answer_math":
        try:
            return Decimal(response.strip()) == Decimal(reference_answer.strip())
        except InvalidOperation:
            return _normalize_answer(response) == _normalize_answer(reference_answer)

    # Code uses exact match only in the mock pipeline. Sandboxed tests come later.
    return _normalize_answer(response) == _normalize_answer(reference_answer)


def incorrect_mock_answer(record: Mapping[str, object]) -> str:
    task_type = str(record["task_type"])
    reference = str(record["reference_answer"])
    if task_type == "multiple_choice":
        choices = list(record["choices"])
        return next(choice for choice in choices if choice != reference)
    if task_type == "short_answer_math":
        return "1" if reference.strip() == "0" else "0"
    return "def add(a, b):\n    return a - b"


def mock_answer(record: Mapping[str, object], model: ModelSpec) -> str:
    identity = f"{record['prompt_id']}:{model.inference_provider}"
    simulated_correct = hashlib.sha256(identity.encode("utf-8")).digest()[0] % 3 != 0
    if simulated_correct:
        return str(record["reference_answer"])
    return incorrect_mock_answer(record)


def generate_results(
    benchmark: List[Dict[str, object]], *, dry_run: bool
) -> List[Dict[str, object]]:
    results = []
    for record in benchmark:
        prompt = render_prompt(record)
        for model in MODEL_SPECS:
            response = call_model(
                model,
                prompt,
                dry_run=dry_run,
                mock_response=mock_answer(record, model) if dry_run else None,
            )
            correct = grade_response(
                str(record["task_type"]),
                response.text,
                str(record["reference_answer"]),
            )
            results.append(
                {
                    "prompt_id": record["prompt_id"],
                    "prompt": prompt,
                    "task_type": record["task_type"],
                    "benchmark_name": record["benchmark_name"],
                    "reference_answer": record["reference_answer"],
                    "model_creator": model.model_creator,
                    "model_name": model.model_name,
                    "inference_provider": model.inference_provider,
                    "model_type": model.model_type,
                    "score": 1.0 if correct else 0.0,
                    "correct": correct,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "estimated_cost_usd": response.estimated_cost_usd,
                    "latency_ms": response.latency_ms,
                }
            )
    validate_results(results)
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
        "--input",
        type=Path,
        default=Path("benchmarks/sample.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/dry_run_results.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        raise SystemExit(
            "Live calls are disabled in Phase 1. Run with --dry-run to spend $0."
        )

    load_dotenv()
    benchmark = load_benchmark(args.input)
    results = generate_results(benchmark, dry_run=True)
    write_results(results, args.output)

    print(f"Loaded {len(benchmark)} benchmark prompts from {args.input}")
    print(f"Simulated {len(MODEL_SPECS)} hosted model/provider combinations")
    print(f"Validated {len(results)} rows with {len(RESULT_FIELDS)} required fields")
    print(f"Wrote normalized results to {args.output}")
    print("Dry run complete: no API calls were made and estimated metrics are synthetic")


if __name__ == "__main__":
    main()

