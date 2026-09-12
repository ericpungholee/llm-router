"""Normalized schema and validation for completed model evaluations."""

import math
from datetime import datetime
from typing import Mapping, Optional, Sequence


RESULT_FIELDS = (
    "prompt_id",
    "prompt",
    "task_type",
    "benchmark_name",
    "reference_answer",
    "model_creator",
    "model_name",
    "api_model_id",
    "inference_provider",
    "model_type",
    "temperature",
    "reasoning_setting",
    "max_output_tokens",
    "timestamp_utc",
    "model_output",
    "score",
    "correct",
    "input_tokens",
    "output_tokens",
    "estimated_cost_usd",
    "latency_ms",
)

TEXT_FIELDS = (
    "prompt_id",
    "prompt",
    "task_type",
    "benchmark_name",
    "reference_answer",
    "model_creator",
    "model_name",
    "api_model_id",
    "inference_provider",
    "model_type",
    "temperature",
    "reasoning_setting",
    "timestamp_utc",
    "model_output",
)


def validate_result(row: Mapping[str, object]) -> None:
    """Raise ValueError if an evaluation row does not match the required schema."""
    missing_fields = [field for field in RESULT_FIELDS if field not in row]
    if missing_fields:
        raise ValueError(f"Missing result fields: {missing_fields}")

    empty_fields = [field for field in TEXT_FIELDS if not str(row[field]).strip()]
    if empty_fields:
        raise ValueError(f"Empty result fields: {empty_fields}")

    if row["task_type"] not in {"multiple_choice", "math", "code"}:
        raise ValueError(f"Unsupported task_type: {row['task_type']}")
    if row["model_type"] not in {"closed", "open_weight"}:
        raise ValueError(f"Unsupported model_type: {row['model_type']}")
    if not isinstance(row["correct"], bool):
        raise ValueError("correct must be a boolean")

    score = float(row["score"])
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("score must be between 0 and 1")

    for field in ("max_output_tokens", "input_tokens", "output_tokens"):
        value = row[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")

    for field in ("estimated_cost_usd", "latency_ms"):
        value = float(row[field])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be non-negative")

    try:
        timestamp = datetime.fromisoformat(str(row["timestamp_utc"]))
    except ValueError as error:
        raise ValueError("timestamp_utc must be a valid ISO-8601 timestamp") from error
    if timestamp.tzinfo is None:
        raise ValueError("timestamp_utc must include a timezone")


def validate_results(
    rows: Sequence[Mapping[str, object]],
    expected_task_types: Optional[Mapping[str, str]] = None,
) -> None:
    if not rows:
        raise ValueError("No evaluation results were generated")
    for row in rows:
        validate_result(row)
        if expected_task_types is not None:
            benchmark_name = str(row["benchmark_name"])
            expected = expected_task_types.get(benchmark_name)
            if expected is None:
                raise ValueError(f"Unknown benchmark in result: {benchmark_name}")
            if row["task_type"] != expected:
                raise ValueError(
                    f"{benchmark_name} result has task_type {row['task_type']!r}; "
                    f"expected {expected!r}"
                )
