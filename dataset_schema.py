"""Normalized schema for one prompt/model evaluation result."""

from typing import Mapping, Sequence


RESULT_FIELDS = (
    "prompt_id",
    "prompt",
    "task_type",
    "benchmark_name",
    "reference_answer",
    "model_creator",
    "model_name",
    "inference_provider",
    "model_type",
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
    "inference_provider",
    "model_type",
)


def validate_result(row: Mapping[str, object]) -> None:
    """Raise ValueError if an evaluation row does not match the required schema."""
    missing_fields = [field for field in RESULT_FIELDS if field not in row]
    if missing_fields:
        raise ValueError(f"Missing result fields: {missing_fields}")

    empty_fields = [field for field in TEXT_FIELDS if not str(row[field]).strip()]
    if empty_fields:
        raise ValueError(f"Empty result fields: {empty_fields}")

    if row["task_type"] not in {"multiple_choice", "short_answer_math", "code"}:
        raise ValueError(f"Unsupported task_type: {row['task_type']}")
    if row["model_type"] not in {"closed", "open_weight"}:
        raise ValueError(f"Unsupported model_type: {row['model_type']}")
    if not isinstance(row["correct"], bool):
        raise ValueError("correct must be a boolean")

    score = float(row["score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be between 0 and 1")

    for field in ("input_tokens", "output_tokens"):
        value = row[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")

    for field in ("estimated_cost_usd", "latency_ms"):
        if float(row[field]) < 0:
            raise ValueError(f"{field} must be non-negative")


def validate_results(rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("No evaluation results were generated")
    for row in rows:
        validate_result(row)

