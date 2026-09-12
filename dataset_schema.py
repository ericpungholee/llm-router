"""Normalized schema and validation for model evaluation attempts."""

import math
from datetime import datetime
from typing import Mapping, Optional, Sequence


RESULT_FIELDS = (
    "prompt_id",
    "prompt",
    "benchmark",
    "task_type",
    "reference_answer",
    "model_creator",
    "model_name",
    "api_model_identifier",
    "inference_provider",
    "model_type",
    "raw_response",
    "parsed_answer",
    "score",
    "correct",
    "input_tokens",
    "output_tokens",
    "estimated_cost_usd",
    "latency_ms",
    "temperature",
    "reasoning_setting",
    "max_output_tokens",
    "timestamp",
    "status",
    "error_type",
    "error_code",
    "error_message",
    "http_status",
    "retry_count",
    "provider_request_id",
    "response_model_identifier",
    "stop_reason",
)

ALWAYS_REQUIRED_TEXT_FIELDS = (
    "prompt_id",
    "prompt",
    "benchmark",
    "task_type",
    "reference_answer",
    "model_creator",
    "model_name",
    "api_model_identifier",
    "inference_provider",
    "model_type",
    "temperature",
    "reasoning_setting",
    "timestamp",
    "status",
)

VALID_STATUSES = {
    "success",
    "parsing_failure",
    "grading_failure",
    "provider_error",
    "skipped_model",
}


def _is_null(value: object) -> bool:
    return value is None or value == ""


def validate_result(row: Mapping[str, object]) -> None:
    """Raise ValueError if an evaluation row does not match the required schema."""
    missing_fields = [field for field in RESULT_FIELDS if field not in row]
    if missing_fields:
        raise ValueError(f"Missing result fields: {missing_fields}")

    empty_fields = [
        field for field in ALWAYS_REQUIRED_TEXT_FIELDS if not str(row[field]).strip()
    ]
    if empty_fields:
        raise ValueError(f"Empty result fields: {empty_fields}")

    if row["task_type"] not in {"multiple_choice", "math", "code"}:
        raise ValueError(f"Unsupported task_type: {row['task_type']}")
    if row["model_type"] not in {"closed", "open_weight"}:
        raise ValueError(f"Unsupported model_type: {row['model_type']}")
    if row["status"] not in VALID_STATUSES:
        raise ValueError(f"Unsupported status: {row['status']}")

    success = row["status"] == "success"
    if success:
        if not str(row["raw_response"]).strip() or not str(row["parsed_answer"]).strip():
            raise ValueError("Successful results require raw_response and parsed_answer")
        if not isinstance(row["correct"], bool):
            raise ValueError("Successful result correct must be a boolean")
        score = float(row["score"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score must be between 0 and 1")
    else:
        if not _is_null(row["score"]) or not _is_null(row["correct"]):
            raise ValueError("Failed/skipped results must have null score and correct")
        if row["status"] != "skipped_model" and not str(row["error_type"]).strip():
            raise ValueError("Failed results require error_type metadata")

    for field in ("max_output_tokens", "retry_count"):
        value = row[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    if row["max_output_tokens"] == 0:
        raise ValueError("max_output_tokens must be positive")

    for field in ("input_tokens", "output_tokens"):
        value = row[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")

    for field in ("estimated_cost_usd", "latency_ms"):
        value = float(row[field])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be non-negative")

    http_status = row["http_status"]
    if not _is_null(http_status) and (
        not isinstance(http_status, int)
        or isinstance(http_status, bool)
        or not 100 <= http_status <= 599
    ):
        raise ValueError("http_status must be null or a valid HTTP status")

    try:
        timestamp = datetime.fromisoformat(str(row["timestamp"]))
    except ValueError as error:
        raise ValueError("timestamp must be a valid ISO-8601 timestamp") from error
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must include a timezone")


def validate_results(
    rows: Sequence[Mapping[str, object]],
    expected_task_types: Optional[Mapping[str, str]] = None,
) -> None:
    if not rows:
        raise ValueError("No evaluation results were generated")
    for row in rows:
        validate_result(row)
        if expected_task_types is not None:
            benchmark_name = str(row["benchmark"])
            expected = expected_task_types.get(benchmark_name)
            if expected is None:
                raise ValueError(f"Unknown benchmark in result: {benchmark_name}")
            if row["task_type"] != expected:
                raise ValueError(
                    f"{benchmark_name} result has task_type {row['task_type']!r}; "
                    f"expected {expected!r}"
                )
