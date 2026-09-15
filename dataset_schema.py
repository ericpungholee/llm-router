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
    "provider_attempts",
    "provider_response_count",
    "response_cost_usd",
    "failure_retryable",
    "failure_scope",
    "configuration_fingerprint",
    "provider_diagnostics",
)

# Keep the original mandatory schema stable as optional telemetry grows.
LEGACY_RESULT_FIELDS = RESULT_FIELDS[:31]

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
    # Additional telemetry is optional in historical CSVs.
    missing_fields = [field for field in LEGACY_RESULT_FIELDS if field not in row]
    if missing_fields:
        raise ValueError(f"Missing result fields: {missing_fields}")

    empty_fields = [
        field
        for field in ALWAYS_REQUIRED_TEXT_FIELDS
        if row[field] is None or not str(row[field]).strip()
    ]
    if empty_fields:
        raise ValueError(f"Empty result fields: {empty_fields}")

    if row["task_type"] not in {"multiple_choice", "math", "code"}:
        raise ValueError(f"Unsupported task_type: {row['task_type']}")
    if row["model_type"] not in {"closed", "open_weight"}:
        raise ValueError(f"Unsupported model_type: {row['model_type']}")
    if row["status"] not in VALID_STATUSES:
        raise ValueError(f"Unsupported status: {row['status']}")
    for field in ("provider_attempts", "provider_response_count"):
        value = row.get(field, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    if row.get("failure_scope", "") not in {"", "pair", "model", "provider"}:
        raise ValueError("Invalid failure_scope")
    if row.get("failure_retryable") not in (None, "", True, False):
        raise ValueError("failure_retryable must be null or boolean")
    response_cost = float(row.get("response_cost_usd", 0))
    if not math.isfinite(response_cost) or response_cost < 0:
        raise ValueError("response_cost_usd must be non-negative")

    success = row["status"] == "success"
    if success:
        if any(
            not isinstance(row[field], str) or not row[field].strip()
            for field in ("raw_response", "parsed_answer")
        ):
            raise ValueError("Successful results require raw_response and parsed_answer")
        if not isinstance(row["correct"], bool):
            raise ValueError("Successful result correct must be a boolean")
        score = float(row["score"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score must be between 0 and 1")
    else:
        if not _is_null(row["score"]) or not _is_null(row["correct"]):
            raise ValueError("Failed/skipped results must have null score and correct")
        if row["status"] != "skipped_model" and (
            row["error_type"] is None or not str(row["error_type"]).strip()
        ):
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
    keys = [
        (row["prompt_id"], row["inference_provider"], row["api_model_identifier"]) for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate prompt/model pairs")
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
