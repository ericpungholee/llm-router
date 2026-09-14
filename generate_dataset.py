"""Generate normalized evaluation data with guarded live-run modes."""

import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # Keep --dry-run usable in a dependency-free checkout.
    def load_dotenv() -> bool:
        path = Path(".env")
        if not path.exists():
            return False
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
        return True

from benchmark_loaders import BenchmarkRecord, load_enabled_benchmarks, load_hard_pilot
from dataset_schema import RESULT_FIELDS, validate_results
from graders import (
    MalformedModelOutput,
    extract_code,
    grade_code,
    grade_math,
    grade_multiple_choice,
    parse_math_answer,
    parse_multiple_choice,
    run_code_tests,
)
from model_registry import ModelConfig, enabled_models, validate_registry
from providers import (
    ProviderError,
    call_model_with_retries,
    provider_display_name,
    validate_provider_registry,
)
from spend_control import (RunPlan, SpendCapExceeded, SpendLimitError, SpendTracker,
                           build_run_plan, run_spend_cap_usd)


SMOKE_MAX_OUTPUT_TOKENS = 128
SMOKE_MAX_API_CALLS = 5
HARD_PILOT_PAIR_COUNT = 75
HARD_PILOT_MAX_RETRIES = 2
HARD_PILOT_MAX_ATTEMPTS = HARD_PILOT_PAIR_COUNT * (1 + HARD_PILOT_MAX_RETRIES)
HARD_PILOT_MAX_SPEND_USD = Decimal("2.00")
HARD_PILOT_DEFINITION = Path("benchmarks/hard_pilot.json")
PAID_TERMINAL_STATUSES = frozenset(("success", "parsing_failure", "grading_failure"))


def render_prompt(record: BenchmarkRecord) -> str:
    if record.task_type == "multiple_choice":
        if not record.choices:
            raise ValueError(f"{record.prompt_id} is missing choices")
        choices = "\n".join(
            f"{label}. {text}" for label, text in record.choices.items()
        )
        return f"{record.prompt}\n{choices}\nAnswer with only the option letter."
    return record.prompt


def smoke_test_prompt() -> BenchmarkRecord:
    return BenchmarkRecord(
        prompt_id="smoke_test:2-plus-2",
        prompt="What is 2 + 2?",
        task_type="multiple_choice",
        benchmark_name="smoke_test",
        reference_answer="B",
        choices={"A": "3", "B": "4", "C": "5"},
    )


def select_pilot_prompts(records: Sequence[BenchmarkRecord]) -> List[BenchmarkRecord]:
    """Select the fixed 15-prompt hard pilot without using model outcomes."""
    return load_hard_pilot(HARD_PILOT_DEFINITION, records)


def incorrect_mock_answer(record: BenchmarkRecord) -> str:
    if record.task_type == "multiple_choice":
        if not record.choices:
            raise ValueError(f"{record.prompt_id} is missing choices")
        return next(option for option in record.choices if option != record.reference_answer)
    if record.task_type == "math":
        return "0"
    return "print(0)"


def mock_answer(record: BenchmarkRecord, model: ModelConfig) -> str:
    identity = f"{record.prompt_id}:{model.key}"
    simulated_correct = hashlib.sha256(identity.encode("utf-8")).digest()[0] % 3 != 0
    if simulated_correct:
        if record.task_type == "code":
            # Real benchmark exports do not include solutions. Keep dry-run
            # fully offline by using a deterministic incorrect placeholder.
            return record.mock_solution or incorrect_mock_answer(record)
        return record.reference_answer
    return incorrect_mock_answer(record)


def parse_answer(record: BenchmarkRecord, raw_response: str) -> str:
    if record.task_type == "multiple_choice":
        if not record.choices:
            raise ValueError(f"{record.prompt_id} is missing choices")
        return parse_multiple_choice(raw_response, tuple(record.choices))
    if record.task_type == "math":
        return parse_math_answer(raw_response)
    if record.task_type == "code":
        return extract_code(raw_response)
    raise ValueError(f"Unsupported task type: {record.task_type}")


def grade_parsed(record: BenchmarkRecord, parsed_answer: str) -> bool:
    if record.task_type == "multiple_choice":
        if not record.choices:
            raise ValueError(f"{record.prompt_id} is missing choices")
        return grade_multiple_choice(
            parsed_answer, record.reference_answer, tuple(record.choices)
        )
    if record.task_type == "math":
        return grade_math(parsed_answer, record.reference_answer)
    if record.task_type == "code":
        return grade_code(parsed_answer, record.tests, run_code_tests)
    raise ValueError(f"Unsupported task type: {record.task_type}")


def planned_calls(
    benchmark: Sequence[BenchmarkRecord],
    models: Optional[Sequence[ModelConfig]] = None,
    *,
    max_output_tokens: Optional[int] = None,
) -> List[Tuple[ModelConfig, str, int]]:
    selected_models = tuple(models) if models is not None else enabled_models()
    return [
        (
            model,
            render_prompt(record),
            max_output_tokens or model.generation.max_output_tokens,
        )
        for record in benchmark
        for model in selected_models
    ]


def result_key(row: Dict[str, object]) -> Tuple[str, str, str]:
    return str(row["prompt_id"]), str(row["inference_provider"]), str(row["api_model_identifier"])


def is_completed_result(row: Dict[str, object]) -> bool:
    return str(row.get("status")) in PAID_TERMINAL_STATUSES


def configuration_fingerprint(model: ModelConfig, output_limit: int) -> str:
    """Non-secret request configuration; changing it releases saved model blocks."""
    settings = (model.inference_provider, model.api_model_identifier,
                model.generation.temperature_record, model.generation.reasoning_setting,
                output_limit)
    return hashlib.sha256(json.dumps(settings).encode()).hexdigest()


def saved_pair_blocks(rows, models, output_limit=None):
    """Budget exhaustion is terminal only for this pair and configuration."""
    configurations = {
        (m.inference_provider, m.api_model_identifier): configuration_fingerprint(
            m, output_limit or m.generation.max_output_tokens
        ) for m in models
    }
    return {
        result_key(row) for row in rows
        if row.get("status") == "provider_error"
        and row.get("error_type") == "output_limit_exhausted"
        and row.get("configuration_fingerprint") == configurations.get(
            (row.get("inference_provider"), row.get("api_model_identifier"))
        )
    }


def record_diagnostics(row, diagnostics, outcome):
    """Append safe metadata so a retry/resume cannot erase earlier evidence."""
    history = json.loads(row.get("provider_diagnostics") or "[]")
    history.append({**diagnostics, "provider_attempt": row["provider_attempts"],
                    "outcome": outcome})
    row["provider_diagnostics"] = json.dumps(history, sort_keys=True)


def saved_model_blocks(rows, models, output_limit=None):
    """Persist invalid/configuration blocks only; auth/billing is rechecked per run.

    Paid responses are terminal regardless of grade. Pair failures and old skipped
    rows are retryable on resume. No credential hashes are written to disk.
    """
    blocked = {}
    for model in models:
        for row in rows:
            if (row.get("status") != "provider_error"
                or row.get("error_type") not in {"invalid_model_error", "configuration_error"}
                or row.get("inference_provider") != model.inference_provider
                or row.get("api_model_identifier") != model.api_model_identifier):
                continue
            fingerprint = row.get("configuration_fingerprint")
            matches = fingerprint == configuration_fingerprint(
                model, output_limit or model.generation.max_output_tokens
            ) if fingerprint else (
                row.get("temperature") == model.generation.temperature_record
                and row.get("reasoning_setting") == model.generation.reasoning_setting
                and row.get("max_output_tokens") == (output_limit or model.generation.max_output_tokens)
            )
            if matches:
                blocked[(model.inference_provider, model.api_model_identifier)] = row
    return blocked


def historical_attempts(row):
    if "provider_attempts" in row:
        return int(row["provider_attempts"])
    if row.get("stop_reason") == "mock_completed" or row.get("status") == "skipped_model":
        return 0
    if row.get("error_type") == "spend_limit_error":
        return int(row.get("retry_count", 0))
    return 1 + int(row.get("retry_count", 0))


def historical_responses(row):
    return int(row.get("provider_response_count", int(is_completed_result(row) and row.get("stop_reason") != "mock_completed")))


def reconcile_pending_configuration(rows, models):
    """Move unpaid pending rows to corrected registry IDs without losing spend.

    Paid outputs retain their exact identity. An output containing a paid model
    removed from the plan is rejected by preflight rather than silently reused.
    """
    by_name = {m.canonical_model_name: m for m in models}
    reconciled = []
    for existing in rows:
        row = dict(existing)
        model = by_name.get(str(row["model_name"]))
        if model and not is_completed_result(row) and (
            row["inference_provider"], row["api_model_identifier"]
        ) != (model.inference_provider, model.api_model_identifier):
            row["configuration_fingerprint"] = row.get("configuration_fingerprint") or (
                f"superseded:{row['inference_provider']}:{row['api_model_identifier']}"
            )
            row.update(inference_provider=model.inference_provider,
                       api_model_identifier=model.api_model_identifier)
        reconciled.append(row)
    if reconciled:
        validate_results(reconciled)
    return reconciled


def _base_result(
    record: BenchmarkRecord,
    prompt: str,
    model: ModelConfig,
    max_output_tokens: int,
) -> Dict[str, object]:
    return {
        "prompt_id": record.prompt_id,
        "prompt": prompt,
        "benchmark": record.benchmark_name,
        "task_type": record.task_type,
        "reference_answer": record.reference_answer,
        "model_creator": model.creator,
        "model_name": model.canonical_model_name,
        "api_model_identifier": model.api_model_identifier or "UNRESOLVED",
        "inference_provider": model.inference_provider,
        "model_type": model.model_type,
        "raw_response": "",
        "parsed_answer": "",
        "score": None,
        "correct": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0,
        "latency_ms": 0.0,
        "temperature": model.generation.temperature_record,
        "reasoning_setting": model.generation.reasoning_setting,
        "max_output_tokens": max_output_tokens,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "provider_error",
        "error_type": "",
        "error_code": "",
        "error_message": "",
        "http_status": None,
        "retry_count": 0,
        "provider_request_id": "",
        "response_model_identifier": "",
        "stop_reason": "",
        "provider_attempts": 0,
        "provider_response_count": 0,
        "response_cost_usd": 0.0,
        "failure_retryable": None,
        "failure_scope": "",
        "configuration_fingerprint": configuration_fingerprint(model, max_output_tokens),
        "provider_diagnostics": "[]",
    }


def _upsert_result(
    rows: List[Dict[str, object]], row: Dict[str, object]
) -> None:
    key = result_key(row)
    for index, existing in enumerate(rows):
        if result_key(existing) == key:
            rows[index] = row
            return
    rows.append(row)


def generate_results(
    benchmark: Sequence[BenchmarkRecord],
    *,
    dry_run: bool,
    run_plan: Optional[RunPlan] = None,
    completed_results: Optional[Sequence[Dict[str, object]]] = None,
    initial_spend_usd: object = None,
    on_result: Optional[Callable[[List[Dict[str, object]]], None]] = None,
    models: Optional[Sequence[ModelConfig]] = None,
    max_output_tokens: Optional[int] = None,
    max_retries: int = 2,
    max_api_attempts: Optional[int] = None,
    print_before_call: bool = False,
    exclude_xai: bool = False,
) -> List[Dict[str, object]]:
    selected_models = tuple(models) if models is not None else enabled_models()
    results = list(completed_results or [])
    completed_keys = {
        result_key(row) for row in results if is_completed_result(row)
    }
    if not dry_run:
        completed_keys |= saved_pair_blocks(results, selected_models, max_output_tokens)
    if initial_spend_usd is None:
        initial_spend_usd = sum((Decimal(str(r["estimated_cost_usd"])) for r in results), Decimal(0))
    tracker = (
        SpendTracker(run_plan, initial_spend_usd=initial_spend_usd)
        if run_plan is not None
        else None
    )
    blocked_models = {} if dry_run else saved_model_blocks(results, selected_models, max_output_tokens)
    blocked_providers = {}
    api_attempts = 0
    stop_for_spend = False

    for record in benchmark:
        if stop_for_spend:
            break
        prompt = render_prompt(record)
        for model in selected_models:
            # Exclusion affects dispatch only. Preserve the full experiment
            # matrix, paid outputs, pending telemetry, and historical reserves.
            if exclude_xai and model.inference_provider == "xai":
                continue
            model_id = model.api_model_identifier or "UNRESOLVED"
            key = (record.prompt_id, model.inference_provider, model_id)
            if key in completed_keys:
                continue
            output_limit = max_output_tokens or model.generation.max_output_tokens
            row = _base_result(record, prompt, model, output_limit)
            prior_row = next((r for r in results if result_key(r) == key), {})
            row["provider_diagnostics"] = prior_row.get("provider_diagnostics") or "[]"
            row["provider_attempts"] = historical_attempts(prior_row) if prior_row else 0
            row["provider_response_count"] = historical_responses(prior_row) if prior_row else 0
            prior_recorded_cost = next(
                (
                    Decimal(str(existing.get("estimated_cost_usd", 0)))
                    for existing in results
                    if result_key(existing) == key
                ),
                Decimal("0"),
            )

            block = blocked_providers.get(model.inference_provider) or blocked_models.get((model.inference_provider, model_id))
            if block:
                # Preserve the original diagnostic row and all historical spend.
                if prior_row.get("status") == "provider_error" and prior_row.get("error_type") in {"invalid_model_error", "configuration_error"}:
                    continue
                row.update(
                    status="skipped_model",
                    error_type="blocked_by_permanent_failure",
                    error_message=f"No call: {block['error_type']}: {block['error_message']}",
                    failure_scope=block.get("failure_scope") or "model",
                    estimated_cost_usd=float(prior_recorded_cost),
                )
                _upsert_result(results, row)
                if on_result:
                    on_result(results)
                continue

            spent_before = tracker.spent_usd if tracker else Decimal("0")
            started = perf_counter()
            pair_attempts = 0

            def before_attempt() -> None:
                nonlocal api_attempts, pair_attempts
                if max_api_attempts is not None and api_attempts >= max_api_attempts:
                    raise SpendLimitError(
                        f"Run stopped before exceeding its {max_api_attempts}-attempt limit."
                    )
                if tracker:
                    tracker.assert_can_call(model, prompt, output_limit)
                api_attempts += 1
                pair_attempts += 1
                row["provider_attempts"] += 1
                if print_before_call:
                    print(
                        f"Calling model: {model.canonical_model_name} "
                        f"[{model_id}] via {provider_display_name(model.inference_provider)} "
                        f"(attempt {api_attempts})",
                        flush=True,
                    )

            def failed_attempt(attempt_error: ProviderError) -> None:
                # Connection failures and malformed responses may have billed
                # inference. Auth/model/parameter rejections do not run inference.
                if tracker and attempt_error.may_have_been_billed:
                    tracker.record_failed_attempt(model, prompt, output_limit)
                diagnostics = attempt_error.diagnostics
                record_diagnostics(row, diagnostics, attempt_error.error_type)
                row.update(
                    input_tokens=diagnostics.get("input_tokens") or 0,
                    output_tokens=diagnostics.get("output_tokens") or 0,
                    stop_reason=diagnostics.get("stop_reason") or "",
                    provider_request_id=diagnostics.get("response_id") or diagnostics.get("request_id") or "",
                    response_model_identifier=diagnostics.get("response_model_identifier") or "",
                )
                # Checkpoint attempts and reserved cost before a possible retry.
                row.update(
                    status="provider_error", error_type=attempt_error.error_type,
                    error_code=attempt_error.error_code, error_message=str(attempt_error),
                    http_status=attempt_error.status_code,
                    failure_retryable=attempt_error.retryable, failure_scope=attempt_error.failure_scope,
                    retry_count=max(0, pair_attempts - 1),
                    estimated_cost_usd=float(prior_recorded_cost + (tracker.spent_usd - spent_before if tracker else Decimal(0))),
                    latency_ms=(perf_counter() - started) * 1000,
                )
                _upsert_result(results, row)
                if on_result:
                    on_result(results)

            try:
                response, retry_count = call_model_with_retries(
                    model,
                    prompt,
                    dry_run=dry_run,
                    mock_response=mock_answer(record, model) if dry_run else None,
                    max_output_tokens=output_limit,
                    max_retries=0 if dry_run else max_retries,
                    before_attempt=None if dry_run else before_attempt,
                    on_failed_attempt=failed_attempt,
                )
                if tracker:
                    try:
                        tracker.record_call(model, response)
                    except SpendCapExceeded:
                        # A returned response is paid even if its reported cost
                        # stops the run. Save/grade it and never dispatch it twice.
                        stop_for_spend = True
                    call_cost = tracker.spent_usd - spent_before
                else:
                    call_cost = Decimal("0")
                row.update(
                    raw_response=response.text,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    estimated_cost_usd=float(prior_recorded_cost + call_cost),
                    latency_ms=response.latency_ms,
                    retry_count=retry_count,
                    provider_request_id=response.request_id,
                    response_model_identifier=response.response_model_identifier,
                    stop_reason=response.stop_reason,
                    provider_response_count=row["provider_response_count"] + (0 if dry_run else 1),
                    response_cost_usd=float(
                        response.provider_reported_cost_usd
                        if response.provider_reported_cost_usd is not None else
                        (Decimal(response.input_tokens) * Decimal(str(model.input_price_per_million_usd))
                         + Decimal(response.output_tokens) * Decimal(str(model.output_price_per_million_usd))) / Decimal(1000000)
                    ),
                )
                row.update(failure_retryable=None, failure_scope="", error_code="", http_status=None)
                record_diagnostics(row, response.diagnostics, "success")
            except ProviderError as error:
                row.update(
                    status="provider_error",
                    error_type=error.error_type,
                    error_code=error.error_code,
                    error_message=str(error),
                    http_status=error.status_code,
                    failure_retryable=error.retryable,
                    failure_scope=error.failure_scope,
                    retry_count=int(getattr(error, "retry_count", 0)),
                    latency_ms=(perf_counter() - started) * 1000,
                    estimated_cost_usd=float(
                        prior_recorded_cost
                        + ((tracker.spent_usd - spent_before) if tracker else Decimal("0"))
                    ),
                )
                if error.failure_scope == "model":
                    blocked_models[(model.inference_provider, model_id)] = row
                elif error.failure_scope == "provider":
                    blocked_providers[model.inference_provider] = row
                _upsert_result(results, row)
                if on_result:
                    on_result(results)
                label = "INVALID MODEL" if error.invalid_model else "PROVIDER FAILURE"
                print(
                    f"{label}: {model.canonical_model_name} [{model_id}] via "
                    f"{provider_display_name(model.inference_provider)}: {error}",
                    flush=True,
                )
                if tracker:
                    print(f"Accumulated spend: ${tracker.spent_usd:.6f}", flush=True)
                continue
            except SpendLimitError as error:
                row.update(
                    status="provider_error",
                    error_type="spend_limit_error",
                    error_message=str(error),
                    latency_ms=(perf_counter() - started) * 1000,
                    estimated_cost_usd=float(
                        prior_recorded_cost
                        + ((tracker.spent_usd - spent_before) if tracker else Decimal("0"))
                    ),
                )
                _upsert_result(results, row)
                if on_result:
                    on_result(results)
                stop_for_spend = True
                break

            # Persist paid raw output before parsing/grading, which can crash.
            row.update(status="grading_failure", error_type="grading_failure",
                       error_message="Response saved; grading not completed. Regrade offline.")
            _upsert_result(results, row)
            if on_result:
                on_result(results)
            try:
                parsed = parse_answer(record, response.text)
                row["parsed_answer"] = parsed
            except MalformedModelOutput as error:
                row.update(
                    status="parsing_failure",
                    error_type="parsing_failure",
                    error_message=str(error),
                )
            else:
                try:
                    correct = grade_parsed(record, parsed)
                except Exception as error:  # A grader failure is metadata, never a wrong answer.
                    row.update(
                        status="grading_failure",
                        error_type="grading_failure",
                        error_message=f"{type(error).__name__}: {error}",
                    )
                else:
                    row.update(
                        status="success",
                        score=1.0 if correct else 0.0,
                        correct=correct,
                        error_type="",
                        error_message="",
                    )

            _upsert_result(results, row)
            completed_keys.add(key)
            if on_result:
                on_result(results)
            if tracker:
                print(f"Accumulated spend: ${tracker.spent_usd:.6f}", flush=True)
            if stop_for_spend:
                break

    expected_task_types = {
        record.benchmark_name: record.task_type for record in benchmark
    }
    # Existing rows can include another mode's benchmark only if a caller chose a
    # shared output path; validate each row structurally in that unusual case.
    validate_results(results, expected_task_types if all(str(r["benchmark"]) in expected_task_types for r in results) else None)
    return results


def write_results(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_results(path: Path) -> List[Dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            typed: Dict[str, object] = dict(row)
            for field in ("max_output_tokens", "input_tokens", "output_tokens", "retry_count"):
                typed[field] = int(str(typed[field]))
            typed["http_status"] = int(str(typed["http_status"])) if typed["http_status"] else None
            for field in ("estimated_cost_usd", "latency_ms"):
                typed[field] = float(str(typed[field]))
            for field in ("provider_attempts", "provider_response_count"):
                if typed.get(field) not in (None, ""):
                    typed[field] = int(str(typed[field]))
                else:
                    typed.pop(field, None)
            if typed.get("response_cost_usd") not in (None, ""):
                typed["response_cost_usd"] = float(str(typed["response_cost_usd"]))
            else:
                typed.pop("response_cost_usd", None)
            typed["failure_retryable"] = (
                str(typed["failure_retryable"]).lower() == "true" if typed.get("failure_retryable") else None
            )
            typed["score"] = float(str(typed["score"])) if typed["score"] else None
            typed["correct"] = (
                str(typed["correct"]).lower() == "true" if typed["correct"] else None
            )
            rows.append(typed)
    validate_results(rows)
    return rows


def regrade_results(
    rows: Sequence[Dict[str, object]],
    records: Sequence[BenchmarkRecord],
) -> List[Dict[str, object]]:
    """Reparse and regrade saved raw responses without calling a provider."""
    records_by_id = {record.prompt_id: record for record in records}
    regraded = []
    for existing in rows:
        row = dict(existing)
        prompt_id = str(row["prompt_id"])
        if prompt_id not in records_by_id:
            raise ValueError(f"Saved result has an unexpected prompt ID: {prompt_id}")
        record = records_by_id[prompt_id]
        raw_response = str(row.get("raw_response", ""))
        if not raw_response.strip():
            regraded.append(row)
            continue

        row.update(
            parsed_answer="",
            score=None,
            correct=None,
            status="grading_failure",
            error_type="",
            error_code="",
            error_message="",
        )
        try:
            parsed = parse_answer(record, raw_response)
            row["parsed_answer"] = parsed
        except MalformedModelOutput as error:
            row.update(
                status="parsing_failure",
                error_type="parsing_failure",
                error_message=str(error),
            )
        else:
            try:
                correct = grade_parsed(record, parsed)
            except Exception as error:
                row.update(
                    status="grading_failure",
                    error_type="grading_failure",
                    error_message=f"{type(error).__name__}: {error}",
                )
            else:
                row.update(
                    status="success",
                    score=1.0 if correct else 0.0,
                    correct=correct,
                )
        regraded.append(row)

    expected_task_types = {
        record.benchmark_name: record.task_type for record in records
    }
    validate_results(regraded, expected_task_types)
    return regraded


def benchmark_records_from_saved_results(
    rows: Sequence[Dict[str, object]],
) -> List[BenchmarkRecord]:
    """Reconstruct grading inputs embedded in a historical result CSV."""
    records = []
    seen = set()
    for row in rows:
        prompt_id = str(row["prompt_id"])
        if prompt_id in seen:
            continue
        seen.add(prompt_id)
        task_type = str(row["task_type"])
        choices = None
        tests = ()
        if task_type == "multiple_choice":
            labels = re.findall(r"(?m)^([A-Z])\.\s", str(row["prompt"]))
            choices = {label: label for label in labels}
        elif task_type == "code":
            decoded = json.loads(str(row["reference_answer"]))
            tests = tuple(
                {"input": str(test["input"]), "output": str(test["output"])}
                for test in decoded
            )
        records.append(
            BenchmarkRecord(
                prompt_id=prompt_id,
                prompt=str(row["prompt"]),
                task_type=task_type,
                benchmark_name=str(row["benchmark"]),
                reference_answer=str(row["reference_answer"]),
                choices=choices,
                tests=tests,
            )
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true", help="Use offline deterministic responses.")
    modes.add_argument("--smoke-test", action="store_true", help="Run one prompt against five models.")
    modes.add_argument(
        "--hard-pilot", "--pilot", dest="pilot", action="store_true",
        help="Run the confirmed 15-prompt, 75-pair hard pilot with up to two retries per pair.",
    )
    modes.add_argument(
        "--regrade-pilot", "--regrade-results", dest="regrade_pilot",
        action="store_true",
        help="Regrade saved CSV responses offline; use --output for the hard pilot.",
    )
    modes.add_argument("--live", action="store_true", help="Run all enabled benchmark prompts live.")
    parser.add_argument(
        "--max-spend-usd",
        "--spend-cap",
        dest="max_spend_usd",
        default=None,
        help="Per-run USD cap (default $1.00; smoke default $0.25).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Authorize --live or --hard-pilot after reviewing the spend plan.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Document that an existing output should be resumed (live modes resume automatically).",
    )
    parser.add_argument(
        "--exclude-xai", action="store_true",
        help="Exclude xAI from dispatch while retaining its rows and cumulative recorded spend.",
    )
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/manifest.json"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _default_output(args: argparse.Namespace) -> Path:
    if args.smoke_test:
        return Path("data/results/smoke_test_results.csv")
    if args.regrade_pilot:
        return Path("data/results/pilot_results.csv")
    if args.pilot:
        return Path("data/results/hard_pilot_results.csv")
    if args.dry_run:
        return Path("data/results/dry_run_results.csv")
    return Path("data/results/live_results.csv")


def main() -> None:
    args = parse_args()
    validate_registry()
    models = enabled_models()
    _, all_records = load_enabled_benchmarks(args.manifest)

    if args.regrade_pilot:
        output = args.output or _default_output(args)
        if not output.exists():
            raise SystemExit(f"Pilot results do not exist: {output}")
        existing_results = read_results(output)
        benchmark = benchmark_records_from_saved_results(existing_results)
        results = regrade_results(existing_results, benchmark)
        write_results(results, output)
        status_counts = {
            status: sum(str(row["status"]) == status for row in results)
            for status in ("success", "parsing_failure", "grading_failure")
        }
        print(f"Regraded {len(results)} existing pilot rows from saved raw responses")
        print(
            "Statuses: "
            + ", ".join(f"{status}={count}" for status, count in status_counts.items())
        )
        print(f"Wrote offline regraded results to {output}")
        print("Offline regrade complete: no provider API calls were made")
        return

    load_dotenv()
    validate_provider_registry(models)

    if args.smoke_test:
        benchmark = [smoke_test_prompt()]
        output_limit = SMOKE_MAX_OUTPUT_TOKENS
        spend_cap = args.max_spend_usd or "0.25"
        max_retries = 0
        max_attempts = SMOKE_MAX_API_CALLS
    elif args.pilot:
        try:
            benchmark = select_pilot_prompts(all_records)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        output_limit = None
        spend_cap = args.max_spend_usd or "1.00"
        max_retries = HARD_PILOT_MAX_RETRIES
        max_attempts = HARD_PILOT_MAX_ATTEMPTS
    else:
        benchmark = list(all_records)
        output_limit = None
        spend_cap = args.max_spend_usd or "1.00"
        max_retries = 2
        max_attempts = None

    output = args.output or _default_output(args)
    live_mode = not args.dry_run
    existing_results = reconcile_pending_configuration(read_results(output), models) if live_mode and output.exists() else []
    plan_keys = {(r.prompt_id, m.inference_provider, m.api_model_identifier or "UNRESOLVED") for r in benchmark for m in models}
    if any(result_key(r) not in plan_keys for r in existing_results):
        raise SystemExit("Existing CSV contains pairs outside this run plan; choose the matching mode/output.")
    completed_keys = {
        result_key(row) for row in existing_results if is_completed_result(row)
    }
    completed_keys |= saved_pair_blocks(existing_results, models, output_limit)
    persistent_blocks = saved_model_blocks(existing_results, models, output_limit)
    remaining = [
        call
        for record in benchmark
        for call in planned_calls((record,), models, max_output_tokens=output_limit)
        if (record.prompt_id, call[0].inference_provider, call[0].api_model_identifier or "UNRESOLVED") not in completed_keys
        and (call[0].inference_provider, call[0].api_model_identifier) not in persistent_blocks
        and not (args.exclude_xai and call[0].inference_provider == "xai")
    ]
    if args.smoke_test and len(remaining) > SMOKE_MAX_API_CALLS:
        raise SystemExit("Smoke test safety invariant failed: more than 5 calls selected")
    if args.pilot and len(planned_calls(benchmark, models)) != HARD_PILOT_PAIR_COUNT:
        raise SystemExit("Pilot safety invariant failed: selection is not exactly 15 prompts x 5 models")
    if args.pilot and len({(r.prompt_id, m.inference_provider, m.api_model_identifier) for r in benchmark for m in models}) != HARD_PILOT_PAIR_COUNT:
        raise SystemExit("Pilot safety invariant failed: pairs are not unique")

    run_plan = None
    if live_mode:
        try:
            if args.pilot and run_spend_cap_usd(spend_cap) > HARD_PILOT_MAX_SPEND_USD:
                raise SystemExit("Hard pilot cap cannot exceed $2.00.")
            run_plan = build_run_plan(remaining, spend_cap=spend_cap, max_retries=max_retries)
        except SpendLimitError as error:
            raise SystemExit(str(error)) from error
        print(run_plan.confirmation_text())
        if args.exclude_xai:
            deferred_xai_pairs = sum(
                (record.prompt_id, model.inference_provider, model.api_model_identifier) not in completed_keys
                for record in benchmark for model in models if model.inference_provider == "xai"
            )
            print(f"xAI dispatch excluded: {deferred_xai_pairs} pending pairs deferred; "
                  "all xAI rows and historical recorded spend remain in the pilot ledger.")
        recorded_spend = sum(Decimal(str(r["estimated_cost_usd"])) for r in existing_results)
        print(f"Previously recorded spend: ${recorded_spend:.6f}")
        maximum_total = recorded_spend + run_plan.estimated_max_cost_usd
        print(f"Conservative maximum total spend: ${maximum_total:.6f}")
        if maximum_total > run_plan.configured_spend_cap_usd:
            print("Conservative maximum exceeds the configured cap; completion is not guaranteed. "
                  "The runtime spend guard will stop before dispatching an attempt that could exceed the cap.")
        print(f"Retry policy: {max_retries} retries; attempt ceiling: {max_attempts or 'uncapped'}")
        if args.pilot:
            print(f"Hard pilot selection: {len(benchmark)} prompts x {len(models)} models = 75 pairs")
        if (args.live or args.pilot) and not args.confirm:
            raise SystemExit("Live run not started: pass --confirm after reviewing the plan.")

    initial_spend = sum(
        (Decimal(str(row["estimated_cost_usd"])) for row in existing_results),
        Decimal("0"),
    )
    results = generate_results(
        benchmark,
        dry_run=args.dry_run,
        run_plan=run_plan,
        completed_results=existing_results,
        initial_spend_usd=initial_spend,
        on_result=lambda rows: write_results(rows, output),
        models=models,
        max_output_tokens=output_limit,
        max_retries=max_retries,
        max_api_attempts=max_attempts,
        print_before_call=live_mode,
        exclude_xai=args.exclude_xai,
    )
    write_results(results, output)

    print(f"Selected {len(benchmark)} prompts and {len(models)} models")
    print(f"Validated {len(results)} rows with {len(RESULT_FIELDS)} required fields")
    print(f"Wrote normalized results to {output}")
    from pilot_analysis import analyze_results
    report = analyze_results(
        results, [r.prompt_id for r in benchmark],
        [(m.inference_provider, m.api_model_identifier) for m in models],
        {r.prompt_id: r.benchmark_name for r in benchmark},
    )
    print(json.dumps({key: report[key] for key in ("matrix_completeness", "by_model")}, indent=2, sort_keys=True))
    if args.dry_run:
        print("Dry run complete: no API calls were made; metrics are synthetic")
    else:
        print("Live mode complete: results were saved after every model/prompt outcome")


if __name__ == "__main__":
    main()
