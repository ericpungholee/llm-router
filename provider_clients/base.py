"""Shared HTTP and response primitives for provider adapters."""

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Mapping, Optional


DEFAULT_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    request_id: str = ""
    response_model_identifier: str = ""
    stop_reason: str = ""
    provider_reported_cost_usd: Optional[float] = None


class ProviderError(RuntimeError):
    """A normalized error that retains the provider's exact response text."""

    def __init__(
        self,
        provider: str,
        model_identifier: str,
        message: str,
        *,
        status_code: Optional[int] = None,
        error_type: str = "provider_error",
        error_code: str = "",
        retryable: bool = False,
        invalid_model: bool = False,
        failure_scope: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model_identifier = model_identifier
        self.status_code = status_code
        self.error_type = error_type
        self.error_code = error_code
        self.retryable = retryable
        self.invalid_model = invalid_model
        # Scope belongs to the normalized provider error, not benchmark logic.
        # Unknown rejections (including prompt-specific bad parameters) stay local.
        self.failure_scope = "pair" if retryable else (failure_scope or {
            "invalid_model_error": "model",
            "configuration_error": "model",
            "authentication_error": "provider",
            "quota_or_billing_error": "model",
        }.get(error_type, "model" if invalid_model else "pair"))
        if self.failure_scope not in {"pair", "model", "provider"}:
            raise ValueError("Invalid provider failure scope")


def _quota_scope(status: int, code: str, kind: str, message: str) -> str:
    """Only explicit account billing exhaustion disables the whole provider.

    A zero model/resource quota does not imply every model on an account is
    unusable. Ordinary rate limits never reach this function as permanent errors.
    """
    text = " ".join((code, kind, message)).lower()
    # Request affordability is not an empty account: cheaper models may work.
    if any(marker in text for marker in ("fewer max_tokens", "can only afford")):
        return "model"
    if status == 402 or any(marker in text for marker in (
        "insufficient_quota", "insufficient_credits", "credit balance",
        "insufficient balance", "billing", "current quota",
    )):
        return "provider"
    return "model"


def _error_details(body: str) -> tuple:
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return "", "", body.strip()
    error = parsed.get("error", parsed) if isinstance(parsed, dict) else parsed
    if isinstance(error, dict):
        code = str(error.get("code") or error.get("status") or "")
        kind = str(error.get("type") or error.get("status") or "")
        message = str(error.get("message") or body).strip()
        return code, kind, message
    return "", "", body.strip()


def _classify_http_error(status: int, code: str, kind: str, message: str) -> tuple:
    searchable = " ".join((code, kind, message)).lower()
    if status in (401, 403):
        return "authentication_error", False, False
    if status == 402 or any(marker in searchable for marker in (
        "insufficient_quota", "insufficient_credits", "credit balance is too low",
        "insufficient balance",
    )) or (
        "quota exceeded" in searchable
        and any(marker in searchable for marker in ("limit: 0", "billing", "current quota"))
    ):
        return "quota_or_billing_error", False, False
    invalid_model = (
        status == 404
        or any(
            marker in searchable
            for marker in ("model_not_found", "invalid_model", "unknown_model", "no endpoints found")
        )
        or (
            "model" in searchable
            and any(
                term in searchable
                for term in ("invalid", "not found", "unknown", "not exist", "does not exist")
            )
        )
    )
    if invalid_model:
        return "invalid_model_error", False, True
    if status in (400, 413, 422):
        return "invalid_parameter_error", False, False
    if status in (408, 409, 425, 429) or 500 <= status <= 599:
        return "transient_provider_error", True, False
    return "provider_error", False, False


def post_json(
    *,
    provider: str,
    model_identifier: str,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, object],
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, object]:
    request_headers = {"Content-Type": "application/json", **headers}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        code, provider_type, exact_message = _error_details(body)
        error_type, retryable, invalid_model = _classify_http_error(
            error.code, code, provider_type, exact_message
        )
        raise ProviderError(
            provider,
            model_identifier,
            exact_message or str(error),
            status_code=error.code,
            error_type=error_type,
            error_code=code,
            retryable=retryable,
            invalid_model=invalid_model,
            failure_scope=(
                _quota_scope(error.code, code, provider_type, exact_message)
                if error_type == "quota_or_billing_error" else None
            ),
        ) from error
    except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
        raise ProviderError(
            provider,
            model_identifier,
            str(error),
            error_type="network_error",
            retryable=True,
        ) from error

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as error:
        raise ProviderError(
            provider,
            model_identifier,
            f"Provider returned invalid JSON: {body[:1000]}",
            error_type="invalid_provider_response",
        ) from error
    if not isinstance(parsed, dict):
        raise ProviderError(
            provider,
            model_identifier,
            "Provider returned a non-object JSON response",
            error_type="invalid_provider_response",
        )
    return parsed


def require_int(data: Mapping[str, object], field: str, provider: str, model_id: str) -> int:
    value = data.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProviderError(
            provider,
            model_id,
            f"Provider response is missing valid {field}",
            error_type="invalid_provider_response",
        )
    return value


def require_text(value: object, provider: str, model_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProviderError(
            provider,
            model_id,
            "Provider response did not contain non-empty text output",
            error_type="invalid_provider_response",
        )
    return value


def response_text(data: Mapping[str, object], provider: str, model_id: str) -> str:
    blocks = data.get("output")
    texts = []
    if isinstance(blocks, list):
        for item in blocks:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                    text = part.get("text")
                    if isinstance(text, str):
                        texts.append(text)
    return require_text("\n".join(texts), provider, model_id)
