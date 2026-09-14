"""One guarded call interface shared by all hosted inference providers."""

import hashlib
import os
import time
from typing import Callable, Dict, Optional, Sequence, Tuple

from model_registry import ModelConfig
from provider_clients import (
    call_anthropic,
    call_deepseek,
    call_openai,
    call_openrouter,
    call_xai,
)
from provider_clients.base import ProviderError, ProviderResponse


API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "xai": "xAI",
    "deepseek": "DeepSeek",
    "openrouter": "OpenRouter",
}

# These are the identifiers implemented and checked against the central registry.
# Adapters never replace one with an alias or fallback model.
SUPPORTED_MODEL_IDS = {
    "openai": frozenset(("gpt-5.6-sol",)),
    "anthropic": frozenset(("claude-opus-5",)),
    "xai": frozenset(("grok-4.6",)),
    "deepseek": frozenset(("deepseek-flash",)),
    "openrouter": frozenset(("qwen/qwen3.8-2.4t-a95b",)),
}


def provider_display_name(provider: str) -> str:
    return PROVIDER_DISPLAY_NAMES.get(provider, provider)


MOCK_BASE_LATENCY_MS = {
    "openai": 700,
    "anthropic": 800,
    "xai": 650,
    "deepseek": 500,
    "openrouter": 600,
}


def estimate_cost_usd(
    model: ModelConfig, input_tokens: int, output_tokens: int
) -> Optional[float]:
    if (
        model.input_price_per_million_usd is None
        or model.output_price_per_million_usd is None
    ):
        return None
    return (
        input_tokens * model.input_price_per_million_usd
        + output_tokens * model.output_price_per_million_usd
    ) / 1_000_000


def missing_api_keys(models: Sequence[ModelConfig]) -> Tuple[str, ...]:
    key_names = {
        API_KEY_ENV[model.inference_provider]
        for model in models
        if model.inference_provider in API_KEY_ENV
    }
    return tuple(sorted(key_name for key_name in key_names if not os.getenv(key_name)))


def validate_provider_registry(models: Sequence[ModelConfig]) -> None:
    """Verify every enabled registry pair has one exact implemented API ID."""
    for model in models:
        provider = model.inference_provider
        if provider not in LIVE_CALLERS or provider not in API_KEY_ENV:
            raise ValueError(f"{model.key} has unsupported provider {provider!r}")
        model_id = model.api_model_identifier
        if not model_id:
            raise ValueError(f"{model.key} has no API model identifier")
        if model_id not in SUPPORTED_MODEL_IDS.get(provider, frozenset()):
            raise ValueError(
                f"{model.key} API identifier {model_id!r} is not implemented for {provider}; "
                "refusing to substitute a different model"
            )


def _token_estimate(text: str) -> int:
    return max(1, round(len(text) / 4))


def _mock_call(model: ModelConfig, prompt: str, response_text_value: str) -> ProviderResponse:
    jitter = hashlib.sha256(f"{model.key}:{prompt}".encode("utf-8")).digest()[0]
    return ProviderResponse(
        text=response_text_value,
        input_tokens=_token_estimate(prompt),
        output_tokens=_token_estimate(response_text_value),
        latency_ms=float(MOCK_BASE_LATENCY_MS[model.inference_provider] + jitter),
        response_model_identifier=model.api_model_identifier or "",
        stop_reason="mock_completed",
    )


LiveCaller = Callable[[ModelConfig, str, str, int], ProviderResponse]

LIVE_CALLERS: Dict[str, LiveCaller] = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "xai": call_xai,
    "deepseek": call_deepseek,
    "openrouter": call_openrouter,
}


def call_model(
    model: ModelConfig,
    prompt: str,
    *,
    dry_run: bool,
    mock_response: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
) -> ProviderResponse:
    """Call one exact hosted model once, or simulate the shape locally."""
    provider = model.inference_provider
    if provider not in API_KEY_ENV or provider not in LIVE_CALLERS:
        raise ValueError(f"Unknown provider: {provider}")

    if dry_run:
        if mock_response is None:
            raise ValueError("mock_response is required in dry-run mode")
        return _mock_call(model, prompt, mock_response)

    validate_provider_registry((model,))
    if not model.live_ready:
        raise RuntimeError(
            f"{model.canonical_model_name} has unresolved API ID or pricing metadata"
        )
    key_name = API_KEY_ENV[provider]
    api_key = os.getenv(key_name)
    if not api_key:
        raise ProviderError(
            provider,
            model.api_model_identifier or "",
            f"Missing required environment variable: {key_name}",
            error_type="authentication_error",
        )
    output_limit = max_output_tokens or model.generation.max_output_tokens
    if output_limit <= 0 or output_limit > model.generation.max_output_tokens:
        raise ValueError("max_output_tokens must be within the registry limit")
    return LIVE_CALLERS[provider](model, prompt, api_key, output_limit)


def call_model_with_retries(
    model: ModelConfig,
    prompt: str,
    *,
    dry_run: bool,
    mock_response: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    max_retries: int = 2,
    before_attempt: Optional[Callable[[], None]] = None,
    on_failed_attempt: Optional[Callable[[ProviderError], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Tuple[ProviderResponse, int]:
    """Call sequentially, retrying transient errors at most ``max_retries`` times."""
    if max_retries < 0 or max_retries > 2:
        raise ValueError("max_retries must be between 0 and 2")
    retries = 0
    while True:
        if before_attempt is not None:
            before_attempt()
        try:
            return (
                call_model(
                    model,
                    prompt,
                    dry_run=dry_run,
                    mock_response=mock_response,
                    max_output_tokens=max_output_tokens,
                ),
                retries,
            )
        except KeyboardInterrupt as interrupt:
            # The call has started and may have dispatched/billed. Do not retry
            # here; checkpoint through the same reservation path as a timeout.
            error = ProviderError(
                model.inference_provider, model.api_model_identifier or "",
                "Interrupted during provider call; maximum possible cost reserved.",
                error_type="interrupted_inflight", retryable=True,
                diagnostics=getattr(interrupt, "diagnostics", {}),
            )
            if on_failed_attempt is not None:
                on_failed_attempt(error)
            raise
        except ProviderError as error:
            if on_failed_attempt is not None:
                on_failed_attempt(error)
            if not error.retryable or retries >= max_retries:
                error.retry_count = retries
                raise
            retries += 1
            sleep(float(2 ** (retries - 1)))


__all__ = (
    "API_KEY_ENV",
    "LIVE_CALLERS",
    "ProviderError",
    "ProviderResponse",
    "SUPPORTED_MODEL_IDS",
    "call_model",
    "call_model_with_retries",
    "estimate_cost_usd",
    "missing_api_keys",
    "provider_display_name",
    "validate_provider_registry",
)
