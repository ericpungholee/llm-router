"""One small call interface shared by all hosted inference providers."""

import hashlib
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

from model_registry import ModelConfig


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

PROVIDER_DISPLAY_NAMES = {
    "openrouter": "OpenRouter",
}


def provider_display_name(provider: str) -> str:
    return PROVIDER_DISPLAY_NAMES.get(provider, provider)

MOCK_BASE_LATENCY_MS = {
    "openai": 700,
    "anthropic": 800,
    "google": 600,
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


def _token_estimate(text: str) -> int:
    return max(1, round(len(text) / 4))


def _mock_call(
    model: ModelConfig, prompt: str, response_text: str
) -> ProviderResponse:
    jitter = hashlib.sha256(
        f"{model.key}:{prompt}".encode("utf-8")
    ).digest()[0]
    return ProviderResponse(
        text=response_text,
        input_tokens=_token_estimate(prompt),
        output_tokens=_token_estimate(response_text),
        latency_ms=float(MOCK_BASE_LATENCY_MS[model.inference_provider] + jitter),
    )


def _not_implemented(
    model: ModelConfig, prompt: str, api_key: str
) -> ProviderResponse:
    del prompt, api_key
    raise NotImplementedError(
        f"Live {model.inference_provider} calls are not enabled in Phase 2"
    )


LIVE_CALLERS: Dict[str, Callable[[ModelConfig, str, str], ProviderResponse]] = {
    "openai": _not_implemented,
    "anthropic": _not_implemented,
    "google": _not_implemented,
    "xai": _not_implemented,
    "deepseek": _not_implemented,
    "openrouter": _not_implemented,
}


def call_model(
    model: ModelConfig,
    prompt: str,
    *,
    dry_run: bool,
    mock_response: Optional[str] = None,
) -> ProviderResponse:
    """Call one hosted model, or simulate the same response shape locally."""
    provider = model.inference_provider
    if provider not in API_KEY_ENV or provider not in LIVE_CALLERS:
        raise ValueError(f"Unknown provider: {provider}")

    if dry_run:
        if mock_response is None:
            raise ValueError("mock_response is required in dry-run mode")
        return _mock_call(model, prompt, mock_response)

    if not model.live_ready:
        raise RuntimeError(
            f"{model.canonical_model_name} has unresolved API ID or pricing metadata"
        )
    key_name = API_KEY_ENV[provider]
    api_key = os.getenv(key_name)
    if not api_key:
        raise RuntimeError(f"Missing required environment variable: {key_name}")
    return LIVE_CALLERS[provider](model, prompt, api_key)
