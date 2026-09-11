"""Small common interface for hosted LLM providers."""

import hashlib
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class ModelSpec:
    model_creator: str
    model_name: str
    inference_provider: str
    model_type: str


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    latency_ms: float


# Model names remain placeholders until the first real evaluation sweep is designed.
MODEL_SPECS = (
    ModelSpec("OpenAI", "openai-model-tbd", "openai", "closed"),
    ModelSpec("Anthropic", "anthropic-model-tbd", "anthropic", "closed"),
    ModelSpec("Google", "gemini-model-tbd", "google", "closed"),
    ModelSpec("DeepSeek", "deepseek-model-tbd", "deepseek", "closed"),
    ModelSpec(
        "Open-weight creator TBD",
        "hosted-open-weight-model-tbd",
        "groq",
        "open_weight",
    ),
)

API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq": "GROQ_API_KEY",
}

# Synthetic rates and latency are used only to exercise the dry-run data path.
MOCK_COST_PER_MILLION_TOKENS = {
    "openai": (2.00, 8.00),
    "anthropic": (3.00, 15.00),
    "google": (1.00, 4.00),
    "deepseek": (0.30, 1.00),
    "groq": (0.10, 0.20),
}
MOCK_BASE_LATENCY_MS = {
    "openai": 700,
    "anthropic": 800,
    "google": 600,
    "deepseek": 500,
    "groq": 250,
}


def _token_estimate(text: str) -> int:
    return max(1, round(len(text) / 4))


def _mock_call(model: ModelSpec, prompt: str, response_text: str) -> ProviderResponse:
    input_tokens = _token_estimate(prompt)
    output_tokens = _token_estimate(response_text)
    input_rate, output_rate = MOCK_COST_PER_MILLION_TOKENS[
        model.inference_provider
    ]
    cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
    jitter = hashlib.sha256(
        f"{model.inference_provider}:{prompt}".encode("utf-8")
    ).digest()[0]
    latency_ms = MOCK_BASE_LATENCY_MS[model.inference_provider] + jitter
    return ProviderResponse(
        text=response_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=cost,
        latency_ms=float(latency_ms),
    )


def _not_implemented(
    model: ModelSpec, prompt: str, api_key: str
) -> ProviderResponse:
    del model, prompt, api_key
    raise NotImplementedError(
        "Live provider calls are intentionally not implemented in Phase 1"
    )


LIVE_CALLERS: Dict[str, Callable[[ModelSpec, str, str], ProviderResponse]] = {
    "openai": _not_implemented,
    "anthropic": _not_implemented,
    "google": _not_implemented,
    "deepseek": _not_implemented,
    "groq": _not_implemented,
}


def call_model(
    model: ModelSpec,
    prompt: str,
    *,
    dry_run: bool,
    mock_response: Optional[str] = None,
) -> ProviderResponse:
    """Call one model through a shared interface, or simulate it in dry-run mode."""
    if model.inference_provider not in API_KEY_ENV:
        raise ValueError(f"Unknown provider: {model.inference_provider}")

    if dry_run:
        if mock_response is None:
            raise ValueError("mock_response is required in dry-run mode")
        return _mock_call(model, prompt, mock_response)

    key_name = API_KEY_ENV[model.inference_provider]
    api_key = os.getenv(key_name)
    if not api_key:
        raise RuntimeError(f"Missing required environment variable: {key_name}")
    return LIVE_CALLERS[model.inference_provider](model, prompt, api_key)

