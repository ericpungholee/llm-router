"""Anthropic Messages API adapter."""

from time import perf_counter

from model_registry import ModelConfig
from provider_clients.base import ProviderResponse, post_json
from provider_clients.normalization import normalize_response


def call_anthropic(model: ModelConfig, prompt: str, api_key: str, max_tokens: int) -> ProviderResponse:
    started = perf_counter()
    data = post_json(
        provider="anthropic",
        model_identifier=model.api_model_identifier or "",
        url="https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        payload={
            "model": model.api_model_identifier,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        },
    )
    return normalize_response(
        data, "anthropic", model.api_model_identifier or "",
        (perf_counter() - started) * 1000, max_tokens, "messages",
    )
