"""xAI Responses API adapter."""

from time import perf_counter

from model_registry import ModelConfig
from provider_clients.base import ProviderResponse, post_json
from provider_clients.normalization import normalize_response


def call_xai(model: ModelConfig, prompt: str, api_key: str, max_tokens: int) -> ProviderResponse:
    model_id = model.api_model_identifier or ""
    started = perf_counter()
    data = post_json(
        provider="xai",
        model_identifier=model_id,
        url="https://api.x.ai/v1/responses",
        headers={"Authorization": f"Bearer {api_key}"},
        payload={
            "model": model_id,
            "input": prompt,
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": "high"},
            "store": False,
        },
    )
    return normalize_response(
        data, "xai", model.api_model_identifier or "",
        (perf_counter() - started) * 1000, max_tokens, "responses",
    )
