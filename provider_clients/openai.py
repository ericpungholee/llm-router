"""OpenAI Responses API adapter."""

from time import perf_counter

from model_registry import ModelConfig
from provider_clients.base import ProviderResponse, post_json
from provider_clients.normalization import normalize_response


def call_openai(model: ModelConfig, prompt: str, api_key: str, max_tokens: int) -> ProviderResponse:
    started = perf_counter()
    data = post_json(
        provider="openai",
        model_identifier=model.api_model_identifier or "",
        url="https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}"},
        payload={
            "model": model.api_model_identifier,
            "input": prompt,
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": "medium"},
            "store": False,
        },
    )
    return normalize_response(
        data, "openai", model.api_model_identifier or "",
        (perf_counter() - started) * 1000, max_tokens, "responses",
    )
