"""DeepSeek Chat Completions API adapter."""

from time import perf_counter

from model_registry import ModelConfig
from provider_clients.base import ProviderResponse, post_json
from provider_clients.normalization import normalize_response


def call_deepseek(model: ModelConfig, prompt: str, api_key: str, max_tokens: int) -> ProviderResponse:
    model_id = model.api_model_identifier or ""
    started = perf_counter()
    data = post_json(
        provider="deepseek",
        model_identifier=model_id,
        url="https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        payload={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
            "stream": False,
        },
    )
    return normalize_response(
        data, "deepseek", model.api_model_identifier or "",
        (perf_counter() - started) * 1000, max_tokens, "chat",
    )
