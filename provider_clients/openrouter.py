"""OpenRouter Chat Completions API adapter for Qwen."""

from time import perf_counter

from model_registry import ModelConfig
from provider_clients.base import ProviderResponse, post_json
from provider_clients.normalization import normalize_response


def call_openrouter(model: ModelConfig, prompt: str, api_key: str, max_tokens: int) -> ProviderResponse:
    model_id = model.api_model_identifier or ""
    started = perf_counter()
    data = post_json(
        provider="openrouter",
        model_identifier=model_id,
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/local/llm-router",
            "X-Title": "LLM Router Evaluation",
        },
        payload={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "reasoning": {"enabled": True, "exclude": True},
            "stream": False,
        },
    )
    return normalize_response(
        data, "openrouter", model.api_model_identifier or "",
        (perf_counter() - started) * 1000, max_tokens, "chat",
    )
