"""Anthropic Messages API adapter."""

from time import perf_counter

from model_registry import ModelConfig
from provider_clients.base import ProviderResponse, post_json, require_int, require_text


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
    content = data.get("content")
    texts = []
    if isinstance(content, list):
        texts = [
            str(block["text"])
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        ]
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return ProviderResponse(
        text=require_text("\n".join(texts), "anthropic", model.api_model_identifier or ""),
        input_tokens=require_int(usage, "input_tokens", "anthropic", model.api_model_identifier or ""),
        output_tokens=require_int(usage, "output_tokens", "anthropic", model.api_model_identifier or ""),
        latency_ms=(perf_counter() - started) * 1000,
        request_id=str(data.get("id") or ""),
        response_model_identifier=str(data.get("model") or ""),
        stop_reason=str(data.get("stop_reason") or ""),
    )
