"""OpenAI Responses API adapter."""

from time import perf_counter

from model_registry import ModelConfig
from provider_clients.base import ProviderResponse, post_json, require_int, response_text


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
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return ProviderResponse(
        text=response_text(data, "openai", model.api_model_identifier or ""),
        input_tokens=require_int(usage, "input_tokens", "openai", model.api_model_identifier or ""),
        output_tokens=require_int(usage, "output_tokens", "openai", model.api_model_identifier or ""),
        latency_ms=(perf_counter() - started) * 1000,
        request_id=str(data.get("id") or ""),
        response_model_identifier=str(data.get("model") or ""),
        stop_reason=str(data.get("status") or ""),
    )
