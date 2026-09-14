"""xAI Responses API adapter."""

from time import perf_counter

from model_registry import ModelConfig
from provider_clients.base import ProviderResponse, post_json, require_int, response_text


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
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return ProviderResponse(
        text=response_text(data, "xai", model_id),
        input_tokens=require_int(usage, "input_tokens", "xai", model_id),
        output_tokens=require_int(usage, "output_tokens", "xai", model_id),
        latency_ms=(perf_counter() - started) * 1000,
        request_id=str(data.get("id") or ""),
        response_model_identifier=str(data.get("model") or ""),
        stop_reason=str(data.get("status") or ""),
    )
