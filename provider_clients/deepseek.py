"""DeepSeek Chat Completions API adapter."""

from time import perf_counter

from model_registry import ModelConfig
from provider_clients.base import ProviderResponse, post_json, require_int, require_text


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
    choices = data.get("choices")
    message = {}
    stop_reason = ""
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        stop_reason = str(choices[0].get("finish_reason") or "")
        if isinstance(choices[0].get("message"), dict):
            message = choices[0]["message"]
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return ProviderResponse(
        text=require_text(message.get("content"), "deepseek", model_id),
        input_tokens=require_int(usage, "prompt_tokens", "deepseek", model_id),
        output_tokens=require_int(usage, "completion_tokens", "deepseek", model_id),
        latency_ms=(perf_counter() - started) * 1000,
        request_id=str(data.get("id") or ""),
        response_model_identifier=str(data.get("model") or ""),
        stop_reason=stop_reason,
    )
