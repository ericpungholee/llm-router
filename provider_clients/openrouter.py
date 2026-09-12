"""OpenRouter Chat Completions API adapter for Qwen."""

from time import perf_counter

from model_registry import ModelConfig
from provider_clients.base import ProviderResponse, post_json, require_int, require_text


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
    choices = data.get("choices")
    message = {}
    stop_reason = ""
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        stop_reason = str(choices[0].get("finish_reason") or "")
        if isinstance(choices[0].get("message"), dict):
            message = choices[0]["message"]
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    reported_cost = usage.get("cost")
    if not isinstance(reported_cost, (int, float)) or isinstance(reported_cost, bool):
        reported_cost = None
    return ProviderResponse(
        text=require_text(message.get("content"), "openrouter", model_id),
        input_tokens=require_int(usage, "prompt_tokens", "openrouter", model_id),
        output_tokens=require_int(usage, "completion_tokens", "openrouter", model_id),
        latency_ms=(perf_counter() - started) * 1000,
        request_id=str(data.get("id") or ""),
        response_model_identifier=str(data.get("model") or ""),
        stop_reason=stop_reason,
        provider_reported_cost_usd=float(reported_cost) if reported_cost is not None else None,
    )
