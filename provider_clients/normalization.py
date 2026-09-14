"""Normalize final answers while retaining only safe response diagnostics.

Never copy response bodies, thinking, reasoning summaries, or refusal text into
diagnostics. Token counts describe usage, not inferred reasoning contents.
"""

from decimal import Decimal

from provider_clients.base import ProviderError, ProviderResponse, require_int


def mapping(value):
    return value if isinstance(value, dict) else {}


def token_count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def normalize_response(data, provider, model_id, latency_ms, max_tokens, api):
    usage = mapping(data.get("usage"))
    chat = api == "chat"
    input_field, output_field = ("prompt_tokens", "completion_tokens") if chat else ("input_tokens", "output_tokens")
    details = mapping(usage.get("completion_tokens_details" if chat else "output_tokens_details"))
    diagnostics = {
        **mapping(data.get("_transport_diagnostics")),
        "response_id": data.get("id") if isinstance(data.get("id"), str) else "",
        "response_model_identifier": data.get("model") if isinstance(data.get("model"), str) else "",
        "input_tokens": token_count(usage.get(input_field)),
        "output_tokens": token_count(usage.get(output_field)),
        "reasoning_tokens": token_count(details.get("reasoning_tokens", details.get("thinking_tokens"))),
        "requested_max_output_tokens": max_tokens,
    }
    cost = None
    if provider == "xai":
        # xAI reports the actual charge after discounts, in integer USD ticks.
        # Keep the original integer for auditability; convert to float only at
        # the ProviderResponse boundary. Never infer this value from tokens.
        raw_ticks = usage.get("cost_in_usd_ticks")
        ticks = token_count(raw_ticks)
        diagnostics.update(
            cost_in_usd_ticks=ticks,
            provider_reported_cost_status=(
                "available" if ticks is not None else
                "missing" if raw_ticks is None else "invalid"
            ),
            response_max_output_tokens=token_count(data.get("max_output_tokens")),
            total_tokens=token_count(usage.get("total_tokens")),
            context_output_tokens=token_count(mapping(usage.get("context_details")).get("output_tokens")),
            num_server_side_tools_used=token_count(usage.get("num_server_side_tools_used")),
        )
        if ticks is not None:
            cost = float(Decimal(ticks) / Decimal(10_000_000_000))
    texts = []
    malformed = False
    refusal = False
    limit = False
    normal = False
    reasoning = False
    stop = ""
    if api in {"responses", "messages"}:
        blocks = data.get("output" if api == "responses" else "content")
        malformed = not isinstance(blocks, list)
        blocks = blocks if isinstance(blocks, list) else []
        types, content_types = [], []
        for block in blocks:
            if not isinstance(block, dict) or not isinstance(block.get("type"), str):
                malformed = True
                continue
            kind = block["type"]
            types.append(kind)
            reasoning |= kind in {"reasoning", "thinking", "redacted_thinking"}
            if api == "responses":
                if kind != "message":
                    continue
                content = block.get("content")
                if not isinstance(content, list):
                    malformed = True
                    continue
            else:
                content = [block]
            for part in content:
                if not isinstance(part, dict) or not isinstance(part.get("type"), str):
                    malformed = True
                    continue
                content_types.append(part["type"])
                refusal |= part["type"] == "refusal"
                if part["type"] in {"text", "output_text"}:
                    if isinstance(part.get("text"), str):
                        texts.append(part["text"])
                    else:
                        malformed = True
        diagnostics.update(output_item_types=types, content_types=content_types)
        if api == "responses":
            stop = data.get("status")
            incomplete = mapping(data.get("incomplete_details")).get("reason")
            diagnostics.update(response_status=stop if isinstance(stop, str) else None,
                               incomplete_reason=incomplete if isinstance(incomplete, str) else None)
            limit = stop == "incomplete" and incomplete == "max_output_tokens"
            refusal |= incomplete == "content_filter"
            normal = stop == "completed"
        else:
            stop = data.get("stop_reason")
            limit = stop == "max_tokens"
            refusal |= stop == "refusal"
            normal = stop in {"end_turn", "stop_sequence"}
    else:
        choices = data.get("choices")
        choice = mapping(choices[0]) if isinstance(choices, list) and choices else {}
        message = mapping(choice.get("message"))
        stop = choice.get("finish_reason")
        content = message.get("content")
        malformed = not choice or not message or ("content" not in message) or (content is not None and not isinstance(content, str))
        if isinstance(content, str):
            texts.append(content)
        reasoning_value = message.get("reasoning_content")
        reasoning = isinstance(reasoning_value, str) and bool(reasoning_value.strip())
        # OpenRouter may report reasoning details instead of reasoning_content.
        reasoning |= bool(message.get("reasoning_details")) or bool(message.get("reasoning"))
        diagnostics.update(reasoning_content_present="reasoning_content" in message,
                           reasoning_content_nonempty=isinstance(reasoning_value, str) and bool(reasoning_value.strip()),
                           final_content_type="null" if content is None else type(content).__name__,
                           finish_reason=stop if isinstance(stop, str) else None)
        limit = stop == "length"
        refusal = bool(message.get("refusal")) or stop == "content_filter"
        normal = stop == "stop"
    stop = stop if isinstance(stop, str) else ""
    diagnostics.update(stop_reason=stop, reasoning_present=reasoning, refusal_present=refusal)
    text = "\n".join(texts)
    diagnostics["visible_text_present"] = bool(text.strip())
    diagnostics["output_tokens_exceed_requested_limit"] = (
        diagnostics["output_tokens"] is not None and diagnostics["output_tokens"] > max_tokens
    )
    try:
        input_tokens = require_int(usage, input_field, provider, model_id)
        output_tokens = require_int(usage, output_field, provider, model_id)
    except ProviderError as error:
        error.diagnostics = diagnostics
        raise

    error_type = None
    if malformed:
        error_type = "invalid_provider_response"
    elif refusal:
        error_type = "refusal"
    elif not text.strip():
        if limit:
            error_type = "output_limit_exhausted"
        elif normal:
            error_type = "reasoning_without_answer" if reasoning else "empty_completion"
        elif api == "responses" and stop == "incomplete":
            error_type = "incomplete_response"
        elif api == "responses" and stop == "failed":
            error_type = "failed_provider_response"
        else:
            error_type = "invalid_provider_response"
    if error_type:
        raise ProviderError(provider, model_id,
                            f"Provider response outcome: {error_type} (stop={stop or 'missing'})",
                            error_type=error_type, diagnostics=diagnostics)
    if provider == "openrouter":
        cost = usage.get("cost")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool):
        cost = None
    return ProviderResponse(
        text=text, input_tokens=input_tokens, output_tokens=output_tokens,
        latency_ms=latency_ms, request_id=diagnostics["response_id"],
        response_model_identifier=diagnostics["response_model_identifier"],
        stop_reason=stop, provider_reported_cost_usd=cost, diagnostics=diagnostics,
    )
