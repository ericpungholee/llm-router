import unittest
import io
import json
import urllib.error
from unittest.mock import patch

from model_registry import enabled_models
from provider_clients.anthropic import call_anthropic
from provider_clients.deepseek import call_deepseek
from provider_clients.openai import call_openai
from provider_clients.openrouter import call_openrouter
from provider_clients.xai import call_xai
from provider_clients.base import ProviderError, _classify_http_error, _quota_scope, post_json


def model_for(provider):
    return next(model for model in enabled_models() if model.inference_provider == provider)


class ProviderAdapterTests(unittest.TestCase):
    def test_http_billing_normalization_preserves_scope(self):
        for message, scope in (("Insufficient credits", "provider"), ("Requires fewer max_tokens", "model")):
            rejection = urllib.error.HTTPError("https://example.invalid", 402, "Payment required", {},
                                              io.BytesIO(json.dumps({"error": {"message": message}}).encode()))
            with patch("provider_clients.base.urllib.request.urlopen", side_effect=rejection):
                with self.assertRaises(ProviderError) as raised:
                    post_json(provider="example", model_identifier="model", url="https://example.invalid", headers={}, payload={})
            self.assertEqual(raised.exception.failure_scope, scope)
            self.assertFalse(raised.exception.retryable)

    def test_billing_exhaustion_and_zero_resource_quota_have_different_scopes(self):
        cases = [
            (402, "", "Insufficient credits", "provider"),
            (402, "", "This request requires more credits, or fewer max_tokens; can only afford 100 tokens", "model"),
            (429, "insufficient_quota", "You exceeded your current quota", "provider"),
            (429, "RESOURCE_EXHAUSTED", "Quota exceeded, limit: 0 for model X", "model"),
        ]
        for status, code, message, scope in cases:
            with self.subTest(status=status, code=code):
                kind, retryable, invalid = _classify_http_error(status, code, "", message)
                self.assertEqual(kind, "quota_or_billing_error")
                self.assertFalse(retryable)
                error = ProviderError("provider", "model", message, status_code=status,
                                      error_type=kind, retryable=retryable, invalid_model=invalid,
                                      failure_scope=_quota_scope(status, code, "", message))
                self.assertEqual(error.failure_scope, scope)

    def test_ordinary_rate_limit_and_server_errors_are_retryable_pair_failures(self):
        for status in (408, 429, 500, 502, 503):
            kind, retryable, invalid = _classify_http_error(status, "", "", "Try again later")
            error = ProviderError("provider", "model", "temporary", error_type=kind,
                                  retryable=retryable, invalid_model=invalid)
            self.assertTrue(error.retryable)
            self.assertEqual(error.failure_scope, "pair")

    def test_zero_quota_is_not_classified_as_transient(self):
        error_type, retryable, invalid_model = _classify_http_error(
            429, "RESOURCE_EXHAUSTED", "", "Quota exceeded, limit: 0"
        )
        self.assertEqual(error_type, "quota_or_billing_error")
        self.assertFalse(retryable)
        self.assertFalse(invalid_model)

    def test_openai_responses_adapter(self):
        data = {
            "id": "resp_1",
            "model": "gpt-5.6-sol",
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "B"}]}],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
        with patch("provider_clients.openai.post_json", return_value=data) as post:
            response = call_openai(model_for("openai"), "prompt", "key", 32)
        self.assertEqual((response.text, response.input_tokens, response.output_tokens), ("B", 10, 2))
        self.assertEqual(post.call_args.kwargs["payload"]["model"], "gpt-5.6-sol")
        self.assertEqual(post.call_args.kwargs["payload"]["max_output_tokens"], 32)

    def test_anthropic_messages_adapter(self):
        data = {
            "id": "msg_1",
            "model": "claude-opus-5",
            "stop_reason": "end_turn",
            "content": [{"type": "thinking", "thinking": "hidden"}, {"type": "text", "text": "B"}],
            "usage": {"input_tokens": 11, "output_tokens": 3},
        }
        with patch("provider_clients.anthropic.post_json", return_value=data) as post:
            response = call_anthropic(model_for("anthropic"), "prompt", "key", 32)
        self.assertEqual((response.text, response.input_tokens, response.output_tokens), ("B", 11, 3))
        payload = post.call_args.kwargs["payload"]
        self.assertEqual(payload["model"], "claude-opus-5")
        self.assertEqual(payload["thinking"], {"type": "adaptive"})

    def test_xai_responses_adapter(self):
        data = {
            "id": "resp_2",
            "model": "grok-4.6",
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "B"}]}],
            "usage": {"input_tokens": 13, "output_tokens": 5},
        }
        with patch("provider_clients.xai.post_json", return_value=data) as post:
            response = call_xai(model_for("xai"), "prompt", "key", 32)
        self.assertEqual((response.text, response.input_tokens, response.output_tokens), ("B", 13, 5))
        self.assertEqual(post.call_args.kwargs["payload"]["reasoning"], {"effort": "high"})

    def test_deepseek_chat_adapter(self):
        data = {
            "id": "chat_1",
            "model": "deepseek-flash",
            "choices": [{"finish_reason": "stop", "message": {"content": "B"}}],
            "usage": {"prompt_tokens": 14, "completion_tokens": 6},
        }
        with patch("provider_clients.deepseek.post_json", return_value=data) as post:
            response = call_deepseek(model_for("deepseek"), "prompt", "key", 32)
        self.assertEqual((response.text, response.input_tokens, response.output_tokens), ("B", 14, 6))
        self.assertEqual(post.call_args.kwargs["payload"]["model"], "deepseek-flash")
        self.assertEqual(post.call_args.kwargs["payload"]["thinking"], {"type": "enabled"})

    def test_openrouter_chat_adapter_keeps_reported_cost(self):
        data = {
            "id": "chat_2",
            "model": "qwen/qwen3.8-2.4t-a95b",
            "choices": [{"finish_reason": "stop", "message": {"content": "B"}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 7, "cost": 0.000072},
        }
        with patch("provider_clients.openrouter.post_json", return_value=data) as post:
            response = call_openrouter(model_for("openrouter"), "prompt", "key", 32)
        self.assertEqual((response.text, response.input_tokens, response.output_tokens), ("B", 15, 7))
        self.assertEqual(response.provider_reported_cost_usd, 0.000072)
        self.assertEqual(post.call_args.kwargs["payload"]["model"], "qwen/qwen3.8-2.4t-a95b")


if __name__ == "__main__":
    unittest.main()
