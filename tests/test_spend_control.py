import os
import unittest
from dataclasses import replace
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from unittest.mock import patch

from model_registry import MODEL_REGISTRY, enabled_models
from providers import (
    API_KEY_ENV,
    LIVE_CALLERS,
    ProviderResponse,
    missing_api_keys,
    provider_display_name,
)
from spend_control import (
    SpendCapExceeded,
    SpendPreflightError,
    SpendTracker,
    build_run_plan,
    estimate_max_spend_usd,
    max_call_cost_usd,
)


class SpendControlTests(unittest.TestCase):
    def test_registry_has_five_enabled_models(self):
        models = enabled_models()
        providers = {model.inference_provider for model in models}
        self.assertEqual(len(models), 5)
        self.assertEqual(
            providers,
            {"openai", "anthropic", "xai", "deepseek", "openrouter"},
        )
        self.assertEqual(set(API_KEY_ENV), providers)
        self.assertEqual(set(LIVE_CALLERS), providers)

    def test_qwen_uses_openrouter_without_dashscope(self):
        qwen = next(model for model in MODEL_REGISTRY if model.key == "qwen_3_8_2_4t_a95b")
        self.assertEqual(qwen.creator, "Qwen / Alibaba")
        self.assertEqual(qwen.canonical_model_name, "Qwen3.8 2.4T-A95B")
        self.assertEqual(qwen.inference_provider, "openrouter")
        self.assertEqual(qwen.model_type, "open_weight")
        self.assertEqual(qwen.api_model_identifier, "qwen/qwen3.8-2.4t-a95b")
        self.assertEqual(qwen.input_price_per_million_usd, 2.00)
        self.assertEqual(qwen.output_price_per_million_usd, 6.00)
        self.assertEqual(API_KEY_ENV["openrouter"], "OPENROUTER_API_KEY")
        self.assertEqual(provider_display_name(qwen.inference_provider), "OpenRouter")
        self.assertNotIn("alibaba_cloud", API_KEY_ENV)
        self.assertNotIn("alibaba_cloud", LIVE_CALLERS)
        self.assertNotIn("DASHSCOPE_API_KEY", API_KEY_ENV.values())
        repository = Path(__file__).resolve().parents[1]
        for path in (".env.example", "model_registry.py", "providers.py", "README.md"):
            self.assertNotIn("DASHSCOPE", (repository / path).read_text(encoding="utf-8"))
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            self.assertIn("OPENROUTER_API_KEY", missing_api_keys([qwen]))

    def test_confirmation_contains_required_live_run_details(self):
        model = MODEL_REGISTRY[0]
        plan = build_run_plan([(model, "short prompt")])
        confirmation = plan.confirmation_text()
        for label in (
            "Remaining prompt/model pairs",
            "Maximum provider attempts",
            "Enabled models",
            "Conservative maximum additional spend",
            "Configured spend cap",
        ):
            self.assertIn(label, confirmation)
        self.assertIn("$1.00", confirmation)

    def test_retry_aware_plan_bounds_three_attempts_for_each_pending_pair(self):
        calls = [(model, "pending prompt", 128) for model in enabled_models()
                 if model.inference_provider != "xai"]
        plan = build_run_plan(calls, spend_cap="2.00", max_retries=2)
        expected = sum((max_call_cost_usd(*call) * 3 for call in calls), Decimal(0))
        self.assertEqual(plan.estimated_max_cost_usd, expected.quantize(Decimal("0.000001"), rounding=ROUND_CEILING))
        self.assertEqual(plan.planned_calls, 4)
        self.assertEqual(plan.maximum_provider_attempts, 12)
        self.assertEqual(plan.configured_spend_cap_usd, Decimal("2.00"))

    def test_retry_estimate_rejects_unsupported_retry_counts(self):
        for retries in (-1, 3, 1.5, True):
            with self.subTest(retries=retries), self.assertRaises(ValueError):
                build_run_plan([], max_retries=retries)

    def test_run_cap_cannot_exceed_global_limit(self):
        model = MODEL_REGISTRY[0]
        with patch.dict(os.environ, {"GLOBAL_MAX_SPEND_USD": "0.50"}):
            with self.assertRaises(SpendPreflightError):
                build_run_plan([(model, "prompt")], spend_cap="0.51")

    def test_unresolved_pricing_blocks_maximum_estimate(self):
        unresolved = replace(MODEL_REGISTRY[0], input_price_per_million_usd=None)
        with self.assertRaises(SpendPreflightError):
            estimate_max_spend_usd([(unresolved, "prompt")])

    def test_tracker_stops_before_a_call_that_could_exceed_cap(self):
        model = MODEL_REGISTRY[0]
        plan = build_run_plan([(model, "prompt")], spend_cap="0.01")
        with self.assertRaises(SpendCapExceeded):
            SpendTracker(plan).assert_can_call(model, "prompt")

    def test_tracker_records_actual_cost_after_each_call(self):
        model = MODEL_REGISTRY[0]
        plan = build_run_plan([(model, "prompt")], spend_cap="1.00")
        tracker = SpendTracker(plan)
        tracker.assert_can_call(model, "prompt")
        cost = tracker.record_call(
            model,
            ProviderResponse("answer", input_tokens=10, output_tokens=20, latency_ms=1),
        )
        self.assertGreater(cost, 0)
        self.assertEqual(tracker.completed_calls, 1)
        self.assertEqual(tracker.spent_usd, cost)

    def test_failed_retry_attempt_counts_against_cap(self):
        model = MODEL_REGISTRY[0]
        plan = build_run_plan([(model, "prompt", 10)], spend_cap="0.01")
        tracker = SpendTracker(plan)
        tracker.assert_can_call(model, "prompt", 10)
        reserved = tracker.record_failed_attempt(model, "prompt", 10)
        self.assertGreater(reserved, 0)
        self.assertEqual(tracker.attempted_calls, 1)
        self.assertEqual(tracker.spent_usd, reserved)

    def test_xai_bound_is_unresolved_in_preflight_runtime_and_failure_reservation(self):
        grok = next(m for m in enabled_models() if m.inference_provider == "xai")
        tracker = SpendTracker(build_run_plan([], spend_cap="2.00"))
        for limit in (128, 4096):
            for operation in (
                lambda: max_call_cost_usd(grok, "prompt", limit),
                lambda: build_run_plan([(grok, "prompt", limit)]),
                lambda: tracker.assert_can_call(grok, "prompt", limit),
                lambda: tracker.record_failed_attempt(grok, "prompt", limit),
            ):
                with self.subTest(limit=limit), self.assertRaisesRegex(SpendPreflightError, "xAI pre-dispatch cost bound is unresolved"):
                    operation()
        self.assertEqual(tracker.spent_usd, Decimal(0))
        self.assertEqual(tracker.attempted_calls, 0)

    def test_xai_reported_billing_overrides_token_estimate_after_response(self):
        from provider_clients.normalization import normalize_response
        grok = next(m for m in enabled_models() if m.inference_provider == "xai")
        # Recording an already returned response needs no dispatch authorization.
        tracker = SpendTracker(build_run_plan([], spend_cap="2.00"), initial_spend_usd="0.8135937")
        data = {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "B"}]}],
            "usage": {"input_tokens": 13, "output_tokens": 10015,
                      "output_tokens_details": {"reasoning_tokens": 9500},
                      "cost_in_usd_ticks": 37_756_000},
        }
        response = normalize_response(data, "xai", grok.api_model_identifier, 1.0, 4096, "responses")
        cost = tracker.record_call(grok, response)
        self.assertEqual(cost, Decimal("0.0037756"))
        self.assertNotEqual(cost, Decimal("0.060116"))
        self.assertEqual(tracker.spent_usd, Decimal("0.8173693"))
        self.assertEqual(tracker.completed_calls, 1)

    def test_xai_reported_zero_cost_is_authoritative(self):
        grok = next(m for m in enabled_models() if m.inference_provider == "xai")
        tracker = SpendTracker(build_run_plan([]))
        self.assertEqual(tracker.record_call(grok, ProviderResponse("B", 13, 10015, 1.0,
                                                                  provider_reported_cost_usd=0.0)), Decimal(0))


if __name__ == "__main__":
    unittest.main()
