import os
import unittest
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
)


class SpendControlTests(unittest.TestCase):
    def test_registry_has_six_enabled_models_without_meta_provider(self):
        models = enabled_models()
        self.assertEqual(len(models), 6)
        self.assertNotIn("meta", {model.inference_provider for model in models})
        self.assertNotIn("meta", API_KEY_ENV)
        self.assertNotIn("meta", LIVE_CALLERS)

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
            "Planned calls",
            "Enabled models",
            "Estimated maximum cost",
            "Configured spend cap",
        ):
            self.assertIn(label, confirmation)
        self.assertIn("$1.00", confirmation)

    def test_run_cap_cannot_exceed_global_limit(self):
        model = MODEL_REGISTRY[0]
        with patch.dict(os.environ, {"GLOBAL_MAX_SPEND_USD": "0.50"}):
            with self.assertRaises(SpendPreflightError):
                build_run_plan([(model, "prompt")], spend_cap="0.51")

    def test_unresolved_pricing_blocks_maximum_estimate(self):
        unresolved = next(model for model in MODEL_REGISTRY if model.input_price_per_million_usd is None)
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


if __name__ == "__main__":
    unittest.main()
