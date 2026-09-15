"""Deterministic provider fixtures and real retry/checkpoint paths; no network."""

import copy
import io
import json
import tempfile
import unittest
import urllib.error
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

from generate_dataset import (
    generate_results,
    planned_calls,
    read_results,
    render_prompt,
    smoke_test_prompt,
    write_results,
)
from model_registry import enabled_models
from provider_clients.base import ProviderError
from providers import LIVE_CALLERS, call_model_with_retries
from spend_control import build_run_plan, max_call_cost_usd

FIXTURES = json.loads((Path(__file__).parent / "fixtures/provider_responses.json").read_text())


def model_for(provider):
    return next(m for m in enabled_models() if m.inference_provider == provider)


class DiagnosticTests(unittest.TestCase):
    def call_fixture(self, provider, fixture):
        with patch(f"provider_clients.{provider}.post_json", return_value=copy.deepcopy(fixture)):
            return LIVE_CALLERS[provider](model_for(provider), "prompt", "dummy", 4096)

    def test_output_budget_fixtures_are_nontransient_pair_outcomes(self):
        for provider, name in (
            ("openai", "openai_incomplete"),
            ("anthropic", "anthropic_limit"),
            ("deepseek", "deepseek_limit"),
        ):
            with self.subTest(provider=provider), self.assertRaises(ProviderError) as caught:
                self.call_fixture(provider, FIXTURES[name])
            error = caught.exception
            self.assertEqual(error.error_type, "output_limit_exhausted")
            self.assertFalse(error.retryable)
            self.assertEqual(error.failure_scope, "pair")
            self.assertTrue(error.may_have_been_billed)
            self.assertEqual(error.diagnostics["output_tokens"], 4096)
            self.assertTrue(error.diagnostics["reasoning_present"])
            self.assertNotIn("SYNTHETIC_PRIVATE", str(vars(error)))

    def test_completed_fixtures_keep_only_visible_answer(self):
        for provider in ("openai", "anthropic", "deepseek"):
            with self.subTest(provider=provider):
                response = self.call_fixture(provider, FIXTURES[provider + "_completed"])
                self.assertEqual(response.text, "B")
                self.assertEqual(response.output_tokens, 80)
                self.assertNotIn("SYNTHETIC_PRIVATE", str(response))

    def test_genuinely_malformed_empty_and_wrong_content_types(self):
        for provider in LIVE_CALLERS:
            with self.subTest(provider=provider), self.assertRaises(ProviderError) as caught:
                self.call_fixture(provider, FIXTURES["malformed_empty"])
            self.assertEqual(caught.exception.error_type, "invalid_provider_response")
        fixture = copy.deepcopy(FIXTURES["deepseek_completed"])
        fixture["choices"][0]["message"]["content"] = {"unexpected": "object"}
        with self.assertRaises(ProviderError) as caught:
            self.call_fixture("deepseek", fixture)
        self.assertEqual(caught.exception.error_type, "invalid_provider_response")

    def test_normal_empty_completion_is_distinct_from_malformed(self):
        for provider in ("openai", "anthropic", "deepseek"):
            fixture = copy.deepcopy(FIXTURES[provider + "_completed"])
            if provider == "openai":
                fixture["output"] = []
            elif provider == "anthropic":
                fixture["content"] = []
            else:
                fixture["choices"][0]["message"] = {"content": None}
            with self.subTest(provider=provider), self.assertRaises(ProviderError) as caught:
                self.call_fixture(provider, fixture)
            self.assertEqual(caught.exception.error_type, "empty_completion")

    def test_reasoning_without_length_finish_is_not_assumed_exhausted(self):
        fixture = copy.deepcopy(FIXTURES["deepseek_limit"])
        fixture["choices"][0]["finish_reason"] = "stop"
        with self.assertRaises(ProviderError) as caught:
            self.call_fixture("deepseek", fixture)
        self.assertEqual(caught.exception.error_type, "reasoning_without_answer")

    def test_explicit_refusals_are_never_graded(self):
        for provider in ("openai", "anthropic", "deepseek"):
            fixture = copy.deepcopy(FIXTURES[provider + "_completed"])
            if provider == "openai":
                fixture["output"] = [
                    {
                        "type": "message",
                        "content": [{"type": "refusal", "refusal": "SYNTHETIC_PRIVATE"}],
                    }
                ]
            elif provider == "anthropic":
                fixture["stop_reason"] = "refusal"
            else:
                fixture["choices"][0]["finish_reason"] = "content_filter"
            with self.subTest(provider=provider), self.assertRaises(ProviderError) as caught:
                self.call_fixture(provider, fixture)
            self.assertEqual(caught.exception.error_type, "refusal")
            self.assertNotIn("SYNTHETIC_PRIVATE", str(vars(caught.exception)))

    def test_unknown_incomplete_reason_is_retained(self):
        fixture = copy.deepcopy(FIXTURES["openai_incomplete"])
        fixture["incomplete_details"]["reason"] = "other"
        with self.assertRaises(ProviderError) as caught:
            self.call_fixture("openai", fixture)
        self.assertEqual(caught.exception.error_type, "incomplete_response")
        self.assertEqual(caught.exception.diagnostics["incomplete_reason"], "other")

    def test_xai_read_timeout_diagnostics_and_configuration(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.headers = {"x-request-id": "req_timeout"}
        response.read.side_effect = TimeoutError("The read operation timed out")
        with patch("provider_clients.base.urllib.request.urlopen", return_value=response) as http:
            with self.assertRaises(ProviderError) as caught:
                LIVE_CALLERS["xai"](model_for("xai"), "prompt", "dummy", 4096)
        error = caught.exception
        self.assertEqual(error.error_type, "network_error")
        self.assertTrue(error.retryable)
        self.assertEqual(error.diagnostics["request_id"], "req_timeout")
        self.assertEqual(error.diagnostics["transport_phase"], "read")
        self.assertTrue(error.diagnostics["timeout"])
        self.assertEqual(http.call_args.kwargs["timeout"], 600)
        payload = json.loads(http.call_args.args[0].data)
        self.assertEqual(payload["reasoning"], {"effort": "high"})
        self.assertEqual(payload["max_output_tokens"], 4096)

    def test_openrouter_affordability_rejection_dispatches_only_once(self):
        rejection = urllib.error.HTTPError(
            "https://example.invalid",
            402,
            "Payment required",
            {},
            io.BytesIO(
                json.dumps(
                    {
                        "error": {
                            "message": "This request requires more credits, or fewer max_tokens"
                        }
                    }
                ).encode()
            ),
        )
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "dummy"}):
            with patch(
                "provider_clients.base.urllib.request.urlopen", side_effect=rejection
            ) as http:
                with self.assertRaises(ProviderError) as caught:
                    call_model_with_retries(
                        model_for("openrouter"), "prompt", dry_run=False, max_retries=2
                    )
        self.assertEqual(http.call_count, 1)
        self.assertEqual(caught.exception.error_type, "quota_or_billing_error")
        self.assertFalse(caught.exception.retryable)
        self.assertEqual(caught.exception.failure_scope, "model")

    def test_interrupt_is_checkpointed_reserved_and_resumable(self):
        # Grok is now blocked before dispatch because its reservation is unknown.
        # Exercise the shared interruption path with a provider that has a bound.
        model = model_for("openai")
        record = smoke_test_prompt()
        bound = max_call_cost_usd(model, render_prompt(record), 128)
        plan = build_run_plan(
            planned_calls([record], [model], max_output_tokens=128),
            spend_cap=bound * Decimal("1.5"),
        )
        response = MagicMock()
        response.__enter__.return_value = response
        response.status, response.headers = 200, {"x-request-id": "req_interrupted"}
        response.read.side_effect = KeyboardInterrupt
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.csv"
            with patch.dict("os.environ", {"OPENAI_API_KEY": "dummy"}):
                with patch(
                    "provider_clients.base.urllib.request.urlopen", return_value=response
                ) as http:
                    with self.assertRaises(KeyboardInterrupt):
                        generate_results(
                            [record],
                            dry_run=False,
                            models=[model],
                            run_plan=plan,
                            max_output_tokens=128,
                            on_result=lambda rows: write_results(rows, path),
                        )
            self.assertEqual(http.call_count, 1)
            saved = read_results(path)
            row = saved[0]
            self.assertEqual(row["error_type"], "interrupted_inflight")
            self.assertEqual(row["provider_attempts"], 1)
            self.assertEqual(Decimal(str(row["estimated_cost_usd"])), bound)
            self.assertIsNone(row["correct"])
            self.assertIsNone(row["score"])
            self.assertTrue(row["failure_retryable"])
            self.assertEqual(row["provider_request_id"], "req_interrupted")
            with patch("providers.call_model") as calls:
                blocked = generate_results(
                    [record],
                    dry_run=False,
                    models=[model],
                    run_plan=plan,
                    max_output_tokens=128,
                    completed_results=saved,
                )
            calls.assert_not_called()
            self.assertEqual(blocked[0]["provider_attempts"], 1)
            self.assertEqual(blocked[0]["estimated_cost_usd"], row["estimated_cost_usd"])
            larger = build_run_plan(
                planned_calls([record], [model], max_output_tokens=128), spend_cap=bound * 3
            )
            ok = self.call_fixture("openai", FIXTURES["openai_completed"])
            with patch("providers.call_model", return_value=ok) as calls:
                resumed = generate_results(
                    [record],
                    dry_run=False,
                    models=[model],
                    run_plan=larger,
                    max_output_tokens=128,
                    completed_results=saved,
                )
            self.assertEqual(calls.call_count, 1)
            self.assertEqual(resumed[0]["provider_attempts"], 2)
            self.assertEqual(resumed[0]["status"], "success")
            self.assertGreater(resumed[0]["estimated_cost_usd"], row["estimated_cost_usd"])
            self.assertEqual(len(json.loads(resumed[0]["provider_diagnostics"])), 2)

    def test_interrupt_during_backoff_does_not_charge_a_phantom_dispatch(self):
        error = ProviderError(
            "xai", "grok-4.6", "timeout", error_type="network_error", retryable=True
        )
        with patch("providers.call_model", side_effect=error) as calls:
            with patch("providers.time.sleep", side_effect=KeyboardInterrupt) as sleep:
                failures = []
                with self.assertRaises(KeyboardInterrupt):
                    call_model_with_retries(
                        model_for("xai"),
                        "prompt",
                        dry_run=False,
                        on_failed_attempt=failures.append,
                        sleep=sleep,
                    )
        self.assertEqual(calls.call_count, 1)
        self.assertEqual(failures, [error])

    def test_budget_failure_checkpoints_metadata_skips_same_config_and_allows_other_pair(self):
        model = model_for("deepseek")
        records = [smoke_test_prompt(), replace(smoke_test_prompt(), prompt_id="second")]
        plan = build_run_plan(planned_calls(records, [model]), spend_cap="1")
        # Explicit side effects make the second identical-text prompt successful.
        with self.assertRaises(ProviderError) as caught:
            self.call_fixture("deepseek", FIXTURES["deepseek_limit"])
        ok = self.call_fixture("deepseek", FIXTURES["deepseek_completed"])
        with patch("providers.call_model", side_effect=[caught.exception, ok]) as calls:
            rows = generate_results(records, dry_run=False, models=[model], run_plan=plan)
        self.assertEqual(calls.call_count, 2)
        self.assertEqual([r["status"] for r in rows], ["provider_error", "success"])
        self.assertEqual(rows[0]["output_tokens"], 4096)
        self.assertGreater(rows[0]["estimated_cost_usd"], 0)
        self.assertNotIn("SYNTHETIC_PRIVATE", json.dumps(rows))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            write_results(rows, path)
            saved = read_results(path)
        with patch("providers.call_model") as calls:
            resumed = generate_results(
                records, dry_run=False, models=[model], completed_results=saved
            )
        calls.assert_not_called()
        self.assertEqual(resumed, saved)
        with patch("providers.call_model", return_value=ok) as calls:
            changed = generate_results(
                records,
                dry_run=False,
                models=[model],
                max_output_tokens=2048,
                completed_results=saved,
            )
        self.assertEqual(calls.call_count, 1)
        self.assertEqual(changed[1], saved[1])

    def test_interrupt_after_failed_attempt_keeps_both_reservations(self):
        model, record = model_for("openai"), smoke_test_prompt()
        bound = max_call_cost_usd(model, render_prompt(record))
        plan = build_run_plan(planned_calls([record], [model]), spend_cap=bound * 3)
        timeout = ProviderError(
            "openai",
            model.api_model_identifier,
            "timeout",
            error_type="network_error",
            retryable=True,
        )
        snapshots = []
        with patch("providers.call_model", side_effect=[timeout, KeyboardInterrupt]) as calls:
            from functools import partial

            with patch(
                "generate_dataset.call_model_with_retries",
                partial(call_model_with_retries, sleep=lambda _: None),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    generate_results(
                        [record],
                        dry_run=False,
                        models=[model],
                        run_plan=plan,
                        on_result=lambda rows: snapshots.append(copy.deepcopy(rows)),
                    )
        self.assertEqual(calls.call_count, 2)
        row = snapshots[-1][0]
        self.assertEqual(row["provider_attempts"], 2)
        self.assertEqual(row["retry_count"], 1)
        self.assertEqual(Decimal(str(row["estimated_cost_usd"])), bound * 2)
        self.assertEqual(
            [d["outcome"] for d in json.loads(row["provider_diagnostics"])],
            ["network_error", "interrupted_inflight"],
        )

    def test_interruption_before_attempt_does_not_reserve_or_dispatch(self):
        with patch("providers.call_model") as calls:
            with patch("builtins.print", side_effect=KeyboardInterrupt):
                failures = []
                with self.assertRaises(KeyboardInterrupt):
                    call_model_with_retries(
                        model_for("xai"),
                        "prompt",
                        dry_run=False,
                        before_attempt=lambda: print("before dispatch"),
                        on_failed_attempt=failures.append,
                    )
        calls.assert_not_called()
        self.assertEqual(failures, [])

    def test_invalid_json_does_not_persist_raw_body(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status, response.headers = 200, {"request-id": "req_bad_json"}
        response.read.return_value = b'{"thinking": "SYNTHETIC_PRIVATE_REASONING"'
        with patch("provider_clients.base.urllib.request.urlopen", return_value=response):
            with self.assertRaises(ProviderError) as caught:
                LIVE_CALLERS["anthropic"](model_for("anthropic"), "prompt", "dummy", 4096)
        self.assertEqual(caught.exception.error_type, "invalid_provider_response")
        self.assertNotIn("SYNTHETIC_PRIVATE", str(vars(caught.exception)))

    def test_xai_usage_above_requested_limit_is_diagnosable(self):
        fixture = copy.deepcopy(FIXTURES["openai_completed"])
        fixture["usage"]["output_tokens"] = 10015
        response = self.call_fixture("xai", fixture)
        self.assertEqual(response.text, "B")
        self.assertEqual(response.stop_reason, "completed")
        self.assertEqual(response.output_tokens, 10015)
        self.assertTrue(response.diagnostics["output_tokens_exceed_requested_limit"])

    def test_xai_cost_ticks_convert_exactly_before_float_boundary(self):
        for ticks, expected in (
            (1, Decimal("0.0000000001")),
            (37_756_000, Decimal("0.0037756")),
            (10_000_000_000, Decimal("1")),
            (0, Decimal("0")),
        ):
            fixture = copy.deepcopy(FIXTURES["openai_completed"])
            fixture["usage"]["cost_in_usd_ticks"] = ticks
            with self.subTest(ticks=ticks):
                response = self.call_fixture("xai", fixture)
                self.assertEqual(Decimal(str(response.provider_reported_cost_usd)), expected)
                self.assertEqual(response.diagnostics["cost_in_usd_ticks"], ticks)

    def test_xai_cost_diagnostics_keep_ticks_and_telemetry_without_reasoning_content(self):
        fixture = copy.deepcopy(FIXTURES["openai_completed"])
        fixture.update(model="grok-4.6", max_output_tokens=4096)
        fixture["output"][0].update(
            encrypted_content="SYNTHETIC_PRIVATE_ENCRYPTED_REASONING",
            summary=[{"type": "summary_text", "text": "SYNTHETIC_PRIVATE_REASONING"}],
        )
        fixture["usage"].update(
            cost_in_usd_ticks=37_756_000,
            total_tokens=93,
            context_details={"output_tokens": 80},
            num_server_side_tools_used=0,
        )
        response = self.call_fixture("xai", fixture)
        self.assertEqual(response.diagnostics["provider_reported_cost_status"], "available")
        for name, expected in (
            ("cost_in_usd_ticks", 37_756_000),
            ("reasoning_tokens", 79),
            ("response_max_output_tokens", 4096),
            ("context_output_tokens", 80),
            ("total_tokens", 93),
        ):
            self.assertEqual(response.diagnostics[name], expected)
        self.assertNotIn("SYNTHETIC_PRIVATE", str(response))
        self.assertNotIn("encrypted_content", json.dumps(response.diagnostics))

    def test_xai_missing_or_invalid_ticks_do_not_fabricate_reported_cost(self):
        for ticks in (None, True, -1, 1.5, "37756000", {"reasoning": "SYNTHETIC_PRIVATE"}):
            fixture = copy.deepcopy(FIXTURES["openai_completed"])
            fixture["usage"]["cost_in_usd_ticks"] = ticks
            with self.subTest(ticks=ticks):
                response = self.call_fixture("xai", fixture)
                self.assertIsNone(response.provider_reported_cost_usd)
                self.assertIsNone(response.diagnostics["cost_in_usd_ticks"])
                self.assertEqual(
                    response.diagnostics["provider_reported_cost_status"],
                    "missing" if ticks is None else "invalid",
                )
                self.assertNotIn("SYNTHETIC_PRIVATE", str(response))


if __name__ == "__main__":
    unittest.main()
