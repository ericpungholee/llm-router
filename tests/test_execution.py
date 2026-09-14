import unittest
import tempfile
from dataclasses import replace
from functools import partial
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

from benchmark_loaders import BenchmarkRecord, load_enabled_benchmarks, load_hard_pilot
from generate_dataset import (
    generate_results,
    is_completed_result,
    main,
    parse_args,
    planned_calls,
    regrade_results,
    select_pilot_prompts,
    smoke_test_prompt,
    read_results,
    write_results,
)
from graders import MalformedModelOutput, parse_math_answer, parse_multiple_choice
from model_registry import enabled_models
from provider_clients.base import ProviderError, ProviderResponse, _classify_http_error
from providers import call_model_with_retries, validate_provider_registry
from spend_control import build_run_plan, max_call_cost_usd


class ModeSelectionTests(unittest.TestCase):
    def test_hard_pilot_alias_requires_confirmation_before_calls(self):
        with patch("sys.argv", ["generate_dataset.py", "--hard-pilot", "--max-spend-usd", "2.00"]):
            self.assertTrue(parse_args().pilot)
            with patch("generate_dataset.call_model_with_retries") as caller:
                with patch("builtins.print"):
                    with self.assertRaisesRegex(SystemExit, "pass --confirm"):
                        main()
            caller.assert_not_called()

    def test_hard_pilot_has_five_prompts_per_benchmark(self):
        _, records = load_enabled_benchmarks(Path("benchmarks/manifest.json"))
        selected = load_hard_pilot(Path("benchmarks/hard_pilot.json"), records)
        self.assertEqual(len(selected), 15)
        for name in ("mmlu_pro", "math_500", "livecodebench"):
            self.assertEqual(sum(row.benchmark_name == name for row in selected), 5)
        self.assertTrue(all(row.difficulty for row in selected))
        self.assertTrue(
            all(row.problem_date for row in selected if row.benchmark_name == "livecodebench")
        )

    def test_smoke_selects_exactly_one_prompt_times_five_models(self):
        records = [smoke_test_prompt()]
        calls = planned_calls(records, enabled_models(), max_output_tokens=128)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(calls), 5)
        self.assertEqual({call[2] for call in calls}, {128})

    def test_pilot_selects_exactly_fifteen_prompts_times_five_models(self):
        _, records = load_enabled_benchmarks(Path("benchmarks/manifest.json"))
        selected = select_pilot_prompts(records)
        self.assertEqual(len(selected), 15)
        self.assertEqual(len(planned_calls(selected, enabled_models())), 75)
        self.assertEqual(len({(r.prompt_id, m.inference_provider, m.api_model_identifier) for r in selected for m in enabled_models()}), 75)
        for name in ("mmlu_pro", "math_500", "livecodebench"):
            self.assertEqual(sum(row.benchmark_name == name for row in selected), 5)

    def test_registry_identifiers_are_implemented_exactly(self):
        validate_provider_registry(enabled_models())


class ParsingTests(unittest.TestCase):
    def test_normalizes_multiple_choice_and_math_final_answer(self):
        self.assertEqual(parse_multiple_choice("Final answer: (b).", ["A", "B"]), "B")
        self.assertEqual(parse_math_answer("Work\nTherefore the final answer is: \\frac{3}{2}"), "\\frac{3}{2}")

    def test_ambiguous_or_unmarked_output_is_a_parsing_failure(self):
        with self.assertRaises(MalformedModelOutput):
            parse_multiple_choice("It might be A or B", ["A", "B"])
        with self.assertRaises(MalformedModelOutput):
            parse_math_answer("I could not solve this problem because it is hard.")


class ExecutionSafetyTests(unittest.TestCase):
    def setUp(self):
        self.model = enabled_models()[0]
        self.records = [smoke_test_prompt(), replace(smoke_test_prompt(), prompt_id="smoke_test:second")]
        self.response = ProviderResponse("B", 5, 1, 12.0)

    def run_mocked(self, effects, records=None, **kwargs):
        # Exercise the real retry loop and attempt callbacks, with no HTTP/sleep.
        with patch("providers.call_model", side_effect=effects) as calls:
            with patch("generate_dataset.call_model_with_retries", partial(call_model_with_retries, sleep=lambda _: None)):
                rows = generate_results(records or self.records, dry_run=False, models=kwargs.pop("models", [self.model]), **kwargs)
        return rows, calls.call_count

    def error(self, error_type="network_error", retryable=True, **kwargs):
        return ProviderError(self.model.inference_provider, self.model.api_model_identifier,
                             "test failure", error_type=error_type, retryable=retryable, **kwargs)

    def test_timeout_exhaustion_continues_same_model_on_prompt_two(self):
        rows, calls = self.run_mocked([self.error(), self.error(), self.error(), self.response], max_retries=2)
        self.assertEqual(calls, 4)
        self.assertEqual([r["status"] for r in rows], ["provider_error", "success"])
        self.assertEqual(rows[0]["retry_count"], 2)
        self.assertEqual(rows[0]["failure_scope"], "pair")
        self.assertEqual([r["provider_attempts"] for r in rows], [3, 1])

    def test_http_500_and_429_exhaustion_do_not_disable_model(self):
        for status in (500, 429):
            with self.subTest(status=status):
                kind, retryable, invalid = _classify_http_error(status, "", "", "temporarily busy")
                error = self.error(kind, retryable, status_code=status, invalid_model=invalid)
                rows, calls = self.run_mocked([error, error, self.response], max_retries=1)
                self.assertEqual(calls, 3)
                self.assertEqual([r["status"] for r in rows], ["provider_error", "success"])
                self.assertEqual(rows[0]["retry_count"], 1)

    def test_429_retry_can_succeed_before_next_prompt(self):
        rows, calls = self.run_mocked([self.error("transient_provider_error", status_code=429), self.response, self.response], max_retries=2)
        self.assertEqual(calls, 3)
        self.assertEqual([r["status"] for r in rows], ["success", "success"])
        self.assertEqual(rows[0]["retry_count"], 1)
        self.assertEqual(rows[0]["failure_scope"], "")
        self.assertIsNone(rows[0]["http_status"])

    def test_invalid_model_blocks_only_exact_provider_model(self):
        other = replace(self.model, key="other", api_model_identifier="other-id")
        rows, calls = self.run_mocked([
            self.error("invalid_model_error", False, invalid_model=True), self.response, self.response,
        ], models=[self.model, other])
        self.assertEqual(calls, 3)
        self.assertEqual([r["status"] for r in rows], ["provider_error", "success", "skipped_model", "success"])
        self.assertEqual(rows[2]["failure_scope"], "model")
        self.assertEqual(rows[2]["provider_attempts"], 0)

    def test_authentication_blocks_provider_this_run_and_rechecks_on_resume(self):
        same_provider = replace(self.model, key="other", api_model_identifier="other-id")
        independent = enabled_models()[1]
        rows, calls = self.run_mocked([
            self.error("authentication_error", False, status_code=401), self.response, self.response,
        ], models=[self.model, same_provider, independent])
        self.assertEqual(calls, 3)
        self.assertEqual([r["status"] for r in rows], ["provider_error", "skipped_model", "success", "skipped_model", "skipped_model", "success"])
        resumed, calls = self.run_mocked([self.response] * 4, models=[self.model, same_provider, independent], completed_results=rows)
        self.assertEqual(calls, 4)
        self.assertTrue(all(r["status"] == "success" for r in resumed))

    def test_configuration_block_persists_until_settings_change(self):
        rows, calls = self.run_mocked([self.error("configuration_error", False)])
        self.assertEqual(calls, 1)
        _, calls = self.run_mocked([], completed_results=rows)
        self.assertEqual(calls, 0)
        changed = replace(self.model, generation=replace(self.model.generation, reasoning_setting="changed"))
        resumed, calls = self.run_mocked([self.response, self.response], models=[changed], completed_results=rows)
        self.assertEqual(calls, 2)
        self.assertTrue(all(r["status"] == "success" for r in resumed))

    def test_invalid_model_block_releases_when_identifier_changes(self):
        rows, _ = self.run_mocked([self.error("invalid_model_error", False, invalid_model=True)])
        changed = replace(self.model, api_model_identifier="fixed-model-id")
        _, calls = self.run_mocked([self.response, self.response], models=[changed], completed_results=rows)
        self.assertEqual(calls, 2)

    def test_prompt_specific_parameters_and_empty_outputs_stay_local(self):
        for kind in ("invalid_parameter_error", "invalid_provider_response"):
            with self.subTest(kind=kind):
                rows, calls = self.run_mocked([self.error(kind, False), self.response])
                self.assertEqual(calls, 2)
                self.assertEqual([r["status"] for r in rows], ["provider_error", "success"])

    def test_resume_retries_transient_failure_and_preserves_cost_and_attempts(self):
        rows, _ = self.run_mocked([self.error(), self.response], max_retries=0)
        rows[0]["estimated_cost_usd"] = 0.03
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            write_results(rows, path)
            saved = read_results(path)
            resumed, calls = self.run_mocked([self.response], completed_results=saved)
            write_results(resumed, path)
            resumed = read_results(path)
        self.assertEqual(calls, 1)
        self.assertEqual(len(resumed), 2)
        self.assertEqual(resumed[0]["provider_attempts"], 2)
        self.assertEqual(resumed[0]["estimated_cost_usd"], 0.03)
        self.assertEqual(resumed[1], saved[1])

    def test_resume_preserves_legacy_paid_and_retries_legacy_skips_and_failures(self):
        from dataset_schema import RESULT_FIELDS
        rows, _ = self.run_mocked([self.error(), self.response], max_retries=0)
        third = replace(self.records[0], prompt_id="smoke_test:third")
        skipped = dict(rows[0], prompt_id=third.prompt_id, status="skipped_model", error_type="model_disabled_after_provider_failure")
        legacy = [{k: v for k, v in r.items() if k in RESULT_FIELDS[:-6]} for r in rows + [skipped]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.csv"
            write_results(legacy, path)
            saved = read_results(path)
        resumed, calls = self.run_mocked([self.response, self.response], records=self.records + [third], completed_results=saved)
        self.assertEqual(calls, 2)
        self.assertEqual(resumed[1], saved[1])
        self.assertEqual([r["provider_attempts"] for r in (resumed[0], resumed[2])], [2, 1])

    def test_resume_does_not_pay_for_parsing_or_grading_failures_twice(self):
        for status in ("parsing_failure", "grading_failure"):
            with self.subTest(status=status):
                row = generate_results([self.records[0]], dry_run=True, models=[self.model])[0]
                row.update(status=status, error_type=status, score=None, correct=None)
                _, calls = self.run_mocked([], records=[self.records[0]], completed_results=[row])
                self.assertEqual(calls, 0)

    def test_grading_failure_has_null_correctness_and_paid_raw_response_checkpoint(self):
        snapshots = []
        with patch("generate_dataset.grade_parsed", side_effect=RuntimeError("grader unavailable")):
            rows, _ = self.run_mocked([self.response], records=[self.records[0]], on_result=lambda r: snapshots.append([dict(x) for x in r]))
        self.assertEqual(rows[0]["status"], "grading_failure")
        self.assertIsNone(rows[0]["correct"])
        self.assertIsNone(rows[0]["score"])
        self.assertEqual(snapshots[0][0]["raw_response"], "B")
        self.assertTrue(is_completed_result(snapshots[0][0]))

    def test_provider_and_model_quota_blocks_do_not_stop_independent_provider(self):
        same = replace(self.model, key="other", api_model_identifier="other-id")
        independent = enabled_models()[1]
        for scope, effects, expected_calls in (
            ("model", [self.error("quota_or_billing_error", False, failure_scope="model"), self.response, self.response, self.response, self.response], 5),
            ("provider", [self.error("quota_or_billing_error", False, failure_scope="provider"), self.response, self.response], 3),
        ):
            with self.subTest(scope=scope):
                rows, calls = self.run_mocked(effects, models=[self.model, same, independent])
                self.assertEqual(calls, expected_calls)
                self.assertEqual(rows[-1]["status"], "success")
                self.assertEqual(rows[3]["status"], "skipped_model")
                self.assertEqual(rows[3]["failure_scope"], scope)

    def test_retry_reserves_are_checkpointed_and_cannot_evade_spend_cap(self):
        from generate_dataset import render_prompt
        bound = max_call_cost_usd(self.model, render_prompt(self.records[0]), 128)
        plan = build_run_plan(planned_calls(self.records, [self.model], max_output_tokens=128), spend_cap=bound * Decimal("2.5"))
        snapshots = []
        rows, calls = self.run_mocked([self.error(), self.error()], run_plan=plan, max_output_tokens=128,
                                      max_retries=2, on_result=lambda r: snapshots.append([dict(x) for x in r]))
        self.assertEqual(calls, 2)
        self.assertEqual(rows[0]["error_type"], "spend_limit_error")
        self.assertAlmostEqual(rows[0]["estimated_cost_usd"], float(bound * 2))
        self.assertAlmostEqual(snapshots[0][0]["estimated_cost_usd"], float(bound))
        self.assertEqual(rows[0]["provider_attempts"], 2)

    def test_response_reporting_spend_above_cap_is_saved_and_not_repaid(self):
        plan = build_run_plan(planned_calls(self.records, [self.model], max_output_tokens=128), spend_cap="0.01")
        response = replace(self.response, provider_reported_cost_usd=0.02)
        rows, calls = self.run_mocked([response], run_plan=plan, max_output_tokens=128)
        self.assertEqual(calls, 1)
        self.assertEqual(rows[0]["status"], "success")
        self.assertEqual(rows[0]["estimated_cost_usd"], 0.02)
        _, calls = self.run_mocked([], records=[self.records[0]], completed_results=rows)
        self.assertEqual(calls, 0)

    def test_attempt_ceiling_is_unchanged_even_when_retries_are_requested(self):
        rows, calls = self.run_mocked([self.error()], max_api_attempts=1, max_retries=2)
        self.assertEqual(calls, 1)
        self.assertEqual(rows[0]["error_type"], "spend_limit_error")

    def test_resume_tracker_includes_prior_cost_by_default(self):
        row = generate_results([self.records[0]], dry_run=True, models=[self.model])[0]
        row.update(estimated_cost_usd=0.01)
        plan = build_run_plan(planned_calls(self.records, [self.model], max_output_tokens=128), spend_cap="0.01")
        rows, calls = self.run_mocked([], run_plan=plan, max_output_tokens=128, completed_results=[row])
        self.assertEqual(calls, 0)
        self.assertEqual(rows[-1]["error_type"], "spend_limit_error")

    def test_generic_offline_regrade_handles_partial_hard_pilot_without_providers(self):
        row = generate_results([self.records[0]], dry_run=True, models=[self.model])[0]
        row.update(status="grading_failure", error_type="grading_failure", correct=None, score=None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hard_pilot.csv"
            write_results([row], path)
            with patch("sys.argv", ["generate_dataset.py", "--regrade-results", "--output", str(path)]):
                with patch("providers.call_model", side_effect=AssertionError("offline only")), patch("generate_dataset.load_dotenv", side_effect=AssertionError("no keys needed")), patch("builtins.print"):
                    main()
            regraded = read_results(path)
        self.assertEqual(regraded[0]["status"], "success")
        self.assertEqual(regraded[0]["raw_response"], row["raw_response"])
        self.assertEqual(regraded[0]["estimated_cost_usd"], row["estimated_cost_usd"])

    def test_cli_resume_corrected_invalid_model_id_preserves_pending_spend(self):
        row = generate_results([self.records[0]], dry_run=True, models=[self.model])[0]
        row.update(api_model_identifier="invalid-old-id", status="provider_error", error_type="invalid_model_error",
                   score=None, correct=None, estimated_cost_usd=0.005,
                   configuration_fingerprint="old-configuration", provider_attempts=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.csv"
            write_results([row], path)
            with patch("sys.argv", ["generate_dataset.py", "--smoke-test", "--max-spend-usd", "0.25", "--resume", "--output", str(path)]):
                with patch("providers.call_model", return_value=self.response) as calls, patch("builtins.print"):
                    main()
            saved = read_results(path)
        self.assertEqual(calls.call_count, 5)
        self.assertEqual(len(saved), 5)
        self.assertEqual(saved[0]["api_model_identifier"], self.model.api_model_identifier)
        self.assertEqual(saved[0]["provider_attempts"], 2)
        self.assertGreater(saved[0]["estimated_cost_usd"], 0.005)


    def test_offline_regrade_preserves_raw_response_and_api_telemetry(self):
        model = enabled_models()[0]
        record = BenchmarkRecord(
            "math_500:test", "Return one half.", "math", "math_500", r"\frac{1}{2}"
        )
        row = generate_results([record], dry_run=True, models=[model])[0]
        row.update(raw_response="x = 1/2", parsed_answer="x = 1/2", score=0.0, correct=False)
        preserved = {
            field: row[field]
            for field in (
                "raw_response",
                "input_tokens",
                "output_tokens",
                "estimated_cost_usd",
                "latency_ms",
                "timestamp",
                "provider_request_id",
            )
        }

        regraded = regrade_results([row], [record])[0]

        self.assertTrue(regraded["correct"])
        self.assertEqual(regraded["score"], 1.0)
        self.assertEqual(regraded["status"], "success")
        self.assertEqual({field: regraded[field] for field in preserved}, preserved)

    def test_resume_skips_paid_terminal_pair(self):
        model = enabled_models()[0]
        record = smoke_test_prompt()
        existing = generate_results([record], dry_run=True, models=[model])[0]
        self.assertTrue(is_completed_result(existing))
        fake = ProviderResponse("B", 5, 1, 1.0)
        caller = Mock(return_value=(fake, 0))
        with patch("generate_dataset.call_model_with_retries", caller):
            results = generate_results(
                [record],
                dry_run=False,
                completed_results=[existing],
                models=[model],
            )
        caller.assert_not_called()
        self.assertEqual(results, [existing])

    def test_parsing_failure_is_not_graded_wrong(self):
        model = enabled_models()[0]
        fake = ProviderResponse("I decline to answer.", 5, 5, 1.0)
        with patch("generate_dataset.call_model_with_retries", return_value=(fake, 0)):
            rows = generate_results(
                [smoke_test_prompt()], dry_run=False, models=[model]
            )
        self.assertEqual(rows[0]["status"], "parsing_failure")
        self.assertIsNone(rows[0]["score"])
        self.assertIsNone(rows[0]["correct"])

    def test_provider_failure_does_not_abort_other_provider(self):
        first, second = enabled_models()[:2]
        response = ProviderResponse("B", 5, 1, 1.0)
        error = ProviderError(
            first.inference_provider,
            first.api_model_identifier or "",
            "exact provider error",
            status_code=401,
            error_type="authentication_error",
        )
        with patch(
            "generate_dataset.call_model_with_retries",
            side_effect=[error, (response, 0)],
        ):
            rows = generate_results(
                [smoke_test_prompt()], dry_run=False, models=[first, second]
            )
        self.assertEqual([row["status"] for row in rows], ["provider_error", "success"])
        self.assertEqual(rows[0]["error_message"], "exact provider error")

    def test_transient_retry_limit_is_two(self):
        model = enabled_models()[0]
        transient = ProviderError(
            model.inference_provider,
            model.api_model_identifier or "",
            "busy",
            status_code=503,
            error_type="transient_provider_error",
            retryable=True,
        )
        with patch("providers.call_model", side_effect=[transient, transient, transient]) as caller:
            with self.assertRaises(ProviderError):
                call_model_with_retries(
                    model,
                    "prompt",
                    dry_run=False,
                    max_retries=2,
                    sleep=lambda _: None,
                )
        self.assertEqual(caller.call_count, 3)

    def test_non_retryable_error_is_never_retried(self):
        model = enabled_models()[0]
        invalid = ProviderError(
            model.inference_provider,
            model.api_model_identifier or "",
            "invalid model",
            status_code=400,
            error_type="invalid_model_error",
            invalid_model=True,
        )
        with patch("providers.call_model", side_effect=invalid) as caller:
            with self.assertRaises(ProviderError):
                call_model_with_retries(
                    model,
                    "prompt",
                    dry_run=False,
                    max_retries=2,
                    sleep=lambda _: None,
                )
        self.assertEqual(caller.call_count, 1)

    def test_saved_invalid_model_error_prevents_future_payment(self):
        model = enabled_models()[0]
        record = smoke_test_prompt()
        existing = generate_results([record], dry_run=True, models=[model])[0]
        existing.update(
            status="provider_error",
            score=None,
            correct=None,
            error_type="invalid_model_error",
            error_message="model not found",
        )
        caller = Mock()
        with patch("generate_dataset.call_model_with_retries", caller):
            rows = generate_results(
                [record], dry_run=False, completed_results=[existing], models=[model]
            )
        caller.assert_not_called()
        self.assertEqual(rows[0]["error_message"], "model not found")


if __name__ == "__main__":
    unittest.main()
