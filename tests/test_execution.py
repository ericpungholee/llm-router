import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from benchmark_loaders import BenchmarkRecord, load_enabled_benchmarks
from generate_dataset import (
    generate_results,
    is_completed_result,
    planned_calls,
    regrade_results,
    select_pilot_prompts,
    smoke_test_prompt,
)
from graders import MalformedModelOutput, parse_math_answer, parse_multiple_choice
from model_registry import enabled_models
from provider_clients.base import ProviderError, ProviderResponse
from providers import call_model_with_retries, validate_provider_registry


class ModeSelectionTests(unittest.TestCase):
    def test_smoke_selects_exactly_one_prompt_times_five_models(self):
        records = [smoke_test_prompt()]
        calls = planned_calls(records, enabled_models(), max_output_tokens=128)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(calls), 5)
        self.assertEqual({call[2] for call in calls}, {128})

    def test_pilot_selects_exactly_nine_prompts_times_five_models(self):
        _, records = load_enabled_benchmarks(Path("benchmarks/manifest.json"))
        selected = select_pilot_prompts(records)
        self.assertEqual(len(selected), 9)
        self.assertEqual(len(planned_calls(selected, enabled_models())), 45)
        for name in ("mmlu_pro", "math_500", "livecodebench"):
            self.assertEqual(sum(row.benchmark_name == name for row in selected), 3)

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
