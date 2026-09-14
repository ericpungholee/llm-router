import unittest
from dataclasses import replace
from unittest.mock import patch

from generate_dataset import generate_results, smoke_test_prompt
from model_registry import enabled_models
from pilot_analysis import analyze_results


class PilotAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.models = enabled_models()[:2]
        self.records = [replace(smoke_test_prompt(), prompt_id=f"smoke_test:{i}") for i in range(3)]
        self.rows = generate_results(self.records, dry_run=True, models=self.models)
        # Complementary model outcomes: cheap solves 0, expensive solves 1;
        # neither solves 2. No probabilistic or LLM analysis is involved.
        for row in self.rows:
            cheap = row["api_model_identifier"] == self.models[0].api_model_identifier
            correct = row["prompt_id"] == ("smoke_test:0" if cheap else "smoke_test:1")
            row.update(correct=correct, score=float(correct), estimated_cost_usd=0.01 if cheap else 0.03,
                       response_cost_usd=0.01 if cheap else 0.03,
                       provider_attempts=1, provider_response_count=1, latency_ms=10 if cheap else 20)

    def test_oracle_baselines_cost_and_complementary_prompt_signal(self):
        report = analyze_results(self.rows)
        comparison = report["routing_comparison"]
        self.assertEqual(comparison["oracle_accuracy"], 2 / 3)
        self.assertEqual(comparison["always_best_single_model"]["accuracy"], 1 / 3)
        self.assertEqual(comparison["always_cheapest_model"]["accuracy"], 1 / 3)
        self.assertAlmostEqual(comparison["always_cheapest_model"]["total_response_cost_usd"], 0.03)
        self.assertAlmostEqual(comparison["oracle_cheapest_correct_cost_usd"], 0.04)
        self.assertEqual(comparison["oracle_unsolved_prompt_ids"], ["smoke_test:2"])
        self.assertEqual([r["prompt_id"] for r in comparison["cheaper_solved_expensive_missed"]], ["smoke_test:0"])
        self.assertEqual([r["prompt_id"] for r in comparison["prompts_requiring_higher_cost_models"]], ["smoke_test:1"])
        self.assertTrue(comparison["evidence_of_routing_signal"])
        self.assertTrue(report["matrix_completeness"]["valid_complete_comparison"])
        self.assertEqual(report, analyze_results(list(reversed(self.rows))))

    def test_failure_categories_are_excluded_from_accuracy_denominator(self):
        model = enabled_models()[0]
        records = [replace(smoke_test_prompt(), prompt_id=f"smoke_test:failure-{i}") for i in range(6)]
        rows = generate_results(records, dry_run=True, models=[model])
        for index, row in enumerate(rows):
            status = ["success", "success", "provider_error", "parsing_failure", "grading_failure", "skipped_model"][index]
            row.update(status=status, correct=(index == 0) if status == "success" else None,
                       score=float(index == 0) if status == "success" else None,
                       error_type="" if status == "success" else status,
                       provider_attempts=0 if status == "skipped_model" else 1,
                       provider_response_count=int(status in {"success", "parsing_failure", "grading_failure"}),
                       estimated_cost_usd=0.01, latency_ms=10 if status in {"success", "parsing_failure", "grading_failure"} else 999)
        report = analyze_results(rows)
        metrics = next(iter(report["by_model"].values()))
        self.assertEqual(metrics["attempted_provider_calls"], 5)
        self.assertEqual(metrics["successful_provider_responses"], 4)
        self.assertEqual(metrics["successfully_graded_responses"], 2)
        self.assertEqual(metrics["correct_responses"], 1)
        self.assertEqual(metrics["accuracy_over_graded"], 0.5)
        for field in ("provider_failures", "parsing_failures", "grading_failures", "skipped_pairs"):
            self.assertEqual(metrics[field], 1)
        self.assertAlmostEqual(metrics["total_recorded_cost_usd"], 0.06)
        self.assertEqual(metrics["average_response_latency_ms"], 10)
        self.assertFalse(report["matrix_completeness"]["valid_complete_comparison"])
        self.assertEqual(report["matrix_completeness"]["successfully_graded_pairs"], 2)
        self.assertEqual(next(iter(report["by_benchmark_and_model"]["smoke_test"].values()))["accuracy_over_graded"], 0.5)

    def test_incomplete_prompts_use_common_subset_and_absent_pairs_are_reported(self):
        report = analyze_results(self.rows[:-1], [r.prompt_id for r in self.records] + ["wholly-absent"],
                                 [(m.inference_provider, m.api_model_identifier) for m in self.models])
        self.assertEqual(report["matrix_completeness"]["expected_pairs"], 8)
        self.assertEqual(report["matrix_completeness"]["successfully_graded_pairs"], 5)
        self.assertEqual(report["routing_comparison"]["comparison_prompt_count"], 2)
        self.assertEqual(report["routing_comparison"]["oracle_accuracy"], 1)
        self.assertEqual(report["routing_comparison"]["always_best_single_model"]["accuracy"], 0.5)
        self.assertEqual(sum(v["unrecorded_pairs"] for v in report["by_model"].values()), 3)

    def test_no_graded_response_has_undefined_accuracy(self):
        row = dict(self.rows[0], status="provider_error", error_type="network_error", correct=None, score=None,
                   provider_response_count=0)
        report = analyze_results([row])
        self.assertIsNone(next(iter(report["by_model"].values()))["accuracy_over_graded"])
        self.assertIsNone(report["routing_comparison"]["oracle_accuracy"])
        self.assertIsNone(report["routing_comparison"]["evidence_of_routing_signal"])

    def test_failure_reserves_do_not_inflate_oracle_response_cost(self):
        rows = [dict(r, estimated_cost_usd=1.0, retry_count=1, provider_attempts=2) for r in self.rows]
        report = analyze_results(rows)
        self.assertEqual(next(iter(report["by_model"].values()))["attempted_provider_calls"], 6)
        self.assertEqual(next(iter(report["by_model"].values()))["total_recorded_cost_usd"], 3)
        self.assertAlmostEqual(report["routing_comparison"]["oracle_cheapest_correct_cost_usd"], 0.04)

    def test_duplicate_pairs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate prompt/model"):
            analyze_results(self.rows + [self.rows[0]])

    def test_analysis_cannot_call_providers(self):
        with patch("providers.call_model", side_effect=AssertionError("must remain offline")):
            analyze_results(self.rows)


if __name__ == "__main__":
    unittest.main()
