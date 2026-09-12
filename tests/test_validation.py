import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from benchmark_loaders import (
    BenchmarkRecord,
    load_manifest,
    validate_benchmark_records,
)
from dataset_schema import validate_result, validate_results
from graders import (
    MalformedModelOutput,
    grade_code,
    grade_multiple_choice,
    run_trusted_dry_run_tests,
)
from model_registry import MODEL_REGISTRY, validate_registry


def valid_result():
    return {
        "prompt_id": "benchmark:1",
        "prompt": "Question",
        "task_type": "multiple_choice",
        "benchmark_name": "benchmark",
        "reference_answer": "A",
        "model_creator": "Creator",
        "model_name": "Model",
        "api_model_id": "model-id",
        "inference_provider": "provider",
        "model_type": "closed",
        "temperature": "provider_default",
        "reasoning_setting": "provider_default",
        "max_output_tokens": 100,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_output": "A",
        "score": 1.0,
        "correct": True,
        "input_tokens": 10,
        "output_tokens": 1,
        "estimated_cost_usd": 0.001,
        "latency_ms": 100.0,
    }


class ValidationTests(unittest.TestCase):
    def test_duplicate_prompt_ids_are_rejected(self):
        record = BenchmarkRecord(
            "duplicate", "Question", "math", "math_500", "1"
        )
        with self.assertRaisesRegex(ValueError, "Duplicate prompt IDs"):
            validate_benchmark_records([record, record])

    def test_missing_reference_answer_is_rejected(self):
        record = BenchmarkRecord("id", "Question", "math", "math_500", "")
        with self.assertRaisesRegex(ValueError, "missing a reference answer"):
            validate_benchmark_records([record])

    def test_missing_model_metadata_is_rejected(self):
        broken = replace(MODEL_REGISTRY[0], creator="")
        with self.assertRaisesRegex(ValueError, "missing creator"):
            validate_registry([broken])

    def test_invalid_cost_is_rejected(self):
        row = valid_result()
        row["estimated_cost_usd"] = -1
        with self.assertRaisesRegex(ValueError, "estimated_cost_usd"):
            validate_result(row)

    def test_inconsistent_result_task_type_is_rejected(self):
        row = valid_result()
        with self.assertRaisesRegex(ValueError, "expected 'math'"):
            validate_results([row], {"benchmark": "math"})

    def test_inconsistent_manifest_task_type_is_rejected(self):
        manifest = {
            "version": 1,
            "benchmarks": [
                {
                    "name": "broken",
                    "task_type": "math",
                    "loader": "mmlu_pro",
                    "source": "source",
                    "source_revision": "revision",
                    "sample_file": "sample.jsonl",
                    "enabled": True,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires 'multiple_choice'"):
                load_manifest(path)

    def test_malformed_multiple_choice_output_is_rejected(self):
        with self.assertRaises(MalformedModelOutput):
            grade_multiple_choice("The answer is A", "A", ["A", "B"])

    def test_code_is_graded_by_executable_tests(self):
        tests = ({"input": "2\n", "output": "4\n"},)
        self.assertTrue(
            grade_code(
                "x = int(input())\nprint(x * 2)",
                tests,
                run_trusted_dry_run_tests,
            )
        )
        self.assertFalse(
            grade_code("print(0)", tests, run_trusted_dry_run_tests)
        )


if __name__ == "__main__":
    unittest.main()

