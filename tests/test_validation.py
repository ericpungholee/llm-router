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
    UnsafeGeneratedCode,
    grade_code,
    grade_math,
    grade_multiple_choice,
    run_trusted_dry_run_tests,
)
from model_registry import MODEL_REGISTRY, validate_registry


def valid_result():
    return {
        "prompt_id": "benchmark:1",
        "prompt": "Question",
        "task_type": "multiple_choice",
        "benchmark": "benchmark",
        "reference_answer": "A",
        "model_creator": "Creator",
        "model_name": "Model",
        "api_model_identifier": "model-id",
        "inference_provider": "provider",
        "model_type": "closed",
        "temperature": "provider_default",
        "reasoning_setting": "provider_default",
        "max_output_tokens": 100,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_response": "A",
        "parsed_answer": "A",
        "score": 1.0,
        "correct": True,
        "input_tokens": 10,
        "output_tokens": 1,
        "estimated_cost_usd": 0.001,
        "latency_ms": 100.0,
        "status": "success",
        "error_type": "",
        "error_code": "",
        "error_message": "",
        "http_status": None,
        "retry_count": 0,
        "provider_request_id": "request-id",
        "response_model_identifier": "model-id",
        "stop_reason": "stop",
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

    def test_pilot_math_equivalence_regressions(self):
        cases = (
            (r"\(\left(3,\frac{\pi}{2}\right)\)", r"\left( 3, \frac{\pi}{2} \right)"),
            (r"(3, \pi/2)", r"\left( 3, \frac{\pi}{2} \right)"),
            (r"\(\frac{3}{2}\)", r"\frac{3}{2}"),
            ("x = 3/2", r"\frac{3}{2}"),
            ("3/2", r"\frac{3}{2}"),
            (r"\( \frac{3}{2} \)", r"\frac{3}{2}"),
            (r"\(\frac{3}{2}\)", r"\frac{3}{2}"),
            ("**x = 83**", "83"),
        )
        for candidate, reference in cases:
            with self.subTest(candidate=candidate):
                self.assertTrue(grade_math(candidate, reference))

    def test_math_grader_handles_exact_numeric_forms_conservatively(self):
        for candidate in ("0.5", "1/2", r"\boxed{\frac{1}{2}}", "x = 0.500"):
            with self.subTest(candidate=candidate):
                self.assertTrue(grade_math(candidate, r"\frac{1}{2}"))
        self.assertFalse(grade_math("0.5001", r"\frac{1}{2}"))
        self.assertFalse(grade_math("sqrt(4)", "2"))

    def test_pilot_import_regressions(self):
        cases = (
            (
                "import sys\nX = int(sys.stdin.readline())\n"
                "ans = sum(i*j for i in range(1,10) for j in range(1,10) if i*j != X)\nprint(ans)",
                ({"input": "1", "output": "2024"}, {"input": "24", "output": "1929"}),
            ),
            (
                "import sys\nX = int(sys.stdin.readline())\ntotal = 2025\ncount = 0\n"
                "for i in range(1,10):\n    for j in range(1,10):\n"
                "        if i*j == X:\n            count += 1\nprint(total - X*count)",
                ({"input": "11", "output": "2025"}, {"input": "24", "output": "1929"}),
            ),
            (
                "import sys\ndef main():\n    data=sys.stdin.read().split()\n"
                "    if not data: return\n    print(2*int(data[0]))\n"
                "if __name__ == '__main__': main()",
                ({"input": "2\n", "output": "4\n"}, {"input": "-7\n", "output": "-14\n"}),
            ),
            (
                "import sys\nn=int(sys.stdin.read().strip())\nprint(2*n)",
                ({"input": "2\n", "output": "4\n"}, {"input": "-7\n", "output": "-14\n"}),
            ),
            (
                "import sys\ndef main():\n    data=sys.stdin.read().split()\n"
                "    a,b=int(data[0]),int(data[1])\n    print(a+b)\n"
                "if __name__ == '__main__': main()",
                ({"input": "3 5\n", "output": "8\n"}, {"input": "-4 1\n", "output": "-3\n"}),
            ),
            (
                "import sys\nA,B=map(int,sys.stdin.read().split())\nprint(A+B)",
                ({"input": "3 5\n", "output": "8\n"}, {"input": "-4 1\n", "output": "-3\n"}),
            ),
            (
                "import sys\ndata=sys.stdin.read().split()\nA,B=map(int,data[:2])\nprint(A+B)",
                ({"input": "3 5\n", "output": "8\n"}, {"input": "-4 1\n", "output": "-3\n"}),
            ),
        )
        for code, tests in cases:
            with self.subTest(code=code):
                self.assertTrue(run_trusted_dry_run_tests(code, tests))

    def test_allowed_standard_library_imports_and_rejects_other_modules(self):
        code = (
            "import sys, math, collections, itertools, functools, heapq, bisect\n"
            "print(math.floor(functools.reduce(lambda a,b:a+b, itertools.repeat(1, 1))))"
        )
        self.assertTrue(
            run_trusted_dry_run_tests(code, ({"input": "", "output": "1"},))
        )
        with self.assertRaises(UnsafeGeneratedCode):
            run_trusted_dry_run_tests(
                "import os\nprint(1)", ({"input": "", "output": "1"},)
            )


if __name__ == "__main__":
    unittest.main()
