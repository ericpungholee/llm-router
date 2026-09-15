"""Local input validation without private datasets, model weights, or network access."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from experiments.route_local import main
from routing_ml.embeddings import file_hash
from routing_ml.local_router import LocalRouter
from routing_ml.training import MODEL_IDS


class LocalRouterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        # A real static bundle exercises loading/validation without deserializing models.
        selection = {
            "primary": {
                "candidate_id": "fixed_gpt5",
                "family": "static",
                "kind": "static",
                "value": 0.0,
                "gate_quantile": 0.0,
            },
            "reference": {"model_id": "gpt-5"},
            "mean_training_cost_usd": {m: 0.01 for m in MODEL_IDS},
        }
        frozen = self.directory / "frozen_selection.json"
        frozen.write_text(json.dumps(selection), encoding="utf-8")
        (self.directory / "artifact_hashes.json").write_text(
            json.dumps({frozen.name: file_hash(frozen)}), encoding="utf-8"
        )
        self.router = LocalRouter(self.directory)

    def test_missing_null_blank_or_duplicate_ids_are_rejected(self):
        cases = [
            pd.DataFrame({"prompt": ["hello"]}),
            *[
                pd.DataFrame({"prompt_id": [pid], "prompt": ["hello"]})
                for pid in (None, float("nan"), "", " ", 12, False)
            ],
            pd.DataFrame({"prompt_id": ["a", "a"], "prompt": ["hello", "world"]}),
            pd.DataFrame([["a", "b", "hello"]], columns=["prompt_id", "prompt_id", "prompt"]),
        ]
        for frame in cases:
            with self.subTest(columns=frame.columns.tolist()):
                with self.assertRaisesRegex(ValueError, "Unique prompt_id"):
                    self.router.predict(frame)

    def test_non_string_text_is_rejected(self):
        for value in (None, 0, False, ["text"]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "text must be strings"):
                    self.router.predict(pd.DataFrame({"prompt_id": ["a"], "prompt": [value]}))

    def test_prompt_ids_order_and_projection_are_preserved(self):
        frame = pd.DataFrame(
            {
                "prompt_id": ["z", "α"],
                "prompt": ["hello", ""],
                "success": [False, True],
                "cost_usd": [99, 0],
            }
        )
        result = self.router.predict(frame)
        self.assertEqual([row["prompt_id"] for row in result], ["z", "α"])
        self.assertTrue(all(row["selected_model_id"] == "gpt-5" for row in result))
        self.assertEqual(result, self.router.predict(frame[["prompt_id", "prompt"]]))
        self.assertEqual(self.router.predict(frame.iloc[:0]), [])

    def test_cli_jsonl_preserves_unicode_ids_and_skips_blank_lines(self):
        requests = self.directory / "requests.jsonl"
        requests.write_text('\n{"prompt_id":"α","prompt":"hello"}\n\n', encoding="utf-8")
        output = io.StringIO()
        with redirect_stdout(output):
            main(["--artifact-dir", str(self.directory), "--input", str(requests)])
        self.assertEqual(json.loads(output.getvalue())["prompt_id"], "α")

    def test_cli_invalid_records_fail_before_loading_artifacts(self):
        requests = self.directory / "requests.jsonl"
        for text in ("{", "[]", "null", '{"prompt":"missing id"}'):
            with self.subTest(text=text):
                requests.write_text(text, encoding="utf-8")
                error = io.StringIO()
                with patch("experiments.route_local.LocalRouter") as loader:
                    with redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                        main(["--input", str(requests)])
                    loader.assert_not_called()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("Input line 1", error.getvalue())
                self.assertNotIn("Traceback", error.getvalue())

    def test_cli_missing_file_and_artifact_have_actionable_errors(self):
        missing = str(self.directory / "missing")
        for arguments in (["--input", missing], ["--artifact-dir", missing, "--prompt", "hello"]):
            with self.subTest(arguments=arguments):
                error = io.StringIO()
                with redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                    main(arguments)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("No such file", error.getvalue())
                self.assertNotIn("Traceback", error.getvalue())


if __name__ == "__main__":
    unittest.main()
