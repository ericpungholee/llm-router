"""Offline reproducibility and payment-boundary tests for the closed pilot."""

import copy
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from generate_dataset import result_key
from hard_pilot_experiment import (
    PINNED_FILES,
    PROVENANCE,
    ROOT,
    SNAPSHOT,
    SOURCE,
    build_report,
    freeze,
    frozen_inputs,
    grading_records,
    select_exhausted,
    treatment_calls,
    treatment_plan,
    validate_treatment_output,
)
from hard_pilot_treatment import execute, prepare
from offline_guard import offline_only
from provider_clients.base import ProviderError
from spend_control import SpendPreflightError, estimate_max_spend_usd


class ClosedPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.models, cls.plan, cls.provenance = frozen_inputs()

    def test_frozen_accounting_and_oracle(self):
        with offline_only() as audit:
            report = build_report()
        self.assertEqual(audit["blocked_network_attempts"], 0)
        self.assertEqual(report["offline_audit"]["provider_calls"], 0)
        self.assertEqual(report["overall"]["successfully_graded_pairs"], 45)
        self.assertEqual(report["overall"]["attempted_provider_calls"], 83)
        self.assertEqual(report["recorded_spend_usd"], "1.9491623")
        comparison = report["analysis"]["routing_comparison"]
        self.assertEqual(comparison["comparison_prompt_count"], 5)
        self.assertEqual(comparison["oracle_accuracy"], 1.0)
        self.assertEqual(comparison["oracle_cheapest_correct_cost_usd"], 0.0029718)
        self.assertEqual(report["common_disagreement_count"], 3)
        self.assertEqual(report["output_exhaustion"]["count"], 20)
        self.assertEqual(
            report["output_exhaustion"]["by_task_type"],
            {"code": 14, "math": 2, "multiple_choice": 4},
        )
        self.assertFalse(report["decision"]["suitable_for_learned_router_training_or_evaluation"])

    def test_freeze_copies_exact_bytes_without_calls_or_source_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / SOURCE
            source.parent.mkdir(parents=True)
            source.write_bytes((ROOT / SNAPSHOT).read_bytes())
            for name in PINNED_FILES:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, target)
            before = (source.read_bytes(), source.stat().st_mtime_ns)
            commit = subprocess.CompletedProcess([], 0, stdout=self.provenance["git_commit"] + "\n")
            with patch("hard_pilot_experiment.subprocess.run", return_value=commit):
                provenance = freeze(root)
            self.assertEqual((source.read_bytes(), source.stat().st_mtime_ns), before)
            self.assertEqual((root / SNAPSHOT).read_bytes(), before[0])
            self.assertEqual(provenance["source_sha256"], provenance["frozen_sha256"])
            self.assertEqual(provenance["source_sha256"], hashlib.sha256(before[0]).hexdigest())
            self.assertTrue(provenance["no_provider_calls_while_freezing"])
            self.assertEqual(
                provenance["offline_audit"], {"provider_calls": 0, "blocked_network_attempts": 0}
            )
            with self.assertRaises(FileExistsError):
                freeze(root)
            self.assertEqual((source.read_bytes(), source.stat().st_mtime_ns), before)

    def test_fresh_checkout_reproduces_report_and_preflight_without_data_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            for path in ROOT.glob("*.py"):
                shutil.copy2(path, checkout / path.name)
            for name in ("benchmarks", "provider_clients", "reports"):
                shutil.copytree(
                    ROOT / name, checkout / name, ignore=shutil.ignore_patterns("__pycache__")
                )
            self.assertFalse((checkout / "data").exists())
            self.assertFalse((checkout / ".git").exists())
            # Block sockets even in the subprocess; its workflow adds provider guards.
            boot = (
                "import runpy,sys,socket,urllib.request; "
                "deny=lambda *a,**k: (_ for _ in ()).throw(AssertionError('offline only')); "
                "socket.socket.connect=deny;socket.create_connection=deny;urllib.request.urlopen=deny; "
                "sys.argv=['reports/reproduce_4096_pilot.py','--output-dir','rebuilt']; "
                "runpy.run_path(sys.argv[0],run_name='__main__')"
            )
            subprocess.run(
                [sys.executable, "-c", boot],
                cwd=checkout,
                check=True,
                capture_output=True,
                text=True,
            )
            for name in ("hard_pilot_4096_analysis.json", "hard_pilot_4096_analysis.md"):
                self.assertEqual(
                    (checkout / "rebuilt" / name).read_bytes(),
                    (ROOT / "reports" / name).read_bytes(),
                )
            preflight = subprocess.run(
                [
                    sys.executable,
                    "hard_pilot_treatment.py",
                    "--max-output-tokens",
                    "8192",
                    "--max-spend-usd",
                    "1.66",
                    "--output",
                    "data/results/treatment.csv",
                ],
                cwd=checkout,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(preflight.returncode, 0)
            self.assertIn('"planned_calls": 20', preflight.stdout)
            self.assertIn("Strict preflight BLOCKED", preflight.stderr)
            self.assertFalse((checkout / "data").exists())
            frozen = checkout / SNAPSHOT
            frozen.write_bytes(frozen.read_bytes() + b"\n")
            failed = subprocess.run(
                [sys.executable, "reports/reproduce_4096_pilot.py", "--output-dir", "rebuilt"],
                cwd=checkout,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("SHA-256 differs", failed.stderr)

    def test_only_explicit_final_exhaustion_is_selected(self):
        selected = select_exhausted(self.rows)
        self.assertEqual(len(selected), 20)
        self.assertEqual(len({result_key(r) for r in selected}), 20)
        self.assertTrue(
            all(
                r["status"] == "provider_error" and r["error_type"] == "output_limit_exhausted"
                for r in selected
            )
        )
        self.assertFalse(
            {result_key(r) for r in selected}
            & {result_key(r) for r in self.rows if r["status"] == "success"}
        )
        self.assertFalse(any(r["inference_provider"] == "xai" for r in selected))
        self.assertFalse(any(r["prompt_id"] == "livecodebench:abc400_g" for r in selected))

    def test_success_even_with_stale_exhaustion_label_is_never_selected(self):
        rows = copy.deepcopy(self.rows)
        row = next(r for r in rows if r["status"] == "success")
        row["error_type"] = "output_limit_exhausted"
        self.assertNotIn(result_key(row), {result_key(r) for r in select_exhausted(rows)})

    def test_exhaustion_history_cannot_select_a_different_final_error(self):
        rows = copy.deepcopy(self.rows)
        row = select_exhausted(rows)[0]
        target = next(r for r in rows if result_key(r) == result_key(row))
        target["error_type"] = "network_error"
        self.assertNotIn(result_key(row), {result_key(r) for r in select_exhausted(rows)})

    def test_grok_cannot_enter_even_if_it_is_explicitly_exhausted(self):
        rows = copy.deepcopy(self.rows)
        row = next(
            r for r in rows if r["inference_provider"] == "xai" and r["status"] == "provider_error"
        )
        row["error_type"] = "output_limit_exhausted"
        self.assertNotIn(result_key(row), {result_key(r) for r in select_exhausted(rows)})
        row["inference_provider"] = "openrouter"
        self.assertNotIn(result_key(row), {result_key(r) for r in select_exhausted(rows)})

    def test_generation_and_grading_inputs_preserved(self):
        registry = {(m.inference_provider, m.api_model_identifier): m for m in self.models}
        calls = treatment_calls(self.rows, self.models, 8192)
        records = grading_records(ROOT, [r for r, _, _, _ in calls])
        for row, model, prompt, budget in calls:
            old = registry[row["inference_provider"], row["api_model_identifier"]]
            self.assertEqual(replace(model, generation=old.generation), old)
            self.assertEqual(model.generation.max_output_tokens, 8192)
            self.assertEqual(prompt, row["prompt"])
            self.assertEqual(budget, 8192)
            self.assertIn(row["prompt_id"], records)
        changed = copy.deepcopy(self.rows)
        row = next(r for r in changed if r["error_type"] == "output_limit_exhausted")
        row["reasoning_setting"] = "different"
        with self.assertRaisesRegex(ValueError, "preserve"):
            treatment_calls(changed, self.models, 8192)
        with self.assertRaises(ValueError):
            treatment_calls(self.rows, self.models, 4096)

    def test_grading_input_drift_is_rejected(self):
        selected = copy.deepcopy(select_exhausted(self.rows))
        selected[0]["prompt"] += " changed"
        with self.assertRaisesRegex(ValueError, "grading inputs differ"):
            grading_records(ROOT, selected)

    def test_costs_and_one_attempt_plan(self):
        for budget, cap, expected in ((8192, "1.66", "1.656349"), (16384, "3.22", "3.216105")):
            report, plan, calls = treatment_plan(self.rows, self.models, budget, cap)
            self.assertEqual(len(calls), 20)
            self.assertEqual(report["registry_contract_maximum_usd"], expected)
            self.assertEqual(report["maximum_attempts_per_pair"], 1)
            self.assertEqual(report["maximum_provider_attempts"], 20)
            self.assertEqual(report["max_retries"], 0)
            self.assertEqual(
                estimate_max_spend_usd([(m, p, b) for _, m, p, b in calls], max_retries=0),
                Decimal(expected),
            )
            self.assertFalse(report["strict_preflight_passed"])
            self.assertIsNone(report["verified_worst_case_spend_usd"])
            self.assertEqual(len(report["cost_bound_issues"]), 2)
        report, _, _ = treatment_plan(self.rows, self.models, 8192, "0.01")
        self.assertFalse(report["within_cap"])
        self.assertFalse(report["strict_preflight_passed"])

    def test_paid_command_stops_before_credentials_files_or_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "treatment.csv"
            prepared = prepare(8192, output, "1.66")
            with (
                patch("hard_pilot_treatment.load_dotenv") as credentials,
                patch("providers.call_model") as caller,
            ):
                with self.assertRaisesRegex(SpendPreflightError, "strict spend preflight"):
                    execute(prepared)
                credentials.assert_not_called()
                caller.assert_not_called()
            self.assertFalse(list(Path(directory).iterdir()))

    def test_configuration_drift_blocks_paid_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in (SNAPSHOT, PROVENANCE, *map(Path, PINNED_FILES)):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / path, target)
            (root / "graders.py").write_text("# changed\n")
            with self.assertRaisesRegex(ValueError, "configuration changed: graders.py"):
                prepare(8192, root / "treatment.csv", "1.66", root)

    def test_output_protection_including_aliases_and_existing_checkpoints(self):
        for p in (SOURCE, SNAPSHOT, PROVENANCE, Path("reports/snapshots/new.csv")):
            with self.assertRaisesRegex(ValueError, "cannot overwrite"):
                validate_treatment_output(ROOT / p)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "treatment.csv"
            output.symlink_to(ROOT / SNAPSHOT)
            with self.assertRaisesRegex(ValueError, "cannot overwrite"):
                validate_treatment_output(output)
            output.unlink()
            # Hard link aliases are existing files, and are equally protected.
            os.link(ROOT / SNAPSHOT, output)
            with self.assertRaises(FileExistsError):
                validate_treatment_output(output)
            output.unlink()
            sidecar = output.with_suffix(".csv.experiment.json")
            sidecar.write_text("{}")
            with self.assertRaises(FileExistsError):
                validate_treatment_output(output)
            sidecar.unlink()
            output.with_suffix(".csv.tmp").write_text("checkpoint")
            with self.assertRaises(FileExistsError):
                validate_treatment_output(output)

    def test_mocked_paid_execution_never_retries_exhaustion_or_transient_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "treatment.csv"
            prepared = prepare(8192, output, "1.66")
            # Synthetic resolved-bound fixture only; never bypass the real gate.
            synthetic = copy.deepcopy(prepared[0])
            synthetic.update(
                strict_preflight_passed=True,
                cost_bound_issues=[],
                verified_worst_case_spend_usd=synthetic["registry_contract_maximum_usd"],
            )
            resolved = (synthetic, *prepared[1:])
            seen = []

            def exhausted(model, prompt, **kwargs):
                seen.append(
                    (
                        model.inference_provider,
                        model.api_model_identifier,
                        prompt,
                        kwargs["max_output_tokens"],
                    )
                )
                kind = "output_limit_exhausted" if len(seen) % 2 else "network_error"
                raise ProviderError(
                    model.inference_provider,
                    model.api_model_identifier or "",
                    "exhausted",
                    error_type=kind,
                    retryable=kind == "network_error",
                )

            with (
                patch("hard_pilot_treatment.prepare", return_value=resolved),
                patch("hard_pilot_treatment.load_dotenv"),
                patch("hard_pilot_treatment.missing_api_keys", return_value=()),
                patch("providers.call_model", side_effect=exhausted) as caller,
                redirect_stdout(io.StringIO()),
            ):
                result = execute(resolved)
                self.assertEqual(caller.call_count, 20)
            self.assertEqual(len(set(seen)), 20)
            self.assertEqual(len(result), 20)
            self.assertTrue(
                all(r["provider_attempts"] == 1 and r["retry_count"] == 0 for r in result)
            )
            self.assertEqual(sum(r["error_type"] == "output_limit_exhausted" for r in result), 10)
            self.assertEqual(sum(r["error_type"] == "network_error" for r in result), 10)
            self.assertTrue(all(r["inference_provider"] != "xai" for r in result))
            self.assertEqual(
                {result_key(r) for r in result},
                {result_key(r) for r in select_exhausted(self.rows)},
            )
            with self.assertRaises(FileExistsError):
                prepare(8192, output, "1.66")

    def test_offline_guard_blocks_provider_and_network_boundary(self):
        with offline_only() as audit:
            import socket

            import providers

            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                providers.call_model(self.models[0], "test", dry_run=False)
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                socket.create_connection(("example.invalid", 443))
        self.assertEqual(audit["provider_calls"], 0)
        self.assertEqual(audit["blocked_network_attempts"], 2)


if __name__ == "__main__":
    unittest.main()
