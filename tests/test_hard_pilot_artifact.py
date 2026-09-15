"""Audit the published experiment input and reproduce it without ignored data."""

import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from generate_dataset import benchmark_records_from_saved_results, read_results, regrade_results

ROOT = Path(__file__).resolve().parents[1]


class HardPilotArtifactTests(unittest.TestCase):
    def test_audit_reconstructs_original_and_regrade_only_changes_grading_fields(self):
        provenance = json.loads(
            (ROOT / "reports/snapshots/hard_pilot_final.provenance.json").read_text()
        )
        snapshot = ROOT / provenance["snapshot_path"]
        self.assertEqual(
            hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            provenance["final_artifact"]["sha256"],
        )
        with snapshot.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            original = list(reader)

        def key(r):
            return (r["prompt_id"], r["inference_provider"], r["api_model_identifier"])

        matrix = {key(r): r for r in original}
        for change in provenance["offline_regrade"]["changed_rows"]:
            for field, values in change["fields"].items():
                self.assertIn(field, provenance["offline_regrade"]["allowed_changed_fields"])
                self.assertEqual(matrix[key(change)][field], values["after"])
                matrix[key(change)][field] = values["before"]
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(original)
        original_bytes = output.getvalue().encode("utf-8")
        self.assertEqual(
            hashlib.sha256(original_bytes).hexdigest(),
            provenance["before_offline_regrade"]["sha256"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "before.csv"
            path.write_bytes(original_bytes)
            before = read_results(path)
        with patch("providers.call_model", side_effect=AssertionError("offline only")):
            after = regrade_results(before, benchmark_records_from_saved_results(before))
        self.assertEqual(after, read_results(snapshot))
        for a, b in zip(before, after):
            for field in a:
                if field not in provenance["offline_regrade"]["allowed_changed_fields"]:
                    self.assertEqual(a[field], b[field], field)
        self.assertEqual(sum(a != b for a, b in zip(before, after)), 1)

    def test_fresh_checkout_without_data_reproduces_all_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            for path in ROOT.glob("*.py"):
                shutil.copy2(path, checkout / path.name)
            for name in ("benchmarks", "provider_clients", "reports"):
                shutil.copytree(
                    ROOT / name, checkout / name, ignore=shutil.ignore_patterns("__pycache__")
                )
            self.assertFalse((checkout / "data").exists())
            result = subprocess.run(
                [sys.executable, "pilot_analysis.py", "reports/snapshots/hard_pilot_final.csv"],
                cwd=checkout,
                check=True,
                capture_output=True,
                text=True,
            )
            expected = json.loads((ROOT / "reports/hard_pilot_existing_analysis.json").read_text())
            self.assertEqual(json.loads(result.stdout), expected)
            subprocess.run(
                [sys.executable, "reports/reproduce_hard_pilot.py", "--output-dir", "rebuilt"],
                cwd=checkout,
                check=True,
                capture_output=True,
                text=True,
            )
            for name in (
                "hard_pilot_existing_analysis.json",
                "hard_pilot_routing_analysis.json",
                "hard_pilot_routing_analysis.md",
            ):
                self.assertEqual(
                    (checkout / "rebuilt" / name).read_bytes(),
                    (ROOT / "reports" / name).read_bytes(),
                    name,
                )
            report = json.loads((checkout / "rebuilt/hard_pilot_routing_analysis.json").read_text())
            self.assertEqual(
                report["provenance"]["snapshot_path"], "reports/snapshots/hard_pilot_final.csv"
            )
            self.assertIn(
                report["provenance"]["snapshot_path"],
                (checkout / "rebuilt/hard_pilot_routing_analysis.md").read_text(),
            )
            self.assertFalse((checkout / "data").exists())
            # An accidental snapshot edit must fail loudly instead of silently changing the experiment.
            snapshot = checkout / "reports/snapshots/hard_pilot_final.csv"
            snapshot.write_bytes(snapshot.read_bytes() + b"\n")
            failed = subprocess.run(
                [sys.executable, "reports/reproduce_hard_pilot.py"],
                cwd=checkout,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("SHA-256 differs", failed.stderr)


if __name__ == "__main__":
    unittest.main()
