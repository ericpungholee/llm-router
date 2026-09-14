"""Offline invariants and full pipeline reproduction on a small archive fixture."""

import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from routing_data.baselines import baseline_artifacts, choose_best_single
from routing_data.features import FEATURE_COLUMNS, extract_prompt_features, feature_table
from routing_data.loading import load_split
from routing_data.matrix import complete_case, coverage_table, normalize_outcomes, validate_pairs, validate_processed
from routing_data.prepare import CONFIG_NAMES, prepare, validate_pool, verify_pool_statistics, write_json
from routing_data.source import download_archive, file_hash, read_archive, verify_file
from routing_data.splits import add_leakage_groups, make_splits, validate_splits

ROOT = Path(__file__).resolve().parents[1]


def example_rows():
    rows = []
    for pid in ("p1", "p2"):
        for model in ("a", "b"):
            rows.append(dict(prompt_id=pid, model_id=model, raw_score=1.0, raw_cost_usd=0.001,
                             input_tokens=10, output_tokens=20, has_output=True, generation_failure=False))
    return pd.DataFrame(rows)


def example_prompts(n=40):
    return pd.DataFrame([dict(prompt_id=f"p{i}", prompt=f"Question {i}", origin_query=f"Question {i}",
                              dataset="code" if i >= n//2 else "math", task_type="code" if i >= n//2 else "math")
                         for i in range(n)])


SPLIT_CONFIG = dict(seed=3407, standard_fractions=[0.7, 0.15, 0.15],
                    ood_validation_fraction=0.15, ood_held_out_datasets=["code"])


class MatrixTests(unittest.TestCase):
    def test_duplicate_pairs_rejected(self):
        rows = example_rows()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_pairs(pd.concat([rows, rows.iloc[:1]]))

    def test_missing_success_not_converted_to_zero(self):
        rows = example_rows()
        rows.loc[0, "raw_score"] = None
        out = normalize_outcomes(rows)
        self.assertTrue(pd.isna(out.loc[0, "success"]))
        self.assertFalse(out.loc[0, "eligible"])

    def test_failed_request_with_zero_score_is_unavailable(self):
        rows = example_rows()
        rows.loc[0, ["raw_score", "raw_cost_usd", "input_tokens", "output_tokens"]] = 0
        rows.loc[0, "generation_failure"] = True
        out = normalize_outcomes(rows)
        self.assertEqual(out.loc[0, "raw_score"], 0)
        self.assertTrue(pd.isna(out.loc[0, "success"]))
        self.assertTrue(pd.isna(out.loc[0, "cost_usd"]))

    def test_paid_empty_final_answer_preserves_recorded_failure(self):
        rows = example_rows()
        rows.loc[0, "has_output"] = False
        rows.loc[0, "raw_score"] = 0
        out = normalize_outcomes(rows)
        self.assertTrue(out.loc[0, "eligible"])
        self.assertEqual(out.loc[0, "success"], 0)

    def test_continuous_scores_require_explicit_rule(self):
        rows = example_rows()
        rows.loc[0, "raw_score"] = 0.5
        with self.assertRaisesRegex(ValueError, "nonbinary"):
            normalize_outcomes(rows)

    def test_costs_negative_nonfinite_or_missing_are_unavailable(self):
        for value in [-1, float("inf"), float("nan"), 0]:
            rows = example_rows()
            rows.loc[0, "raw_cost_usd"] = value
            normalized = normalize_outcomes(rows)
            self.assertFalse(normalized.loc[0, "eligible"])
            self.assertTrue(pd.isna(normalized.loc[0, "cost_usd"]))
            self.assertEqual(normalized.loc[0, "success"], 1)

    def test_impossible_token_counts_quarantined(self):
        for value in [-1, 1.5, float("nan"), float("inf"), 0]:
            rows = example_rows().astype({"output_tokens": "float64"})
            rows.loc[0, "output_tokens"] = value
            self.assertFalse(normalize_outcomes(rows).loc[0, "eligible"])

    def test_complete_case_excludes_missing_model_and_missing_label(self):
        prompts = pd.DataFrame([dict(prompt_id=x, dataset="math", task_type="math") for x in ["p1", "p2"]])
        rows = normalize_outcomes(example_rows().iloc[:-1])
        complete, missing = complete_case(rows, prompts, ["a", "b"])
        self.assertEqual(set(complete.prompt_id), {"p1"})
        self.assertEqual(missing.set_index("prompt_id").loc["p2", "missing_models"], 1)
        all_rows = example_rows()
        all_rows.loc[0, "raw_score"] = None
        complete, _ = complete_case(normalize_outcomes(all_rows), prompts, ["a", "b"])
        self.assertEqual(set(complete.prompt_id), {"p2"})

    def test_coverage_uses_full_prompt_universe(self):
        rows = normalize_outcomes(example_rows().iloc[:-1])
        prompts = pd.DataFrame([dict(prompt_id=x, dataset="math", task_type="math") for x in ["p1", "p2", "p3"]])
        table = coverage_table(rows, prompts, ["a", "b"], ["model_id"]).set_index("model_id")
        self.assertAlmostEqual(table.loc["a", "coverage"], 2/3)
        self.assertAlmostEqual(table.loc["b", "coverage"], 1/3)
        self.assertEqual(table.loc["b", "missing_pairs"], 2)


class SplitFeatureBaselineTests(unittest.TestCase):
    def test_deterministic_splits_independent_of_row_order(self):
        prompts = add_leakage_groups(example_prompts())
        a = make_splits(prompts, SPLIT_CONFIG)
        b = make_splits(prompts.sample(frac=1, random_state=77), SPLIT_CONFIG)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a.standard_split.value_counts().to_dict(), {"train": 28, "validation": 6, "test": 6})

    def test_duplicate_question_templates_grouped(self):
        p = example_prompts()
        p.loc[1, "origin_query"] = "  QUESTION   0 "
        p = add_leakage_groups(p)
        self.assertEqual(p.set_index("prompt_id").loc["p0", "leakage_group"], p.set_index("prompt_id").loc["p1", "leakage_group"])
        s = make_splits(p, SPLIT_CONFIG).set_index("prompt_id")
        self.assertEqual(s.loc["p0", "standard_split"], s.loc["p1", "standard_split"])

    def test_ood_domains_absent_from_training_and_overlap_purged(self):
        p = example_prompts()
        p.loc[0, "origin_query"] = p.loc[20, "origin_query"]
        p = add_leakage_groups(p)
        s = make_splits(p, SPLIT_CONFIG)
        combined = p.merge(s.drop(columns="leakage_group"), on="prompt_id")
        self.assertEqual(set(combined[combined.ood_split == "train"].dataset), {"math"})
        self.assertEqual(combined.set_index("prompt_id").loc["p0", "ood_split"], "excluded_overlap")
        self.assertTrue((combined[combined.dataset == "code"].ood_split == "test").all())

    def test_split_validation_detects_prompt_leakage(self):
        p = example_prompts()
        p.loc[1, "origin_query"] = p.loc[0, "origin_query"]
        p = add_leakage_groups(p)
        s = make_splits(p, SPLIT_CONFIG)
        s.loc[s.prompt_id == "p0", "standard_split"] = "train"
        s.loc[s.prompt_id == "p1", "standard_split"] = "test"
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_splits(p, pd.DataFrame(), s, SPLIT_CONFIG)

    def test_all_outcomes_share_prompt_split(self):
        p = add_leakage_groups(example_prompts())
        s = make_splits(p, SPLIT_CONFIG)
        rows = pd.concat([s, s], ignore_index=True)
        rows.loc[0, "standard_split"] = "test" if s.loc[0, "standard_split"] != "test" else "train"
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_splits(p, rows, s, SPLIT_CONFIG)

    def test_features_ignore_all_post_response_columns(self):
        p = example_prompts()
        baseline = feature_table(p)
        p["raw_output"] = "Secret answer"
        p["success"] = 1
        p["response_latency"] = 123
        p["output_tokens"] = 999
        p["cost_usd"] = 12
        p["model_generated_metadata"] = "difficulty: easy"
        p["dataset"] = "arbitrary"
        pd.testing.assert_frame_equal(feature_table(p), baseline)
        self.assertEqual(list(baseline.columns), ["prompt_id"] + FEATURE_COLUMNS)

    def test_feature_counts(self):
        f = extract_prompt_features("A. 2 + 2?\nB. 5\n```python\nx=4\n```")
        self.assertEqual(f["answer_choice_count"], 2)
        self.assertEqual(f["code_fence_count"], 2)
        self.assertEqual(f["digit_count"], 4)
        self.assertEqual(f["contains_math"], 1)

    def test_best_single_validation_only_even_if_test_winner_differs(self):
        val = pd.DataFrame([dict(model_id="a", success=1, standard_split="validation"),
                            dict(model_id="b", success=0, standard_split="validation")])
        self.assertEqual(choose_best_single(val, "standard_split", {"a": 2, "b": 1}), "a")
        test = val.copy()
        test["success"] = 1 - test.success
        test["standard_split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation"):
            choose_best_single(test, "standard_split", {"a": 2, "b": 1})

    def test_frozen_model_pool_deterministic_and_unique(self):
        path = ROOT / "configs/router_model_pool.json"
        config = json.loads(path.read_text())
        before = file_hash(path)
        self.assertEqual(validate_pool(config), validate_pool(copy.deepcopy(config)))
        self.assertEqual(file_hash(path), before)
        config["models"].append(config["models"][0])
        with self.assertRaises(ValueError):
            validate_pool(config)

    def test_stale_pool_statistics_rejected_without_reselection(self):
        rows = normalize_outcomes(example_rows())
        prompts = pd.DataFrame({"prompt_id": ["p1", "p2"]})
        pool = dict(coverage_denominator=2, models=[dict(model_id="a", coverage=0.5)])
        before = copy.deepcopy(pool)
        with self.assertRaisesRegex(ValueError, "Stale"):
            verify_pool_statistics(pool, rows, prompts)
        self.assertEqual(pool, before)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.raw = cls.base / "raw"
        cls.config = cls.base / "config"
        cls.raw.mkdir(); cls.config.mkdir()
        tasks = dict(SPLIT_CONFIG, schema_version=1, tasks=[dict(dataset=d, source_split="test", task_type=d, label_rule="binary") for d in ["math", "code"]])
        pool = dict(models=[dict(model_id=f"m{i}", coverage=1.0, reason_selected="Fixture coverage") for i in range(5)])
        cls.archive = cls.raw / "bench-release.tar.gz"
        with tarfile.open(cls.archive, "w:gz") as tar:
            for dataset in ["math", "code"]:
                for model in range(5):
                    records = []
                    for idx in range(30):
                        records.append(dict(index=idx, prompt=f"{dataset} question {idx}", origin_query=f"{dataset} question {idx}",
                                            score=float((idx + model) % 3 != 0), cost=(model + 1) / 1000,
                                            prompt_tokens=10, completion_tokens=20, raw_output="answer", prediction="answer", ground_truth="reference"))
                    if dataset == "math" and model == 0:
                        records[0]["score"] = None
                    data = dict(dataset_name=dataset, split="test", model_name=f"m{model}", counts=len(records), records=records,
                                performance=0.5, cost=sum(r["cost"] for r in records), demo=False)
                    payload = json.dumps(data).encode()
                    member = tarfile.TarInfo(f"bench-release/{dataset}/test/m{model}/20250101_000000.json")
                    member.size = len(payload)
                    tar.addfile(member, io.BytesIO(payload))
        source = dict(dataset_revision="fixture-revision", repository_commit="fixture-commit",
                      archive=dict(filename=cls.archive.name, size_bytes=cls.archive.stat().st_size,
                                   sha256=file_hash(cls.archive), url="https://example.invalid/never-called"))
        for name, cfg in zip(CONFIG_NAMES, [source, pool, tasks]):
            write_json(cls.config / name, cfg)
        cls.output = cls.base / "processed"
        with patch("builtins.print"):
            cls.summary = prepare(cls.raw, cls.output, cls.base / "reports", config_dir=cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_processed_artifact_schema(self):
        outcomes = pd.read_parquet(self.output / "outcomes.parquet")
        prompts = pd.read_parquet(self.output / "prompts.parquet")
        features = pd.read_parquet(self.output / "prompt_features.parquet")
        self.assertEqual(len(prompts), 59)
        self.assertEqual(len(outcomes), 295)
        validate_processed(outcomes, prompts, features, [f"m{i}" for i in range(5)])
        self.assertTrue({"raw_score", "success", "success_label", "cost_usd", "input_tokens", "output_tokens", "standard_split", "ood_split"}.issubset(outcomes.columns))
        features["output_tokens"] = 1
        with self.assertRaisesRegex(ValueError, "allowlist"):
            validate_processed(outcomes, prompts, features, [f"m{i}" for i in range(5)])

    def test_missing_label_preserved_in_audit(self):
        raw = pd.read_parquet(self.output / "audit_selected_outcomes.parquet")
        nulls = raw[raw.raw_score.isna()]
        self.assertEqual(len(nulls), 1)
        self.assertTrue(nulls.success.isna().all())

    def test_reproducibility_metadata_and_all_hashes(self):
        p = json.loads((self.output / "provenance.json").read_text())
        self.assertEqual(p["source"]["archive"]["sha256"], file_hash(self.archive))
        self.assertEqual(p["split_seed"], 3407)
        self.assertTrue(p["no_training"] and p["no_provider_calls"])
        self.assertEqual(set(p["config_sha256"]), set(CONFIG_NAMES))
        for filename, info in p["artifacts"].items():
            self.assertEqual(file_hash(self.output / filename), info["sha256"])
            self.assertEqual(len(pd.read_parquet(self.output / filename)), info["rows"])

    def test_end_to_end_rerun_is_byte_deterministic(self):
        again = self.base / "again"
        with patch("builtins.print"):
            summary = prepare(self.raw, again, self.base / "reports_again", config_dir=self.config)
        self.assertEqual(summary, self.summary)
        for path in self.output.iterdir():
            self.assertEqual(file_hash(path), file_hash(again / path.name), path.name)

    def test_no_test_information_selects_baselines(self):
        outcomes = pd.read_parquet(self.output / "outcomes.parquet")
        _, _, _, before = baseline_artifacts(outcomes)
        outcomes.loc[outcomes.standard_split == "test", "success"] = 0
        outcomes.loc[(outcomes.standard_split == "test") & (outcomes.model_id == "m4"), "success"] = 1
        _, _, _, after = baseline_artifacts(outcomes)
        self.assertEqual(before["standard"]["best_single_model"], after["standard"]["best_single_model"])
        self.assertEqual(before["standard"]["always_cheapest_model"], after["standard"]["always_cheapest_model"])

    def test_oracle_unsolved_prompt_fallback_and_null_success_cost(self):
        outcomes = pd.read_parquet(self.output / "outcomes.parquet")
        pid = outcomes[outcomes.standard_split == "test"].prompt_id.iloc[0]
        outcomes.loc[outcomes.prompt_id == pid, "success"] = 0
        _, _, oracle, meta = baseline_artifacts(outcomes)
        row = oracle[(oracle.regime == "standard") & (oracle.prompt_id == pid)].iloc[0]
        self.assertEqual(row.any_success, 0)
        self.assertIsNone(row.cheapest_success_model_id)
        self.assertTrue(pd.isna(row.cheapest_success_cost_usd))
        self.assertGreater(row.oracle_policy_cost_usd, 0)

    def test_training_loader_alignment_and_feature_separation(self):
        prompts, x, y, cost = load_split(self.output)
        self.assertEqual(list(x.columns), FEATURE_COLUMNS)
        self.assertTrue(x.index.equals(y.index) and y.index.equals(cost.index))
        self.assertEqual(list(y.columns), [f"m{i}" for i in range(5)])
        self.assertEqual(set(prompts.standard_split), {"train"})

    def test_download_existing_archive_never_contacts_network(self):
        config = json.loads((self.config / CONFIG_NAMES[0]).read_text())
        with patch("routing_data.source.subprocess.run", side_effect=AssertionError("network")):
            self.assertEqual(download_archive(self.raw, config, download=True), self.archive)

    def test_bad_source_hash_and_missing_archive_fail_closed(self):
        source = json.loads((self.config / CONFIG_NAMES[0]).read_text())
        bad = dict(source["archive"], sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            verify_file(self.archive, bad)
        with self.assertRaises(FileNotFoundError):
            download_archive(self.base / "absent", source)

    def test_archive_path_identity_mismatch_rejected(self):
        bad = self.base / "bad.tar.gz"
        with tarfile.open(bad, "w:gz") as tar:
            payload = json.dumps(dict(dataset_name="math", split="test", model_name="different", records=[])).encode()
            member = tarfile.TarInfo("bench-release/math/test/m0/file.json")
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))
        with self.assertRaisesRegex(ValueError, "identifiers"):
            read_archive(bad, dict(tasks=[]), [])

    def test_duplicate_archive_members_rejected(self):
        bad = self.base / "duplicate.tar.gz"
        with tarfile.open(bad, "w:gz") as tar:
            payload = json.dumps(dict(dataset_name="math", split="test", model_name="m0", records=[], counts=0)).encode()
            for _ in range(2):
                member = tarfile.TarInfo("bench-release/math/test/m0/file.json")
                member.size = len(payload)
                tar.addfile(member, io.BytesIO(payload))
        with self.assertRaisesRegex(ValueError, "Duplicate archive"):
            read_archive(bad, dict(tasks=[]), [])


if __name__ == "__main__":
    unittest.main()
