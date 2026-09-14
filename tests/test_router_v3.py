"""Leakage, frozen-rule and real-artifact checks for the cross-fitting audit."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from experiments.crossfit_tfidf_router import run_fold, start_campaign
from experiments.tfidf_logreg_router import model_signature, sha256
from routing_ml.crossfit import (CrossfitPart, fit_fixed_router, fold_ids, freeze_references,
                                 frozen_actions, partition_assignments, read_part, validate_assignments)
from routing_ml.training import MODEL_IDS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/llmrouterbench"
ARTIFACTS = ROOT / "artifacts/router_v3"
AVAILABLE = (ARTIFACTS / "results.json").exists()


def synthetic_prompts():
    return pd.DataFrame(dict(prompt_id=[f"id{i:03d}" for i in range(160)],
                             prompt=[f"{'math algebra' if i % 2 else 'history knowledge'} prompt number {i}" for i in range(160)],
                             dataset=["simpleqa"] * 80 + ["livecodebench"] * 80,
                             leakage_group=[f"group{i//2:03d}" for i in range(160)])).set_index("prompt_id")


def part(prompts, name, regime="standard"):
    y = np.array([[int((i + j) % 3 != 0) for j in range(8)] for i in range(len(prompts))])
    return CrossfitPart(regime, name, prompts.copy(), pd.DataFrame(y, index=prompts.index, columns=MODEL_IDS),
                        pd.DataFrame(np.tile(np.arange(1, 9) / 100, (len(prompts), 1)), index=prompts.index, columns=MODEL_IDS))


class CrossfitUnitTests(unittest.TestCase):
    def test_group_partitions_exact_coverage_and_determinism_without_outcomes(self):
        p = synthetic_prompts()
        a = partition_assignments(p)
        pd.testing.assert_frame_equal(a, partition_assignments(p.sample(frac=1, random_state=77)), check_exact=True)
        pd.testing.assert_frame_equal(a, partition_assignments(p.assign(success="poison", response="do not use", cost_usd=-1)), check_exact=True)
        self.assertTrue(validate_assignments(a, p))
        self.assertEqual(a[a.role == "test"].prompt_id.nunique(), len(p))
        self.assertEqual(a.groupby(["fold", "leakage_group"]).role.nunique().max(), 1)
        broken = a.copy()
        first = broken.index[0]
        broken.loc[first, "role"] = "test" if broken.loc[first, "role"] != "test" else "train"
        with self.assertRaises(ValueError):
            validate_assignments(broken, p)

    def test_ood_excludes_code_and_paired_evaluation_covers_code_once(self):
        p = synthetic_prompts()
        a = partition_assignments(p)
        held_out = []
        for k in range(5):
            for role in ("train", "validation"):
                ids = fold_ids(a, k, "ood", role)
                self.assertFalse(p.loc[ids].dataset.eq("livecodebench").any())
                self.assertTrue(set(ids) <= set(fold_ids(a, k, "standard", role)))
            ids = fold_ids(a, k, "ood", "test")
            self.assertTrue(p.loc[ids].dataset.eq("livecodebench").all())
            self.assertTrue(set(ids) <= set(fold_ids(a, k, "standard", "test")))
            held_out.extend(ids)
        self.assertEqual(len(held_out), len(set(held_out)))
        self.assertEqual(set(held_out), set(p[p.dataset == "livecodebench"].index))

    def test_fixed_fit_train_only_vocabulary_no_cv_response_features_or_state_reuse(self):
        p = synthetic_prompts().iloc[:60].copy()
        train = part(p.iloc[:40], "train")
        train.prompts["response"] = "forbiddenresponse"
        with patch("routing_ml.training.train_router", side_effect=AssertionError("No new C search")):
            model = fit_fixed_router(train)
            replay = fit_fixed_router(train)
        self.assertEqual(model.selected_c, 1)
        self.assertNotIn("forbiddenresponse", model.preprocessor.vocabulary_)
        request = SimpleNamespace(regime="standard", prompts=p.iloc[40:].assign(prompt="unseenvalidationtoken"))
        prediction = model.predict(request)
        self.assertNotIn("unseenvalidationtoken", model.preprocessor.vocabulary_)
        self.assertEqual(model_signature(model), model_signature(replay))
        self.assertIsNot(model.preprocessor, replay.preprocessor)
        pd.testing.assert_index_equal(prediction.index, request.prompts.index)
        with self.assertRaises(ValueError):
            fit_fixed_router(part(p, "validation"))
        with self.assertRaises(ValueError):
            model.predict(SimpleNamespace(regime="ood", prompts=p))

    def test_baselines_use_validation_and_costs_only_training(self):
        p = synthetic_prompts()
        train, val = part(p.iloc[:40], "train"), part(p.iloc[40:60], "validation")
        val.y.iloc[:, :] = 0
        val.y[MODEL_IDS[6]] = 1
        train.y.iloc[:, :] = 1
        val.costs.iloc[:, :] = 999
        frozen = freeze_references(train, val)
        self.assertEqual(frozen["best_single"], MODEL_IDS[6])
        np.testing.assert_allclose(list(frozen["mean_training_cost_usd"].values()), train.costs.mean())
        self.assertEqual(frozen["domain_static_models"]["simpleqa"], MODEL_IDS[6])
        with self.assertRaises(ValueError):
            freeze_references(train, part(p.iloc[40:60], "test"))
        with self.assertRaises(ValueError):
            freeze_references(train, part(p.iloc[:20], "validation"))

    def test_frozen_actions_margin_and_domain_metadata_never_affect_primary(self):
        p = synthetic_prompts()
        train, val = part(p.iloc[:40], "train"), part(p.iloc[40:60], "validation")
        val.y.iloc[:, :] = 0
        val.y[MODEL_IDS[6]] = 1
        frozen = freeze_references(train, val)
        pred = pd.DataFrame(np.full((2, 8), .1), columns=MODEL_IDS)
        pred[MODEL_IDS[6]] = .7
        pred[MODEL_IDS[0]] = [.8, .71]
        actions = frozen_actions(pred, ["simpleqa", "unseen_domain"], frozen)
        np.testing.assert_array_equal(actions["router"], [0, 6])
        np.testing.assert_array_equal(actions["privileged_domain_static"], [6, 6])
        np.testing.assert_array_equal(actions["router"], frozen_actions(pred, ["poison", "poison"], frozen)["router"])
        with self.assertRaises(ValueError):
            frozen_actions(pred.loc[:, list(reversed(MODEL_IDS))], ["a", "a"], frozen)

    def test_runner_saves_model_policy_predictions_actions_before_evaluation_read(self):
        prompts = synthetic_prompts()
        assignments = partition_assignments(prompts)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            events = []
            def observed_read(directory, all_prompts, ids, regime, name):
                if name == "test":
                    path = output / regime / "fold_0"
                    for filename in ("models.joblib", "frozen_policy.json", "test_predictions.parquet", "test_decisions.parquet"):
                        self.assertTrue((path / filename).exists(), filename)
                events.append(name)
                return part(all_prompts.loc[ids], name, regime)
            with patch("experiments.crossfit_tfidf_router.read_part", side_effect=observed_read):
                run_fold(output, output, prompts, assignments, "standard", 0)
            self.assertEqual(events, ["train", "validation", "test"])
            with self.assertRaisesRegex(ValueError, "not empty"):
                start_campaign(output, output)

    def test_candidate_order_and_ood_part_are_enforced(self):
        p = synthetic_prompts()
        with self.assertRaises(ValueError):
            part(p, "train", "ood")
        a = part(p, "train")
        with self.assertRaises(ValueError):
            CrossfitPart(a.regime, a.name, a.prompts, a.y.loc[:, list(reversed(MODEL_IDS))], a.costs)


@unittest.skipUnless(AVAILABLE, "Run the v3 offline cross-fitting experiment for real artifacts")
class CrossfitArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ARTIFACTS / "results.json").read_text())
        cls.prompts = pd.read_parquet(DATA / "prompts.parquet").set_index("prompt_id").sort_index()
        cls.assignments = pd.read_parquet(ARTIFACTS / "assignments.parquet")

    def test_all_artifact_hashes_and_original_data_reports_match(self):
        manifest = json.loads((ARTIFACTS / "artifact_hashes.json").read_text())
        for name, expected in manifest.items():
            self.assertEqual(sha256(ARTIFACTS / name), expected, name)
        lock = self.report["provenance"]
        for name, expected in lock["source_processed_data_hashes"].items():
            self.assertEqual(sha256(DATA / name), expected, name)
        for name, expected in lock["previous_report_hashes"].items():
            self.assertEqual(sha256(ROOT / name), expected, name)
        for name, expected in lock["source_hashes"].items():
            self.assertEqual(sha256(ARTIFACTS / "source_snapshot" / name), expected, name)

    def test_real_partitions_coverage_training_costs_baselines_and_independent_models(self):
        pd.testing.assert_frame_equal(self.assignments, partition_assignments(self.prompts), check_exact=True)
        hashes = []
        for regime in ("standard", "ood"):
            for k in range(5):
                path = ARTIFACTS / regime / f"fold_{k}"
                frozen = json.loads((path / "frozen_policy.json").read_text())
                train = read_part(DATA, self.prompts, fold_ids(self.assignments, k, regime, "train"), regime, "train")
                val = read_part(DATA, self.prompts, fold_ids(self.assignments, k, regime, "validation"), regime, "validation")
                reference = freeze_references(train, val)
                for name, value in reference.items():
                    self.assertEqual(frozen[name], value)
                model = joblib.load(path / "models.joblib")
                self.assertEqual(model_signature(model), frozen["model_signature"])
                hashes.append(model.metadata["vocabulary_hash"])
                self.assertEqual(model.selected_c, 1)
                self.assertEqual(model.metadata["feature_columns"], ["prompt"])
                if k == 0:
                    self.assertTrue(frozen["training_replay_exact"])
        self.assertEqual(len(set(hashes)), 10)

    def test_every_saved_probability_and_action_matches_reload_and_scalar_rule(self):
        for regime in ("standard", "ood"):
            for k in range(5):
                path = ARTIFACTS / regime / f"fold_{k}"
                frozen = json.loads((path / "frozen_policy.json").read_text())
                model = joblib.load(path / "models.joblib")
                for role in ("train", "validation", "test"):
                    prediction = pd.read_parquet(path / f"{role}_predictions.parquet").set_index("prompt_id")
                    ids = fold_ids(self.assignments, k, regime, role)
                    request = SimpleNamespace(regime=regime, prompts=self.prompts.loc[ids])
                    pd.testing.assert_frame_equal(prediction, model.predict(request), check_exact=True)
                decisions = pd.read_parquet(path / "test_decisions.parquet").set_index("prompt_id")
                costs = frozen["mean_training_cost_usd"]
                best = frozen["best_single"]
                expected = []
                for row in prediction.to_dict(orient="records"):
                    eligible = [m for m in MODEL_IDS if costs[m] < costs[best] and row[m] >= row[best] + frozen["margin"]]
                    expected.append(min(eligible, key=lambda m: (costs[m], -row[m], m)) if eligible else best)
                self.assertEqual(decisions.router.tolist(), expected)
                self.assertFalse({"success", "cost_usd", "response", "dataset"} & set(decisions.columns))

    def test_evaluation_aligns_with_source_outcomes_and_summary_and_intervals(self):
        for regime in ("standard", "ood"):
            path = ARTIFACTS / regime
            pred = pd.read_parquet(path / "oof_predictions.parquet").set_index("prompt_id")
            self.assertFalse(pred.index.has_duplicates)
            self.assertEqual(tuple(pred.columns), MODEL_IDS)
            ids = self.prompts.index if regime == "standard" else self.prompts[self.prompts.dataset == "livecodebench"].index
            pd.testing.assert_index_equal(pred.index, ids)
            data = read_part(DATA, self.prompts, ids, regime, "test")
            e = pd.read_parquet(path / "oof_evaluation.parquet")
            self.assertFalse(e.duplicated(["prompt_id", "policy_id"]).any())
            rows = data.y.index.get_indexer(e.prompt_id)
            cols = [MODEL_IDS.index(m) for m in e.selected_model_id]
            np.testing.assert_array_equal(e.success, data.y.to_numpy()[rows, cols])
            np.testing.assert_array_equal(e.cost_usd, data.costs.to_numpy()[rows, cols])
            r = self.report["regimes"][regime]
            for pid, metrics in r["micro"].items():
                values = e[e.policy_id == pid]
                self.assertAlmostEqual(values.success.mean(), metrics["quality"])
                self.assertAlmostEqual(values.cost_usd.mean(), metrics["mean_cost_usd"])
                self.assertAlmostEqual(values.groupby("dataset").success.mean().mean(), r["macro"][pid]["quality"])
            samples = pd.read_parquet(path / "primary_bootstrap_samples.parquet")
            self.assertEqual(len(samples), 2000)
            for key, ci in r["primary"]["bootstrap"]["ci_95"].items():
                np.testing.assert_array_equal(np.quantile(samples[key], [.025, .975]), ci)
            self.assertFalse(r["primary"]["bootstrap"]["independent_confirmation"])
        self.assertEqual(self.report["matched_code"]["n"], 1055)


if __name__ == "__main__":
    unittest.main()
