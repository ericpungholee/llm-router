"""Leakage, decision semantics, uncertainty, and offline experiment integration."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from routing_data.features import FEATURE_COLUMNS, feature_table
from routing_ml.bootstrap import group_strata, paired_bootstrap, sample_groups
from routing_ml.metrics import calibration, nondominated, oracle_actions
from routing_ml.policies import Policy, best_single, estimate_costs, grid, route, select_validation, validate_predictions
from routing_ml.training import MODEL_IDS, Split, fit_predictors, inputs, train_router

ROOT = Path(__file__).resolve().parents[1]


def synthetic(name="train", regime="standard", n=60):
    prompts = pd.DataFrame([dict(prompt_id=f"{regime}_{name}_{i:03d}",
                                 prompt=f"{'algebra equation' if i % 2 else 'history question'} common word identifier{i} {name}token",
                                 dataset="math" if i < n // 2 else "knowledge",
                                 leakage_group=f"{regime}_{name}_group{i // 2}",
                                 standard_split=name, ood_split=name) for i in range(n)]).set_index("prompt_id")
    features = feature_table(prompts.reset_index()).set_index("prompt_id")
    y = pd.DataFrame([[int((i + j) % 3 != 0) for j in range(8)] for i in range(n)], index=prompts.index, columns=MODEL_IDS)
    costs = pd.DataFrame(np.tile(np.arange(1, 9) / 1000, (n, 1)), index=prompts.index, columns=MODEL_IDS)
    return Split(regime, name, prompts, features, y, costs)


class TrainingTests(unittest.TestCase):
    def test_train_only_vocab_and_every_cv_fold_refits(self):
        train = synthetic()
        calls = []
        original = TfidfVectorizer.fit_transform
        def observe(vectorizer, documents, *args, **kwargs):
            calls.append((vectorizer, list(documents)))
            return original(vectorizer, documents, *args, **kwargs)
        with patch.object(TfidfVectorizer, "fit_transform", observe), patch("routing_ml.training.load_split", side_effect=AssertionError("CV cannot load any split")):
            model, cv, assignments = train_router(train)
        self.assertEqual(len(calls), 6)
        self.assertEqual(len({id(p) for p, _ in calls}), 6)
        self.assertEqual(len(cv), 5 * 3 * 8)
        for fold, (prep, documents) in enumerate(calls[:5]):
            heldout = assignments.loc[assignments.fold == fold, "prompt_id"]
            expected = train.prompts.loc[~train.prompts.index.isin(heldout), "prompt"].tolist()
            self.assertEqual(documents, expected)
            self.assertFalse(set(documents) & set(train.prompts.loc[heldout, "prompt"]))
        self.assertIn("traintoken", model.preprocessor.vocabulary_)
        model.predict(synthetic("validation"))
        model.predict(synthetic("test"))
        self.assertNotIn("validationtoken", model.preprocessor.vocabulary_)
        self.assertNotIn("testtoken", model.preprocessor.vocabulary_)

    def test_cv_rejects_validation_and_test(self):
        for name in ("validation", "test"):
            with self.assertRaisesRegex(ValueError, "train only"):
                train_router(synthetic(name))

    def test_only_prompt_and_allowlisted_features_enter_x(self):
        a = synthetic()
        text_before, numeric_before = inputs(a, "tfidf"), inputs(a, "handcrafted")
        for column in ("raw_output", "success", "cost_usd", "latency", "output_tokens", "ground_truth", "actual_model"):
            a.prompts[column] = "forbidden"
            a.features[column] = -123456
        a.prompts["dataset"] = "forbidden"
        self.assertEqual(text_before, inputs(a, "tfidf"))
        np.testing.assert_array_equal(numeric_before, inputs(a, "handcrafted"))
        self.assertEqual(inputs(a, "handcrafted").shape[1], 16)

    def test_handcrafted_scaling_is_train_only(self):
        train = synthetic()
        model, _, _ = train_router(train, "handcrafted")
        means = train.features[FEATURE_COLUMNS].mean().to_numpy()
        np.testing.assert_allclose(model.preprocessor.mean_, means)
        val = synthetic("validation")
        val.features[FEATURE_COLUMNS] *= 100000
        model.predict(val)
        np.testing.assert_array_equal(model.preprocessor.mean_, means)

    def test_candidate_order_and_prompt_alignment_rejected(self):
        a = synthetic()
        with self.assertRaisesRegex(ValueError, "order"):
            Split(a.regime, a.name, a.prompts, a.features, a.y.iloc[:, ::-1], a.costs)
        with self.assertRaisesRegex(ValueError, "alignment"):
            Split(a.regime, a.name, a.prompts, a.features.iloc[::-1], a.y, a.costs)
        self.assertEqual(tuple(json.loads((ROOT / "configs/router_model_pool.json").read_text())["models"][i]["model_id"] for i in range(8)), MODEL_IDS)

    def test_single_class_predictor_fallback(self):
        x = np.ones((10, 2))
        y = np.zeros((10, 8))
        y[:, 3] = 1
        models, fallback = fit_predictors(x, y, 1)
        self.assertEqual(len(fallback), 8)
        np.testing.assert_array_equal(models[3].predict_proba(x)[:, 1], 1)
        np.testing.assert_array_equal(models[0].predict_proba(x)[:, 1], 0)

    def test_seed_3407_reproduces_folds_c_and_probabilities(self):
        a = synthetic()
        m1, cv1, folds1 = train_router(a)
        m2, cv2, folds2 = train_router(a)
        pd.testing.assert_frame_equal(cv1, cv2, check_exact=True)
        pd.testing.assert_frame_equal(folds1, folds2, check_exact=True)
        pd.testing.assert_frame_equal(m1.predict(a), m2.predict(a), check_exact=True)
        self.assertEqual(m1.metadata, m2.metadata)
        self.assertEqual(m1.metadata["seed"], 3407)

    def test_independent_regimes_and_zero_code_in_ood_training(self):
        a, b = synthetic(), synthetic(regime="ood")
        b.prompts["prompt"] = b.prompts.prompt + " oodexclusive"
        ma, _, _ = train_router(a)
        mb, _, _ = train_router(b)
        self.assertIsNot(ma.preprocessor, mb.preprocessor)
        self.assertNotIn("oodexclusive", ma.preprocessor.vocabulary_)
        self.assertIn("oodexclusive", mb.preprocessor.vocabulary_)
        for x, y in zip(ma.predictors, mb.predictors):
            self.assertIsNot(x, y)
        with self.assertRaisesRegex(ValueError, "across"):
            ma.predict(b)
        for split in ("train", "validation"):
            b = synthetic(split, "ood")
            b.prompts.iloc[0, b.prompts.columns.get_loc("dataset")] = "livecodebench"
            with self.assertRaisesRegex(ValueError, "LiveCodeBench"):
                b.__post_init__()

    def test_prediction_ids_columns_and_order(self):
        a = synthetic()
        model, _, _ = train_router(a)
        pred = model.predict(a)
        validate_predictions(a, pred)
        self.assertTrue(pred.index.equals(a.prompts.index))
        self.assertEqual(tuple(pred.columns), MODEL_IDS)
        self.assertFalse(set(pred.columns) & {"success", "label", "cost_usd", "dataset"})
        with self.assertRaisesRegex(ValueError, "align"):
            validate_predictions(a, pred.iloc[::-1])


class PolicyTests(unittest.TestCase):
    def test_frozen_grid(self):
        self.assertEqual(len(grid()), 29)
        self.assertEqual([p.value for p in grid()[:21]], [i / 20 for i in range(21)])

    def test_threshold_qualifying_cost_probability_and_id_ties(self):
        ids = ("z", "a", "b")
        p = [[0.8, 0.9, 0.95], [0.9, 0.9, 1.0], [0.4, 0.4, 0.7]]
        result = route(p, Policy("test", "threshold", 0.5), [1, 1, 2], ids)
        np.testing.assert_array_equal(result, [1, 1, 2])

    def test_threshold_fallback_probability_cost_and_id_ties(self):
        p = [[0.3, 0.4, 0.2], [0.4, 0.4, 0.4], [0.1, 0.1, 0.2]]
        np.testing.assert_array_equal(route(p, Policy("t", "threshold", 1), [2, 2, 3], ("z", "a", "b")), [1, 1, 2])
        np.testing.assert_array_equal(route([[0.2, 0.2]], Policy("t", "threshold", 1), [1, 2], ("z", "a")), [0])

    def test_utility_uses_normalized_training_cost_and_ties(self):
        p = [[0.6, 0.9]]
        self.assertEqual(route(p, Policy("u", "utility", 0), [1, 10], ("a", "b"))[0], 1)
        self.assertEqual(route(p, Policy("u", "utility", 1), [1, 10], ("a", "b"))[0], 0)
        self.assertEqual(route([[0.5, 1.0]], Policy("u", "utility", 1), [1, 2], ("z", "a"))[0], 0)
        self.assertEqual(route([[0.5, 0.5]], Policy("u", "utility", 0), [1, 1], ("z", "a"))[0], 1)

    def test_cost_estimation_accepts_training_only(self):
        a = synthetic()
        costs, normalized = estimate_costs(a)
        np.testing.assert_allclose(costs, np.arange(1, 9) / 1000)
        np.testing.assert_allclose(normalized, np.arange(1, 9) / 8)
        for split in ("validation", "test"):
            b = synthetic(split)
            b.costs[:] = 999
            with self.assertRaisesRegex(ValueError, "train only"):
                estimate_costs(b)

    def test_best_single_is_validation_selected_with_cost_id_ties(self):
        val = synthetic("validation")
        val.y[:] = 0
        val.y[MODEL_IDS[5]] = 1
        val.y[MODEL_IDS[6]] = 1
        costs = np.arange(1, 9, dtype=float)
        self.assertEqual(best_single(val, costs).model_id, MODEL_IDS[5])
        costs[5] = costs[6]
        self.assertEqual(best_single(val, costs).model_id, "gpt-5")
        with self.assertRaisesRegex(ValueError, "validation only"):
            best_single(synthetic("test"), costs)

    def test_validation_selection_never_accepts_test_and_fallback_feasible(self):
        val = synthetic("validation")
        val.y[:] = 0
        val.y["gpt-5"] = 1
        p = pd.DataFrame(0.1, index=val.prompts.index, columns=MODEL_IDS)
        p[MODEL_IDS[0]] = 0.9
        costs = np.arange(1, 9)
        chosen, best, table, _ = select_validation(val, p, costs)
        self.assertEqual(chosen.policy_id, "best_single")
        self.assertEqual(best.model_id, "gpt-5")
        self.assertTrue(table[table.policy_id == "best_single"].feasible.iloc[0])
        test = synthetic("test")
        with self.assertRaisesRegex(ValueError, "validation only"):
            select_validation(test, p, costs)

    def test_validation_selection_minimizes_realized_cost(self):
        val = synthetic("validation")
        val.y[:] = 1
        val.costs.iloc[:, 0] = 0.0001
        p = pd.DataFrame(0.9, index=val.prompts.index, columns=MODEL_IDS)
        chosen, _, table, _ = select_validation(val, p, np.arange(1, 9))
        self.assertAlmostEqual(table[table.policy_id == chosen.policy_id].mean_cost_usd.iloc[0], 0.0001)


class EvaluationTests(unittest.TestCase):
    def test_bootstrap_samples_whole_groups_within_strata(self):
        prompts = pd.DataFrame(dict(dataset=["a", "a", "a", "b", "b", "b"], leakage_group=["g1", "g1", "g2", "g3", "g4", "g4"]))
        rng, strata = np.random.default_rng(3407), group_strata(prompts)
        for _ in range(30):
            idx = sample_groups(strata, rng)
            counts = np.bincount(idx, minlength=6)
            self.assertEqual(counts[0], counts[1])
            self.assertEqual(counts[4], counts[5])
            self.assertEqual(counts[0] + counts[2], 2)
            self.assertEqual(counts[3] + counts[4], 2)
        prompts.loc[3, "leakage_group"] = "g1"
        with self.assertRaisesRegex(ValueError, "crosses"):
            group_strata(prompts)

    def test_paired_bootstrap_known_delta_and_savings_and_seed(self):
        prompts = synthetic().prompts
        n = len(prompts)
        samples, summary = paired_bootstrap(prompts, np.ones(n), np.ones(n), np.ones(n), np.full(n, 2), replicates=50)
        self.assertEqual(summary["ci_95"]["quality_delta"], [0, 0])
        self.assertEqual(summary["ci_95"]["cost_savings"], [0.5, 0.5])
        self.assertTrue(summary["noninferiority_supported"])
        other, _ = paired_bootstrap(prompts, np.ones(n), np.ones(n), np.ones(n), np.full(n, 2), replicates=50)
        pd.testing.assert_frame_equal(samples, other, check_exact=True)

    def test_calibration_bins_include_one_and_loss_is_clipped(self):
        metrics, rel = calibration([0, 1, 1, 0], [0, 0.1, 1, 1])
        self.assertEqual(rel["count"].sum(), 4)
        self.assertEqual(rel[rel.bin == 9]["count"].iloc[0], 2)
        self.assertAlmostEqual(metrics["brier"], (0.81 + 1) / 4)
        self.assertAlmostEqual(metrics["ece"], (0.9 + 1) / 4)
        self.assertTrue(np.isfinite(metrics["log_loss"]))

    def test_pareto_retains_equal_points_and_drops_dominated(self):
        table = pd.DataFrame(dict(mean_cost_usd=[1, 1, 2, 3, 4], quality=[0.5, 0.5, 0.4, 0.8, 0.8]))
        np.testing.assert_array_equal(nondominated(table), [True, True, False, True, False])

    def test_oracle_uses_realized_success_cost_and_paid_fallback(self):
        part = synthetic(n=6)
        part.y[:] = 0
        part.y.iloc[0, 2:4] = 1
        part.costs.iloc[0, 3] = 0.00001
        actions = oracle_actions(part, np.arange(1, 9))
        self.assertEqual(actions[0], 3)
        np.testing.assert_array_equal(actions[1:], 0)

    def test_offline_run_freezes_policy_before_loading_test(self):
        spec = importlib.util.spec_from_file_location("router_v1_experiment", ROOT / "experiments/tfidf_logreg_router.py")
        experiment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(experiment)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "standard"
            def read(directory, regime, split):
                if split == "test":
                    self.assertTrue((output / "frozen_policy.json").exists())
                    self.assertTrue((output / "models.joblib").exists())
                return synthetic(split, regime)
            with patch.object(experiment, "checked_read", side_effect=read), patch.object(experiment, "verify_source", return_value={}), patch.object(experiment, "figures"):
                result = experiment.run_one(Path(tmp), output, Path(tmp), "standard", "tfidf")
            self.assertTrue(result["determinism_full_training_replay_verified"])
            self.assertEqual(len(pd.read_parquet(output / "bootstrap_samples.parquet")), 2000)
            for split in ("train", "validation", "test"):
                pred = pd.read_parquet(output / f"{split}_predictions.parquet")
                self.assertEqual(pred.columns.tolist(), ["prompt_id"] + list(MODEL_IDS))
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertTrue(metadata["frozen_policy"]["frozen_before_test_read"])


if __name__ == "__main__":
    unittest.main()
