"""Independent checks of actual saved experiment outputs, when available locally."""

import hashlib
import json
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd

from routing_data.loading import load_split
from routing_ml.training import MODEL_IDS, read_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/llmrouterbench"
ARTIFACTS = ROOT / "artifacts/router_v1"
RUNS = [(r, k, ARTIFACTS / r / ("handcrafted" if k == "handcrafted" else ""))
        for r in ("standard", "ood") for k in ("tfidf", "handcrafted")]
AVAILABLE = all((p / "results.json").exists() for _, _, p in RUNS) and DATA.exists()


@unittest.skipUnless(AVAILABLE, "Run the four offline router experiments to verify real artifacts")
class SavedExperimentTests(unittest.TestCase):
    def test_all_saved_artifact_hashes_match(self):
        for _, _, path in RUNS:
            hashes = json.loads((path / "artifact_hashes.json").read_text())
            for name, expected in hashes.items():
                self.assertEqual(hashlib.sha256((path / name).read_bytes()).hexdigest(), expected)

    def test_predictions_match_frozen_prompt_ids_and_reloaded_models(self):
        for regime, _, path in RUNS:
            model = joblib.load(path / "models.joblib")
            for split in ("train", "validation", "test"):
                pred = pd.read_parquet(path / f"{split}_predictions.parquet").set_index("prompt_id")
                part = read_split(DATA, regime, split)
                pd.testing.assert_frame_equal(pred, model.predict(part), check_exact=True)
                self.assertEqual(tuple(pred.columns), MODEL_IDS)

    def test_independent_scalar_routing_matches_every_frozen_grid_decision(self):
        for _, _, path in RUNS:
            frozen = json.loads((path / "frozen_policy.json").read_text())
            costs = frozen["mean_training_cost_usd"]
            max_cost = max(costs.values())
            table = pd.read_parquet(path / "validation_policy_grid.parquet")
            probabilities = pd.read_parquet(path / "test_predictions.parquet").set_index("prompt_id")
            decisions = pd.read_parquet(path / "test_routing_decisions.parquet")
            for policy in table.itertuples():
                actual = decisions[decisions.policy_id == policy.policy_id].set_index("prompt_id").loc[probabilities.index, "selected_model_id"]
                expected = []
                for row in probabilities.itertuples(index=False, name=None):
                    p = dict(zip(MODEL_IDS, row))
                    if policy.kind == "static":
                        choice = policy.model_id
                    elif policy.kind == "utility":
                        choice = min(MODEL_IDS, key=lambda m: (-(p[m] - policy.value * costs[m] / max_cost), costs[m], m))
                    else:
                        eligible = [m for m in MODEL_IDS if p[m] >= policy.value]
                        choice = min(eligible, key=lambda m: (costs[m], -p[m], m)) if eligible else min(MODEL_IDS, key=lambda m: (-p[m], costs[m], m))
                    expected.append(choice)
                self.assertEqual(actual.tolist(), expected)

    def test_saved_selection_is_validation_feasible_minimum_cost(self):
        for regime, _, path in RUNS:
            _, _, val_y, _ = load_split(DATA, regime, "validation")
            _, _, _, train_c = load_split(DATA, regime, "train")
            frozen = json.loads((path / "frozen_policy.json").read_text())
            mean_c = train_c.mean()
            best = min(MODEL_IDS, key=lambda m: (-val_y[m].mean(), mean_c[m], m))
            self.assertEqual(best, frozen["best_single"]["model_id"])
            np.testing.assert_allclose([frozen["mean_training_cost_usd"][m] for m in MODEL_IDS], mean_c)
            table = pd.read_parquet(path / "validation_policy_grid.parquet")
            feasible = table[table.quality >= val_y[best].mean() - 0.005]
            winner = min(feasible.itertuples(), key=lambda r: (r.mean_cost_usd, -r.quality, r.policy_id))
            self.assertEqual(winner.policy_id, frozen["selected_policy"]["policy_id"])

    def test_source_outcomes_reproduce_all_saved_selected_labels_and_costs(self):
        for regime, _, path in RUNS:
            for split in ("validation", "test"):
                _, _, y, costs = load_split(DATA, regime, split)
                evaluation = pd.read_parquet(path / f"{split}_evaluation.parquet")
                rows = y.index.get_indexer(evaluation.prompt_id)
                columns = np.array([MODEL_IDS.index(m) for m in evaluation.selected_model_id])
                np.testing.assert_array_equal(evaluation.success, y.to_numpy(dtype=float)[rows, columns])
                np.testing.assert_array_equal(evaluation.cost_usd, costs.to_numpy(dtype=float)[rows, columns])
                self.assertFalse({"success", "cost_usd"} & set(pd.read_parquet(path / f"{split}_routing_decisions.parquet").columns))

    def test_baselines_match_frozen_preprocessing_references(self):
        reference = pd.read_parquet(DATA / "baseline_choices.parquet")
        for regime, _, path in RUNS:
            evaluation = pd.read_parquet(path / "test_evaluation.parquet")
            for ours, original in [("best_single", "best_single"), ("always_cheapest", "always_cheapest"), ("oracle", "oracle_cheapest_success_with_fallback")]:
                a = evaluation[evaluation.policy_id == ours].sort_values("prompt_id")
                b = reference[(reference.regime == regime) & (reference.policy == original)].sort_values("prompt_id")
                self.assertEqual(a.prompt_id.tolist(), b.prompt_id.tolist())
                self.assertEqual(a.selected_model_id.tolist(), b.model_id.tolist())
                np.testing.assert_array_equal(a.success, b.success)
                np.testing.assert_array_equal(a.cost_usd, b.cost_usd)

    def test_bootstrap_intervals_and_micro_macro_numbers_recompute(self):
        for regime, _, path in RUNS:
            result = json.loads((path / "results.json").read_text())
            evaluation = pd.read_parquet(path / "test_evaluation.parquet")
            for pid in ("deployable_router", "best_single", "always_cheapest", "oracle"):
                e = evaluation[evaluation.policy_id == pid]
                self.assertAlmostEqual(e.success.mean(), result["test"][pid]["quality"])
                self.assertAlmostEqual(e.groupby("dataset").success.mean().mean(), result["macro"][pid]["quality"])
            samples = pd.read_parquet(path / "bootstrap_samples.parquet")
            self.assertEqual(len(samples), 2000)
            for name, ci in result["bootstrap"]["ci_95"].items():
                np.testing.assert_allclose(np.quantile(samples[name], [0.025, 0.975]), ci, atol=1e-15)
            # Independent resampling implementation for this frozen singleton-group dataset.
            a = evaluation[evaluation.policy_id == "deployable_router"].reset_index(drop=True)
            b = evaluation[evaluation.policy_id == "best_single"].reset_index(drop=True)
            self.assertEqual(a.leakage_group.nunique(), len(a))
            strata = [a[a.dataset == d].sort_values("leakage_group").index.to_numpy() for d in sorted(a.dataset.unique())]
            rng = np.random.default_rng(3407)
            first = []
            for _ in range(25):
                idx = np.concatenate([indices[rng.integers(len(indices), size=len(indices))] for indices in strata])
                first.append([float((a.success.to_numpy()[idx] - b.success.to_numpy()[idx]).mean()),
                              float(1 - a.cost_usd.to_numpy()[idx].mean() / b.cost_usd.to_numpy()[idx].mean())])
            np.testing.assert_allclose(samples[["quality_delta", "cost_savings"]].iloc[:25], first, atol=1e-15)

    def test_cv_folds_and_ood_fits_are_independent_and_train_only(self):
        for regime, _, path in RUNS:
            train = read_split(DATA, regime, "train")
            folds = pd.read_parquet(path / "cv_assignments.parquet")
            self.assertEqual(set(folds.prompt_id), set(train.prompts.index))
            self.assertFalse(folds.prompt_id.duplicated().any())
            self.assertEqual(folds.groupby("leakage_group").fold.nunique().max(), 1)
            if regime == "ood":
                self.assertFalse(train.prompts.dataset.eq("livecodebench").any())
        standard = joblib.load(ARTIFACTS / "standard/models.joblib")
        ood = joblib.load(ARTIFACTS / "ood/models.joblib")
        self.assertNotEqual(standard.metadata["training_prompt_ids_hash"], ood.metadata["training_prompt_ids_hash"])
        self.assertNotEqual(standard.metadata["vocabulary_hash"], ood.metadata["vocabulary_hash"])


if __name__ == "__main__":
    unittest.main()
