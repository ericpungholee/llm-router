"""Independent verification of local encoder and actual frozen v2 outputs."""

import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from routing_ml.embeddings import (
    ENCODER_CONFIG,
    encode_prompts,
    file_hash,
    load_cache,
    verify_encoder,
)
from routing_ml.semantic_training import FAMILIES, predict_family
from routing_ml.training import MODEL_IDS, read_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/llmrouterbench"
ARTIFACTS = ROOT / "artifacts/router_v2"
ENCODER_AVAILABLE = (ARTIFACTS / "encoder/manifest.json").exists()
RUNS_AVAILABLE = all((ARTIFACTS / r / "results.json").exists() for r in ("standard", "ood"))


@unittest.skipUnless(
    ENCODER_AVAILABLE, "Download the pinned local v2 encoder to verify real inference"
)
class LocalEncoderTests(unittest.TestCase):
    def test_real_encoder_offline_determinism_and_response_projection(self):
        prompts = pd.DataFrame(
            dict(
                prompt_id=["a", "b", "c"],
                prompt=["What is two plus three?", "What is two plus three?", "word " * 1100],
            )
        )
        before, stats, metadata = encode_prompts(prompts, ARTIFACTS / "encoder", progress=None)
        prompts["ground_truth"] = "forbidden secret label"
        prompts["raw_output"] = "forbidden model response"
        prompts["cost_usd"] = 99999
        prompts["dataset"] = "privileged benchmark identity"
        after, _, _ = encode_prompts(prompts, ARTIFACTS / "encoder", progress=None)
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(before[0], before[1])
        self.assertEqual(before.shape, (3, 384))
        self.assertEqual(stats.loc[2, "content_tokens"], 1100)
        self.assertEqual(stats.loc[2, "chunks"], 3)
        self.assertEqual(metadata["encoder_config"], ENCODER_CONFIG)
        verify_encoder(ARTIFACTS / "encoder")


@unittest.skipUnless(RUNS_AVAILABLE, "Run both v2 regimes to verify their real artifacts")
class SavedCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embeddings, _ = load_cache(ARTIFACTS / "embeddings")

    def test_all_artifact_and_original_data_hashes_match(self):
        lock = json.loads((ARTIFACTS / "campaign_lock.json").read_text())
        for name, expected in lock["source_processed_data_hashes"].items():
            self.assertEqual(file_hash(DATA / name), expected)
        for name, expected in lock["source_sha256"].items():
            self.assertEqual(file_hash(ARTIFACTS / "source_snapshot" / name), expected)
        for regime in ("standard", "ood"):
            directory = ARTIFACTS / regime
            for name, expected in json.loads(
                (directory / "artifact_hashes.json").read_text()
            ).items():
                self.assertEqual(file_hash(directory / name), expected)

    def test_all_predictions_and_cv_selection_reproduce(self):
        for regime in ("standard", "ood"):
            for family in FAMILIES:
                directory = ARTIFACTS / regime / family
                model = joblib.load(directory / "models.joblib")
                scores = (
                    pd.read_parquet(directory / "cv_scores.parquet").groupby("C").log_loss.mean()
                )
                self.assertEqual(model.selected_c, min(scores.index, key=lambda c: (scores[c], c)))
                for split in ("train", "validation", "test"):
                    part = read_split(DATA, regime, split)
                    saved = pd.read_parquet(directory / f"{split}_predictions.parquet").set_index(
                        "prompt_id"
                    )
                    self.assertEqual(tuple(saved.columns), MODEL_IDS)
                    pd.testing.assert_frame_equal(
                        saved, predict_family(model, part, self.embeddings), check_exact=True
                    )

    def test_novelty_references_and_cv_ids_are_training_only(self):
        for regime in ("standard", "ood"):
            train = read_split(DATA, regime, "train")
            gate = joblib.load(ARTIFACTS / regime / "novelty_gate.joblib")
            np.testing.assert_array_equal(gate.reference, self.embeddings.loc[train.prompts.index])
            np.testing.assert_array_equal(gate.groups, train.prompts.leakage_group)
            similarity = pd.read_parquet(ARTIFACTS / regime / "train_novelty.parquet")
            self.assertEqual(similarity.prompt_id.tolist(), train.prompts.index.tolist())
            for q, cutoff in gate.cutoffs.items():
                self.assertAlmostEqual(cutoff, np.quantile(similarity.similarity, q))
            for family in FAMILIES:
                folds = pd.read_parquet(ARTIFACTS / regime / family / "cv_assignments.parquet")
                self.assertEqual(set(folds.prompt_id), set(train.prompts.index))
                self.assertEqual(folds.groupby("leakage_group").fold.nunique().max(), 1)
            if regime == "ood":
                self.assertFalse(train.prompts.dataset.eq("livecodebench").any())

    def test_saved_primary_is_validation_minimum_feasible(self):
        for regime in ("standard", "ood"):
            directory = ARTIFACTS / regime
            frozen = json.loads((directory / "frozen_selection.json").read_text())
            table = pd.read_parquet(directory / "validation_candidates.parquet")
            samples = pd.read_parquet(directory / "validation_bootstrap_max_t.parquet")
            critical = np.quantile(samples.max_t, 0.95)
            self.assertAlmostEqual(critical, frozen["selection"]["critical_value"])
            self.assertEqual(len(samples), 2000)
            self.assertEqual(frozen["selection"]["anchor_candidate_count"], 4360)
            self.assertEqual(frozen["selection"]["comparators"], list(MODEL_IDS))
            for metric in ("micro", "macro"):
                np.testing.assert_allclose(
                    table[metric + "_lower"],
                    table[metric + "_delta"] - critical * table[metric + "_standard_error"],
                    atol=1e-12,
                )
            feasible = table[
                (table.micro_lower >= -0.005)
                & (table.macro_lower >= -0.005)
                & (table.worst_dataset_delta >= -0.005)
            ]
            winner = min(
                feasible.itertuples(), key=lambda r: (r.mean_cost_usd, -r.quality, r.candidate_id)
            )
            self.assertEqual(winner.candidate_id, frozen["primary"]["candidate_id"])
            actions = pd.read_parquet(directory / "test_actions.parquet")
            np.testing.assert_array_equal(
                actions.primary, actions[frozen["primary"]["candidate_id"]]
            )

    def test_scalar_policy_reference_for_every_candidate_on_test_sample(self):
        for regime in ("standard", "ood"):
            directory = ARTIFACTS / regime
            frozen = json.loads((directory / "frozen_selection.json").read_text())
            costs = frozen["mean_training_cost_usd"]
            baseline = frozen["best_single"]["model_id"]
            policies = pd.read_parquet(directory / "validation_candidates.parquet")
            selected = pd.read_parquet(directory / "test_actions.parquet").set_index("prompt_id")
            sample_ids = selected.index[np.linspace(0, len(selected) - 1, 19).astype(int)]
            predictions = {
                f: pd.read_parquet(directory / f / "test_predictions.parquet")
                .set_index("prompt_id")
                .loc[sample_ids]
                for f in FAMILIES
            }
            similarity = (
                pd.read_parquet(directory / "test_novelty.parquet")
                .set_index("prompt_id")
                .similarity
            )
            for policy in policies.itertuples():
                for pid in sample_ids:
                    if policy.kind == "static" or (
                        policy.gate_quantile
                        and similarity[pid] < frozen["novelty_cutoffs"][str(policy.gate_quantile)]
                    ):
                        chosen = baseline
                    else:
                        p = predictions[policy.family].loc[pid].to_dict()
                        if policy.kind == "advantage":
                            eligible = [
                                m
                                for m in MODEL_IDS
                                if costs[m] < costs[baseline] and p[m] >= p[baseline] + policy.value
                            ]
                            chosen = (
                                min(eligible, key=lambda m: (costs[m], -p[m], m))
                                if eligible
                                else baseline
                            )
                        elif policy.kind == "threshold":
                            eligible = [m for m in MODEL_IDS if p[m] >= policy.value]
                            chosen = (
                                min(eligible, key=lambda m: (costs[m], -p[m], m))
                                if eligible
                                else min(MODEL_IDS, key=lambda m: (-p[m], costs[m], m))
                            )
                        else:
                            chosen = min(
                                MODEL_IDS,
                                key=lambda m: (
                                    -(p[m] - policy.value * costs[m] / max(costs.values())),
                                    costs[m],
                                    m,
                                ),
                            )
                    self.assertEqual(MODEL_IDS[selected.loc[pid, policy.candidate_id]], chosen)

    def test_all_cost_quality_points_match_source_outcomes(self):
        for regime in ("standard", "ood"):
            directory = ARTIFACTS / regime
            test = read_split(DATA, regime, "test")
            actions = pd.read_parquet(directory / "test_actions.parquet").set_index("prompt_id")
            self.assertEqual(actions.index.tolist(), test.prompts.index.tolist())
            y = test.y.to_numpy(dtype=float)[np.arange(len(actions))[:, None], actions.to_numpy()]
            c = test.costs.to_numpy(dtype=float)[
                np.arange(len(actions))[:, None], actions.to_numpy()
            ]
            summary = pd.read_parquet(directory / "test_policy_metrics.parquet")
            micro = (
                summary[summary.dataset == "__micro__"].set_index("policy_id").loc[actions.columns]
            )
            np.testing.assert_allclose(micro.quality, y.mean(axis=0))
            np.testing.assert_allclose(micro.mean_cost_usd, c.mean(axis=0))
            self.assertFalse(
                {"success", "cost_usd"}
                & set(pd.read_parquet(directory / "test_routing_decisions.parquet").columns)
            )

    def test_primary_bootstrap_bounds_and_success_conditions_recompute(self):
        for regime in ("standard", "ood"):
            directory = ARTIFACTS / regime
            r = json.loads((directory / "results.json").read_text())["comparison"]["primary"]
            samples = pd.read_parquet(directory / "primary_bootstrap_samples.parquet")
            for name, ci in r["bootstrap"]["ci_95"].items():
                np.testing.assert_allclose(ci, np.quantile(samples[name], [0.025, 0.975]))
            self.assertEqual(r["promising_exploratory"], all(r["conditions"].values()))

    def test_label_free_local_router_matches_frozen_decisions_and_probabilities(self):
        from routing_ml.local_router import LocalRouter

        for regime in ("standard", "ood"):
            directory = ARTIFACTS / regime
            router = LocalRouter(directory)
            part = read_split(DATA, regime, "test")
            # Both selected v2 routers require only TF-IDF; no encoder or outcomes.
            text = part.prompts[["prompt"]].reset_index()
            actual = router.predict(text)
            selected = pd.read_parquet(directory / "test_actions.parquet").set_index("prompt_id")
            self.assertEqual(
                [r["selected_model_id"] for r in actual], [MODEL_IDS[i] for i in selected.primary]
            )
            if router.candidate.family != "static":
                saved = pd.read_parquet(
                    directory / router.candidate.family / "test_predictions.parquet"
                ).set_index("prompt_id")
                np.testing.assert_array_equal(
                    [[row["probabilities"][m] for m in MODEL_IDS] for row in actual], saved
                )
            text["dataset"] = "injected privileged data"
            text["success"] = 0
            text["cost_usd"] = 1e9
            self.assertEqual(actual, router.predict(text))

    def test_local_router_rejects_duplicate_ids_and_artifact_tampering(self):
        import shutil
        import tempfile

        from routing_ml.local_router import LocalRouter

        router = LocalRouter(ARTIFACTS / "standard")
        with self.assertRaisesRegex(ValueError, "Unique"):
            router.predict(pd.DataFrame(dict(prompt_id=["x", "x"], prompt=["hello", "world"])))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for name in ("artifact_hashes.json", "frozen_selection.json"):
                shutil.copyfile(ARTIFACTS / "standard" / name, path / name)
            with (path / "frozen_selection.json").open("a") as f:
                f.write(" ")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                LocalRouter(path)


if __name__ == "__main__":
    unittest.main()
