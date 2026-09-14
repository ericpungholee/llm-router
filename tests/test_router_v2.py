"""Tests of the new semantic representation and conservative selection protocol."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from routing_ml.conservative import (Candidate, GATE_QUANTILES, NoveltyGate, candidate_actions, candidates,
                                     comparative_route, conservative_select, resampling_weights, simultaneous_bounds)
from routing_ml.embeddings import (DIMENSIONS, ENCODER_CONFIG, load_cache, prompt_identity, token_chunks,
                                   weighted_prompt_vectors)
from routing_ml.semantic_training import FAMILIES, dense_inputs, train_semantic
from routing_ml.training import MODEL_IDS
from test_router_v1 import synthetic

ROOT = Path(__file__).resolve().parents[1]


def synthetic_embeddings(part):
    rng = np.random.default_rng(3407)
    values = rng.normal(size=(len(part.prompts), DIMENSIONS)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return pd.DataFrame(values, index=part.prompts.index)


class EmbeddingTests(unittest.TestCase):
    def test_chunking_retains_every_token_in_order_and_empty_prompt(self):
        for n in (0, 1, 510, 511, 1020, 1537):
            tokens = list(range(n))
            chunks = token_chunks(tokens)
            self.assertEqual(sum(chunks, []), tokens)
            self.assertTrue(all(len(c) <= 510 for c in chunks))
            self.assertGreaterEqual(len(chunks), 1)

    def test_prompt_identity_ignores_all_response_and_metadata_columns(self):
        rows = synthetic().prompts.reset_index()
        before = prompt_identity(rows)
        for column in ("ground_truth", "success", "cost_usd", "raw_output", "latency", "dataset"):
            rows[column] = "forbidden"
        self.assertEqual(before, prompt_identity(rows))
        rows.loc[0, "prompt"] += " changed"
        self.assertNotEqual(before, prompt_identity(rows))

    def test_weighted_chunk_pooling_preserves_alignment(self):
        vectors = np.array([[1, 0], [0, 1], [0, -1]], dtype=np.float32)
        result = weighted_prompt_vectors(vectors, [0, 1, 0], [3, 2, 4], 2)
        np.testing.assert_allclose(result, [[0.6, -0.8], [0, 1]])

    def test_cache_rejects_wrong_source_identity_and_corruption(self):
        from routing_ml.embeddings import file_hash, json_file
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            rows = pd.DataFrame(dict(prompt_id=["a", "b"], prompt=["one", "two"]))
            values = np.zeros((2, DIMENSIONS), dtype=np.float32)
            values[:, 0] = 1
            np.save(path / "vectors.npy", values, allow_pickle=False)
            rows[["prompt_id"]].to_parquet(path / "prompt_index.parquet", index=False)
            manifest = dict(encoder_config=ENCODER_CONFIG, prompt_text_identity=prompt_identity(rows),
                            files={p.name: file_hash(p) for p in path.iterdir()})
            json_file(path / "manifest.json", manifest)
            actual, _ = load_cache(path, rows)
            self.assertEqual(actual.index.tolist(), ["a", "b"])
            rows.loc[0, "prompt"] = "altered"
            with self.assertRaisesRegex(ValueError, "text/ID"):
                load_cache(path, rows)
            (path / "vectors.npy").write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_cache(path)


class SemanticTrainingTests(unittest.TestCase):
    def test_features_scaling_and_cv_are_training_only(self):
        train = synthetic()
        e = synthetic_embeddings(train)
        model, cv, folds = train_semantic(train, e, progress=None)
        np.testing.assert_allclose(model.preprocessor.mean_, e.mean().to_numpy(), atol=1e-8)
        for audit in model.metadata["cv_preprocessing"]:
            holdout = folds.loc[folds.fold == audit["fold"], "prompt_id"]
            expected = e.loc[~e.index.isin(holdout)].to_numpy(dtype=float)
            np.testing.assert_allclose(audit["mean"], expected.mean(axis=0))
        self.assertEqual(len(cv), 120)
        for name in ("validation", "test"):
            part = synthetic(name)
            with self.assertRaisesRegex(ValueError, "train only"):
                train_semantic(part, synthetic_embeddings(part), progress=None)

    def test_combined_features_allowlist_and_alignment(self):
        part = synthetic()
        e = synthetic_embeddings(part)
        expected = dense_inputs(part, e, "embedding_handcrafted")
        self.assertEqual(expected.shape[1], 400)
        part.features["response_tokens"] = 123456
        part.prompts["ground_truth"] = "secret"
        np.testing.assert_array_equal(dense_inputs(part, e, "embedding_handcrafted"), expected)
        with self.assertRaisesRegex(ValueError, "align"):
            dense_inputs(part, e.iloc[::-1], "embedding")

    def test_seed_reproduction_and_regime_independence(self):
        part = synthetic()
        e = synthetic_embeddings(part)
        a, cv_a, _ = train_semantic(part, e, progress=None)
        b, cv_b, _ = train_semantic(part, e, progress=None)
        pd.testing.assert_frame_equal(cv_a, cv_b, check_exact=True)
        pd.testing.assert_frame_equal(a.predict(part, e), b.predict(part, e), check_exact=True)
        ood = synthetic(regime="ood")
        c, _, _ = train_semantic(ood, synthetic_embeddings(ood), progress=None)
        self.assertIsNot(a.preprocessor, c.preprocessor)
        for x, y in zip(a.predictors, c.predictors):
            self.assertIsNot(x, y)
        with self.assertRaisesRegex(ValueError, "across"):
            a.predict(ood, synthetic_embeddings(ood))


class ConservativeTests(unittest.TestCase):
    def test_fixed_search_budget(self):
        choices = candidates()
        self.assertEqual(len(choices), 544)
        self.assertEqual(len({c.candidate_id for c in choices}), 544)
        self.assertEqual(set(c.family for c in choices), set(FAMILIES))

    def test_comparative_rule_uses_strictly_cheaper_models_and_falls_back(self):
        p = np.full((3, 8), 0.1)
        p[:, 6] = 0.7
        p[0, 0] = 0.8
        p[1, 0] = 0.75
        p[2, 7] = 0.99
        actual = comparative_route(p, np.arange(1, 9), 6, 0.1)
        np.testing.assert_array_equal(actual, [0, 6, 6])

    def test_comparative_cost_probability_and_id_ties(self):
        p = np.full((2, 8), 0.1)
        p[:, 6] = 0.5
        p[:, 2:4] = 0.8
        p[0, 3] = 0.9
        costs = np.arange(1, 9, dtype=float)
        costs[2:4] = 2
        actual = comparative_route(p, costs, 6, 0.1)
        # Equal probabilities/costs break lexicographically: deepseek-r1 before v3.
        np.testing.assert_array_equal(actual, [3, 3])

    def test_novelty_cutoffs_fit_training_only_and_exclude_whole_groups(self):
        part = synthetic(n=12)
        e = synthetic_embeddings(part)
        e.iloc[1] = e.iloc[0]
        gate, scores = NoveltyGate.fit(part, e)
        self.assertLess(scores[0], 0.9)
        self.assertAlmostEqual(gate.score(e.iloc[:1].to_numpy())[0], 1, places=5)
        for q in GATE_QUANTILES[1:]:
            self.assertAlmostEqual(gate.cutoffs[q], np.quantile(scores, q))
        validation = synthetic("validation")
        with self.assertRaisesRegex(ValueError, "train only"):
            NoveltyGate.fit(validation, synthetic_embeddings(validation))

    def test_gate_fallback_and_equality_at_cutoff(self):
        part = synthetic("validation", n=12)
        p = pd.DataFrame(0.5, index=part.prompts.index, columns=MODEL_IDS)
        p.iloc[:, 0] = 0.9
        gate = NoveltyGate("standard", np.eye(2), np.array(["a", "b"]), {0.05: 0.8})
        candidate = Candidate("x", "embedding", "threshold", 0.5, 0.05)
        scores = np.array([0.79, 0.8] + [0.9]*10)
        selected = candidate_actions({"embedding": p}, np.arange(1, 9), 6, scores, gate, [candidate])[:, 0]
        self.assertEqual(selected[0], 6)
        np.testing.assert_array_equal(selected[1:], 0)

    def test_resampling_preserves_groups_and_equal_dataset_macro_weights(self):
        prompts = pd.DataFrame(dict(dataset=["a", "a", "a", "b", "b"], leakage_group=["x", "x", "y", "z", "w"]))
        micro, macro = resampling_weights(prompts, replicates=30)
        np.testing.assert_allclose(micro.sum(axis=1), 1)
        np.testing.assert_allclose(macro.sum(axis=1), 1)
        np.testing.assert_allclose(micro[:, 0], micro[:, 1])
        np.testing.assert_allclose(macro[:, :3].sum(axis=1), 0.5)
        np.testing.assert_allclose(macro[:, 3:].sum(axis=1), 0.5)

    def test_simultaneous_bounds_include_all_static_comparators(self):
        prompts = synthetic().prompts
        rng = np.random.default_rng(34)
        y = rng.integers(0, 2, (len(prompts), 8))
        selected = np.column_stack([y[:, 0], y[:, 1], np.ones(len(prompts))])
        stats, samples, metadata = simultaneous_bounds(prompts, selected, y, replicates=100)
        self.assertEqual(metadata["comparators"], list(MODEL_IDS))
        self.assertEqual(stats["micro"]["lower"].shape, (3, 8))
        self.assertAlmostEqual(stats["micro"]["lower"][0, 0], 0, places=12)
        self.assertTrue((stats["micro"]["lower"] <= stats["micro"]["delta"] + 1e-12).all())
        np.testing.assert_allclose(np.quantile(samples.max_t, 0.95), metadata["critical_value"])

    def test_selector_baseline_fallback_and_all_anchor_uncertainty(self):
        validation = synthetic("validation", n=20)
        validation.y[:] = 0
        validation.y["gpt-5"] = 1
        p = pd.DataFrame(0.1, index=validation.prompts.index, columns=MODEL_IDS)
        p.iloc[:, 0] = 0.99
        gate = NoveltyGate("standard", np.eye(2), np.array(["a", "b"]), {q: 0.0 for q in GATE_QUANTILES[1:]})
        chosen, best, table, _, metadata, actions = conservative_select(validation, {f: p for f in FAMILIES}, np.arange(1, 9), np.ones(20), gate, replicates=30)
        self.assertEqual(chosen.candidate_id, "best_single")
        self.assertEqual(best.model_id, "gpt-5")
        self.assertEqual(metadata["anchor_candidate_count"], 545 * 8)
        self.assertEqual(metadata["possible_anchor_identities"], list(MODEL_IDS))
        self.assertEqual(actions.shape, (20, 545))
        with self.assertRaisesRegex(ValueError, "validation only"):
            conservative_select(synthetic("test"), {}, np.arange(1, 9), [], gate)


class CampaignIntegrationTests(unittest.TestCase):
    def test_freeze_precedes_test_outcome_load(self):
        from experiments.embedding_logreg_router import run_regime
        parts = {s: synthetic(s, n=20) for s in ("train", "validation", "test")}
        embeddings = pd.concat([synthetic_embeddings(p) for p in parts.values()])
        for part in parts.values():
            part.y[:] = 0
            part.y["gpt-5"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def read(data, regime, split):
                if split == "test":
                    self.assertTrue((root / "standard/frozen_selection.json").exists())
                return parts[split]
            from routing_ml.conservative import conservative_select as original
            def quick_select(*args, **kwargs):
                return original(*args, **kwargs, replicates=30)
            with patch("experiments.embedding_logreg_router.checked_read", side_effect=read), patch("experiments.embedding_logreg_router.verify_source", return_value={}), patch("experiments.embedding_logreg_router.conservative_select", side_effect=quick_select):
                result = run_regime(root, root, root, "standard", embeddings, {}, {"source_processed_data_hashes": {}, "protocol_sha256": "fixture"})
            self.assertTrue(result["semantic_training_replay_exact"])
            self.assertEqual(result["comparison"]["primary"]["cost_savings"], 0)
            self.assertFalse(result["comparison"]["primary"]["promising_exploratory"])


if __name__ == "__main__":
    unittest.main()
