"""Fixed references, joint validation evidence and real v4 campaign invariants."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from experiments.fixed_reference_router import canonical_part, run_regime, source_evidence, subset
from experiments.tfidf_logreg_router import sha256, model_signature
from routing_ml.bootstrap import group_strata, sample_groups
from routing_ml.crossfit import CrossfitPart, read_part
from routing_ml.fixed_reference import (Evidence, POLICY_IDS, REFERENCE_INDEX, evidence_from_actions,
                                        grid_actions, joint_bounds, select_policy, source_partitions)
from routing_ml.local_router import LocalRouter
from routing_ml.training import MODEL_IDS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/llmrouterbench"
ARTIFACTS = ROOT / "artifacts/router_v4"
AVAILABLE = (ARTIFACTS / "results.json").exists()


def synthetic_part(name="train", regime="standard"):
    n = 80
    p = pd.DataFrame(dict(prompt_id=[f"{name}_{i}" for i in range(n)],
                          prompt=[("mathquadratic algebra equation " if i < n//2 else "historyempire factual answer ") + str(i) for i in range(n)],
                          dataset=["aime"]*(n//2) + ["simpleqa"]*(n//2), leakage_group=[f"{name}_group_{i//2}" for i in range(n)])).set_index("prompt_id")
    y = np.tile((np.arange(n) % 3 != 0)[:, None], (1, 8)).astype(float)
    c = np.tile(np.arange(1, 9, dtype=float), (n, 1))
    return CrossfitPart(regime, name, p, pd.DataFrame(y, index=p.index, columns=MODEL_IDS), pd.DataFrame(c, index=p.index, columns=MODEL_IDS))


def synthetic_evidence(name):
    part = synthetic_part("validation")
    part.prompts.index = pd.Index([f"{name}_{i}" for i in range(80)], name="prompt_id")
    part.prompts.leakage_group = [f"{name}_g{i//2}" for i in range(80)]
    y = np.ones((80, 6))
    y[:12, 0] = 0
    y[::4, 1] = 0
    c = np.tile([1, 2, 3, 4, 5, 6], (80, 1)).astype(float)
    return Evidence(name, "standard", part.prompts, y - 1, c, y)


class FixedReferenceUnitTests(unittest.TestCase):
    def test_reference_is_gpt5_even_if_another_model_has_higher_validation_quality(self):
        p = pd.DataFrame(np.full((2, 8), .1), columns=MODEL_IDS)
        p[MODEL_IDS[REFERENCE_INDEX]] = [.8, .8]
        p[MODEL_IDS[0]] = [.9, .79]
        costs = np.arange(1, 9)
        actions = grid_actions(p, costs)
        np.testing.assert_array_equal(actions.fixed_gpt5, [REFERENCE_INDEX]*2)
        np.testing.assert_array_equal(actions['advantage_0.02'], [0, REFERENCE_INDEX])
        with self.assertRaises(ValueError):
            grid_actions(p.loc[:, list(reversed(MODEL_IDS))], costs)
        with self.assertRaises(ValueError):
            grid_actions(p, np.zeros(8))

    def test_source_domains_have_disjoint_groups_and_cover_only_training(self):
        train = synthetic_part()
        seen = []
        for domain, (fitting, heldout) in source_partitions(train).items():
            self.assertFalse(set(fitting) & set(heldout))
            self.assertFalse(set(train.prompts.loc[fitting].leakage_group) & set(train.prompts.loc[heldout].leakage_group))
            self.assertFalse(train.prompts.loc[fitting].dataset.eq(domain).any())
            self.assertTrue(train.prompts.loc[heldout].dataset.eq(domain).all())
            seen.extend(heldout)
        self.assertEqual(set(seen), set(train.prompts.index))
        self.assertEqual(len(seen), len(train.prompts))
        with self.assertRaises(ValueError):
            source_partitions(synthetic_part("test"))
        train.prompts.loc[train.prompts.index[40], "leakage_group"] = train.prompts.leakage_group.iloc[0]
        with self.assertRaises(ValueError):
            source_partitions(train)

    def test_auxiliary_fit_vocabularies_exclude_heldout_domain_and_use_own_costs(self):
        train = synthetic_part()
        with tempfile.TemporaryDirectory() as tmp:
            evidence = source_evidence(train, Path(tmp))
            self.assertEqual(evidence.prompts.index.tolist(), train.prompts.index.tolist())
            for domain, (fitting, heldout) in source_partitions(train).items():
                path = Path(tmp) / 'source_holdout' / domain
                model = joblib.load(path / 'models.joblib')
                excluded_word = 'mathquadratic' if domain == 'aime' else 'historyempire'
                self.assertNotIn(excluded_word, model.preprocessor.vocabulary_)
                metadata = json.loads((path / 'metadata.json').read_text())
                np.testing.assert_allclose([metadata['mean_training_cost_usd'][m] for m in MODEL_IDS], train.costs.loc[fitting].mean())

    def test_ood_evidence_rejects_code_and_test_role(self):
        val = synthetic_part('validation', 'ood')
        actions = pd.DataFrame(np.full((80, 6), REFERENCE_INDEX), index=val.prompts.index, columns=POLICY_IDS)
        val.prompts.dataset = 'livecodebench'
        with self.assertRaises(ValueError):
            evidence_from_actions(val, actions, 'ordinary')
        with self.assertRaises(ValueError):
            evidence_from_actions(synthetic_part('test'), actions, 'ordinary')

    def test_joint_bounds_match_independent_group_resampling_and_maximum(self):
        blocks = {n: synthetic_evidence(n) for n in ('ordinary', 'source_holdout')}
        bounds, samples, meta = joint_bounds(blocks, replicates=40)
        rng = np.random.default_rng(3407)
        maxima = []
        for name in ('ordinary', 'source_holdout'):
            e = blocks[name]
            domains = sorted(e.prompts.dataset.unique())
            def stats(idx):
                ds = e.prompts.dataset.to_numpy()[idx]
                rows = e.quality_difference[idx]
                d = np.stack([rows[ds == domain].mean(axis=0) for domain in domains])
                return np.vstack([rows.mean(axis=0), d.mean(axis=0), d])
            observed = stats(np.arange(len(e.prompts)))
            strata = group_strata(e.prompts)
            draws = np.array([stats(sample_groups(strata, rng)) for _ in range(40)])
            std = draws.std(axis=0, ddof=1)
            normalized = np.divide(draws-observed, std, out=np.zeros_like(draws), where=std > 1e-12)
            maxima.append(np.maximum(0, normalized.max(axis=(1,2))))
        np.testing.assert_array_equal(samples.ordinary_max_t, maxima[0])
        np.testing.assert_array_equal(samples.source_holdout_max_t, maxima[1])
        np.testing.assert_array_equal(samples.joint_max_t, np.maximum(*maxima))
        self.assertEqual(meta['critical_value'], float(np.quantile(np.maximum(*maxima), .95)))
        np.testing.assert_array_equal(bounds.joint_lower, bounds.quality_delta-meta['critical_value']*bounds.standard_error)

    def test_selection_is_min_cost_feasible_and_reference_always_feasible(self):
        blocks = {n: synthetic_evidence(n) for n in ('ordinary', 'source_holdout')}
        selected, control, candidates, bounds, samples, meta = select_policy(blocks, replicates=40)
        self.assertEqual(selected, 'advantage_0.05')
        self.assertEqual(control, selected)
        self.assertTrue(candidates.set_index('policy_id').loc['fixed_gpt5', 'feasible'])
        rerun = select_policy(blocks, replicates=40)
        pd.testing.assert_frame_equal(samples, rerun[4], check_exact=True)
        blocks['source_holdout'].prompts.index = blocks['ordinary'].prompts.index
        with self.assertRaises(ValueError):
            joint_bounds(blocks, replicates=10)

    def test_runner_freezes_gpt5_and_actions_before_requesting_test_outcomes(self):
        parts = {name: synthetic_part(name) for name in ('train', 'validation', 'test')}
        # A different validation best-single must not silently replace GPT-5.
        parts['validation'].y.iloc[:, :] = 0
        parts['validation'].y[MODEL_IDS[0]] = 1
        prompts = pd.concat([p.prompts.assign(standard_split=name) for name, p in parts.items()])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            events = []
            def read_checked(data, request_prompts, regime, name):
                if name == 'test':
                    path = output / regime
                    for filename in ('frozen_selection.json', 'test_predictions.parquet', 'test_actions.parquet', 'test_routing_decisions.parquet'):
                        self.assertTrue((path / filename).exists(), filename)
                    frozen = json.loads((path / 'frozen_selection.json').read_text())
                    self.assertEqual(frozen['reference']['model_id'], 'gpt-5')
                    self.assertEqual(frozen['validation_best_single']['model_id'], MODEL_IDS[0])
                events.append(name)
                return parts[name]
            with patch('experiments.fixed_reference_router.canonical_part', side_effect=read_checked), \
                 patch('experiments.fixed_reference_router.evaluate', return_value={'comparisons': {'primary': {}}}):
                run_regime(output, prompts, output, 'standard')
            self.assertEqual(events, ['train', 'validation', 'test'])


@unittest.skipUnless(AVAILABLE, 'Run v4 locally to verify saved campaign artifacts')
class FixedReferenceArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((ARTIFACTS/'results.json').read_text())
        cls.prompts = pd.read_parquet(DATA/'prompts.parquet').set_index('prompt_id').sort_index()

    def test_all_hashes_and_prior_reports_match(self):
        hashes = json.loads((ARTIFACTS/'artifact_hashes.json').read_text())
        for name, expected in hashes.items():
            self.assertEqual(sha256(ARTIFACTS/name), expected, name)
        lock = self.result['provenance']
        for name, expected in lock['source_processed_data_hashes'].items():
            self.assertEqual(sha256(DATA/name), expected)
        for name, expected in lock['prior_report_hashes'].items():
            self.assertEqual(sha256(ROOT/name), expected)

    def test_auxiliary_models_independent_domain_excluded_and_train_only(self):
        signatures = []
        for regime in ('standard','ood'):
            train = canonical_part(DATA, self.prompts, regime, 'train')
            assignments = pd.read_parquet(ARTIFACTS/regime/'source_assignments.parquet')
            self.assertEqual(set(assignments.prompt_id), set(train.prompts.index))
            seen = []
            for domain, rows in assignments.groupby('excluded_domain'):
                path = ARTIFACTS/regime/'source_holdout'/domain
                metadata = json.loads((path/'metadata.json').read_text())
                fitting = rows[rows.role=='train']
                heldout = rows[rows.role=='validation']
                self.assertFalse(fitting.dataset.eq(domain).any())
                self.assertTrue(heldout.dataset.eq(domain).all())
                self.assertFalse(set(fitting.leakage_group) & set(heldout.leakage_group))
                self.assertEqual(metadata['training_prompt_count'],len(fitting))
                self.assertNotIn(domain,metadata['training_dataset_counts'])
                if regime=='ood':
                    self.assertFalse(rows.dataset.eq('livecodebench').any())
                model=joblib.load(path/'models.joblib')
                self.assertEqual(model_signature(model),metadata['model_signature'])
                signatures.append(metadata['vocabulary_hash'])
                pd.testing.assert_frame_equal(pd.read_parquet(path/'heldout_predictions.parquet').set_index('prompt_id'),
                                              model.predict(SimpleNamespace(regime=regime,prompts=self.prompts.loc[heldout.prompt_id])),check_exact=True)
                np.testing.assert_allclose([metadata['mean_training_cost_usd'][m] for m in MODEL_IDS],train.costs.loc[fitting.prompt_id].mean())
                seen.extend(heldout.prompt_id)
            self.assertEqual(len(seen),len(set(seen)))
            self.assertEqual(set(seen),set(train.prompts.index))
        self.assertEqual(len(set(signatures)),11)

    def test_frozen_selection_matches_all_validation_bounds_and_costs(self):
        for regime in ('standard','ood'):
            path=ARTIFACTS/regime
            frozen=json.loads((path/'frozen_selection.json').read_text())
            self.assertEqual(frozen['reference']['model_id'],'gpt-5')
            self.assertFalse(frozen['test_used_for_selection'])
            bounds=pd.read_parquet(path/'validation_quality_bounds.parquet')
            candidates=pd.read_parquet(path/'validation_policy_grid.parquet')
            expected=bounds.groupby('policy_id').joint_lower.min().ge(-.005)
            self.assertEqual(candidates.set_index('policy_id').feasible.to_dict(),expected.to_dict())
            winner=candidates[candidates.feasible].sort_values(['mean_cost_usd','quality','policy_id'],ascending=[True,False,True]).iloc[0].policy_id
            self.assertEqual(winner,frozen['primary']['candidate_id'])
            samples=pd.read_parquet(path/'validation_bootstrap_max_t.parquet')
            self.assertEqual(len(samples),2000)
            np.testing.assert_array_equal(samples.joint_max_t,np.maximum(samples.ordinary_max_t,samples.source_holdout_max_t))
            self.assertEqual(np.quantile(samples.joint_max_t,.95),frozen['selection_metadata']['critical_value'])
            train=canonical_part(DATA,self.prompts,regime,'train')
            np.testing.assert_allclose([frozen['mean_training_cost_usd'][m] for m in MODEL_IDS],train.costs.mean())

    def test_all_final_predictions_evaluation_and_runtime_align(self):
        for regime in ('standard','ood'):
            path=ARTIFACTS/regime
            model=joblib.load(path/'tfidf/models.joblib')
            metadata=json.loads((path/'tfidf/metadata.json').read_text())
            self.assertTrue(metadata['training_replay_exact'])
            for split in ('train','validation','test'):
                ids=self.prompts.index[self.prompts[regime+'_split']==split]
                request=SimpleNamespace(regime=regime,prompts=self.prompts.loc[ids])
                predictions=pd.read_parquet(path/f'{split}_predictions.parquet').set_index('prompt_id')
                pd.testing.assert_frame_equal(predictions,model.predict(request),check_exact=True)
            e=pd.read_parquet(path/'test_evaluation.parquet')
            test=canonical_part(DATA,self.prompts,regime,'test')
            rows=test.y.index.get_indexer(e.prompt_id)
            columns=[MODEL_IDS.index(m) for m in e.selected_model_id]
            np.testing.assert_array_equal(e.success,test.y.to_numpy()[rows,columns])
            np.testing.assert_array_equal(e.cost_usd,test.costs.to_numpy()[rows,columns])
            decisions=pd.read_parquet(path/'test_routing_decisions.parquet').set_index('prompt_id')
            self.assertFalse({'success','cost_usd','dataset','leakage_group'} & set(decisions.columns))
            routed=LocalRouter(path).predict(test.prompts.reset_index()[['prompt_id','prompt']])
            self.assertEqual([r['selected_model_id'] for r in routed],decisions.primary.tolist())
            for record in routed:
                if record['probabilities'] is not None:
                    np.testing.assert_array_equal([record['probabilities'][m] for m in MODEL_IDS],predictions.loc[record['prompt_id']])

    def test_report_metrics_bootstrap_quantiles_and_fallback_are_truthful(self):
        for regime in ('standard','ood'):
            r=self.result['regimes'][regime]
            e=pd.read_parquet(ARTIFACTS/regime/'test_evaluation.parquet')
            for name,metrics in r['micro'].items():
                rows=e[e.policy_id==name]
                self.assertAlmostEqual(metrics['quality'],rows.success.mean())
                self.assertAlmostEqual(metrics['mean_cost_usd'],rows.cost_usd.mean())
            for name,c in r['comparisons'].items():
                samples=pd.read_parquet(ARTIFACTS/regime/f'{name}_bootstrap_samples.parquet')
                for metric,interval in c['bootstrap']['ci_95'].items():
                    np.testing.assert_array_equal(np.quantile(samples[metric],[.025,.975]),interval)
                self.assertEqual(c['promising_exploratory'],all(c['conditions'].values()))
            if r['frozen_selection']['primary']['candidate_id']=='fixed_gpt5':
                self.assertEqual(r['comparisons']['primary']['cost_savings'],0)
                self.assertFalse(r['comparisons']['primary']['promising_exploratory'])


if __name__=='__main__':
    unittest.main()
