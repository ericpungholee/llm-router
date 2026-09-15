#!/usr/bin/env python3
"""Train and evaluate fixed-GPT-5 routing with source-domain validation offline."""

import argparse
import importlib.metadata
import json
import shutil
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.embedding_logreg_router import summarize_actions, test_comparison
from experiments.tfidf_logreg_router import (
    COUNTS,
    clean,
    model_signature,
    sha256,
    verify_source,
    write_json,
)
from routing_ml.bootstrap import paired_bootstrap
from routing_ml.crossfit import CrossfitPart, fit_fixed_router, read_part
from routing_ml.fixed_reference import (
    MARGINS,
    POLICY_IDS,
    REFERENCE,
    REFERENCE_INDEX,
    evidence_from_actions,
    evidence_from_predictions,
    grid_actions,
    select_policy,
    source_partitions,
)
from routing_ml.metrics import (
    classifier_diagnostics,
    oracle_actions,
    pareto_table,
    selection_tables,
)
from routing_ml.policies import Policy, best_single, cheapest, estimate_costs
from routing_ml.training import MODEL_IDS, SEED, digest


def start_campaign(output, data):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty --output-dir; existing campaigns are protected")
    hashes = verify_source(data)
    output.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / "reports/router_v4_experiment_spec.md", Path(__file__).resolve()]
    paths += sorted((ROOT / "routing_ml").glob("*.py"))
    paths += [
        ROOT / "experiments/tfidf_logreg_router.py",
        ROOT / "experiments/embedding_logreg_router.py",
        ROOT / "experiments/requirements-router-v1.txt",
        ROOT / "experiments/route_local.py",
    ]
    source_hashes = {}
    for p in paths:
        name = str(p.relative_to(ROOT))
        source_hashes[name] = sha256(p)
        destination = output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, destination)
    prior = {
        str(p.relative_to(ROOT)): sha256(p)
        for version in (1, 2, 3)
        for p in sorted((ROOT / "reports").glob(f"router_v{version}*"))
        if p.is_file()
    }
    lock = dict(
        seed=SEED,
        fixed_reference=REFERENCE,
        source_processed_data_hashes=hashes,
        source_hashes=source_hashes,
        prior_report_hashes=prior,
        environment=dict(
            python=sys.version,
            packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        ),
        interpretation="exploratory on previously inspected benchmark outcomes",
    )
    write_json(output / "campaign_lock.json", lock)
    return lock


def canonical_part(data, prompts, regime, name):
    ids = prompts.index[prompts[regime + "_split"] == name]
    if len(ids) != COUNTS[regime][name]:
        raise ValueError("Frozen canonical split count mismatch")
    return read_part(data, prompts, ids, regime, name)


def subset(part, ids, role):
    return CrossfitPart(
        part.regime,
        role,
        part.prompts.loc[ids].copy(),
        part.y.loc[ids].copy(),
        part.costs.loc[ids].copy(),
    )


def fit(train, output, excluded_domain=None, replay=False):
    start = time.perf_counter()
    model = fit_fixed_router(train)
    model.metadata.update(
        C_selection="C=1 fixed before v4; no tuning",
        fit_split="canonical_train" if excluded_domain is None else "source_domain_train",
        excluded_source_domain=excluded_domain,
    )
    prediction = model.predict(train)
    signature = model_signature(model)
    if replay:
        other = fit_fixed_router(train)
        other.metadata.update(
            C_selection="C=1 fixed before v4; no tuning",
            fit_split="canonical_train" if excluded_domain is None else "source_domain_train",
            excluded_source_domain=excluded_domain,
        )
        pd.testing.assert_frame_equal(prediction, other.predict(train), check_exact=True)
        if signature != model_signature(other):
            raise AssertionError("Training replay differs")
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "models.joblib", compress=3)
    pd.testing.assert_frame_equal(
        prediction, joblib.load(output / "models.joblib").predict(train), check_exact=True
    )
    costs, normalized = estimate_costs(train)
    metadata = dict(
        **model.metadata,
        model_signature=signature,
        training_seconds=time.perf_counter() - start,
        training_replay_exact=bool(replay),
        mean_training_cost_usd=dict(zip(MODEL_IDS, costs)),
        normalized_training_cost=dict(zip(MODEL_IDS, normalized)),
    )
    write_json(output / "metadata.json", metadata)
    return model, prediction, costs


def source_evidence(train, output):
    predictions, actions, assignments = [], [], []
    for domain, (fit_ids, score_ids) in source_partitions(train).items():
        fitting, heldout = subset(train, fit_ids, "train"), subset(train, score_ids, "validation")
        path = output / "source_holdout" / domain
        model, _, costs = fit(fitting, path, excluded_domain=domain)
        pred = model.predict(heldout)
        action = grid_actions(pred, costs)
        pred.reset_index().to_parquet(path / "heldout_predictions.parquet", index=False)
        action.reset_index().to_parquet(path / "heldout_actions.parquet", index=False)
        predictions.append(pred)
        actions.append(action)
        for role, part in (("train", fitting), ("validation", heldout)):
            assignments.append(
                part.prompts[["dataset", "leakage_group"]]
                .reset_index()
                .assign(excluded_domain=domain, role=role)
            )
        print(
            f"{train.regime}: held out source domain {domain}; fitted {len(fit_ids)}, validated {len(score_ids)}",
            flush=True,
        )
    predictions = pd.concat(predictions)
    actions = pd.concat(actions)
    if predictions.index.has_duplicates or set(predictions.index) != set(train.prompts.index):
        raise ValueError("Source holdout coverage failure")
    predictions = predictions.loc[train.prompts.index]
    actions = actions.loc[train.prompts.index]
    predictions.reset_index().to_parquet(output / "source_holdout_predictions.parquet", index=False)
    actions.reset_index().to_parquet(output / "source_holdout_actions.parquet", index=False)
    pd.concat(assignments, ignore_index=True).to_parquet(
        output / "source_assignments.parquet", index=False
    )
    heldout = subset(train, train.prompts.index, "validation")
    evidence = evidence_from_actions(heldout, actions, "source_holdout")
    save_evidence(evidence, output)
    return evidence


def save_evidence(evidence, output):
    records = []
    for j, pid in enumerate(POLICY_IDS):
        records.append(
            pd.DataFrame(
                dict(
                    prompt_id=evidence.prompts.index,
                    policy_id=pid,
                    dataset=evidence.prompts.dataset.to_numpy(),
                    leakage_group=evidence.prompts.leakage_group.to_numpy(),
                    success=evidence.quality[:, j],
                    quality_delta=evidence.quality_difference[:, j],
                    cost_usd=evidence.costs[:, j],
                )
            )
        )
    pd.concat(records, ignore_index=True).to_parquet(
        output / f"{evidence.name}_validation_evaluation.parquet", index=False
    )


def runtime_candidate(policy_id):
    if policy_id == "fixed_gpt5":
        return dict(
            candidate_id=policy_id, family="static", kind="static", value=0, gate_quantile=0
        )
    return dict(
        candidate_id=policy_id,
        family="tfidf",
        kind="advantage",
        value=MARGINS[POLICY_IDS.index(policy_id)],
        gate_quantile=0,
    )


def evaluate(test, prediction, actions, costs, frozen, output):
    policy_ids = list(actions.columns) + ["oracle"]
    oracle = oracle_actions(test, costs)
    matrix = np.column_stack([actions.to_numpy(), oracle]).astype(np.int8)
    summary, y, c = summarize_actions(test, matrix, policy_ids)
    summary.to_parquet(output / "test_policy_metrics.parquet", index=False)
    frontier = pareto_table(summary)
    frontier.to_parquet(output / "pareto_table.parquet", index=False)
    records = []
    for j, pid in enumerate(policy_ids):
        records.append(
            pd.DataFrame(
                dict(
                    prompt_id=test.prompts.index,
                    policy_id=pid,
                    selected_model_id=np.asarray(MODEL_IDS)[matrix[:, j]],
                    dataset=test.prompts.dataset.to_numpy(),
                    leakage_group=test.prompts.leakage_group.to_numpy(),
                    success=y[:, j],
                    cost_usd=c[:, j],
                )
            )
        )
    pd.concat(records, ignore_index=True).to_parquet(
        output / "test_evaluation.parquet", index=False
    )
    baseline = np.full(len(test.prompts), REFERENCE_INDEX)
    comparisons = {}
    for name in ("primary", "ordinary_only_control", "fixed_margin_control"):
        samples, comparison = test_comparison(test, actions[name].to_numpy(), baseline, oracle)
        comparison["bootstrap"].update(
            independent_confirmation=False,
            interpretation="exploratory; conditional on frozen fit and selection",
        )
        comparisons[name] = comparison
        samples.to_parquet(output / f"{name}_bootstrap_samples.parquet", index=False)
    primary = actions.primary.to_numpy()
    per_domain = []
    for domain in sorted(test.prompts.dataset.unique()):
        ids = test.prompts.index[test.prompts.dataset == domain]
        part = subset(test, ids, "test")
        mask = test.prompts.index.get_indexer(ids)
        samples, comparison = test_comparison(part, primary[mask], baseline[mask], oracle[mask])
        samples.to_parquet(output / f"dataset_{domain}_bootstrap_samples.parquet", index=False)
        per_domain.append(dict(dataset=domain, n=len(ids), **comparison))
    comparisons["primary"]["all_dataset_interval_margins_pass"] = all(
        d["bootstrap"]["ci_95"]["quality_delta"][0] >= -0.005 for d in per_domain
    )
    diagnostics, reliability = classifier_diagnostics(test, prediction, primary)
    diagnostics.to_parquet(output / "test_classifier_metrics.parquet", index=False)
    reliability.to_parquet(output / "test_reliability.parquet", index=False)
    distribution, counts = selection_tables(test, prediction, primary)
    distribution.to_parquet(output / "selection_distribution.parquet", index=False)
    counts.to_parquet(output / "dataset_model_counts.parquet", index=False)
    micro = summary[summary.dataset == "__micro__"].set_index("policy_id")
    macro = summary[summary.dataset == "__macro__"].set_index("policy_id")
    composition = []
    for excluded in (("simpleqa",), ("simpleqa", "mmlupro")):
        mask = ~test.prompts.dataset.isin(excluded).to_numpy()
        r, b = policy_ids.index("primary"), policy_ids.index("fixed_gpt5")
        composition.append(
            dict(
                excluded=list(excluded),
                n=int(mask.sum()),
                quality_delta=float((y[mask, r] - y[mask, b]).mean()),
                cost_savings=float(1 - c[mask, r].mean() / c[mask, b].mean()),
            )
        )
    source_metadata = {
        p.parent.name: json.loads(p.read_text())
        for p in sorted((output / "source_holdout").glob("*/metadata.json"))
    }
    return dict(
        regime=test.regime,
        selected_C=1,
        fixed_reference=REFERENCE,
        frozen_selection=frozen,
        comparisons=comparisons,
        micro=micro.to_dict(orient="index"),
        macro=macro.to_dict(orient="index"),
        per_dataset=per_domain,
        distribution=distribution.to_dict(orient="records"),
        dataset_model_counts=counts.to_dict(orient="records"),
        classifier_metrics=diagnostics.to_dict(orient="records"),
        composition_checks=composition,
        primary_on_pareto=bool(frontier.set_index("policy_id").loc["primary", "pareto_deployable"]),
        source_fit_metadata=source_metadata,
        independent_confirmation=False,
    )


def run_regime(data, prompts, output, regime):
    path = output / regime
    path.mkdir(parents=True)
    train = canonical_part(data, prompts, regime, "train")
    validation = canonical_part(data, prompts, regime, "validation")
    model, train_prediction, costs = fit(train, path / "tfidf", replay=True)
    train_prediction.reset_index().to_parquet(path / "train_predictions.parquet", index=False)
    validation_prediction = model.predict(validation)
    validation_prediction.reset_index().to_parquet(
        path / "validation_predictions.parquet", index=False
    )
    for part, prediction in ((train, train_prediction), (validation, validation_prediction)):
        diagnostics, reliability = classifier_diagnostics(part, prediction)
        diagnostics.to_parquet(path / f"{part.name}_classifier_metrics.parquet", index=False)
        reliability.to_parquet(path / f"{part.name}_reliability.parquet", index=False)
    ordinary = evidence_from_predictions(validation, validation_prediction, costs, "ordinary")
    save_evidence(ordinary, path)
    source = source_evidence(train, path)
    selected, control, candidates, bounds, samples, selection_metadata = select_policy(
        dict(ordinary=ordinary, source_holdout=source)
    )
    candidates.to_parquet(path / "validation_policy_grid.parquet", index=False)
    bounds.to_parquet(path / "validation_quality_bounds.parquet", index=False)
    samples.to_parquet(path / "validation_bootstrap_max_t.parquet", index=False)
    fixed_control = "advantage_0.02" if regime == "standard" else "advantage_0.05"
    selected_best = best_single(validation, costs)
    frozen = dict(
        primary=runtime_candidate(selected),
        reference=Policy("fixed_gpt5", "static", model_id=REFERENCE).to_dict(),
        validation_best_single=selected_best.to_dict(),
        controls=dict(ordinary_only=control, fixed_margin=fixed_control),
        mean_training_cost_usd=dict(zip(MODEL_IDS, costs)),
        normalized_training_cost=dict(zip(MODEL_IDS, costs / costs.max())),
        candidate_model_order=list(MODEL_IDS),
        seed=SEED,
        selected_C=1,
        selection_metadata=selection_metadata,
        test_used_for_selection=False,
        training_prompt_ids_hash=digest(train.prompts.index.tolist()),
        validation_prompt_ids_hash=digest(validation.prompts.index.tolist()),
        source_holdout_prompt_ids_hash=digest(source.prompts.index.tolist()),
    )
    write_json(path / "frozen_selection.json", frozen)
    frozen_hash = sha256(path / "frozen_selection.json")
    print(
        f"{regime}: frozen primary={selected}; ordinary-only={control}; joint feasible={selection_metadata['feasible_count']}/6",
        flush=True,
    )
    # Request-side text only until all test actions have been persisted.
    test_ids = prompts.index[prompts[regime + "_split"] == "test"]
    request = SimpleNamespace(regime=regime, prompts=prompts.loc[test_ids])
    prediction = model.predict(request)
    prediction.reset_index().to_parquet(path / "test_predictions.parquet", index=False)
    actions = grid_actions(prediction, costs)
    actions["primary"] = actions[selected]
    actions["ordinary_only_control"] = actions[control]
    actions["fixed_margin_control"] = actions[fixed_control]
    for j, mid in enumerate(MODEL_IDS):
        actions["static_" + mid] = j
    actions["validation_best_single"] = MODEL_IDS.index(selected_best.model_id)
    actions["always_cheapest"] = cheapest(costs)
    actions.reset_index().to_parquet(path / "test_actions.parquet", index=False)
    decisions = actions.apply(lambda column: np.asarray(MODEL_IDS)[column.to_numpy()])
    decisions.reset_index().to_parquet(path / "test_routing_decisions.parquet", index=False)
    test = canonical_part(data, prompts, regime, "test")
    result = evaluate(test, prediction, actions, costs, frozen, path)
    if sha256(path / "frozen_selection.json") != frozen_hash:
        raise AssertionError("Selection changed after test evaluation")
    write_json(path / "results.json", result)
    write_json(
        path / "artifact_hashes.json",
        {
            str(p.relative_to(path)): sha256(p)
            for p in sorted(path.rglob("*"))
            if p.is_file() and p.name != "artifact_hashes.json"
        },
    )
    print(f"{regime}: {json.dumps(clean(result['comparisons']['primary']))}", flush=True)
    return clean(result)


def matched_code(output):
    frames = []
    for regime in ("standard", "ood"):
        e = pd.read_parquet(output / regime / "test_evaluation.parquet")
        frames.append(
            e[(e.policy_id == "primary") & (e.dataset == "livecodebench")]
            .set_index("prompt_id")
            .sort_index()
        )
    standard, ood = frames
    shared = standard.index.intersection(ood.index)
    a, b = ood.loc[shared], standard.loc[shared]
    samples, bootstrap = paired_bootstrap(a, a.success, b.success, a.cost_usd, b.cost_usd)
    samples.to_parquet(output / "matched_code_bootstrap_samples.parquet", index=False)
    return dict(
        shared_n=len(shared),
        standard_quality=float(b.success.mean()),
        ood_shared_quality=float(a.success.mean()),
        ood_full_quality=float(ood.success.mean()),
        quality_delta_shared_ood_minus_standard=float((a.success - b.success).mean()),
        bootstrap=bootstrap,
        independent_confirmation=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/llmrouterbench")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/router_v4")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    from routing_ml.reporting_v4 import generate_report

    if args.report_only:
        generate_report(args.output_dir, args.reports_dir)
        return

    def blocked(*args, **kwargs):
        raise RuntimeError("This experiment permits no network/provider calls")

    with ExitStack() as stack:
        for target in (
            "socket.socket.connect",
            "socket.socket.connect_ex",
            "socket.create_connection",
            "socket.getaddrinfo",
            "urllib.request.urlopen",
        ):
            stack.enter_context(patch(target, side_effect=blocked))
        stack.enter_context(threadpool_limits(limits=1))
        lock = start_campaign(args.output_dir, args.data_dir)
        prompts = (
            pd.read_parquet(args.data_dir / "prompts.parquet").set_index("prompt_id").sort_index()
        )
        results = {}
        for regime in ("standard", "ood"):
            results[regime] = run_regime(args.data_dir, prompts, args.output_dir, regime)
        if verify_source(args.data_dir) != lock["source_processed_data_hashes"]:
            raise AssertionError("Processed data changed")
        for name, expected in lock["prior_report_hashes"].items():
            if sha256(ROOT / name) != expected:
                raise AssertionError("Prior report changed")
        result = dict(
            regimes=results,
            provenance=lock,
            matched_code=matched_code(args.output_dir),
            prior_results_and_processed_data_unchanged=True,
            independent_confirmation=False,
        )
        write_json(args.output_dir / "results.json", result)
        write_json(
            args.output_dir / "artifact_hashes.json",
            {
                str(p.relative_to(args.output_dir)): sha256(p)
                for p in sorted(args.output_dir.rglob("*"))
                if p.is_file() and p.name != "artifact_hashes.json"
            },
        )
        generate_report(args.output_dir, args.reports_dir)


if __name__ == "__main__":
    main()
