#!/usr/bin/env python3
"""Frozen router-v2 local campaign: representations, comparative routing, validation risk."""

import argparse
import json
import shutil
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.tfidf_logreg_router import (
    checked_read,
    clean,
    model_signature,
    sha256,
    verify_source,
    write_json,
)
from routing_ml.bootstrap import paired_bootstrap
from routing_ml.conservative import (
    NoveltyGate,
    candidate_actions,
    candidates,
    conservative_select,
)
from routing_ml.embeddings import ENCODER_CONFIG, load_cache
from routing_ml.metrics import (
    classifier_diagnostics,
    nondominated,
    oracle_actions,
    selection_tables,
)
from routing_ml.policies import cheapest, route, select_validation
from routing_ml.semantic_training import FAMILIES, predict_family, train_semantic
from routing_ml.training import MODEL_IDS, train_router


def campaign_lock(output, data):
    """Preserve a reviewable protocol and source snapshot before supervised fitting."""
    sources = verify_source(data)
    paths = sorted((ROOT / "routing_ml").glob("*.py")) + [
        ROOT / "experiments/embedding_logreg_router.py",
        ROOT / "experiments/tfidf_logreg_router.py",
        ROOT / "experiments/requirements-router-v2.txt",
        ROOT / "experiments/requirements-router-v1.txt",
        ROOT / "reports/router_v2_experiment_spec.md",
    ]
    manifest = dict(
        source_processed_data_hashes=sources,
        protocol_sha256=sha256(ROOT / "reports/router_v2_experiment_spec.md"),
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in paths},
        encoder_config=ENCODER_CONFIG,
        existing_test_status="previously inspected; exploratory only",
    )
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "campaign_lock.json"
    if lock.exists():
        previous = json.loads(lock.read_text())
        if previous != manifest:
            raise ValueError(
                "Campaign protocol/source changed; preserve this run and use a new output directory"
            )
    else:
        snapshot = output / "source_snapshot"
        for p in paths:
            destination = snapshot / p.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, destination)
        write_json(lock, manifest)
    return manifest


def summarize_actions(part, actions, policy_ids):
    y = part.y.to_numpy(dtype=float)[np.arange(len(part.prompts))[:, None], actions]
    c = part.costs.to_numpy(dtype=float)[np.arange(len(part.prompts))[:, None], actions]
    datasets = part.prompts.dataset.to_numpy()
    summaries = []
    for dataset in ["__micro__"] + sorted(set(datasets)):
        mask = np.ones(len(y), dtype=bool) if dataset == "__micro__" else datasets == dataset
        summaries.append(
            pd.DataFrame(
                dict(
                    policy_id=policy_ids,
                    dataset=dataset,
                    n=int(mask.sum()),
                    quality=y[mask].mean(axis=0),
                    mean_cost_usd=c[mask].mean(axis=0),
                    median_cost_usd=np.median(c[mask], axis=0),
                )
            )
        )
    macro = (
        pd.concat(summaries[1:])
        .groupby("policy_id", sort=False)[["quality", "mean_cost_usd"]]
        .mean()
        .reindex(policy_ids)
    )
    macro["dataset"], macro["n"], macro["median_cost_usd"] = "__macro__", len(y), np.nan
    summaries.append(macro.reset_index())
    return pd.concat(summaries, ignore_index=True), y, c


def test_comparison(part, action, best_action, oracle_action):
    y, c = part.y.to_numpy(dtype=float), part.costs.to_numpy(dtype=float)
    rows = np.arange(len(y))
    samples, bootstrap = paired_bootstrap(
        part.prompts, y[rows, action], y[rows, best_action], c[rows, action], c[rows, best_action]
    )
    summary, _, _ = summarize_actions(
        part,
        np.column_stack([action, best_action, oracle_action]),
        ["router", "best_single", "oracle"],
    )
    micro = summary[summary.dataset == "__micro__"].set_index("policy_id")
    macro = summary[summary.dataset == "__macro__"].set_index("policy_id")
    by_dataset = summary[~summary.dataset.str.startswith("__")].pivot(
        index="dataset", columns="policy_id", values="quality"
    )
    delta = float(micro.loc["router", "quality"] - micro.loc["best_single", "quality"])
    savings = float(
        1 - micro.loc["router", "mean_cost_usd"] / micro.loc["best_single", "mean_cost_usd"]
    )
    oracle_gap = float(micro.loc["oracle", "quality"] - micro.loc["best_single", "quality"])
    conditions = dict(
        savings_at_least_10_percent=savings >= 0.1,
        micro_noninferiority=bootstrap["ci_95"]["quality_delta"][0] >= -0.005,
        macro_noninferiority=bootstrap["ci_95"]["macro_quality_delta"][0] >= -0.005,
        dataset_point_guard=bool(((by_dataset.router - by_dataset.best_single) >= -0.005).all()),
    )
    return samples, dict(
        quality=float(micro.loc["router", "quality"]),
        cost_usd=float(micro.loc["router", "mean_cost_usd"]),
        quality_delta=delta,
        cost_savings=savings,
        macro_quality=float(macro.loc["router", "quality"]),
        macro_quality_delta=float(
            macro.loc["router", "quality"] - macro.loc["best_single", "quality"]
        ),
        oracle_gap_recovered=delta / oracle_gap if oracle_gap > 0 else None,
        remaining_oracle_gap=float(micro.loc["oracle", "quality"] - micro.loc["router", "quality"]),
        bootstrap=bootstrap,
        conditions=conditions,
        promising_exploratory=all(conditions.values()),
    )


def evaluate_frozen(test, predictions, costs, gate, similarities, chosen, best, controls, output):
    baseline_index = MODEL_IDS.index(best.model_id)
    catalog = candidates()
    main_actions = candidate_actions(
        predictions, costs, baseline_index, similarities, gate, catalog
    )
    policy_ids = [c.candidate_id for c in catalog] + ["best_single"]
    main_actions = np.column_stack(
        [main_actions, np.full(len(test.prompts), baseline_index, dtype=np.int8)]
    )
    primary_action = main_actions[:, policy_ids.index(chosen.candidate_id)]
    refs = {
        "static_" + m: np.full(len(test.prompts), j, dtype=np.int8) for j, m in enumerate(MODEL_IDS)
    }
    refs["always_cheapest"] = np.full(len(test.prompts), cheapest(costs), dtype=np.int8)
    refs["oracle"] = oracle_actions(test, costs)
    refs["primary"] = primary_action
    for family, policy in controls.items():
        refs["control_" + family] = route(predictions[family], policy, costs)
    actions = np.column_stack([main_actions] + list(refs.values())).astype(np.int8)
    policy_ids += list(refs)
    pd.DataFrame(actions, index=test.prompts.index, columns=policy_ids).reset_index().to_parquet(
        output / "test_actions.parquet", index=False
    )
    summary, y, c = summarize_actions(test, actions, policy_ids)
    summary.to_parquet(output / "test_policy_metrics.parquet", index=False)
    frontier = summary[summary.dataset == "__micro__"].copy()
    eligible = frontier.policy_id != "oracle"
    frontier["pareto_deployable"] = False
    frontier.loc[eligible, "pareto_deployable"] = nondominated(frontier[eligible])
    frontier["pareto_with_oracle"] = nondominated(frontier)
    frontier.to_parquet(output / "pareto_table.parquet", index=False)
    comparison = {}
    for name in ["primary"] + ["control_" + family for family in FAMILIES]:
        samples, result = test_comparison(test, refs[name], main_actions[:, -1], refs["oracle"])
        samples.to_parquet(output / f"{name}_bootstrap_samples.parquet", index=False)
        comparison[name] = result
    primary_family = chosen.family if chosen.family != "static" else "embedding"
    distribution, cross = selection_tables(test, predictions[primary_family], primary_action)
    distribution.to_parquet(output / "selection_distribution.parquet", index=False)
    cross.to_parquet(output / "dataset_selection_counts.parquet", index=False)
    decisions = []
    for name in ["primary", "best_single", "always_cheapest", "oracle"] + [
        "control_" + f for f in FAMILIES
    ]:
        j = policy_ids.index(name)
        decisions.append(
            pd.DataFrame(
                dict(
                    prompt_id=test.prompts.index,
                    policy_id=name,
                    selected_model_id=np.asarray(MODEL_IDS)[actions[:, j]],
                    dataset=test.prompts.dataset.to_numpy(),
                    leakage_group=test.prompts.leakage_group.to_numpy(),
                    success=y[:, j],
                    cost_usd=c[:, j],
                )
            )
        )
    evaluation = pd.concat(decisions, ignore_index=True)
    evaluation.to_parquet(output / "test_evaluation.parquet", index=False)
    evaluation.drop(columns=["success", "cost_usd", "dataset", "leakage_group"]).to_parquet(
        output / "test_routing_decisions.parquet", index=False
    )
    datasets = {
        d: summary[summary.dataset == d]
        .set_index("policy_id")[["quality", "mean_cost_usd", "n"]]
        .loc[["primary", "best_single", "always_cheapest", "oracle"]]
        .to_dict(orient="index")
        for d in sorted(test.prompts.dataset.unique())
    }
    base = summary[summary.dataset == "__micro__"].set_index("policy_id")
    return dict(
        comparison=comparison,
        baselines=base.loc[["best_single", "always_cheapest", "oracle"]].to_dict(orient="index"),
        datasets=datasets,
        selection_distribution=distribution.to_dict(orient="records"),
        dataset_selection_counts=cross.to_dict(orient="records"),
        primary_prediction_reference_family=primary_family,
        gate_rejection_fraction=float(np.mean(similarities < gate.cutoffs[chosen.gate_quantile]))
        if chosen.gate_quantile
        else 0.0,
        primary_on_pareto_frontier=bool(
            frontier.loc[frontier.policy_id == "primary", "pareto_deployable"].iloc[0]
        ),
        frontier=frontier[frontier.pareto_deployable].to_dict(orient="records"),
    ), primary_action


def run_regime(data, root_output, reports, regime, embeddings, cache_metadata, lock):
    output = root_output / regime
    if (output / "results.json").exists():
        raise ValueError(
            "Completed regime already exists; use --report-only or a fresh output directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    train = checked_read(data, regime, "train")
    costs = train.costs.to_numpy(dtype=float).mean(axis=0)
    models, train_predictions, family_metadata = {}, {}, {}
    for family in FAMILIES:
        start = time.perf_counter()
        if family in {"tfidf", "handcrafted"}:
            model, cv, folds = train_router(
                train, family, lambda message: print(message, flush=True)
            )
        else:
            model, cv, folds = train_semantic(train, embeddings.loc[train.prompts.index], family)
        directory = output / family
        directory.mkdir(exist_ok=True)
        joblib.dump(model, directory / "models.joblib", compress=3)
        models[family] = model
        cv.to_parquet(directory / "cv_scores.parquet", index=False)
        folds.to_parquet(directory / "cv_assignments.parquet", index=False)
        prediction = predict_family(model, train, embeddings)
        pd.testing.assert_frame_equal(
            prediction,
            predict_family(joblib.load(directory / "models.joblib"), train, embeddings),
            check_exact=True,
        )
        prediction.reset_index().to_parquet(directory / "train_predictions.parquet", index=False)
        train_predictions[family] = prediction
        family_metadata[family] = dict(
            selected_C=model.selected_c,
            model_signature=model_signature(model),
            training_seconds=time.perf_counter() - start,
            metadata=model.metadata,
        )
        write_json(directory / "metadata.json", family_metadata[family])
    # Full repeated CV + fit for semantic heads verifies the new learning path.
    replay, replay_cv, _ = train_semantic(train, embeddings.loc[train.prompts.index], "embedding")
    pd.testing.assert_frame_equal(
        replay_cv, pd.read_parquet(output / "embedding/cv_scores.parquet"), check_exact=True
    )
    if model_signature(replay) != model_signature(models["embedding"]):
        raise AssertionError("Semantic training replay differs")
    del replay
    gate, train_similarity = NoveltyGate.fit(train, embeddings.loc[train.prompts.index])
    joblib.dump(gate, output / "novelty_gate.joblib", compress=3)
    pd.DataFrame(dict(prompt_id=train.prompts.index, similarity=train_similarity)).to_parquet(
        output / "train_novelty.parquet", index=False
    )
    validation = checked_read(data, regime, "validation")
    validation_predictions = {}
    controls, control_tables = {}, {}
    for family, model in models.items():
        p = predict_family(model, validation, embeddings)
        validation_predictions[family] = p
        p.reset_index().to_parquet(output / family / "validation_predictions.parquet", index=False)
        policy, _, table, _ = select_validation(validation, p, costs)
        controls[family], control_tables[family] = policy, table
        table.to_parquet(output / family / "control_validation_grid.parquet", index=False)
    validation_similarity = gate.score(embeddings.loc[validation.prompts.index].to_numpy())
    print(
        f"{regime}: computing simultaneous validation bounds before loading test outcomes",
        flush=True,
    )
    chosen, best, table, samples, selection_metadata, val_actions = conservative_select(
        validation, validation_predictions, costs, validation_similarity, gate
    )
    table.to_parquet(output / "validation_candidates.parquet", index=False)
    samples.to_parquet(output / "validation_bootstrap_max_t.parquet", index=False)
    pd.DataFrame(
        val_actions, index=validation.prompts.index, columns=table.candidate_id
    ).reset_index().to_parquet(output / "validation_actions.parquet", index=False)
    pd.DataFrame(
        dict(prompt_id=validation.prompts.index, similarity=validation_similarity)
    ).to_parquet(output / "validation_novelty.parquet", index=False)
    frozen = dict(
        primary=chosen.to_dict(),
        best_single=best.to_dict(),
        controls={f: p.to_dict() for f, p in controls.items()},
        mean_training_cost_usd=dict(zip(MODEL_IDS, costs)),
        normalized_training_cost=(costs / costs.max()).tolist(),
        novelty_cutoffs=gate.cutoffs,
        selection=selection_metadata,
        model_signatures={f: m["model_signature"] for f, m in family_metadata.items()},
        protocol_sha256=lock["protocol_sha256"],
        frozen_before_test_outcomes_loaded=True,
    )
    write_json(output / "frozen_selection.json", frozen)
    frozen_hash = sha256(output / "frozen_selection.json")
    print(
        f"{regime}: frozen {chosen.candidate_id}; feasible={selection_metadata['feasible_count']}; max-t={selection_metadata['critical_value']:.3f}",
        flush=True,
    )
    # First test outcome load for this regime. Nothing below can alter fitting or selection.
    test = checked_read(data, regime, "test")
    test_predictions, diagnostics, reliability = {}, [], []
    for family, model in models.items():
        p = predict_family(model, test, embeddings)
        test_predictions[family] = p
        p.reset_index().to_parquet(output / family / "test_predictions.parquet", index=False)
        for part, prediction in [
            (train, train_predictions[family]),
            (validation, validation_predictions[family]),
            (test, p),
        ]:
            d, rel = classifier_diagnostics(part, prediction)
            diagnostics.append(d.assign(family=family))
            reliability.append(rel.assign(family=family))
    test_similarity = gate.score(embeddings.loc[test.prompts.index].to_numpy())
    pd.DataFrame(dict(prompt_id=test.prompts.index, similarity=test_similarity)).to_parquet(
        output / "test_novelty.parquet", index=False
    )
    results, primary_action = evaluate_frozen(
        test, test_predictions, costs, gate, test_similarity, chosen, best, controls, output
    )
    diagnostics = pd.concat(diagnostics, ignore_index=True)
    reliability = pd.concat(reliability, ignore_index=True)
    diagnostics.to_parquet(output / "classifier_metrics.parquet", index=False)
    reliability.to_parquet(output / "reliability.parquet", index=False)
    import importlib.metadata

    results.update(
        regime=regime,
        primary=chosen.to_dict(),
        best_single=best.to_dict(),
        controls={f: p.to_dict() for f, p in controls.items()},
        selection_metadata=selection_metadata,
        validation_primary=table[table.candidate_id == chosen.candidate_id].iloc[0].to_dict(),
        families={
            f: {k: m[k] for k in ("selected_C", "model_signature", "training_seconds")}
            for f, m in family_metadata.items()
        },
        test_classifier_metrics=diagnostics[diagnostics.split == "test"].to_dict(orient="records"),
        split_counts={p.name: len(p.prompts) for p in (train, validation, test)},
        semantic_training_replay_exact=True,
        source_processed_data_hashes=lock["source_processed_data_hashes"],
        frozen_selection_sha256=frozen_hash,
        protocol_sha256=lock["protocol_sha256"],
        embedding_cache_metadata=cache_metadata,
        novelty_cutoffs=gate.cutoffs,
        environment=dict(
            python=sys.version,
            packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        ),
        interpretation="exploratory; existing tests inspected during v1; no v2 test-based selection",
    )
    if (
        sha256(output / "frozen_selection.json") != frozen_hash
        or verify_source(data) != lock["source_processed_data_hashes"]
    ):
        raise AssertionError("Frozen policy or data changed")
    write_json(output / "results.json", results)
    write_json(
        output / "artifact_hashes.json",
        {
            str(p.relative_to(output)): sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "artifact_hashes.json"
        },
    )
    print(
        f"{regime}: primary results {json.dumps(clean(results['comparison']['primary']))}",
        flush=True,
    )
    return clean(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/llmrouterbench")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/router_v2")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--regime", choices=["standard", "ood", "all"], default="all")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    from routing_ml.reporting_v2 import generate_report

    if args.report_only:
        generate_report(args.output_dir, args.reports_dir)
        return
    lock = campaign_lock(args.output_dir, args.data_dir)
    prompt_text = (
        pd.read_parquet(args.data_dir / "prompts.parquet", columns=["prompt_id", "prompt"])
        .sort_values("prompt_id")
        .reset_index(drop=True)
    )
    embeddings, cache_metadata = load_cache(args.output_dir / "embeddings", prompt_text)

    def blocked(*args, **kwargs):
        raise RuntimeError("Network calls are forbidden in this local campaign")

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
        for regime in ("standard", "ood"):
            if args.regime not in {regime, "all"}:
                continue
            if regime == "ood" and not (args.output_dir / "standard/results.json").exists():
                raise ValueError("Complete standard v2 before OOD")
            run_regime(
                args.data_dir,
                args.output_dir,
                args.reports_dir,
                regime,
                embeddings,
                cache_metadata,
                lock,
            )
            generate_report(args.output_dir, args.reports_dir)


if __name__ == "__main__":
    main()
