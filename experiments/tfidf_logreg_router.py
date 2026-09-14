#!/usr/bin/env python3
"""Run the frozen offline router experiment; no preprocessing or provider calls."""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from routing_ml.bootstrap import paired_bootstrap
from routing_ml.metrics import classifier_diagnostics, envelopes, evaluate_policies, outcome_summary, pareto_table, selection_tables
from routing_ml.policies import estimate_costs, route, select_validation
from routing_ml.reporting import figures, write_report
from routing_ml.training import MODEL_IDS, SEED, digest, read_split, train_router

COUNTS = {"standard": {"train": 6094, "validation": 1307, "test": 1305},
          "ood": {"train": 6503, "validation": 1148, "test": 1055}}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_source(directory):
    provenance = json.loads((directory / "provenance.json").read_text())
    names = ["prompts.parquet", "outcomes.parquet", "splits.parquet", "prompt_features.parquet", "static_model_summary.parquet"]
    hashes = {name: sha256(directory / name) for name in names}
    for name in names:
        if hashes[name] != provenance["artifacts"][name]["sha256"]:
            raise ValueError(f"Frozen artifact hash mismatch: {name}")
    for name, expected in provenance["config_sha256"].items():
        hashes[name] = sha256(directory / name)
        if hashes[name] != expected:
            raise ValueError(f"Frozen config hash mismatch: {name}")
    pool = json.loads((directory / "router_model_pool.json").read_text())
    if tuple(m["model_id"] for m in pool["models"]) != MODEL_IDS:
        raise ValueError("Frozen candidate order mismatch")
    # Validate persisted assignments, without using outcomes to generate or revise splits.
    prompts = pd.read_parquet(directory / "prompts.parquet", columns=["prompt_id", "leakage_group", "standard_split", "ood_split"])
    splits = pd.read_parquet(directory / "splits.parquet")
    pd.testing.assert_frame_equal(prompts.set_index("prompt_id").sort_index(), splits.set_index("prompt_id").sort_index()[prompts.columns[1:]])
    for regime in COUNTS:
        if prompts.groupby("leakage_group")[regime + "_split"].nunique().max() > 1:
            raise ValueError("Group leakage across persisted splits")
        if prompts[regime + "_split"].value_counts().to_dict() != COUNTS[regime]:
            raise ValueError("Frozen split counts mismatch")
    hashes["provenance.json"] = sha256(directory / "provenance.json")
    return hashes


def model_signature(model):
    h = hashlib.sha256()
    h.update(digest(model.metadata).encode())
    prep = model.preprocessor
    for array in ([prep.idf_] if model.kind == "tfidf" else [prep.mean_, prep.scale_]):
        h.update(np.asarray(array, dtype="<f8").tobytes())
    for predictor in model.predictors:
        if hasattr(predictor, "coef_"):
            h.update(np.asarray(predictor.coef_, dtype="<f8").tobytes())
            h.update(np.asarray(predictor.intercept_, dtype="<f8").tobytes())
        else:
            h.update(str(predictor.prevalence).encode())
    return h.hexdigest()


def checked_read(directory, regime, split):
    part = read_split(directory, regime, split)
    if len(part.prompts) != COUNTS[regime][split]:
        raise ValueError("Frozen split size mismatch")
    return part


def run_one(directory, output, reports, regime, kind):
    output.mkdir(parents=True, exist_ok=True)
    progress = lambda message: print(message, flush=True)
    sources = verify_source(directory)
    train = checked_read(directory, regime, "train")
    start = time.perf_counter()
    model, cv, assignments = train_router(train, kind, progress)
    training_seconds = time.perf_counter() - start
    train_prediction = model.predict(train)
    signature = model_signature(model)
    replay_seconds = 0.0
    replay_verified = False
    if regime == "standard" and kind == "tfidf":
        progress("standard/tfidf: repeating complete train-only CV and fit to verify determinism")
        start = time.perf_counter()
        replay, replay_cv, replay_assignments = train_router(train, kind, progress)
        pd.testing.assert_frame_equal(cv, replay_cv, check_exact=True)
        pd.testing.assert_frame_equal(assignments, replay_assignments, check_exact=True)
        pd.testing.assert_frame_equal(train_prediction, replay.predict(train), check_exact=True)
        if signature != model_signature(replay):
            raise AssertionError("Training replay changed fitted parameters")
        replay_verified = True
        replay_seconds = time.perf_counter() - start
        del replay
    joblib.dump(model, output / "models.joblib", compress=3)
    pd.testing.assert_frame_equal(train_prediction, joblib.load(output / "models.joblib").predict(train), check_exact=True)
    cv.to_parquet(output / "cv_scores.parquet", index=False)
    assignments.to_parquet(output / "cv_assignments.parquet", index=False)
    train_prediction.reset_index().to_parquet(output / "train_predictions.parquet", index=False)
    costs, normalized = estimate_costs(train)
    validation = checked_read(directory, regime, "validation")
    val_prediction = model.predict(validation)
    val_prediction.reset_index().to_parquet(output / "validation_predictions.parquet", index=False)
    selected, best, val_grid, budgets = select_validation(validation, val_prediction, costs)
    val_grid.to_parquet(output / "validation_policy_grid.parquet", index=False)
    frozen = dict(selected_policy=selected.to_dict(), best_single=best.to_dict(), budget_policies=budgets,
                  mean_training_cost_usd=dict(zip(MODEL_IDS, costs)), normalized_training_cost=dict(zip(MODEL_IDS, normalized)),
                  selection_split="validation", frozen_before_test_read=True, test_used_for_selection=False)
    write_json(output / "frozen_policy.json", frozen)
    frozen_hash = sha256(output / "frozen_policy.json")
    progress(f"{regime}/{kind}: selected C={model.selected_c:g}, validation policy={selected.policy_id}; frozen before test load")
    # This is the first test load; all training and validation decisions are already persisted.
    test = checked_read(directory, regime, "test")
    start = time.perf_counter()
    test_prediction = model.predict(test)
    prediction_seconds = time.perf_counter() - start
    test_prediction.reset_index().to_parquet(output / "test_predictions.parquet", index=False)
    start = time.perf_counter()
    route(test_prediction, selected, costs)
    routing_seconds = time.perf_counter() - start
    diagnostics, reliability, static = [], [], []
    test_evaluation = test_summary = test_actions = None
    for part, prediction in [(train, train_prediction), (validation, val_prediction), (test, test_prediction)]:
        action = route(prediction, selected, costs)
        d, rel = classifier_diagnostics(part, prediction, action)
        diagnostics.append(d)
        reliability.append(rel)
        for j, model_id in enumerate(MODEL_IDS):
            summary, _, _ = outcome_summary(part, np.full(len(part.prompts), j))
            static.extend(dict(split=part.name, model_id=model_id, **row) for row in summary)
        if part.name != "train":
            summary, decisions, evaluation, actions = evaluate_policies(part, prediction, costs, selected, best)
            summary.to_parquet(output / f"{part.name}_policy_metrics.parquet", index=False)
            decisions.to_parquet(output / f"{part.name}_routing_decisions.parquet", index=False)
            evaluation.to_parquet(output / f"{part.name}_evaluation.parquet", index=False)
            if part.name == "test":
                test_summary, test_evaluation, test_actions = summary, evaluation, actions
    diagnostics = pd.concat(diagnostics, ignore_index=True)
    reliability = pd.concat(reliability, ignore_index=True)
    diagnostics.to_parquet(output / "classifier_metrics.parquet", index=False)
    reliability.to_parquet(output / "reliability.parquet", index=False)
    pd.DataFrame(static).to_parquet(output / "static_model_summary.parquet", index=False)
    frontier = pareto_table(test_summary)
    frontier.to_parquet(output / "pareto_table.parquet", index=False)
    selected_rows = test_evaluation[test_evaluation.policy_id == "deployable_router"]
    best_rows = test_evaluation[test_evaluation.policy_id == "best_single"]
    samples, bootstrap = paired_bootstrap(test.prompts, selected_rows.success, best_rows.success, selected_rows.cost_usd, best_rows.cost_usd)
    samples.to_parquet(output / "bootstrap_samples.parquet", index=False)
    distribution, cross = selection_tables(test, test_prediction, test_actions["deployable_router"])
    distribution.to_parquet(output / "selection_distribution.parquet", index=False)
    cross.to_parquet(output / "dataset_selection_counts.parquet", index=False)
    def aggregate(dataset):
        return test_summary[test_summary.dataset == dataset].drop(columns=["split", "dataset"]).set_index("policy_id").to_dict(orient="index")
    micro, macro = aggregate("__micro__"), aggregate("__macro__")
    def comparison(rows):
        r, b = rows["deployable_router"], rows["best_single"]
        return r["quality"] - b["quality"], 1 - r["mean_cost_usd"] / b["mean_cost_usd"]
    delta, savings = comparison(micro)
    macro_delta, macro_savings = comparison(macro)
    oracle_gap = micro["oracle"]["quality"] - micro["best_single"]["quality"]
    code_paths = sorted((ROOT / "routing_ml").glob("*.py")) + [Path(__file__), ROOT / "experiments/requirements-router-v1.txt"]
    metadata = dict(**model.metadata, regime=regime, feature_kind=kind, model_signature=signature,
                    source_processed_data_hashes=sources, spec_sha256=sha256(ROOT / "reports/ml_experiment_spec.md"),
                    code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in code_paths},
                    environment=dict(python=sys.version, platform=platform.platform(), machine=platform.machine(),
                                     packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                                     threadpool_info=threadpool_info()),
                    frozen_policy=frozen, frozen_policy_sha256=frozen_hash,
                    determinism_full_training_replay_verified=replay_verified, model_reload_exact=True,
                    no_network_calls=True, split_counts=COUNTS[regime], bootstrap=bootstrap)
    write_json(output / "metadata.json", metadata)
    prefix = regime if kind == "tfidf" else regime + "_handcrafted"
    figures(frontier, reliability, diagnostics, reports / "figures", prefix)
    oracle_rows = test_evaluation[test_evaluation.policy_id == "oracle"]
    result = dict(regime=regime, kind=kind, selected_C=model.selected_c, selected_policy=selected.to_dict(), best_single=best.to_dict(),
                  split_counts=COUNTS[regime], test=micro, macro=macro,
                  datasets={d: aggregate(d) for d in sorted(test.prompts.dataset.unique())},
                  comparison=dict(quality_delta=delta, cost_savings=savings, macro_quality_delta=macro_delta,
                                  macro_cost_savings=macro_savings, oracle_gap_recovered=delta / oracle_gap if oracle_gap > 0 else None,
                                  remaining_oracle_gap=micro["oracle"]["quality"] - micro["deployable_router"]["quality"]),
                  oracle_conditional_cost_usd=float(oracle_rows.loc[oracle_rows.success == 1, "cost_usd"].mean()),
                  validation_best_quality=float(val_grid.loc[val_grid.policy_id == "best_single", "quality"].iloc[0]),
                  validation_selected=val_grid[val_grid.policy_id == selected.policy_id].iloc[0].to_dict(),
                  validation_policy_grid=val_grid.to_dict(orient="records"),
                  cv_mean_per_model_log_loss=model.metadata["cv_mean_per_model_log_loss"],
                  bootstrap=bootstrap, test_classifier_metrics=diagnostics[diagnostics.split == "test"].to_dict(orient="records"),
                  selection_distribution=distribution.to_dict(orient="records"), dataset_selection_counts=cross.to_dict(orient="records"),
                  pareto_frontier=frontier[frontier.pareto_deployable].to_dict(orient="records"),
                  envelopes=envelopes(test_summary, best, costs, budgets), figure_prefix=prefix,
                  vocabulary_size=model.metadata["vocabulary_size"], model_signature=signature,
                  determinism_full_training_replay_verified=replay_verified,
                  source_processed_data_hashes=sources, spec_sha256=metadata["spec_sha256"],
                  timing=dict(training_seconds=training_seconds, determinism_replay_seconds=replay_seconds,
                              test_prediction_ms_per_prompt=prediction_seconds * 1000 / len(test.prompts),
                              test_routing_ms_per_prompt=routing_seconds * 1000 / len(test.prompts)))
    if verify_source(directory) != sources or sha256(output / "frozen_policy.json") != frozen_hash:
        raise AssertionError("Frozen inputs/policy changed during evaluation")
    write_json(output / "results.json", result)
    write_json(output / "artifact_hashes.json", {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "artifact_hashes.json"})
    progress(f"{regime}/{kind}: test quality={micro['deployable_router']['quality']:.6f}, savings={savings:.6f}, delta={delta:.6f}, CI={bootstrap['ci_95']['quality_delta']}")
    return clean(result)


def update_reports(output, reports):
    results = {}
    for regime in COUNTS:
        for kind in ("tfidf", "handcrafted"):
            path = output / regime / ("handcrafted" if kind == "handcrafted" else "") / "results.json"
            if path.exists():
                results[regime + "_" + kind] = json.loads(path.read_text())
    if "standard_tfidf" in results and "ood_tfidf" in results:
        a, b = results["standard_tfidf"], results["ood_tfidf"]
        standard = a["datasets"]["livecodebench"]["deployable_router"]
        ood = b["test"]["deployable_router"]
        se = pd.read_parquet(output / "standard/test_evaluation.parquet")
        oe = pd.read_parquet(output / "ood/test_evaluation.parquet")
        se = se[(se.policy_id == "deployable_router") & (se.dataset == "livecodebench")]
        oe = oe[(oe.policy_id == "deployable_router") & oe.prompt_id.isin(se.prompt_id)]
        results["ood_comparison"] = dict(standard_code_quality=standard["quality"], standard_code_n=standard["n"],
                                         ood_code_quality=ood["quality"], ood_code_n=ood["n"],
                                         standard_code_cost_usd=standard["mean_cost_usd"], ood_code_cost_usd=ood["mean_cost_usd"],
                                         ood_minus_standard_quality=ood["quality"] - standard["quality"],
                                         common_code_n=len(oe), ood_common_quality=float(oe.success.mean()),
                                         ood_common_minus_standard_quality=float(oe.success.mean() - se.success.mean()))
    reports.mkdir(parents=True, exist_ok=True)
    write_json(reports / "router_v1_results.json", results)
    write_report(results, reports / "router_v1_results.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/llmrouterbench")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/router_v1")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--regime", choices=["standard", "ood", "all"], default="all")
    parser.add_argument("--features", choices=["tfidf", "handcrafted", "all"], default="all")
    args = parser.parse_args()
    def blocked(*args, **kwargs):
        raise RuntimeError("Network calls are forbidden in this local experiment")
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection", "socket.getaddrinfo", "urllib.request.urlopen"):
            stack.enter_context(patch(target, side_effect=blocked))
        stack.enter_context(threadpool_limits(limits=1))
        for kind in ("tfidf", "handcrafted"):
            if args.features not in {kind, "all"}:
                continue
            for regime in COUNTS:
                if args.regime not in {regime, "all"}:
                    continue
                if regime == "ood" or kind == "handcrafted":
                    prerequisite = args.output_dir / "standard/results.json"
                    if not prerequisite.exists() or not json.loads(prerequisite.read_text())["determinism_full_training_replay_verified"]:
                        raise ValueError("Complete deterministic standard TF-IDF experiment first")
                out = args.output_dir / regime
                if kind == "handcrafted":
                    out = out / "handcrafted"
                run_one(args.data_dir, out, args.reports_dir, regime, kind)
                update_reports(args.output_dir, args.reports_dir)


if __name__ == "__main__":
    main()
