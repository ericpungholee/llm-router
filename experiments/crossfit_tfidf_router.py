#!/usr/bin/env python3
"""Run the fixed v3 grouped robustness check offline, without new tuning."""

import argparse
from contextlib import ExitStack
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.tfidf_logreg_router import clean, model_signature, sha256, verify_source, write_json
from routing_ml.bootstrap import paired_bootstrap
from routing_ml.crossfit import (MARGINS, fit_fixed_router, fold_ids, freeze_references,
                                 frozen_actions, partition_assignments, read_part)
from routing_ml.metrics import classifier_diagnostics, oracle_actions, outcome_summary, pareto_table, selection_tables
from routing_ml.training import MODEL_IDS, SEED, digest


def start_campaign(output, data):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Campaign directory is not empty; choose a new --output-dir")
    sources = verify_source(data)
    output.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / "reports/router_v3_robustness_spec.md", Path(__file__).resolve()]
    paths += sorted((ROOT / "routing_ml").glob("*.py"))
    paths += [ROOT / "experiments/tfidf_logreg_router.py", ROOT / "experiments/requirements-router-v1.txt"]
    source_hashes = {}
    for p in paths:
        name = str(p.relative_to(ROOT))
        source_hashes[name] = sha256(p)
        destination = output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, destination)
    lock = dict(seed=SEED, source_processed_data_hashes=sources, source_hashes=source_hashes,
                environment=dict(python=sys.version, packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}),
                previous_report_hashes={str(p.relative_to(ROOT)): sha256(p) for version in (1, 2)
                                        for p in sorted((ROOT / "reports").glob(f"router_v{version}*")) if p.is_file()},
                interpretation="descriptive reused-benchmark cross-fitting; not independent confirmation")
    write_json(output / "campaign_lock.json", lock)
    return lock


def run_fold(data, output, prompts, assignments, regime, fold):
    path = output / regime / f"fold_{fold}"
    path.mkdir(parents=True)
    ids = {name: fold_ids(assignments, fold, regime, name) for name in ("train", "validation", "test")}
    train = read_part(data, prompts, ids["train"], regime, "train")
    validation = read_part(data, prompts, ids["validation"], regime, "validation")
    start = time.perf_counter()
    model = fit_fixed_router(train)
    seconds = time.perf_counter() - start
    train_prediction = model.predict(train)
    signature = model_signature(model)
    replay_exact = None
    if fold == 0:
        replay = fit_fixed_router(train)
        pd.testing.assert_frame_equal(train_prediction, replay.predict(train), check_exact=True)
        if signature != model_signature(replay):
            raise AssertionError("Fixed training replay differs")
        replay_exact = True
    joblib.dump(model, path / "models.joblib", compress=3)
    pd.testing.assert_frame_equal(train_prediction, joblib.load(path / "models.joblib").predict(train), check_exact=True)
    for name, part in (("train", train), ("validation", validation)):
        model.predict(part).reset_index().to_parquet(path / f"{name}_predictions.parquet", index=False)
    frozen = freeze_references(train, validation)
    frozen.update(fold=fold, split_counts={k: len(v) for k, v in ids.items()},
                  model_signature=signature, training_seconds=seconds, training_replay_exact=replay_exact,
                  training_metadata=model.metadata, decisions_saved_before_evaluation_labels=True)
    write_json(path / "frozen_policy.json", frozen)
    frozen_hash = sha256(path / "frozen_policy.json")
    # Label-free prediction and routing interface. No evaluation outcomes exist here.
    request = SimpleNamespace(regime=regime, prompts=prompts.loc[ids["test"]])
    prediction = model.predict(request)
    prediction.reset_index().to_parquet(path / "test_predictions.parquet", index=False)
    actions = frozen_actions(prediction, request.prompts.dataset.to_numpy(), frozen)
    decisions = pd.DataFrame({p: np.asarray(MODEL_IDS)[a] for p, a in actions.items()}, index=prediction.index)
    decisions.reset_index().to_parquet(path / "test_decisions.parquet", index=False)
    test = read_part(data, prompts, ids["test"], regime, "test")
    costs = np.array([frozen["mean_training_cost_usd"][m] for m in MODEL_IDS])
    actions["oracle"] = oracle_actions(test, costs)
    evaluation = []
    for policy_id, action in actions.items():
        _, y, c = outcome_summary(test, action)
        evaluation.append(pd.DataFrame(dict(prompt_id=test.prompts.index, fold=fold, policy_id=policy_id,
                                           dataset=test.prompts.dataset.to_numpy(), leakage_group=test.prompts.leakage_group.to_numpy(),
                                           selected_model_id=np.asarray(MODEL_IDS)[action], success=y, cost_usd=c)))
    pd.concat(evaluation, ignore_index=True).to_parquet(path / "test_evaluation.parquet", index=False)
    if sha256(path / "frozen_policy.json") != frozen_hash:
        raise AssertionError("Frozen reference changed during evaluation")
    print(f"{regime} fold {fold + 1}/5: fit {len(train.prompts)}, validation {len(validation.prompts)}, "
          f"held out {len(test.prompts)}, best={frozen['best_single']}; replay={replay_exact}", flush=True)


def summarize_rows(evaluation):
    records = []
    for pid, values in evaluation.groupby("policy_id", sort=True):
        domains = values.groupby("dataset").agg(quality=("success", "mean"), mean_cost_usd=("cost_usd", "mean"), n=("success", "size"))
        records.append(dict(policy_id=pid, dataset="__micro__", quality=float(values.success.mean()),
                            mean_cost_usd=float(values.cost_usd.mean()), n=len(values)))
        records.extend(dict(policy_id=pid, dataset=d, **row) for d, row in domains.to_dict(orient="index").items())
        records.append(dict(policy_id=pid, dataset="__macro__", quality=float(domains.quality.mean()),
                            mean_cost_usd=float(domains.mean_cost_usd.mean()), n=len(values)))
    return pd.DataFrame(records)


def comparison(evaluation, reference, output, label):
    router = evaluation[evaluation.policy_id == "router"].sort_values("prompt_id").set_index("prompt_id")
    baseline = evaluation[evaluation.policy_id == reference].sort_values("prompt_id").set_index("prompt_id")
    if not router.index.equals(baseline.index):
        raise ValueError("Evaluation identity mismatch")
    samples, bootstrap = paired_bootstrap(router, router.success, baseline.success, router.cost_usd, baseline.cost_usd)
    samples.to_parquet(output / f"{label}_bootstrap_samples.parquet", index=False)
    bootstrap.update(interpretation="descriptive conditional intervals; overlapping fold fits and prior benchmark adaptation excluded",
                     independent_confirmation=False)
    return dict(quality_delta=float((router.success - baseline.success).mean()),
                cost_savings=float(1 - router.cost_usd.mean() / baseline.cost_usd.mean()), bootstrap=bootstrap)


def summarize_regime(data, output, prompts, assignments, regime):
    path = output / regime
    evaluation = pd.concat([pd.read_parquet(path / f"fold_{k}/test_evaluation.parquet") for k in range(5)], ignore_index=True)
    evaluation = evaluation.sort_values(["policy_id", "prompt_id"]).reset_index(drop=True)
    prediction = pd.concat([pd.read_parquet(path / f"fold_{k}/test_predictions.parquet") for k in range(5)]).set_index("prompt_id").sort_index()
    expected_ids = prompts.index if regime == "standard" else prompts[prompts.dataset == "livecodebench"].index
    if not prediction.index.equals(expected_ids) or prediction.index.has_duplicates:
        raise ValueError("Out-of-fold coverage failure")
    prediction.reset_index().to_parquet(path / "oof_predictions.parquet", index=False)
    evaluation.to_parquet(path / "oof_evaluation.parquet", index=False)
    summary = summarize_rows(evaluation)
    summary.to_parquet(path / "summary.parquet", index=False)
    frontier = pareto_table(summary)
    frontier.to_parquet(path / "pareto.parquet", index=False)
    micro = summary[summary.dataset == "__micro__"].set_index("policy_id")
    macro = summary[summary.dataset == "__macro__"].set_index("policy_id")
    comp = comparison(evaluation, "best_single", path, "primary")
    domain = comparison(evaluation, "privileged_domain_static", path, "domain_reference")
    per_dataset = []
    for dataset in sorted(evaluation.dataset.unique()):
        e = evaluation[evaluation.dataset == dataset]
        dcomp = comparison(e, "best_single", path, "dataset_" + dataset)
        s = summary[(summary.dataset == dataset)].set_index("policy_id")
        per_dataset.append(dict(dataset=dataset, n=int(s.loc["router", "n"]),
                                router=s.loc["router"].to_dict(), best_single=s.loc["best_single"].to_dict(), **dcomp))
    folds = []
    for k in range(5):
        s = summarize_rows(evaluation[evaluation.fold == k])
        m = s[s.dataset == "__micro__"].set_index("policy_id")
        frozen = json.loads((path / f"fold_{k}/frozen_policy.json").read_text())
        folds.append(dict(fold=k, best_single=frozen["best_single"], split_counts=frozen["split_counts"],
                          training_seconds=frozen["training_seconds"], vocabulary_hash=frozen["training_metadata"]["vocabulary_hash"],
                          quality_delta=float(m.loc["router", "quality"] - m.loc["best_single", "quality"]),
                          cost_savings=float(1 - m.loc["router", "mean_cost_usd"] / m.loc["best_single", "mean_cost_usd"]),
                          router=m.loc["router"].to_dict(), baseline=m.loc["best_single"].to_dict()))
    test = read_part(data, prompts, prediction.index, regime, "test")
    selected = evaluation[evaluation.policy_id == "router"].set_index("prompt_id").loc[prediction.index]
    action = np.array([MODEL_IDS.index(m) for m in selected.selected_model_id])
    diagnostics, reliability = classifier_diagnostics(test, prediction, action)
    diagnostics.to_parquet(path / "classifier_metrics.parquet", index=False)
    reliability.to_parquet(path / "reliability.parquet", index=False)
    distribution, counts = selection_tables(test, prediction, action)
    distribution.to_parquet(path / "selection_distribution.parquet", index=False)
    counts.to_parquet(path / "dataset_model_counts.parquet", index=False)
    excluded = []
    for names in (("simpleqa",), ("simpleqa", "mmlupro")):
        e = evaluation[~evaluation.dataset.isin(names)]
        s = summarize_rows(e)
        m = s[s.dataset == "__micro__"].set_index("policy_id")
        excluded.append(dict(excluded=list(names), n=int(m.loc["router", "n"]),
                             quality_delta=float(m.loc["router", "quality"] - m.loc["best_single", "quality"]),
                             cost_savings=float(1 - m.loc["router", "mean_cost_usd"] / m.loc["best_single", "mean_cost_usd"])))
    denominator = float(micro.loc["oracle", "quality"] - micro.loc["best_single", "quality"])
    ci = comp["bootstrap"]["ci_95"]
    checks = dict(savings_at_least_10_percent=comp["cost_savings"] >= 0.1,
                  micro_interval_margin=ci["quality_delta"][0] >= -0.005,
                  macro_interval_margin=ci["macro_quality_delta"][0] >= -0.005,
                  dataset_point_guard=all(d["quality_delta"] >= -0.005 for d in per_dataset))
    result = dict(regime=regime, fixed_C=1.0, margin=MARGINS[regime], n=len(prediction),
                  micro=micro.to_dict(orient="index"), macro=macro.to_dict(orient="index"),
                  primary=comp, privileged_reference_comparison=domain, per_dataset=per_dataset,
                  folds=folds, fold_checks=dict(quality_margin_passes=sum(f["quality_delta"] >= -0.005 for f in folds),
                                               savings_10_percent_passes=sum(f["cost_savings"] >= .1 for f in folds)),
                  descriptive_checks=checks, all_descriptive_checks_pass=all(checks.values()),
                  oracle_gap_recovered=comp["quality_delta"] / denominator if denominator else None,
                  remaining_oracle_gap=float(micro.loc["oracle", "quality"] - micro.loc["router", "quality"]),
                  router_on_pareto=bool(frontier.set_index("policy_id").loc["router", "pareto_deployable"]),
                  classifier_metrics=diagnostics.to_dict(orient="records"), distribution=distribution.to_dict(orient="records"),
                  dataset_model_counts=counts.to_dict(orient="records"), composition_checks=excluded)
    write_json(path / "results.json", result)
    print(f"{regime}: quality={micro.loc['router', 'quality']:.6f}, delta={comp['quality_delta']:.6f}, "
          f"savings={comp['cost_savings']:.6f}; descriptive checks={checks}", flush=True)
    return clean(result)


def matched_code(output):
    frames = []
    for regime in ("standard", "ood"):
        e = pd.read_parquet(output / regime / "oof_evaluation.parquet")
        frames.append(e[(e.policy_id == "router") & (e.dataset == "livecodebench")].set_index("prompt_id").sort_index())
    standard, ood = frames
    if not standard.index.equals(ood.index) or not standard.fold.equals(ood.fold):
        raise ValueError("Matched code coverage mismatch")
    samples, bootstrap = paired_bootstrap(ood, ood.success, standard.success, ood.cost_usd, standard.cost_usd)
    samples.to_parquet(output / "matched_code_bootstrap_samples.parquet", index=False)
    return dict(n=len(ood), standard_quality=float(standard.success.mean()), ood_quality=float(ood.success.mean()),
                quality_delta_ood_minus_standard=float((ood.success - standard.success).mean()),
                cost_savings_ood_vs_standard=float(1 - ood.cost_usd.mean() / standard.cost_usd.mean()),
                bootstrap=bootstrap, interpretation="matched prompts; independently fitted systems with different training domains")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/llmrouterbench")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/router_v3")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    from routing_ml.reporting_v3 import generate_report
    if args.report_only:
        generate_report(args.output_dir, args.reports_dir)
        return
    def blocked(*args, **kwargs):
        raise RuntimeError("Network/provider calls are forbidden in the robustness check")
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection", "socket.getaddrinfo", "urllib.request.urlopen"):
            stack.enter_context(patch(target, side_effect=blocked))
        stack.enter_context(threadpool_limits(limits=1))
        lock = start_campaign(args.output_dir, args.data_dir)
        prompts = pd.read_parquet(args.data_dir / "prompts.parquet").set_index("prompt_id").sort_index()
        assignments = partition_assignments(prompts)
        assignments.to_parquet(args.output_dir / "assignments.parquet", index=False)
        lock["assignments_sha256"] = sha256(args.output_dir / "assignments.parquet")
        write_json(args.output_dir / "campaign_lock.json", lock)
        results = {}
        for regime in ("standard", "ood"):
            for fold in range(5):
                run_fold(args.data_dir, args.output_dir, prompts, assignments, regime, fold)
            results[regime] = summarize_regime(args.data_dir, args.output_dir, prompts, assignments, regime)
        if verify_source(args.data_dir) != lock["source_processed_data_hashes"]:
            raise AssertionError("Processed source changed")
        for name, expected in lock["previous_report_hashes"].items():
            if sha256(ROOT / name) != expected:
                raise AssertionError("Previous experiment report changed")
        report = dict(regimes=results, matched_code=matched_code(args.output_dir),
                      seed=SEED, provenance=lock, independent_confirmation=False,
                      previous_reports_and_processed_data_unchanged=True)
        write_json(args.output_dir / "results.json", report)
        write_json(args.output_dir / "artifact_hashes.json", {str(p.relative_to(args.output_dir)): sha256(p)
                   for p in sorted(args.output_dir.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"})
        generate_report(args.output_dir, args.reports_dir)


if __name__ == "__main__":
    main()
