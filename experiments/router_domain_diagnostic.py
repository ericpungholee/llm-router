#!/usr/bin/env python3
"""Post-hoc diagnostic only: validation-selected domain-static reference.

This privileged-metadata reference is not eligible for v2 primary selection.
It does not change, refit, or replace any frozen router.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from experiments.embedding_logreg_router import summarize_actions
from experiments.tfidf_logreg_router import write_json
from routing_ml.bootstrap import paired_bootstrap
from routing_ml.training import MODEL_IDS, read_split


def select_domain_models(validation, costs):
    validation.require("validation")
    choices = {}
    for dataset in sorted(validation.prompts.dataset.unique()):
        means = validation.y.loc[validation.prompts.dataset == dataset].mean()
        choices[dataset] = min(MODEL_IDS, key=lambda m: (-means[m], costs[MODEL_IDS.index(m)], m))
    global_means = validation.y.mean()
    fallback = min(MODEL_IDS, key=lambda m: (-global_means[m], costs[MODEL_IDS.index(m)], m))
    return choices, fallback


def main():
    data = ROOT / "data/processed/llmrouterbench"
    root = ROOT / "artifacts/router_v2"
    report = dict(status="post-hoc diagnostic; not preregistered primary or deployment candidate",
                  privileged_input="dataset name", selection_split="validation",
                  unseen_dataset_rule="validation-selected global best-single", regimes={})
    for regime in ("standard", "ood"):
        train, val = read_split(data, regime, "train"), read_split(data, regime, "validation")
        costs = train.costs.to_numpy(dtype=float).mean(axis=0)
        choices, fallback = select_domain_models(val, costs)
        # Domain model IDs are frozen from validation before loading test outcomes.
        test = read_split(data, regime, "test")
        domain = np.asarray([MODEL_IDS.index(choices.get(d, fallback)) for d in test.prompts.dataset])
        actions = pd.read_parquet(root / regime / "test_actions.parquet").set_index("prompt_id")
        if not actions.index.equals(test.prompts.index):
            raise ValueError("Prompt ID alignment failure")
        primary = actions.primary.to_numpy()
        summary, y, c = summarize_actions(test, np.column_stack([primary, domain]), ["frozen_primary", "privileged_domain_static"])
        samples, bootstrap = paired_bootstrap(test.prompts, y[:, 0], y[:, 1], c[:, 0], c[:, 1])
        micro = summary[summary.dataset == "__micro__"].set_index("policy_id")
        report["regimes"][regime] = dict(selected_domain_models=choices, fallback=fallback,
                                        metrics=summary.to_dict(orient="records"), bootstrap=bootstrap,
                                        primary_minus_domain_quality=float(micro.loc["frozen_primary", "quality"] - micro.loc["privileged_domain_static", "quality"]),
                                        primary_cost_savings_vs_domain=float(1 - micro.loc["frozen_primary", "mean_cost_usd"] / micro.loc["privileged_domain_static", "mean_cost_usd"]))
    write_json(ROOT / "reports/router_v2_domain_diagnostic.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
