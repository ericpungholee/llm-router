"""Outcome evaluation is separate from predictions and deployable policy selection."""

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

from routing_ml.policies import Policy, cheapest, grid, route, validate_predictions
from routing_ml.training import MODEL_IDS, clipped_log_loss


def calibration(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = np.minimum((p * 10).astype(int), 9)
    rows = []
    for b in range(10):
        mask = bins == b
        if mask.any():
            rows.append(
                dict(
                    bin=b,
                    left=b / 10,
                    right=(b + 1) / 10,
                    count=int(mask.sum()),
                    mean_probability=float(p[mask].mean()),
                    observed_success=float(y[mask].mean()),
                )
            )
    reliability = pd.DataFrame(rows)
    ece = sum(r["count"] * abs(r["mean_probability"] - r["observed_success"]) for r in rows) / len(
        y
    )
    return dict(
        brier=float(np.mean((p - y) ** 2)), log_loss=clipped_log_loss(y, p), ece=float(ece)
    ), reliability


def binary_metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    cal, rel = calibration(y, p)
    both_classes = len(np.unique(y)) == 2
    precision, recall, _ = precision_recall_curve(y, p) if both_classes else (None, None, None)
    return dict(
        prevalence=float(y.mean()),
        roc_auc=float(roc_auc_score(y, p)) if both_classes else None,
        pr_auc=float(average_precision_score(y, p)) if y.sum() else None,
        pr_auc_trapezoid=float(auc(recall, precision)) if both_classes else None,
        accuracy_at_0_5=float(np.mean((p >= 0.5) == y)),
        **cal,
    ), rel


def classifier_diagnostics(part, prediction, selected=None):
    validate_predictions(part, prediction)
    y, p = part.y.to_numpy(dtype=float), prediction.to_numpy()
    metrics, reliabilities = [], []
    for j, model_id in enumerate(MODEL_IDS):
        scores, rel = binary_metrics(y[:, j], p[:, j])
        metrics.append(dict(split=part.name, model_id=model_id, **scores))
        reliabilities.append(rel.assign(split=part.name, model_id=model_id))
    macro = pd.DataFrame(metrics).drop(columns=["split", "model_id"]).mean().to_dict()
    metrics.append(dict(split=part.name, model_id="__macro__", **macro))
    if selected is not None:
        rows = np.arange(len(y))
        scores, rel = binary_metrics(y[rows, selected], p[rows, selected])
        metrics.append(dict(split=part.name, model_id="__selected_action__", **scores))
        reliabilities.append(rel.assign(split=part.name, model_id="__selected_action__"))
    return pd.DataFrame(metrics), pd.concat(reliabilities, ignore_index=True)


def outcome_summary(part, selected):
    rows = np.arange(len(part.prompts))
    y = part.y.to_numpy(dtype=float)[rows, selected]
    costs = part.costs.to_numpy(dtype=float)[rows, selected]
    datasets = part.prompts.dataset.to_numpy()
    records = []
    for dataset in ["__micro__"] + sorted(set(datasets)):
        mask = np.ones(len(rows), dtype=bool) if dataset == "__micro__" else datasets == dataset
        records.append(
            dict(
                dataset=dataset,
                n=int(mask.sum()),
                quality=float(y[mask].mean()),
                mean_cost_usd=float(costs[mask].mean()),
                median_cost_usd=float(np.median(costs[mask])),
                coverage=1.0,
            )
        )
    macro = pd.DataFrame(records[1:])[["quality", "mean_cost_usd"]].mean().to_dict()
    records.append(
        dict(dataset="__macro__", n=len(rows), median_cost_usd=None, coverage=1.0, **macro)
    )
    return records, y, costs


def oracle_actions(part, costs):
    y, c = part.y.to_numpy(dtype=float), part.costs.to_numpy(dtype=float)
    ids = np.broadcast_to(np.asarray(MODEL_IDS), c.shape)
    successful = np.lexsort((ids, c, y != 1), axis=1)[:, 0]
    return np.where(y.any(axis=1), successful, cheapest(costs))


def evaluate_policies(part, prediction, costs, selected, best):
    validate_predictions(part, prediction)
    policies = grid() + [Policy("static_" + m, "static", model_id=m) for m in MODEL_IDS]
    policies += [
        best,
        Policy("always_cheapest", "static", model_id=MODEL_IDS[cheapest(costs)]),
        Policy("deployable_router", selected.kind, selected.value, selected.model_id),
    ]
    actions = {p.policy_id: route(prediction, p, costs) for p in policies}
    actions["oracle"] = oracle_actions(part, costs)
    c = part.costs.to_numpy(dtype=float)
    actions["hindsight_cheapest_cost"] = np.lexsort(
        (np.broadcast_to(np.asarray(MODEL_IDS), c.shape), c), axis=1
    )[:, 0]
    summaries, decisions, evaluation = [], [], []
    rows = np.arange(len(c))
    for policy_id, action in actions.items():
        summary, success, realized = outcome_summary(part, action)
        summaries.extend(dict(policy_id=policy_id, split=part.name, **r) for r in summary)
        identity = dict(
            prompt_id=part.prompts.index,
            policy_id=policy_id,
            selected_model_id=np.asarray(MODEL_IDS)[action],
            selected_probability=prediction.to_numpy()[rows, action],
        )
        decisions.append(pd.DataFrame(identity))
        evaluation.append(
            pd.DataFrame(
                dict(
                    **identity,
                    dataset=part.prompts.dataset.to_numpy(),
                    leakage_group=part.prompts.leakage_group.to_numpy(),
                    success=success,
                    cost_usd=realized,
                )
            )
        )
    return (
        pd.DataFrame(summaries),
        pd.concat(decisions, ignore_index=True),
        pd.concat(evaluation, ignore_index=True),
        actions,
    )


def nondominated(table):
    c, q = table.mean_cost_usd.to_numpy(), table.quality.to_numpy()
    return np.array(
        [not np.any((c <= c[i]) & (q >= q[i]) & ((c < c[i]) | (q > q[i]))) for i in range(len(c))]
    )


def pareto_table(summary):
    table = summary[summary.dataset == "__micro__"].copy().reset_index(drop=True)
    table["pareto_deployable"] = False
    eligible = ~table.policy_id.isin(["oracle", "hindsight_cheapest_cost"])
    table.loc[eligible, "pareto_deployable"] = nondominated(table[eligible])
    table["pareto_with_oracle"] = False
    eligible = table.policy_id != "hindsight_cheapest_cost"
    table.loc[eligible, "pareto_with_oracle"] = nondominated(table[eligible])
    return table.sort_values(["mean_cost_usd", "quality", "policy_id"])


def envelopes(summary, best, costs, budgets):
    micro = summary[summary.dataset == "__micro__"].set_index("policy_id")
    candidates = micro.loc[[p.policy_id for p in grid()]]
    baseline = micro.loc["best_single"]
    savings = []
    for epsilon in (0.005, 0.0):
        feasible = candidates[candidates.quality >= baseline.quality - epsilon]
        winner = (
            None
            if feasible.empty
            else feasible.sort_values(["mean_cost_usd", "quality"], ascending=[True, False]).iloc[0]
        )
        savings.append(
            dict(
                epsilon=epsilon,
                feasible=winner is not None,
                policy_id=None if winner is None else winner.name,
                cost_savings=None
                if winner is None
                else float(1 - winner.mean_cost_usd / baseline.mean_cost_usd),
            )
        )
    static = micro.loc[["static_" + m for m in MODEL_IDS]]
    budget_results = []
    for frozen in budgets:
        b = frozen["budget_usd"]
        routed, fixed = candidates[candidates.mean_cost_usd <= b], static[static.mean_cost_usd <= b]
        feasible = not routed.empty and not fixed.empty
        pid = frozen["validation_selected_policy_id"]
        budget_results.append(
            dict(
                **frozen,
                descriptive_feasible=feasible,
                descriptive_quality_gain=float(routed.quality.max() - fixed.quality.max())
                if feasible
                else None,
                frozen_test_quality=None if pid is None else float(micro.loc[pid, "quality"]),
                frozen_test_cost_usd=None
                if pid is None
                else float(micro.loc[pid, "mean_cost_usd"]),
                frozen_test_budget_violation=None
                if pid is None
                else bool(micro.loc[pid, "mean_cost_usd"] > b),
            )
        )
    return dict(
        descriptive_only=True, savings_at_best_single_quality=savings, fixed_budget=budget_results
    )


def selection_tables(part, prediction, action):
    records = []
    p = prediction.to_numpy()
    y, c = part.y.to_numpy(dtype=float), part.costs.to_numpy(dtype=float)
    for j, model_id in enumerate(MODEL_IDS):
        mask = action == j
        records.append(
            dict(
                model_id=model_id,
                count=int(mask.sum()),
                fraction=float(mask.mean()),
                mean_predicted_success=float(p[mask, j].mean()) if mask.any() else None,
                actual_success=float(y[mask, j].mean()) if mask.any() else None,
                mean_cost_usd=float(c[mask, j].mean()) if mask.any() else None,
            )
        )
    cross = pd.crosstab(part.prompts.dataset.to_numpy(), np.asarray(MODEL_IDS)[action]).reindex(
        columns=MODEL_IDS, fill_value=0
    )
    cross.index.name = "dataset"
    return pd.DataFrame(records), cross.reset_index()
