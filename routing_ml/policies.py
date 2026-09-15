"""Prompt-probability policies; selection accepts validation only."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from routing_ml.training import MODEL_IDS

THRESHOLDS = tuple(i / 20 for i in range(21))
LAMBDAS = (0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10)
EPSILON = 0.005


@dataclass(frozen=True)
class Policy:
    policy_id: str
    kind: str
    value: float = 0.0
    model_id: str = ""

    def to_dict(self):
        return asdict(self)


def grid():
    return [Policy(f"threshold_{t:.2f}", "threshold", t) for t in THRESHOLDS] + [
        Policy(f"utility_{v:g}", "utility", v) for v in LAMBDAS
    ]


def estimate_costs(train):
    train.require("train")
    costs = train.costs.to_numpy(dtype=float).mean(axis=0)
    return costs, costs / costs.max()


def validate_predictions(part, prediction):
    if not prediction.index.equals(part.prompts.index) or tuple(prediction.columns) != MODEL_IDS:
        raise ValueError("Predictions do not align with prompt IDs and frozen model order")
    if (
        not np.isfinite(prediction.to_numpy()).all()
        or not ((prediction >= 0) & (prediction <= 1)).all().all()
    ):
        raise ValueError("Invalid probabilities")


def route(prediction, policy, costs, model_ids=MODEL_IDS):
    """Use predicted success and train-fixed costs only, including all tie breaks."""
    p = np.asarray(prediction, dtype=float)
    costs = np.asarray(costs, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(model_ids) or costs.shape != (len(model_ids),):
        raise ValueError("Policy matrix shape/order mismatch")
    if not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("Training costs must be positive finite")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probabilities")
    if policy.kind == "static":
        return np.full(len(p), model_ids.index(policy.model_id), dtype=int)
    ids = np.broadcast_to(np.asarray(model_ids), p.shape)
    c = np.broadcast_to(costs, p.shape)
    if policy.kind == "utility":
        score = p - policy.value * (costs / costs.max())
        return np.lexsort((ids, c, -score), axis=1)[:, 0]
    if policy.kind == "threshold":
        qualifies = p >= policy.value
        eligible = np.lexsort((ids, -p, c, ~qualifies), axis=1)[:, 0]
        fallback = np.lexsort((ids, c, -p), axis=1)[:, 0]
        return np.where(qualifies.any(axis=1), eligible, fallback)
    raise ValueError("Unknown policy")


def best_single(validation, costs):
    validation.require("validation")
    means = validation.y.to_numpy(dtype=float).mean(axis=0)
    j = min(range(len(MODEL_IDS)), key=lambda i: (-means[i], costs[i], MODEL_IDS[i]))
    return Policy("best_single", "static", model_id=MODEL_IDS[j])


def cheapest(costs):
    return min(range(len(MODEL_IDS)), key=lambda i: (costs[i], MODEL_IDS[i]))


def select_validation(validation, prediction, costs):
    validation.require("validation")
    validate_predictions(validation, prediction)
    best = best_single(validation, costs)
    candidates = grid() + [best]
    y, c = validation.y.to_numpy(dtype=float), validation.costs.to_numpy(dtype=float)
    rows = np.arange(len(y))
    records = []
    for policy in candidates:
        selected = route(prediction, policy, costs)
        records.append(
            dict(
                **policy.to_dict(),
                quality=float(y[rows, selected].mean()),
                mean_cost_usd=float(c[rows, selected].mean()),
            )
        )
    table = pd.DataFrame(records)
    target = float(table.loc[table.policy_id == "best_single", "quality"].iloc[0]) - EPSILON
    table["feasible"] = table.quality >= target
    winner = (
        table[table.feasible]
        .sort_values(["mean_cost_usd", "quality", "policy_id"], ascending=[True, False, True])
        .iloc[0]
    )
    selected = next(p for p in candidates if p.policy_id == winner.policy_id)
    budgets = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        budget = fraction * costs[MODEL_IDS.index(best.model_id)]
        feasible = table[(table.kind != "static") & (table.mean_cost_usd <= budget)]
        chosen = (
            None
            if feasible.empty
            else feasible.sort_values(
                ["quality", "mean_cost_usd", "policy_id"], ascending=[False, True, True]
            )
            .iloc[0]
            .policy_id
        )
        budgets.append(
            dict(fraction=fraction, budget_usd=float(budget), validation_selected_policy_id=chosen)
        )
    return selected, best, table, budgets
