"""Comparative policies anchored to GPT-5 and source-only domain validation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from routing_ml.bootstrap import group_strata, sample_groups
from routing_ml.conservative import comparative_route
from routing_ml.policies import validate_predictions
from routing_ml.training import MODEL_IDS, SEED

REFERENCE = "gpt-5"
REFERENCE_INDEX = MODEL_IDS.index(REFERENCE)
MARGINS = (0.0, 0.02, 0.05, 0.1, 0.2)
POLICY_IDS = tuple(f"advantage_{m:.2f}" for m in MARGINS) + ("fixed_gpt5",)


def grid_actions(prediction, costs):
    if tuple(prediction.columns) != MODEL_IDS or not np.isfinite(prediction.to_numpy()).all():
        raise ValueError("Prediction order/values mismatch")
    if ((prediction < 0) | (prediction > 1)).any().any():
        raise ValueError("Invalid probabilities")
    costs = np.asarray(costs, float)
    if costs.shape != (len(MODEL_IDS),) or not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("Invalid training costs")
    actions = [comparative_route(prediction, costs, REFERENCE_INDEX, margin) for margin in MARGINS]
    actions.append(np.full(len(prediction), REFERENCE_INDEX))
    return pd.DataFrame(
        np.column_stack(actions).astype(np.int8), index=prediction.index, columns=POLICY_IDS
    )


def source_partitions(train):
    train.require("train")
    p = train.prompts
    if p.groupby("leakage_group").dataset.nunique().max() != 1:
        raise ValueError("A leakage group crosses datasets")
    if train.regime == "ood" and p.dataset.eq("livecodebench").any():
        raise ValueError("OOD source includes code")
    if p.dataset.nunique() < 2:
        raise ValueError("Need multiple source domains")
    return {
        d: (p.index[p.dataset != d], p.index[p.dataset == d]) for d in sorted(p.dataset.unique())
    }


@dataclass
class Evidence:
    name: str
    regime: str
    prompts: pd.DataFrame
    quality_difference: np.ndarray
    costs: np.ndarray
    quality: np.ndarray

    def __post_init__(self):
        if self.name not in {"ordinary", "source_holdout"}:
            raise ValueError("Unknown validation evidence block")
        if self.prompts.index.has_duplicates:
            raise ValueError("Duplicated evidence prompts")
        if self.regime not in {"standard", "ood"}:
            raise ValueError("Unknown regime")
        if self.regime == "ood" and self.prompts.dataset.eq("livecodebench").any():
            raise ValueError("Code cannot enter OOD policy selection")
        for a in (self.quality_difference, self.costs, self.quality):
            if (
                np.asarray(a).shape != (len(self.prompts), len(POLICY_IDS))
                or not np.isfinite(a).all()
            ):
                raise ValueError("Evidence matrix alignment failure")
        if (
            not np.isin(self.quality, [0, 1]).all()
            or not np.isin(self.quality_difference, [-1, 0, 1]).all()
        ):
            raise ValueError("Invalid binary quality evidence")
        if (self.costs <= 0).any() or (self.quality_difference[:, -1] != 0).any():
            raise ValueError("Invalid cost/reference evidence")


def evidence_from_actions(part, actions, name):
    part.require("validation")
    if not actions.index.equals(part.prompts.index) or tuple(actions.columns) != POLICY_IDS:
        raise ValueError("Evidence/action identity mismatch")
    a = actions.to_numpy()
    if not np.issubdtype(a.dtype, np.integer) or (a < 0).any() or (a >= len(MODEL_IDS)).any():
        raise ValueError("Invalid action indices")
    y, c = part.y.to_numpy(dtype=float), part.costs.to_numpy(dtype=float)
    rows = np.arange(len(y))[:, None]
    selected_y = y[rows, a]
    return Evidence(
        name,
        part.regime,
        part.prompts.copy(),
        selected_y - y[:, REFERENCE_INDEX, None],
        c[rows, a],
        selected_y,
    )


def evidence_from_predictions(part, prediction, costs, name):
    part.require("validation")
    validate_predictions(part, prediction)
    return evidence_from_actions(part, grid_actions(prediction, costs), name)


def _statistics(prompts, values, indices=None):
    if indices is None:
        indices = np.arange(len(prompts))
    datasets = prompts.dataset.to_numpy()[indices]
    selected = values[indices]
    names = sorted(set(datasets))
    domain = np.stack([selected[datasets == d].mean(axis=0) for d in names])
    return ["__micro__", "__macro__"] + names, np.vstack(
        [selected.mean(axis=0), domain.mean(axis=0), domain]
    )


def joint_bounds(blocks, replicates=2000, seed=SEED):
    """Conditional joint bounds over six policies, both blocks and all domains."""
    if set(blocks) != {"ordinary", "source_holdout"}:
        raise ValueError("Both source and ordinary validation blocks are required")
    if len({e.regime for e in blocks.values()}) != 1:
        raise ValueError("Evidence cannot cross regimes")
    ordinary, source = blocks["ordinary"], blocks["source_holdout"]
    if set(ordinary.prompts.index) & set(source.prompts.index) or set(
        ordinary.prompts.leakage_group
    ) & set(source.prompts.leakage_group):
        raise ValueError("Validation blocks overlap")
    rng = np.random.default_rng(seed)
    rows, maxima = [], {}
    for name in ("ordinary", "source_holdout"):
        e = blocks[name]
        strata = group_strata(e.prompts)
        labels, observed = _statistics(e.prompts, e.quality_difference)
        boot = np.empty((replicates,) + observed.shape)
        for b in range(replicates):
            _, boot[b] = _statistics(e.prompts, e.quality_difference, sample_groups(strata, rng))
        se = boot.std(axis=0, ddof=1)
        centered = np.divide(boot - observed, se, out=np.zeros_like(boot), where=se > 1e-12)
        maxima[name] = np.maximum(0, centered.max(axis=(1, 2)))
        for i, metric in enumerate(labels):
            rows.extend(
                dict(
                    block=name,
                    metric=metric,
                    policy_id=pid,
                    quality_delta=float(observed[i, j]),
                    standard_error=float(se[i, j]),
                )
                for j, pid in enumerate(POLICY_IDS)
            )
    max_t = np.maximum(maxima["ordinary"], maxima["source_holdout"])
    critical = float(np.quantile(max_t, 0.95))
    ordinary_critical = float(np.quantile(maxima["ordinary"], 0.95))
    table = pd.DataFrame(rows)
    table["joint_lower"] = table.quality_delta - critical * table.standard_error
    table["ordinary_only_lower"] = np.where(
        table.block.eq("ordinary"),
        table.quality_delta - ordinary_critical * table.standard_error,
        np.nan,
    )
    samples = pd.DataFrame(
        dict(
            replicate=np.arange(replicates),
            joint_max_t=max_t,
            ordinary_max_t=maxima["ordinary"],
            source_holdout_max_t=maxima["source_holdout"],
        )
    )
    metadata = dict(
        replicates=replicates,
        seed=seed,
        reference=REFERENCE,
        critical_value=critical,
        ordinary_only_critical_value=ordinary_critical,
        comparison_count=len(table),
        method="paired stratified whole-group joint max-t lower bounds",
        confidence=0.95,
        conditional_on_fitted_models=True,
        independent_confirmation=False,
        source_auxiliary_fit_dependence_not_estimated=True,
    )
    return table, samples, metadata


def select_policy(blocks, replicates=2000, seed=SEED):
    bounds, samples, metadata = joint_bounds(blocks, replicates, seed)
    ordinary = blocks["ordinary"]
    rows = []
    for j, pid in enumerate(POLICY_IDS):
        b = bounds[bounds.policy_id == pid]
        lower = float(b.joint_lower.min())
        ordinary_lower = float(b.loc[b.block == "ordinary", "ordinary_only_lower"].min())
        rows.append(
            dict(
                policy_id=pid,
                quality=float(ordinary.quality[:, j].mean()),
                mean_cost_usd=float(ordinary.costs[:, j].mean()),
                worst_joint_lower=lower,
                worst_ordinary_lower=ordinary_lower,
                feasible=lower >= -0.005,
                ordinary_only_feasible=ordinary_lower >= -0.005,
            )
        )
    candidates = pd.DataFrame(rows)
    for column in ("feasible", "ordinary_only_feasible"):
        if not candidates.loc[candidates.policy_id == "fixed_gpt5", column].item():
            raise AssertionError("Static reference must always be feasible")

    def choose(column):
        return (
            candidates[candidates[column]]
            .sort_values(["mean_cost_usd", "quality", "policy_id"], ascending=[True, False, True])
            .iloc[0]
            .policy_id
        )

    selected = choose("feasible")
    control = choose("ordinary_only_feasible")
    metadata.update(
        selected_policy=selected,
        ordinary_only_control=control,
        feasible_count=int(candidates.feasible.sum()),
    )
    return selected, control, candidates, bounds, samples, metadata
