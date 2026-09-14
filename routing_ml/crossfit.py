"""Separate grouped partitions for a fixed-rule robustness check, not new tuning."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from routing_ml.conservative import comparative_route
from routing_ml.policies import best_single, cheapest, estimate_costs
from routing_ml.training import (LOGREG_PARAMS, MODEL_IDS, SEED, TFIDF_PARAMS,
                                 RouterModel, digest, fit_predictors, inputs,
                                 preprocessing, preprocessing_hash)

MARGINS = {"standard": 0.02, "ood": 0.05}
FIXED_C = 1.0


def partition_assignments(prompts):
    """Use identities, dataset strata and groups only; never read supervision."""
    prompts = prompts.loc[:, ["dataset", "leakage_group"]].sort_index()
    if prompts.index.has_duplicates:
        raise ValueError("Duplicate prompt IDs")
    if prompts.groupby("leakage_group").dataset.nunique().max() != 1:
        raise ValueError("Groups cross dataset strata")
    ids = prompts.index.to_numpy()
    groups = prompts.leakage_group.to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    records = []
    for fold, (remaining, test) in enumerate(outer.split(ids, prompts.dataset, groups)):
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        fit, validation = next(inner.split(ids[remaining], prompts.dataset.iloc[remaining], groups[remaining]))
        parts = {"train": remaining[fit], "validation": remaining[validation], "test": test}
        for name, indices in parts.items():
            records.extend(dict(prompt_id=ids[i], fold=fold, role=name,
                                dataset=prompts.dataset.iloc[i], leakage_group=groups[i]) for i in indices)
    table = pd.DataFrame(records).sort_values(["fold", "prompt_id"]).reset_index(drop=True)
    validate_assignments(table, prompts)
    return table


def validate_assignments(table, prompts):
    if set(table.fold) != set(range(5)) or table.duplicated(["fold", "prompt_id"]).any():
        raise ValueError("Invalid fold identities")
    for _, fold in table.groupby("fold"):
        if set(fold.prompt_id) != set(prompts.index) or set(fold.role) != {"train", "validation", "test"}:
            raise ValueError("Incomplete fold")
        if fold.groupby("leakage_group").role.nunique().max() != 1:
            raise ValueError("Group leakage across fold roles")
    test = table[table.role == "test"]
    if test.prompt_id.duplicated().any() or set(test.prompt_id) != set(prompts.index):
        raise ValueError("Each prompt must be evaluated exactly once")
    return True


def fold_ids(assignments, fold, regime, name):
    if regime not in MARGINS or name not in {"train", "validation", "test"}:
        raise ValueError("Unknown regime/role")
    rows = assignments[(assignments.fold == fold) & (assignments.role == name)]
    if regime == "ood":
        code = rows.dataset.eq("livecodebench")
        rows = rows[code if name == "test" else ~code]
    return pd.Index(rows.prompt_id, name="prompt_id")


@dataclass
class CrossfitPart:
    """Experimental role distinct from the unchanged canonical split metadata."""
    regime: str
    name: str
    prompts: pd.DataFrame
    y: pd.DataFrame
    costs: pd.DataFrame

    def __post_init__(self):
        ids = self.prompts.index
        if self.regime not in MARGINS or self.name not in {"train", "validation", "test"}:
            raise ValueError("Unknown regime/role")
        if ids.has_duplicates or not ids.equals(self.y.index) or not ids.equals(self.costs.index):
            raise ValueError("Prompt alignment failure")
        if tuple(self.y.columns) != MODEL_IDS or tuple(self.costs.columns) != MODEL_IDS:
            raise ValueError("Frozen candidate order mismatch")
        if not np.isin(self.y.to_numpy(), [0, 1]).all():
            raise ValueError("Invalid success labels")
        if not np.isfinite(self.costs.to_numpy()).all() or (self.costs.to_numpy() <= 0).any():
            raise ValueError("Invalid costs")
        if self.regime == "ood" and self.name != "test" and self.prompts.dataset.eq("livecodebench").any():
            raise ValueError("OOD fitting/validation contains code")

    def require(self, name):
        if self.name != name:
            raise ValueError(f"This operation accepts {name} only")


def read_part(directory, prompts, ids, regime, name):
    # Arrow predicate pushdown requests only the role's outcomes. Test outcomes
    # are requested by the runner only after its model and actions are saved.
    outcomes = pd.read_parquet(directory / "outcomes.parquet",
                               columns=["prompt_id", "model_id", "success", "cost_usd"],
                               filters=[("prompt_id", "in", ids.tolist())])
    y = outcomes.pivot(index="prompt_id", columns="model_id", values="success").reindex(index=ids, columns=MODEL_IDS)
    costs = outcomes.pivot(index="prompt_id", columns="model_id", values="cost_usd").reindex(index=ids, columns=MODEL_IDS)
    return CrossfitPart(regime, name, prompts.loc[ids].copy(), y, costs)


def fit_fixed_router(train):
    train.require("train")
    with threadpool_limits(limits=1):
        prep = preprocessing("tfidf")
        x = prep.fit_transform(inputs(train, "tfidf"))
        predictors, fallbacks = fit_predictors(x, train.y.to_numpy(dtype=float), FIXED_C)
    metadata = dict(seed=SEED, selected_C=FIXED_C, C_selection="frozen from v2; no v3 tuning",
                    candidate_model_order=list(MODEL_IDS), tfidf_parameters=TFIDF_PARAMS,
                    logistic_parameters=LOGREG_PARAMS, feature_columns=["prompt"],
                    training_prompt_ids_hash=digest(train.prompts.index.tolist()),
                    training_prompt_count=len(train.prompts),
                    training_dataset_counts=train.prompts.dataset.value_counts().sort_index().to_dict(),
                    vocabulary_hash=preprocessing_hash(prep, "tfidf"),
                    vocabulary_size=len(prep.vocabulary_), final_constant_fallbacks=fallbacks,
                    fit_split="crossfit_train", refit_train_validation=False)
    return RouterModel(train.regime, "tfidf", FIXED_C, prep, predictors, metadata)


def freeze_references(train, validation):
    train.require("train")
    validation.require("validation")
    if train.regime != validation.regime:
        raise ValueError("Regime mismatch")
    if set(train.prompts.leakage_group) & set(validation.prompts.leakage_group):
        raise ValueError("Training/validation groups overlap")
    costs, _ = estimate_costs(train)
    best = best_single(validation, costs)
    domains = {}
    for d in sorted(validation.prompts.dataset.unique()):
        means = validation.y.loc[validation.prompts.dataset == d].mean().to_numpy()
        j = min(range(len(MODEL_IDS)), key=lambda j: (-means[j], costs[j], MODEL_IDS[j]))
        domains[d] = MODEL_IDS[j]
    return dict(regime=train.regime, selected_C=FIXED_C, margin=MARGINS[train.regime],
                best_single=best.model_id, mean_training_cost_usd=dict(zip(MODEL_IDS, costs)),
                domain_static_models=domains, always_cheapest=MODEL_IDS[cheapest(costs)],
                validation_prompt_ids_hash=digest(validation.prompts.index.tolist()),
                test_used_for_selection=False)


def frozen_actions(prediction, datasets, frozen):
    if tuple(prediction.columns) != MODEL_IDS or len(prediction) != len(datasets):
        raise ValueError("Prediction/model/dataset alignment failure")
    if not np.isfinite(prediction.to_numpy()).all() or ((prediction < 0) | (prediction > 1)).any().any():
        raise ValueError("Invalid probabilities")
    costs = np.array([frozen["mean_training_cost_usd"][m] for m in MODEL_IDS])
    best = MODEL_IDS.index(frozen["best_single"])
    n = len(prediction)
    result = dict(router=comparative_route(prediction, costs, best, frozen["margin"]),
                  best_single=np.full(n, best),
                  always_cheapest=np.full(n, MODEL_IDS.index(frozen["always_cheapest"])),
                  privileged_domain_static=np.array([MODEL_IDS.index(frozen["domain_static_models"].get(d, frozen["best_single"])) for d in datasets]))
    result.update({"static_" + m: np.full(n, i) for i, m in enumerate(MODEL_IDS)})
    return result
