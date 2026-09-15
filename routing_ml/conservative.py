"""Fixed comparative policies and joint validation uncertainty for router v2."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from routing_ml.bootstrap import group_strata, sample_groups
from routing_ml.policies import Policy, best_single, grid, route, validate_predictions
from routing_ml.semantic_training import FAMILIES
from routing_ml.training import MODEL_IDS, SEED

MARGINS = (0, 0.02, 0.05, 0.1, 0.2)
GATE_QUANTILES = (0, 0.01, 0.05, 0.1)


def finite_product(a, b):
    # Bounded inputs can raise spurious flags in the pinned macOS BLAS build.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        result = a @ b
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite matrix product")
    return result


@dataclass
class NoveltyGate:
    regime: str
    reference: np.ndarray
    groups: np.ndarray
    cutoffs: dict

    def score(self, values, groups=None):
        values = np.asarray(values, dtype=np.float32)
        scores = []
        for start in range(0, len(values), 128):
            similarity = finite_product(values[start : start + 128], self.reference.T)
            if (np.abs(similarity) > 1.0001).any():
                raise ValueError("Novelty requires unit-normalized embeddings")
            # Independent non-BLAS check of one column in each block.
            reference = np.einsum(
                "ij,j->i", values[start : start + 128], self.reference[0], optimize=False
            )
            np.testing.assert_allclose(similarity[:, 0], reference, atol=2e-6)
            if groups is not None:
                similarity[
                    np.asarray(groups[start : start + 128])[:, None] == self.groups[None, :]
                ] = -np.inf
            best = similarity.max(axis=1)
            if not np.isfinite(best).all():
                raise ValueError("No independent training group for novelty reference")
            scores.extend(best.tolist())
        return np.asarray(scores)

    @classmethod
    def fit(cls, train, embeddings):
        train.require("train")
        if not embeddings.index.equals(train.prompts.index):
            raise ValueError("Novelty training embeddings are misaligned")
        values = embeddings.to_numpy(dtype=np.float32)
        gate = cls(train.regime, values.copy(), train.prompts.leakage_group.to_numpy().copy(), {})
        similarities = gate.score(values, groups=gate.groups)
        gate.cutoffs = {q: float(np.quantile(similarities, q)) for q in GATE_QUANTILES if q > 0}
        return gate, similarities


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    kind: str
    value: float
    gate_quantile: float

    def to_dict(self):
        return asdict(self)


def candidates():
    result = []
    for family in FAMILIES:
        rules = [(p.policy_id, p.kind, p.value) for p in grid()]
        rules += [(f"advantage_{m:.2f}", "advantage", m) for m in MARGINS]
        for policy_id, kind, value in rules:
            for q in GATE_QUANTILES:
                result.append(
                    Candidate(f"{family}__{policy_id}__gate_{q:.2f}", family, kind, value, q)
                )
    return result


def comparative_route(prediction, costs, baseline_index, margin):
    p = np.asarray(prediction, dtype=float)
    c = np.asarray(costs, dtype=float)
    eligible = (c < c[baseline_index])[None, :] & (p >= p[:, baseline_index, None] + margin)
    order = np.lexsort(
        (
            np.broadcast_to(np.asarray(MODEL_IDS), p.shape),
            -p,
            np.broadcast_to(c, p.shape),
            ~eligible,
        ),
        axis=1,
    )[:, 0]
    return np.where(eligible.any(axis=1), order, baseline_index)


def candidate_actions(predictions, costs, baseline_index, similarities, gate, choices=None):
    choices = candidates() if choices is None else choices
    arrays = []
    for candidate in choices:
        p = predictions[candidate.family].to_numpy()
        action = (
            comparative_route(p, costs, baseline_index, candidate.value)
            if candidate.kind == "advantage"
            else route(p, Policy(candidate.candidate_id, candidate.kind, candidate.value), costs)
        )
        if candidate.gate_quantile:
            action = np.where(
                similarities < gate.cutoffs[candidate.gate_quantile], baseline_index, action
            )
        arrays.append(action)
    return np.column_stack(arrays).astype(np.int8)


def resampling_weights(prompts, replicates=2000, seed=SEED):
    strata, rng = group_strata(prompts), np.random.default_rng(seed)
    micro = np.empty((replicates, len(prompts)))
    macro = np.empty_like(micro)
    domains = [np.concatenate(groups) for groups in strata]
    for b in range(replicates):
        selected = sample_groups(strata, rng)
        counts = np.bincount(selected, minlength=len(prompts))
        micro[b] = counts / counts.sum()
        for indices in domains:
            macro[b, indices] = counts[indices] / counts[indices].sum() / len(domains)
    return micro, macro


def simultaneous_bounds(prompts, selected_y, baseline_y, replicates=2000, seed=SEED):
    """Joint max-t across every candidate, all static comparators, micro and macro."""
    selected_y, baseline_y = np.asarray(selected_y, float), np.asarray(baseline_y, float)
    if selected_y.shape[0] != len(prompts) or baseline_y.shape != (len(prompts), len(MODEL_IDS)):
        raise ValueError("Selection outcome alignment mismatch")
    weights = resampling_weights(prompts, replicates, seed)
    datasets = prompts.dataset.to_numpy()
    statistics = {}
    max_t = np.zeros(replicates)
    for name, w in zip(("micro", "macro"), weights):
        if name == "micro":
            observed_r, observed_b = selected_y.mean(axis=0), baseline_y.mean(axis=0)
        else:
            observed_r = np.mean(
                [selected_y[datasets == d].mean(axis=0) for d in sorted(set(datasets))], axis=0
            )
            observed_b = np.mean(
                [baseline_y[datasets == d].mean(axis=0) for d in sorted(set(datasets))], axis=0
            )
        boot_r, boot_b = finite_product(w, selected_y), finite_product(w, baseline_y)
        delta = observed_r[:, None] - observed_b[None, :]
        standard_error = np.empty_like(delta)
        for j in range(len(MODEL_IDS)):
            differences = boot_r - boot_b[:, j, None]
            se = differences.std(axis=0, ddof=1)
            standard_error[:, j] = se
            valid = se > 1e-12
            if valid.any():
                centered = (differences[:, valid] - delta[valid, j]) / se[valid]
                max_t = np.maximum(max_t, centered.max(axis=1))
        statistics[name] = dict(delta=delta, standard_error=standard_error)
    critical = float(np.quantile(max_t, 0.95))
    for name in statistics:
        statistics[name]["lower"] = (
            statistics[name]["delta"] - critical * statistics[name]["standard_error"]
        )
    metadata = dict(
        method="paired stratified whole-group simultaneous bootstrap max-t",
        replicates=replicates,
        seed=seed,
        critical_value=critical,
        confidence=0.95,
        comparators=list(MODEL_IDS),
        metrics=["micro", "macro"],
        unique_action_vectors=selected_y.shape[1],
        approximate_not_distribution_free=True,
        group_counts={
            d: int(prompts.loc[prompts.dataset == d, "leakage_group"].nunique())
            for d in sorted(set(datasets))
        },
    )
    return statistics, pd.DataFrame(dict(replicate=np.arange(replicates), max_t=max_t)), metadata


def conservative_select(validation, predictions, costs, similarities, gate, replicates=2000):
    validation.require("validation")
    if gate.regime != validation.regime:
        raise ValueError("Cannot reuse novelty gate across regimes")
    for p in predictions.values():
        validate_predictions(validation, p)
    best = best_single(validation, costs)
    baseline_index = MODEL_IDS.index(best.model_id)
    choices = candidates()
    all_anchors = [
        np.column_stack(
            [
                candidate_actions(predictions, costs, j, similarities, gate, choices),
                np.full(len(validation.prompts), j, dtype=np.int8),
            ]
        )
        for j in range(len(MODEL_IDS))
    ]
    actions = all_anchors[baseline_index]
    choices.append(Candidate("best_single", "static", "static", 0, 0))
    unique, _, inverse = np.unique(
        np.column_stack(all_anchors), axis=1, return_index=True, return_inverse=True
    )
    inverse = inverse.reshape(-1)[
        baseline_index * len(choices) : (baseline_index + 1) * len(choices)
    ]
    rows = np.arange(len(validation.prompts))[:, None]
    y = validation.y.to_numpy(dtype=float)
    selected_y = y[rows, unique]
    bounds, samples, metadata = simultaneous_bounds(validation.prompts, selected_y, y, replicates)
    realized_y = y[rows, actions]
    realized_c = validation.costs.to_numpy(dtype=float)[rows, actions]
    domain_deltas = []
    domains = validation.prompts.dataset.to_numpy()
    domain_values = {}
    for dataset in sorted(set(domains)):
        mask = domains == dataset
        delta = realized_y[mask].mean(axis=0) - y[mask, baseline_index].mean()
        domain_deltas.append(delta)
        domain_values[dataset] = delta
    table = pd.DataFrame([c.to_dict() for c in choices])
    table["quality"] = realized_y.mean(axis=0)
    table["mean_cost_usd"] = realized_c.mean(axis=0)
    table["micro_delta"] = bounds["micro"]["delta"][inverse, baseline_index]
    table["macro_delta"] = bounds["macro"]["delta"][inverse, baseline_index]
    table["micro_lower"] = bounds["micro"]["lower"][inverse, baseline_index]
    table["macro_lower"] = bounds["macro"]["lower"][inverse, baseline_index]
    table["micro_standard_error"] = bounds["micro"]["standard_error"][inverse, baseline_index]
    table["macro_standard_error"] = bounds["macro"]["standard_error"][inverse, baseline_index]
    table["worst_dataset_delta"] = np.min(domain_deltas, axis=0)
    for dataset, values in domain_values.items():
        table["delta_" + dataset] = values
    table["feasible"] = (
        (table.micro_lower >= -0.005)
        & (table.macro_lower >= -0.005)
        & (table.worst_dataset_delta >= -0.005)
    )
    # Mathematically identical action vectors have zero difference from fallback.
    table.loc[table.candidate_id == "best_single", "feasible"] = True
    winner = (
        table[table.feasible]
        .sort_values(["mean_cost_usd", "quality", "candidate_id"], ascending=[True, False, True])
        .iloc[0]
    )
    chosen = next(c for c in choices if c.candidate_id == winner.candidate_id)
    metadata.update(
        candidate_count=len(choices),
        possible_anchor_identities=list(MODEL_IDS),
        anchor_candidate_count=sum(a.shape[1] for a in all_anchors),
        feasible_count=int(table.feasible.sum()),
        best_single=best.to_dict(),
        selection_split="validation",
        selected_candidate=chosen.to_dict(),
    )
    return chosen, best, table, samples, metadata, actions
