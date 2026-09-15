"""Explicit label/resource availability and complete-case construction."""

import numpy as np
import pandas as pd


def validate_pairs(rows):
    if rows[["prompt_id", "model_id"]].isna().any().any():
        raise ValueError("Missing prompt/model identifier")
    if rows.duplicated(["prompt_id", "model_id"]).any():
        raise ValueError(
            "Duplicate prompt/model pair; no outcome averaging or implicit deduplication"
        )


def normalize_outcomes(rows):
    out = rows.copy()
    validate_pairs(out)
    scores = out.raw_score
    if (~scores.dropna().isin([0, 1])).any():
        raise ValueError(
            "Selected binary task contains nonbinary score; define a task rule explicitly"
        )
    for col in ("raw_score", "raw_cost_usd", "input_tokens", "output_tokens"):
        if not pd.api.types.is_numeric_dtype(out[col]):
            raise ValueError(f"Nonnumeric {col}")
    # A missing final answer with positive generation usage remains the recorded
    # failure under the benchmark budget. Request errors are unavailable labels.
    no_generation = ~out.has_output & ~(out.output_tokens > 0)
    out["label_available"] = scores.notna() & ~out.generation_failure & ~no_generation
    out["success_label"] = scores.where(out.label_available).astype("Int8")
    out["success"] = out.success_label
    out["cost_available"] = np.isfinite(out.raw_cost_usd) & (out.raw_cost_usd > 0)
    out["cost_usd"] = out.raw_cost_usd.where(out.cost_available)
    out["cost_source"] = np.where(out.cost_available, "benchmark_record", "unavailable")
    out["tokens_available"] = True
    for col in ("input_tokens", "output_tokens"):
        out["tokens_available"] &= np.isfinite(out[col]) & (out[col] > 0) & (out[col] % 1 == 0)
    out["eligible"] = out.label_available & out.cost_available & out.tokens_available
    out["unavailable_reason"] = [
        ";".join(
            name
            for name, bad in [
                ("missing_or_failed_label", not label),
                ("missing_or_invalid_cost", not cost),
                ("missing_or_invalid_tokens", not tokens),
            ]
            if bad
        )
        for label, cost, tokens in zip(
            out.label_available, out.cost_available, out.tokens_available
        )
    ]
    return out


def coverage_table(rows, prompts, models, group_columns):
    validate_pairs(rows)
    keys = prompts[["prompt_id"] + [c for c in group_columns if c != "model_id"]].copy()
    grid = keys.merge(pd.DataFrame({"model_id": models}), how="cross")
    grid = grid.merge(
        rows[
            [
                "prompt_id",
                "model_id",
                "eligible",
                "label_available",
                "cost_available",
                "tokens_available",
            ]
        ],
        how="left",
        on=["prompt_id", "model_id"],
        validate="one_to_one",
        indicator=True,
    )
    grid["outcome_present"] = grid["_merge"] == "both"
    for c in ["eligible", "label_available", "cost_available", "tokens_available"]:
        # Only availability masks use False; success labels are never filled.
        grid[c] = grid[c].eq(True)
    summary = (
        grid.groupby(group_columns)
        .agg(
            expected_pairs=("prompt_id", "size"),
            present_pairs=("outcome_present", "sum"),
            eligible_pairs=("eligible", "sum"),
            labeled_pairs=("label_available", "sum"),
            priced_pairs=("cost_available", "sum"),
            valid_token_pairs=("tokens_available", "sum"),
        )
        .reset_index()
    )
    summary["outcome_coverage"] = summary.present_pairs / summary.expected_pairs
    summary["coverage"] = summary.eligible_pairs / summary.expected_pairs
    summary["missing_pairs"] = summary.expected_pairs - summary.eligible_pairs
    return summary


def complete_case(rows, prompts, models):
    validate_pairs(rows)
    if len(set(models)) != len(models) or not models:
        raise ValueError("Model pool contains duplicate/empty IDs")
    if set(rows.model_id) - set(models):
        raise ValueError("Unknown model in selected outcomes")
    if prompts.prompt_id.duplicated().any():
        raise ValueError("Duplicate prompt IDs")
    if set(rows.prompt_id) - set(prompts.prompt_id):
        raise ValueError("Outcome refers to unknown prompt")
    counts = rows[rows.eligible].groupby("prompt_id").model_id.nunique()
    missing = prompts[["prompt_id", "dataset", "task_type"]].copy()
    missing["available_models"] = missing.prompt_id.map(counts).fillna(0).astype(int)
    missing["missing_models"] = len(models) - missing.available_models
    missing["complete"] = missing.missing_models == 0
    ids = set(missing.loc[missing.complete, "prompt_id"])
    complete = rows[rows.prompt_id.isin(ids)].copy()
    if len(complete) != len(ids) * len(models) or not complete.eligible.all():
        raise ValueError("Incomplete matrix")
    return complete, missing


def validate_processed(outcomes, prompts, features, models):
    from routing_data.features import FEATURE_COLUMNS

    validate_pairs(outcomes)
    if prompts.prompt_id.duplicated().any():
        raise ValueError("Duplicate prompt metadata")
    if set(outcomes.prompt_id) != set(prompts.prompt_id) or set(features.prompt_id) != set(
        prompts.prompt_id
    ):
        raise ValueError("Artifact prompt keys disagree")
    if len(outcomes) != len(prompts) * len(models) or set(outcomes.model_id) != set(models):
        raise ValueError("Artifact matrix incomplete")
    for col in ("success", "success_label"):
        if outcomes[col].isna().any() or not outcomes[col].isin([0, 1]).all():
            raise ValueError("Missing/nonbinary processed labels")
    if not outcomes.success.equals(outcomes.success_label):
        raise ValueError("Label aliases disagree")
    if not np.isfinite(outcomes.cost_usd).all() or (outcomes.cost_usd < 0).any():
        raise ValueError("Invalid processed costs")
    for col in ("input_tokens", "output_tokens"):
        if (
            outcomes[col].isna().any()
            or (outcomes[col] <= 0).any()
            or (outcomes[col] % 1 != 0).any()
        ):
            raise ValueError("Invalid processed tokens")
    if (
        list(features.columns) != ["prompt_id"] + FEATURE_COLUMNS
        or features.prompt_id.duplicated().any()
    ):
        raise ValueError("Feature allowlist/schema violation")
    if features[FEATURE_COLUMNS].isna().any().any() or (features[FEATURE_COLUMNS] < 0).any().any():
        raise ValueError("Invalid prompt feature values")
