"""Small training-facing interface; features and supervision stay separate."""

from pathlib import Path

import pandas as pd

from routing_data.features import FEATURE_COLUMNS


def load_split(directory, regime="standard", split="train"):
    """Return aligned request rows, numeric X, success Y, and realized costs.

    Columns of Y/costs follow the frozen pool order. Costs are evaluation targets,
    never input features. Text vectorizers must be fitted only on train.prompt.
    """
    import json

    if regime not in {"standard", "ood"} or split not in {"train", "validation", "test"}:
        raise ValueError("Unknown evaluation split")
    directory = Path(directory)
    pool = json.loads((directory / "router_model_pool.json").read_text())
    models = [m["model_id"] for m in pool["models"]]
    prompts = pd.read_parquet(directory / "prompts.parquet")
    prompts = prompts[prompts[regime + "_split"] == split].set_index("prompt_id").sort_index()
    features = pd.read_parquet(directory / "prompt_features.parquet").set_index("prompt_id")
    outcomes = pd.read_parquet(
        directory / "outcomes.parquet", columns=["prompt_id", "model_id", "success", "cost_usd"]
    )
    y = outcomes.pivot(index="prompt_id", columns="model_id", values="success").reindex(
        index=prompts.index, columns=models
    )
    costs = outcomes.pivot(index="prompt_id", columns="model_id", values="cost_usd").reindex(
        index=prompts.index, columns=models
    )
    if y.isna().any().any() or costs.isna().any().any():
        raise ValueError("Incomplete processed matrix")
    return prompts, features.loc[prompts.index, FEATURE_COLUMNS], y, costs
