"""Descriptive static and ex-post references. No learned models or API calls."""

import pandas as pd


def choose_best_single(validation, split_column, training_costs):
    if validation.empty or set(validation[split_column]) != {"validation"}:
        raise ValueError("Best-single selection accepts validation outcomes only")
    scores = validation.groupby("model_id").success.mean()
    return min(scores.index, key=lambda m: (-scores[m], training_costs[m], m))


def model_summary(rows, prompt_count):
    summary = rows.groupby("model_id").agg(
        outcomes=("prompt_id", "size"), accuracy=("success", "mean"),
        mean_raw_score=("raw_score", "mean"), mean_cost_usd=("cost_usd", "mean"),
        median_cost_usd=("cost_usd", "median"))
    summary["coverage"] = summary.outcomes / prompt_count
    return summary.reset_index()


def baseline_artifacts(outcomes):
    choices, summaries, oracles = [], [], []
    metadata = {}
    for regime in ("standard", "ood"):
        column = regime + "_split"
        train = outcomes[outcomes[column] == "train"]
        val = outcomes[outcomes[column] == "validation"]
        test = outcomes[outcomes[column] == "test"]
        costs = train.groupby("model_id").cost_usd.mean().to_dict()
        cheapest = min(costs, key=lambda m: (costs[m], m))
        best = choose_best_single(val, column, costs)
        test_costs = test.groupby("model_id").cost_usd.mean().to_dict()
        test_cheapest = min(test_costs, key=lambda m: (test_costs[m], m))
        for split in ("train", "validation", "test"):
            part = outcomes[outcomes[column] == split]
            for dataset in ["__all__"] + sorted(part.dataset.unique()):
                sub = part if dataset == "__all__" else part[part.dataset == dataset]
                sm = model_summary(sub, sub.prompt_id.nunique())
                sm["regime"], sm["split"], sm["dataset"] = regime, split, dataset
                summaries.append(sm)
        for policy, model in [("always_cheapest", cheapest), ("best_single", best)]:
            part = test[test.model_id == model][["prompt_id", "model_id", "success", "cost_usd"]].copy()
            part["regime"], part["policy"] = regime, policy
            choices.append(part)
        for pid, group in test.groupby("prompt_id", sort=True):
            ranked = group.sort_values(["cost_usd", "model_id"])
            successful = ranked[ranked.success == 1]
            any_success = int(not successful.empty)
            fallback = group[group.model_id == cheapest].iloc[0]
            winner = successful.iloc[0] if any_success else fallback
            oracles.append(dict(regime=regime, prompt_id=pid, any_success=any_success,
                                cheapest_success_model_id=winner.model_id if any_success else None,
                                cheapest_success_cost_usd=float(winner.cost_usd) if any_success else None,
                                fallback_model_id=cheapest, oracle_policy_cost_usd=float(winner.cost_usd)))
            for policy, row in [("oracle_cheapest_success_with_fallback", winner), ("hindsight_cheapest_cost", ranked.iloc[0])]:
                choices.append(pd.DataFrame([dict(regime=regime, policy=policy, prompt_id=pid,
                                                  model_id=row.model_id, success=int(row.success), cost_usd=float(row.cost_usd))]))
        metadata[regime] = dict(always_cheapest_model=cheapest, best_single_model=best,
                                best_single_selection_split="validation", cost_estimation_split="train",
                                cheapest_single_on_test_descriptive=test_cheapest,
                                training_mean_cost_usd=costs,
                                validation_success=val.groupby("model_id").success.mean().to_dict(),
                                test_prompt_count=int(test.prompt_id.nunique()))
    choice_table = pd.concat(choices, ignore_index=True).sort_values(["regime", "policy", "prompt_id"]).reset_index(drop=True)
    for regime in metadata:
        sub = choice_table[choice_table.regime == regime]
        stats = sub.groupby("policy").agg(success_rate=("success", "mean"), mean_cost_usd=("cost_usd", "mean"))
        metadata[regime]["test_policies"] = stats.to_dict(orient="index")
        metadata[regime]["oracle_success_headroom"] = float(stats.loc["oracle_cheapest_success_with_fallback", "success_rate"] - stats.loc["best_single", "success_rate"])
        baseline_cost = stats.loc["best_single", "mean_cost_usd"]
        metadata[regime]["oracle_cost_savings_vs_best_single"] = float(1 - stats.loc["oracle_cheapest_success_with_fallback", "mean_cost_usd"] / baseline_cost)
    return choice_table, pd.concat(summaries, ignore_index=True), pd.DataFrame(oracles), metadata
