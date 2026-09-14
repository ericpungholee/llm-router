"""Paired percentile bootstrap of whole groups within fixed dataset strata."""

import numpy as np
import pandas as pd

from routing_ml.training import SEED


def group_strata(prompts):
    if prompts.groupby("leakage_group").dataset.nunique().max() > 1:
        raise ValueError("Bootstrap group crosses dataset strata")
    strata = []
    for dataset in sorted(prompts.dataset.unique()):
        indices = np.flatnonzero(prompts.dataset.to_numpy() == dataset)
        groups = prompts.leakage_group.to_numpy()[indices]
        strata.append([indices[groups == g] for g in sorted(set(groups))])
    return strata


def sample_groups(strata, rng):
    return np.concatenate([np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                           for groups in strata])


def paired_bootstrap(prompts, router_y, best_y, router_c, best_c, replicates=2000, seed=SEED):
    values = np.asarray([router_y, best_y, router_c, best_c], dtype=float)
    if values.shape != (4, len(prompts)):
        raise ValueError("Bootstrap arrays must align with prompts")
    strata, rng = group_strata(prompts), np.random.default_rng(seed)
    samples = []
    datasets = prompts.dataset.to_numpy()
    for replicate in range(replicates):
        idx = sample_groups(strata, rng)
        ry, by, rc, bc = values[:, idx]
        macro = [values[:, idx[datasets[idx] == d]].mean(axis=1) for d in sorted(set(datasets))]
        mry, mby, mrc, mbc = np.mean(macro, axis=0)
        samples.append(dict(replicate=replicate, quality_delta=float((ry - by).mean()),
                            cost_savings=float(1 - rc.mean() / bc.mean()),
                            macro_quality_delta=float(mry - mby), macro_cost_savings=float(1 - mrc / mbc)))
    samples = pd.DataFrame(samples)
    ci = {name: np.quantile(samples[name], [0.025, 0.975]).tolist() for name in samples if name != "replicate"}
    return samples, dict(replicates=replicates, seed=seed, method="paired stratified whole-group percentile",
                         percentile_method="linear", ci_95=ci,
                         noninferiority_supported=bool(ci["quality_delta"][0] >= -0.005),
                         training_and_selection_fixed=True,
                         groups_per_stratum={d: len(g) for d, g in zip(sorted(set(datasets)), strata)})
