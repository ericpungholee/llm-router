"""Hash-ordered, stratified prompt groups; independent of all outcome labels."""

import hashlib
import unicodedata

import pandas as pd


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def add_leakage_groups(prompts):
    """Connect identical normalized requests OR original questions across datasets.

    This also protects different prompt templates wrapping the same question.
    It is not a claim to detect every semantic paraphrase in public benchmarks.
    """
    p = prompts.sort_values("prompt_id").copy()
    parent = {pid: pid for pid in p.prompt_id}

    def root(pid):
        while parent[pid] != pid:
            parent[pid] = parent[parent[pid]]
            pid = parent[pid]
        return pid

    seen = {}
    for row in p.itertuples(index=False):
        for text in (row.prompt, row.origin_query):
            key = normalize_text(text)
            if not key:
                continue
            if key in seen:
                a, b = sorted([root(row.prompt_id), root(seen[key])])
                parent[b] = a
            else:
                seen[key] = row.prompt_id
    p["leakage_group"] = [digest(root(pid)) for pid in p.prompt_id]
    return p


def _assign_groups(prompts, seed, regime, fractions, labels):
    result = {}
    # A group spanning datasets gets a deterministic combined stratum.
    groups = prompts.groupby("leakage_group").dataset.agg(lambda s: "|".join(sorted(set(s))))
    for _, ids in groups.groupby(groups):
        order = sorted(ids.index, key=lambda x: (digest(f"{seed}:{regime}:{x}"), x))
        # Largest remainder apportionment; exact 70/15/15 up to whole groups.
        ideals = [len(order) * f for f in fractions]
        sizes = [int(n) for n in ideals]
        remainder_order = sorted(range(len(sizes)), key=lambda i: (-(ideals[i] - sizes[i]), i))
        for i in remainder_order[:len(order) - sum(sizes)]:
            sizes[i] += 1
        start = 0
        for label, size in zip(labels, sizes):
            result.update({g: label for g in order[start:start + size]})
            start += size
    return result


def make_splits(prompts, config):
    if prompts.prompt_id.duplicated().any():
        raise ValueError("Duplicate prompt IDs")
    fractions = config["standard_fractions"]
    if len(fractions) != 3 or any(f < 0 for f in fractions) or abs(sum(fractions) - 1) > 1e-12:
        raise ValueError("Invalid split fractions")
    val_fraction = config["ood_validation_fraction"]
    if not 0 < val_fraction < 1:
        raise ValueError("Invalid OOD validation fraction")
    standard = _assign_groups(prompts, config["seed"], "standard", fractions, ["train", "validation", "test"])
    held = set(config["ood_held_out_datasets"])
    if not held or not held.issubset(set(prompts.dataset)):
        raise ValueError("Unknown/empty OOD holdout")
    test_groups = set(prompts.loc[prompts.dataset.isin(held), "leakage_group"])
    remaining = prompts[~prompts.leakage_group.isin(test_groups)]
    if remaining.empty:
        raise ValueError("OOD holdout leaves no training prompts")
    ood = _assign_groups(remaining, config["seed"], "ood", [1-val_fraction, val_fraction], ["train", "validation"])
    out = prompts[["prompt_id", "leakage_group"]].copy()
    out["standard_split"] = out.leakage_group.map(standard)
    # If a training-domain prompt duplicates a held-out one, purge it; do not
    # expand OOD test with in-domain data or expose its labels during training.
    out["ood_split"] = ["test" if ds in held else "excluded_overlap" if g in test_groups else ood[g]
                        for ds, g in zip(prompts.dataset, prompts.leakage_group)]
    return out.sort_values("prompt_id").reset_index(drop=True)


def validate_splits(prompts, outcomes, splits, config):
    if splits.prompt_id.duplicated().any() or set(splits.prompt_id) != set(prompts.prompt_id):
        raise ValueError("Split keys do not match prompts")
    p = prompts[["prompt_id", "prompt", "dataset"]].merge(splits, validate="one_to_one")
    for column in ("standard_split", "ood_split"):
        if p[column].isna().any():
            raise ValueError("Missing split")
        active = p[p[column] != "excluded_overlap"].copy()
        active["normalized_prompt"] = active.prompt.map(normalize_text)
        for key in ("leakage_group", "normalized_prompt"):
            if (active.groupby(key)[column].nunique() > 1).any():
                raise ValueError("Prompt leakage across splits")
        if column in outcomes:
            joined = outcomes[["prompt_id", column]].merge(splits[["prompt_id", column]], on="prompt_id", suffixes=("_row", "_prompt"), validate="many_to_one")
            if not joined[f"{column}_row"].equals(joined[f"{column}_prompt"]):
                raise ValueError("Outcomes for a prompt disagree on split")
    if p[p.ood_split == "train"].dataset.isin(config["ood_held_out_datasets"]).any():
        raise ValueError("OOD dataset present in training")
