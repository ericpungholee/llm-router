"""Deterministic orchestration and auditable artifacts for the first ML phase."""

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow

from routing_data.baselines import baseline_artifacts
from routing_data.features import FEATURE_COLUMNS, FEATURE_VERSION, feature_table
from routing_data.matrix import (
    complete_case,
    coverage_table,
    normalize_outcomes,
    validate_processed,
)
from routing_data.source import download_archive, file_hash, read_archive
from routing_data.splits import add_leakage_groups, make_splits, validate_splits

ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAMES = ["llmrouterbench_source.json", "router_model_pool.json", "router_tasks.json"]


def write_json(path, value):
    def default(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)

    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=default) + "\n",
        encoding="utf-8",
    )


def markdown_table(frame):
    columns = list(frame.columns)

    def value(v):
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v).replace("|", "\\|")

    return (
        "| "
        + " | ".join(columns)
        + " |\n| "
        + " | ".join(["---"] * len(columns))
        + " |\n"
        + "\n".join(
            "| " + " | ".join(value(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        )
        + "\n"
    )


def audit_summary(raw, inventory, fields):
    summary = dict(
        fields,
        files=len(inventory),
        json_payload_bytes=int(inventory.bytes.sum()),
        outcome_records=len(raw),
        datasets=sorted(raw.dataset.unique()),
        models=sorted(raw.model_id.unique()),
        source_prompt_keys=len(raw[["dataset", "source_split", "source_index"]].drop_duplicates()),
        dataset_prompt_keys=len(raw[["dataset", "prompt_sha256"]].drop_duplicates()),
        unique_prompt_texts=int(raw.prompt_sha256.nunique()),
        duplicate_source_outcomes=int(
            raw.duplicated(["dataset", "source_split", "model_id", "source_index"]).sum()
        ),
        duplicate_dataset_prompt_outcomes=int(
            raw.duplicated(["dataset", "model_id", "prompt_sha256"]).sum()
        ),
        null_scores=int(raw.raw_score.isna().sum()),
        zero_cost_records=int((raw.raw_cost_usd == 0).sum()),
        negative_cost_records=int((raw.raw_cost_usd < 0).sum()),
        null_cost_records=int(raw.raw_cost_usd.isna().sum()),
        negative_input_tokens=int((raw.input_tokens < 0).sum()),
        negative_output_tokens=int((raw.output_tokens < 0).sum()),
        fractional_input_tokens=int((raw.input_tokens.dropna() % 1 != 0).sum()),
        fractional_output_tokens=int((raw.output_tokens.dropna() % 1 != 0).sum()),
        missing_input_tokens=int(raw.input_tokens.isna().sum()),
        missing_output_tokens=int(raw.output_tokens.isna().sum()),
        generation_failure_records=int(raw.generation_failure.sum()),
        empty_output_records=int((~raw.has_output).sum()),
        demo_records=int(raw.demo.sum()),
        demo_files=int(inventory.demo.sum()),
        path_identity_exceptions=int((~inventory.path_identity_consistent).sum()),
        header_cost_disagreements=int(
            (abs(inventory.header_cost_usd - inventory.record_cost_sum_usd) > 1e-6).sum()
        ),
        count_disagreements=int((inventory.records != inventory.reported_counts).sum()),
    )
    conflicts = raw.groupby(["dataset", "source_split", "source_index"]).prompt_sha256.nunique()
    summary["source_index_text_conflicts"] = [
        dict(dataset=ds, source_split=sp, source_index=int(idx), distinct_texts=int(n))
        for (ds, sp, idx), n in conflicts[conflicts > 1].items()
    ]
    groups = raw.groupby(["dataset", "source_split"])
    datasets = groups.agg(
        outcomes=("source_index", "size"),
        source_indices=("source_index", "nunique"),
        prompt_texts=("prompt_sha256", "nunique"),
        models=("model_id", "nunique"),
        null_scores=("raw_score", lambda s: s.isna().sum()),
        zero_costs=("raw_cost_usd", lambda s: (s == 0).sum()),
    ).reset_index()
    summary["dataset_inventory"] = datasets.to_dict(orient="records")
    summary["score_histograms"] = {
        ds: {str(score): int(n) for score, n in g.raw_score.value_counts(dropna=False).items()}
        for ds, g in raw.groupby("dataset")
    }
    model_coverage = []
    for (ds, split), group in groups:
        n = group.prompt_sha256.nunique()
        for model in summary["models"]:
            m = group[group.model_id == model]
            model_coverage.append(
                dict(
                    dataset=ds,
                    source_split=split,
                    model_id=model,
                    prompt_universe=n,
                    observed_prompts=int(m.prompt_sha256.nunique()),
                    labeled_outcomes=int(m.raw_score.notna().sum()),
                    positive_cost_outcomes=int((m.raw_cost_usd > 0).sum()),
                )
            )
    return summary, pd.DataFrame(model_coverage)


def validate_pool(config):
    ids = [m["model_id"] for m in config["models"]]
    if not 5 <= len(ids) <= 8 or len(ids) != len(set(ids)):
        raise ValueError("Expected 5-8 distinct fixed model identifiers")
    for model in config["models"]:
        if not model["reason_selected"] or not 0 <= model["coverage"] <= 1:
            raise ValueError("Invalid pool config")
    return ids


def verify_pool_statistics(pool, rows, prompts):
    """Catch stale frozen selection statistics; never reselect from test scores."""
    if "coverage_denominator" not in pool:
        return  # Small synthetic fixtures need no dataset-specific expected counts.
    n = len(prompts)
    if n != pool["coverage_denominator"]:
        raise ValueError("Frozen model-pool prompt universe changed")
    for model in pool["models"]:
        part = rows[rows.model_id == model["model_id"]]
        expected = dict(
            coverage=float(part.eligible.sum() / n),
            outcome_coverage=len(part) / n,
            valid_outcomes=int(part.eligible.sum()),
            mean_score=float(part.raw_score.mean()),
            mean_success=float(part.success.mean()),
            mean_cost=float(part.cost_usd.mean()),
        )
        for key, observed in expected.items():
            if abs(model[key] - observed) > 1e-12:
                raise ValueError(f"Stale frozen model-pool statistic: {model['model_id']} {key}")


def write_preparation_report(report_dir, summary, coverages, static):
    text = "# LLMRouterBench preparation result\n\n"
    text += (
        f"Primary dataset: {summary['complete_prompts']:,} complete prompts × {summary['models']} models = "
        f"{summary['outcome_rows']:,} outcomes. No router has been trained; no provider or embedding API calls were made.\n\n"
    )
    text += "## Selected-model coverage before complete-case filtering\n\n" + markdown_table(
        coverages["model"]
    )
    text += "\n## Task retention and splits\n\n" + markdown_table(
        pd.DataFrame(summary["dataset_splits"])
    )
    text += "\n## Baseline references\n\n"
    for regime, data in summary["baselines"].items():
        text += f"### {regime}\n\n"
        text += (
            f"Always-cheapest fixed model: `{data['always_cheapest_model']}` (training mean cost). "
            f"Best-single: `{data['best_single_model']}` (validation success only). "
            f"Cheapest fixed model by test mean cost, descriptive only: `{data['cheapest_single_on_test_descriptive']}`.\n\n"
        )
        text += markdown_table(
            pd.DataFrame([dict(policy=k, **v) for k, v in data["test_policies"].items()])
        )
        text += (
            f"\nEx-post oracle success headroom: {100 * data['oracle_success_headroom']:.3f} percentage points; "
            f"oracle cost reduction relative to best-single: {100 * data['oracle_cost_savings_vs_best_single']:.3f}%. "
            "The oracle and hindsight-cheapest policies use unobservable outcomes/costs and are not achievable routing claims. "
            "On unsolved prompts, oracle cost includes the training-cheapest fallback call; conditional cheapest-success cost remains null.\n\n"
        )
    text += "## Static model summaries on standard test\n\n"
    text += markdown_table(
        static[
            (static.regime == "standard") & (static.split == "test") & (static.dataset == "__all__")
        ].drop(columns=["regime", "split", "dataset"])
    )
    text += (
        "\nFull task-level statistics, both evaluation regimes, and per-prompt baseline/oracle decisions are in Parquet. "
        "See `llmrouterbench_processed_summary.json` for exact counts and `ml_experiment_spec.md` for definitions.\n\n"
    )
    text += "## Missingness and leakage\n\n"
    text += (
        f"{summary['prompts_before_filter'] - summary['complete_prompts']} prompts excluded out of {summary['prompts_before_filter']}; "
        f"{summary['ineligible_pairs']} unavailable model outcomes. Raw scores and resource anomalies remain in audit artifacts. "
        "Availability masks are explicit; absent labels are not zero-filled. No complete-case imputation.\n\n"
        "Splits use only prompt text, source dataset, duplicate groups, and seed 3407. All selected outcomes for a request "
        "share its split. Normalized original questions and rendered requests form duplicate groups across datasets. "
        "OOD holds out all LiveCodeBench; overlapping non-held-out prompts would be purged. "
        "Features contain exactly the allowlisted prompt-derived counts; dataset/task labels are separate metadata. "
        "Best-single selection rejects non-validation rows. Cost estimates use training rows only. "
        "Complete-case filtering may bias evaluation toward callable, accounted-for requests; it is not a reliability benchmark.\n\n"
    )
    text += "## Prompt-visible features\n\n" + ", ".join(f"`{c}`" for c in FEATURE_COLUMNS) + ".\n"
    (report_dir / "llmrouterbench_preparation.md").write_text(text, encoding="utf-8")


def prepare(raw_dir, output_dir, report_dir, download=False, config_dir=None):
    config_dir = Path(config_dir or ROOT / "configs")
    configs = {name: json.loads((config_dir / name).read_text()) for name in CONFIG_NAMES}
    source, pool, tasks = (configs[name] for name in CONFIG_NAMES)
    models = validate_pool(pool)
    if any(t["label_rule"] != "binary" for t in tasks["tasks"]):
        raise ValueError("Initial experiment supports explicitly binary tasks only")
    task_keys = [(t["dataset"], t["source_split"]) for t in tasks["tasks"]]
    if len(set(task_keys)) != len(task_keys):
        raise ValueError("Duplicate task selection")
    output_dir, report_dir = Path(output_dir), Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    archive = download_archive(raw_dir, source, download)
    raw, selected, inventory, fields = read_archive(archive, tasks, models)
    audit, raw_coverage = audit_summary(raw, inventory, fields)
    if audit["duplicate_source_outcomes"]:
        raise ValueError("Duplicate raw source outcomes")
    normalized = normalize_outcomes(selected)
    if (
        normalized.groupby(["dataset", "source_split", "source_index"]).prompt_id.nunique().max()
        > 1
    ):
        raise ValueError("Selected models saw inconsistent request texts for a source index")
    prompt_columns = [
        "prompt_id",
        "prompt",
        "origin_query",
        "dataset",
        "task_type",
        "source_split",
        "source_index",
        "prompt_sha256",
        "origin_sha256",
    ]
    prompts = (
        normalized[prompt_columns].drop_duplicates().sort_values("prompt_id").reset_index(drop=True)
    )
    if prompts.prompt_id.duplicated().any():
        raise ValueError(
            "Ambiguous duplicate prompt metadata; explicit deduplication policy required"
        )
    verify_pool_statistics(pool, normalized, prompts)
    coverages = {
        name: coverage_table(normalized, prompts, models, groups)
        for name, groups in [
            ("model", ["model_id"]),
            ("dataset", ["dataset"]),
            ("task_type", ["task_type"]),
            ("model_dataset", ["model_id", "dataset"]),
        ]
    }
    complete, missing = complete_case(normalized, prompts, models)
    if complete.empty:
        raise ValueError("No complete prompt vectors")
    prompts = add_leakage_groups(prompts[prompts.prompt_id.isin(complete.prompt_id)])
    splits = make_splits(prompts, tasks)
    prompts = prompts.drop(columns=["origin_query"]).merge(
        splits.drop(columns="leakage_group"), on="prompt_id", validate="one_to_one"
    )
    feature_rows = feature_table(prompts)
    columns = [
        "prompt_id",
        "prompt",
        "dataset",
        "task_type",
        "model_id",
        "raw_score",
        "success_label",
        "success",
        "cost_usd",
        "input_tokens",
        "output_tokens",
        "cost_source",
        "source_split",
        "source_index",
        "source_file",
    ]
    outcomes = complete[columns].merge(
        splits[["prompt_id", "standard_split", "ood_split"]], on="prompt_id", validate="many_to_one"
    )
    outcomes = outcomes.sort_values(["prompt_id", "model_id"]).reset_index(drop=True)
    for col in ("input_tokens", "output_tokens", "source_index"):
        outcomes[col] = outcomes[col].astype("int64")
    validate_processed(outcomes, prompts, feature_rows, models)
    validate_splits(prompts, outcomes, splits, tasks)
    print(
        f"Validated {len(prompts):,} complete prompts; computing descriptive references", flush=True
    )
    choices, static, oracles, baselines = baseline_artifacts(outcomes)
    dataset_splits = []
    for ds, group in missing.groupby("dataset"):
        row = dict(dataset=ds, before_filter=len(group), complete_prompts=int(group.complete.sum()))
        for regime in ("standard", "ood"):
            for split in ("train", "validation", "test"):
                row[f"{regime}_{split}"] = int(
                    ((prompts.dataset == ds) & (prompts[f"{regime}_split"] == split)).sum()
                )
        dataset_splits.append(row)
    summary = dict(
        schema_version=1,
        source_revision=source["dataset_revision"],
        prompts_before_filter=len(missing),
        complete_prompts=len(prompts),
        models=len(models),
        model_ids=models,
        outcome_rows=len(outcomes),
        ineligible_pairs=int((~normalized.eligible).sum()),
        missing_outcome_rows=int(len(missing) * len(models) - len(normalized)),
        matrix_missingness_after_filter=0,
        seed=tasks["seed"],
        standard_split_counts=prompts.standard_split.value_counts().to_dict(),
        ood_split_counts=prompts.ood_split.value_counts().to_dict(),
        ood_held_out_datasets=tasks["ood_held_out_datasets"],
        dataset_splits=dataset_splits,
        coverage={k: v.to_dict(orient="records") for k, v in coverages.items()},
        unavailable_reason_counts=normalized.loc[~normalized.eligible, "unavailable_reason"]
        .value_counts()
        .to_dict(),
        duplicate_leakage_groups=int((prompts.groupby("leakage_group").size() > 1).sum()),
        feature_version=FEATURE_VERSION,
        feature_columns=FEATURE_COLUMNS,
        baselines=baselines,
        leakage_checks={
            "unique_prompt_model_pairs": True,
            "prompt_group_splits": True,
            "outcome_split_consistency": True,
            "ood_absent_from_training": True,
            "feature_allowlist": True,
            "validation_only_best_single": True,
            "training_only_cost_estimates": True,
        },
    )
    tables = {
        "outcomes": outcomes,
        "prompts": prompts.sort_values("prompt_id"),
        "splits": splits,
        "prompt_features": feature_rows,
        "baseline_choices": choices,
        "static_model_summary": static,
        "oracle_targets": oracles,
        "missingness_by_prompt": missing.sort_values("prompt_id"),
        "audit_raw_outcomes": raw,
        "audit_source_files": inventory,
        "audit_raw_coverage": raw_coverage,
        "audit_selected_outcomes": normalized.drop(columns=["prompt", "origin_query"]).sort_values(
            ["prompt_id", "model_id"]
        ),
    }
    tables.update({"missingness_by_" + k: v for k, v in coverages.items()})
    artifacts = {}
    for name, frame in tables.items():
        path = output_dir / (name + ".parquet")
        frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
        artifacts[path.name] = dict(
            sha256=file_hash(path),
            size_bytes=path.stat().st_size,
            rows=len(frame),
            schema={c: str(dtype) for c, dtype in frame.dtypes.items()},
        )
    write_json(output_dir / "matrix_metadata.json", summary)
    write_json(report_dir / "llmrouterbench_processed_summary.json", summary)
    write_json(report_dir / "llmrouterbench_raw_audit.json", audit)
    write_preparation_report(report_dir, summary, coverages, static)
    outcomes.head(80).to_csv(output_dir / "outcomes_sample.csv", index=False)
    config_hashes = {}
    for name in CONFIG_NAMES:
        (output_dir / name).write_bytes((config_dir / name).read_bytes())
        config_hashes[name] = file_hash(config_dir / name)
    code_paths = sorted((ROOT / "routing_data").glob("*.py")) + [
        ROOT / "scripts/prepare_llmrouterbench.py",
        ROOT / "experiments/requirements-llmrouterbench.txt",
    ]
    provenance = dict(
        schema_version=1,
        source=source,
        config_sha256=config_hashes,
        code_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in code_paths},
        artifacts=artifacts,
        matrix_metadata_sha256=file_hash(output_dir / "matrix_metadata.json"),
        environment=dict(
            python=platform.python_version(),
            pandas=pd.__version__,
            numpy=np.__version__,
            pyarrow=pyarrow.__version__,
        ),
        split_algorithm="sha256-group-order-largest-remainder-v1",
        split_seed=tasks["seed"],
        reproduction_command=".venv/bin/python scripts/prepare_llmrouterbench.py --download",
        no_training=True,
        no_provider_calls=True,
    )
    write_json(output_dir / "provenance.json", provenance)
    write_json(report_dir / "llmrouterbench_reproducibility.json", provenance)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "complete_prompts",
                    "outcome_rows",
                    "standard_split_counts",
                    "ood_split_counts",
                ]
            },
            indent=2,
        )
    )
    return summary
