"""Freeze the 4096 study and plan a separate exhaustion-only treatment offline."""

import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import Decimal
from itertools import combinations
from pathlib import Path

from benchmark_loaders import load_enabled_benchmarks, load_hard_pilot
from dataset_schema import validate_results
from generate_dataset import read_results, render_prompt, result_key
from model_registry import MODEL_REGISTRY
from offline_guard import offline_only
from pilot_analysis import _metrics, analyze_results, model_key, model_label, response_cost
from spend_control import build_run_plan, max_call_cost_usd

ROOT = Path(__file__).resolve().parent
SOURCE = Path("data/results/hard_pilot_results.csv")
SNAPSHOT = Path("reports/snapshots/hard_pilot_4096_final.csv")
PROVENANCE = SNAPSHOT.with_suffix(".provenance.json")
RECOMMENDED_BUDGET = 8192
PINNED_FILES = (
    "model_registry.py",
    "benchmarks/manifest.json",
    "benchmarks/hard_pilot.json",
    "benchmark_loaders.py",
    "graders.py",
    "generate_dataset.py",
    "providers.py",
    "provider_clients/base.py",
    "provider_clients/normalization.py",
    "provider_clients/openai.py",
    "provider_clients/anthropic.py",
    "provider_clients/deepseek.py",
    "provider_clients/openrouter.py",
    "benchmarks/samples/mmlu_pro.jsonl",
    "benchmarks/samples/math_500.jsonl",
    "benchmarks/samples/livecodebench.jsonl",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_spend(rows):
    return str(sum((Decimal(str(r["estimated_cost_usd"])) for r in rows), Decimal(0)))


def freeze(root=ROOT):
    """Exclusive byte copy, with no regrading or writes to the source."""
    root = Path(root)
    with offline_only() as audit:
        source, snapshot, provenance = (root / p for p in (SOURCE, SNAPSHOT, PROVENANCE))
        if snapshot.exists() or provenance.exists():
            raise FileExistsError("The frozen experiment already exists; never replace it")
        original = source.read_bytes()
        source_hash = hashlib.sha256(original).hexdigest()
        stat_before = source.stat()
        rows = read_results(source)
        validate_results(rows)
        if {r["max_output_tokens"] for r in rows} != {4096}:
            raise ValueError("Source is not a uniform recorded 4096-token experiment")
        metrics = _metrics(rows, 75)
        configs = [asdict(m) for m in MODEL_REGISTRY]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        pins = {name: sha256(root / name) for name in PINNED_FILES}
        if source.read_bytes() != original or source.stat().st_mtime_ns != stat_before.st_mtime_ns:
            raise ValueError("Source changed during freeze; no snapshot written")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open("xb") as handle:
            handle.write(original)
        frozen_hash = sha256(snapshot)
        if frozen_hash != source_hash or source.read_bytes() != original:
            raise ValueError("Source/frozen bytes changed during freeze")
        data = {
            "experiment": "hard_pilot_4096_final",
            "source_path": str(SOURCE),
            "snapshot_path": str(SNAPSHOT),
            "source_sha256": source_hash,
            "frozen_sha256": frozen_hash,
            "row_count": len(rows),
            "size_bytes": len(original),
            "recorded_spend_usd": exact_spend(rows),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "metrics": metrics,
            "output_limit_exhausted_count": sum(
                r["error_type"] == "output_limit_exhausted" for r in rows
            ),
            "http_200_exhausted_responses": sum(
                d.get("http_status") == 200 and d.get("outcome") == "output_limit_exhausted"
                for r in rows
                for d in json.loads(r.get("provider_diagnostics") or "[]")
            ),
            "response_count_definition": "Successful normalized provider responses as recorded/inferred by the existing ledger; HTTP-200 exhausted responses are separate.",
            "generation_configuration": configs,
            "observed_generation_configuration": [
                {
                    "provider": p,
                    "model_id": mid,
                    "temperature": t,
                    "reasoning_setting": effort,
                    "max_output_tokens": budget,
                }
                for p, mid, t, effort, budget in sorted(
                    {
                        (
                            r["inference_provider"],
                            r["api_model_identifier"],
                            r["temperature"],
                            r["reasoning_setting"],
                            r["max_output_tokens"],
                        )
                        for r in rows
                    }
                )
            ],
            "configuration_file_sha256": pins,
            "history": "Completed working CSV includes earlier attempts/resumes; no new calls, offline regrade, sanitization, or field changes during freezing.",
            "source_unchanged": True,
            "no_provider_calls_while_freezing": True,
            "offline_audit": dict(audit),
            "immutability": "Exclusive creation; SHA verified on every read; tracked artifact, never a resume/merge destination.",
        }
        with provenance.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
        return data


def frozen_inputs(root=ROOT, *, verify_configuration=False):
    root = Path(root)
    provenance = json.loads((root / PROVENANCE).read_text())
    if sha256(root / SNAPSHOT) != provenance["frozen_sha256"]:
        raise ValueError("Frozen snapshot SHA-256 differs from provenance")
    rows = read_results(root / SNAPSHOT)
    validate_results(rows)
    if (
        len(rows) != provenance["row_count"]
        or exact_spend(rows) != provenance["recorded_spend_usd"]
    ):
        raise ValueError("Frozen snapshot accounting differs from provenance")
    if verify_configuration:
        for name, expected in provenance["configuration_file_sha256"].items():
            if sha256(root / name) != expected:
                raise ValueError(f"Treatment configuration changed: {name}")
    from model_registry import GenerationSettings, ModelConfig

    models = tuple(
        ModelConfig(**{**c, "generation": GenerationSettings(**c["generation"])})
        for c in provenance["generation_configuration"]
        if c["enabled"]
    )
    plan = json.loads((root / "benchmarks/hard_pilot.json").read_text())["prompts"]
    return rows, models, plan, provenance


def select_exhausted(rows):
    validate_results(rows)
    return sorted(
        (
            r
            for r in rows
            if r["status"] == "provider_error"
            and r["error_type"] == "output_limit_exhausted"
            and r["inference_provider"] != "xai"
            and "grok" not in r["api_model_identifier"].lower()
            and "grok" not in r["model_name"].lower()
        ),
        key=result_key,
    )


def treatment_calls(rows, models, budget):
    if budget not in (8192, 16384):
        raise ValueError("Treatment budget must be 8192 or 16384")
    registry = {(m.inference_provider, m.api_model_identifier): m for m in models}
    calls = []
    for row in select_exhausted(rows):
        model = registry[model_key(row)]
        if (
            row["max_output_tokens"] != 4096
            or row["reasoning_setting"] != model.generation.reasoning_setting
            or row["temperature"] != model.generation.temperature_record
        ):
            raise ValueError("Treatment must preserve the frozen generation settings")
        model = replace(model, generation=replace(model.generation, max_output_tokens=budget))
        calls.append((row, model, row["prompt"], budget))
    return calls


def cost_bound_issues(rows):
    # A contradictory observed response cannot establish a monetary ceiling.
    issues = []
    for row in rows:
        if row["inference_provider"] == "xai":
            continue
        for diagnostic in json.loads(row.get("provider_diagnostics") or "[]"):
            output = diagnostic.get("output_tokens")
            limit = diagnostic.get("requested_max_output_tokens")
            if isinstance(output, int) and isinstance(limit, int) and output > limit:
                issues.append(
                    {
                        "prompt_id": row["prompt_id"],
                        "model": model_label(model_key(row)),
                        "requested_max_output_tokens": limit,
                        "output_tokens": output,
                        "reasoning_tokens": diagnostic.get("reasoning_tokens"),
                        "reason": "Observed billed output exceeds recorded request ceiling; strict pre-dispatch bound unresolved",
                    }
                )
    return issues


def validate_treatment_output(output, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    protected = {(root / p).resolve() for p in (SOURCE, SNAPSHOT, PROVENANCE)}
    if output in protected or output.is_relative_to(root / "reports/snapshots"):
        raise ValueError("Treatment output cannot overwrite a working CSV or frozen snapshot")
    for path in (
        output,
        output.with_suffix(output.suffix + ".tmp"),
        output.with_suffix(output.suffix + ".experiment.json"),
    ):
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                "Treatment output/checkpoint already exists; no overwrite or paid resume"
            )
    return output


def treatment_plan(rows, models, budget, spend_cap="1.66"):
    calls = treatment_calls(rows, models, budget)
    tuples = [(m, prompt, b) for _, m, prompt, b in calls]
    plan = build_run_plan(tuples, spend_cap=spend_cap, max_retries=0)
    selected_models = {model_label(model_key(r)) for r, _, _, _ in calls}
    issues = [issue for issue in cost_bound_issues(rows) if issue["model"] in selected_models]
    cap_ok = plan.estimated_max_cost_usd <= plan.configured_spend_cap_usd
    return (
        {
            "experiment": f"hard_pilot_exhaustion_{budget}",
            "parent_snapshot": str(SNAPSHOT),
            "max_output_tokens": budget,
            "planned_calls": len(calls),
            "max_retries": 0,
            "maximum_attempts_per_pair": 1,
            "maximum_provider_attempts": plan.maximum_provider_attempts,
            "registry_contract_maximum_usd": str(plan.estimated_max_cost_usd),
            "verified_worst_case_spend_usd": None if issues else str(plan.estimated_max_cost_usd),
            "spend_cap_usd": str(plan.configured_spend_cap_usd),
            "within_cap": cap_ok,
            "strict_preflight_passed": cap_ok and not issues and bool(calls),
            "cost_bound_issues": issues,
            "by_model": {
                label: {
                    "calls": sum(model_label(model_key(r)) == label for r, _, _, _ in calls),
                    "registry_contract_maximum_usd": str(
                        sum(
                            (
                                max_call_cost_usd(m, prompt, b)
                                for r, m, prompt, b in calls
                                if model_label(model_key(r)) == label
                            ),
                            Decimal(0),
                        )
                    ),
                }
                for label in sorted(selected_models)
            },
            "pairs": [
                {
                    "prompt_id": r["prompt_id"],
                    "model": model_label(model_key(r)),
                    "benchmark": r["benchmark"],
                    "task_type": r["task_type"],
                    "source_final_state": r["error_type"],
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "reasoning_setting": m.generation.reasoning_setting,
                    "temperature": m.generation.temperature_record,
                }
                for r, m, prompt, _ in calls
            ],
            "executed": False,
        },
        plan,
        calls,
    )


def pair_comparison(rows, a, b, prompts):
    matrix = {(r["prompt_id"], model_key(r)): r for r in rows}
    common = [
        p
        for p in sorted(prompts)
        if all((p, m) in matrix and matrix[p, m]["status"] == "success" for m in (a, b))
    ]
    wins_a = sum(matrix[p, a]["correct"] and not matrix[p, b]["correct"] for p in common)
    wins_b = sum(matrix[p, b]["correct"] and not matrix[p, a]["correct"] for p in common)
    both_correct = sum(matrix[p, a]["correct"] and matrix[p, b]["correct"] for p in common)
    return {
        "models": [model_label(a), model_label(b)],
        "n": len(common),
        "prompt_ids": common,
        "accuracy_a": sum(matrix[p, a]["correct"] for p in common) / len(common)
        if common
        else None,
        "accuracy_b": sum(matrix[p, b]["correct"] for p in common) / len(common)
        if common
        else None,
        "a_only_correct": wins_a,
        "b_only_correct": wins_b,
        "both_correct": both_correct,
        "both_wrong": len(common) - wins_a - wins_b - both_correct,
        "disagreement": wins_a + wins_b,
        "disagreement_rate": (wins_a + wins_b) / len(common) if common else None,
    }


def build_report(root=ROOT):
    with offline_only() as audit:
        rows, models, plan, provenance = frozen_inputs(root, verify_configuration=True)
        membership = {p: b for b, ps in plan.items() for p in ps}
        keys = sorted((m.inference_provider, m.api_model_identifier) for m in models)
        base = analyze_results(rows, list(membership), keys, membership)
        common = base["routing_comparison"]["comparison_prompt_ids"]
        matrix = {(r["prompt_id"], model_key(r)): r for r in rows}
        static = {}
        for m in keys:
            selected = [matrix[p, m] for p in common]
            costs = [response_cost(r) for r in selected]
            total = (
                sum(costs, Decimal(0)) if selected and all(c is not None for c in costs) else None
            )
            static[model_label(m)] = {
                "n": len(selected),
                "correct": sum(r["correct"] for r in selected),
                "accuracy": sum(r["correct"] for r in selected) / len(selected)
                if selected
                else None,
                "total_response_cost_usd": str(total) if total is not None else None,
            }
        exhausted = select_exhausted(rows)
        diagnostics = []
        for r in exhausted:
            ds = [
                d
                for d in json.loads(r.get("provider_diagnostics") or "[]")
                if d.get("outcome") == "output_limit_exhausted"
            ]
            d = ds[-1] if ds else {}
            diagnostics.append(
                {
                    "prompt_id": r["prompt_id"],
                    "model": model_label(model_key(r)),
                    "benchmark": r["benchmark"],
                    "task_type": r["task_type"],
                    "requested_max_output_tokens": d.get(
                        "requested_max_output_tokens", r["max_output_tokens"]
                    ),
                    "output_tokens": d.get("output_tokens", r["output_tokens"]),
                    "reasoning_tokens": d.get("reasoning_tokens"),
                    "stop_reason": d.get("stop_reason", r["stop_reason"]),
                    "incomplete_reason": d.get("incomplete_reason"),
                    "visible_text_present": d.get(
                        "visible_text_present", bool(r["raw_response"].strip())
                    ),
                }
            )
        potential = {}
        for name, subset in [
            ("five_model", keys),
            ("four_non_xai_models", [m for m in keys if m[0] != "xai"]),
        ]:
            potential[name] = [
                p
                for p in sorted(membership)
                if all(
                    matrix[p, m]["status"] == "success"
                    or result_key(matrix[p, m]) in {result_key(r) for r in exhausted}
                    for m in subset
                )
            ]
        pairs = [
            {
                "all": pair_comparison(rows, a, b, membership),
                "by_benchmark": {
                    bench: pair_comparison(rows, a, b, ps) for bench, ps in plan.items()
                },
            }
            for a, b in combinations(keys, 2)
        ]
        return {
            "provenance": provenance,
            "analysis": base,
            "overall": _metrics(rows, 75),
            "recorded_spend_usd": exact_spend(rows),
            "by_benchmark": {
                b: _metrics([r for r in rows if r["benchmark"] == b], len(ps) * len(keys))
                for b, ps in plan.items()
            },
            "common_vectors": [
                {
                    "prompt_id": p,
                    "benchmark": membership[p],
                    "correctness": {model_label(m): matrix[p, m]["correct"] for m in keys},
                }
                for p in common
            ],
            "static_baselines": static,
            "pairwise": pairs,
            "disagreement_matrix": {
                model_label(a): {
                    model_label(b): pair_comparison(rows, a, b, membership) for b in keys
                }
                for a in keys
            },
            "common_disagreement_count": sum(
                len({matrix[p, m]["correct"] for m in keys}) > 1 for p in common
            ),
            "output_exhaustion": {
                "count": len(exhausted),
                "pairs": diagnostics,
                "by_model": {
                    model_label(m): sum(model_key(r) == m for r in exhausted) for m in keys
                },
                "by_benchmark": dict(Counter(r["benchmark"] for r in exhausted)),
                "by_task_type": dict(Counter(r["task_type"] for r in exhausted)),
                "by_model_benchmark_task": [
                    {"model": m, "benchmark": b, "task_type": t, "count": n}
                    for (m, b, t), n in sorted(
                        Counter(
                            (model_label(model_key(r)), r["benchmark"], r["task_type"])
                            for r in exhausted
                        ).items()
                    )
                ],
            },
            "cost_bound_issues": cost_bound_issues(rows),
            "treatments": {
                str(b): treatment_plan(rows, models, b, "1.66" if b == 8192 else "3.22")[0]
                for b in (8192, 16384)
            },
            "maximum_completeness_after_all_selected_recover": potential,
            "decision": {
                "suitable_for_learned_router_training_or_evaluation": False,
                "reason": "Systematic output-budget missingness and only five common prompts; no held-out evaluation.",
                "prompt_visible_baseline_meaningfully_evaluable": False,
                "recommendation": "8192-token feasibility treatment, conditional on resolving strict spend bounds; no ML training.",
                "mixed_budget_results_are_uniform_budget_evaluation": False,
            },
            "offline_audit": dict(audit),
        }


def grading_records(root, rows):
    _, pool = load_enabled_benchmarks(Path(root) / "benchmarks/manifest.json")
    records = load_hard_pilot(Path(root) / "benchmarks/hard_pilot.json", pool)
    by_id = {r.prompt_id: r for r in records}
    for row in rows:
        record = by_id[row["prompt_id"]]
        reference = (
            record.reference_answer
            if record.task_type != "code"
            else json.dumps(list(record.tests), sort_keys=True)
        )
        if (
            render_prompt(record) != row["prompt"]
            or record.task_type != row["task_type"]
            or record.benchmark_name != row["benchmark"]
            or reference != row["reference_answer"]
        ):
            raise ValueError("Treatment prompt/benchmark/grading inputs differ from frozen pair")
    return by_id
