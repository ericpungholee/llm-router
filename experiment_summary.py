"""Print the complete experiment plan without making any model API calls."""

import argparse
import os
from collections import Counter
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        path = Path(".env")
        if not path.exists():
            return False
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
        return True

from benchmark_loaders import load_enabled_benchmarks
from dataset_schema import RESULT_FIELDS
from model_registry import enabled_models, validate_registry
from providers import missing_api_keys, provider_display_name
from generate_dataset import planned_calls
from spend_control import (
    DEFAULT_RUN_SPEND_CAP_USD,
    SpendPreflightError,
    build_run_plan,
    global_max_spend_usd,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    validate_registry()
    specs, records = load_enabled_benchmarks(args.manifest)
    models = enabled_models()

    print("Experiment summary (read-only; no model API calls)\n")
    print("Enabled models:")
    for model in models:
        api_id = model.api_model_identifier or "UNRESOLVED"
        readiness = "live-ready metadata" if model.live_ready else "NOT live-ready"
        print(
            f"- {model.canonical_model_name} | creator={model.creator} | "
            f"provider={provider_display_name(model.inference_provider)} | type={model.model_type} | "
            f"api_id={api_id} | {readiness}"
        )

    counts = Counter(record.benchmark_name for record in records)
    print("\nEnabled benchmark samples:")
    for spec in specs:
        if spec.enabled:
            print(
                f"- {spec.name}: {counts[spec.name]} {spec.task_type} prompt(s) "
                f"from {spec.source}@{spec.source_revision}"
            )

    calls = planned_calls(records, models)
    print("\nEvaluation matrix:")
    print(f"- Enabled models: {len(models)}")
    print(f"- Benchmark prompts: {len(records)}")
    print(f"- Expected model calls: {len(calls)}")
    print(f"- Expected normalized result rows: {len(calls)}")
    print(f"- Result fields per row: {len(RESULT_FIELDS)}")

    print("\nLive-run spend confirmation preview:")
    print(f"- Planned calls: {len(calls)}")
    print(f"- Enabled models: {', '.join(model.canonical_model_name for model in models)}")
    print(f"- Configured spend cap: ${DEFAULT_RUN_SPEND_CAP_USD:.2f}")
    print(f"- Global maximum spend: ${global_max_spend_usd():.2f}")
    try:
        plan = build_run_plan(calls)
    except SpendPreflightError as error:
        print(f"- Estimated maximum cost: unavailable ({error})")
    else:
        print(f"- Estimated maximum cost: ${plan.estimated_max_cost_usd:.6f}")

    missing_keys = missing_api_keys(models)
    print("\nMissing API keys:")
    if missing_keys:
        for key_name in missing_keys:
            print(f"- {key_name}")
    else:
        print("- none")

    unresolved_ids = [model for model in models if model.api_model_identifier is None]
    print("\nModels with unresolved API identifiers:")
    if unresolved_ids:
        for model in unresolved_ids:
            print(f"- {model.canonical_model_name}: {model.resolution_note}")
    else:
        print("- none")

    unresolved_prices = [
        model
        for model in models
        if model.input_price_per_million_usd is None
        or model.output_price_per_million_usd is None
    ]
    print("\nModels with unresolved pricing:")
    if unresolved_prices:
        for model in unresolved_prices:
            print(f"- {model.canonical_model_name}: {model.resolution_note}")
    else:
        print("- none")

    print("\nRead-only summary complete; no provider API calls were made.")


if __name__ == "__main__":
    main()
