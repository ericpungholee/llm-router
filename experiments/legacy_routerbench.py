"""Historical RouterBench preparation script from the first project iteration."""

import ast
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


DATASET_REPO = "withmartian/routerbench"
DATASET_FILE = "routerbench_0shot.pkl"
WEAK_MODEL = "mistralai/mistral-7b-chat"
STRONG_MODEL = "gpt-4-1106-preview"


def clean_prompt(raw_prompt: str) -> str:
    try:
        prompt_parts = ast.literal_eval(raw_prompt)
    except (SyntaxError, ValueError):
        return raw_prompt
    if isinstance(prompt_parts, (list, tuple)):
        return "\n\n".join(str(part) for part in prompt_parts)
    return str(prompt_parts)


def main() -> None:
    base_dir = Path("data/legacy/routerbench")
    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=DATASET_FILE,
        repo_type="dataset",
        local_dir=raw_dir,
    )
    dataset = pd.read_pickle(dataset_path)

    cost_fields = [column for column in dataset.columns if column.endswith("|total_cost")]
    model_names = [column.removesuffix("|total_cost") for column in cost_fields]
    performance_fields = [model for model in model_names if model in dataset.columns]

    print(f"Dataset: {DATASET_REPO}/{DATASET_FILE}")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Columns: {dataset.columns.tolist()}")
    print(f"Model names: {model_names}")
    print(f"Performance fields: {performance_fields}")
    print(f"Cost fields: {cost_fields}")
    print("\nOne full sample (row 0):")
    with pd.option_context("display.max_columns", None, "display.max_colwidth", None):
        print(dataset.iloc[0].to_string())

    weak_scores = pd.to_numeric(dataset[WEAK_MODEL], errors="raise")
    strong_scores = pd.to_numeric(dataset[STRONG_MODEL], errors="raise")
    clean = pd.DataFrame(
        {
            "prompt": dataset["prompt"].map(clean_prompt),
            "weak_model_score": weak_scores,
            "strong_model_score": strong_scores,
            "weak_model_cost": pd.to_numeric(
                dataset[f"{WEAK_MODEL}|total_cost"], errors="raise"
            ),
            "strong_model_cost": pd.to_numeric(
                dataset[f"{STRONG_MODEL}|total_cost"], errors="raise"
            ),
            "routing_label": (strong_scores > weak_scores).map(
                {True: "strong", False: "weak"}
            ),
        }
    )
    output_path = processed_dir / "routerbench_binary.csv"
    clean.to_csv(output_path, index=False)
    print(f"Wrote {len(clean):,} historical rows to {output_path}")


if __name__ == "__main__":
    main()

