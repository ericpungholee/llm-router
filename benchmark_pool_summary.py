"""Print the hard candidate pool and pilot subset without provider calls."""

from collections import Counter
from pathlib import Path

from benchmark_loaders import load_enabled_benchmarks, load_hard_pilot

MANIFEST = Path("benchmarks/manifest.json")
HARD_PILOT = Path("benchmarks/hard_pilot.json")


def deterministic_grading_issue(record):
    if record.task_type == "multiple_choice" and (
        not record.choices or record.reference_answer not in record.choices
    ):
        return "missing choices or invalid reference option"
    if record.task_type == "math" and not record.reference_answer.strip():
        return "missing exact reference answer"
    if record.task_type == "code" and not record.tests:
        return "missing stdin tests"
    return None


def main() -> None:
    _, records = load_enabled_benchmarks(MANIFEST)
    hard_pilot = load_hard_pilot(HARD_PILOT, records)
    benchmark_counts = Counter(record.benchmark_name for record in records)
    difficulty_counts = Counter((record.benchmark_name, record.difficulty) for record in records)
    split_counts = Counter((record.benchmark_name, record.source_split) for record in records)
    issues = [
        (record.prompt_id, issue)
        for record in records
        if (issue := deterministic_grading_issue(record)) is not None
    ]

    print(f"Candidate prompts: {len(records)}")
    print("By benchmark:")
    for benchmark, count in sorted(benchmark_counts.items()):
        print(f"- {benchmark}: {count}")
    print("Difficulty metadata:")
    for (benchmark, difficulty), count in sorted(difficulty_counts.items()):
        print(f"- {benchmark} / {difficulty}: {count}")
    print("Source splits:")
    for (benchmark, split), count in sorted(split_counts.items()):
        print(f"- {benchmark} / {split}: {count}")
    print("Hard-pilot prompt IDs:")
    for record in hard_pilot:
        date = f" / {record.problem_date}" if record.problem_date else ""
        print(f"- {record.prompt_id} / {record.difficulty} / {record.task_category}{date}")
    print("Deterministic grading issues for stored references/tests:")
    if issues:
        for prompt_id, issue in issues:
            print(f"- {prompt_id}: {issue}")
    else:
        print("- none")
    print(
        "Coding test scope: official public stdin cases; compressed private "
        "LiveCodeBench tests are not vendored."
    )
    print("No provider API calls were made.")


if __name__ == "__main__":
    main()
