"""Manifest and source-specific loaders for deterministic benchmark samples."""

import json
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    task_type: str
    loader: str
    source: str
    source_revision: str
    sample_file: Path
    enabled: bool


@dataclass(frozen=True)
class BenchmarkRecord:
    prompt_id: str
    prompt: str
    task_type: str
    benchmark_name: str
    reference_answer: str
    choices: Optional[Mapping[str, str]] = None
    tests: Tuple[Mapping[str, str], ...] = ()
    mock_solution: Optional[str] = None
    task_category: str = ""
    difficulty: str = ""
    source_split: str = ""
    problem_date: Optional[str] = None


LOADER_TASK_TYPES = {
    "mmlu_pro": "multiple_choice",
    "math_500": "math",
    "livecodebench": "code",
}


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSON at {path}:{line_number}") from error
    if not rows:
        raise ValueError(f"No sample rows found in {path}")
    return rows


def _require_text(row: Mapping[str, object], field: str, source: str) -> str:
    value = row.get(field)
    if value is None or not str(value).strip():
        raise ValueError(f"{source} sample is missing {field}")
    return str(value)


def load_mmlu_pro(spec: BenchmarkSpec) -> List[BenchmarkRecord]:
    records = []
    for row in _read_jsonl(spec.sample_file):
        question_id = _require_text(row, "question_id", spec.name)
        question = _require_text(row, "question", spec.name)
        answer = _require_text(row, "answer", spec.name).upper()
        options = row.get("options")
        if not isinstance(options, list) or not options:
            raise ValueError(f"{spec.name} sample requires a non-empty options list")
        if len(options) > len(string.ascii_uppercase):
            raise ValueError(f"{spec.name} sample has too many options")
        choices = dict(zip(string.ascii_uppercase, map(str, options)))
        if answer not in choices:
            raise ValueError(f"{spec.name} answer {answer!r} is not a valid option")
        records.append(
            BenchmarkRecord(
                prompt_id=f"{spec.name}:{question_id}",
                prompt=question,
                task_type=spec.task_type,
                benchmark_name=spec.name,
                reference_answer=answer,
                choices=choices,
                task_category=str(row.get("category", "general")),
                difficulty=str(row.get("difficulty", "hard")),
                source_split=str(row.get("source_split", "test")),
            )
        )
    return records


def load_math_500(spec: BenchmarkSpec) -> List[BenchmarkRecord]:
    records = []
    for row in _read_jsonl(spec.sample_file):
        problem_id = _require_text(row, "unique_id", spec.name)
        problem = _require_text(row, "problem", spec.name)
        answer = _require_text(row, "answer", spec.name)
        records.append(
            BenchmarkRecord(
                prompt_id=f"{spec.name}:{problem_id}",
                prompt=f"{problem}\nReturn only the final answer.",
                task_type=spec.task_type,
                benchmark_name=spec.name,
                reference_answer=answer,
                task_category=str(row.get("subject", "mathematics")),
                difficulty=str(row.get("difficulty", f"level_{row.get('level', 'unknown')}")),
                source_split=str(row.get("source_split", "test")),
            )
        )
    return records


def load_livecodebench(spec: BenchmarkSpec) -> List[BenchmarkRecord]:
    records = []
    for row in _read_jsonl(spec.sample_file):
        question_id = _require_text(row, "question_id", spec.name)
        question = _require_text(row, "question_content", spec.name)
        raw_tests = row.get("public_test_cases")
        if isinstance(raw_tests, str):
            raw_tests = json.loads(raw_tests)
        if not isinstance(raw_tests, list) or not raw_tests:
            raise ValueError(f"{spec.name} sample requires executable public tests")
        tests = tuple(
            {"input": _require_text(test, "input", spec.name), "output": _require_text(test, "output", spec.name)}
            for test in raw_tests
        )
        mock_solution_value = row.get("mock_solution")
        mock_solution = (
            str(mock_solution_value)
            if mock_solution_value is not None and str(mock_solution_value).strip()
            else None
        )
        platform = str(row.get("platform", "competition"))
        records.append(
            BenchmarkRecord(
                prompt_id=f"{spec.name}:{question_id}",
                prompt=f"{question}\nReturn only a complete Python program.",
                task_type=spec.task_type,
                benchmark_name=spec.name,
                reference_answer=json.dumps(tests, ensure_ascii=False),
                tests=tests,
                mock_solution=mock_solution,
                task_category=f"competitive_programming:{platform}",
                difficulty=str(row.get("difficulty", "unknown")),
                source_split=str(row.get("source_split", "test")),
                problem_date=str(row["contest_date"]) if row.get("contest_date") else None,
            )
        )
    return records


LOADERS = {
    "mmlu_pro": load_mmlu_pro,
    "math_500": load_math_500,
    "livecodebench": load_livecodebench,
}


def load_manifest(path: Path) -> List[BenchmarkSpec]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("version") != 1 or not isinstance(manifest.get("benchmarks"), list):
        raise ValueError("Benchmark manifest must have version 1 and a benchmarks list")

    specs = []
    for entry in manifest["benchmarks"]:
        missing = [
            field
            for field in (
                "name",
                "task_type",
                "loader",
                "source",
                "source_revision",
                "sample_file",
                "enabled",
            )
            if field not in entry
        ]
        if missing:
            raise ValueError(f"Benchmark manifest entry is missing {missing}")
        loader = str(entry["loader"])
        if loader not in LOADERS:
            raise ValueError(f"Unknown benchmark loader: {loader}")
        task_type = str(entry["task_type"])
        if task_type != LOADER_TASK_TYPES[loader]:
            raise ValueError(
                f"Benchmark {entry['name']} has task_type {task_type!r}, but "
                f"loader {loader!r} requires {LOADER_TASK_TYPES[loader]!r}"
            )
        specs.append(
            BenchmarkSpec(
                name=str(entry["name"]),
                task_type=task_type,
                loader=loader,
                source=str(entry["source"]),
                source_revision=str(entry["source_revision"]),
                sample_file=path.parent / str(entry["sample_file"]),
                enabled=bool(entry["enabled"]),
            )
        )

    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("Benchmark manifest contains duplicate benchmark names")
    return specs


def validate_benchmark_records(records: Sequence[BenchmarkRecord]) -> None:
    if not records:
        raise ValueError("No benchmark prompts are enabled")
    prompt_ids = [record.prompt_id for record in records]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("Duplicate prompt IDs found across enabled benchmarks")
    for record in records:
        if not record.reference_answer.strip():
            raise ValueError(f"{record.prompt_id} is missing a reference answer")
        if record.task_type not in {"multiple_choice", "math", "code"}:
            raise ValueError(f"{record.prompt_id} has invalid task_type")
        for field_name in ("task_category", "difficulty", "source_split"):
            if not getattr(record, field_name).strip():
                raise ValueError(f"{record.prompt_id} is missing {field_name}")


def load_enabled_benchmarks(path: Path) -> Tuple[List[BenchmarkSpec], List[BenchmarkRecord]]:
    specs = load_manifest(path)
    records = []
    for spec in specs:
        if spec.enabled:
            records.extend(LOADERS[spec.loader](spec))
    validate_benchmark_records(records)
    return specs, records


def load_hard_pilot(
    path: Path, records: Sequence[BenchmarkRecord]
) -> List[BenchmarkRecord]:
    """Load an ordered, outcome-blind hard-pilot subset definition."""
    with path.open(encoding="utf-8") as handle:
        definition = json.load(handle)
    if definition.get("version") != 1 or not isinstance(definition.get("prompts"), dict):
        raise ValueError("Hard-pilot definition must have version 1 and prompt groups")
    ordered_ids = [
        str(prompt_id)
        for benchmark_name in ("mmlu_pro", "math_500", "livecodebench")
        for prompt_id in definition["prompts"].get(benchmark_name, [])
    ]
    if len(ordered_ids) != 15 or len(set(ordered_ids)) != 15:
        raise ValueError("Hard pilot must contain exactly 15 unique prompt IDs")
    records_by_id = {record.prompt_id: record for record in records}
    missing = [prompt_id for prompt_id in ordered_ids if prompt_id not in records_by_id]
    if missing:
        raise ValueError(f"Hard-pilot prompt IDs are missing from the candidate pool: {missing}")
    selected = [records_by_id[prompt_id] for prompt_id in ordered_ids]
    for benchmark_name in ("mmlu_pro", "math_500", "livecodebench"):
        count = sum(record.benchmark_name == benchmark_name for record in selected)
        if count != 5:
            raise ValueError(f"Hard pilot requires 5 {benchmark_name} prompts, found {count}")
    return selected
