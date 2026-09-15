"""Read only pinned public JSON archives, without executing upstream code."""

import hashlib
import json
import subprocess
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

import pandas as pd

from routing_data.splits import digest

# One observed, excluded SWE-bench directory has a stale model name. This is
# a path exception, never a general model-ID alias or outcome merge.
PATH_MODEL_EXCEPTIONS = {
    "bench-release/swe-bench/verified/qwen3-235b-a22b-thinking/SWE-Bench-verified_qwen3-235b-a22b-thinking-2507-20251026_124335.json": (
        "swe-bench",
        "verified",
        "qwen3-235b-a22b-thinking-2507",
    )
}


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path, spec):
    if not path.is_file() or path.stat().st_size != spec["size_bytes"]:
        raise ValueError(f"Missing or wrong-sized source: {path}")
    if file_hash(path) != spec["sha256"]:
        raise ValueError(f"Source SHA-256 mismatch: {path}")


def download_archive(raw_dir, config, download=False):
    spec = config["archive"]
    path = Path(raw_dir) / spec["filename"]
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"{path} missing; pass --download to fetch the pinned public archive"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        # No shell, environment credentials, HF token, or provider SDK needed.
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "3",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                spec["url"],
            ],
            check=True,
        )
        verify_file(partial, spec)
        partial.replace(path)
    verify_file(path, spec)
    return path


def output_failure(raw):
    text = str(raw or "").strip().lower()
    return text.startswith(
        ("processing failed:", "generation failed:", "failed to generate", "error:")
    )


def read_archive(archive, task_config, models):
    """Stream all 7 GB JSON payloads, retaining no completions or answer keys.

    All raw scores/resources survive in compact audit rows. Only selected request
    texts are retained. Archive members are read as bytes, never extracted.
    """
    tasks = {(t["dataset"], t["source_split"]): t for t in task_config["tasks"]}
    rows, selected, files = [], [], []
    fields, header_fields, extra_fields = Counter(), Counter(), Counter()
    member_names = set()
    for_model = set(models)
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            parts = PurePosixPath(member.name).parts
            if ".." in parts or member.name.startswith("/") or member.issym() or member.islnk():
                raise ValueError("Unsafe archive member")
            if not member.isfile():
                continue
            if member.name in member_names:
                raise ValueError("Duplicate archive member")
            member_names.add(member.name)
            if (
                len(parts) not in (4, 5)
                or parts[0] != "bench-release"
                or not member.name.endswith(".json")
            ):
                raise ValueError(f"Unexpected release member: {member.name}")
            payload = bundle.extractfile(member).read()
            data = json.loads(payload)
            ds, split, model = data["dataset_name"], data["split"], data["model_name"]
            path_identity = (ds, split, model) if len(parts) == 5 else (ds, model)
            path_agrees = tuple(parts[1:-1]) == path_identity
            if not path_agrees and PATH_MODEL_EXCEPTIONS.get(member.name) != (ds, split, model):
                raise ValueError(f"Inconsistent model/dataset identifiers: {member.name}")
            if len(parts) == 4 and ds not in {
                "arenahard",
                "arenahard_coding",
                "arenahard_math",
                "arenahard_creative_writing",
            }:
                raise ValueError("Unexpected source layout without split directory")
            if not isinstance(data["records"], list):
                raise ValueError("Expected records list")
            header_fields.update(data.keys())
            cost_sum = 0.0
            for record in data["records"]:
                fields.update(record.keys())
                extra_fields.update((record.get("extra_fields") or {}).keys())
                prompt, origin = record.get("prompt"), record.get("origin_query")
                if not isinstance(prompt, str) or not isinstance(origin, str):
                    raise ValueError("Expected prompt/origin strings")
                index = record["index"]
                if isinstance(index, bool) or not isinstance(index, int):
                    raise ValueError("Expected integral record index")
                cost = record.get("cost")
                if cost is not None:
                    cost_sum += cost
                row = dict(
                    dataset=ds,
                    source_split=split,
                    model_id=model,
                    source_index=index,
                    prompt_sha256=digest(prompt),
                    origin_sha256=digest(origin),
                    raw_score=record.get("score"),
                    raw_cost_usd=cost,
                    input_tokens=record.get("prompt_tokens"),
                    output_tokens=record.get("completion_tokens"),
                    has_output=bool(record.get("raw_output")),
                    generation_failure=output_failure(record.get("raw_output")),
                    source_file=member.name,
                    demo=bool(data.get("demo", False)),
                )
                rows.append(row)
                if (ds, split) in tasks and model in for_model and not row["demo"]:
                    selected.append(
                        dict(
                            row,
                            prompt=prompt,
                            origin_query=origin,
                            prompt_id=f"llmrouterbench:{ds}:{row['prompt_sha256']}",
                            task_type=tasks[ds, split]["task_type"],
                        )
                    )
            files.append(
                dict(
                    source_file=member.name,
                    dataset=ds,
                    source_split=split,
                    model_id=model,
                    path_identity_consistent=path_agrees,
                    bytes=member.size,
                    sha256=hashlib.sha256(payload).hexdigest(),
                    records=len(data["records"]),
                    reported_counts=data["counts"],
                    demo=bool(data.get("demo", False)),
                    data_fingerprint=data.get("data_fingerprint"),
                    header_cost_usd=data.get("cost"),
                    record_cost_sum_usd=cost_sum,
                )
            )
            if len(files) % 100 == 0:
                print(f"Audited {len(files)} result files / {len(rows):,} records", flush=True)
    raw = pd.DataFrame(rows).sort_values(["source_file", "source_index"]).reset_index(drop=True)
    inventory = pd.DataFrame(files).sort_values("source_file").reset_index(drop=True)
    if inventory.duplicated(["dataset", "source_split", "model_id"]).any():
        raise ValueError("Duplicate runs: pin explicit source files before proceeding")
    stats = dict(
        record_fields=dict(sorted(fields.items())),
        header_fields=dict(sorted(header_fields.items())),
        extra_fields=dict(sorted(extra_fields.items())),
    )
    return raw, pd.DataFrame(selected), inventory, stats
