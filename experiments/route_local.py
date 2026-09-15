#!/usr/bin/env python3
"""Route prompt text locally with frozen artifacts. Does not execute an LLM call."""

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from routing_ml.local_router import LocalRouter


def read_requests(args):
    if args.prompt is not None:
        return [{"prompt_id": "request_0", "prompt": args.prompt}]
    rows = []
    for number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Input line {number} is not valid JSON: {error.msg}") from error
        if not isinstance(row, dict) or not {"prompt_id", "prompt"}.issubset(row):
            raise ValueError(f"Input line {number} requires prompt_id and prompt fields")
        rows.append({"prompt_id": row["prompt_id"], "prompt": row["prompt"]})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/router_v2/standard")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="One prompt to route")
    source.add_argument("--input", type=Path, help="JSONL records with prompt_id and prompt")
    args = parser.parse_args(argv)

    def blocked(*args, **kwargs):
        raise RuntimeError("This router performs local inference only")

    with ExitStack() as stack:
        for target in (
            "socket.socket.connect",
            "socket.socket.connect_ex",
            "socket.create_connection",
            "socket.getaddrinfo",
            "urllib.request.urlopen",
        ):
            stack.enter_context(patch(target, side_effect=blocked))
        try:
            rows = read_requests(args)
            results = LocalRouter(args.artifact_dir).predict(
                pd.DataFrame(rows, columns=["prompt_id", "prompt"])
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        for result in results:
            print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
