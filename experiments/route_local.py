#!/usr/bin/env python3
"""Route prompt text locally with frozen artifacts. Does not execute an LLM call."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from routing_ml.local_router import LocalRouter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/router_v2/standard")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="One prompt to route")
    source.add_argument("--input", type=Path, help="JSONL records with prompt_id and prompt")
    args = parser.parse_args()
    rows = ([dict(prompt_id="request_0", prompt=args.prompt)] if args.prompt is not None else
            [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()])
    def blocked(*args, **kwargs):
        raise RuntimeError("This router performs local inference only")
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection", "socket.getaddrinfo", "urllib.request.urlopen"):
            stack.enter_context(patch(target, side_effect=blocked))
        for result in LocalRouter(args.artifact_dir).predict(pd.DataFrame(rows, columns=["prompt_id", "prompt"])):
            print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
