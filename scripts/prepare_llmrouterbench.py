#!/usr/bin/env python3
"""Prepare the frozen official public benchmark; --download is public data only."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_data.prepare import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download the hash-pinned official archive if absent")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/llmrouterbench")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/llmrouterbench")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    prepare(args.raw_dir, args.output_dir, args.report_dir, download=args.download)


if __name__ == "__main__":
    main()
