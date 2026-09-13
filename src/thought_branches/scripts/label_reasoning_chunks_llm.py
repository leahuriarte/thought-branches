#!/usr/bin/env python3
"""Classify reasoning chunks with an LLM DAG-style labeler."""

from __future__ import annotations

import argparse
from pathlib import Path

from thought_branches.io_utils import read_jsonl, write_jsonl
from thought_branches.llm_labeling import label_chunk_groups
from thought_branches.paths import GENERATIONS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, default=GENERATIONS)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labeler-model", default="deepseek/deepseek-v3.2")
    parser.add_argument("--max-tokens", type=int, default=3500)
    args = parser.parse_args()

    rows = label_chunk_groups(
        generations=read_jsonl(args.generations),
        chunks=read_jsonl(args.chunks),
        model=args.labeler_model,
        max_tokens=args.max_tokens,
    )
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} LLM-labeled reasoning chunks to {args.out}")


if __name__ == "__main__":
    main()
