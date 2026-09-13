#!/usr/bin/env python3
"""Chunk generated final outputs or readable reasoning traces into JSONL rows."""

from __future__ import annotations

import argparse
from pathlib import Path

from thought_branches.chunking import split_text_into_chunks
from thought_branches.io_utils import read_jsonl, write_jsonl
from thought_branches.paths import CHUNKS, GENERATIONS


def text_for_source(generation: dict, source: str) -> str:
    if source == "output":
        return str(generation.get("output_text", ""))
    if source == "reasoning":
        return str(generation.get("reasoning_text", ""))
    raise ValueError(f"unknown chunk source: {source!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, default=GENERATIONS, help="Input generations JSONL.")
    parser.add_argument("--out", type=Path, default=CHUNKS, help="Output chunks JSONL.")
    parser.add_argument(
        "--source",
        choices=["output", "reasoning"],
        default="output",
        help="Chunk final output text or extracted readable reasoning text.",
    )
    parser.add_argument("--min-chars", type=int, default=24)
    parser.add_argument("--max-chars", type=int, default=700)
    args = parser.parse_args()

    rows = []
    for generation in read_jsonl(args.generations):
        text = text_for_source(generation, args.source)
        if not text.strip():
            continue
        for chunk in split_text_into_chunks(text, min_chars=args.min_chars, max_chars=args.max_chars):
            rows.append(
                {
                    "output_id": generation.get("output_id", ""),
                    "job_id": generation.get("job_id", ""),
                    "run_id": generation.get("run_id", ""),
                    "actor": generation.get("actor", ""),
                    "condition": generation.get("condition", ""),
                    "task": generation.get("task", ""),
                    "item_id": generation.get("item_id", ""),
                    "sample_index": generation.get("sample_index", ""),
                    "chunk_source": args.source,
                    "chunk_index": chunk.index,
                    "char_start": chunk.start,
                    "char_end": chunk.end,
                    "chunk_text": chunk.text,
                }
            )
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} chunks to {args.out}")


if __name__ == "__main__":
    main()
