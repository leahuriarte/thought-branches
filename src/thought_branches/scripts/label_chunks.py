#!/usr/bin/env python3
"""Attach heuristic labels to chunk rows."""

from __future__ import annotations

import argparse
from pathlib import Path

from thought_branches.io_utils import read_jsonl, write_jsonl
from thought_branches.labeling import label_chunk, label_reason
from thought_branches.paths import CHUNKS, LABELED_CHUNKS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=CHUNKS, help="Input chunks JSONL.")
    parser.add_argument("--out", type=Path, default=LABELED_CHUNKS, help="Output labeled chunks JSONL.")
    args = parser.parse_args()

    rows = []
    for row in read_jsonl(args.chunks):
        labels = label_chunk(str(row.get("chunk_text", "")))
        labeled = dict(row)
        labeled["labels"] = labels
        labeled["label_reason"] = label_reason(str(row.get("chunk_text", "")), labels)
        rows.append(labeled)
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} labeled chunks to {args.out}")


if __name__ == "__main__":
    main()
