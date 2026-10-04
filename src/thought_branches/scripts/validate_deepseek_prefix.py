#!/usr/bin/env python3
"""Prepare or run a small DeepSeek native reasoning-prefix fidelity check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thought_branches.deepseek import DIRECT_DEEPSEEK_MODELS, DeepSeekClient
from thought_branches.io_utils import read_jsonl, write_jsonl
from thought_branches.paths import INPUTS, OUTPUT_API
from thought_branches.prefix_validation import (
    DEFAULT_VALIDATION_CASES,
    attach_response,
    summarize_validation,
    validation_rows,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=INPUTS / "deepseek_prefix_validation_cases.jsonl",
        help="Optional JSONL cases. Built-in cases are used when this path does not exist.",
    )
    parser.add_argument("--model", choices=sorted(DIRECT_DEEPSEEK_MODELS), default="deepseek-flash")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", choices=["low", "high", "max"], default="low")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--out", type=Path, default=OUTPUT_API / "deepseek_prefix_validation.jsonl")
    parser.add_argument("--summary", type=Path, default=OUTPUT_API / "deepseek_prefix_validation_summary.json")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Make paid API requests. Without this flag, only the exact request payloads are written.",
    )
    args = parser.parse_args()

    cases = read_jsonl(args.cases) if args.cases.exists() else DEFAULT_VALIDATION_CASES
    rows = validation_rows(
        cases=cases,
        model=args.model,
        repeats=args.repeats,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        top_p=args.top_p,
    )
    if args.live:
        client = DeepSeekClient()
        completed = []
        for row in rows:
            try:
                response = client.reasoning_prefix_completion(row["request"])
                completed.append(attach_response(row, response))
            except Exception as exc:  # noqa: BLE001 - validation artifacts should retain per-request failures.
                failed = dict(row)
                failed.update({"status": "failure", "error": str(exc)})
                completed.append(failed)
        rows = completed

    write_jsonl(args.out, rows)
    summary = summarize_validation(rows, model=args.model)
    summary["mode"] = "live" if args.live else "prepared_only"
    write_json(args.summary, summary)
    print(f"wrote {len(rows)} {'live results' if args.live else 'prepared requests'} to {args.out}")
    print(f"wrote validation summary to {args.summary}")
    if not args.live:
        print("no API requests were made; pass --live only after reviewing the prepared payloads")


if __name__ == "__main__":
    main()
