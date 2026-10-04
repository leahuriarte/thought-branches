#!/usr/bin/env python3
"""Prepare native DeepSeek branch jobs from LLM-classified reasoning chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

from thought_branches.deepseek import DIRECT_DEEPSEEK_MODELS
from thought_branches.io_utils import read_jsonl, write_jsonl
from thought_branches.native_branching import build_native_branch_jobs
from thought_branches.paths import GENERATIONS, OUTPUT_API


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, default=GENERATIONS, help="Input source generations.")
    parser.add_argument(
        "--labeled-chunks",
        type=Path,
        required=True,
        help=(
            "Reasoning chunks labeled by tb-label-reasoning-llm using "
            "llm_dag_v3_motivation_commitment."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_API / "branch_continuation_jobs.jsonl",
        help="Output native branch jobs JSONL.",
    )
    parser.add_argument("--model", choices=sorted(DIRECT_DEEPSEEK_MODELS), default="deepseek-flash")
    parser.add_argument("--samples-per-branch", type=int, default=1)
    parser.add_argument("--max-seeds", type=int, default=None, help="Maximum source outputs to branch.")
    parser.add_argument("--min-pre-prefix-chars", type=int, default=1)
    args = parser.parse_args()

    jobs = build_native_branch_jobs(
        generations=read_jsonl(args.generations),
        labeled_chunks=read_jsonl(args.labeled_chunks),
        target_model=args.model,
        samples_per_branch=args.samples_per_branch,
        max_seeds=args.max_seeds,
        min_pre_prefix_chars=args.min_pre_prefix_chars,
    )
    write_jsonl(args.out, jobs)
    print(f"wrote {len(jobs)} native DeepSeek branch jobs to {args.out}")
    print("no API requests were made; run tb-run-branch-continuations --live after reviewing the jobs")


if __name__ == "__main__":
    main()
