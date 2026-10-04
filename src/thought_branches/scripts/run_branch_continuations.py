#!/usr/bin/env python3
"""Validate or run native reasoning-prefix branch jobs through DeepSeek."""

from __future__ import annotations

import argparse
from pathlib import Path

from thought_branches.branch_generation import (
    branch_request,
    run_native_branch_jobs,
    validate_identical_branch_inputs,
)
from thought_branches.io_utils import read_jsonl
from thought_branches.paths import OUTPUT_API


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        type=Path,
        default=OUTPUT_API / "branch_continuation_jobs.jsonl",
        help="Native reasoning-prefix branch jobs JSONL.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_API / "branch_continuation_generations.jsonl",
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=OUTPUT_API / "branch_continuation_failures.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--reasoning-effort", choices=["low", "high", "max"], default="low")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Make paid DeepSeek API calls. Without this flag, only validate request payloads.",
    )
    args = parser.parse_args()

    jobs = read_jsonl(args.jobs)
    requests = [
        branch_request(
            job,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            top_p=args.top_p,
        )
        for job in jobs
    ]
    validate_identical_branch_inputs(jobs, requests)
    if not args.live:
        print(f"validated {len(jobs)} native DeepSeek branch request payloads")
        print("no API requests were made; pass --live to execute the prepared jobs")
        return

    written, failed = run_native_branch_jobs(
        jobs=jobs,
        generations_path=args.out,
        failures_path=args.failures,
        limit=args.limit,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        top_p=args.top_p,
    )
    print(f"wrote {written} native branch generations to {args.out}")
    if failed:
        print(f"logged {failed} failures to {args.failures}")


if __name__ == "__main__":
    main()
