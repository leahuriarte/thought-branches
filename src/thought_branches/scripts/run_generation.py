#!/usr/bin/env python3
"""Run prepared generation jobs through OpenRouter."""

from __future__ import annotations

import argparse
from pathlib import Path

from thought_branches.generation import run_jobs
from thought_branches.io_utils import read_jsonl
from thought_branches.paths import GENERATION_FAILURES, GENERATION_JOBS, GENERATIONS


REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


def reasoning_config(effort: str) -> dict[str, str | bool] | None:
    if effort == "none":
        return None
    return {"effort": effort, "exclude": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=GENERATION_JOBS, help="Generation jobs JSONL.")
    parser.add_argument("--out", type=Path, default=GENERATIONS, help="Successful generations JSONL.")
    parser.add_argument("--failures", type=Path, default=GENERATION_FAILURES, help="Generation failures JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum new jobs to attempt.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument(
        "--reasoning-effort",
        choices=sorted(REASONING_EFFORTS),
        default="none",
        help="Request readable reasoning tokens when supported. Use low/medium/high for trace collection.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write deterministic placeholders without API calls.")
    args = parser.parse_args()

    jobs = read_jsonl(args.jobs)
    written, failed = run_jobs(
        jobs=jobs,
        generations_path=args.out,
        failures_path=args.failures,
        dry_run=args.dry_run,
        limit=args.limit,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning=reasoning_config(args.reasoning_effort),
    )
    print(f"wrote {written} generations to {args.out}")
    if failed:
        print(f"logged {failed} failures to {args.failures}")


if __name__ == "__main__":
    main()
