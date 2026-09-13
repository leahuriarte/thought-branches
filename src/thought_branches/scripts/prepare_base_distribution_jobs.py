#!/usr/bin/env python3
"""Prepare base-distribution jobs for best-effort, utility, and role-play conditions."""

from __future__ import annotations

import argparse
from pathlib import Path

from thought_branches.jobs import build_generation_jobs, parse_set, write_generation_jobs_file
from thought_branches.paths import GENERATION_JOBS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", default="all", help="Comma-separated conditions, or all.")
    parser.add_argument("--actors", default="all", help="Comma-separated actor ids, or all.")
    parser.add_argument("--tasks", default="all", help="Comma-separated task ids, or all.")
    parser.add_argument("--items-per-task", type=int, default=2, help="Limit items per task.")
    parser.add_argument("--pairs-per-actor-domain", type=int, default=2, help="Utility pairs per actor/domain.")
    parser.add_argument("--samples-per-condition", type=int, default=1, help="Samples per condition/item.")
    parser.add_argument("--out", type=Path, default=GENERATION_JOBS, help="Output JSONL path.")
    args = parser.parse_args()

    jobs = build_generation_jobs(
        conditions=parse_set(args.conditions),
        actors=parse_set(args.actors),
        tasks=parse_set(args.tasks),
        items_per_task=args.items_per_task,
        pairs_per_actor_domain=args.pairs_per_actor_domain,
        samples_per_condition=args.samples_per_condition,
    )
    write_generation_jobs_file(jobs, args.out)
    print(f"wrote {len(jobs)} generation jobs to {args.out}")


if __name__ == "__main__":
    main()
