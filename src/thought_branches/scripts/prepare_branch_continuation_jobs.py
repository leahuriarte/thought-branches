#!/usr/bin/env python3
"""Prepare motivated/unmotivated branch-continuation jobs from readable reasoning traces."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from thought_branches.branching import (
    MOTIVATED_BRANCH,
    UNMOTIVATED_BRANCH,
    build_prefix_continuation_prompt,
    first_motivation_span,
    motivation_prefixes,
    readable_reasoning_text,
)
from thought_branches.io_utils import read_jsonl, write_jsonl
from thought_branches.paths import GENERATIONS, OUTPUT_API


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, default=GENERATIONS, help="Input generations with raw responses.")
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_API / "branch_continuation_jobs.jsonl",
        help="Output branch continuation jobs JSONL.",
    )
    parser.add_argument("--max-seeds", type=int, default=None, help="Maximum source outputs to branch.")
    parser.add_argument("--min-pre-prefix-chars", type=int, default=1)
    args = parser.parse_args()

    jobs = []
    seeds = 0
    for row in read_jsonl(args.generations):
        trace = readable_reasoning_text(row)
        span = first_motivation_span(trace) if trace else None
        if span is None:
            continue
        prefixes = motivation_prefixes(trace, span)
        if len(prefixes["before_motivation"]) < args.min_pre_prefix_chars:
            continue
        job = row.get("job", {})
        if not isinstance(job, dict) or not str(job.get("prompt", "")).strip():
            continue
        original_prompt = str(job["prompt"])
        branch_prompts = {
            MOTIVATED_BRANCH: build_prefix_continuation_prompt(
                original_prompt=original_prompt,
                reasoning_prefix=prefixes["through_motivation_segment"],
                branch_label=MOTIVATED_BRANCH,
            ),
            UNMOTIVATED_BRANCH: build_prefix_continuation_prompt(
                original_prompt=original_prompt,
                reasoning_prefix=prefixes["before_motivation"],
                branch_label=UNMOTIVATED_BRANCH,
            ),
        }
        for branch_label, prompt in branch_prompts.items():
            digest = hashlib.sha256(f"{row['output_id']}\n{branch_label}\n{prompt}".encode("utf-8")).hexdigest()[:10]
            jobs.append(
                {
                    "job_id": f"branch:{row['output_id']}:{branch_label}:v{digest}",
                    "source_output_id": row["output_id"],
                    "source_condition": row.get("condition", ""),
                    "branch_label": branch_label,
                    "actor": row.get("actor", ""),
                    "model": row.get("model", ""),
                    "condition": branch_label,
                    "system_prompt": job.get("system_prompt", ""),
                    "prompt": prompt,
                    "task": row.get("task", job.get("task", "")),
                    "task_label": job.get("task_label", ""),
                    "item_id": row.get("item_id", job.get("item_id", "")),
                    "item_index": job.get("item_index", ""),
                    "item_label": row.get("item_label", job.get("item_label", "")),
                    "base_prompt": job.get("base_prompt", ""),
                    "axis": job.get("axis", ""),
                    "axis_definition": job.get("axis_definition", ""),
                    "sample_index": row.get("sample_index", ""),
                    "motivation_category": span.category,
                    "motivation_start": span.start,
                    "motivation_end": span.end,
                    "source_job": job,
                }
            )
        seeds += 1
        if args.max_seeds is not None and seeds >= args.max_seeds:
            break
    write_jsonl(args.out, jobs)
    print(f"wrote {len(jobs)} branch continuation jobs to {args.out}")


if __name__ == "__main__":
    main()
