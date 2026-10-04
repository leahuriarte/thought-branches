#!/usr/bin/env python3
"""Run prepared native token-ID SGLang branch jobs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from thought_branches.io_utils import read_jsonl
from thought_branches.paths import OUTPUT_API
from thought_branches.sglang_backend import SGLangClient, run_sglang_jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=OUTPUT_API / "sglang_branch_jobs.jsonl")
    parser.add_argument("--out", type=Path, default=OUTPUT_API / "sglang_branch_generations.jsonl")
    parser.add_argument("--failures", type=Path, default=OUTPUT_API / "sglang_branch_failures.jsonl")
    parser.add_argument("--base-url", default=os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    jobs = read_jsonl(args.jobs)
    for job in jobs:
        if job.get("continuation_mode") != "sglang_token_prefix":
            raise ValueError(f"job {job.get('job_id')} is not an SGLang token-prefix job")
        expected = job.get("conversation_token_ids", []) + job.get("prefix_token_ids", [])
        if job.get("input_token_ids") != expected:
            raise ValueError(f"job {job.get('job_id')} input IDs do not equal conversation + exact prefix IDs")
    provenance = jobs[0].get("source_provenance", {}) if jobs else {"backend": "sglang_native_generate"}
    client = SGLangClient(args.base_url)
    written, failed = run_sglang_jobs(
        jobs=jobs, client=client, output_path=args.out, failures_path=args.failures,
        provenance=provenance, concurrency=args.concurrency, retries=args.retries, limit=args.limit,
    )
    print(f"wrote {written} local token-prefix continuations to {args.out}; failures: {failed}")


if __name__ == "__main__":
    main()
