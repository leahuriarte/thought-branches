#!/usr/bin/env python3
"""Prepare local SGLang branches from locally generated and labeled traces."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path

from thought_branches.branching import MOTIVATED_BRANCH, UNMOTIVATED_BRANCH
from thought_branches.io_utils import read_jsonl, write_jsonl
from thought_branches.native_branching import first_llm_motivation_chunk, MOTIVATION_COMMITMENT_FLAG
from thought_branches.paths import OUTPUT_API
from thought_branches.sglang_backend import (
    build_sglang_branch_job, reasoning_ids_and_text, sampling_params, token_count_at_text_offset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True, help="Local SGLang base generation JSONL")
    parser.add_argument("--labeled-chunks", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUTPUT_API / "sglang_branch_jobs.jsonl")
    parser.add_argument("--samples-per-branch", type=int, default=1)
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.samples_per_branch < 1:
        parser.error("--samples-per-branch must be positive")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers on the generation host to load the local tokenizer") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    chunks_by_output = defaultdict(list)
    for chunk in read_jsonl(args.labeled_chunks):
        chunks_by_output[str(chunk.get("output_id", ""))].append(chunk)

    jobs = []
    seeds = 0
    for source in read_jsonl(args.generations):
        if source.get("provenance", {}).get("backend") != "sglang_native_generate":
            continue
        trace = str(source.get("reasoning_text", ""))
        reasoning_ids = source.get("reasoning_token_ids", [])
        if not trace or not reasoning_ids:
            continue
        chunks = chunks_by_output.get(str(source.get("output_id", "")), [])
        if not chunks:
            continue
        selected = first_llm_motivation_chunk(chunks)
        if selected is None:
            continue
        start, end = int(selected["char_start"]), int(selected["char_end"])
        if trace[start:end].strip() != str(selected.get("chunk_text", "")).strip():
            raise ValueError(f"chunk offsets do not match local source trace {source.get('output_id')}")
        boundaries = {
            UNMOTIVATED_BRANCH: start,
            MOTIVATED_BRANCH: end,
        }
        for branch, char_offset in boundaries.items():
            token_count = token_count_at_text_offset(tokenizer, reasoning_ids, trace, char_offset)
            prefix_ids = reasoning_ids[:token_count]
            if not prefix_ids:
                continue
            for sample_index in range(args.samples_per_branch):
                job_seed = int.from_bytes(hashlib.sha256(
                    f"{args.seed}:{source.get('output_id')}:{branch}:{sample_index}".encode()
                ).digest()[:4], "big")
                params = sampling_params(
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                    top_p=args.top_p, seed=job_seed,
                )
                jobs.append(build_sglang_branch_job(
                    source=source, branch_label=branch, prefix_token_ids=prefix_ids,
                    continuation_sample_index=sample_index, sampling=params,
                ))
        seeds += 1
        if args.max_seeds is not None and seeds >= args.max_seeds:
            break
    write_jsonl(args.out, jobs)
    print(f"wrote {len(jobs)} token-ID SGLang branch jobs from {seeds} same-backend seeds to {args.out}")


if __name__ == "__main__":
    main()
