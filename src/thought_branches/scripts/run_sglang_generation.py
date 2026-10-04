#!/usr/bin/env python3
"""Generate local base traces through SGLang's token-ID endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from thought_branches.io_utils import read_jsonl
from thought_branches.paths import GENERATION_JOBS, OUTPUT_API
from thought_branches.sglang_backend import (
    SGLangClient, chat_template_token_ids, run_sglang_jobs, runtime_provenance,
    sampling_params, tokenizer_file_hash,
)


def _model_manifest_hash(model_path: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in model_path.rglob("*") if p.is_file()):
        stat = path.stat()
        digest.update(f"{path.relative_to(model_path)}\0{stat.st_size}\n".encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=GENERATION_JOBS)
    parser.add_argument("--out", type=Path, default=OUTPUT_API / "sglang_generations.jsonl")
    parser.add_argument("--failures", type=Path, default=OUTPUT_API / "sglang_failures.jsonl")
    parser.add_argument("--base-url", default=os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", default="/workspace/models/Qwen3.5-9B")
    parser.add_argument("--model-revision", default="local-unpinned")
    parser.add_argument("--tokenizer-revision", default="local-unpinned")
    parser.add_argument("--gpu-type", default="NVIDIA L40S")
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers on the generation host to load the local tokenizer") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    client = SGLangClient(args.base_url)
    settings = {
        "host": "127.0.0.1", "port": 30000, "tp_size": 1, "mem_fraction_static": 0.8,
        "context_length": args.context_limit, "max_running_requests": 8,
        "reasoning_parser": "qwen3", "thinking_enabled": True,
    }
    provenance = runtime_provenance(
        client, model_id=args.model_id, model_revision=args.model_revision,
        model_hash=_model_manifest_hash(args.model_path), tokenizer_revision=args.tokenizer_revision,
        tokenizer_hash=tokenizer_file_hash(args.model_path), gpu_type=args.gpu_type,
        context_limit=args.context_limit, server_settings=settings,
    )
    provenance["model_id"] = args.model_id
    provenance["model_hash_kind"] = "sha256_of_relative_paths_and_file_sizes"
    jobs: list[dict[str, Any]] = []
    for source in read_jsonl(args.jobs):
        if source.get("continuation_mode") == "native_reasoning_prefix":
            raise ValueError("refusing DeepSeek native branch jobs; prepare local SGLang jobs from local base traces")
        messages = []
        if str(source.get("system_prompt", "")).strip():
            messages.append({"role": "system", "content": str(source["system_prompt"])})
        messages.append({"role": "user", "content": str(source["prompt"])})
        prompt_ids = chat_template_token_ids(tokenizer, messages)
        seed_digest = hashlib.sha256(f"{args.seed}:{source['job_id']}".encode()).digest()
        job_seed = int.from_bytes(seed_digest[:4], "big")
        params = sampling_params(
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, seed=job_seed,
        )
        identity = f"sglang\n{args.model_id}\n{source['job_id']}\n{json.dumps(params, sort_keys=True)}"
        job_id = "sglang:" + hashlib.sha256(identity.encode()).hexdigest()[:20]
        jobs.append({
            "job_id": job_id, "source_job_id": source["job_id"], "model": args.model_id,
            "condition": source.get("condition", ""), "messages": messages,
            "prompt_token_ids": prompt_ids, "input_token_ids": prompt_ids, "prefix_token_ids": [],
            "sampling_params": params, "job": source,
        })
    written, failed = run_sglang_jobs(
        jobs=jobs, client=client, output_path=args.out, failures_path=args.failures,
        provenance=provenance, concurrency=args.concurrency, retries=args.retries, limit=args.limit,
        tokenizer=tokenizer,
    )
    print(f"wrote {written} local SGLang generations to {args.out}; failures: {failed} ({args.failures})")


if __name__ == "__main__":
    main()
