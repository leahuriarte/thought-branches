"""Run native reasoning-prefix branch jobs through DeepSeek's beta endpoint."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from thought_branches.deepseek import DeepSeekClient, build_reasoning_prefix_request, response_parts
from thought_branches.io_utils import append_jsonl, read_jsonl
from thought_branches.native_branching import NATIVE_REASONING_PREFIX_MODE


def branch_request(job: dict[str, Any], *, max_tokens: int, reasoning_effort: str, top_p: float) -> dict[str, Any]:
    if job.get("continuation_mode") != NATIVE_REASONING_PREFIX_MODE:
        raise ValueError(f"job {job.get('job_id', '')!r} is not a native reasoning-prefix job")
    messages: list[dict[str, str]] = []
    system_prompt = str(job.get("system_prompt", ""))
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": str(job["original_prompt"])})
    return build_reasoning_prefix_request(
        model=str(job["model"]),
        messages=messages,
        reasoning_prefix=str(job["reasoning_prefix"]),
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        top_p=top_p,
    )


def validate_identical_branch_inputs(jobs: list[dict[str, Any]], requests: list[dict[str, Any]]) -> None:
    """Require repeated samples from one source/branch to have identical API payloads."""
    if len(jobs) != len(requests):
        raise ValueError("jobs and requests must have the same length")
    digests_by_branch: dict[tuple[str, str], set[str]] = {}
    for job, request in zip(jobs, requests):
        key = (str(job.get("source_output_id", "")), str(job.get("branch_label", "")))
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()
        digests_by_branch.setdefault(key, set()).add(digest)
    mismatched = [key for key, digests in digests_by_branch.items() if len(digests) != 1]
    if mismatched:
        raise ValueError(f"repeated branch samples do not have identical request payloads: {mismatched}")


def existing_output_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("output_id", "")) for row in read_jsonl(path) if row.get("success") is True}


def run_native_branch_jobs(
    *,
    jobs: list[dict[str, Any]],
    generations_path: Path,
    failures_path: Path,
    limit: int | None,
    max_tokens: int,
    reasoning_effort: str,
    top_p: float,
) -> tuple[int, int]:
    requests = [
        branch_request(job, max_tokens=max_tokens, reasoning_effort=reasoning_effort, top_p=top_p)
        for job in jobs
    ]
    validate_identical_branch_inputs(jobs, requests)
    done = existing_output_ids(generations_path)
    client = DeepSeekClient()
    written = 0
    failed = 0
    attempted = 0
    for job, request in zip(jobs, requests):
        output_id = str(job["job_id"])
        if output_id in done:
            continue
        if limit is not None and attempted >= limit:
            break
        attempted += 1
        started = time.time()
        try:
            raw_response = client.reasoning_prefix_completion(request)
            reasoning_suffix, output_text, finish_reason = response_parts(raw_response)
            if not output_text.strip():
                raise RuntimeError("empty final output")
            prefix = str(job["reasoning_prefix"])
            request_sha256 = hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()
            append_jsonl(
                generations_path,
                {
                    "output_id": output_id,
                    "job_id": job["job_id"],
                    "source_output_id": job.get("source_output_id", ""),
                    "actor": job.get("actor", ""),
                    "model": job["model"],
                    "source_model": job.get("source_model", ""),
                    "condition": job.get("condition", ""),
                    "branch_label": job.get("branch_label", ""),
                    "task": job.get("task", ""),
                    "item_id": job.get("item_id", ""),
                    "item_label": job.get("item_label", ""),
                    "sample_index": job.get("continuation_sample_index", 0),
                    "source_sample_index": job.get("source_sample_index", 0),
                    "success": True,
                    "reasoning_prefix": prefix,
                    "reasoning_suffix": reasoning_suffix,
                    "reasoning_text": reasoning_suffix,
                    "reconstructed_reasoning": prefix + reasoning_suffix,
                    "prefix_continued": bool(reasoning_suffix.strip()),
                    "completion_complete": finish_reason == "stop",
                    "output_text": output_text,
                    "finish_reason": finish_reason,
                    "latency_s": round(time.time() - started, 3),
                    "max_tokens": max_tokens,
                    "reasoning_effort": reasoning_effort,
                    "top_p": top_p,
                    "usage": raw_response.get("usage", {}),
                    "raw_response": raw_response,
                    "request": request,
                    "request_sha256": request_sha256,
                    "job": job,
                },
            )
            written += 1
        except Exception as exc:  # noqa: BLE001 - preserve per-job API/runtime failures.
            append_jsonl(
                failures_path,
                {
                    "output_id": output_id,
                    "job_id": job.get("job_id", ""),
                    "source_output_id": job.get("source_output_id", ""),
                    "success": False,
                    "error": str(exc),
                    "latency_s": round(time.time() - started, 3),
                    "request": request,
                    "job": job,
                },
            )
            failed += 1
    return written, failed
