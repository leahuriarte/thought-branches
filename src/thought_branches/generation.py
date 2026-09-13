"""Run generation jobs and write JSONL outputs."""

from __future__ import annotations

import sys
import time
from typing import Any

from thought_branches.branching import readable_reasoning_text
from thought_branches.io_utils import append_jsonl, read_jsonl
from thought_branches.openrouter import OpenRouterClient, response_text


def messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    rows = []
    if system_prompt:
        rows.append({"role": "system", "content": system_prompt})
    rows.append({"role": "user", "content": user_prompt})
    return rows


def request_snapshot(
    job: dict[str, Any],
    *,
    temperature: float | None,
    max_tokens: int,
    reasoning: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "script": "thought_branches.scripts.run_generation",
        "argv": sys.argv,
        "model": job["model"],
        "messages": messages(job.get("system_prompt", ""), job["prompt"]),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": reasoning,
    }


def existing_output_ids(path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("output_id", "")) for row in read_jsonl(path) if row.get("success") is True}


def run_jobs(
    *,
    jobs: list[dict[str, Any]],
    generations_path,
    failures_path,
    dry_run: bool,
    limit: int | None,
    temperature: float | None,
    max_tokens: int,
    reasoning: dict[str, Any] | None,
) -> tuple[int, int]:
    done = existing_output_ids(generations_path)
    client = None if dry_run else OpenRouterClient()
    written = 0
    failed = 0
    attempted = 0
    for job in jobs:
        output_id = job["job_id"]
        if output_id in done:
            continue
        if limit is not None and attempted >= limit:
            break
        attempted += 1
        started = time.time()
        snapshot = request_snapshot(job, temperature=temperature, max_tokens=max_tokens, reasoning=reasoning)
        try:
            if dry_run:
                raw_response: dict[str, Any] = {}
                text = f"[dry run] {job['condition']} output for {job['actor']} on {job['task']}."
                finish_reason = "dry_run"
            else:
                assert client is not None
                raw_response = client.chat_completion(
                    model=job["model"],
                    messages=snapshot["messages"],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning=reasoning,
                )
                text = response_text(raw_response)
                choices = raw_response.get("choices", [{}])
                finish_reason = str(choices[0].get("finish_reason", "")) if choices else ""
            if not text.strip():
                raise RuntimeError("empty output_text")
            reasoning_text = readable_reasoning_text({"raw_response": raw_response})
            append_jsonl(
                generations_path,
                {
                    "output_id": output_id,
                    "job_id": job["job_id"],
                    "run_id": job.get("run_id", ""),
                    "actor": job["actor"],
                    "model": job["model"],
                    "condition": job["condition"],
                    "task": job["task"],
                    "item_id": job["item_id"],
                    "item_label": job["item_label"],
                    "sample_index": job["sample_index"],
                    "success": True,
                    "output_text": text,
                    "reasoning_text": reasoning_text,
                    "latency_s": round(time.time() - started, 3),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "reasoning": reasoning,
                    "finish_reason": finish_reason,
                    "usage": raw_response.get("usage", {}),
                    "raw_response": raw_response,
                    "request": snapshot,
                    "job": job,
                },
            )
            written += 1
        except Exception as exc:  # noqa: BLE001 - failure rows should preserve API/runtime details.
            append_jsonl(
                failures_path,
                {
                    "output_id": output_id,
                    "job_id": job["job_id"],
                    "run_id": job.get("run_id", ""),
                    "actor": job.get("actor", ""),
                    "model": job.get("model", ""),
                    "condition": job.get("condition", ""),
                    "success": False,
                    "error": str(exc),
                    "latency_s": round(time.time() - started, 3),
                    "request": snapshot,
                    "job": job,
                },
            )
            failed += 1
    return written, failed
