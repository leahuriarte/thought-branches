"""Prepare native reasoning-prefix branch jobs from LLM-classified chunks."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from thought_branches.branching import MOTIVATED_BRANCH, UNMOTIVATED_BRANCH, readable_reasoning_text


LLM_MOTIVATION_LABELING_METHOD = "llm_dag_v3_motivation_commitment"
MOTIVATION_COMMITMENT_FLAG = "motivation_commitment"
MOTIVATION_FEATURE_FLAGS = {
    MOTIVATION_COMMITMENT_FLAG,
    "motivation_values",
    "motivation_effort",
    "motivation_role_reputation",
}
NATIVE_REASONING_PREFIX_MODE = "native_reasoning_prefix"


def first_llm_motivation_chunk(chunks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first chunk the LLM labeled as motivational."""
    invalid_methods = {
        str(chunk.get("labeling_method", ""))
        for chunk in chunks
        if chunk.get("labeling_method") != LLM_MOTIVATION_LABELING_METHOD
    }
    if invalid_methods:
        raise ValueError(
            "branch preparation requires every chunk to use "
            f"{LLM_MOTIVATION_LABELING_METHOD!r}; found {sorted(invalid_methods)!r}"
        )
    invalid_sources = {
        str(chunk.get("chunk_source", ""))
        for chunk in chunks
        if chunk.get("chunk_source") not in (None, "", "reasoning")
    }
    if invalid_sources:
        raise ValueError(f"branch preparation requires reasoning chunks; found {sorted(invalid_sources)!r}")
    for chunk in sorted(chunks, key=lambda row: int(row.get("chunk_index", 0))):
        flags = chunk.get("feature_flags", [])
        if isinstance(flags, list) and MOTIVATION_COMMITMENT_FLAG in map(str, flags):
            return chunk
    return None


def prefixes_from_labeled_chunk(trace: str, chunk: dict[str, Any]) -> dict[str, str]:
    start = int(chunk.get("char_start", -1))
    end = int(chunk.get("char_end", -1))
    if start < 0 or end <= start or end > len(trace):
        raise ValueError(f"invalid labeled chunk offsets: start={start}, end={end}, trace_chars={len(trace)}")
    chunk_text = str(chunk.get("chunk_text", ""))
    if trace[start:end].strip() != chunk_text.strip():
        raise ValueError("labeled chunk text does not match its source reasoning trace offsets")
    return {
        UNMOTIVATED_BRANCH: trace[:start].rstrip(),
        MOTIVATED_BRANCH: trace[:end].rstrip(),
    }


def build_native_branch_jobs(
    *,
    generations: list[dict[str, Any]],
    labeled_chunks: list[dict[str, Any]],
    target_model: str,
    samples_per_branch: int,
    max_seeds: int | None,
    min_pre_prefix_chars: int,
) -> list[dict[str, Any]]:
    if samples_per_branch < 1:
        raise ValueError("samples_per_branch must be positive")
    if max_seeds is not None and max_seeds < 1:
        raise ValueError("max_seeds must be positive when provided")
    if min_pre_prefix_chars < 1:
        raise ValueError("min_pre_prefix_chars must be positive")

    chunks_by_output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in labeled_chunks:
        chunks_by_output[str(chunk.get("output_id", ""))].append(chunk)

    jobs: list[dict[str, Any]] = []
    selected_seeds = 0
    for row in generations:
        output_id = str(row.get("output_id", ""))
        chunks = chunks_by_output.get(output_id, [])
        if not chunks:
            continue
        selected_chunk = first_llm_motivation_chunk(chunks)
        if selected_chunk is None:
            continue
        trace = readable_reasoning_text(row) or str(row.get("reasoning_text", ""))
        if not trace:
            continue
        prefixes = prefixes_from_labeled_chunk(trace, selected_chunk)
        if len(prefixes[UNMOTIVATED_BRANCH]) < min_pre_prefix_chars:
            continue

        source_job = row.get("job", {})
        if not isinstance(source_job, dict) or not str(source_job.get("prompt", "")).strip():
            continue
        original_prompt = str(source_job["prompt"])
        source_sample_index = int(row.get("sample_index", 0))
        selected_flags = sorted(
            MOTIVATION_FEATURE_FLAGS.intersection(map(str, selected_chunk.get("feature_flags", [])))
        )
        for branch_label in (UNMOTIVATED_BRANCH, MOTIVATED_BRANCH):
            reasoning_prefix = prefixes[branch_label]
            prefix_digest = hashlib.sha256(reasoning_prefix.encode("utf-8")).hexdigest()[:12]
            for continuation_sample_index in range(samples_per_branch):
                identity = (
                    f"{output_id}\n{branch_label}\n{target_model}\n{prefix_digest}\n"
                    f"{continuation_sample_index}"
                )
                digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
                jobs.append(
                    {
                        "job_id": f"native-branch:{output_id}:{branch_label}:s{continuation_sample_index}:v{digest}",
                        "continuation_mode": NATIVE_REASONING_PREFIX_MODE,
                        "source_output_id": output_id,
                        "source_condition": row.get("condition", ""),
                        "source_model": row.get("model", ""),
                        "source_sample_index": source_sample_index,
                        "continuation_sample_index": continuation_sample_index,
                        "branch_label": branch_label,
                        "actor": row.get("actor", ""),
                        "model": target_model,
                        "condition": branch_label,
                        "system_prompt": source_job.get("system_prompt", ""),
                        "original_prompt": original_prompt,
                        "reasoning_prefix": reasoning_prefix,
                        "reasoning_prefix_sha256": hashlib.sha256(
                            reasoning_prefix.encode("utf-8")
                        ).hexdigest(),
                        "task": row.get("task", source_job.get("task", "")),
                        "task_label": source_job.get("task_label", ""),
                        "item_id": row.get("item_id", source_job.get("item_id", "")),
                        "item_index": source_job.get("item_index", ""),
                        "item_label": row.get("item_label", source_job.get("item_label", "")),
                        "base_prompt": source_job.get("base_prompt", ""),
                        "axis": source_job.get("axis", ""),
                        "axis_definition": source_job.get("axis_definition", ""),
                        "selected_chunk_index": int(selected_chunk.get("chunk_index", 0)),
                        "selected_chunk_text": str(selected_chunk.get("chunk_text", "")),
                        "selected_chunk_feature_flags": selected_flags,
                        "labeler_model": selected_chunk.get("labeler_model", ""),
                        "labeling_method": selected_chunk.get("labeling_method", ""),
                        "source_job": source_job,
                    }
                )
        selected_seeds += 1
        if max_seeds is not None and selected_seeds >= max_seeds:
            break
    return jobs
