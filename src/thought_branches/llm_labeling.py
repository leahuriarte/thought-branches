"""LLM-based reasoning chunk classification inspired by the original DAG labeler."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from thought_branches.openrouter import OpenRouterClient, response_text


PRIMARY_FUNCTIONS = [
    "task_parse",
    "condition_cue_processing",
    "constraint_tracking",
    "goal_setting",
    "structure_planning",
    "argument_planning",
    "evidence_selection",
    "counterargument_planning",
    "tone_style_planning",
    "drafting_or_rehearsing",
    "revision_quality_control",
    "finalization",
    "other",
]

FEATURE_FLAGS = [
    "mentions_incentive",
    "suppresses_incentive",
    "mentions_best_effort",
    "uses_role_identity",
    "mentions_utility_target",
    "quality_goal",
    "drafts_final_text",
    "handles_counterargument",
    "checks_constraints",
]


def _json_from_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def build_label_prompt(
    *,
    condition: str,
    task_prompt: str,
    chunks: list[dict[str, Any]],
) -> str:
    chunk_text = "\n\n".join(
        f"Chunk {chunk['chunk_index']}:\n{chunk['chunk_text']}" for chunk in chunks
    )
    return f"""You are labeling readable reasoning traces from language-model generations.

The trace is split into numbered chunks. Label each chunk by what it is doing in the reasoning process, not by whether you agree with it.

Use this schema for each chunk:
- primary_function: exactly one of {PRIMARY_FUNCTIONS}
- cognitive_label: a concise 2-5 word verb-noun phrase, e.g. "parse essay task", "plan herd-immunity argument", "suppress prize mention", "draft thesis".
- feature_flags: zero or more of {FEATURE_FLAGS}
- rationale: one short sentence explaining the label.

Guidelines inspired by the original Thought Branches labeler:
- Capture the verb: what is the model doing?
- Distinguish planning from action/drafting.
- Prefer a specific cognitive_label over a generic one.
- Assign only one primary_function even if multiple feature_flags apply.
- condition_cue_processing is for noticing or managing prompt-condition cues such as best effort, role identity, prize/donation/sponsor, or utility target.
- drafting_or_rehearsing is for text that is already proposed final-answer language or sentence-level rehearsal.
- revision_quality_control is for checking strength, clarity, word count, flow, or constraints after planning/drafting.

Condition: {condition}

Original task prompt:
---
{task_prompt}
---

Chunks:
---
{chunk_text}
---

Return only JSON in this exact shape:
{{
  "chunks": {{
    "0": {{
      "primary_function": "task_parse",
      "cognitive_label": "parse essay task",
      "feature_flags": [],
      "rationale": "The chunk restates the assignment and constraints."
    }}
  }}
}}
"""


def validate_label(raw: dict[str, Any], chunk: dict[str, Any]) -> dict[str, Any]:
    primary = str(raw.get("primary_function", "other"))
    if primary not in PRIMARY_FUNCTIONS:
        primary = "other"
    flags = raw.get("feature_flags", [])
    if not isinstance(flags, list):
        flags = []
    flags = [str(flag) for flag in flags if str(flag) in FEATURE_FLAGS]
    cognitive_label = str(raw.get("cognitive_label", "")).strip() or primary.replace("_", " ")
    rationale = str(raw.get("rationale", "")).strip()
    return {
        "primary_function": primary,
        "cognitive_label": cognitive_label[:120],
        "feature_flags": flags,
        "rationale": rationale[:500],
    }


def label_chunk_groups(
    *,
    generations: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    model: str,
    max_tokens: int,
) -> list[dict[str, Any]]:
    generation_by_output = {str(row.get("output_id", "")): row for row in generations}
    chunks_by_output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_output[str(chunk.get("output_id", ""))].append(chunk)

    client = OpenRouterClient()
    labeled_rows: list[dict[str, Any]] = []
    for output_id, group in chunks_by_output.items():
        group = sorted(group, key=lambda row: int(row.get("chunk_index", 0)))
        generation = generation_by_output.get(output_id, {})
        job = generation.get("job", {}) if isinstance(generation.get("job"), dict) else {}
        prompt = build_label_prompt(
            condition=str(generation.get("condition") or group[0].get("condition", "")),
            task_prompt=str(job.get("prompt", "")),
            chunks=group,
        )
        response = client.chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
            reasoning=None,
        )
        payload = _json_from_text(response_text(response))
        labels = payload.get("chunks", {})
        if not isinstance(labels, dict):
            labels = {}
        for chunk in group:
            raw_label = labels.get(str(chunk.get("chunk_index")), {})
            if not isinstance(raw_label, dict):
                raw_label = {}
            labeled = dict(chunk)
            structured = validate_label(raw_label, chunk)
            labeled.update(structured)
            labeled["labeler_model"] = model
            labeled["labeling_method"] = "llm_dag_v1"
            labeled_rows.append(labeled)
    return labeled_rows
