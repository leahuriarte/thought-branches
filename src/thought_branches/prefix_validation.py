"""Build and summarize small native reasoning-prefix fidelity checks."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any

from thought_branches.deepseek import (
    DEEPSEEK_BETA_BASE_URL,
    UTILITY_COVERED_DEEPSEEK_MODEL,
    build_reasoning_prefix_request,
    response_parts,
)


DEFAULT_VALIDATION_CASES: list[dict[str, str]] = [
    {
        "case_id": "paragraph_boundary",
        "cut_kind": "paragraph_boundary",
        "system_prompt": "",
        "user_prompt": "Explain whether a city should replace some downtown parking with a public park.",
        "reasoning_prefix": (
            "We need a concise, balanced answer that still reaches a recommendation. "
            "The main tradeoffs are public space, access for drivers, local business effects, and cost.\n\n"
        ),
    },
    {
        "case_id": "sentence_boundary",
        "cut_kind": "sentence_boundary",
        "system_prompt": "",
        "user_prompt": "Compare two ways a school could reduce food waste.",
        "reasoning_prefix": (
            "I should compare the options on likely waste reduction, implementation burden, and student acceptance. "
        ),
    },
    {
        "case_id": "mid_sentence",
        "cut_kind": "mid_sentence",
        "system_prompt": "",
        "user_prompt": "Recommend whether a library should extend weekend hours.",
        "reasoning_prefix": (
            "The answer should weigh patron demand against staffing cost. A useful recommendation would be to extend "
        ),
    },
]


def messages_for_case(case: dict[str, str]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if case.get("system_prompt", "").strip():
        messages.append({"role": "system", "content": case["system_prompt"]})
    messages.append({"role": "user", "content": case["user_prompt"]})
    return messages


def validation_rows(
    *,
    cases: list[dict[str, str]],
    model: str,
    repeats: int,
    max_tokens: int,
    reasoning_effort: str,
    top_p: float,
) -> list[dict[str, Any]]:
    if repeats < 2:
        raise ValueError("repeats must be at least 2 so diversity can be checked")
    rows: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for case in cases:
        case_id = str(case.get("case_id", "")).strip()
        if not case_id or case_id in seen_case_ids:
            raise ValueError("validation case_id values must be non-empty and unique")
        seen_case_ids.add(case_id)
        prefix = str(case.get("reasoning_prefix", ""))
        request = build_reasoning_prefix_request(
            model=model,
            messages=messages_for_case(case),
            reasoning_prefix=prefix,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            top_p=top_p,
        )
        request_digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()
        for sample_index in range(repeats):
            rows.append(
                {
                    "validation_id": f"{case_id}:s{sample_index}",
                    "case_id": case_id,
                    "cut_kind": str(case.get("cut_kind", "unspecified")),
                    "sample_index": sample_index,
                    "reasoning_prefix": prefix,
                    "endpoint": f"{DEEPSEEK_BETA_BASE_URL}/chat/completions",
                    "request_digest": request_digest,
                    "request": request,
                    "status": "prepared",
                }
            )
    return rows


def attach_response(row: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    reasoning_suffix, output_text, finish_reason = response_parts(response)
    result = dict(row)
    result.update(
        {
            "status": "success",
            "reasoning_suffix": reasoning_suffix,
            "reconstructed_reasoning": str(row["reasoning_prefix"]) + reasoning_suffix,
            "output_text": output_text,
            "finish_reason": finish_reason,
            "usage": response.get("usage", {}),
            "raw_response": response,
        }
    )
    return result


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def summarize_validation(rows: list[dict[str, Any]], *, model: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("case_id", ""))].append(row)

    cases: dict[str, Any] = {}
    for case_id, group in sorted(groups.items()):
        successful = [row for row in group if row.get("status") == "success"]
        suffixes = [_normalized(str(row.get("reasoning_suffix", ""))) for row in successful]
        nonempty = [suffix for suffix in suffixes if suffix]
        request_digests = {str(row.get("request_digest", "")) for row in group}
        finish_reasons = Counter(str(row.get("finish_reason", "")) for row in successful)
        completed_outputs = [
            row
            for row in successful
            if row.get("finish_reason") == "stop" and str(row.get("output_text", "")).strip()
        ]
        cases[case_id] = {
            "cut_kind": str(group[0].get("cut_kind", "")),
            "samples": len(group),
            "successful_samples": len(successful),
            "nonempty_reasoning_continuations": len(nonempty),
            "unique_reasoning_continuations": len(set(nonempty)),
            "diversity_ratio": round(len(set(nonempty)) / len(nonempty), 3) if nonempty else 0.0,
            "identical_request_payloads": len(request_digests) == 1,
            "finish_reasons": dict(sorted(finish_reasons.items())),
            "completed_final_outputs": len(completed_outputs),
        }

    complete = bool(cases) and all(
        entry["successful_samples"] == entry["samples"]
        and entry["nonempty_reasoning_continuations"] == entry["samples"]
        for entry in cases.values()
    )
    diverse = bool(cases) and all(entry["unique_reasoning_continuations"] > 1 for entry in cases.values())
    mid_sentence_present = any(entry["cut_kind"] == "mid_sentence" for entry in cases.values())
    final_outputs_complete = bool(cases) and all(
        entry["completed_final_outputs"] == entry["samples"] for entry in cases.values()
    )
    prefix_transport_pass = complete and mid_sentence_present
    return {
        "model": model,
        "utility_covered_experimental_model": UTILITY_COVERED_DEEPSEEK_MODEL,
        "same_experimental_model": model == UTILITY_COVERED_DEEPSEEK_MODEL,
        "eligible_for_utility_pilot": False,
        "prefix_transport_checks_pass": prefix_transport_pass,
        "diversity_checks_pass": diverse,
        "end_to_end_completion_checks_pass": prefix_transport_pass and diverse and final_outputs_complete,
        "automated_transport_checks_pass": prefix_transport_pass and diverse and final_outputs_complete,
        "manual_prefix_fidelity_review_required": True,
        "cases": cases,
        "interpretation": (
            "Prefix transport, diversity, and final-output completion are reported separately. Even when all "
            "automated checks pass, manually inspect every prefix/suffix seam. This direct-API model is not the "
            "DeepSeek V3.2 model with saved utility pairs."
        ),
    }
