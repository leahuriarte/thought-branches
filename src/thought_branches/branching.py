"""Helpers for turning labeled/chunked traces into branch prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


MOTIVATION_PATTERNS: dict[str, str] = {
    "explicit_incentive": (
        r"\bmotivat(?:e|ed|es|ion|ional)\b|\bincentive(?:s)?\b|\breward(?:s)?\b|"
        r"\bprize(?:s)?\b|\bstakes?\b"
    ),
    "success_contingent_outcome": (
        r"\bif (?:this output|my output|the output|it|my essay|the essay|this essay|"
        r"my translation|the translation|this translation) (?:is )?"
        r"(?:selected|judged better|chosen|wins?)\b|"
        r"\b(?:donat(?:e|ed|ion|ions)|fund(?:s|ed|ing)?|sponsor(?:s|ed|ship)?|"
        r"intervention(?:s)?)\b"
    ),
    "self_directed_quality_goal": (
        r"\b(?:I should|I need to|I want to|I'll|I will|my goal|the goal|aim|objective)\b"
        r".{0,120}\b(?:win|selected|best|better|strong|compelling|quality|polish|excellent|help)\b"
    ),
    "exhortation_or_reputation": (
        r"\b(?:absolute best effort|give (?:my|the) best|try harder|extremely important|"
        r"high-stakes|reputation|prestigious|expert reviewers?|world-class)\b"
    ),
}

TRACE_TEXT_KEYS = ("reasoning", "reasoning_content")
MOTIVATED_BRANCH = "motivated_branch"
UNMOTIVATED_BRANCH = "unmotivated_branch"


@dataclass(frozen=True)
class MotivationSpan:
    start: int
    end: int
    category: str
    matched_text: str
    segment: str


def response_message(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_response", {})
    choices = raw.get("choices", []) if isinstance(raw, dict) else []
    if not choices or not isinstance(choices[0], dict):
        return {}
    message = choices[0].get("message", {})
    return message if isinstance(message, dict) else {}


def readable_reasoning_text(row: dict[str, Any]) -> str:
    message = response_message(row)
    chunks: list[str] = []
    for key in TRACE_TEXT_KEYS:
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            chunks.append(value.strip())
    details = message.get("reasoning_details")
    if isinstance(details, list):
        for item in details:
            if not isinstance(item, dict) or item.get("type") == "reasoning.encrypted":
                continue
            for key in ("text", "summary"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    chunks.append(value.strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk in seen:
            continue
        seen.add(chunk)
        deduped.append(chunk)
    return "\n\n".join(deduped)


def _segment_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, flags=re.DOTALL):
        para_start, para_end = match.span()
        paragraph = match.group(0)
        if len(paragraph) <= 360:
            spans.append((para_start, para_end, paragraph))
            continue
        offset = 0
        for sent in re.split(r"(?<=[.!?])\s+", paragraph):
            stripped = sent.strip()
            if not stripped:
                offset += len(sent) + 1
                continue
            sent_start = paragraph.find(stripped, offset)
            if sent_start < 0:
                sent_start = offset
            start = para_start + sent_start
            spans.append((start, start + len(stripped), stripped))
            offset = sent_start + len(stripped)
    return spans


def first_motivation_span(text: str) -> MotivationSpan | None:
    best: tuple[int, int, int, str, re.Match[str], str] | None = None
    for start, end, segment in _segment_spans(text):
        for category, pattern in MOTIVATION_PATTERNS.items():
            match = re.search(pattern, segment, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            absolute_start = start + match.start()
            if best is None or absolute_start < best[0]:
                best = (absolute_start, start, end, category, match, segment)
    if best is None:
        return None
    _, segment_start, end, category, match, segment = best
    return MotivationSpan(
        start=segment_start,
        end=end,
        category=category,
        matched_text=match.group(0),
        segment=segment.strip(),
    )


def motivation_prefixes(text: str, span: MotivationSpan) -> dict[str, str]:
    return {
        "before_motivation": text[: span.start].rstrip(),
        "through_motivation_segment": text[: span.end].rstrip(),
    }


def build_prefix_continuation_prompt(
    *,
    original_prompt: str,
    reasoning_prefix: str,
    branch_label: str,
) -> str:
    if branch_label not in {MOTIVATED_BRANCH, UNMOTIVATED_BRANCH}:
        raise ValueError(f"unknown branch label: {branch_label!r}")
    if branch_label == MOTIVATED_BRANCH:
        branch_instruction = (
            "The prefix includes the next reasoning thought. Treat that prefix as the reasoning context "
            "already written before producing the final answer."
        )
    else:
        branch_instruction = (
            "The prefix stops immediately before a motivation-related thought. Treat only this prefix as "
            "the reasoning context already written before producing the final answer."
        )
    return (
        "You are completing the original task below after a partial reasoning trace was already written.\n\n"
        "Original task:\n"
        f"{original_prompt.strip()}\n\n"
        "Reasoning prefix:\n"
        f"{reasoning_prefix.strip() or '[empty prefix]'}\n\n"
        f"{branch_instruction}\n\n"
        "Write only the final answer for the original task. Do not mention this branch experiment, "
        "the reasoning prefix, or any hidden evaluation/sponsor/donation details unless the original "
        "task explicitly asks for them."
    )
