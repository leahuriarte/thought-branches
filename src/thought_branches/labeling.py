"""Heuristic labels for reasoning-trace chunks."""

from __future__ import annotations

import re


LABEL_PATTERNS: dict[str, str] = {
    "task_parse": (
        r"\b(user wants|request|topic|format|requirements?|asked to|need to include|"
        r"write (?:a|an)|essay|translation|postmortem|abstract)\b"
    ),
    "constraint_tracking": (
        r"\b(250-400|word count|words|length|within the range|output only|do not mention|"
        r"avoid mentioning|focus solely|constraint|requirements?)\b"
    ),
    "incentive_awareness": (
        r"\b(prize|donat(?:e|ed|ion|ions)|sponsor(?:ing)?|competition|judged|selected|wins?)\b"
    ),
    "best_effort_orientation": (
        r"\b(best effort|absolute best|extremely important|serious|persuasive|compelling tone|"
        r"world-class|reputation|prestigious|carefully)\b"
    ),
    "role_identity": r"\b(as a|I am|I'll act as|my role|expert|writer|translator|engineer)\b",
    "outline_planning": (
        r"\b(outline|structure|introduction|body paragraph|conclusion|hook|thesis statement|"
        r"let me structure|let me plan|plan the structure)\b"
    ),
    "argument_selection": (
        r"\b(arguments? to include|argument \d|supporting arguments?|herd immunity|"
        r"counterargument|rebuttal|public health|individual choice|legal precedent|equity)\b"
    ),
    "example_selection": (
        r"\b(concrete example|for example|instance|measles outbreak|pertussis|polio|"
        r"california|washington state|recent outbreak)\b"
    ),
    "evidence_use": (
        r"\b(demonstrates?|consequences?|effectiveness|evidence|medical consensus|"
        r"scientific consensus|data|historical|precedent)\b"
    ),
    "tone_planning": r"\b(tone|balanced|firm|acknowledg(?:e|ing)|emphasiz(?:e|ing)|persuasive)\b",
    "counterargument_handling": (
        r"\b(counterarguments?|parental rights|individual concerns|individual liberty|"
        r"freedom concerns|refute|rebut|public safety logic)\b"
    ),
    "drafting_final_answer": (
        r"\b(drafting|draft|revised draft|start strong|let me write|I'll write|"
        r"therefore|in an era|first,|second,|ultimately)\b"
    ),
    "draft_content": (
        r"\b(vaccin(?:e|es|ation|ations)|immuni(?:zation|sation|ty|zed|sed)|"
        r"public school|classroom|child(?:ren)?|parents?|disease|measles|pertussis|"
        r"whooping cough|herd immunity|community health|individual freedom|public health|"
        r"medical decisions?|susceptible|infection|outbreaks?|social contract)\b"
        r"|\b(resistance|strict enforcement|dangerous pockets|devastating consequences|"
        r"health of the many|responsibility of citizenship)\b"
    ),
    "revision_polish": (
        r"\b(refin(?:e|ing)|polish|flow|transition|language|compelling|strong vocabulary|"
        r"integrated smoothly|revised)\b"
    ),
    "length_check": r"\b(word count check|total:|~\d+\s*words|fits.*250-400|within the word limit)\b",
    "final_review": (
        r"\b(final review|final polish|checking|against constraints|thesis\\?|"
        r"supporting arguments\\?|I will output this version)\b"
    ),
}


def label_chunk(text: str) -> list[str]:
    labels = [
        label
        for label, pattern in LABEL_PATTERNS.items()
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    ]
    return labels or ["reasoning_step_other"]


def label_reason(text: str, labels: list[str]) -> str:
    if labels == ["reasoning_step_other"]:
        return "Classified as a reasoning step that did not match a more specific built-in heuristic."
    return "Matched reasoning heuristic(s): " + ", ".join(labels)
