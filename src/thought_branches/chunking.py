"""Chunk model outputs into readable reasoning/action units."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    start: int
    end: int


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
TAG_RE = re.compile(r"</?[A-Za-z_][A-Za-z0-9_:.-]*[^>]*>")


def _candidate_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for tag in TAG_RE.finditer(text):
        if tag.start() > cursor:
            spans.extend(_paragraph_sentence_spans(text, cursor, tag.start()))
        spans.append((tag.start(), tag.end(), tag.group(0)))
        cursor = tag.end()
    if cursor < len(text):
        spans.extend(_paragraph_sentence_spans(text, cursor, len(text)))
    return spans


def _paragraph_sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    block = text[start:end]
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", block, flags=re.DOTALL):
        para_start = start + match.start()
        paragraph = match.group(0)
        if len(paragraph) <= 360:
            out.append((para_start, para_start + len(paragraph), paragraph))
            continue
        offset = 0
        for sentence in SENTENCE_BOUNDARY_RE.split(paragraph):
            stripped = sentence.strip()
            if not stripped:
                offset += len(sentence)
                continue
            sent_start = paragraph.find(stripped, offset)
            if sent_start < 0:
                sent_start = offset
            absolute = para_start + sent_start
            out.append((absolute, absolute + len(stripped), stripped))
            offset = sent_start + len(stripped)
    return out


def split_text_into_chunks(text: str, *, min_chars: int = 24, max_chars: int = 700) -> list[Chunk]:
    """Split text into paragraph/sentence chunks and merge tiny fragments."""

    candidates = _candidate_spans(text)
    merged: list[tuple[int, int, str]] = []
    for start, end, chunk in candidates:
        stripped = chunk.strip()
        if not stripped:
            continue
        if merged and (len(stripped) < min_chars or len(merged[-1][2]) + len(stripped) <= min_chars):
            prev_start, _, prev_text = merged[-1]
            between = text[merged[-1][1] : start]
            combined = f"{prev_text}{between}{stripped}"
            merged[-1] = (prev_start, end, combined.strip())
            continue
        if len(stripped) > max_chars:
            merged.extend(_split_long_chunk(text, start, end, max_chars=max_chars))
        else:
            merged.append((start, end, stripped))
    return [Chunk(index=i, text=chunk, start=start, end=end) for i, (start, end, chunk) in enumerate(merged)]


def _split_long_chunk(text: str, start: int, end: int, *, max_chars: int) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    cursor = start
    while cursor < end:
        target = min(cursor + max_chars, end)
        split_at = text.rfind("\n", cursor, target)
        if split_at <= cursor:
            split_at = text.rfind(" ", cursor, target)
        if split_at <= cursor:
            split_at = target
        chunk = text[cursor:split_at].strip()
        if chunk:
            chunk_start = text.find(chunk, cursor, split_at)
            out.append((chunk_start, chunk_start + len(chunk), chunk))
        cursor = split_at
        while cursor < end and text[cursor].isspace():
            cursor += 1
    return out
