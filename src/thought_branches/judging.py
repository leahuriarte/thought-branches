"""Blind pairwise judging of base-distribution conditions.

Ported from the utility_behavior_gap repo's judge-comparison pipeline
(``utility_behavior_gap.scripts.run_judging``), adapted to thought-branches'
generation records: one row per condition per item, with the prompt/job
metadata embedded inline rather than joined from a separate pair-job
manifest.
"""

from __future__ import annotations

import itertools
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from thought_branches.constants import JUDGE_MODEL_IDS
from thought_branches.io_utils import read_jsonl
from thought_branches.openrouter import MalformedOpenRouterResponse, OpenRouterClient, response_text
from thought_branches.prompts import build_judge_prompt


JUDGE_REASONING = {"effort": "minimal", "exclude": True}

GroupKey = tuple[str, str, str]  # (actor, task, item_id)


def group_key(row: dict[str, Any]) -> GroupKey:
    return (str(row.get("actor", "")), str(row.get("task", "")), str(row.get("item_id", "")))


def base_condition_groups(generations: list[dict[str, Any]]) -> dict[GroupKey, dict[str, dict[str, Any]]]:
    """Group generation rows by (actor, task, item), keyed by condition within each group."""
    groups: dict[GroupKey, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in generations:
        condition = str(row.get("condition", ""))
        if not condition:
            continue
        groups[group_key(row)][condition] = row
    return groups


def pair_id(key: GroupKey, condition_a: str, condition_b: str) -> str:
    actor, task, item = key
    return f"{actor}:{task}:{item}:{condition_a}__vs__{condition_b}"


def parse_winner(text: str) -> str:
    match = re.search(r"answer\s*:\s*(x|y|tie)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    match = re.search(r"winner\s*:\s*(a|b|tie)\b", text, flags=re.IGNORECASE)
    if match:
        value = match.group(1).lower()
        return {"a": "x", "b": "y"}.get(value, value)
    return "unresolved"


def judge_prompt_for(row_x: dict[str, Any], row_y: dict[str, Any]) -> str:
    job = row_x.get("job", {}) if isinstance(row_x.get("job"), dict) else {}
    return build_judge_prompt(
        task=str(row_x.get("task", "")),
        item_label=str(row_x.get("item_label", "")),
        base_prompt=str(job.get("prompt", "")),
        response_x=str(row_x.get("output_text", "")),
        response_y=str(row_y.get("output_text", "")),
    )


PendingVote = tuple[GroupKey, str, str, dict[str, Any], str, dict[str, Any], str, bool]


def build_pending(
    groups: dict[GroupKey, dict[str, dict[str, Any]]],
    *,
    judges: list[str],
    seed: int,
    orders: str,
    done: set[tuple[str, str, bool]],
) -> list[PendingVote]:
    """orders='both': one vote per (pair, judge) in each presentation order (position bias cancels).
    orders='single': one vote per (pair, judge) in a random presentation order."""
    rng = random.Random(seed)
    pending: list[PendingVote] = []
    for key, by_condition in groups.items():
        conditions = sorted(by_condition)
        if len(conditions) < 2:
            continue
        for condition_a, condition_b in itertools.combinations(conditions, 2):
            pid = pair_id(key, condition_a, condition_b)
            row_a, row_b = by_condition[condition_a], by_condition[condition_b]
            for judge_model in judges:
                if orders == "both":
                    flips = [flip for flip in (False, True) if (pid, judge_model, flip) not in done]
                else:
                    already = (pid, judge_model, False) in done or (pid, judge_model, True) in done
                    flips = [] if already else [rng.random() < 0.5]
                for flip in flips:
                    pending.append((key, pid, condition_a, row_a, condition_b, row_b, judge_model, flip))
    return pending


def existing_vote_keys(path: Path) -> set[tuple[str, str, bool]]:
    keys: set[tuple[str, str, bool]] = set()
    if not path.exists():
        return keys
    for row in read_jsonl(path):
        if row.get("success") is False:
            continue
        keys.add((str(row["pair_id"]), str(row["judge_model"]), bool(row.get("flipped"))))
    return keys


def run_judge_request(
    *,
    key: GroupKey,
    pid: str,
    condition_a: str,
    row_a: dict[str, Any],
    condition_b: str,
    row_b: dict[str, Any],
    judge_model: str,
    flip: bool,
    temperature: float,
    max_tokens: int,
    dry_run: bool,
) -> dict[str, Any]:
    row_x, condition_x = (row_b, condition_b) if flip else (row_a, condition_a)
    row_y, condition_y = (row_a, condition_a) if flip else (row_b, condition_b)
    prompt = judge_prompt_for(row_x, row_y)
    started = time.time()
    if dry_run:
        raw_text, success = "Answer: TIE", True
    else:
        client = OpenRouterClient()
        try:
            response = client.chat_completion(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning=JUDGE_REASONING,
            )
            raw_text, success = response_text(response), True
        except MalformedOpenRouterResponse:
            raw_text, success = "", False
    parsed = parse_winner(raw_text)
    winner_condition = {"x": condition_x, "y": condition_y, "tie": "tie"}.get(parsed, "unresolved")
    return {
        "pair_id": pid,
        "actor": key[0],
        "task": key[1],
        "item_id": key[2],
        "condition_a": condition_a,
        "condition_b": condition_b,
        "judge_model": judge_model,
        "flipped": flip,
        "vote_raw": raw_text,
        "parsed_winner": parsed,
        "winner_condition": winner_condition,
        "success": success,
        "latency_s": round(time.time() - started, 3),
        "output_a_id": row_a.get("output_id", ""),
        "output_b_id": row_b.get("output_id", ""),
    }


def summarize_votes(votes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate votes into a per-pair breakdown and per-condition win/loss/tie totals."""
    pairs: dict[str, dict[str, Any]] = {}
    condition_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0})
    for vote in votes:
        if vote.get("success") is False:
            continue
        pid = str(vote["pair_id"])
        condition_a, condition_b = vote["condition_a"], vote["condition_b"]
        entry = pairs.setdefault(
            pid,
            {
                "actor": vote.get("actor", ""),
                "task": vote.get("task", ""),
                "item_id": vote.get("item_id", ""),
                "condition_a": condition_a,
                "condition_b": condition_b,
                "wins_a": 0,
                "wins_b": 0,
                "ties": 0,
                "unresolved": 0,
                "votes": [],
            },
        )
        winner = vote["winner_condition"]
        if winner == condition_a:
            entry["wins_a"] += 1
        elif winner == condition_b:
            entry["wins_b"] += 1
        elif winner == "tie":
            entry["ties"] += 1
        else:
            entry["unresolved"] += 1
        entry["votes"].append(
            {
                "judge_model": vote["judge_model"],
                "flipped": vote.get("flipped", False),
                "parsed_winner": vote.get("parsed_winner", ""),
                "winner_condition": winner,
            }
        )
        if winner in (condition_a, condition_b, "tie"):
            loser = condition_b if winner == condition_a else condition_a if winner == condition_b else None
            if loser is None:
                condition_totals[condition_a]["ties"] += 1
                condition_totals[condition_b]["ties"] += 1
            else:
                condition_totals[winner]["wins"] += 1
                condition_totals[loser]["losses"] += 1
    return {"pairs": pairs, "condition_totals": dict(condition_totals)}


def render_judge_summary_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Judge Comparison Summary", ""]
    lines.append("## Win / loss / tie totals by condition")
    lines.append("")
    lines.append("| condition | wins | losses | ties |")
    lines.append("|---|---:|---:|---:|")
    for condition, totals in sorted(summary["condition_totals"].items()):
        lines.append(f"| {condition} | {totals['wins']} | {totals['losses']} | {totals['ties']} |")
    lines.append("")
    lines.append("## Pairwise detail")
    lines.append("")
    for pid, entry in sorted(summary["pairs"].items()):
        lines.append(
            f"### {entry['condition_a']} vs {entry['condition_b']} "
            f"({entry['actor']}, {entry['task']}, {entry['item_id']})"
        )
        lines.append("")
        lines.append(
            f"- **{entry['condition_a']}** wins: {entry['wins_a']}, "
            f"**{entry['condition_b']}** wins: {entry['wins_b']}, "
            f"ties: {entry['ties']}, unresolved: {entry['unresolved']}"
        )
        for vote in entry["votes"]:
            orientation = "flipped" if vote["flipped"] else "unflipped"
            lines.append(f"  - {vote['judge_model']} ({orientation}): {vote['winner_condition']}")
        lines.append("")
    return "\n".join(lines)


def default_judges() -> list[str]:
    return list(JUDGE_MODEL_IDS)
