#!/usr/bin/env python3
"""Build a standalone HTML viewer for generations and reasoning chunks."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from thought_branches.io_utils import read_jsonl
from thought_branches.judging import summarize_votes
from thought_branches.paths import OUTPUTS


DEFAULT_GENERATIONS = OUTPUTS / "api" / "all4_liquid_essay_generations_clean.jsonl"
DEFAULT_REASONING_CHUNKS = OUTPUTS / "chunks" / "all4_liquid_essay_reasoning_chunks_classified_clean.jsonl"
DEFAULT_OUT = OUTPUTS / "viewer" / "all4_liquid_essay_viewer.html"

CONDITION_ORDER = {
    "best_effort": 0,
    "high_utility": 1,
    "low_utility": 2,
    "role_play": 3,
}


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def pre_text(value: Any) -> str:
    return esc(value).replace("\n", "<br>")


def condition_title(condition: str) -> str:
    return condition.replace("_", " ").title()


def label_badges(labels: list[str]) -> str:
    return "".join(f'<span class="badge">{esc(label)}</span>' for label in labels)


def chunk_primary_label(chunk: dict[str, Any]) -> str:
    if chunk.get("primary_function"):
        return str(chunk["primary_function"])
    labels = chunk.get("labels", [])
    if isinstance(labels, list) and labels:
        return str(labels[0])
    return "unlabeled"


def chunk_feature_labels(chunk: dict[str, Any]) -> list[str]:
    if isinstance(chunk.get("feature_flags"), list):
        return [str(flag) for flag in chunk["feature_flags"]]
    labels = chunk.get("labels", [])
    if isinstance(labels, list):
        return [str(label) for label in labels[1:]]
    return []


def metadata_table(row: dict[str, Any]) -> str:
    utility_pair = row.get("job", {}).get("utility_pair", {}) if isinstance(row.get("job"), dict) else {}
    fields = [
        ("output_id", row.get("output_id", "")),
        ("actor", row.get("actor", "")),
        ("model", row.get("model", "")),
        ("task", row.get("task", "")),
        ("item", row.get("item_label", "")),
        ("finish_reason", row.get("finish_reason", "")),
        ("output_chars", len(row.get("output_text") or "")),
        ("reasoning_chars", len(row.get("reasoning_text") or "")),
        ("latency_s", row.get("latency_s", "")),
        ("max_tokens", row.get("max_tokens", "")),
    ]
    if utility_pair:
        fields.extend(
            [
                ("domain", utility_pair.get("domain", "")),
                ("high_description", utility_pair.get("high_description", "")),
                ("low_description", utility_pair.get("low_description", "")),
                ("delta_u", utility_pair.get("delta_u", "")),
            ]
        )
    rows = "\n".join(
        f"<tr><th>{esc(key)}</th><td>{esc(value)}</td></tr>"
        for key, value in fields
        if str(value) != ""
    )
    return f'<table class="meta">{rows}</table>'


def chunk_summary(chunks: list[dict[str, Any]]) -> str:
    counts = Counter(chunk_primary_label(chunk) for chunk in chunks)
    if not counts:
        return '<p class="muted">No classified reasoning chunks.</p>'
    badges = "".join(
        f'<span class="summary-badge">{esc(label)} <strong>{count}</strong></span>'
        for label, count in counts.most_common()
    )
    return f'<div class="label-summary">{badges}</div>'


def render_chunks(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return '<p class="muted">No chunks for this output.</p>'
    parts = []
    for chunk in sorted(chunks, key=lambda row: int(row.get("chunk_index", 0))):
        primary = chunk_primary_label(chunk)
        features = chunk_feature_labels(chunk)
        cognitive_label = chunk.get("cognitive_label", "")
        rationale = chunk.get("rationale", chunk.get("label_reason", ""))
        all_badges = [primary] + features
        subline = []
        if cognitive_label:
            subline.append(f'<strong>{esc(cognitive_label)}</strong>')
        if rationale:
            subline.append(esc(rationale))
        parts.append(
            f"""
            <article class="chunk" data-labels="{esc(' '.join(all_badges))}">
              <div class="chunk-head">
                <span class="chunk-index">#{esc(chunk.get('chunk_index', ''))}</span>
                <span>{label_badges(all_badges)}</span>
              </div>
              <div class="chunk-subline">{' · '.join(subline)}</div>
              <p>{pre_text(chunk.get('chunk_text', ''))}</p>
            </article>
            """
        )
    return "\n".join(parts)


def render_prompt(row: dict[str, Any]) -> str:
    job = row.get("job", {}) if isinstance(row.get("job"), dict) else {}
    system_prompt = job.get("system_prompt", "")
    prompt = job.get("prompt", "")
    if not system_prompt and not prompt:
        return '<p class="muted">No prompt stored.</p>'
    system_block = (
        f"""
        <h4>System</h4>
        <div class="prompt-box">{pre_text(system_prompt)}</div>
        """
        if system_prompt
        else ""
    )
    return (
        system_block
        + f"""
        <h4>User</h4>
        <div class="prompt-box">{pre_text(prompt)}</div>
        """
    )


def render_judge_summary(votes: list[dict[str, Any]]) -> str:
    if not votes:
        return ""
    summary = summarize_votes(votes)
    totals = summary["condition_totals"]
    if not totals:
        return ""
    totals_rows = "\n".join(
        f"<tr><td>{esc(condition_title(condition))}</td><td>{esc(t['wins'])}</td>"
        f"<td>{esc(t['losses'])}</td><td>{esc(t['ties'])}</td></tr>"
        for condition, t in sorted(totals.items(), key=lambda kv: CONDITION_ORDER.get(kv[0], 99))
    )
    pair_rows = "\n".join(
        f"<tr><td>{esc(condition_title(entry['condition_a']))} vs {esc(condition_title(entry['condition_b']))}</td>"
        f"<td>{esc(entry['wins_a'])}</td><td>{esc(entry['wins_b'])}</td>"
        f"<td>{esc(entry['ties'])}</td><td>{esc(entry['unresolved'])}</td></tr>"
        for _, entry in sorted(summary["pairs"].items())
    )
    return f"""
    <section class="panel judge-summary">
      <h3>Judge Comparison</h3>
      <p class="muted">Blind pairwise judging across conditions, both presentation orders ({len(votes)} judge votes).</p>
      <table class="judge-table">
        <thead><tr><th>Condition</th><th>Wins</th><th>Losses</th><th>Ties</th></tr></thead>
        <tbody>{totals_rows}</tbody>
      </table>
      <h4>Pairwise results</h4>
      <table class="judge-table">
        <thead><tr><th>Pair</th><th>A wins</th><th>B wins</th><th>Ties</th><th>Unresolved</th></tr></thead>
        <tbody>{pair_rows}</tbody>
      </table>
    </section>
    """


def render_branch(row: dict[str, Any], chunks: list[dict[str, Any]]) -> str:
    condition = str(row.get("condition", ""))
    return f"""
    <section class="branch" id="{esc(condition)}">
      <header class="branch-title">
        <div>
          <p class="eyebrow">Condition</p>
          <h2>{esc(condition_title(condition))}</h2>
        </div>
        <div class="status">{esc(row.get('finish_reason', ''))}</div>
      </header>

      <div class="grid">
        <div class="panel">
          <h3>Run Info</h3>
          {metadata_table(row)}
          <h3>Reasoning Labels</h3>
          {chunk_summary(chunks)}
        </div>
        <div class="panel">
          <h3>Prompt</h3>
          <details>
            <summary>Show prompt</summary>
            {render_prompt(row)}
          </details>
        </div>
      </div>

      <div class="panel">
        <h3>Final Output</h3>
        <div class="output-text">{pre_text(row.get('output_text', ''))}</div>
      </div>

      <div class="panel">
        <h3>Reasoning Trace</h3>
        <details open>
          <summary>Raw extracted reasoning text</summary>
          <div class="reasoning-text">{pre_text(row.get('reasoning_text', ''))}</div>
        </details>
      </div>

      <div class="panel">
        <h3>Classified Reasoning Chunks</h3>
        {render_chunks(chunks)}
      </div>
    </section>
    """


def render_html(
    generations: list[dict[str, Any]],
    chunks_by_output: dict[str, list[dict[str, Any]]],
    judge_votes: list[dict[str, Any]] | None = None,
) -> str:
    generations = sorted(
        generations,
        key=lambda row: (CONDITION_ORDER.get(str(row.get("condition", "")), 99), str(row.get("condition", ""))),
    )
    total_chunks = sum(len(chunks_by_output.get(str(row.get("output_id", "")), [])) for row in generations)
    nav = "\n".join(
        f'<a href="#{esc(row.get("condition", ""))}">{esc(condition_title(str(row.get("condition", ""))))}</a>'
        for row in generations
    )
    branches = "\n".join(
        render_branch(row, chunks_by_output.get(str(row.get("output_id", "")), [])) for row in generations
    )
    judge_summary = render_judge_summary(judge_votes or [])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Thought Branches Viewer</title>
  <style>
    :root {{
      --bg: #f7f7f4;
      --text: #1d2527;
      --muted: #667276;
      --line: #d9ded9;
      --panel: #ffffff;
      --accent: #0f766e;
      --accent-soft: #dff3ef;
      --code: #f1f4f2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      background: var(--bg);
      font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header.top {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: rgba(247, 247, 244, 0.95);
      border-bottom: 1px solid var(--line);
      padding: 16px 24px;
      backdrop-filter: blur(8px);
    }}
    h1, h2, h3, h4, p {{ margin-top: 0; }}
    h1 {{ margin-bottom: 4px; font-size: 24px; }}
    h2 {{ margin: 0; font-size: 22px; }}
    h3 {{ margin-bottom: 12px; font-size: 16px; }}
    h4 {{ margin: 14px 0 6px; font-size: 13px; color: var(--muted); }}
    nav {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }}
    nav a {{
      color: var(--accent);
      background: var(--accent-soft);
      border: 1px solid #b9e0da;
      border-radius: 6px;
      padding: 5px 9px;
      text-decoration: none;
      font-weight: 600;
    }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    .branch {{ margin-bottom: 32px; scroll-margin-top: 96px; }}
    .branch-title {{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      border-bottom: 2px solid var(--text);
      padding-bottom: 10px;
      margin-bottom: 16px;
    }}
    .eyebrow {{ margin: 0 0 3px; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0; }}
    .status {{ color: var(--accent); font-weight: 700; }}
    .grid {{ display: grid; grid-template-columns: minmax(280px, 0.9fr) minmax(320px, 1.1fr); gap: 16px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }}
    .meta {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .meta th {{ width: 150px; text-align: left; color: var(--muted); vertical-align: top; padding: 5px 8px 5px 0; }}
    .meta td {{ padding: 5px 0; border-bottom: 1px solid #edf0ed; word-break: break-word; }}
    .output-text, .reasoning-text, .prompt-box {{
      white-space: normal;
      background: var(--code);
      border: 1px solid #e0e5e1;
      border-radius: 6px;
      padding: 14px;
    }}
    .output-text {{ font-size: 15px; }}
    .reasoning-text, .prompt-box {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
    details summary {{ cursor: pointer; color: var(--accent); font-weight: 700; margin-bottom: 10px; }}
    .chunk {{
      border-top: 1px solid var(--line);
      padding: 12px 0;
    }}
    .chunk:first-of-type {{ border-top: 0; }}
    .chunk p {{ margin: 8px 0 0; }}
    .chunk-head {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
    .chunk-index {{ color: var(--muted); font-weight: 700; font-variant-numeric: tabular-nums; }}
    .chunk-subline {{ margin-top: 6px; color: var(--muted); font-size: 13px; }}
    .badge, .summary-badge {{
      display: inline-block;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #07534d;
      border: 1px solid #b9e0da;
      padding: 2px 8px;
      margin: 2px 4px 2px 0;
      font-size: 12px;
      font-weight: 650;
    }}
    .summary-badge strong {{ margin-left: 4px; }}
    .muted {{ color: var(--muted); }}
    .judge-table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 12px; }}
    .judge-table th, .judge-table td {{ text-align: left; padding: 5px 8px; border-bottom: 1px solid #edf0ed; }}
    .judge-table th {{ color: var(--muted); font-weight: 650; }}
    @media (max-width: 850px) {{
      header.top {{ padding: 14px 16px; }}
      main {{ padding: 16px; }}
      .grid {{ grid-template-columns: 1fr; }}
      .branch-title {{ align-items: flex-start; flex-direction: column; gap: 8px; }}
    }}
  </style>
</head>
<body>
  <header class="top">
    <h1>Thought Branches Viewer</h1>
    <p class="muted">{len(generations)} generations, {total_chunks} classified reasoning chunks</p>
    <nav>{nav}</nav>
  </header>
  <main>
    {judge_summary}
    {branches}
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument("--reasoning-chunks", type=Path, default=DEFAULT_REASONING_CHUNKS)
    parser.add_argument(
        "--judge-votes",
        type=Path,
        default=None,
        help="Judge votes JSONL from tb-run-judging (default: derived from --generations, if it exists).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    generations = read_jsonl(args.generations)
    chunks_by_output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if args.reasoning_chunks.exists():
        for chunk in read_jsonl(args.reasoning_chunks):
            chunks_by_output[str(chunk.get("output_id", ""))].append(chunk)

    judge_votes_path = args.judge_votes
    if judge_votes_path is None:
        stem = args.generations.stem
        if stem.endswith("_generations"):
            stem = stem[: -len("_generations")]
        candidate = args.generations.parent / f"{stem}_judge_votes.jsonl"
        if candidate.exists():
            judge_votes_path = candidate
    judge_votes = read_jsonl(judge_votes_path) if judge_votes_path and judge_votes_path.exists() else []

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(generations, chunks_by_output, judge_votes), encoding="utf-8")
    print(f"wrote viewer to {args.out}")


if __name__ == "__main__":
    main()
