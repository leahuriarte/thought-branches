#!/usr/bin/env python3
"""Run blind pairwise judging across the base-distribution conditions for each item.

This is the thought-branches port of the utility_behavior_gap repo's
judge-comparison pipeline: for every (actor, task, item) group with 2+
condition generations (e.g. best_effort / high_utility / low_utility /
role_play), every pair of conditions is judged blindly, in both
presentation orders, by each judge model in
``thought_branches.constants.JUDGE_MODEL_IDS``. Position bias cancels
within each pair because both orders are judged.

Example, on the 4 base runs behind the deepseek_v32_all4_essay_3000_low
viewer:

    tb-run-judging \\
      --generations outputs/api/deepseek_v32_all4_essay_3000_low_generations.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from thought_branches.io_utils import append_jsonl, read_jsonl
from thought_branches.judging import (
    base_condition_groups,
    build_pending,
    default_judges,
    existing_vote_keys,
    render_judge_summary_markdown,
    run_judge_request,
    summarize_votes,
)
from thought_branches.paths import OUTPUT_API


def votes_path_for(generations_path: Path) -> Path:
    stem = generations_path.stem
    if stem.endswith("_generations"):
        stem = stem[: -len("_generations")]
    return OUTPUT_API / f"{stem}_judge_votes.jsonl"


def summary_path_for(votes_path: Path) -> Path:
    return votes_path.with_name(votes_path.stem.replace("_judge_votes", "_judge_summary") + ".md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", type=Path, required=True, help="Generations JSONL to judge.")
    parser.add_argument("--out", type=Path, default=None, help="Judge votes JSONL (default: derived from --generations).")
    parser.add_argument("--summary-out", type=Path, default=None, help="Judge summary Markdown (default: derived from --out).")
    parser.add_argument(
        "--judges",
        nargs="+",
        default=None,
        help="Judge model ids to use (default: thought_branches.constants.JUDGE_MODEL_IDS).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--orders",
        choices=["single", "both"],
        default="both",
        help="single: one vote per pair x judge, random order. both: one vote per pair x judge in each order.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum new judge votes to run.")
    parser.add_argument("--dry-run", action="store_true", help="Write deterministic placeholder votes without API calls.")
    args = parser.parse_args()

    generations = read_jsonl(args.generations)
    groups = base_condition_groups(generations)
    votes_path = args.out or votes_path_for(args.generations)
    summary_path = args.summary_out or summary_path_for(votes_path)
    judges = args.judges or default_judges()

    done = existing_vote_keys(votes_path)
    pending = build_pending(groups, judges=judges, seed=args.seed, orders=args.orders, done=done)
    if args.limit is not None:
        pending = pending[: args.limit]

    written = 0
    for key, pid, condition_a, row_a, condition_b, row_b, judge_model, flip in pending:
        row = run_judge_request(
            key=key,
            pid=pid,
            condition_a=condition_a,
            row_a=row_a,
            condition_b=condition_b,
            row_b=row_b,
            judge_model=judge_model,
            flip=flip,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            dry_run=args.dry_run,
        )
        append_jsonl(votes_path, row)
        written += 1
    print(f"wrote {written} new judge votes to {votes_path}")

    all_votes = read_jsonl(votes_path)
    summary = summarize_votes(all_votes)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(render_judge_summary_markdown(summary), encoding="utf-8")
    print(f"wrote judge summary to {summary_path}")


if __name__ == "__main__":
    main()
