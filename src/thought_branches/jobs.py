"""Build base-distribution generation jobs."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from thought_branches.constants import ACTORS, ACTOR_MODEL_ID, TASK_LABEL
from thought_branches.io_utils import read_csv_rows, write_jsonl
from thought_branches.paths import GENERATION_JOBS, SELECTED_UTILITY_PAIRS, TASK_ITEMS
from thought_branches.prompts import prompt_for_condition


BASE_CONDITIONS = {"best_effort", "high_utility", "low_utility", "role_play"}


def parse_set(value: str) -> set[str] | None:
    stripped = value.strip()
    if not stripped or stripped == "all":
        return None
    return {part.strip() for part in stripped.split(",") if part.strip()}


def stable_digest(*parts: str) -> str:
    return hashlib.sha256("\n---\n".join(parts).encode("utf-8")).hexdigest()[:10]


def run_id_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%SZ")


def task_rows(*, tasks: set[str] | None, items_per_task: int | None) -> list[dict[str, str]]:
    rows = read_csv_rows(TASK_ITEMS)
    if tasks is not None:
        rows = [row for row in rows if row["task"] in tasks]
    out: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for row in rows:
        task = row["task"]
        if items_per_task is not None and counts.get(task, 0) >= items_per_task:
            continue
        copy = dict(row)
        copy["item_index"] = str(counts.get(task, 0))
        out.append(copy)
        counts[task] = counts.get(task, 0) + 1
    return out


FALLBACK_UTILITY_ACTOR = "__fallback__"


def limit_pairs_per_actor_domain(
    rows: list[dict[str, str]],
    *,
    pairs_per_actor_domain: int | None,
) -> list[dict[str, str]]:
    kept: list[dict[str, str]] = []
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["actor"], row["domain"])
        if pairs_per_actor_domain is not None and counts.get(key, 0) >= pairs_per_actor_domain:
            continue
        kept.append(row)
        counts[key] = counts.get(key, 0) + 1
    return kept


def utility_pairs_by_actor(*, actors: set[str] | None, pairs_per_actor_domain: int | None) -> dict[str, list[dict[str, str]]]:
    all_rows = [row for row in read_csv_rows(SELECTED_UTILITY_PAIRS) if row["pair_set"] == "default"]
    rows = [row for row in all_rows if actors is None or row["actor"] in actors]
    rows = limit_pairs_per_actor_domain(rows, pairs_per_actor_domain=pairs_per_actor_domain)
    fallback_rows = limit_pairs_per_actor_domain(all_rows, pairs_per_actor_domain=pairs_per_actor_domain)
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(row["actor"], []).append(row)
    out[FALLBACK_UTILITY_ACTOR] = fallback_rows
    return out


def utility_pair_for_slot(pairs: list[dict[str, str]], slot_index: int) -> dict[str, str]:
    if not pairs:
        return {}
    return pairs[slot_index % len(pairs)]


def build_generation_jobs(
    *,
    conditions: set[str] | None = None,
    actors: set[str] | None = None,
    tasks: set[str] | None = None,
    items_per_task: int | None = None,
    pairs_per_actor_domain: int | None = 2,
    samples_per_condition: int = 1,
) -> list[dict[str, Any]]:
    selected_conditions = conditions or set(BASE_CONDITIONS)
    unknown = selected_conditions - BASE_CONDITIONS
    if unknown:
        raise ValueError(f"unknown condition(s): {', '.join(sorted(unknown))}")
    selected_actors = sorted(actors or set(ACTORS))
    unknown_actors = set(selected_actors) - set(ACTOR_MODEL_ID)
    if unknown_actors:
        raise ValueError(f"missing model id for actor(s): {', '.join(sorted(unknown_actors))}")
    if samples_per_condition < 1:
        raise ValueError("samples_per_condition must be at least 1")

    tasks_in_order = task_rows(tasks=tasks, items_per_task=items_per_task)
    utility_by_actor = utility_pairs_by_actor(actors=set(selected_actors), pairs_per_actor_domain=pairs_per_actor_domain)
    run_id = f"base_distributions__{run_id_timestamp()}"
    jobs: list[dict[str, Any]] = []
    for actor in selected_actors:
        actor_pairs = utility_by_actor.get(actor) or utility_by_actor.get(FALLBACK_UTILITY_ACTOR, [])
        for task_index, task_row in enumerate(tasks_in_order):
            for condition in sorted(selected_conditions):
                for sample_index in range(samples_per_condition):
                    pair = utility_pair_for_slot(actor_pairs, task_index + sample_index)
                    system_prompt, prompt = prompt_for_condition(
                        condition,
                        task_row,
                        high_outcome=pair.get("high_description", ""),
                        low_outcome=pair.get("low_description", ""),
                    )
                    digest = stable_digest(actor, condition, task_row["task"], task_row["item_id"], prompt, system_prompt)
                    job_id = (
                        f"{condition}:{actor}:{task_row['task']}:"
                        f"i{task_row['item_index']}:s{sample_index}:v{digest}"
                    )
                    jobs.append(
                        {
                            "job_id": job_id,
                            "run_id": run_id,
                            "actor": actor,
                            "model": ACTOR_MODEL_ID[actor],
                            "condition": condition,
                            "sample_index": sample_index,
                            "system_prompt": system_prompt,
                            "prompt": prompt,
                            "task": task_row["task"],
                            "task_label": TASK_LABEL.get(task_row["task"], task_row["task"]),
                            "item_id": task_row["item_id"],
                            "item_index": task_row["item_index"],
                            "item_label": task_row["item_label"],
                            "base_prompt": task_row.get("base_prompt", ""),
                            "axis": task_row.get("axis", ""),
                            "axis_definition": task_row.get("axis_definition", ""),
                            "utility_pair": {
                                key: pair.get(key, "")
                                for key in (
                                    "domain",
                                    "domain_label",
                                    "pair_idx",
                                    "category",
                                    "high_description",
                                    "low_description",
                                    "high_utility",
                                    "low_utility",
                                    "delta_u",
                                )
                            },
                            "prompt_variant_id": digest,
                        }
                    )
    return jobs


def write_generation_jobs_file(jobs: list[dict[str, Any]], path=GENERATION_JOBS) -> None:
    write_jsonl(path, jobs)
