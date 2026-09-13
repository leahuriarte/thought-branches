from thought_branches.branching import (
    MOTIVATED_BRANCH,
    UNMOTIVATED_BRANCH,
    build_prefix_continuation_prompt,
    first_motivation_span,
    motivation_prefixes,
    readable_reasoning_text,
)
from thought_branches.chunking import split_text_into_chunks
from thought_branches.jobs import build_generation_jobs
from thought_branches.labeling import label_chunk
from thought_branches.prompts import prompt_for_condition


def test_prompt_conditions_are_distinct():
    task = {
        "task": "essay",
        "item_label": "Public museums should be free to all visitors",
        "base_prompt": "",
    }

    best_effort = prompt_for_condition("best_effort", task)
    high_utility = prompt_for_condition("high_utility", task, high_outcome="100 people are helped.")
    role_play = prompt_for_condition("role_play", task)

    assert "absolute best effort" in best_effort[1]
    assert "100 people are helped" in high_utility[1]
    assert "world-class essayist" in role_play[0]


def test_build_generation_jobs_can_make_tiny_dry_run_manifest():
    jobs = build_generation_jobs(
        conditions={"best_effort", "role_play"},
        actors={"gpt-5.4-mini-or"},
        tasks={"essay"},
        items_per_task=1,
        pairs_per_actor_domain=1,
        samples_per_condition=1,
    )

    assert len(jobs) == 2
    assert {job["condition"] for job in jobs} == {"best_effort", "role_play"}
    assert all(job["prompt"].strip() for job in jobs)


def test_chunk_and_label_helpers():
    text = "First, I will outline the essay.\n\nThis prize should not be mentioned in the final answer."
    chunks = split_text_into_chunks(text)

    assert len(chunks) == 2
    assert "outline_planning" in label_chunk(chunks[0].text)
    assert "incentive_awareness" in label_chunk(chunks[1].text)


def test_branch_prefix_helpers():
    trace = (
        "I should identify the topic and constraints.\n\n"
        "If this essay wins, the sponsor will fund the intervention, so quality matters."
    )

    span = first_motivation_span(trace)

    assert span is not None
    prefixes = motivation_prefixes(trace, span)
    assert prefixes["before_motivation"] == "I should identify the topic and constraints."

    motivated = build_prefix_continuation_prompt(
        original_prompt="Write the essay.",
        reasoning_prefix=prefixes["through_motivation_segment"],
        branch_label=MOTIVATED_BRANCH,
    )
    unmotivated = build_prefix_continuation_prompt(
        original_prompt="Write the essay.",
        reasoning_prefix=prefixes["before_motivation"],
        branch_label=UNMOTIVATED_BRANCH,
    )

    assert "includes the next reasoning thought" in motivated
    assert "stops immediately before" in unmotivated


def test_readable_reasoning_text_extracts_raw_response_reasoning():
    row = {
        "raw_response": {
            "choices": [
                {
                    "message": {
                        "content": "Final answer.",
                        "reasoning": "Plan the answer.",
                        "reasoning_details": [
                            {"type": "reasoning.text", "text": "Check constraints."},
                            {"type": "reasoning.encrypted", "text": "hidden"},
                        ],
                    }
                }
            ]
        }
    }

    trace = readable_reasoning_text(row)

    assert "Plan the answer." in trace
    assert "Check constraints." in trace
    assert "Final answer." not in trace
    assert "hidden" not in trace
