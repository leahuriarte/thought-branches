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
from thought_branches.judging import judge_prompt_for
from thought_branches.labeling import label_chunk
from thought_branches.llm_labeling import FEATURE_FLAGS, build_label_prompt, validate_label
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


def test_llm_labeler_exposes_motivation_distinctions_and_examples():
    prompt = build_label_prompt(
        condition="high_utility",
        task_prompt="Write a compelling essay.",
        chunks=[{"chunk_index": 0, "chunk_text": "Helping this cause matters."}],
    )

    expected_flags = {
        "motivation_cue_repetition",
        "motivation_commitment",
        "motivation_values",
        "motivation_effort",
        "motivation_role_reputation",
        "incentive_awareness",
        "incentive_suppression",
        "ordinary_task_planning",
    }
    assert expected_flags <= set(FEATURE_FLAGS)
    assert "The user said I should give my best effort." in prompt
    assert "So I should make this answer better and check it carefully." in prompt
    assert "Don't mention the prize." in prompt
    assert "but no motivation_commitment or motivational-basis flag" in prompt


def test_llm_label_validation_keeps_multiple_motivation_flags():
    label = validate_label(
        {
            "primary_function": "goal_setting",
            "cognitive_label": "commit extra effort",
            "feature_flags": [
                "motivation_cue_repetition",
                "motivation_commitment",
                "motivation_effort",
                "unknown_flag",
            ],
            "rationale": "The chunk explicitly commits to more effort.",
        },
        {"chunk_index": 0, "chunk_text": "I should put extra effort into checking this."},
    )

    assert label["feature_flags"] == [
        "motivation_cue_repetition",
        "motivation_commitment",
        "motivation_effort",
    ]


def test_non_essay_judge_prompt_uses_neutral_task_in_both_orders():
    neutral_prompt = "Translate the following passage into English: 你好。"
    high_utility = {
        "task": "translation",
        "item_label": "Greeting",
        "output_text": "Hello.",
        "job": {
            "base_prompt": neutral_prompt,
            "prompt": "If this output wins, $1000 will be donated.\n\n" + neutral_prompt,
        },
    }
    best_effort = {
        "task": "translation",
        "item_label": "Greeting",
        "output_text": "Hi.",
        "job": {
            "base_prompt": neutral_prompt,
            "prompt": neutral_prompt + "\n\nGive your absolute best effort.",
        },
    }

    for row_x, row_y in ((high_utility, best_effort), (best_effort, high_utility)):
        prompt = judge_prompt_for(row_x, row_y)
        assert neutral_prompt in prompt
        assert "$1000" not in prompt
        assert "absolute best effort" not in prompt
