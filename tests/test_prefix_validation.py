from thought_branches.deepseek import build_reasoning_prefix_request
from thought_branches.prefix_validation import summarize_validation, validation_rows


def test_native_reasoning_prefix_request_uses_documented_beta_fields():
    request = build_reasoning_prefix_request(
        model="deepseek-flash",
        messages=[{"role": "user", "content": "Answer the task."}],
        reasoning_prefix="I should first compare",
        max_tokens=256,
        reasoning_effort="low",
        top_p=0.95,
    )

    assert request["messages"][-1] == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "I should first compare",
        "prefix": True,
    }
    assert request["thinking"] == {"type": "enabled"}
    assert request["reasoning_effort"] == "low"
    assert "temperature" not in request


def test_repeated_validation_requests_are_byte_identical_per_prefix():
    rows = validation_rows(
        cases=[
            {
                "case_id": "mid_sentence",
                "cut_kind": "mid_sentence",
                "system_prompt": "",
                "user_prompt": "Make a recommendation.",
                "reasoning_prefix": "The strongest option is",
            }
        ],
        model="deepseek-flash",
        repeats=3,
        max_tokens=128,
        reasoning_effort="low",
        top_p=0.95,
    )

    assert len({row["request_digest"] for row in rows}) == 1
    assert rows[0]["endpoint"] == "https://api.deepseek.com/beta/chat/completions"
    assert [row["sample_index"] for row in rows] == [0, 1, 2]


def test_validation_summary_never_marks_current_direct_model_utility_eligible():
    rows = [
        {
            "case_id": "mid_sentence",
            "cut_kind": "mid_sentence",
            "status": "success",
            "reasoning_suffix": suffix,
            "output_text": "A final answer.",
            "finish_reason": "stop",
            "request_digest": "same",
        }
        for suffix in (" option A.", " option B.")
    ]

    summary = summarize_validation(rows, model="deepseek-flash")

    assert summary["automated_transport_checks_pass"] is True
    assert summary["prefix_transport_checks_pass"] is True
    assert summary["diversity_checks_pass"] is True
    assert summary["end_to_end_completion_checks_pass"] is True
    assert summary["same_experimental_model"] is False
    assert summary["eligible_for_utility_pilot"] is False
    assert summary["manual_prefix_fidelity_review_required"] is True
