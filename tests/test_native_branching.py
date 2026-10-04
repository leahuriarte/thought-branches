from pathlib import Path
from tempfile import TemporaryDirectory

from thought_branches.branch_generation import branch_request, validate_identical_branch_inputs
from thought_branches.branching import MOTIVATED_BRANCH, UNMOTIVATED_BRANCH
from thought_branches.generation import run_jobs
from thought_branches.native_branching import (
    LLM_MOTIVATION_LABELING_METHOD,
    build_native_branch_jobs,
    first_llm_motivation_chunk,
)


def _fixture_rows():
    trace = (
        "First I should identify the task constraints.\n\n"
        "Helping this cause matters, so I should put extra effort into the answer.\n\n"
        "Next I should draft the response."
    )
    motivation_text = "Helping this cause matters, so I should put extra effort into the answer."
    start = trace.index(motivation_text)
    generation = {
        "output_id": "source-1",
        "actor": "deepseek-v3.2-or",
        "model": "deepseek/deepseek-v3.2",
        "condition": "high_utility",
        "task": "essay",
        "item_id": "essay-1",
        "item_label": "A topic",
        "sample_index": 4,
        "reasoning_text": trace,
        "job": {
            "prompt": "Write the original essay.",
            "system_prompt": "",
            "base_prompt": "Write the original essay.",
            "task": "essay",
            "item_id": "essay-1",
        },
    }
    chunks = [
        {
            "output_id": "source-1",
            "chunk_index": 0,
            "char_start": 0,
            "char_end": start - 2,
            "chunk_text": "First I should identify the task constraints.",
            "feature_flags": ["ordinary_task_planning"],
            "labeling_method": LLM_MOTIVATION_LABELING_METHOD,
            "labeler_model": "classifier/model",
        },
        {
            "output_id": "source-1",
            "chunk_index": 1,
            "char_start": start,
            "char_end": start + len(motivation_text),
            "chunk_text": motivation_text,
            "feature_flags": ["motivation_commitment", "motivation_values", "motivation_effort"],
            "labeling_method": LLM_MOTIVATION_LABELING_METHOD,
            "labeler_model": "classifier/model",
        },
    ]
    return generation, chunks


def test_native_branch_jobs_use_llm_chunk_and_direct_prefix_fields():
    generation, chunks = _fixture_rows()
    jobs = build_native_branch_jobs(
        generations=[generation],
        labeled_chunks=chunks,
        target_model="deepseek-flash",
        samples_per_branch=2,
        max_seeds=1,
        min_pre_prefix_chars=1,
    )

    assert len(jobs) == 4
    assert {job["branch_label"] for job in jobs} == {MOTIVATED_BRANCH, UNMOTIVATED_BRANCH}
    assert all(job["model"] == "deepseek-flash" for job in jobs)
    assert all(job["source_model"] == "deepseek/deepseek-v3.2" for job in jobs)
    assert all(job["labeling_method"] == LLM_MOTIVATION_LABELING_METHOD for job in jobs)
    assert all("prompt" not in job for job in jobs)

    requests = [
        branch_request(job, max_tokens=2048, reasoning_effort="low", top_p=0.95) for job in jobs
    ]
    validate_identical_branch_inputs(jobs, requests)
    for job, request in zip(jobs, requests):
        assert request["messages"][-1]["prefix"] is True
        assert request["messages"][-1]["reasoning_content"] == job["reasoning_prefix"]
        assert request["messages"][-2] == {"role": "user", "content": "Write the original essay."}


def test_branch_selection_rejects_old_or_heuristic_labels():
    _, chunks = _fixture_rows()
    chunks[0]["labeling_method"] = "llm_dag_v1"

    try:
        first_llm_motivation_chunk(chunks)
    except ValueError as exc:
        assert "llm_dag_v3_motivation_commitment" in str(exc)
    else:
        raise AssertionError("old labels should be rejected")


def test_branch_selection_skips_repeated_cue_without_commitment():
    _, chunks = _fixture_rows()
    chunks[1]["feature_flags"] = ["motivation_cue_repetition"]

    assert first_llm_motivation_chunk(chunks) is None


def test_openrouter_runner_rejects_native_branch_jobs():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        try:
            run_jobs(
                jobs=[{"continuation_mode": "native_reasoning_prefix"}],
                generations_path=root / "generations.jsonl",
                failures_path=root / "failures.jsonl",
                dry_run=True,
                limit=None,
                temperature=None,
                max_tokens=10,
                reasoning=None,
            )
        except ValueError as exc:
            assert "tb-run-branch-continuations" in str(exc)
        else:
            raise AssertionError("native branch jobs should not run through OpenRouter")
