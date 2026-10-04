from pathlib import Path
from tempfile import TemporaryDirectory

from thought_branches.sglang_backend import (
    build_sglang_branch_job, chat_template_token_ids, exact_continuation_prompt, run_sglang_jobs,
)


class TinyTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and add_generation_prompt and enable_thinking
        return [101, 102, 103]


class FakeClient:
    base_url = "http://127.0.0.1:30000"

    def __init__(self):
        self.seen = []

    def generate(self, prompt_token_ids, sampling_params):
        self.seen.append((prompt_token_ids, sampling_params))
        return {
            "text": "continued", "output_ids": [400, 401],
            "meta_info": {"prompt_tokens": len(prompt_token_ids), "finish_reason": {"type": "length"}},
        }


def test_prompt_uses_template_ids_then_exact_mid_sentence_prefix_ids():
    conversation_ids = chat_template_token_ids(TinyTokenizer(), [{"role": "user", "content": "Question"}])
    # A prefix may end mid-sentence; preserve these IDs as captured, with no re-encoding.
    prefix_ids = [8123, 19, 77]
    assert exact_continuation_prompt(conversation_ids, prefix_ids) == [101, 102, 103, 8123, 19, 77]


def test_runner_submits_full_ids_and_records_generated_suffix_ids():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        client = FakeClient()
        jobs = [{
            "job_id": "local:base:one", "model": "/models/qwen", "input_token_ids": [10, 11, 12, 13],
            "prefix_token_ids": [12, 13], "sampling_params": {"sampling_seed": 9, "max_new_tokens": 4},
        }]
        written, failed = run_sglang_jobs(
            jobs=jobs, client=client, output_path=root / "out.jsonl", failures_path=root / "fail.jsonl",
            provenance={"backend": "sglang_native_generate"}, concurrency=1,
        )
        assert (written, failed) == (1, 0)
        assert client.seen[0][0] == [10, 11, 12, 13]
        import json
        row = json.loads((root / "out.jsonl").read_text().strip())
        assert row["input_token_ids"] == [10, 11, 12, 13]
        assert row["prefix_token_ids"] == [12, 13]
        assert row["generated_suffix_token_ids"] == [400, 401]
        assert row["seed"] == 9


def test_branch_job_requires_local_sglang_seed_and_preserves_prefix_ids():
    source = {
        "output_id": "base-1", "job_id": "base-job", "success": True,
        "model": "/models/qwen", "input_token_ids": [1, 2, 3],
        "provenance": {"backend": "sglang_native_generate", "model_id": "/models/qwen", "model_hash": "abc"},
        "job": {"prompt": "Task"},
    }
    branch = build_sglang_branch_job(
        source=source, branch_label="motivated", prefix_token_ids=[41, 42, 43],
        continuation_sample_index=0, sampling={"random_seed": 8},
    )
    assert branch["input_token_ids"] == [1, 2, 3, 41, 42, 43]
    assert branch["prefix_token_ids"] == [41, 42, 43]
    assert branch["continuation_mode"] == "sglang_token_prefix"

    source["provenance"] = {"backend": "openrouter"}
    try:
        build_sglang_branch_job(
            source=source, branch_label="motivated", prefix_token_ids=[41],
            continuation_sample_index=0, sampling={},
        )
    except ValueError as exc:
        assert "locally generated" in str(exc)
    else:
        raise AssertionError("non-local seeds must be rejected")
