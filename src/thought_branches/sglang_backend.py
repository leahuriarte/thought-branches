"""Local SGLang native-token generation helpers.

This module intentionally uses SGLang's low-level ``/generate`` endpoint. It
does not use OpenAI chat completions or turn a reasoning prefix into a message.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from thought_branches.io_utils import append_jsonl, read_jsonl


DEFAULT_SGLANG_URL = "http://127.0.0.1:30000"


class SGLangClient:
    def __init__(self, base_url: str = DEFAULT_SGLANG_URL, *, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"SGLang POST {path} failed: {exc}") from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"SGLang POST {path} returned a non-object response")
        if "error" in result:
            raise RuntimeError(f"SGLang POST {path} returned error: {result['error']}")
        return result

    def generate(self, prompt_token_ids: list[int], sampling_params: dict[str, Any]) -> dict[str, Any]:
        if not prompt_token_ids or any(not isinstance(token, int) for token in prompt_token_ids):
            raise ValueError("prompt_token_ids must be a non-empty list of integers")
        response = self.post_json(
            "/generate",
            {"input_ids": prompt_token_ids, "sampling_params": sampling_params},
        )
        output_ids = response.get("output_ids")
        if not isinstance(output_ids, list) or any(not isinstance(token, int) for token in output_ids):
            raise RuntimeError("SGLang response omitted integer output_ids")
        return response

    def get_json(self, path: str) -> dict[str, Any]:
        request = urllib.request.Request(self.base_url + path, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"SGLang GET {path} failed: {exc}") from exc
        return result if isinstance(result, dict) else {"value": result}


def chat_template_token_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    """Render the original conversation with thinking enabled, preserving IDs."""
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("expected one conversation from tokenizer chat template")
        ids = ids[0]
    return [int(token) for token in ids]


def exact_continuation_prompt(conversation_token_ids: list[int], prefix_token_ids: list[int]) -> list[int]:
    """Append already-tokenized assistant reasoning IDs without re-tokenizing text."""
    if not conversation_token_ids:
        raise ValueError("conversation_token_ids cannot be empty")
    if not prefix_token_ids:
        raise ValueError("prefix_token_ids cannot be empty")
    return [int(token) for token in conversation_token_ids + prefix_token_ids]


def reasoning_ids_and_text(tokenizer: Any, generated_ids: list[int]) -> tuple[list[int], str]:
    """Extract Qwen3 reasoning IDs up to (but excluding) the ``</think>`` token."""
    end_ids = token_ids_for_text(tokenizer, "</think>")
    if not end_ids:
        raise ValueError("tokenizer does not encode the Qwen3 </think> marker")
    end_at = next(
        (i for i in range(len(generated_ids) - len(end_ids) + 1)
         if generated_ids[i:i + len(end_ids)] == end_ids),
        None,
    )
    if end_at is None:
        raise ValueError("base generation has no </think> marker; cannot identify a reasoning trace")
    ids = [int(token) for token in generated_ids[:end_at]]
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    # The output can open with a special <think> marker because the prompt starts
    # from Qwen's generation header; remove that marker while retaining its exact IDs.
    marker = "<think>"
    if text.startswith(marker):
        text = text[len(marker):].lstrip("\n")
    return ids, text


def token_count_at_text_offset(tokenizer: Any, token_ids: list[int], text: str, offset: int) -> int:
    """Find an exact token boundary for a character offset or reject the alignment."""
    if offset < 0 or offset > len(text):
        raise ValueError(f"text offset {offset} is outside 0..{len(text)}")
    for count in range(len(token_ids) + 1):
        decoded = tokenizer.decode(
            token_ids[:count], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if decoded.startswith("<think>"):
            decoded = decoded[len("<think>"):].lstrip("\n")
        if decoded == text[:offset]:
            return count
    raise ValueError(
        f"character offset {offset} falls inside a token or decoder normalization prevents exact alignment"
    )


def build_sglang_branch_job(
    *, source: dict[str, Any], branch_label: str, prefix_token_ids: list[int],
    continuation_sample_index: int, sampling: dict[str, Any],
) -> dict[str, Any]:
    """Build a branch from same-run SGLang base prompt IDs and captured reasoning IDs."""
    provenance = source.get("provenance", {})
    if provenance.get("backend") != "sglang_native_generate":
        raise ValueError("branch seeds must be locally generated with the SGLang backend")
    if source.get("success") is not True:
        raise ValueError("branch seed must be a successful local generation")
    prompt_ids = source.get("input_token_ids")
    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise ValueError("local base generation is missing original prompt token IDs")
    if not prefix_token_ids:
        raise ValueError("branch reasoning prefix must contain token IDs")
    original_job = source.get("job", {})
    stable = json.dumps(
        [source.get("output_id"), branch_label, prefix_token_ids, continuation_sample_index,
         provenance.get("model_hash")], separators=(",", ":"),
    )
    digest = hashlib.sha256(stable.encode()).hexdigest()[:16]
    job_id = f"sglang-branch:{source.get('output_id')}:{branch_label}:s{continuation_sample_index}:v{digest}"
    return {
        "job_id": job_id, "continuation_mode": "sglang_token_prefix",
        "source_output_id": source.get("output_id"), "source_job_id": source.get("job_id"),
        "branch_label": branch_label, "model": provenance.get("model_id", source.get("model", "")),
        "conversation_token_ids": prompt_ids, "prefix_token_ids": [int(t) for t in prefix_token_ids],
        "input_token_ids": exact_continuation_prompt(prompt_ids, prefix_token_ids),
        "sampling_params": dict(sampling), "source_provenance": provenance,
        "source_job": original_job,
    }


def token_ids_for_text(tokenizer: Any, text: str) -> list[int]:
    """Encode new source text without adding tokenizer special tokens."""
    return [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]


def tokenizer_file_hash(model_path: str | Path) -> str:
    """Hash tokenizer files only, avoiding a full multi-GB model scan."""
    root = Path(model_path)
    names = (
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt", "tokenizer.model",
    )
    digest = hashlib.sha256()
    found = False
    for name in names:
        path = root / name
        if path.is_file():
            found = True
            digest.update(name.encode())
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
    if not found:
        raise FileNotFoundError(f"no recognized tokenizer files found under {root}")
    return digest.hexdigest()


def sampling_params(*, max_new_tokens: int, temperature: float, top_p: float, seed: int) -> dict[str, Any]:
    if max_new_tokens < 1 or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("invalid sampling parameters")
    return {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "sampling_seed": seed,
        "skip_special_tokens": False,
    }


def run_sglang_jobs(
    *,
    jobs: list[dict[str, Any]],
    client: SGLangClient,
    output_path: Path,
    failures_path: Path,
    provenance: dict[str, Any],
    concurrency: int = 4,
    retries: int = 2,
    limit: int | None = None,
    tokenizer: Any | None = None,
) -> tuple[int, int]:
    """Run token-ID jobs concurrently; append one durable JSONL record per job."""
    if not 1 <= concurrency <= 8:
        raise ValueError("concurrency must be between 1 and 8")
    if retries < 0:
        raise ValueError("retries cannot be negative")
    done = {str(row.get("output_id", "")) for row in read_jsonl(output_path)} if output_path.exists() else set()
    pending = [job for job in jobs if str(job["job_id"]) not in done]
    if limit is not None:
        pending = pending[:limit]

    def execute(job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], float]:
        started = time.time()
        request_ids = [int(token) for token in job["input_token_ids"]]
        params = dict(job["sampling_params"])
        for attempt in range(retries + 1):
            try:
                response = client.generate(request_ids, params)
                meta = response.get("meta_info", {})
                prompt_count = meta.get("prompt_tokens")
                if prompt_count is not None and int(prompt_count) != len(request_ids):
                    raise RuntimeError(
                        f"server consumed {prompt_count} prompt tokens, expected exact {len(request_ids)} IDs"
                    )
                completion_count = meta.get("completion_tokens")
                if completion_count is not None and int(completion_count) != len(response["output_ids"]):
                    raise RuntimeError(
                        f"server reports {completion_count} generated tokens but returned "
                        f"{len(response['output_ids'])} output IDs; refusing ambiguous token alignment"
                    )
                return job, response, time.time() - started
            except Exception:
                if attempt == retries:
                    raise
                time.sleep(min(2 ** attempt, 8))
        raise AssertionError("unreachable")

    written = failed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(execute, job): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            try:
                _, response, elapsed = future.result()
                suffix_ids = [int(token) for token in response["output_ids"]]
                trace_fields: dict[str, Any] = {}
                if tokenizer is not None:
                    try:
                        trace_ids, trace_text = reasoning_ids_and_text(tokenizer, suffix_ids)
                        end_ids = token_ids_for_text(tokenizer, "</think>")
                        answer_ids = suffix_ids[len(trace_ids) + len(end_ids):]
                        answer_text = tokenizer.decode(
                            answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        trace_fields = {
                            "reasoning_token_ids": trace_ids, "reasoning_text": trace_text,
                            "output_text": answer_text,
                            "completion_complete": response.get("meta_info", {}).get("finish_reason", {}).get("type") == "stop",
                        }
                    except ValueError:
                        # Keep the raw token output and explicit parse status. A capped
                        # thinking trace can lack </think> and is not branchable yet.
                        trace_fields = {"reasoning_token_ids": [], "reasoning_text": "",
                                        "reasoning_parse_status": "incomplete_or_unparseable"}
                source_job = job.get("job", {})
                append_jsonl(output_path, {
                    "output_id": str(job["job_id"]), "job_id": job["job_id"], "success": True,
                    "model": job.get("model", ""), "input_token_ids": job["input_token_ids"],
                    "actor": source_job.get("actor", ""), "condition": job.get("condition", ""),
                    "sample_index": source_job.get("sample_index", 0),
                    "task": source_job.get("task", ""), "item_id": source_job.get("item_id", ""),
                    "item_label": source_job.get("item_label", ""), "run_id": source_job.get("run_id", ""),
                    "prefix_token_ids": job.get("prefix_token_ids", []),
                    "generated_suffix_token_ids": suffix_ids,
                    "text": response.get("text", ""), "meta_info": response.get("meta_info", {}),
                    "sampling_params": job["sampling_params"], "seed": job["sampling_params"].get("sampling_seed"),
                    "latency_s": round(elapsed, 3), "provenance": provenance,
                    "job": job.get("job", job), "sglang_job": job,
                    **trace_fields,
                })
                written += 1
            except Exception as exc:  # Each failed generation remains inspectable/retryable.
                append_jsonl(failures_path, {
                    "output_id": str(job.get("job_id", "")), "job_id": job.get("job_id", ""),
                    "success": False, "error": str(exc), "job": job, "provenance": provenance,
                })
                failed += 1
    return written, failed


def runtime_provenance(client: SGLangClient, *, model_id: str, model_revision: str, model_hash: str,
                       tokenizer_revision: str, tokenizer_hash: str, gpu_type: str,
                       context_limit: int, server_settings: dict[str, Any]) -> dict[str, Any]:
    """Collect runtime facts from local files and server metadata for each run."""
    version = "unknown"
    try:
        version = str(client.get_json("/get_server_info").get("version", "unknown"))
    except RuntimeError:
        pass
    if version == "unknown":
        try:
            version = importlib.metadata.version("sglang")
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "backend": "sglang_native_generate", "base_url": client.base_url, "model_id": model_id,
        "model_revision": model_revision, "model_hash": model_hash,
        "tokenizer_revision": tokenizer_revision, "tokenizer_hash": tokenizer_hash,
        "sglang_version": version, "gpu_type": gpu_type, "context_limit": context_limit,
        "server_settings": server_settings,
    }
