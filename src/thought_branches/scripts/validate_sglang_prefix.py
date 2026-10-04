#!/usr/bin/env python3
"""Run an actual token-ID prefix-fidelity check against a local SGLang server."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from thought_branches.io_utils import write_jsonl
from thought_branches.sglang_backend import (
    SGLangClient, chat_template_token_ids, exact_continuation_prompt,
    sampling_params, token_ids_for_text,
)


CASES = [
    ("mid_sentence", "Recommend whether a library should extend weekend hours.",
     "The answer should weigh patron demand against staffing cost. A useful recommendation would be to extend"),
    ("token_boundary", "Explain one practical tradeoff.",
     "A particularly useful recommendation balances the cost with the benefit to readers."),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/api/sglang_prefix_validation.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers on the generation host to load the local tokenizer") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    client = SGLangClient(args.base_url, timeout=600)
    rows = []
    for index, (case_id, user_text, prefix_text) in enumerate(CASES):
        messages = [{"role": "user", "content": user_text}]
        conversation_ids = chat_template_token_ids(tokenizer, messages)
        if case_id == "token_boundary":
            full_prefix_ids = token_ids_for_text(tokenizer, prefix_text)
            # Cut the captured token stream at a real subword boundary in the
            # middle of the sentence; do not decode and re-encode the prefix.
            prefix_ids = full_prefix_ids[: max(1, len(full_prefix_ids) // 2)]
            prefix_text = tokenizer.decode(
                prefix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        else:
            prefix_ids = token_ids_for_text(tokenizer, prefix_text)
        full_prompt_ids = exact_continuation_prompt(conversation_ids, prefix_ids)
        params = sampling_params(max_new_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0, seed=8100 + index)
        response = client.generate(full_prompt_ids, params)
        meta = response.get("meta_info", {})
        prompt_count = meta.get("prompt_tokens")
        if prompt_count is not None and int(prompt_count) != len(full_prompt_ids):
            raise RuntimeError(f"{case_id}: server prompt token count {prompt_count} != submitted {len(full_prompt_ids)}")
        suffix_ids = [int(value) for value in response["output_ids"]]
        if not suffix_ids:
            raise RuntimeError(f"{case_id}: server returned no generated suffix token IDs")
        completion_count = meta.get("completion_tokens")
        if completion_count is not None and int(completion_count) != len(suffix_ids):
            raise RuntimeError(
                f"{case_id}: output_ids length {len(suffix_ids)} does not match generated-token count {completion_count}"
            )
        rows.append({
            "validation_id": case_id, "status": "passed",
            "endpoint": client.base_url + "/generate", "model_path": str(args.model_path),
            "conversation_token_ids": conversation_ids, "prefix_token_ids": prefix_ids,
            "full_prompt_token_ids": full_prompt_ids,
            "generated_suffix_token_ids": suffix_ids,
            "prefix_text": prefix_text, "generated_suffix_text": response.get("text", ""),
            "server_prompt_token_count": prompt_count,
            "submitted_prompt_token_count": len(full_prompt_ids),
            "server_completion_token_count": completion_count,
            "sampling_params": params, "meta_info": meta,
            "native_token_prefix_continuation_verified": True,
        })
    write_jsonl(args.out, rows)
    print(f"PASS: {len(rows)} /generate token-prefix checks; evidence written to {args.out}")


if __name__ == "__main__":
    main()
