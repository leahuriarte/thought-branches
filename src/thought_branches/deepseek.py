"""Minimal client for DeepSeek's native reasoning-prefix completion beta."""

from __future__ import annotations

import http.client
import json
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any

from thought_branches.io_utils import load_env_file
from thought_branches.paths import ROOT


DEEPSEEK_BETA_BASE_URL = "https://api.deepseek.com/beta"
DIRECT_DEEPSEEK_MODELS = {"deepseek-flash", "deepseek-v4-pro"}
UTILITY_COVERED_DEEPSEEK_MODEL = "deepseek/deepseek-v3.2"
REASONING_EFFORTS = {"low", "high", "max"}


def require_deepseek_key() -> str:
    load_env_file(ROOT / ".env")
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for live DeepSeek validation")
    return key


def build_reasoning_prefix_request(
    *,
    model: str,
    messages: list[dict[str, str]],
    reasoning_prefix: str,
    max_tokens: int,
    reasoning_effort: str,
    top_p: float,
) -> dict[str, Any]:
    """Build the documented beta request for continuing a reasoning prefix."""
    if model not in DIRECT_DEEPSEEK_MODELS:
        supported = ", ".join(sorted(DIRECT_DEEPSEEK_MODELS))
        raise ValueError(f"unsupported direct DeepSeek model {model!r}; choose one of: {supported}")
    if not messages or messages[-1].get("role") == "assistant":
        raise ValueError("messages must end with the original user turn")
    if not reasoning_prefix:
        raise ValueError("reasoning_prefix must not be empty")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"reasoning_effort must be one of {sorted(REASONING_EFFORTS)}")
    if not 0.95 <= top_p <= 1.0:
        raise ValueError("DeepSeek thinking-mode top_p must be between 0.95 and 1.0")

    prefix_message: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "reasoning_content": reasoning_prefix,
        "prefix": True,
    }
    return {
        "model": model,
        "messages": [*messages, prefix_message],
        "thinking": {"type": "enabled"},
        "reasoning_effort": reasoning_effort,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }


class DeepSeekClient:
    def __init__(self, *, timeout_s: float = 120.0, max_retries: int = 3) -> None:
        self.api_key = require_deepseek_key()
        self.base_url = os.environ.get("DEEPSEEK_BETA_BASE_URL", DEEPSEEK_BETA_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def reasoning_prefix_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return json.loads(response.read().decode("utf-8", errors="replace"))
            except json.JSONDecodeError as exc:
                if attempt == self.max_retries:
                    raise RuntimeError("DeepSeek request failed: malformed non-JSON response") from exc
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if attempt == self.max_retries or exc.code < 500:
                    raise RuntimeError(f"DeepSeek request failed ({exc.code}): {detail}") from exc
            except urllib.error.URLError as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"DeepSeek request failed: {exc}") from exc
            except http.client.IncompleteRead as exc:
                if attempt == self.max_retries:
                    raise RuntimeError("DeepSeek request failed: incomplete response body") from exc
            except (TimeoutError, socket.timeout) as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"DeepSeek request timed out after {self.timeout_s}s") from exc
            time.sleep(min(2**attempt, 10))
        raise RuntimeError("DeepSeek request failed after retries")


def response_parts(response: dict[str, Any]) -> tuple[str, str, str]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("DeepSeek response has no usable choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("DeepSeek response has no assistant message")
    reasoning = message.get("reasoning_content")
    content = message.get("content")
    return (
        "" if reasoning is None else str(reasoning),
        "" if content is None else str(content),
        str(choices[0].get("finish_reason", "")),
    )
