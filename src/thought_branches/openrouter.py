"""Minimal OpenRouter chat-completion client."""

from __future__ import annotations

import http.client
import json
import os
import socket
import time
import urllib.error
import urllib.request
from copy import deepcopy
from typing import Any

from thought_branches.io_utils import load_env_file
from thought_branches.paths import ROOT


class MalformedOpenRouterResponse(RuntimeError):
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        snippet = json.dumps(response, sort_keys=True)[:1000]
        super().__init__(f"OpenRouter response did not contain a chat choice: {snippet}")


def require_openrouter_key() -> str:
    load_env_file(ROOT / ".env")
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key or key == "xxx":
        raise RuntimeError("Set OPENROUTER_API_KEY in .env or the shell environment before live API calls.")
    return key


class OpenRouterClient:
    def __init__(self, *, timeout_s: float = 120.0, max_retries: int = 3) -> None:
        self.api_key = require_openrouter_key()
        self.base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.site_url = os.environ.get("OPENROUTER_SITE_URL", "").strip()
        self.app_name = os.environ.get("OPENROUTER_APP_NAME", "Thought Branches").strip()

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int,
        reasoning: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            payload["temperature"] = temperature
        if reasoning is not None:
            payload["reasoning"] = reasoning
        body = json.dumps(payload).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self.site_url and self.site_url != "xxx":
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return json.loads(response.read().decode("utf-8", errors="replace"))
            except json.JSONDecodeError as exc:
                if attempt == self.max_retries:
                    raise RuntimeError("OpenRouter request failed: malformed non-JSON response") from exc
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if attempt == self.max_retries or exc.code < 500:
                    raise RuntimeError(f"OpenRouter request failed ({exc.code}): {detail}") from exc
            except urllib.error.URLError as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
            except http.client.IncompleteRead as exc:
                if attempt == self.max_retries:
                    raise RuntimeError("OpenRouter request failed: incomplete response body") from exc
            except (TimeoutError, socket.timeout) as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"OpenRouter request timed out after {self.timeout_s}s") from exc
            time.sleep(min(2**attempt, 10))
        raise RuntimeError("OpenRouter request failed after retries")


def response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not choices:
        raise MalformedOpenRouterResponse(response)
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise MalformedOpenRouterResponse(response)
    content = message.get("content", "")
    return "" if content is None else str(content)


def response_without_message_content(response: dict[str, Any]) -> dict[str, Any]:
    sanitized = deepcopy(response)
    choices = sanitized.get("choices")
    if not isinstance(choices, list):
        return sanitized
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict) or "content" not in message:
            continue
        content = message.pop("content")
        message["content_omitted"] = True
        message["content_char_count"] = 0 if content is None else len(str(content))
        message["content_was_null"] = content is None
    return sanitized
